#!/usr/bin/env python3
"""Fine-tune Transkun using artifacts in eval_utils/4_fine_tuning/model_info.

This script mirrors the notebook workflow and runs the core steps end-to-end:
1) validate paths
2) seed a fresh fine-tune checkpoint with pretrained weights
3) run transkun.train
4) verify with a one-file transcription on the test split
"""

#python fine_tune.py --dry-run --skip-train
#Run full fine-tune + verify:
#python fine_tune.py


from __future__ import annotations

import argparse
import copy
import os
import pickle
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import torch


THIS_FILE = Path(__file__).resolve()
FINE_TUNE_ROOT = THIS_FILE.parent
PROJECT_ROOT = FINE_TUNE_ROOT.parent.parent
DEFAULT_MODEL_INFO_DIR = FINE_TUNE_ROOT / "model_info"
DEFAULT_RUN_DIR = FINE_TUNE_ROOT / "output" / "finetune_v2_run"
DEFAULT_DATASET_PATH = Path("/scratch/gilbreth/li5042/datasets")


def log(msg: str) -> None:
	print(f"[fine_tune] {msg}")


def resolve_device(device: str) -> str:
	normalized = device.strip().lower()
	if normalized in {"auto", "gpu"}:
		normalized = "cuda" if torch.cuda.is_available() else "cpu"

	if normalized.startswith("cuda") and not torch.cuda.is_available():
		raise RuntimeError("CUDA device requested, but torch.cuda.is_available() is False")

	return normalized


def load_checkpoint_compat(weight_path: Path, map_location: str):
	try:
		return torch.load(weight_path, map_location=map_location, weights_only=True)
	except TypeError:
		return torch.load(weight_path, map_location=map_location)


def extract_audio_filename(entry) -> str:
	if isinstance(entry, dict):
		for key in ("audio_filename", "audioPath", "audio_path", "wav", "audio"):
			value = entry.get(key)
			if isinstance(value, str) and value:
				return value

	if isinstance(entry, (list, tuple)):
		for item in entry:
			if isinstance(item, str) and item.lower().endswith(".wav"):
				return item
			if isinstance(item, dict):
				try:
					return extract_audio_filename(item)
				except ValueError:
					pass

	raise ValueError(f"Could not extract an audio filename from test entry: {entry!r}")


def resolve_sample_wav_path(dataset_path: Path, sample_rel: str) -> Path:
	"""Resolve sample audio path across common metadata conventions.

	Some metadata stores paths like "2009/...wav", others like "MAESTRO/2009/...wav".
	"""
	rel = Path(sample_rel)
	if rel.is_absolute():
		return rel

	candidates = [dataset_path / rel, dataset_path.parent / rel]

	parts = rel.parts
	if parts and parts[0].lower() == "maestro":
		trimmed = Path(*parts[1:]) if len(parts) > 1 else Path("")
		candidates.insert(0, dataset_path / trimmed)

	seen = set()
	for candidate in candidates:
		candidate_str = str(candidate)
		if candidate_str in seen:
			continue
		seen.add(candidate_str)
		if candidate.exists():
			return candidate

	return candidates[0]


def run_command(cmd: list[str], cwd: Path, env: dict[str, str], dry_run: bool) -> None:
	log("Command:")
	print(shlex.join(cmd))
	if dry_run:
		log("Dry-run enabled; command not executed.")
		return
	subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def validate_paths(paths: list[Path]) -> None:
	missing = [p for p in paths if not p.exists()]
	if missing:
		missing_str = "\n".join(str(p) for p in missing)
		raise FileNotFoundError(f"Missing required inputs:\n{missing_str}")


def get_split_size(path: Path) -> int:
	with path.open("rb") as f:
		data = pickle.load(f)
	if not isinstance(data, list):
		raise TypeError(f"Expected list metadata in {path}, got {type(data)}")
	return len(data)


def seed_finetune_checkpoint(
	pretrained_ckpt: Path,
	pretrained_conf: Path,
	finetune_ckpt: Path,
	finetune_conf: Path,
	device: str,
	max_lr: float,
	weight_decay: float,
	n_iter: int,
	dry_run: bool,
) -> None:
	log("Seeding fine-tuning checkpoint with pretrained weights and a fresh optimizer/LR schedule.")

	if dry_run:
		log(f"Dry-run: would copy {pretrained_conf} -> {finetune_conf}")
		log(f"Dry-run: would create seeded checkpoint at {finetune_ckpt}")
		return

	import moduleconf
	from transkun.TrainUtil import initializeCheckpoint, save_checkpoint

	finetune_ckpt.parent.mkdir(parents=True, exist_ok=True)
	shutil.copy2(pretrained_conf, finetune_conf)

	conf_manager = moduleconf.parseFromFile(str(finetune_conf))
	transkun_class = conf_manager["Model"].module.TransKun
	conf = conf_manager["Model"].config

	(
		_start_epoch,
		_start_iter,
		model,
		loss_tracker,
		_best_state_dict,
		optimizer,
		lr_scheduler,
	) = initializeCheckpoint(
		transkun_class,
		device=device,
		max_lr=max_lr,
		weight_decay=weight_decay,
		nIter=n_iter,
		conf=conf,
	)

	pretrained_state = load_checkpoint_compat(pretrained_ckpt, map_location=device)
	source_state = pretrained_state.get(
		"best_state_dict",
		pretrained_state.get("state_dict", pretrained_state),
	)
	load_result = model.load_state_dict(source_state, strict=False)

	best_state_dict = copy.deepcopy(model.state_dict())
	save_checkpoint(
		str(finetune_ckpt),
		epoch=0,
		nIter=0,
		model=model,
		lossTracker=loss_tracker,
		best_state_dict=best_state_dict,
		optimizer=optimizer,
		lrScheduler=lr_scheduler,
	)

	log(f"Seeded checkpoint: {finetune_ckpt}")
	log(f"Run config: {finetune_conf}")
	log(f"Missing keys: {len(load_result.missing_keys)}")
	log(f"Unexpected keys: {len(load_result.unexpected_keys)}")


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Fine-tune Transkun using model_info artifacts, then run one-file verification."
	)
	parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
	parser.add_argument("--model-info-dir", type=Path, default=DEFAULT_MODEL_INFO_DIR)
	parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
	parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)

	parser.add_argument("--n-process", type=int, default=1)
	parser.add_argument("--batch-size", type=int, default=4)
	parser.add_argument("--data-loader-workers", type=int, default=8)
	parser.add_argument("--max-lr", type=float, default=1e-4)
	parser.add_argument("--weight-decay", type=float, default=1e-4)
	parser.add_argument("--n-iter", type=int, default=50000)
	parser.add_argument("--device", default="auto", help="auto|cpu|cuda|cuda:N")
	parser.add_argument("--allow-tf32", action="store_true")

	parser.add_argument("--skip-seed", action="store_true")
	parser.add_argument("--skip-train", action="store_true")
	parser.add_argument("--skip-verify", action="store_true")
	parser.add_argument("--dry-run", action="store_true")

	args = parser.parse_args()

	project_root = args.project_root.resolve()
	model_info_dir = args.model_info_dir.resolve()
	dataset_path = args.dataset_path.resolve()
	run_dir = args.run_dir.resolve()
	run_dir.mkdir(parents=True, exist_ok=True)

	if str(project_root) not in sys.path:
		sys.path.insert(0, str(project_root))

	model_params_dir = model_info_dir / "model_params"
	training_data_dir = model_info_dir / "training_data"

	pretrained_ckpt = model_params_dir / "checkpoint.pt"
	pretrained_conf = model_params_dir / "model.conf"
	train_meta = training_data_dir / "train.pickle"
	val_meta = training_data_dir / "val.pickle"
	test_meta = training_data_dir / "test.pickle"

	finetune_ckpt = run_dir / "checkpoint_finetuned.pt"
	finetune_conf = run_dir / "model_finetune.conf"
	validation_mid = run_dir / "validation_test.mid"

	validate_paths([
		project_root,
		model_info_dir,
		dataset_path,
		pretrained_ckpt,
		pretrained_conf,
		train_meta,
		val_meta,
		test_meta,
	])

	train_size = get_split_size(train_meta)
	val_size = get_split_size(val_meta)
	test_size = get_split_size(test_meta)

	log(f"Split sizes: train={train_size} val={val_size} test={test_size}")
	if train_size == 0:
		log("WARN: train split is empty")
	if val_size == 0:
		log("WARN: val split is empty")
	if test_size == 0:
		log("WARN: test split is empty")

	device = resolve_device(args.device)
	log(f"Project root: {project_root}")
	log(f"Model info dir: {model_info_dir}")
	log(f"Run dir: {run_dir}")
	log(f"Dataset root: {dataset_path}")
	log(f"Device: {device}")

	if not args.skip_seed:
		seed_finetune_checkpoint(
			pretrained_ckpt=pretrained_ckpt,
			pretrained_conf=pretrained_conf,
			finetune_ckpt=finetune_ckpt,
			finetune_conf=finetune_conf,
			device=device,
			max_lr=args.max_lr,
			weight_decay=args.weight_decay,
			n_iter=args.n_iter,
			dry_run=args.dry_run,
		)
	else:
		log("Skipping checkpoint seeding (--skip-seed).")

	if not finetune_ckpt.exists() or not finetune_conf.exists():
		if args.dry_run:
			log(
				"Dry-run: fine-tune checkpoint/config do not exist yet; this is expected "
				"when seeding/training commands are not executed."
			)
		else:
			raise FileNotFoundError(
				"Fine-tune checkpoint/config not found. Run without --skip-seed first, "
				"or point --run-dir to an existing seeded run directory."
			)

	env = os.environ.copy()
	old_python_path = env.get("PYTHONPATH", "")
	env["PYTHONPATH"] = (
		f"{project_root}:{old_python_path}" if old_python_path else str(project_root)
	)

	train_cmd = [
		sys.executable,
		"-m",
		"transkun.train",
		"--nProcess",
		str(args.n_process),
		"--datasetPath",
		str(dataset_path),
		"--datasetMetaFile_train",
		str(train_meta),
		"--datasetMetaFile_val",
		str(val_meta),
		"--batchSize",
		str(args.batch_size),
		"--dataLoaderWorkers",
		str(args.data_loader_workers),
		"--max_lr",
		str(args.max_lr),
		"--weight_decay",
		str(args.weight_decay),
		"--nIter",
		str(args.n_iter),
		"--modelConf",
		str(finetune_conf),
	]
	if args.allow_tf32:
		train_cmd.append("--allow_tf32")
	train_cmd.append(str(finetune_ckpt))

	if args.skip_train:
		log("Skipping training (--skip-train).")
	else:
		if train_size == 0:
			log("WARN: Skipping training because train split is empty.")
		elif val_size == 0:
			log("WARN: Skipping training because val split is empty (transkun.train requires non-empty validation data).")
		else:
			log("Launching fine-tuning via transkun.train...")
			run_command(train_cmd, cwd=project_root, env=env, dry_run=args.dry_run)

	if args.skip_verify:
		log("Skipping verification transcription (--skip-verify).")
		return
	if test_size == 0:
		log("WARN: Skipping verification transcription because test split is empty.")
		return

	with open(test_meta, "rb") as f:
		test_entries = pickle.load(f)
	if not test_entries:
		raise RuntimeError(f"No entries found in test metadata: {test_meta}")

	sample_rel = extract_audio_filename(test_entries[0])
	sample_wav = resolve_sample_wav_path(dataset_path, sample_rel)
	if not sample_wav.exists():
		if args.dry_run:
			log(
				"Dry-run: sample test audio does not exist at resolved path yet: "
				f"{sample_wav}"
			)
		else:
			raise FileNotFoundError(
				f"Sample test audio not found: {sample_wav} (metadata entry: {sample_rel})"
			)

	verify_cmd = [
		sys.executable,
		"-m",
		"transkun.transcribe",
		str(sample_wav),
		str(validation_mid),
		"--device",
		device,
		"--weight",
		str(finetune_ckpt),
		"--conf",
		str(finetune_conf),
	]

	log("Running one-file verification transcription...")
	run_command(verify_cmd, cwd=project_root, env=env, dry_run=args.dry_run)

	log(f"Verification output MIDI: {validation_mid}")


if __name__ == "__main__":
	main()
