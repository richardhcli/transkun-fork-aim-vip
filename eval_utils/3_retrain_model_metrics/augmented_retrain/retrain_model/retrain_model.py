#!/usr/bin/env python3
"""Orchestrate full Transkun retraining with optional augmentation staging.

This script combines the behavior of generate_prereq_files.sh and
training_script_dry_run.sh and is designed for long retraining runs.
"""

from __future__ import annotations

import argparse
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, List, Optional


DEFAULT_DATASET_DIR = Path("/scratch/gilbreth/li5042/datasets/MAESTRO")
DEFAULT_METADATA_DIR = Path(
	"/scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/3_retrain_model_metrics/output/MAESTRO_METADATA"
)
DEFAULT_SAVE_DIR = Path(
	"/scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/3_retrain_model_metrics/transkun_checkpoints"
)
DEFAULT_AUG_STAGE_DIR = Path(
	"/scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/3_retrain_model_metrics/output/AUGMENTATION_STAGING"
)


def timestamp() -> str:
	return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
	print(f"[{timestamp()}] {msg}", flush=True)
	print(f"[{timestamp()}] {msg}", file=sys.stderr, flush=True)


def run_cmd(cmd: List[str], *, stdout_file: Optional[Path] = None) -> None:
	log("Running: " + " ".join(cmd))
	if stdout_file is None:
		subprocess.run(cmd, check=True)
		return

	stdout_file.parent.mkdir(parents=True, exist_ok=True)
	with stdout_file.open("w", encoding="utf-8") as f:
		subprocess.run(cmd, check=True, stdout=f)


def parse_multi_paths(values: Optional[List[str]]) -> List[Path]:
	if not values:
		return []
	paths: List[Path] = []
	for item in values:
		for token in item.split(","):
			token = token.strip()
			if token:
				paths.append(Path(token))
	return paths


def collect_files(sources: Iterable[Path], exts: Iterable[str]) -> List[Path]:
	ext_set = {e.lower() for e in exts}
	found: List[Path] = []

	for src in sources:
		if not src.exists():
			raise FileNotFoundError(f"Augmentation source not found: {src}")

		if src.is_file():
			if src.suffix.lower() in ext_set:
				found.append(src)
			continue

		for p in src.rglob("*"):
			if p.is_file() and p.suffix.lower() in ext_set:
				found.append(p)

	return found


def stage_augmentation_files(
	files: List[Path], stage_root: Path, kind: str, clear_stage: bool
) -> Optional[Path]:
	if not files:
		return None

	target = stage_root / kind
	if clear_stage and target.exists():
		shutil.rmtree(target)
	target.mkdir(parents=True, exist_ok=True)

	for idx, src in enumerate(files):
		base_name = f"{idx:06d}_{src.name}"
		dst = target / base_name
		if dst.exists():
			stem = dst.stem
			suffix = dst.suffix
			bump = 1
			while True:
				candidate = target / f"{stem}_{bump}{suffix}"
				if not candidate.exists():
					dst = candidate
					break
				bump += 1
		dst.symlink_to(src)

	log(f"Prepared {len(files)} {kind} augmentation files in {target}")
	return target


def ensure_prerequisites(args: argparse.Namespace, py_exec: str) -> None:
	args.metadata_dir.mkdir(parents=True, exist_ok=True)

	train_pickle = args.metadata_dir / "train.pickle"
	val_pickle = args.metadata_dir / "val.pickle"
	conf_json = args.metadata_dir / "transkun_base.json"

	metadata_missing = not train_pickle.exists() or not val_pickle.exists()
	conf_missing = not conf_json.exists()

	if args.regenerate_metadata or metadata_missing:
		csv_path = args.dataset_dir / args.metadata_csv
		run_cmd(
			[
				py_exec,
				"-m",
				"transkun.createDatasetMaestro",
				str(args.dataset_dir),
				str(csv_path),
				str(args.metadata_dir),
			]
		)
	else:
		log("Skipping metadata generation: train/val pickle files already exist.")

	if args.regenerate_conf or conf_missing:
		run_cmd(
			[py_exec, "-m", "moduleconf.generate", "Model:transkun.ModelTransformer"],
			stdout_file=conf_json,
		)
	else:
		log("Skipping conf generation: transkun_base.json already exists.")


def run_training(args: argparse.Namespace, py_exec: str, checkpoint_path: Path) -> int:
	cmd: List[str] = [
		py_exec,
		"-m",
		"transkun.train",
		"--nProcess",
		str(args.n_process),
		"--datasetPath",
		str(args.dataset_dir),
		"--datasetMetaFile_train",
		str(args.metadata_dir / "train.pickle"),
		"--datasetMetaFile_val",
		str(args.metadata_dir / "val.pickle"),
		"--modelConf",
		str(args.metadata_dir / "transkun_base.json"),
		"--nIter",
		str(args.n_iter),
		"--batchSize",
		str(args.batch_size),
		"--dataLoaderWorkers",
		str(args.dataloader_workers),
		"--max_lr",
		str(args.max_lr),
		"--weight_decay",
		str(args.weight_decay),
		str(checkpoint_path),
	]

	if args.allow_tf32:
		cmd.append("--allow_tf32")
	if args.hop_size is not None:
		cmd.extend(["--hopSize", str(args.hop_size)])
	if args.chunk_size is not None:
		cmd.extend(["--chunkSize", str(args.chunk_size)])

	if args.augment:
		cmd.append("--augment")
		if args.noise_folder is not None:
			cmd.extend(["--noiseFolder", str(args.noise_folder)])
		if args.ir_folder is not None:
			cmd.extend(["--irFolder", str(args.ir_folder)])

	max_seconds = int(args.max_hours * 3600)
	log(f"Starting training with max runtime {args.max_hours:.2f} hours.")

	proc = subprocess.Popen(cmd)
	start = time.time()
	while True:
		rc = proc.poll()
		if rc is not None:
			return rc

		elapsed = time.time() - start
		if elapsed >= max_seconds:
			log("Reached max runtime. Sending SIGTERM to training process.")
			proc.terminate()
			try:
				return proc.wait(timeout=120)
			except subprocess.TimeoutExpired:
				log("Training process did not exit after SIGTERM; sending SIGKILL.")
				proc.kill()
				return proc.wait()

		time.sleep(20)


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description="Retrain Transkun from scratch with optional augmentation staging"
	)
	parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
	parser.add_argument("--metadata-csv", default="maestro-v3.0.0.csv")
	parser.add_argument("--metadata-dir", type=Path, default=DEFAULT_METADATA_DIR)
	parser.add_argument("--save-dir", type=Path, default=DEFAULT_SAVE_DIR)
	parser.add_argument(
		"--checkpoint-name", default="checkpoint_baseline.pt", help="Output checkpoint filename"
	)

	parser.add_argument("--n-process", type=int, default=1)
	parser.add_argument("--n-iter", type=int, default=10_000_000)
	parser.add_argument("--max-hours", type=float, default=48.0)
	parser.add_argument("--batch-size", type=int, default=4)
	parser.add_argument("--dataloader-workers", type=int, default=8)
	parser.add_argument("--max-lr", type=float, default=2e-4)
	parser.add_argument("--weight-decay", type=float, default=1e-4)
	parser.add_argument("--hop-size", type=float)
	parser.add_argument("--chunk-size", type=float)
	parser.add_argument("--allow-tf32", action="store_true")

	parser.add_argument("--regenerate-metadata", action="store_true")
	parser.add_argument("--regenerate-conf", action="store_true")
	parser.add_argument("--force-fresh", action="store_true", default=True)
	parser.add_argument("--no-force-fresh", action="store_false", dest="force_fresh")

	parser.add_argument("--augment", action="store_true")
	parser.add_argument("--noise-source", action="append", help="Noise source file/dir (repeatable or comma-separated)")
	parser.add_argument("--ir-source", action="append", help="IR source file/dir (repeatable or comma-separated)")
	parser.add_argument("--augmentation-stage-dir", type=Path, default=DEFAULT_AUG_STAGE_DIR)
	parser.add_argument("--clear-augmentation-stage", action="store_true")

	return parser


def main() -> int:
	parser = build_parser()
	args = parser.parse_args()
	py_exec = sys.executable

	args.save_dir.mkdir(parents=True, exist_ok=True)
	checkpoint_path = args.save_dir / args.checkpoint_name

	log("Preparing prerequisite metadata and model config...")
	ensure_prerequisites(args, py_exec)

	noise_sources = parse_multi_paths(args.noise_source)
	ir_sources = parse_multi_paths(args.ir_source)

	args.noise_folder = None
	args.ir_folder = None

	if args.augment:
		noise_files = collect_files(noise_sources, exts=[".wav", ".flac", ".mp3", ".ogg"]) if noise_sources else []
		ir_files = collect_files(ir_sources, exts=[".wav", ".flac", ".mp3", ".ogg"]) if ir_sources else []

		args.noise_folder = stage_augmentation_files(
			noise_files, args.augmentation_stage_dir, "noise", args.clear_augmentation_stage
		)
		args.ir_folder = stage_augmentation_files(
			ir_files, args.augmentation_stage_dir, "ir", args.clear_augmentation_stage
		)

		if args.noise_folder is None and args.ir_folder is None:
			log("[WARN] --augment set but no augmentation files were staged.")

	if args.force_fresh and checkpoint_path.exists():
		log(f"Removing existing checkpoint to force scratch training: {checkpoint_path}")
		checkpoint_path.unlink()

	log("Launching Transkun retraining run...")
	rc = run_training(args, py_exec, checkpoint_path)
	if rc == 0:
		log("Training process exited successfully.")
	else:
		log(f"Training process exited with code {rc}.")
	return rc


if __name__ == "__main__":
	try:
		raise SystemExit(main())
	except KeyboardInterrupt:
		log("Interrupted by user (KeyboardInterrupt).")
		raise SystemExit(130)

