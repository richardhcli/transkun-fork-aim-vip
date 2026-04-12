#!/usr/bin/env python3
"""Step 1: Generate fine-tuning artifacts from augmented datasets.

This script reuses augmented_retrain/preprocess_datasets/prepare_transkun_pickles.py
and writes normalized train/val/test pickle files into 4_fine_tuning/model_info/training_data.
"""

from __future__ import annotations

import argparse
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path


def log(message: str) -> None:
    print(message, flush=True)
    print(message, file=sys.stderr, flush=True)


def find_repo_root(start: Path) -> Path:
    for parent in [start] + list(start.parents):
        if (parent / "transkun" / "train.py").exists():
            return parent
    raise FileNotFoundError("Could not find repository root containing transkun/train.py")


def detect_default_workers() -> int:
    slurm = os.environ.get("SLURM_CPUS_ON_NODE")
    if slurm:
        try:
            return max(1, int(slurm))
        except ValueError:
            pass
    return max(1, os.cpu_count() or 1)


def run_checked(cmd: list[str], cwd: Path) -> None:
    log("Running: " + " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def copy_model_params(pretrained_dir: Path, model_params_dir: Path) -> None:
    checkpoint_src = pretrained_dir / "checkpoint.pt"
    model_conf_src = pretrained_dir / "model.conf"

    for path in (checkpoint_src, model_conf_src):
        if not path.exists():
            raise FileNotFoundError(f"Missing required pretrained artifact: {path}")

    model_params_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint_src, model_params_dir / "checkpoint.pt")
    shutil.copy2(model_conf_src, model_params_dir / "model.conf")


def summarize_split_sizes(metadata_dir: Path) -> None:
    split_paths = {
        "train": metadata_dir / "train.pickle",
        "val": metadata_dir / "val.pickle",
        "test": metadata_dir / "test.pickle",
    }

    for split_name, split_path in split_paths.items():
        if not split_path.exists():
            log(f"WARN: {split_name}.pickle is missing at {split_path}")
            continue

        with split_path.open("rb") as f:
            data = pickle.load(f)

        if not isinstance(data, list):
            log(f"WARN: {split_name}.pickle is not a list (type={type(data)})")
            continue

        log(f"Generated {split_name}.pickle samples={len(data)}")
        if len(data) == 0:
            log(f"WARN: {split_name}.pickle is empty")


def generate_model_conf(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "moduleconf.generate", "Model:transkun.ModelTransformer"]
    with output_path.open("w", encoding="utf-8") as f:
        subprocess.run(cmd, check=True, stdout=f)


def ensure_model_conf(metadata_dir: Path, baseline_conf: Path | None = None) -> None:
    dst_conf = metadata_dir / "transkun_base.json"
    dst_conf.parent.mkdir(parents=True, exist_ok=True)

    if baseline_conf is not None and baseline_conf.exists():
        src_conf = baseline_conf.resolve()
        if src_conf == dst_conf.resolve():
            log(f"Using existing model conf: {dst_conf}")
            return

        shutil.copy2(src_conf, dst_conf)
        log(f"Copied baseline model conf: {src_conf} -> {dst_conf}")
        return

    generate_model_conf(dst_conf)
    log(f"Generated model conf: {dst_conf}")


def write_smoke_single_pair_metadata(
    metadata_dir: Path,
    dataset_root: Path,
    midi_rel: str,
    audio_rel: str,
    split_name: str,
) -> None:
    midi_rel_norm = midi_rel.strip().replace("\\", "/")
    audio_rel_norm = audio_rel.strip().replace("\\", "/")
    midi_path = dataset_root / midi_rel_norm
    audio_path = dataset_root / audio_rel_norm

    if not midi_path.exists():
        raise FileNotFoundError(f"Smoke MIDI file not found: {midi_path}")
    if not audio_path.exists():
        raise FileNotFoundError(f"Smoke audio file not found: {audio_path}")

    with wave.open(str(audio_path), "rb") as wf:
        fs = int(wf.getframerate())
        n_samples = int(wf.getnframes())
        n_channel = int(wf.getnchannels())

    sample = {
        "split": split_name,
        "midi_filename": midi_rel_norm,
        "audio_filename": audio_rel_norm,
        "duration": float(n_samples) / float(fs),
        "notes": [],
        "fs": fs,
        "nSamples": n_samples,
        "nChannel": n_channel,
    }

    split_key = split_name.strip().lower()
    train_samples = [sample] if split_key == "train" else []
    val_samples = [sample] if split_key in {"val", "validation"} else []
    test_samples = [sample] if split_key == "test" else []

    metadata_dir.mkdir(parents=True, exist_ok=True)
    with (metadata_dir / "train.pickle").open("wb") as f:
        pickle.dump(train_samples, f, pickle.HIGHEST_PROTOCOL)
    with (metadata_dir / "val.pickle").open("wb") as f:
        pickle.dump(val_samples, f, pickle.HIGHEST_PROTOCOL)
    with (metadata_dir / "test.pickle").open("wb") as f:
        pickle.dump(test_samples, f, pickle.HIGHEST_PROTOCOL)

    log("Smoke metadata generated from a single hard-coded pair")


def main() -> int:
    here = Path(__file__).resolve().parent
    repo_root = find_repo_root(here)

    preprocess_dir = (
        repo_root
        / "eval_utils"
        / "3_retrain_model_metrics"
        / "augmented_retrain"
        / "preprocess_datasets"
    )

    parser = argparse.ArgumentParser(description="Generate fine-tuning artifacts from augmented datasets")
    parser.add_argument("--dataset-root", type=Path, default=Path("/scratch/gilbreth/li5042/datasets"))
    parser.add_argument("--csv-dir", type=Path, default=preprocess_dir / "CSVs")
    parser.add_argument("--metadata-dir", type=Path, default=here / "model_info" / "training_data")
    parser.add_argument("--model-params-dir", type=Path, default=here / "model_info" / "model_params")
    parser.add_argument(
        "--pretrained-dir",
        type=Path,
        default=repo_root / "transkun" / "pretrained" / "transkunV2" / "checkpointMSimpler",
    )
    parser.add_argument("--train-csv", type=Path, default=None)
    parser.add_argument("--maestro-csv", type=Path, default=None)
    parser.add_argument("--smoke-midi-rel", type=str, default=None)
    parser.add_argument("--smoke-audio-rel", type=str, default=None)
    parser.add_argument("--smoke-split", type=str, default="train")
    parser.add_argument("--workers", type=int, default=detect_default_workers())
    parser.add_argument("--rows-per-chunk", type=int, default=512)
    parser.add_argument("--skip-preprocess", action="store_true")
    parser.add_argument("--skip-merge", action="store_true")
    parser.add_argument("--no-pedal-extension", action="store_true")
    parser.add_argument("--skip-model-conf", action="store_true")
    parser.add_argument("--skip-model-params-copy", action="store_true")
    parser.add_argument("--strip-prefix", action="append", default=[])
    parser.add_argument("--fail-on-missing", action="store_true")
    args = parser.parse_args()

    metadata_dir = args.metadata_dir.resolve()
    metadata_dir.mkdir(parents=True, exist_ok=True)

    smoke_requested = (args.smoke_midi_rel is not None) or (args.smoke_audio_rel is not None)
    if smoke_requested:
        if args.smoke_midi_rel is None or args.smoke_audio_rel is None:
            raise ValueError("--smoke-midi-rel and --smoke-audio-rel must be provided together")

        write_smoke_single_pair_metadata(
            metadata_dir=metadata_dir,
            dataset_root=args.dataset_root.resolve(),
            midi_rel=args.smoke_midi_rel,
            audio_rel=args.smoke_audio_rel,
            split_name=args.smoke_split,
        )

        if not args.skip_model_conf:
            baseline_conf = here / "model_info" / "training_data" / "transkun_base.json"
            ensure_model_conf(metadata_dir, baseline_conf=baseline_conf)

        if args.skip_model_params_copy:
            log(f"Skipping model params copy (--skip-model-params-copy). Using: {args.model_params_dir.resolve()}")
        else:
            copy_model_params(args.pretrained_dir.resolve(), args.model_params_dir.resolve())

        summarize_split_sizes(metadata_dir)
        log(f"Metadata generated in: {metadata_dir}")
        log(f"Model params prepared in: {args.model_params_dir.resolve()}")
        return 0

    with tempfile.TemporaryDirectory(prefix="finetune_pickles_raw_") as tmp:
        raw_metadata_dir = Path(tmp) / "raw"

        prepare_cmd = [
            sys.executable,
            str(preprocess_dir / "prepare_transkun_pickles.py"),
            "--dataset-root",
            str(args.dataset_root),
            "--csv-dir",
            str(args.csv_dir),
            "--output-dir",
            str(raw_metadata_dir),
            "--workers",
            str(args.workers),
            "--rows-per-chunk",
            str(args.rows_per_chunk),
        ]
        if args.train_csv is not None:
            prepare_cmd.extend(["--train-csv", str(args.train_csv)])
        if args.maestro_csv is not None:
            prepare_cmd.extend(["--maestro-csv", str(args.maestro_csv)])
        if args.skip_preprocess:
            prepare_cmd.append("--skip-preprocess")
        if args.skip_merge:
            prepare_cmd.append("--skip-merge")
        if args.no_pedal_extension:
            prepare_cmd.append("--no-pedal-extension")
        if args.skip_model_conf:
            prepare_cmd.append("--skip-model-conf")

        run_checked(prepare_cmd, cwd=repo_root)

        preprocess_cmd = [
            sys.executable,
            str(here / "preprocessing.py"),
            "--input-dir",
            str(raw_metadata_dir),
            "--output-dir",
            str(metadata_dir),
            "--dataset-root",
            str(args.dataset_root),
        ]
        for prefix in args.strip_prefix:
            preprocess_cmd.extend(["--strip-prefix", prefix])
        if args.fail_on_missing:
            preprocess_cmd.append("--fail-on-missing")

        run_checked(preprocess_cmd, cwd=repo_root)

        if not args.skip_model_conf:
            src_conf = raw_metadata_dir / "transkun_base.json"
            if not src_conf.exists():
                raise FileNotFoundError(f"Missing generated model conf: {src_conf}")
            ensure_model_conf(metadata_dir, baseline_conf=src_conf)

    if args.skip_model_params_copy:
        log(f"Skipping model params copy (--skip-model-params-copy). Using: {args.model_params_dir.resolve()}")
    else:
        copy_model_params(args.pretrained_dir.resolve(), args.model_params_dir.resolve())

    summarize_split_sizes(metadata_dir)

    log(f"Metadata generated in: {metadata_dir}")
    log(f"Model params prepared in: {args.model_params_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
