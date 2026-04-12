#!/usr/bin/env python3
"""Step 3: Launch Transkun training using native transkun.train."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def log_both(message: str) -> None:
    print(message, flush=True)
    print(message, file=sys.stderr, flush=True)


def find_repo_root(start: Path) -> Path:
    for parent in [start] + list(start.parents):
        if (parent / "transkun" / "train.py").exists():
            return parent
    raise FileNotFoundError("Could not find repository root containing transkun/train.py")


def resolve_metadata_dir(output_root: Path) -> Path:
    """Metadata may be in output_root/metadata or directly in output_root."""
    nested = output_root / "metadata"
    if (nested / "train.pickle").exists() and (nested / "val.pickle").exists():
        return nested
    return output_root


def detect_n_process() -> int:
    """Auto-detect GPU count from SLURM or CUDA environment."""
    slurm_gpus = os.environ.get("SLURM_GPUS_ON_NODE")
    if slurm_gpus:
        try:
            return max(1, int(slurm_gpus))
        except ValueError:
            pass

    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices:
        devices = [device for device in visible_devices.split(",") if device.strip()]
        if devices:
            return len(devices)

    try:
        import torch
        count = torch.cuda.device_count()
        if count > 0:
            return count
    except Exception:
        pass

    return 1


def main() -> int:
    here = Path(__file__).resolve().parent
    repo_root = find_repo_root(here)

    parser = argparse.ArgumentParser(description="Train Transkun using native transkun.train")
    parser.add_argument("--dataset-root", type=Path, default=Path("/scratch/gilbreth/li5042/datasets"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=repo_root / "eval_utils" / "3_retrain_model_metrics" / "output" / "AUGMENTED_METADATA",
    )
    parser.add_argument("--checkpoint-name", default="checkpoint_augmented.pt")
    parser.add_argument("--n-process", type=int, default=0)
    parser.add_argument("--n-iter", type=int, default=180000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--dataloader-workers", type=int, default=8)
    parser.add_argument("--max-lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--hop-size", type=float)
    parser.add_argument("--chunk-size", type=float)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--noise-folder", type=Path)
    parser.add_argument("--ir-folder", type=Path)
    parser.add_argument("--force-fresh", action="store_true")
    args = parser.parse_args()

    metadata_dir = resolve_metadata_dir(args.output_root)
    checkpoint_dir = args.output_root / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_pickle = metadata_dir / "train.pickle"
    val_pickle = metadata_dir / "val.pickle"
    model_conf = metadata_dir / "transkun_base.json"

    log_both(f"Using metadata directory: {metadata_dir}")

    for p in (train_pickle, val_pickle, model_conf):
        if not p.exists():
            raise FileNotFoundError(f"Required artifact missing: {p}")

    checkpoint_path = checkpoint_dir / args.checkpoint_name
    if args.force_fresh and checkpoint_path.exists():
        checkpoint_path.unlink()

    n_process = args.n_process if args.n_process and args.n_process > 0 else detect_n_process()
    log_both(f"Using n-process: {n_process}")

    # Call native transkun.train with all arguments forwarded.
    cmd = [
        sys.executable,
        "-m",
        "transkun.train",
        "--nProcess",
        str(n_process),
        "--datasetPath",
        str(args.dataset_root),
        "--datasetMetaFile_train",
        str(train_pickle),
        "--datasetMetaFile_val",
        str(val_pickle),
        "--modelConf",
        str(model_conf),
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

    cmd.append(str(checkpoint_path))

    log_both("Running: " + " ".join(cmd))
    subprocess.run(cmd, cwd=str(repo_root), check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
