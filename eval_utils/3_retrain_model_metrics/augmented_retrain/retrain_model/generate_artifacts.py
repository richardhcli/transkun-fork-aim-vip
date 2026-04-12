#!/usr/bin/env python3
"""Step 1: Generate Transkun training artifacts via prepare_transkun_pickles wrapper."""

from __future__ import annotations

import argparse
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


def main() -> int:
    here = Path(__file__).resolve().parent
    repo_root = find_repo_root(here)

    parser = argparse.ArgumentParser(description="Generate train/val/test artifacts using prepare_transkun_pickles")
    parser.add_argument("--dataset-root", type=Path, default=Path("/scratch/gilbreth/li5042/datasets"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=repo_root / "eval_utils" / "3_retrain_model_metrics" / "output" / "AUGMENTED_METADATA",
    )
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--maestro-csv",
        type=Path,
        default=None,
    )
    parser.add_argument("--skip-preprocess", action="store_true")
    parser.add_argument("--skip-merge", action="store_true")
    parser.add_argument("--no-pedal-extension", action="store_true")
    args = parser.parse_args()

    preprocess_dir = (
        repo_root
        / "eval_utils"
        / "3_retrain_model_metrics"
        / "augmented_retrain"
        / "preprocess_datasets"
    )
    csv_dir = preprocess_dir / "CSVs"

    cmd = [
        sys.executable,
        str(preprocess_dir / "prepare_transkun_pickles.py"),
        "--dataset-root",
        str(args.dataset_root),
        "--csv-dir",
        str(csv_dir),
        "--output-dir",
        str(args.output_root),
    ]

    if args.train_csv is not None:
        cmd.extend(["--train-csv", str(args.train_csv)])
    if args.maestro_csv is not None:
        cmd.extend(["--maestro-csv", str(args.maestro_csv)])

    if args.skip_preprocess:
        cmd.append("--skip-preprocess")
    if args.skip_merge:
        cmd.append("--skip-merge")
    if args.no_pedal_extension:
        cmd.append("--no-pedal-extension")

    log_both("Generating artifacts via prepare_transkun_pickles.py")
    log_both("Running: " + " ".join(cmd))
    subprocess.run(cmd, cwd=str(repo_root), check=True)

    log_both(f"Artifacts generated in: {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
