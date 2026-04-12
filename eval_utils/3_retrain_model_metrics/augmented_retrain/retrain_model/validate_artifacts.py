#!/usr/bin/env python3
"""Step 2: Validate generated artifacts match Transkun train.py expectations."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path


def log_both(message: str) -> None:
    print(message, flush=True)
    print(message, file=sys.stderr, flush=True)


def resolve_metadata_dir(output_root: Path) -> Path:
    nested = output_root / "metadata"
    if (nested / "train.pickle").exists() and (nested / "val.pickle").exists():
        return nested
    return output_root


def validate_pickle(
    path: Path,
    dataset_root: Path,
    name: str,
    min_samples: int,
    sample_check_count: int,
) -> None:
    required = {"split", "midi_filename", "audio_filename", "duration", "notes", "fs", "nSamples", "nChannel"}
    log_both(f"Validating {name} pickle: {path}")

    if not path.exists():
        raise FileNotFoundError(f"Missing {name} pickle: {path}")

    with path.open("rb") as f:
        data = pickle.load(f)

    log_both(f"Loaded {name} pickle with {len(data)} samples")

    if not isinstance(data, list):
        raise TypeError(f"{name} pickle must contain a list, got {type(data)}")
    if len(data) < min_samples:
        raise ValueError(f"{name} pickle has {len(data)} samples, expected at least {min_samples}")

    check_n = min(sample_check_count, len(data))
    for i in range(check_n):
        row = data[i]
        if not isinstance(row, dict):
            raise TypeError(f"{name}[{i}] is not a dict")
        missing_keys = required - set(row.keys())
        if missing_keys:
            raise KeyError(f"{name}[{i}] missing keys: {sorted(missing_keys)}")

        midi_path = dataset_root / str(row["midi_filename"])
        audio_path = dataset_root / str(row["audio_filename"])
        if not midi_path.exists():
            raise FileNotFoundError(f"{name}[{i}] missing midi file: {midi_path}")
        if not audio_path.exists():
            raise FileNotFoundError(f"{name}[{i}] missing audio file: {audio_path}")

        notes = row["notes"]
        if not isinstance(notes, list):
            raise TypeError(f"{name}[{i}].notes is not a list")
        if notes:
            first = notes[0]
            for attr in ("start", "end", "pitch", "velocity"):
                if not hasattr(first, attr):
                    raise TypeError(f"{name}[{i}].notes[0] missing attribute: {attr}")

    log_both(f"Validated {name} pickle: {path} (n={len(data)})")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate generated Transkun artifacts")
    parser.add_argument("--dataset-root", type=Path, default=Path("/scratch/gilbreth/li5042/datasets"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/3_retrain_model_metrics/output/AUGMENTED_METADATA"),
    )
    parser.add_argument("--sample-check-count", type=int, default=10)
    parser.add_argument("--min-train-samples", type=int, default=100)
    parser.add_argument("--min-val-samples", type=int, default=10)
    parser.add_argument("--min-test-samples", type=int, default=10)
    args = parser.parse_args()

    metadata_dir = resolve_metadata_dir(args.output_root)
    train_pickle = metadata_dir / "train.pickle"
    val_pickle = metadata_dir / "val.pickle"
    test_pickle = metadata_dir / "test.pickle"
    model_conf = metadata_dir / "transkun_base.json"

    log_both(f"Using metadata directory: {metadata_dir}")

    validate_pickle(train_pickle, args.dataset_root, "train", args.min_train_samples, args.sample_check_count)
    validate_pickle(val_pickle, args.dataset_root, "val", args.min_val_samples, args.sample_check_count)
    validate_pickle(test_pickle, args.dataset_root, "test", args.min_test_samples, args.sample_check_count)

    if not model_conf.exists():
        raise FileNotFoundError(f"Missing model conf: {model_conf}")
    log_both(f"Validated model conf: {model_conf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
