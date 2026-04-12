#!/usr/bin/env python3
"""Step 2: Validate fine-tuning artifacts for Transkun training and inference."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path


REQUIRED_SAMPLE_KEYS = {
    "split",
    "midi_filename",
    "audio_filename",
    "duration",
    "notes",
    "fs",
    "nSamples",
    "nChannel",
}


def log(message: str) -> None:
    print(message, flush=True)
    print(message, file=sys.stderr, flush=True)


def validate_pickle(
    path: Path,
    dataset_root: Path,
    split_name: str,
    min_samples: int,
    sample_check_count: int,
) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {split_name} pickle: {path}")

    with path.open("rb") as f:
        data = pickle.load(f)

    if not isinstance(data, list):
        raise TypeError(f"{split_name} pickle must contain a list, got {type(data)}")
    if len(data) == 0:
        log(f"WARN: {split_name} pickle is empty: {path}")
    if len(data) < min_samples:
        raise ValueError(f"{split_name} pickle has {len(data)} samples, expected at least {min_samples}")

    n_check = min(sample_check_count, len(data))
    for idx in range(n_check):
        row = data[idx]
        if not isinstance(row, dict):
            raise TypeError(f"{split_name}[{idx}] is not a dict")

        missing_keys = REQUIRED_SAMPLE_KEYS - set(row.keys())
        if missing_keys:
            raise KeyError(f"{split_name}[{idx}] missing keys: {sorted(missing_keys)}")

        midi_path = dataset_root / str(row["midi_filename"])
        audio_path = dataset_root / str(row["audio_filename"])
        if not midi_path.exists():
            raise FileNotFoundError(f"{split_name}[{idx}] missing midi file: {midi_path}")
        if not audio_path.exists():
            raise FileNotFoundError(f"{split_name}[{idx}] missing audio file: {audio_path}")

        notes = row["notes"]
        if not isinstance(notes, list):
            raise TypeError(f"{split_name}[{idx}].notes is not a list")
        if notes:
            first_note = notes[0]
            for attr in ("start", "end", "pitch", "velocity"):
                if not hasattr(first_note, attr):
                    raise TypeError(f"{split_name}[{idx}].notes[0] missing attribute: {attr}")

    log(f"Validated {split_name}.pickle: {path} (samples={len(data)})")


def main() -> int:
    here = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="Validate fine-tuning artifacts")
    parser.add_argument("--dataset-root", type=Path, default=Path("/scratch/gilbreth/li5042/datasets"))
    parser.add_argument("--metadata-dir", type=Path, default=here / "model_info" / "training_data")
    parser.add_argument("--model-params-dir", type=Path, default=here / "model_info" / "model_params")
    parser.add_argument("--sample-check-count", type=int, default=10)
    parser.add_argument("--min-train-samples", type=int, default=100)
    parser.add_argument("--min-val-samples", type=int, default=10)
    parser.add_argument("--min-test-samples", type=int, default=10)
    args = parser.parse_args()

    metadata_dir = args.metadata_dir.resolve()
    model_params_dir = args.model_params_dir.resolve()

    log(f"Using metadata directory: {metadata_dir}")
    log(f"Using model params directory: {model_params_dir}")

    validate_pickle(
        metadata_dir / "train.pickle",
        args.dataset_root,
        "train",
        args.min_train_samples,
        args.sample_check_count,
    )
    validate_pickle(
        metadata_dir / "val.pickle",
        args.dataset_root,
        "val",
        args.min_val_samples,
        args.sample_check_count,
    )
    validate_pickle(
        metadata_dir / "test.pickle",
        args.dataset_root,
        "test",
        args.min_test_samples,
        args.sample_check_count,
    )

    metadata_conf = metadata_dir / "transkun_base.json"
    if not metadata_conf.exists():
        raise FileNotFoundError(f"Missing model conf: {metadata_conf}")
    log(f"Validated model conf: {metadata_conf}")

    checkpoint_path = model_params_dir / "checkpoint.pt"
    model_conf_path = model_params_dir / "model.conf"
    for path in (checkpoint_path, model_conf_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing required model params artifact: {path}")
    log(f"Validated model params: {checkpoint_path} and {model_conf_path}")

    log("All fine-tuning artifacts validated successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
