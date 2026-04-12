#!/usr/bin/env python3
"""Merge per-dataset CSV metadata into one training CSV with optional weighting."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Dict, List

CSV_FIELDS = ["split", "midi_filename", "audio_filename"]

DATASET_FOLDER_MAP = {
    "A-MAPS": "A-MAPS",
    "MAESTRO": "MAESTRO",
    "SMD_v2": "SMD_v2",
    "msmd_data": "msmd_data",
    "BiMMuDa": "BiMMuDa",
    "POP909": "POP909",
}


def parse_weights(raw_weights: List[str]) -> Dict[str, float]:
    weights: Dict[str, float] = {}
    for item in raw_weights:
        if "=" not in item:
            raise ValueError(f"Invalid --weight '{item}'. Expected DATASET=VALUE")
        name, value = item.split("=", 1)
        name = name.strip()
        if name not in DATASET_FOLDER_MAP:
            raise ValueError(f"Unknown dataset in --weight: {name}")
        w = float(value)
        if w < 0:
            raise ValueError(f"Weight must be >= 0, got {w} for {name}")
        weights[name] = w
    return weights


def read_rows(dataset_csv: Path, dataset_name: str, keep_non_train: bool) -> List[Dict[str, str]]:
    folder_prefix = DATASET_FOLDER_MAP[dataset_name]
    out: List[Dict[str, str]] = []
    with dataset_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = row["split"].strip()
            if split != "train" and not keep_non_train:
                continue
            out.append(
                {
                    "split": "train",
                    "midi_filename": f"{folder_prefix}/{row['midi_filename'].strip()}",
                    "audio_filename": f"{folder_prefix}/{row['audio_filename'].strip()}",
                }
            )
    return out


def apply_weight(rows: List[Dict[str, str]], weight: float, rng: random.Random) -> List[Dict[str, str]]:
    if weight <= 0 or not rows:
        return []

    target_n = int(round(len(rows) * weight))
    if target_n <= 0:
        return []
    if target_n == len(rows):
        return list(rows)
    if target_n < len(rows):
        return rng.sample(rows, target_n)
    return [rng.choice(rows) for _ in range(target_n)]


def write_rows(rows: List[Dict[str, str]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    here = Path(__file__).resolve().parent
    csv_dir = here / "CSVs"

    parser = argparse.ArgumentParser(description="Merge per-dataset CSVs into one training CSV")
    parser.add_argument("--csv-dir", type=Path, default=csv_dir)
    parser.add_argument("--output-csv", type=Path, default=csv_dir / "merged_train.csv")
    parser.add_argument("--datasets", nargs="+", default=list(DATASET_FOLDER_MAP.keys()))
    parser.add_argument("--weight", action="append", default=[], help="Optional dataset weight: DATASET=VALUE")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep-maestro-non-train", action="store_true", help="Include MAESTRO validation/test rows in merged training CSV")
    parser.add_argument("--no-shuffle", action="store_true")
    args = parser.parse_args()

    for d in args.datasets:
        if d not in DATASET_FOLDER_MAP:
            raise ValueError(f"Unsupported dataset: {d}")

    weights = parse_weights(args.weight)
    rng = random.Random(args.seed)
    merged: List[Dict[str, str]] = []

    print("Merging CSVs...")
    for dataset_name in args.datasets:
        csv_path = args.csv_dir / f"{dataset_name}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Dataset CSV not found: {csv_path}")

        base_rows = read_rows(
            csv_path,
            dataset_name,
            keep_non_train=(dataset_name == "MAESTRO" and args.keep_maestro_non_train),
        )
        weight = weights.get(dataset_name, 1.0)
        final_rows = apply_weight(base_rows, weight, rng)
        merged.extend(final_rows)

        print(
            f"  - {dataset_name}: base={len(base_rows)} weight={weight} merged={len(final_rows)}"
        )

    if not args.no_shuffle:
        rng.shuffle(merged)

    write_rows(merged, args.output_csv)
    print(f"Wrote merged training CSV: {args.output_csv}")
    print(f"Total merged training rows: {len(merged)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
