#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from preprocess_common import pair_by_stem, print_summary, write_rows


DEFAULT_ROOT = Path("/scratch/gilbreth/li5042/datasets/A-MAPS")
DEFAULT_OUT = Path(
    "/scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/3_retrain_model_metrics/augmented_retrain/CSVs/A-MAPS.csv"
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate MAESTRO-style CSV for A-MAPS")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    rows, stats = pair_by_stem(args.dataset_root, split="train")
    write_rows(rows, args.output_csv)
    print_summary("A-MAPS", args.output_csv, rows, stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
