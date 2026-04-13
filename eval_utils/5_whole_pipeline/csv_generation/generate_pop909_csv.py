#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from csv_generation_common import (
    pair_by_stem,
    print_dataset_summary,
    transform_rows_for_transkun,
    write_csv_rows,
)


def main() -> int:
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="Generate POP909 CSV for Transkun pickle generation")
    parser.add_argument("--dataset-root", type=Path, default=Path("/scratch/gilbreth/li5042/datasets/POP909"))
    parser.add_argument("--output-csv", type=Path, default=script_dir / "CSVs" / "POP909.csv")
    parser.add_argument("--logs-dir", type=Path, default=script_dir / "logs")
    args = parser.parse_args()

    raw_rows = pair_by_stem(args.dataset_root, split="train")
    rows, stats = transform_rows_for_transkun(
        dataset_name="POP909",
        dataset_root=args.dataset_root,
        raw_rows=raw_rows,
        logs_dir=args.logs_dir,
    )

    write_csv_rows(rows, args.output_csv)
    print_dataset_summary("POP909", args.output_csv, stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
