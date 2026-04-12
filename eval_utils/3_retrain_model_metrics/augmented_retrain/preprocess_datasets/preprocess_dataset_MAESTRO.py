#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from preprocess_common import read_maestro_minimal_rows, write_rows


DEFAULT_MAESTRO_CSV = Path("/scratch/gilbreth/li5042/datasets/MAESTRO/maestro-v3.0.0.csv")
DEFAULT_OUT = Path(
    "/scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/3_retrain_model_metrics/augmented_retrain/CSVs/MAESTRO.csv"
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate MAESTRO-style CSV for MAESTRO")
    parser.add_argument("--maestro-csv", type=Path, default=DEFAULT_MAESTRO_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    rows = read_maestro_minimal_rows(args.maestro_csv)
    write_rows(rows, args.output_csv)
    print(f"[MAESTRO] Wrote {len(rows)} rows -> {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
