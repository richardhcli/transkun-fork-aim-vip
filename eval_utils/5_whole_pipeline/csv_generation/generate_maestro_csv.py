#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

def main() -> int:
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="Generate MAESTRO CSV for Transkun pickle generation")
    parser.add_argument("--dataset-root", type=Path, default=Path("/scratch/gilbreth/li5042/datasets/MAESTRO"))
    parser.add_argument("--maestro-csv", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=script_dir / "CSVs" / "MAESTRO.csv")
    parser.add_argument("--logs-dir", type=Path, default=script_dir / "logs")
    args = parser.parse_args()

    # Kept for CLI compatibility with other generators and orchestrator.
    _ = args.logs_dir

    maestro_csv = args.maestro_csv if args.maestro_csv is not None else (args.dataset_root / "maestro-v3.0.0.csv")
    if not maestro_csv.exists():
        raise FileNotFoundError(f"MAESTRO metadata CSV not found: {maestro_csv}")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    if args.output_csv.exists():
        # If a previous run preserved read-only bits, make target writable before overwrite.
        os.chmod(args.output_csv, 0o644)
    shutil.copyfile(maestro_csv, args.output_csv)
    os.chmod(args.output_csv, 0o644)
    print(f"[MAESTRO] Copied source CSV -> {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
