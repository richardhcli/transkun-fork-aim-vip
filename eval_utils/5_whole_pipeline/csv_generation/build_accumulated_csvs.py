#!/usr/bin/env python3
"""Build accumulated CSVs from raw per-dataset CSV files.

This module is intentionally separated from per-dataset CSV generation.
It consumes the JSON path map emitted by generate_all_csv.py (output_json)
and writes three derived CSV files:

1) entire.csv
   - Accumulates all rows from all dataset CSVs.

2) user_testing.csv
   - Contains exactly one row (the first deterministic row) from each dataset CSV.
   - This is intended as lightweight cross-dataset test coverage.

3) smoketest.csv
   - Contains exactly one row (the first deterministic row) from the MAESTRO CSV.

For compatibility with transkun Data.createDatasetMaestroCSV-style path usage,
this builder converts each midi/audio path to an absolute path by joining
(dataset_root + relative_path) for each dataset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

from csv_generation_common import read_csv_rows, write_csv_rows


SPLIT_PRIORITY = {"train": 0, "validation": 1, "val": 1, "test": 2}


def log(msg: str) -> None:
    print(msg, flush=True)


def sorted_rows(rows: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    return sorted(
        rows,
        key=lambda row: (
            SPLIT_PRIORITY.get(str(row["split"]).strip().lower(), 99),
            str(row["midi_filename"]),
            str(row["audio_filename"]),
        ),
    )


def normalize_rel_for_dataset(path_value: str, dataset_name: str) -> str:
    """Normalize a CSV path and strip accidental dataset-name prefix.

    Per-dataset CSVs should be dataset-root-relative. If a row accidentally
    contains "<dataset_name>/...", strip that prefix before path joining.
    """

    normalized = str(path_value).strip().replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]

    normalized = normalized.lstrip("/")
    parts = [part for part in normalized.split("/") if part]
    if parts and parts[0].lower() == dataset_name.lower():
        parts = parts[1:]

    return "/".join(parts)


def to_absolute_row(row: Dict[str, str], dataset_root: Path, dataset_name: str) -> Dict[str, str]:
    """Convert dataset-relative midi/audio paths to absolute paths.

    This mirrors the idea used in Data.createDatasetMaestroCSV:
    resolve midi/audio by prefixing with datasetPath (dataset root).
    """

    midi_raw = str(row["midi_filename"]) if row.get("midi_filename") is not None else ""
    audio_raw = str(row["audio_filename"]) if row.get("audio_filename") is not None else ""

    midi_path = Path(midi_raw)
    audio_path = Path(audio_raw)

    if midi_path.is_absolute():
        midi_abs = midi_path
    else:
        midi_rel = normalize_rel_for_dataset(midi_raw, dataset_name)
        midi_abs = dataset_root / midi_rel

    if audio_path.is_absolute():
        audio_abs = audio_path
    else:
        audio_rel = normalize_rel_for_dataset(audio_raw, dataset_name)
        audio_abs = dataset_root / audio_rel

    return {
        "split": str(row["split"]).strip(),
        "midi_filename": str(midi_abs.resolve()),
        "audio_filename": str(audio_abs.resolve()),
    }


def _choose_output_path(
    explicit_path: Path | None,
    combined_csvs: Dict[str, str],
    key: str,
    default_dir: Path,
) -> Path:
    if explicit_path is not None:
        return explicit_path.resolve()

    mapped = combined_csvs.get(key)
    if mapped:
        return Path(mapped).resolve()

    return (default_dir / f"{key}.csv").resolve()


def build_accumulated_csvs_from_map(
    dataset_root_map_json: Path,
    entire_csv: Path | None = None,
    user_testing_csv: Path | None = None,
    smoketest_csv: Path | None = None,
) -> Dict[str, Path]:
    with dataset_root_map_json.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    dataset_roots_raw = payload.get("dataset_roots", {})
    dataset_csvs_raw = payload.get("dataset_csvs", {})
    combined_csvs_raw = payload.get("combined_csvs", {})
    dataset_order = payload.get("dataset_order") or list(dataset_csvs_raw.keys())

    if not dataset_roots_raw:
        raise ValueError(f"dataset_root_map.json missing dataset_roots: {dataset_root_map_json}")
    if not dataset_csvs_raw:
        raise ValueError(f"dataset_root_map.json missing dataset_csvs: {dataset_root_map_json}")

    dataset_roots = {name: Path(path).resolve() for name, path in dataset_roots_raw.items()}
    dataset_csvs = {name: Path(path).resolve() for name, path in dataset_csvs_raw.items()}

    default_output_dir = dataset_root_map_json.resolve().parent
    entire_out = _choose_output_path(entire_csv, combined_csvs_raw, "entire", default_output_dir)
    user_testing_out = _choose_output_path(user_testing_csv, combined_csvs_raw, "user_testing", default_output_dir)
    smoketest_out = _choose_output_path(smoketest_csv, combined_csvs_raw, "smoketest", default_output_dir)

    entire_rows: List[Dict[str, str]] = []
    user_testing_rows: List[Dict[str, str]] = []
    maestro_rows_abs: List[Dict[str, str]] = []

    for dataset_name in dataset_order:
        if dataset_name not in dataset_roots:
            raise KeyError(f"Missing dataset root for {dataset_name} in {dataset_root_map_json}")
        if dataset_name not in dataset_csvs:
            raise KeyError(f"Missing dataset CSV for {dataset_name} in {dataset_root_map_json}")

        dataset_root = dataset_roots[dataset_name]
        dataset_csv_path = dataset_csvs[dataset_name]

        if not dataset_csv_path.exists():
            raise FileNotFoundError(f"Dataset CSV not found for {dataset_name}: {dataset_csv_path}")

        raw_rows = read_csv_rows(dataset_csv_path)
        ordered_rows = sorted_rows(raw_rows)
        abs_rows = [to_absolute_row(row, dataset_root, dataset_name) for row in ordered_rows]

        # entire.csv accumulates all rows from all datasets.
        entire_rows.extend(abs_rows)

        # user_testing.csv includes one representative row from each dataset.
        if not abs_rows:
            log(f"Warning: Dataset {dataset_name} produced no rows; skipping from user_testing.csv")
        else:
            user_testing_rows.append(abs_rows[0])

        if dataset_name.lower() == "maestro":
            maestro_rows_abs = abs_rows

        log(f"Accumulation input: {dataset_name} rows={len(abs_rows)}")

    # smoketest.csv uses one representative row from MAESTRO only.
    if not maestro_rows_abs:
        raise RuntimeError("MAESTRO rows are required to build smoketest.csv")
    smoketest_rows = [maestro_rows_abs[0]]

    write_csv_rows(entire_rows, entire_out)
    write_csv_rows(user_testing_rows, user_testing_out)
    write_csv_rows(smoketest_rows, smoketest_out)

    log(f"Wrote entire CSV ({len(entire_rows)} rows): {entire_out}")
    log(f"Wrote user testing CSV ({len(user_testing_rows)} rows): {user_testing_out}")
    log(f"Wrote smoketest CSV ({len(smoketest_rows)} rows): {smoketest_out}")

    return {
        "entire": entire_out,
        "user_testing": user_testing_out,
        "smoketest": smoketest_out,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build accumulated CSVs from dataset_root_map.json")
    parser.add_argument("--dataset-root-map-json", type=Path, default="/scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/5_whole_pipeline/pickle_generation/CSVs/dataset_root_map.json")
    parser.add_argument("--entire-csv", type=Path, default=None)
    parser.add_argument("--user-testing-csv", type=Path, default=None)
    parser.add_argument("--smoketest-csv", type=Path, default=None)
    args = parser.parse_args()

    build_accumulated_csvs_from_map(
        dataset_root_map_json=args.dataset_root_map_json.resolve(),
        entire_csv=args.entire_csv,
        user_testing_csv=args.user_testing_csv,
        smoketest_csv=args.smoketest_csv,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
