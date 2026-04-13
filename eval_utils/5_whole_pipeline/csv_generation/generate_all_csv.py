#!/usr/bin/env python3
"""Generate all per-dataset CSVs and concatenated CSV outputs for pickle generation."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Sequence, Tuple


def resolve_pickle_generation_dir() -> Path:
	main_script_dir = os.environ.get("MAIN_SCRIPT_DIR", "").strip()
	if not main_script_dir:
		raise EnvironmentError(
			"MAIN_SCRIPT_DIR is not set. Export MAIN_SCRIPT_DIR in the launcher shell before running generate_all_csv.py"
		)

	script_dir = Path(main_script_dir).expanduser().resolve() / "pickle_generation"
	if not script_dir.exists():
		raise FileNotFoundError(f"pickle_generation directory not found under MAIN_SCRIPT_DIR: {script_dir}")

	return script_dir


SCRIPT_DIR = resolve_pickle_generation_dir()
script_dir_str = str(SCRIPT_DIR)
if script_dir_str not in sys.path:
	sys.path.insert(0, script_dir_str)

from build_accumulated_csvs import build_accumulated_csvs_from_map
from csv_generation_common import read_csv_rows


DATASET_GENERATORS: Sequence[Tuple[str, str, str]] = (
	("A-MAPS", "A-MAPS", "generate_a_maps_csv.py"),
	("BiMMuDa", "BiMMuDa", "generate_bimmuda_csv.py"),
	("MAESTRO", "MAESTRO", "generate_maestro_csv.py"),
	("POP909", "POP909", "generate_pop909_csv.py"),
	("SMD_v2", "SMD_v2", "generate_smd_v2_csv.py"),
	("msmd_data", "msmd_data", "generate_msmd_data_csv.py"),
)


def log(msg: str) -> None:
	print(msg, flush=True)


def run_generator(
	python_bin: str,
	script_dir: Path,
	datasets_root: Path,
	output_dir: Path,
	logs_dir: Path,
	dataset_name: str,
	dataset_folder: str,
	script_name: str,
) -> Path:
	script_path = script_dir / script_name
	if not script_path.exists():
		raise FileNotFoundError(f"Generator script not found: {script_path}")

	dataset_root = datasets_root / dataset_folder
	output_csv = output_dir / f"{dataset_name}.csv"

	cmd = [
		python_bin,
		str(script_path),
		"--dataset-root",
		str(dataset_root),
		"--output-csv",
		str(output_csv),
		"--logs-dir",
		str(logs_dir),
	]

	if dataset_name == "MAESTRO":
		cmd.extend(["--maestro-csv", str(dataset_root / "maestro-v3.0.0.csv")])

	log(f"Running {dataset_name} generator: {' '.join(cmd)}")
	subprocess.run(cmd, check=True)
	return output_csv


def write_dataset_root_map_json(
	output_json: Path,
	datasets_root: Path,
	per_dataset_csv_paths: Dict[str, Path],
	entire_csv: Path,
	user_testing_csv: Path,
	smoketest_csv: Path,
) -> None:
	payload = {
		"datasets_root": str(datasets_root.resolve()),
		"csv_path_prefix_semantics": "Paths in CSV files are relative to each dataset root in dataset_roots.",
		"dataset_order": [dataset_name for dataset_name, _folder, _script in DATASET_GENERATORS],
		"dataset_roots": {
			dataset_name: str((datasets_root / dataset_name).resolve())
			for dataset_name, _dataset_folder, _script_name in DATASET_GENERATORS
		},
		"dataset_csvs": {k: str(v.resolve()) for k, v in per_dataset_csv_paths.items()},
		"combined_csvs": {
			"entire": str(entire_csv.resolve()),
			"user_testing": str(user_testing_csv.resolve()),
			"smoketest": str(smoketest_csv.resolve()),
		},
	}

	output_json.parent.mkdir(parents=True, exist_ok=True)
	with output_json.open("w", encoding="utf-8") as f:
		json.dump(payload, f, indent=2, sort_keys=True)

	log(f"Wrote dataset root map JSON: {output_json}")


def main() -> int:
	script_dir = SCRIPT_DIR

	parser = argparse.ArgumentParser(description="Generate all per-dataset and concatenated CSVs")
	parser.add_argument("--datasets-root", type=Path, default=Path("/scratch/gilbreth/li5042/datasets"))
	parser.add_argument("--output-dir", type=Path, default=script_dir / "CSVs")
	parser.add_argument("--logs-dir", type=Path, default=script_dir / "logs")
	parser.add_argument("--python-bin", type=str, default=sys.executable)
	parser.add_argument("--entire-csv", type=Path, default=None)
	parser.add_argument("--user-testing-csv", type=Path, default=None)
	parser.add_argument("--smoketest-csv", type=Path, default=None)
	parser.add_argument("--dataset-root-map-json", type=Path, default=None)
	args = parser.parse_args()

	output_dir = args.output_dir.resolve()
	logs_dir = args.logs_dir.resolve()
	output_dir.mkdir(parents=True, exist_ok=True)
	logs_dir.mkdir(parents=True, exist_ok=True)

	entire_csv = args.entire_csv.resolve() if args.entire_csv is not None else (output_dir / "entire.csv")
	user_testing_csv = (
		args.user_testing_csv.resolve() if args.user_testing_csv is not None else (output_dir / "user_testing.csv")
	)
	smoketest_csv = (
		args.smoketest_csv.resolve() if args.smoketest_csv is not None else (output_dir / "smoketest.csv")
	)
	dataset_root_map_json = (
		args.dataset_root_map_json.resolve()
		if args.dataset_root_map_json is not None
		else (output_dir / "dataset_root_map.json")
	)

	per_dataset_csv_paths: Dict[str, Path] = {}

	for dataset_name, dataset_folder, script_name in DATASET_GENERATORS:
		output_csv = run_generator(
			python_bin=args.python_bin,
			script_dir=script_dir,
			datasets_root=args.datasets_root.resolve(),
			output_dir=output_dir,
			logs_dir=logs_dir,
			dataset_name=dataset_name,
			dataset_folder=dataset_folder,
			script_name=script_name,
		)
		per_dataset_csv_paths[dataset_name] = output_csv
		rows = read_csv_rows(output_csv)
		log(f"Loaded {len(rows)} rows from {output_csv}")

	# Export path-prefix metadata as soon as per-dataset CSVs are available.
	write_dataset_root_map_json(
		dataset_root_map_json,
		args.datasets_root.resolve(),
		per_dataset_csv_paths,
		entire_csv,
		user_testing_csv,
		smoketest_csv,
	)

	# Refactored accumulation step: build combined CSVs from already-generated
	# per-dataset raw CSVs described by output_json.
	build_accumulated_csvs_from_map(
		dataset_root_map_json,
		entire_csv=entire_csv,
		user_testing_csv=user_testing_csv,
		smoketest_csv=smoketest_csv,
	)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
