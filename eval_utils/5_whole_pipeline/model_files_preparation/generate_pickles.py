#!/usr/bin/env python3
"""Build train/val/test pickle files from a MAESTRO-compatible CSV.

CSV format must include columns:
- split
- midi_filename
- audio_filename

If validation/test rows are missing, this script copies one train row into each
missing split so Transkun always receives non-empty train/val/test pickles.
"""

from __future__ import annotations

import argparse
import csv
import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


def env_path(
    name: str,
    default: Path | str | None = None,
    *,
    must_exist: bool = False,
    expect_dir: bool | None = None,
) -> Path:
    raw = os.environ.get(name, "").strip()
    if not raw:
        if default is None:
            raise EnvironmentError(f"{name} is not set")
        raw = str(default)

    path = Path(raw).expanduser().resolve()
    if must_exist and not path.exists():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    if expect_dir is True and path.exists() and not path.is_dir():
        raise NotADirectoryError(f"{name} is not a directory: {path}")
    if expect_dir is False and path.exists() and not path.is_file():
        raise FileNotFoundError(f"{name} is not a file: {path}")

    return path


CSV_FIELDS: Tuple[str, str, str] = ("split", "midi_filename", "audio_filename")

WORKING_DIR = env_path("WORKING_DIR", must_exist=True, expect_dir=True)
if not (WORKING_DIR / "transkun" / "train.py").exists():
    raise FileNotFoundError(
        f"WORKING_DIR must point to transkun_fork repo root: {WORKING_DIR}"
    )

WHOLE_PIPELINE_DIR = env_path(
    "WHOLE_PIPELINE_DIR",
    default=WORKING_DIR / "eval_utils" / "5_whole_pipeline",
    must_exist=True,
    expect_dir=True,
)
CSV_DIR = env_path(
    "CSV_DIR",
    default=WHOLE_PIPELINE_DIR / "csv_generation" / "CSVs",
    must_exist=True,
    expect_dir=True,
)
OUTPUT_DIR = env_path(
    "OUTPUT_DIR",
    default=WHOLE_PIPELINE_DIR / "output",
    must_exist=False,
)

DEFAULT_DATASET_ROOT = env_path(
    "DATASET_ROOT",
    default="/scratch/gilbreth/li5042/datasets",
    must_exist=False,
)
DEFAULT_INPUT_CSV = env_path(
    "INPUT_CSV_FULL",
    default=CSV_DIR / "entire.csv",
    must_exist=False,
)
DEFAULT_OUTPUT_DIR = env_path(
    "PICKLE_DIR_FULL",
    default=OUTPUT_DIR / "pickles" / "full",
    must_exist=False,
)


def log(message: str) -> None:
    print(message, flush=True)


def normalize_path(value: str) -> str:
    text = str(value).strip().replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    return text


def load_rows(input_csv: Path) -> List[Dict[str, str]]:
    with input_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing_columns = [field for field in CSV_FIELDS if field not in (reader.fieldnames or [])]
        if missing_columns:
            raise ValueError(f"CSV is missing required columns: {missing_columns}")

        rows: List[Dict[str, str]] = []
        for row in reader:
            rows.append(
                {
                    "split": str(row["split"]).strip().lower(),
                    "midi_filename": normalize_path(str(row["midi_filename"])),
                    "audio_filename": normalize_path(str(row["audio_filename"])),
                }
            )

    return rows


def split_rows(rows: Sequence[Dict[str, str]]) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
    train_rows: List[Dict[str, str]] = []
    val_rows: List[Dict[str, str]] = []
    test_rows: List[Dict[str, str]] = []

    for row in rows:
        split = row["split"]
        canonical = dict(row)

        if split == "train":
            canonical["split"] = "train"
            train_rows.append(canonical)
        elif split in {"validation", "val"}:
            canonical["split"] = "validation"
            val_rows.append(canonical)
        elif split == "test":
            canonical["split"] = "test"
            test_rows.append(canonical)
        else:
            raise ValueError(
                f"Unsupported split value: {split!r}. Allowed: train, validation/val, test"
            )

    if not train_rows:
        raise ValueError("Input CSV must contain at least one train row")

    seed_row = dict(train_rows[0])

    if not val_rows:
        fallback = dict(seed_row)
        fallback["split"] = "validation"
        val_rows = [fallback]
        log("Validation split was empty; copied one train row into validation.")

    if not test_rows:
        fallback = dict(seed_row)
        fallback["split"] = "test"
        test_rows = [fallback]
        log("Test split was empty; copied one train row into test.")

    return train_rows, val_rows, test_rows


def write_split_csv(path: Path, rows: Sequence[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(CSV_FIELDS))
        writer.writeheader()
        writer.writerows(rows)


def add_duration(samples: List[Dict[str, object]]) -> None:
    for sample in samples:
        if "duration" not in sample:
            sample["duration"] = float(sample["nSamples"]) / float(sample["fs"])


def create_samples_for_split(
    data_module,
    dataset_root: Path,
    split_csv: Path,
    extend_pedal: bool,
) -> List[Dict[str, object]]:
    samples = data_module.createDatasetMaestroCSV(
        str(dataset_root),
        str(split_csv),
        extendSustainPedal=extend_pedal,
    )
    add_duration(samples)
    return samples


def dump_pickle(path: Path, samples: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(samples, f, pickle.HIGHEST_PROTOCOL)


def generate_model_conf(output_dir: Path) -> Path:
    conf_path = output_dir / "transkun_base.json"
    cmd = [sys.executable, "-m", "moduleconf.generate", "Model:transkun.ModelTransformer"]
    with conf_path.open("w", encoding="utf-8") as f:
        subprocess.run(cmd, check=True, stdout=f)
    return conf_path


def main() -> int:
    repo_root = WORKING_DIR
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from transkun import Data  # pylint: disable=import-outside-toplevel

    parser = argparse.ArgumentParser(description="Build train/val/test pickles from one MAESTRO-style CSV")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Root used to resolve relative midi/audio paths",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help="CSV with split,midi_filename,audio_filename",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write train/val/test pickles",
    )
    parser.add_argument("--no-pedal-extension", action="store_true", help="Disable sustain pedal extension")
    parser.add_argument("--skip-model-conf", action="store_true", help="Skip generating transkun_base.json")
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    input_csv = args.input_csv.resolve()
    output_dir = args.output_dir.resolve()

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV does not exist: {input_csv}")

    rows = load_rows(input_csv)
    train_rows, val_rows, test_rows = split_rows(rows)

    log(
        "Split counts after fallback: "
        f"train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}"
    )

    extend_pedal = not args.no_pedal_extension

    with tempfile.TemporaryDirectory(prefix="universal_pickles_") as tmp:
        tmp_dir = Path(tmp)
        train_csv = tmp_dir / "train.csv"
        val_csv = tmp_dir / "val.csv"
        test_csv = tmp_dir / "test.csv"

        write_split_csv(train_csv, train_rows)
        write_split_csv(val_csv, val_rows)
        write_split_csv(test_csv, test_rows)
        
        import pretty_midi

        def check_midi_in_rows(rows: List[Dict[str, str]], root: Path, split_name: str) -> None:
            for row in rows:
                orig_midi_path = root / row["midi_filename"]
                if not orig_midi_path.exists():
                    continue
                try:
                    pm = pretty_midi.PrettyMIDI(str(orig_midi_path))
                    if len(pm.instruments) > 1:
                        log(f"WARNING: Multi-track MIDI detected during dataloader generation (should have been pre-flattened)! File: {orig_midi_path}")
                except Exception as e:
                    log(f"WARNING: Failed to parse {orig_midi_path}: {e}")

        log("Verifying MIDI files are flattened...")
        check_midi_in_rows(train_rows, dataset_root, "train")
        check_midi_in_rows(val_rows, dataset_root, "val")
        check_midi_in_rows(test_rows, dataset_root, "test")

        train_samples = create_samples_for_split(Data, dataset_root, train_csv, extend_pedal)
        val_samples = create_samples_for_split(Data, dataset_root, val_csv, extend_pedal)
        test_samples = create_samples_for_split(Data, dataset_root, test_csv, extend_pedal)

    if not train_samples:
        raise RuntimeError("Generated train samples are empty; cannot continue")
    if not val_samples:
        raise RuntimeError("Generated val samples are empty; fallback should have prevented this")
    if not test_samples:
        raise RuntimeError("Generated test samples are empty; fallback should have prevented this")

    dump_pickle(output_dir / "train.pickle", train_samples)
    dump_pickle(output_dir / "val.pickle", val_samples)
    dump_pickle(output_dir / "test.pickle", test_samples)

    log(f"Wrote train.pickle with {len(train_samples)} samples")
    log(f"Wrote val.pickle with {len(val_samples)} samples")
    log(f"Wrote test.pickle with {len(test_samples)} samples")

    if not args.skip_model_conf:
        conf_path = generate_model_conf(output_dir)
        log(f"Wrote model conf: {conf_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
