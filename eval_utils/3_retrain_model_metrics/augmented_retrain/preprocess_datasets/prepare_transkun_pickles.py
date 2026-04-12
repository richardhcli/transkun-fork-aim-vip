#!/usr/bin/env python3
"""Build Transkun metadata by wrapping native preprocess and Data helpers."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
import os
import pickle
import subprocess
import sys
import tempfile
import wave
from pathlib import Path
from typing import Dict, List, Tuple


DATASET_PREPROCESSORS = [
    ("A-MAPS", "preprocess_dataset_A-MAPS.py"),
    ("MAESTRO", "preprocess_dataset_MAESTRO.py"),
    ("SMD_v2", "preprocess_dataset_SMD_v2.py"),
    ("msmd_data", "preprocess_dataset_msmd_data.py"),
    ("BiMMuDa", "preprocess_dataset_BiMMuDa.py"),
    ("POP909", "preprocess_dataset_POP909.py"),
]


def log(msg: str) -> None:
    print(msg, flush=True)
    print(msg, file=sys.stderr, flush=True)


def ensure_transkun_importable() -> None:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "transkun").is_dir():
            parent_str = str(parent)
            if parent_str not in sys.path:
                sys.path.insert(0, parent_str)
            return
    raise FileNotFoundError("Could not find repository root containing transkun package")


def load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [
            {
                "split": row["split"].strip(),
                "midi_filename": row["midi_filename"].strip(),
                "audio_filename": row["audio_filename"].strip(),
            }
            for row in reader
        ]


def with_dataset_prefix(rows: List[Dict[str, str]], dataset_name: str) -> List[Dict[str, str]]:
    """Prefix relative midi/audio paths with dataset_name when missing.

    MAESTRO rows are commonly stored as "2018/..." while this pipeline uses
    a global dataset root (e.g. /scratch/.../datasets), which expects
    "MAESTRO/2018/...".
    """

    dataset_prefix = dataset_name.strip().strip("/")
    prefixed_rows: List[Dict[str, str]] = []

    for row in rows:
        new_row = dict(row)
        for key in ("midi_filename", "audio_filename"):
            value = str(new_row.get(key, "")).strip().replace("\\", "/")
            if not value:
                continue

            parts = [p for p in value.split("/") if p]
            if not parts:
                continue

            if parts[0].lower() != dataset_prefix.lower():
                new_row[key] = f"{dataset_prefix}/{'/'.join(parts)}"
            else:
                new_row[key] = "/".join(parts)

        prefixed_rows.append(new_row)

    return prefixed_rows


def write_rows(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "midi_filename", "audio_filename"])
        writer.writeheader()
        writer.writerows(rows)


def run_subprocess(cmd: List[str], cwd: Path) -> None:
    log("Running: " + " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def run_preprocess_scripts(script_dir: Path, dataset_root: Path, csv_dir: Path) -> None:
    for dataset_name, script_name in DATASET_PREPROCESSORS:
        script_path = script_dir / script_name
        output_csv = csv_dir / f"{dataset_name}.csv"

        cmd = [sys.executable, str(script_path), "--output-csv", str(output_csv)]
        if dataset_name == "MAESTRO":
            cmd.extend(["--maestro-csv", str(dataset_root / "MAESTRO" / "maestro-v3.0.0.csv")])
        else:
            cmd.extend(["--dataset-root", str(dataset_root / dataset_name)])

        run_subprocess(cmd, cwd=script_dir)


def merge_training_csv(script_dir: Path, csv_dir: Path) -> Path:
    merged_csv = csv_dir / "merged_train.csv"
    cmd = [
        sys.executable,
        str(script_dir / "merge_dataset_csvs.py"),
        "--csv-dir",
        str(csv_dir),
        "--output-csv",
        str(merged_csv),
    ]
    run_subprocess(cmd, cwd=script_dir)
    return merged_csv


def filter_44100_wav_rows(rows: List[Dict[str, str]], dataset_root: Path) -> Tuple[List[Dict[str, str]], int, int, int]:
    kept: List[Dict[str, str]] = []
    non_wav = 0
    missing = 0
    wrong_rate = 0

    for row in rows:
        audio_path = dataset_root / row["audio_filename"]
        if not audio_path.exists():
            missing += 1
            continue
        if audio_path.suffix.lower() != ".wav":
            non_wav += 1
            continue
        try:
            with wave.open(str(audio_path), "rb") as wf:
                fs = wf.getframerate()
        except wave.Error:
            non_wav += 1
            continue

        if fs != 44100:
            wrong_rate += 1
            continue

        kept.append(row)

    return kept, non_wav, missing, wrong_rate


def build_maestro_like_json(path: Path, samples: List[Dict[str, object]]) -> None:
    rows = [
        {
            "split": str(sample["split"]),
            "midi_filename": str(sample["midi_filename"]),
            "audio_filename": str(sample["audio_filename"]),
        }
        for sample in samples
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f)


def add_duration(samples: List[Dict[str, object]]) -> None:
    for sample in samples:
        if "duration" not in sample:
            sample["duration"] = float(sample["nSamples"]) / float(sample["fs"])


def detect_default_workers() -> int:
    slurm = os.environ.get("SLURM_CPUS_ON_NODE")
    if slurm:
        try:
            return max(1, int(slurm))
        except ValueError:
            pass
    try:
        return max(1, os.cpu_count() or 1)
    except Exception:
        return 1


def chunk_rows(rows: List[Dict[str, str]], rows_per_chunk: int) -> List[List[Dict[str, str]]]:
    n = max(1, rows_per_chunk)
    return [rows[i : i + n] for i in range(0, len(rows), n)]


def _process_chunk(
    chunk_rows_list: List[Dict[str, str]],
    dataset_root_str: str,
    extend_pedal: bool,
    worker_tmp_dir_str: str,
    chunk_index: int,
) -> Tuple[int, str, int, int]:
    ensure_transkun_importable()
    from transkun import Data  # pylint: disable=import-outside-toplevel

    worker_tmp_dir = Path(worker_tmp_dir_str)
    csv_path = worker_tmp_dir / f"chunk_{chunk_index:05d}.csv"
    out_path = worker_tmp_dir / f"chunk_{chunk_index:05d}.pickle"

    write_rows(csv_path, chunk_rows_list)
    samples = Data.createDatasetMaestroCSV(
        str(dataset_root_str),
        str(csv_path),
        extendSustainPedal=extend_pedal,
    )
    add_duration(samples)

    with out_path.open("wb") as f:
        pickle.dump(samples, f, pickle.HIGHEST_PROTOCOL)

    return chunk_index, str(out_path), len(chunk_rows_list), len(samples)


def create_samples_parallel(
    rows: List[Dict[str, str]],
    dataset_root: Path,
    extend_pedal: bool,
    workers: int,
    rows_per_chunk: int,
    name: str,
) -> List[Dict[str, object]]:
    if not rows:
        log(f"No rows for {name}; skipping parsing.")
        return []

    workers_eff = max(1, workers)
    if workers_eff == 1:
        ensure_transkun_importable()
        from transkun import Data  # pylint: disable=import-outside-toplevel

        with tempfile.TemporaryDirectory(prefix=f"{name}_single_") as tmp:
            tmp_csv = Path(tmp) / f"{name}.csv"
            write_rows(tmp_csv, rows)
            samples = Data.createDatasetMaestroCSV(
                str(dataset_root),
                str(tmp_csv),
                extendSustainPedal=extend_pedal,
            )
            add_duration(samples)
            return samples

    chunks = chunk_rows(rows, rows_per_chunk)
    max_workers = min(workers_eff, len(chunks))
    log(
        f"Parsing {name} metadata in parallel: rows={len(rows)} chunks={len(chunks)} workers={max_workers}"
    )

    merged: List[Dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix=f"{name}_chunks_") as tmp:
        tmp_dir = Path(tmp)
        results: Dict[int, str] = {}
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(
                    _process_chunk,
                    chunk,
                    str(dataset_root),
                    extend_pedal,
                    str(tmp_dir),
                    idx,
                ): idx
                for idx, chunk in enumerate(chunks)
            }

            for future in as_completed(future_to_idx):
                idx, out_path, in_rows, out_rows = future.result()
                results[idx] = out_path
                log(
                    f"[{name}] chunk {idx + 1}/{len(chunks)} done: input_rows={in_rows} output_samples={out_rows}"
                )

        for idx in range(len(chunks)):
            out_path = Path(results[idx])
            with out_path.open("rb") as f:
                merged.extend(pickle.load(f))

    log(f"Completed parallel parse for {name}: samples={len(merged)}")
    return merged


def write_split_pickles(output_dir: Path, samples: List[Dict[str, object]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    train = [s for s in samples if s.get("split") == "train"]
    val = [s for s in samples if s.get("split") == "validation"]
    test = [s for s in samples if s.get("split") == "test"]

    with (output_dir / "train.pickle").open("wb") as f:
        pickle.dump(train, f, pickle.HIGHEST_PROTOCOL)
    with (output_dir / "val.pickle").open("wb") as f:
        pickle.dump(val, f, pickle.HIGHEST_PROTOCOL)
    with (output_dir / "test.pickle").open("wb") as f:
        pickle.dump(test, f, pickle.HIGHEST_PROTOCOL)

    log(f"Wrote {len(train)} samples -> {output_dir / 'train.pickle'}")
    log(f"Wrote {len(val)} samples -> {output_dir / 'val.pickle'}")
    log(f"Wrote {len(test)} samples -> {output_dir / 'test.pickle'}")


def generate_model_conf(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "moduleconf.generate", "Model:transkun.ModelTransformer"]
    with output_path.open("w", encoding="utf-8") as f:
        subprocess.run(cmd, check=True, stdout=f)


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    default_output = Path(
        "/scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/3_retrain_model_metrics/output/AUGMENTED_METADATA"
    )

    parser = argparse.ArgumentParser(description="Build Transkun pickles via native Data CSV/JSON paths")
    parser.add_argument("--dataset-root", type=Path, default=Path("/scratch/gilbreth/li5042/datasets"))
    parser.add_argument("--csv-dir", type=Path, default=script_dir / "CSVs")
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument("--train-csv", type=Path, default=None)
    parser.add_argument("--maestro-csv", type=Path, default=None)
    parser.add_argument("--json-path", type=Path, default=None)
    parser.add_argument("--no-pedal-extension", action="store_true")
    parser.add_argument("--workers", type=int, default=detect_default_workers())
    parser.add_argument("--rows-per-chunk", type=int, default=512)
    parser.add_argument("--skip-preprocess", action="store_true")
    parser.add_argument("--skip-merge", action="store_true")
    parser.add_argument("--skip-model-conf", action="store_true")
    args = parser.parse_args()

    ensure_transkun_importable()
    from transkun import Data  # pylint: disable=import-outside-toplevel

    csv_dir = args.csv_dir
    csv_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_preprocess:
        log("Generating per-dataset CSV files...")
        run_preprocess_scripts(script_dir, args.dataset_root, csv_dir)

    if args.train_csv is None:
        if args.skip_merge:
            train_csv = csv_dir / "merged_train.csv"
        else:
            log("Merging per-dataset CSV files into merged_train.csv...")
            train_csv = merge_training_csv(script_dir, csv_dir)
    else:
        train_csv = args.train_csv

    maestro_csv = args.maestro_csv if args.maestro_csv is not None else (csv_dir / "MAESTRO.csv")

    if not train_csv.exists():
        raise FileNotFoundError(f"Missing train CSV: {train_csv}")
    if not maestro_csv.exists():
        raise FileNotFoundError(f"Missing MAESTRO CSV: {maestro_csv}")

    log("Filtering CSV rows to WAV files at 44.1kHz...")
    train_rows = load_rows(train_csv)
    maestro_rows = with_dataset_prefix(load_rows(maestro_csv), "MAESTRO")

    train_filtered, train_non_wav, train_missing, train_wrong_rate = filter_44100_wav_rows(
        train_rows, args.dataset_root
    )
    maestro_filtered, maestro_non_wav, maestro_missing, maestro_wrong_rate = filter_44100_wav_rows(
        maestro_rows, args.dataset_root
    )

    train_44100_csv = csv_dir / "merged_train_44100.csv"
    maestro_44100_csv = csv_dir / "MAESTRO_44100.csv"
    write_rows(train_44100_csv, train_filtered)
    write_rows(maestro_44100_csv, maestro_filtered)

    log(
        "Train CSV filter stats: "
        f"input={len(train_rows)} kept={len(train_filtered)} non_wav={train_non_wav} "
        f"missing={train_missing} wrong_rate={train_wrong_rate}"
    )
    log(
        "MAESTRO CSV filter stats: "
        f"input={len(maestro_rows)} kept={len(maestro_filtered)} non_wav={maestro_non_wav} "
        f"missing={maestro_missing} wrong_rate={maestro_wrong_rate}"
    )

    extend_pedal = not args.no_pedal_extension

    log("Loading filtered CSV metadata with Data.createDatasetMaestroCSV...")
    train_samples = create_samples_parallel(
        rows=train_filtered,
        dataset_root=args.dataset_root,
        extend_pedal=extend_pedal,
        workers=args.workers,
        rows_per_chunk=args.rows_per_chunk,
        name="train",
    )
    maestro_samples = create_samples_parallel(
        rows=maestro_filtered,
        dataset_root=args.dataset_root,
        extend_pedal=extend_pedal,
        workers=args.workers,
        rows_per_chunk=args.rows_per_chunk,
        name="maestro",
    )

    maestro_like_json = args.json_path if args.json_path is not None else (args.output_dir / "augmented_maestro_like.json")
    build_maestro_like_json(maestro_like_json, train_samples + maestro_samples)
    log(f"Wrote maestro-like JSON -> {maestro_like_json}")

    log("Re-loading maestro-like JSON with Data.createDatasetMaestro...")
    all_samples = Data.createDatasetMaestro(
        str(args.dataset_root),
        str(maestro_like_json),
        extendSustainPedal=extend_pedal,
    )
    add_duration(all_samples)

    write_split_pickles(args.output_dir, all_samples)

    if not args.skip_model_conf:
        conf_path = args.output_dir / "transkun_base.json"
        generate_model_conf(conf_path)
        log(f"Wrote model conf -> {conf_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
