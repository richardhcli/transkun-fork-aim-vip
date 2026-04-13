#!/usr/bin/env python3
"""Shared helpers for Transkun-compatible CSV generation."""

from __future__ import annotations

import csv
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

CSV_FIELDS = ["split", "midi_filename", "audio_filename"]
TARGET_SAMPLE_RATE_HZ = 44100
GENERATED_FLATTENED_MIDI_DIR = ".transkun_flattened_midis"


@dataclass
class TransformStats:
    raw_rows: int = 0
    kept_rows: int = 0
    skipped_missing_midi: int = 0
    skipped_missing_audio: int = 0
    skipped_non_wav_audio: int = 0
    skipped_invalid_wav: int = 0
    skipped_wrong_sample_rate: int = 0


def normalize_rel_path(path_str: str) -> str:
    value = str(path_str).strip().replace("\\", "/")
    while value.startswith("./"):
        value = value[2:]
    value = value.lstrip("/")
    return "/".join([part for part in value.split("/") if part])


def strip_dataset_prefix(path_str: str, dataset_name: str) -> str:
    normalized = normalize_rel_path(path_str)
    parts = normalized.split("/") if normalized else []
    if parts and parts[0].lower() == dataset_name.lower():
        return "/".join(parts[1:])
    return normalized


def read_csv_rows(csv_path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "split": str(row["split"]).strip(),
                    "midi_filename": normalize_rel_path(str(row["midi_filename"])),
                    "audio_filename": normalize_rel_path(str(row["audio_filename"])),
                }
            )
    return rows


def write_csv_rows(rows: Sequence[Dict[str, str]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _path_has_dirname(path: Path, dirname: str) -> bool:
    dirname_lower = dirname.lower()
    return any(part.lower() == dirname_lower for part in path.parts)


def collect_files(root: Path, extensions: Iterable[str]) -> List[Path]:
    exts = {e.lower() for e in extensions}
    return sorted(
        [
            p
            for p in root.rglob("*")
            if p.is_file()
            and p.suffix.lower() in exts
            and not _path_has_dirname(p, GENERATED_FLATTENED_MIDI_DIR)
        ]
    )


def pair_by_stem(
    dataset_root: Path,
    split: str = "train",
    midi_exts: Iterable[str] = (".mid", ".midi"),
    audio_exts: Iterable[str] = (".wav",),
) -> List[Dict[str, str]]:
    midi_files = collect_files(dataset_root, midi_exts)
    audio_files = collect_files(dataset_root, audio_exts)

    midi_by_stem: Dict[str, List[Path]] = {}
    audio_by_stem: Dict[str, List[Path]] = {}

    for midi_path in midi_files:
        midi_by_stem.setdefault(midi_path.stem, []).append(midi_path)
    for audio_path in audio_files:
        audio_by_stem.setdefault(audio_path.stem, []).append(audio_path)

    rows: List[Dict[str, str]] = []
    for stem in sorted(set(midi_by_stem) & set(audio_by_stem)):
        midi_matches = midi_by_stem[stem]
        audio_matches = audio_by_stem[stem]
        if len(midi_matches) != 1 or len(audio_matches) != 1:
            continue

        midi_rel = midi_matches[0].relative_to(dataset_root).as_posix()
        audio_rel = audio_matches[0].relative_to(dataset_root).as_posix()
        rows.append(
            {
                "split": split,
                "midi_filename": midi_rel,
                "audio_filename": audio_rel,
            }
        )

    rows.sort(key=lambda item: item["midi_filename"])
    return rows


def pair_smd_v2(dataset_root: Path) -> List[Dict[str, str]]:
    midi_root = dataset_root / "midi"
    audio_root = dataset_root / "wav_44100_stereo"
    if not audio_root.exists():
        audio_root = dataset_root / "wav"

    midi_files = sorted(
        [
            p
            for p in midi_root.rglob("*")
            if p.suffix.lower() in {".mid", ".midi"}
            and not _path_has_dirname(p, GENERATED_FLATTENED_MIDI_DIR)
        ]
    )
    audio_by_stem = {
        p.stem: p
        for p in sorted([p for p in audio_root.rglob("*") if p.suffix.lower() == ".wav"])
    }

    rows: List[Dict[str, str]] = []
    for midi_path in midi_files:
        audio_path = audio_by_stem.get(midi_path.stem)
        if audio_path is None:
            continue

        rows.append(
            {
                "split": "train",
                "midi_filename": midi_path.relative_to(dataset_root).as_posix(),
                "audio_filename": audio_path.relative_to(dataset_root).as_posix(),
            }
        )

    rows.sort(key=lambda item: item["midi_filename"])
    return rows


def read_maestro_rows(maestro_csv: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with maestro_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "split": str(row["split"]).strip(),
                    "midi_filename": normalize_rel_path(str(row["midi_filename"])),
                    "audio_filename": normalize_rel_path(str(row["audio_filename"])),
                }
            )
    rows.sort(key=lambda item: (item["split"], item["midi_filename"]))
    return rows


def check_wav_sample_rate(audio_abs_path: Path) -> Tuple[bool, int | None, str]:
    if not audio_abs_path.exists():
        return False, None, "missing_audio"

    if audio_abs_path.suffix.lower() != ".wav":
        return False, None, "non_wav_audio"

    try:
        with wave.open(str(audio_abs_path), "rb") as wf:
            hz = int(wf.getframerate())
    except (wave.Error, EOFError, OSError):
        return False, None, "invalid_wav"

    if hz != TARGET_SAMPLE_RATE_HZ:
        return False, hz, "wrong_sample_rate"

    return True, hz, "ok"


def _write_wav_rate_log(path: Path, issues: Sequence[Dict[str, str]]) -> None:
    fields = ["dataset", "audio_filename", "actual_hz", "absolute_audio_path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(issues)


def _write_skip_log(path: Path, issues: Sequence[Dict[str, str]]) -> None:
    fields = ["dataset", "reason", "midi_filename", "audio_filename", "details"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(issues)


def transform_rows_for_transkun(
    dataset_name: str,
    dataset_root: Path,
    raw_rows: Sequence[Dict[str, str]],
    logs_dir: Path,
) -> Tuple[List[Dict[str, str]], TransformStats]:
    stats = TransformStats(raw_rows=len(raw_rows))

    accepted: List[Dict[str, str]] = []
    wav_rate_issues: List[Dict[str, str]] = []
    skipped_rows: List[Dict[str, str]] = []

    for row in raw_rows:
        split = str(row["split"]).strip()
        midi_rel = normalize_rel_path(str(row["midi_filename"]))
        audio_rel = normalize_rel_path(str(row["audio_filename"]))

        midi_abs = dataset_root / midi_rel
        audio_abs = dataset_root / audio_rel

        if not midi_abs.exists():
            stats.skipped_missing_midi += 1
            skipped_rows.append(
                {
                    "dataset": dataset_name,
                    "reason": "missing_midi",
                    "midi_filename": midi_rel,
                    "audio_filename": audio_rel,
                    "details": str(midi_abs),
                }
            )
            continue

        ok, hz, reason = check_wav_sample_rate(audio_abs)
        if not ok:
            if reason == "missing_audio":
                stats.skipped_missing_audio += 1
            elif reason == "non_wav_audio":
                stats.skipped_non_wav_audio += 1
            elif reason == "invalid_wav":
                stats.skipped_invalid_wav += 1
            elif reason == "wrong_sample_rate":
                stats.skipped_wrong_sample_rate += 1
                wav_rate_issues.append(
                    {
                        "dataset": dataset_name,
                        "audio_filename": audio_rel,
                        "actual_hz": "" if hz is None else str(hz),
                        "absolute_audio_path": str(audio_abs),
                    }
                )

            skipped_rows.append(
                {
                    "dataset": dataset_name,
                    "reason": reason,
                    "midi_filename": midi_rel,
                    "audio_filename": audio_rel,
                    "details": "" if hz is None else str(hz),
                }
            )
            continue

        accepted.append(
            {
                "split": split,
                "midi_filename": midi_rel,
                "audio_filename": audio_rel,
            }
        )

    accepted.sort(key=lambda item: (item["split"], item["midi_filename"]))
    stats.kept_rows = len(accepted)

    wav_log_path = logs_dir / f"{dataset_name}_wav_rate_issues.csv"
    skip_log_path = logs_dir / f"{dataset_name}_skipped_rows.csv"
    _write_wav_rate_log(wav_log_path, wav_rate_issues)
    _write_skip_log(skip_log_path, skipped_rows)

    return accepted, stats


def print_dataset_summary(dataset_name: str, output_csv: Path, stats: TransformStats) -> None:
    print(f"[{dataset_name}] raw={stats.raw_rows} kept={stats.kept_rows}")
    print(
        f"[{dataset_name}] skipped: missing_midi={stats.skipped_missing_midi} "
        f"missing_audio={stats.skipped_missing_audio} "
        f"non_wav={stats.skipped_non_wav_audio} invalid_wav={stats.skipped_invalid_wav} "
        f"wrong_rate={stats.skipped_wrong_sample_rate}"
    )
    print(f"[{dataset_name}] wrote CSV -> {output_csv}")
