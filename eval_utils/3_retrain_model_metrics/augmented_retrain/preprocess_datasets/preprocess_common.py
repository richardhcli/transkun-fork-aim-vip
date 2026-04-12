#!/usr/bin/env python3
"""Shared helpers for generating MAESTRO-style CSV metadata files."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

CSV_FIELDS = ["split", "midi_filename", "audio_filename"]


@dataclass
class PairingStats:
    midi_count: int
    audio_count: int
    paired_count: int
    ambiguous_stems: int
    missing_audio_for_midi: int
    rejected_multi_instrument_midi: int = 0
    rejected_invalid_midi: int = 0


def as_posix_relative(path: Path, root: Path) -> str:
    return str(path.relative_to(root).as_posix())


def write_rows(rows: Sequence[Dict[str, str]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def read_maestro_minimal_rows(maestro_csv: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with maestro_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "split": row["split"],
                    "midi_filename": row["midi_filename"],
                    "audio_filename": row["audio_filename"],
                }
            )
    return rows


def collect_files(root: Path, extensions: Iterable[str]) -> List[Path]:
    exts = {e.lower() for e in extensions}
    return sorted(
        [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    )


def filter_single_instrument_midis(midi_files: Sequence[Path]) -> Tuple[List[Path], int, int]:
    """Keep only MIDI files with exactly one instrument.

    Returns (filtered_files, rejected_multi_instrument, rejected_invalid_midi).
    """
    filtered: List[Path] = []
    rejected_multi = 0
    rejected_invalid = 0

    try:
        import pretty_midi  # pylint: disable=import-outside-toplevel
    except Exception as exc:
        raise RuntimeError(
            "pretty_midi is required to validate single-instrument MIDI files"
        ) from exc

    for midi_path in midi_files:
        try:
            midi_obj = pretty_midi.PrettyMIDI(str(midi_path))
        except Exception:
            rejected_invalid += 1
            continue

        if len(midi_obj.instruments) == 1:
            filtered.append(midi_path)
        else:
            rejected_multi += 1

    return filtered, rejected_multi, rejected_invalid


def pair_by_stem(
    dataset_root: Path,
    midi_exts: Iterable[str] = (".mid", ".midi"),
    audio_exts: Iterable[str] = (".wav", ".flac", ".mp3", ".ogg"),
    split: str = "train",
) -> Tuple[List[Dict[str, str]], PairingStats]:
    midi_files_all = collect_files(dataset_root, midi_exts)
    midi_files, rejected_multi, rejected_invalid = filter_single_instrument_midis(midi_files_all)
    audio_files = collect_files(dataset_root, audio_exts)

    midi_by_stem: Dict[str, List[Path]] = {}
    audio_by_stem: Dict[str, List[Path]] = {}

    for m in midi_files:
        midi_by_stem.setdefault(m.stem, []).append(m)
    for a in audio_files:
        audio_by_stem.setdefault(a.stem, []).append(a)

    rows: List[Dict[str, str]] = []
    ambiguous = 0

    for stem in sorted(set(midi_by_stem) & set(audio_by_stem)):
        mids = midi_by_stem[stem]
        auds = audio_by_stem[stem]
        if len(mids) == 1 and len(auds) == 1:
            rows.append(
                {
                    "split": split,
                    "midi_filename": as_posix_relative(mids[0], dataset_root),
                    "audio_filename": as_posix_relative(auds[0], dataset_root),
                }
            )
        else:
            ambiguous += 1

    rows.sort(key=lambda r: r["midi_filename"])

    stats = PairingStats(
        midi_count=len(midi_files),
        audio_count=len(audio_files),
        paired_count=len(rows),
        ambiguous_stems=ambiguous,
        missing_audio_for_midi=sum(1 for stem in midi_by_stem if stem not in audio_by_stem),
        rejected_multi_instrument_midi=rejected_multi,
        rejected_invalid_midi=rejected_invalid,
    )
    return rows, stats


def print_summary(dataset_name: str, output_csv: Path, rows: Sequence[Dict[str, str]], stats: PairingStats) -> None:
    print(f"[{dataset_name}] Wrote {len(rows)} rows -> {output_csv}")
    print(
        f"[{dataset_name}] MIDI files: {stats.midi_count}, audio files: {stats.audio_count}, paired: {stats.paired_count}"
    )
    print(
        f"[{dataset_name}] Rejected MIDI files: multi-instrument={stats.rejected_multi_instrument_midi}, invalid={stats.rejected_invalid_midi}"
    )
    print(
        f"[{dataset_name}] Missing audio stems for MIDI: {stats.missing_audio_for_midi}, ambiguous stems: {stats.ambiguous_stems}"
    )
