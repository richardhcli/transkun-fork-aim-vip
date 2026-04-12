#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from preprocess_common import PairingStats, as_posix_relative, print_summary, write_rows


DEFAULT_ROOT = Path("/scratch/gilbreth/li5042/datasets/SMD_v2")
DEFAULT_OUT = Path(
    "/scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/3_retrain_model_metrics/augmented_retrain/CSVs/SMD_v2.csv"
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate MAESTRO-style CSV for SMD_v2")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    midi_root = args.dataset_root / "midi"
    audio_root = args.dataset_root / "wav_44100_stereo"
    if not audio_root.exists():
        audio_root = args.dataset_root / "wav"

    midi_files = sorted([p for p in midi_root.rglob("*") if p.suffix.lower() in {".mid", ".midi"}])
    audio_by_stem = {
        p.stem: p
        for p in sorted([p for p in audio_root.rglob("*") if p.suffix.lower() in {".wav", ".flac", ".mp3", ".ogg"}])
    }

    rows = []
    missing_audio = 0
    for midi_path in midi_files:
        audio_path = audio_by_stem.get(midi_path.stem)
        if audio_path is None:
            missing_audio += 1
            continue

        rows.append(
            {
                "split": "train",
                "midi_filename": as_posix_relative(midi_path, args.dataset_root),
                "audio_filename": as_posix_relative(audio_path, args.dataset_root),
            }
        )

    stats = PairingStats(
        midi_count=len(midi_files),
        audio_count=len(audio_by_stem),
        paired_count=len(rows),
        ambiguous_stems=0,
        missing_audio_for_midi=missing_audio,
    )

    write_rows(rows, args.output_csv)
    print_summary("SMD_v2", args.output_csv, rows, stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
