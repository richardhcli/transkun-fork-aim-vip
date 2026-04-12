#!/usr/bin/env python3
"""Normalize and validate Transkun metadata pickle files.

This utility updates train/val/test pickle entries so audio/midi paths are
consistent with the dataset root used during fine-tuning.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Iterable


def normalize_rel_path(path_str: str, strip_prefixes: tuple[str, ...]) -> str:
	path = path_str.replace("\\", "/")
	parts = [p for p in path.split("/") if p]
	if not parts:
		return path_str

	strip_prefixes_lower = {p.lower() for p in strip_prefixes if p}
	while parts and parts[0].lower() in strip_prefixes_lower:
		parts = parts[1:]

	return "/".join(parts)


def _summarize_missing(label: str, missing: Iterable[str]) -> None:
	missing_unique = sorted(set(missing))
	print(f"[{label}] missing files: {len(missing_unique)}")
	if missing_unique:
		print(f"[{label}] first missing file: {missing_unique[0]}")


def process_split_file(
	split_path: Path,
	output_dir: Path,
	strip_prefixes: tuple[str, ...],
	dataset_root: Path | None,
	fail_on_missing: bool,
) -> None:
	with split_path.open("rb") as f:
		entries = pickle.load(f)

	if not isinstance(entries, list):
		raise TypeError(f"Expected list in {split_path}, got {type(entries)}")

	n_changed_audio = 0
	n_changed_midi = 0
	missing_audio = []
	missing_midi = []

	for item in entries:
		if not isinstance(item, dict):
			continue

		audio = item.get("audio_filename")
		if isinstance(audio, str):
			normalized_audio = normalize_rel_path(audio, strip_prefixes)
			if normalized_audio != audio:
				n_changed_audio += 1
			item["audio_filename"] = normalized_audio
			if dataset_root is not None:
				audio_path = dataset_root / item["audio_filename"]
				if not audio_path.exists():
					missing_audio.append(str(audio_path))

		midi = item.get("midi_filename")
		if isinstance(midi, str):
			normalized_midi = normalize_rel_path(midi, strip_prefixes)
			if normalized_midi != midi:
				n_changed_midi += 1
			item["midi_filename"] = normalized_midi
			if dataset_root is not None:
				midi_path = dataset_root / item["midi_filename"]
				if not midi_path.exists():
					missing_midi.append(str(midi_path))

	output_dir.mkdir(parents=True, exist_ok=True)
	output_path = output_dir / split_path.name
	with output_path.open("wb") as f:
		pickle.dump(entries, f, pickle.HIGHEST_PROTOCOL)

	print(f"[{split_path.name}] total entries: {len(entries)}")
	print(f"[{split_path.name}] normalized audio paths: {n_changed_audio}")
	print(f"[{split_path.name}] normalized midi paths: {n_changed_midi}")

	if dataset_root is not None:
		_summarize_missing(f"{split_path.name}:audio", missing_audio)
		_summarize_missing(f"{split_path.name}:midi", missing_midi)
		if fail_on_missing and (missing_audio or missing_midi):
			raise FileNotFoundError(
				f"Missing files found while validating {split_path.name}; "
				"re-run with corrected dataset root or disable --fail-on-missing"
			)


def main() -> None:
	parser = argparse.ArgumentParser(description="Normalize Transkun metadata pickle paths.")
	parser.add_argument(
		"--input-dir",
		type=Path,
		required=True,
		help="Directory containing train.pickle, val.pickle, test.pickle",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=None,
		help="Directory to write normalized pickle files",
	)
	parser.add_argument(
		"--strip-prefix",
		action="append",
		default=[],
		help="Leading path prefix to remove (can be passed multiple times)",
	)
	parser.add_argument(
		"--dataset-root",
		type=Path,
		default=None,
		help="Optional dataset root used to validate normalized paths",
	)
	parser.add_argument(
		"--in-place",
		action="store_true",
		help="Overwrite pickle files in input dir instead of writing to a separate output dir",
	)
	parser.add_argument(
		"--fail-on-missing",
		action="store_true",
		help="Fail if any normalized audio/midi file path does not exist under dataset root",
	)
	args = parser.parse_args()

	input_dir = args.input_dir.resolve()
	if args.in_place and args.output_dir is not None:
		raise ValueError("Use either --in-place or --output-dir, not both")
	if not args.in_place and args.output_dir is None:
		raise ValueError("Either --in-place or --output-dir must be provided")

	output_dir = input_dir if args.in_place else args.output_dir.resolve()
	dataset_root = args.dataset_root.resolve() if args.dataset_root else None
	strip_prefixes = tuple(args.strip_prefix)

	for name in ("train.pickle", "val.pickle", "test.pickle"):
		split_path = input_dir / name
		if not split_path.exists():
			raise FileNotFoundError(f"Missing required split file: {split_path}")
		process_split_file(
			split_path,
			output_dir,
			strip_prefixes,
			dataset_root,
			fail_on_missing=args.fail_on_missing,
		)


if __name__ == "__main__":
	main()
