#!/usr/bin/env python3
"""One-time script to convert multi-instrument MIDI files in place into single-instrument files.

This script scans a dataset directory for mid/midi files, identifies those with >1 instrument
track, flattens them into a single instrument track, saves a backup of the original,
verifies the written file has only 1 track, and logs all actions and errors to a JSON log.

Features:
- Modifies files in place but saves the original to `<name>_original.mid.backup`.
- Validation check ensures the newly written file parses back with `len(pm.instruments) <= 1`.
- Skips already single-track files.
- Fails securely: if rewriting fails or verification fails, it restores the backup to protect the original dataset.
"""

import argparse
import json
import logging
import os
import pretty_midi
import shutil
import sys
import traceback
from pathlib import Path
from typing import Dict, List


def setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("midi_flatten")
    logger.setLevel(logging.INFO)
    
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("[%(levelname)s] %(asctime)s - %(message)s")

    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


def flatten_midi(midi_path: Path, pm: pretty_midi.PrettyMIDI) -> pretty_midi.PrettyMIDI:
    """Takes a parsed PrettyMIDI object and flattens its instruments into the first one."""
    main_inst = pm.instruments[0]
    
    for inst in pm.instruments[1:]:
        main_inst.notes.extend(inst.notes)
        main_inst.control_changes.extend(inst.control_changes)
        main_inst.pitch_bends.extend(inst.pitch_bends)
        
    main_inst.notes.sort(key=lambda n: n.start)
    main_inst.control_changes.sort(key=lambda c: c.time)
    main_inst.pitch_bends.sort(key=lambda b: b.time)
    
    pm.instruments = [main_inst]
    return pm


def verify_flattened(midi_path: Path) -> bool:
    """Parse the file again and check if it has only one instrument."""
    try:
        pm = pretty_midi.PrettyMIDI(str(midi_path))
        return len(pm.instruments) <= 1
    except Exception:
        return False


def process_file(midi_path: Path, logger: logging.Logger) -> Dict[str, object]:
    """Process a single MIDI file. Modifies in place and returns status dict."""
    status = {
        "path": str(midi_path.resolve()),
        "status": "skipped",
        "original_instruments": 0,
        "final_instruments": 0,
        "error": None,
    }

    try:
        try:
            pm = pretty_midi.PrettyMIDI(str(midi_path))
        except Exception as parse_err:
            status["status"] = "error"
            status["error"] = f"Parse error: {parse_err}"
            logger.debug(f"Skipping {midi_path.name} (parse error).")
            return status

        num_inst = len(pm.instruments)
        status["original_instruments"] = num_inst

        if num_inst <= 1:
            status["final_instruments"] = num_inst
            return status

        logger.info(f"Flattening {midi_path.name} ({num_inst} instruments)...")
        status["status"] = "modified"

        backup_path = midi_path.with_suffix(midi_path.suffix + ".backup")
        if backup_path.exists():
            logger.warning(f"Backup already exists for {midi_path.name}, skipping backup creation.")
        else:
            shutil.copy2(midi_path, backup_path)

        try:
            flattened_pm = flatten_midi(midi_path, pm)
            
            # Make the file writable before mutating
            midi_path.chmod(midi_path.stat().st_mode | 0o200)
            
            flattened_pm.write(str(midi_path))

            if not verify_flattened(midi_path):
                raise RuntimeError(f"Verification failed: new file {midi_path.name} still has >1 track or is malformed.")

            status["final_instruments"] = 1
            logger.info(f"Successfully flattened {midi_path.name}.")

        except Exception as proc_err:
            logger.error(f"Failed during processing {midi_path.name}: {proc_err}")
            logger.info(f"Restoring backup {backup_path.name} -> {midi_path.name}")
            shutil.copy2(backup_path, midi_path)
            status["status"] = "error_restored"
            status["error"] = str(proc_err)

    except Exception as e:
        status["status"] = "fatal_error"
        status["error"] = str(e)
        logger.error(f"Fatal error on {midi_path.name}:\n{traceback.format_exc()}")

    return status


def main() -> int:
    parser = argparse.ArgumentParser(description="Flatten multi-track MIDI files in a dataset in place.")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Directory to scan for .mid/.midi files")
    parser.add_argument("--output-json", type=Path, default=Path("flatten_log.json"), help="Output JSON status file")
    parser.add_argument("--output-log", type=Path, default=Path("flatten_script.log"), help="Output text log file")
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    if not dataset_dir.is_dir():
        print(f"Error: dataset directory {dataset_dir} does not exist.")
        return 1

    logger = setup_logger(args.output_log)
    logger.info(f"Scanning {dataset_dir} for MIDI files...")

    midi_files: List[Path] = []
    midi_files.extend(dataset_dir.rglob("*.mid"))
    midi_files.extend(dataset_dir.rglob("*.midi"))

    logger.info(f"Found {len(midi_files)} files. Starting processing...")

    results = []
    modified_count = 0
    skipped_count = 0
    error_count = 0

    for idx, path in enumerate(midi_files, 1):
        if idx % 100 == 0:
            logger.info(f"Processed {idx}/{len(midi_files)} files...")
        
        # Don't process our own backups
        if path.suffix == ".backup":
            continue

        res = process_file(path, logger)
        results.append(res)
        
        if res["status"] == "modified":
            modified_count += 1
        elif res["status"] == "skipped":
            skipped_count += 1
        elif "error" in res["status"]:
            error_count += 1

    logger.info("Writing JSON report...")
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info("=" * 40)
    logger.info("FLATTEN MIDI REPORT")
    logger.info("=" * 40)
    logger.info(f"Total files checked : {len(midi_files)}")
    logger.info(f"Already single track: {skipped_count}")
    logger.info(f"Successfully flat   : {modified_count}")
    logger.info(f"Errors (restored)   : {error_count}")
    logger.info(f"Logs written to     : {args.output_json} and {args.output_log}")

    return 0 if error_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
