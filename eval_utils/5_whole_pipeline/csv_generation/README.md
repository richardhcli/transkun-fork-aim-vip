# Pickle Generation Pipeline

This folder builds Transkun-compatible CSV metadata for all target piano datasets.

## Goal

Produce:
- Per-dataset CSVs in `CSVs/*.csv`
- `CSVs/entire.csv` (accumulates all rows from all dataset CSVs)
- `CSVs/user_testing.csv` (one sample, e.g. first row, from each dataset CSV)
- `CSVs/smoketest.csv` (one sample, e.g. first row, from MAESTRO CSV only)
- `CSVs/dataset_root_map.json` (dataset-name -> dataset-root mapping used as path prefix metadata)

These CSVs are then consumed by pickle generation/training stages.

## Entry Point

Run the orchestrator:

```bash
export MAIN_SCRIPT_DIR=/scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/5_whole_pipeline
python /scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/5_whole_pipeline/pickle_generation/generate_all_csv.py
```

The script expects `MAIN_SCRIPT_DIR` to be exported and resolves its script location as:
- `MAIN_SCRIPT_DIR/pickle_generation`

## Per-Dataset Generators

The orchestrator calls:
- `generate_a_maps_csv.py`
- `generate_bimmuda_csv.py`
- `generate_maestro_csv.py`
- `generate_pop909_csv.py`
- `generate_smd_v2_csv.py`
- `generate_msmd_data_csv.py`

## Accumulation Builder

Accumulated CSV generation is intentionally moved to a dedicated file:
- `build_accumulated_csvs.py`

This module reads `dataset_root_map.json` (the `output_json` map from the orchestrator)
and builds new CSVs from already-generated raw per-dataset CSVs.

Behavior:
- `entire.csv`: contains every row from all dataset CSVs.
- `user_testing.csv`: contains one deterministic representative row per dataset.
- `smoketest.csv`: contains one deterministic representative row from MAESTRO.

All accumulated CSV rows use absolute paths for `midi_filename` and `audio_filename`.
This mimics `Data.createDatasetMaestroCSV` path joining behavior by applying
`dataset_root + relative_csv_path` for each dataset row.

## Data Rules Enforced

### 1) No dataset mutation

CSV generation does not create or rewrite MIDI/audio files.
No flattened MIDI artifacts are created.

Paths in CSV rows always point to original dataset files, relative to that dataset root.

### 2) Audio sample-rate validation

Only `.wav` files at 44100 Hz are accepted.

Rows are skipped for:
- missing audio
- non-wav audio
- invalid/corrupt wav
- wav at non-44100 rate

### 3) Relative path format

CSV `midi_filename` and `audio_filename` are dataset-root-relative paths.

Example (correct):
- `2018/file.mid`

Example (not used in per-dataset CSV):
- `MAESTRO/2018/file.mid`

Note: accumulated CSVs (`entire.csv`, `user_testing.csv`, `smoketest.csv`) are
written with absolute paths, while per-dataset CSVs remain dataset-root-relative.

### 4) MAESTRO handling

`generate_maestro_csv.py` copies MAESTRO's metadata CSV directly to output (no regeneration/flattening in this step).

### 5) Dataset root mapping JSON

`dataset_root_map.json` is exported to describe how to prefix CSV relative paths back to absolute paths.
It includes:
- `datasets_root`
- `dataset_roots` (dataset-name -> absolute dataset root)
- `dataset_order` (dataset order used when aggregating CSVs)
- `dataset_csvs` (dataset-name -> generated CSV path)
- `combined_csvs` (`entire`, `user_testing`, `smoketest`)

## Failsafes

- `generate_all_csv.py` fails early if `MAIN_SCRIPT_DIR` is missing.
- It fails if a required generator script is missing.
- It fails if a dataset used for `user_testing.csv` has zero rows.
- It fails if MAESTRO does not contain at least one row in each split needed by `smoketest.csv`.

## Logs

Logs are written to `logs/`:
- `<dataset>_skipped_rows.csv`
  - reason, midi path, audio path, and details for skipped rows
- `<dataset>_wav_rate_issues.csv`
  - all non-44100 wav files and detected frequencies

No logs are produced for flattening, because flattening is disabled.

## What Usually Takes the Most Time

The dominant runtime is usually dataset traversal plus WAV sample-rate checks on large datasets.

Most expensive parts:
- traversing deep dataset directory trees
- validating WAV headers/sample rates at scale

If runtime is high, typical mitigations are:
- keep datasets on fast storage
- run with sufficient CPU resources during metadata build

## Known Empty-CSV Causes

Common reasons a dataset CSV may end up empty:
- dataset has MIDI but no WAV files
- no stem matches between MIDI and audio
- all audio rows filtered out by sample-rate rule
- dataset was interrupted before CSV writing completed

Legacy generated `.transkun_flattened_midis` folders are ignored during pairing.
