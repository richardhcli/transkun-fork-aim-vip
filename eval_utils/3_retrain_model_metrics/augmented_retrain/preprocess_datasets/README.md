# Augmented Retraining Plan

This folder is the data-preprocessing domain for augmented retraining. It prepares MAESTRO-style CSV metadata for multiple datasets, merges them into one weighted training CSV, and builds Transkun-compatible pickle files for training/validation/test.

## Goal

Create per-dataset CSV metadata with the exact columns expected by Transkun-style metadata loaders:

- split
- midi_filename
- audio_filename

Rules used here:

- MAESTRO keeps original split values from maestro-v3.0.0.csv.
- All non-MAESTRO datasets are assigned split=train.
- Paths in midi_filename and audio_filename are relative to each dataset root (same style used by MAESTRO metadata).

## Generated Scripts

- preprocess_dataset_A-MAPS.py
- preprocess_dataset_MAESTRO.py
- preprocess_dataset_SMD_v2.py
- preprocess_dataset_msmd_data.py
- preprocess_dataset_BiMMuDa.py
- preprocess_dataset_POP909.py
- preprocess_common.py
- merge_dataset_csvs.py
- prepare_transkun_pickles.py

Dataset preprocess scripts read from /scratch/gilbreth/li5042/datasets/<DATASET> by default and write CSV to:

/scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/3_retrain_model_metrics/augmented_retrain/preprocess_datasets/CSVs

## Run Metadata Generation

From this folder:

```bash
python preprocess_dataset_MAESTRO.py
python preprocess_dataset_A-MAPS.py
python preprocess_dataset_SMD_v2.py
python preprocess_dataset_msmd_data.py
python preprocess_dataset_BiMMuDa.py
python preprocess_dataset_POP909.py
```

## Merge Per-Dataset CSVs Into One Training CSV

Use merge_dataset_csvs.py to create one merged training CSV for retraining.

```bash
python merge_dataset_csvs.py
```

Output by default:

- CSVs/merged_train.csv

Optional dataset weighting (upsample/downsample by duplication/subsampling):

```bash
python merge_dataset_csvs.py \
	--weight MAESTRO=1.0 \
	--weight POP909=0.6 \
	--weight msmd_data=1.2 \
	--weight BiMMuDa=1.0 \
	--weight SMD_v2=1.5 \
	--weight A-MAPS=0.0
```

Notes:

- The merged CSV is always split=train rows.
- Path fields are prefixed with dataset folder names (for example MAESTRO/... or POP909/...), so a single dataset root can be used later.

## Pairing Behavior

For non-MAESTRO datasets, scripts pair MIDI/audio files by file stem:

- MIDI extensions: .mid, .midi
- Audio extensions: .wav, .flac, .mp3, .ogg

Each run prints summary counts:

- total MIDI files
- total audio files
- paired rows written
- missing audio stems for MIDI
- ambiguous stem matches

If a dataset is missing audio files in the current workspace snapshot (for example A-MAPS), the CSV is still written with headers and any found pairs.

## Retraining Plan With More Data

1. Generate all per-dataset CSVs with the preprocess_dataset_*.py scripts.
2. Merge them into one training CSV with merge_dataset_csvs.py (with optional weights).
3. Build train/val/test pickle files with prepare_transkun_pickles.py.
4. Re-run long training from scratch (48h target) using retrain_model.py with the new pickle paths.
5. Evaluate and compare against MAESTRO-only baseline using your existing verification/inference scripts.

## Build Transkun-Compatible Pickles

prepare_transkun_pickles.py outputs files that match expectations in transkun/train.py and your previous prereq flow:

- train.pickle
- val.pickle
- test.pickle
- transkun_base.json (unless --skip-model-conf)

Default output folder:

- /scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/3_retrain_model_metrics/output/AUGMENTED_METADATA

Run:

```bash
python prepare_transkun_pickles.py
```

For a faster multi-core run:

```bash
python prepare_transkun_pickles.py --workers 24 --rows-per-chunk 512
```

This metadata output is used by both:

- augmented retraining from scratch
- fine-tuning in eval_utils/4_fine_tuning

Wrapper script:

```bash
bash run_prepare_transkun_pickles.sh
```

This writes logs to:

- prepare_transkun_pickles.out
- prepare_transkun_pickles.err

## Notes On Transkun Compatibility

The CSV field style follows the MAESTRO convention consumed by Transkun createDatasetMaestro/Data.createDatasetMaestroCSV:

- split
- midi_filename
- audio_filename

This keeps metadata shape consistent while allowing additional datasets to be integrated into a combined retraining workflow.

For pickle metadata, each sample includes fields used by Data.DatasetMaestro and DatasetMaestroIterator:

- split
- midi_filename
- audio_filename
- duration
- notes
- fs
- nSamples
- nChannel

Important: skipped rows do not mean the entire dataset was skipped. The script keeps every row it can parse and skips rows that fail existence/format checks before Transkun metadata creation.

Current filtering behavior in this folder:

- Audio path must exist.
- Audio extension must be `.wav`.
- Audio sampling rate must be 44.1 kHz.

MIDI parsing and note extraction are delegated to `Data.createDatasetMaestroCSV`/`Data.createDatasetMaestro` in Transkun.
