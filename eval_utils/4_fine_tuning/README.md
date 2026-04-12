# Fine-Tuning Pipeline (Augmented Metadata)

This folder now mirrors the `augmented_retrain` staged workflow, but the training step is **fine-tuning from pretrained Transkun weights** instead of training from scratch.

## Pipeline Steps

1. `generate_artifacts.py`
- Reuses `eval_utils/3_retrain_model_metrics/augmented_retrain/preprocess_datasets/prepare_transkun_pickles.py`.
- Builds augmented `train.pickle`, `val.pickle`, `test.pickle` from the combined datasets.
- Normalizes and validates metadata paths with `preprocessing.py`.
- Refreshes `model_info/model_params/checkpoint.pt` and `model_info/model_params/model.conf` from pretrained Transkun V2 artifacts.

2. `validate_artifacts.py`
- Checks that metadata files exist and have expected Transkun fields.
- Verifies sampled `audio_filename` and `midi_filename` paths resolve under dataset root.
- Verifies pretrained model artifacts are present.

3. `fine_tune.py`
- Seeds a new fine-tuning checkpoint from pretrained weights.
- Launches `transkun.train` using generated metadata.
- Optionally runs one-file test transcription on the test split.

## SLURM Scripts

- `run_pipeline.sh`:
  Base execution script for Step 1 -> Step 2 -> Step 3.

- `run_verify.sh`:
  Single verification script for metadata/model validation plus one-file transcription.

- `generate_prereq_files.sh`:
  Compatibility wrapper to generate fine-tuning prerequisites only.

## Default Paths

- Dataset root: `/scratch/gilbreth/li5042/datasets`
- Metadata output: `model_info/training_data`
- Model params: `model_info/model_params`
- Fine-tuning run dir: `output/finetune_v2_run`

## Common Commands

Run artifact generation only:

```bash
python generate_artifacts.py \
  --dataset-root /scratch/gilbreth/li5042/datasets \
  --workers 24 \
  --rows-per-chunk 512 \
  --fail-on-missing
```

Validate artifacts:

```bash
python validate_artifacts.py --dataset-root /scratch/gilbreth/li5042/datasets
```

Run full pipeline locally:

```bash
bash run_pipeline.sh
```

Submit full pipeline with SLURM:

```bash
sbatch run_pipeline.sh
```

Run verification only:

```bash
bash run_verify.sh
```

Run isolated smoke test (single hard-coded MAESTRO train pair):

```bash
bash run_smoke_test.sh
```

The smoke test writes only to `output/smoke/*`, duplicates model params into that
isolated folder first, and keeps the shared `model_info/model_params` and
`model_info/training_data` untouched.

## Notes

- `fine_tune.py` now defaults to dataset root `/scratch/gilbreth/li5042/datasets` so augmented path prefixes such as `MAESTRO/...`, `POP909/...`, and others resolve correctly.
- For repeated runs, you can skip generation and/or validation from `run_pipeline.sh` using `--skip-generate` and `--skip-validate`.
