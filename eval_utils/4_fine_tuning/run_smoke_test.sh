#!/bin/bash
# Minimal smoke test for the 4_fine_tuning pipeline.
# Uses one hard-coded MAESTRO train pair and isolated smoke output paths.

set -euo pipefail

timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

log() {
    echo "[$(timestamp)] $*"
}

run_step() {
    local step_name="$1"
    shift

    log "Running ${step_name}..."
    "$@" \
        > >(tee "$LOG_DIR/${step_name}.out") \
        2> >(tee "$LOG_DIR/${step_name}.err" >&2)
    log "Completed ${step_name}."
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

DATASET_ROOT="${DATASET_ROOT:-/scratch/gilbreth/li5042/datasets}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SMOKE_TIMEOUT_SECONDS="${SMOKE_TIMEOUT_SECONDS:-600}"

SMOKE_BASE_DIR="$SCRIPT_DIR/output/smoke"
SMOKE_RUN_ID="smoke_$(date '+%Y%m%d_%H%M%S')"
SMOKE_ROOT="$SMOKE_BASE_DIR/$SMOKE_RUN_ID"

CSV_DIR="$SMOKE_ROOT/csv"
MODEL_INFO_DIR="$SMOKE_ROOT/model_info"
METADATA_DIR="$MODEL_INFO_DIR/training_data"
MODEL_PARAMS_DIR="$MODEL_INFO_DIR/model_params"
RUN_DIR="$SMOKE_ROOT/run"
LOG_DIR="$SMOKE_ROOT/logs"

mkdir -p "$CSV_DIR" "$METADATA_DIR" "$MODEL_PARAMS_DIR" "$RUN_DIR" "$LOG_DIR"

# Hard-coded row from MAESTRO metadata:
# Alban Berg, Sonata Op. 1, train, 2018,
# 2018/MIDI-Unprocessed_Chamber3_MID--AUDIO_10_R3_2018_wav--1.midi,
# 2018/MIDI-Unprocessed_Chamber3_MID--AUDIO_10_R3_2018_wav--1.wav, 698.661160312
SMOKE_MIDI_REL="2018/MIDI-Unprocessed_Chamber3_MID--AUDIO_10_R3_2018_wav--1.midi"
SMOKE_AUDIO_REL="2018/MIDI-Unprocessed_Chamber3_MID--AUDIO_10_R3_2018_wav--1.wav"

SMOKE_MIDI_ABS="$DATASET_ROOT/MAESTRO/$SMOKE_MIDI_REL"
SMOKE_AUDIO_ABS="$DATASET_ROOT/MAESTRO/$SMOKE_AUDIO_REL"

if [[ ! -f "$SMOKE_MIDI_ABS" ]]; then
    log "ERROR: Missing smoke MIDI file: $SMOKE_MIDI_ABS"
    exit 1
fi
if [[ ! -f "$SMOKE_AUDIO_ABS" ]]; then
    log "ERROR: Missing smoke audio file: $SMOKE_AUDIO_ABS"
    exit 1
fi

SMOKE_TRAIN_CSV="$CSV_DIR/smoke_train.csv"
SMOKE_MAESTRO_CSV="$CSV_DIR/smoke_maestro.csv"

cat > "$SMOKE_TRAIN_CSV" <<EOF
split,midi_filename,audio_filename
train,MAESTRO/$SMOKE_MIDI_REL,MAESTRO/$SMOKE_AUDIO_REL
EOF

# Intentionally empty MAESTRO split CSV (header only) to test empty val/test handling.
cat > "$SMOKE_MAESTRO_CSV" <<EOF
split,midi_filename,audio_filename
EOF

# Duplicate existing .pt/.conf into isolated smoke model params directory
# so smoke execution never affects shared artifacts.
MAIN_MODEL_PARAMS_DIR="$SCRIPT_DIR/model_info/model_params"
cp -f "$MAIN_MODEL_PARAMS_DIR/checkpoint.pt" "$MODEL_PARAMS_DIR/checkpoint.pt"
cp -f "$MAIN_MODEL_PARAMS_DIR/model.conf" "$MODEL_PARAMS_DIR/model.conf"

log "Smoke run root: $SMOKE_ROOT"
log "Smoke train CSV: $SMOKE_TRAIN_CSV"
log "Smoke MAESTRO CSV: $SMOKE_MAESTRO_CSV"
log "Dataset root: $DATASET_ROOT"

run_step "step1_generate" \
    timeout "$SMOKE_TIMEOUT_SECONDS" \
    "$PYTHON_BIN" "$SCRIPT_DIR/generate_artifacts.py" \
    --dataset-root "$DATASET_ROOT" \
    --metadata-dir "$METADATA_DIR" \
    --model-params-dir "$MODEL_PARAMS_DIR" \
    --smoke-midi-rel "MAESTRO/$SMOKE_MIDI_REL" \
    --smoke-audio-rel "MAESTRO/$SMOKE_AUDIO_REL" \
    --smoke-split train \
    --skip-model-params-copy

run_step "step2_validate" \
    timeout "$SMOKE_TIMEOUT_SECONDS" \
    "$PYTHON_BIN" "$SCRIPT_DIR/validate_artifacts.py" \
    --dataset-root "$DATASET_ROOT" \
    --metadata-dir "$METADATA_DIR" \
    --model-params-dir "$MODEL_PARAMS_DIR" \
    --sample-check-count 1 \
    --min-train-samples 1 \
    --min-val-samples 0 \
    --min-test-samples 0

run_step "step3_fine_tune" \
    timeout "$SMOKE_TIMEOUT_SECONDS" \
    "$PYTHON_BIN" "$SCRIPT_DIR/fine_tune.py" \
    --dataset-path "$DATASET_ROOT" \
    --model-info-dir "$MODEL_INFO_DIR" \
    --run-dir "$RUN_DIR" \
    --n-process 1 \
    --batch-size 1 \
    --data-loader-workers 0 \
    --max-lr 1e-5 \
    --weight-decay 1e-4 \
    --n-iter 1 \
    --device auto

if [[ ! -f "$RUN_DIR/checkpoint_finetuned.pt" ]]; then
    log "ERROR: Expected smoke checkpoint not found: $RUN_DIR/checkpoint_finetuned.pt"
    exit 1
fi
if [[ ! -f "$RUN_DIR/model_finetune.conf" ]]; then
    log "ERROR: Expected smoke model conf not found: $RUN_DIR/model_finetune.conf"
    exit 1
fi

echo "$SMOKE_ROOT" > "$SMOKE_BASE_DIR/latest_run_path.txt"
log "Smoke test completed successfully."
log "Artifacts: $SMOKE_ROOT"
