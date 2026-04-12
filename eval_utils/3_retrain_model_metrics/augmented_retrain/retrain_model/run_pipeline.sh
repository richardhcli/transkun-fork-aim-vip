#!/bin/bash
#SBATCH --account yunglu
#SBATCH --partition=a100-80gb
#SBATCH --qos=normal
#SBATCH --ntasks=1 --cpus-per-task=32
#SBATCH --nodes=1 --gpus-per-node=1
#SBATCH --mem=128G
#SBATCH --time=6:00:00
#SBATCH --job-name transkun_aug_pipeline
#SBATCH --output=/scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/3_retrain_model_metrics/augmented_retrain/retrain_model/output/run_pipeline.out
#SBATCH --error=/scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/3_retrain_model_metrics/augmented_retrain/retrain_model/output/run_pipeline.err

# ============================== SLURM DIRECTIVES ===============================
# Submit this script with:
#   sbatch run_pipeline.sh [optional flags passed directly if your cluster allows]
#
# For direct shell usage (non-SLURM), this also works:
#   bash run_pipeline.sh --skip-generate --skip-validate
# ==============================================================================
# Timestamp helper for readable, chronological feedback.
timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

log() {
    echo "[$(timestamp)] $*"
}

# Use absolute script directory so SLURM spool execution still finds step scripts.
SCRIPT_DIR="/scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/3_retrain_model_metrics/augmented_retrain/retrain_model"

# Load Transkun environment just like other project SLURM scripts.
if [[ -f "/scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/0_environment_setup/setup_environment.sh" ]]; then
    # shellcheck source=/dev/null
    source /scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/0_environment_setup/setup_environment.sh
    log "Loaded Transkun environment setup."
else
    log "WARN: setup_environment.sh not found. Using current shell environment."
fi

set -euo pipefail

# =============================== DEFAULT CONFIG =================================
# Centralized output root: metadata, checkpoints, and logs all live underneath.
OUTPUT_ROOT="/scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/3_retrain_model_metrics/output/AUGMENTED_METADATA"
DATASET_ROOT="/scratch/gilbreth/li5042/datasets"
TRAIN_CSV=""
MAESTRO_CSV=""
WORKERS=16
CHUNKSIZE=32
NO_PEDAL_EXTENSION=0

# Step controls (requested):
# - SKIP_GENERATE=1 skips Step 1
# - SKIP_VALIDATE=1 skips Step 2
# - SKIP_TRAINING=1 skips Step 3
SKIP_GENERATE=1
SKIP_VALIDATE=0
SKIP_TRAINING=0

SAMPLE_CHECK_COUNT=10
MIN_TRAIN_SAMPLES=100
MIN_VAL_SAMPLES=10
MIN_TEST_SAMPLES=10

CHECKPOINT_NAME="checkpoint_augmented.pt"
N_PROCESS=0
N_ITER=180000
BATCH_SIZE=4
DATALOADER_WORKERS=8
MAX_LR="2e-4"
WEIGHT_DECAY="1e-4"
ALLOW_TF32=0
HOP_SIZE=""
CHUNK_SIZE=""
AUGMENT=0
NOISE_FOLDER=""
IR_FOLDER=""
FORCE_FRESH=0

detect_n_process() {
    if [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
        echo "$SLURM_GPUS_ON_NODE"
        return
    fi

    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        local device_count=0
        IFS=',' read -r -a _visible_devices <<< "$CUDA_VISIBLE_DEVICES"
        device_count=${#_visible_devices[@]}
        if [[ "$device_count" -gt 0 ]]; then
            echo "$device_count"
            return
        fi
    fi

    echo 1
}

# =============================== ARGUMENT PARSING ===============================
while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
        --dataset-root) DATASET_ROOT="$2"; shift 2 ;;
        --train-csv) TRAIN_CSV="$2"; shift 2 ;;
        --maestro-csv) MAESTRO_CSV="$2"; shift 2 ;;
        --workers) WORKERS="$2"; shift 2 ;;
        --chunksize) CHUNKSIZE="$2"; shift 2 ;;
        --no-pedal-extension) NO_PEDAL_EXTENSION=1; shift ;;
        --skip-generate) SKIP_GENERATE=1; shift ;;
        --skip-validate) SKIP_VALIDATE=1; shift ;;
        --skip-training) SKIP_TRAINING=1; shift ;;
        --sample-check-count) SAMPLE_CHECK_COUNT="$2"; shift 2 ;;
        --min-train-samples) MIN_TRAIN_SAMPLES="$2"; shift 2 ;;
        --min-val-samples) MIN_VAL_SAMPLES="$2"; shift 2 ;;
        --min-test-samples) MIN_TEST_SAMPLES="$2"; shift 2 ;;
        --checkpoint-name) CHECKPOINT_NAME="$2"; shift 2 ;;
        --n-process) N_PROCESS="$2"; shift 2 ;;
        --n-iter) N_ITER="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --dataloader-workers) DATALOADER_WORKERS="$2"; shift 2 ;;
        --max-lr) MAX_LR="$2"; shift 2 ;;
        --weight-decay) WEIGHT_DECAY="$2"; shift 2 ;;
        --allow-tf32) ALLOW_TF32=1; shift ;;
        --hop-size) HOP_SIZE="$2"; shift 2 ;;
        --chunk-size) CHUNK_SIZE="$2"; shift 2 ;;
        --augment) AUGMENT=1; shift ;;
        --noise-folder) NOISE_FOLDER="$2"; shift 2 ;;
        --ir-folder) IR_FOLDER="$2"; shift 2 ;;
        --force-fresh) FORCE_FRESH=1; shift ;;
        *) echo "Unknown option: $1"; exit 2 ;;
    esac
done

# Prepare centralized directories.
METADATA_DIR="$OUTPUT_ROOT/metadata"
LOG_DIR="$OUTPUT_ROOT/logs"
mkdir -p "$METADATA_DIR" "$LOG_DIR"

log "Starting augmented retraining pipeline"
log "OUTPUT_ROOT=$OUTPUT_ROOT"
log "DATASET_ROOT=$DATASET_ROOT"
log "SKIP_GENERATE=$SKIP_GENERATE SKIP_VALIDATE=$SKIP_VALIDATE SKIP_TRAINING=$SKIP_TRAINING"
log "WORKERS=$WORKERS CHUNKSIZE=$CHUNKSIZE"

if [[ "$N_PROCESS" -le 0 ]]; then
    N_PROCESS="$(detect_n_process)"
fi
log "N_PROCESS=$N_PROCESS"

GEN_ARGS=(
    --dataset-root "$DATASET_ROOT"
    --output-root "$OUTPUT_ROOT"
    --workers "$WORKERS"
    --chunksize "$CHUNKSIZE"
)
if [[ -n "$TRAIN_CSV" ]]; then GEN_ARGS+=(--train-csv "$TRAIN_CSV"); fi
if [[ -n "$MAESTRO_CSV" ]]; then GEN_ARGS+=(--maestro-csv "$MAESTRO_CSV"); fi
if [[ "$NO_PEDAL_EXTENSION" -eq 1 ]]; then GEN_ARGS+=(--no-pedal-extension); fi

VAL_ARGS=(
    --dataset-root "$DATASET_ROOT"
    --output-root "$OUTPUT_ROOT"
    --sample-check-count "$SAMPLE_CHECK_COUNT"
    --min-train-samples "$MIN_TRAIN_SAMPLES"
    --min-val-samples "$MIN_VAL_SAMPLES"
    --min-test-samples "$MIN_TEST_SAMPLES"
)

TRAIN_ARGS=(
    --dataset-root "$DATASET_ROOT"
    --output-root "$OUTPUT_ROOT"
    --checkpoint-name "$CHECKPOINT_NAME"
    --n-process "$N_PROCESS"
    --n-iter "$N_ITER"
    --batch-size "$BATCH_SIZE"
    --dataloader-workers "$DATALOADER_WORKERS"
    --max-lr "$MAX_LR"
    --weight-decay "$WEIGHT_DECAY"
)
if [[ "$ALLOW_TF32" -eq 1 ]]; then TRAIN_ARGS+=(--allow-tf32); fi
if [[ -n "$HOP_SIZE" ]]; then TRAIN_ARGS+=(--hop-size "$HOP_SIZE"); fi
if [[ -n "$CHUNK_SIZE" ]]; then TRAIN_ARGS+=(--chunk-size "$CHUNK_SIZE"); fi
if [[ "$AUGMENT" -eq 1 ]]; then TRAIN_ARGS+=(--augment); fi
if [[ -n "$NOISE_FOLDER" ]]; then TRAIN_ARGS+=(--noise-folder "$NOISE_FOLDER"); fi
if [[ -n "$IR_FOLDER" ]]; then TRAIN_ARGS+=(--ir-folder "$IR_FOLDER"); fi
if [[ "$FORCE_FRESH" -eq 1 ]]; then TRAIN_ARGS+=(--force-fresh); fi

# ================================ STEP 1: GENERATE ===============================
if [[ "$SKIP_GENERATE" -eq 0 ]]; then
    log "[Step 1/3] Generating artifacts..."
    python "$SCRIPT_DIR/generate_artifacts.py" "${GEN_ARGS[@]}" \
        > >(tee "$LOG_DIR/step1_generate.out") \
        2> >(tee "$LOG_DIR/step1_generate.err" >&2)
    log "[Step 1/3] Completed. See $LOG_DIR/step1_generate.out"
else
    log "[Step 1/3] Skipped (--skip-generate)."
fi

# ================================ STEP 2: VALIDATE ===============================
if [[ "$SKIP_VALIDATE" -eq 0 ]]; then
    log "[Step 2/3] Validating artifacts..."
    python "$SCRIPT_DIR/validate_artifacts.py" "${VAL_ARGS[@]}" \
        > >(tee "$LOG_DIR/step2_validate.out") \
        2> >(tee "$LOG_DIR/step2_validate.err" >&2)
    log "[Step 2/3] Completed. See $LOG_DIR/step2_validate.out"
else
    log "[Step 2/3] Skipped (--skip-validate)."
fi

# ================================ STEP 3: TRAIN ==================================
if [[ "$SKIP_TRAINING" -eq 0 ]]; then
    log "[Step 3/3] Launching training..."
    python "$SCRIPT_DIR/train_from_artifacts.py" "${TRAIN_ARGS[@]}" \
        > >(tee "$LOG_DIR/step3_train.out") \
        2> >(tee "$LOG_DIR/step3_train.err" >&2)
    log "[Step 3/3] Completed. See $LOG_DIR/step3_train.out"
else
    log "[Step 3/3] Skipped (--skip-training)."
fi

log "Pipeline finished successfully. Logs in: $LOG_DIR"
