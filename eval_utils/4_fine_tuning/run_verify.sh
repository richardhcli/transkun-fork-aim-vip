#!/bin/bash
#SBATCH --account yunglu
#SBATCH --partition=a100-80gb
#SBATCH --qos=normal
#SBATCH --ntasks=1 --cpus-per-task=8
#SBATCH --nodes=1 --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --job-name transkun_finetune_verify
#SBATCH --output=/scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/4_fine_tuning/output/run_verify.out
#SBATCH --error=/scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/4_fine_tuning/output/run_verify.err
#SBATCH --chdir=/scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/4_fine_tuning

# Verification job:
# 1) validate metadata/model artifacts
# 2) run one-file test transcription with existing fine-tuned checkpoint

timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

log() {
    echo "[$(timestamp)] $*"
}

set -euo pipefail

SCRIPT_DIR="/scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/4_fine_tuning"

PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="$SCRIPT_DIR/output/logs"
mkdir -p "$LOG_DIR" "$SCRIPT_DIR/output"

if [[ -f "$PROJECT_ROOT/eval_utils/0_environment_setup/setup_environment.sh" ]]; then
    # shellcheck source=/dev/null
    source "$PROJECT_ROOT/eval_utils/0_environment_setup/setup_environment.sh"
    log "Loaded Transkun environment setup."
else
    log "WARN: setup_environment.sh not found. Using current shell environment."
fi

DATASET_ROOT="/scratch/gilbreth/li5042/datasets"
MODEL_INFO_DIR="$SCRIPT_DIR/model_info"
METADATA_DIR="$MODEL_INFO_DIR/training_data"
MODEL_PARAMS_DIR="$MODEL_INFO_DIR/model_params"
RUN_DIR="$SCRIPT_DIR/output/finetune_v2_run"
DEVICE="auto"
SAMPLE_CHECK_COUNT=10

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset-root) DATASET_ROOT="$2"; shift 2 ;;
        --model-info-dir) MODEL_INFO_DIR="$2"; shift 2 ;;
        --metadata-dir) METADATA_DIR="$2"; shift 2 ;;
        --model-params-dir) MODEL_PARAMS_DIR="$2"; shift 2 ;;
        --run-dir) RUN_DIR="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --sample-check-count) SAMPLE_CHECK_COUNT="$2"; shift 2 ;;
        *)
            echo "Unknown option: $1"
            exit 2
            ;;
    esac
done

log "Starting fine-tuning verification"
log "DATASET_ROOT=$DATASET_ROOT"
log "METADATA_DIR=$METADATA_DIR"
log "MODEL_PARAMS_DIR=$MODEL_PARAMS_DIR"
log "RUN_DIR=$RUN_DIR"

python "$SCRIPT_DIR/validate_artifacts.py" \
    --dataset-root "$DATASET_ROOT" \
    --metadata-dir "$METADATA_DIR" \
    --model-params-dir "$MODEL_PARAMS_DIR" \
    --sample-check-count "$SAMPLE_CHECK_COUNT" \
    > >(tee "$LOG_DIR/verify_step1_validate.out") \
    2> >(tee "$LOG_DIR/verify_step1_validate.err" >&2)

python "$SCRIPT_DIR/fine_tune.py" \
    --dataset-path "$DATASET_ROOT" \
    --model-info-dir "$MODEL_INFO_DIR" \
    --run-dir "$RUN_DIR" \
    --device "$DEVICE" \
    --skip-seed \
    --skip-train \
    > >(tee "$LOG_DIR/verify_step2_transcribe.out") \
    2> >(tee "$LOG_DIR/verify_step2_transcribe.err" >&2)

log "Verification completed successfully."
