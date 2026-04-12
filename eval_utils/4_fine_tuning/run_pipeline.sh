#!/bin/bash
#SBATCH --account yunglu
#SBATCH --partition=a100-80gb
#SBATCH --qos=normal
#SBATCH --ntasks=1 --cpus-per-task=32
#SBATCH --nodes=1 --gpus-per-node=1
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --job-name transkun_finetune_pipeline
#SBATCH --output=/scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/4_fine_tuning/output/run_pipeline.out
#SBATCH --error=/scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/4_fine_tuning/output/run_pipeline.err

# Full fine-tuning pipeline for augmented datasets:
# 1) generate artifacts (pickles + model params)
# 2) validate artifacts
# 3) fine-tune and optionally run one-file verification

timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

log() {
    echo "[$(timestamp)] $*"
}

set -euo pipefail

resolve_script_path() {
    local script_path="${BASH_SOURCE[0]}"

    if [[ "$script_path" == /var/spool/slurm/* ]] && [[ -n "${SLURM_JOB_ID:-}" ]]; then
        if command -v scontrol >/dev/null 2>&1; then
            local cmd_path
            cmd_path="$(scontrol show job "$SLURM_JOB_ID" | tr ' ' '\n' | awk -F= '$1=="Command"{print substr($0, index($0, "=") + 1)}')"
            if [[ -n "$cmd_path" ]]; then
                if [[ "$cmd_path" != /* ]] && [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
                    cmd_path="${SLURM_SUBMIT_DIR}/$cmd_path"
                fi
                if [[ -f "$cmd_path" ]]; then
                    script_path="$cmd_path"
                fi
            fi
        fi
    fi

    echo "$script_path"
}

SCRIPT_PATH="$(resolve_script_path)"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
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
CSV_DIR="$PROJECT_ROOT/eval_utils/3_retrain_model_metrics/augmented_retrain/preprocess_datasets/CSVs"
METADATA_DIR="$SCRIPT_DIR/model_info/training_data"
MODEL_PARAMS_DIR="$SCRIPT_DIR/model_info/model_params"
RUN_DIR="$SCRIPT_DIR/output/finetune_v2_run"

WORKERS=24
ROWS_PER_CHUNK=512
N_PROCESS=1
N_ITER=50000
BATCH_SIZE=4
DATALOADER_WORKERS=8
MAX_LR="1e-4"
WEIGHT_DECAY="1e-4"
DEVICE="auto"
ALLOW_TF32=0

TRAIN_CSV=""
MAESTRO_CSV=""
SKIP_PREPROCESS=0
SKIP_MERGE=0
NO_PEDAL_EXTENSION=0
FAIL_ON_MISSING=1

SKIP_GENERATE=0
SKIP_VALIDATE=0
SKIP_TRAIN=0
SKIP_VERIFY=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset-root) DATASET_ROOT="$2"; shift 2 ;;
        --csv-dir) CSV_DIR="$2"; shift 2 ;;
        --metadata-dir) METADATA_DIR="$2"; shift 2 ;;
        --model-params-dir) MODEL_PARAMS_DIR="$2"; shift 2 ;;
        --run-dir) RUN_DIR="$2"; shift 2 ;;
        --workers) WORKERS="$2"; shift 2 ;;
        --rows-per-chunk) ROWS_PER_CHUNK="$2"; shift 2 ;;
        --n-process) N_PROCESS="$2"; shift 2 ;;
        --n-iter) N_ITER="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --dataloader-workers) DATALOADER_WORKERS="$2"; shift 2 ;;
        --max-lr) MAX_LR="$2"; shift 2 ;;
        --weight-decay) WEIGHT_DECAY="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --allow-tf32) ALLOW_TF32=1; shift ;;
        --train-csv) TRAIN_CSV="$2"; shift 2 ;;
        --maestro-csv) MAESTRO_CSV="$2"; shift 2 ;;
        --skip-preprocess) SKIP_PREPROCESS=1; shift ;;
        --skip-merge) SKIP_MERGE=1; shift ;;
        --no-pedal-extension) NO_PEDAL_EXTENSION=1; shift ;;
        --no-fail-on-missing) FAIL_ON_MISSING=0; shift ;;
        --skip-generate) SKIP_GENERATE=1; shift ;;
        --skip-validate) SKIP_VALIDATE=1; shift ;;
        --skip-train) SKIP_TRAIN=1; shift ;;
        --skip-verify) SKIP_VERIFY=1; shift ;;
        *)
            echo "Unknown option: $1"
            exit 2
            ;;
    esac
done

log "Starting fine-tuning pipeline"
log "DATASET_ROOT=$DATASET_ROOT"
log "METADATA_DIR=$METADATA_DIR"
log "MODEL_PARAMS_DIR=$MODEL_PARAMS_DIR"
log "RUN_DIR=$RUN_DIR"

GEN_ARGS=(
    --dataset-root "$DATASET_ROOT"
    --csv-dir "$CSV_DIR"
    --metadata-dir "$METADATA_DIR"
    --model-params-dir "$MODEL_PARAMS_DIR"
    --workers "$WORKERS"
    --rows-per-chunk "$ROWS_PER_CHUNK"
)
if [[ -n "$TRAIN_CSV" ]]; then GEN_ARGS+=(--train-csv "$TRAIN_CSV"); fi
if [[ -n "$MAESTRO_CSV" ]]; then GEN_ARGS+=(--maestro-csv "$MAESTRO_CSV"); fi
if [[ "$SKIP_PREPROCESS" -eq 1 ]]; then GEN_ARGS+=(--skip-preprocess); fi
if [[ "$SKIP_MERGE" -eq 1 ]]; then GEN_ARGS+=(--skip-merge); fi
if [[ "$NO_PEDAL_EXTENSION" -eq 1 ]]; then GEN_ARGS+=(--no-pedal-extension); fi
if [[ "$FAIL_ON_MISSING" -eq 1 ]]; then GEN_ARGS+=(--fail-on-missing); fi

VAL_ARGS=(
    --dataset-root "$DATASET_ROOT"
    --metadata-dir "$METADATA_DIR"
    --model-params-dir "$MODEL_PARAMS_DIR"
)

FT_ARGS=(
    --dataset-path "$DATASET_ROOT"
    --model-info-dir "$SCRIPT_DIR/model_info"
    --run-dir "$RUN_DIR"
    --n-process "$N_PROCESS"
    --batch-size "$BATCH_SIZE"
    --data-loader-workers "$DATALOADER_WORKERS"
    --max-lr "$MAX_LR"
    --weight-decay "$WEIGHT_DECAY"
    --n-iter "$N_ITER"
    --device "$DEVICE"
)
if [[ "$ALLOW_TF32" -eq 1 ]]; then FT_ARGS+=(--allow-tf32); fi

if [[ "$SKIP_GENERATE" -eq 0 ]]; then
    log "[Step 1/3] Generating fine-tuning artifacts..."
    python "$SCRIPT_DIR/generate_artifacts.py" "${GEN_ARGS[@]}" \
        > >(tee "$LOG_DIR/step1_generate.out") \
        2> >(tee "$LOG_DIR/step1_generate.err" >&2)
    log "[Step 1/3] Completed."
else
    log "[Step 1/3] Skipped (--skip-generate)."
fi

if [[ "$SKIP_VALIDATE" -eq 0 ]]; then
    log "[Step 2/3] Validating fine-tuning artifacts..."
    python "$SCRIPT_DIR/validate_artifacts.py" "${VAL_ARGS[@]}" \
        > >(tee "$LOG_DIR/step2_validate.out") \
        2> >(tee "$LOG_DIR/step2_validate.err" >&2)
    log "[Step 2/3] Completed."
else
    log "[Step 2/3] Skipped (--skip-validate)."
fi

if [[ "$SKIP_TRAIN" -eq 1 && "$SKIP_VERIFY" -eq 1 ]]; then
    log "[Step 3/3] Skipped (both --skip-train and --skip-verify requested)."
else
    if [[ "$SKIP_TRAIN" -eq 1 ]]; then
        FT_ARGS+=(--skip-seed --skip-train)
    fi
    if [[ "$SKIP_VERIFY" -eq 1 ]]; then
        FT_ARGS+=(--skip-verify)
    fi

    log "[Step 3/3] Running fine-tune stage..."
    python "$SCRIPT_DIR/fine_tune.py" "${FT_ARGS[@]}" \
        > >(tee "$LOG_DIR/step3_fine_tune.out") \
        2> >(tee "$LOG_DIR/step3_fine_tune.err" >&2)
    log "[Step 3/3] Completed."
fi

log "Fine-tuning pipeline finished successfully. Logs: $LOG_DIR"
