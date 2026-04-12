#!/bin/bash
#SBATCH --account yunglu
#SBATCH --partition=a100-80gb
#SBATCH --qos=standby
#SBATCH --ntasks=1 --cpus-per-task=8
#SBATCH --nodes=1 --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --job-name transkun_finetune_prereq
#SBATCH --output=/scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/4_fine_tuning/output/generate_prereq_files.out
#SBATCH --error=/scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/4_fine_tuning/output/generate_prereq_files.err

#====================================================================================================================================
SCRIPT_DIR=/scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/4_fine_tuning


#====================================================================================================================================
cd "$SCRIPT_DIR"

set -euo pipefail

timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

log() {
    echo "[$(timestamp)] $*"
}

PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

OUT_DIR="$SCRIPT_DIR/output"
OUT_FILE="$OUT_DIR/generate_prereq_files.out"
ERR_FILE="$OUT_DIR/generate_prereq_files.err"

MODEL_INFO_DIR="$SCRIPT_DIR/model_info"
MODEL_PARAMS_DIR="$MODEL_INFO_DIR/model_params"
TRAINING_DATA_DIR="$MODEL_INFO_DIR/training_data"

SCRATCH_ROOT="$SCRATCH"
DATASET_ROOT="$SCRATCH_ROOT/datasets"
WORKERS="${WORKERS:-24}"
ROWS_PER_CHUNK="${ROWS_PER_CHUNK:-512}"

mkdir -p "$OUT_DIR" "$MODEL_PARAMS_DIR" "$TRAINING_DATA_DIR"

source "$PROJECT_ROOT/eval_utils/0_environment_setup/setup_environment.sh"
cd "$PROJECT_ROOT"

log "Generating augmented fine-tuning prerequisites via generate_artifacts.py" | tee -a "$OUT_FILE"
log "DATASET_ROOT=$DATASET_ROOT" | tee -a "$OUT_FILE"
log "WORKERS=$WORKERS ROWS_PER_CHUNK=$ROWS_PER_CHUNK" | tee -a "$OUT_FILE"

python "$SCRIPT_DIR/generate_artifacts.py" \
    --dataset-root "$DATASET_ROOT" \
    --workers "$WORKERS" \
    --rows-per-chunk "$ROWS_PER_CHUNK" \
    --metadata-dir "$TRAINING_DATA_DIR" \
    --fail-on-missing \
    1>>"$OUT_FILE" 2>>"$ERR_FILE"

log "Fine-tuning prerequisites generated successfully." | tee -a "$OUT_FILE"
log "training_data: $TRAINING_DATA_DIR" | tee -a "$OUT_FILE"
log "model_params: $MODEL_PARAMS_DIR" | tee -a "$OUT_FILE"