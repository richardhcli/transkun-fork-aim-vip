#!/bin/bash
#SBATCH --account yunglu
#SBATCH --partition=a100-80gb
#SBATCH --qos=normal
#SBATCH --ntasks=1 --cpus-per-task=32
#SBATCH --nodes=1 --gpus-per-node=2
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --job-name transkun_metrics
#SBATCH --output=/scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/5_whole_pipeline/output/compute_metrics.out
#SBATCH --error=/scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/5_whole_pipeline/output/compute_metrics.err

# Configure paths
MAIN_SCRIPT_DIR="/scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/5_whole_pipeline"
MAESTRO_DIR="/scratch/gilbreth/li5042/datasets/MAESTRO"

# This is the directory where transcribe evaluation stored the predictions
# The python script automatically looks for the 'predicted_midis' folder inside it
EST_DIR="$MAIN_SCRIPT_DIR/output/metrics/full_eval_1776146937"

# Source environment
source "$MAIN_SCRIPT_DIR/prologue.sh"
cd /scratch/gilbreth/li5042/transkun/transkun_fork

echo "Computing comparison metrics against ground truth..."
echo "Predictions directory: $EST_DIR"

python "$MAIN_SCRIPT_DIR/compute_comparison_metrics.py" \
  --maestro_dir "$MAESTRO_DIR" \
  --est_dir "$EST_DIR" \
  --workers 32 \
  --split "test"

EXIT_CODE=$?

echo "Metrics calculation finished with code $EXIT_CODE"

source "$MAIN_SCRIPT_DIR/epilogue.sh"