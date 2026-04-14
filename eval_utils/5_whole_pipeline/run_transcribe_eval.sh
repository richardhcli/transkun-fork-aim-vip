#!/bin/bash
#SBATCH --account yunglu
#SBATCH --partition=a100-80gb
#SBATCH --qos=normal
#SBATCH --ntasks=1 --cpus-per-task=32
#SBATCH --nodes=1 --gpus-per-node=2
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --job-name transkun_eval
#SBATCH --output=/scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/5_whole_pipeline/output/transcribe_eval.out
#SBATCH --error=/scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/5_whole_pipeline/output/transcribe_eval.err

# Configure paths
MAIN_SCRIPT_DIR="/scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/5_whole_pipeline"
MAESTRO_DIR="/scratch/gilbreth/li5042/datasets/MAESTRO"

# We reuse the exact same directory from your timed-out job. 
# The StateManager in transcribe_maestro_test.py will automatically skip completed tracks and resume!
OUTPUT_DIR="$MAIN_SCRIPT_DIR/output/metrics/full_eval_1776146937"
CHECKPOINT_PT="$MAIN_SCRIPT_DIR/output/checkpoints/full/checkpoint.pt"

# Source environment
source "$MAIN_SCRIPT_DIR/prologue.sh"
cd /scratch/gilbreth/li5042/transkun/transkun_fork

echo "Resuming MAESTRO test transcription from checkpoint..."
echo "Output directory: $OUTPUT_DIR"

python "$MAIN_SCRIPT_DIR/transcribe_maestro_test.py" \
  --maestro_dir "$MAESTRO_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --checkpoint_pt "$CHECKPOINT_PT"

EXIT_CODE=$?

echo "Transcription process finished with code $EXIT_CODE"

source "$MAIN_SCRIPT_DIR/epilogue.sh"