#!/bin/bash
#SBATCH --account yunglu
#SBATCH --partition=a100-80gb
#SBATCH --qos=normal
#SBATCH --ntasks=1 --cpus-per-task=8
#SBATCH --nodes=1 --gpus-per-node=2
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --job-name transkun_job
#SBATCH --output=/scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/2_recreate_metrics/official/output/evaluate_metrics.log.out
#SBATCH --error=/scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/2_recreate_metrics/official/output/evaluate_metrics.log.err

#===========================================
#run command: 
# sbatch /scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/2_recreate_metrics/official/run_official_metrics.sh

#SBATCH --qos=standby


MAESTRO_DATASET_PREPROCESSED=/scratch/gilbreth/li5042/datasets/MAESTRO
export MAESTRO_DATASET_PREPROCESSED

PREDICTION_OUTPUT_DIR=/scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/2_recreate_metrics/official/output
PREDICTED_MIDIS_DIR="$PREDICTION_OUTPUT_DIR/predicted_midis"
export PREDICTION_OUTPUT_DIR

CPU_PER_TASK=8
export CPU_PER_TASK

source /scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/0_environment_setup/setup_environment.sh

#OUTPUT_DIR=/scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/2_recreate_metrics/output
#export OUTPUT_DIR

# 1. Activate your environment
source /scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/0_environment_setup/setup_environment.sh

# 2. Run the multi-GPU batch transcription
# python /scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/2_recreate_metrics/transcribe_maestro_test.py \
#     --maestro_dir "$MAESTRO_DATASET_PREPROCESSED" \
#     --output_dir "$PREDICTION_OUTPUT_DIR"

cd /scratch/gilbreth/li5042/transkun/transkun_fork/transkun

python /scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/2_recreate_metrics/official/run_official_metrics.py \
    --maestro_dir "$MAESTRO_DATASET_PREPROCESSED" \
    --est_dir "$PREDICTED_MIDIS_DIR" \
    --workers 16 \
    --output_json "$PREDICTION_OUTPUT_DIR/transkun_paper_metrics.json"