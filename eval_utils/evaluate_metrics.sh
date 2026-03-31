#!/bin/bash
#SBATCH --account yunglu
#SBATCH --partition=a100-80gb
#SBATCH --qos=standby
#SBATCH --ntasks=1 --cpus-per-task=8
#SBATCH --nodes=1 --gpus-per-node=2
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --job-name transkun_job
#SBATCH --output=/scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/output/myjob.out
#SBATCH --error=/scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/output/myjob.err

#===========================================
#run command: 
# sbatch /scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/evaluate_metrics.sh


MAESTRO_DATASET_PREPROCESSED=/scratch/gilbreth/li5042/datasets/MAESTRO
export MAESTRO_DATASET_PREPROCESSED

PREDICTION_OUTPUT_DIR=/scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/2_recreate_metrics/output
export PREDICTION_OUTPUT_DIR

CPU_PER_TASK=8
export CPU_PER_TASK

#OUTPUT_DIR=/scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/2_recreate_metrics/output
#export OUTPUT_DIR

# 1. Activate your environment
#source transkun/transkun_fork/eval_utils/environment_setup/setup_environment.sh

# 2. Run the batch transcription (GPU-bound, sequential)
# python /scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/transcribe_all.py \
#     --maestro_dir "$MAESTRO_DATASET_PREPROCESSED" \
#     --output_dir "$PREDICTION_OUTPUT_DIR" \
#     --device cuda
# 2. Run the multi-GPU batch transcription
python /scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/2_recreate_metrics/transcribe_all_multiGPU.py \
    --maestro_dir "$MAESTRO_DATASET_PREPROCESSED" \
    --output_dir "$PREDICTION_OUTPUT_DIR"


# 3. Run the holistic evaluation (CPU-bound, highly parallel)
python /scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/2_recreate_metrics/compute_comparison_metrics.py \
    --maestro_dir "$MAESTRO_DATASET_PREPROCESSED" \
    --est_dir "$PREDICTION_OUTPUT_DIR" \
    --workers $CPU_PER_TASK