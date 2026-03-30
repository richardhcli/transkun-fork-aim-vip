

MAESTRO_DATASET_PREPROCESSED=/scratch/gilbreth/li5042/datasets/maestro_dataset
PREDICTION_OUTPUT_DIR=/scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/output/
OUTPUT_DIR=/scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/output
export MAESTRO_DATASET_PREPROCESSED
export PREDICTION_OUTPUT_DIR
export OUTPUT_DIR

# 1. Activate your environment
#source transkun/transkun_fork/eval_utils/environment_setup/setup_environment.sh

# 2. Run the batch transcription (GPU-bound, sequential)
# python /scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/transcribe_all.py \
#     --maestro_dir "$MAESTRO_DATASET_PREPROCESSED" \
#     --output_dir "$PREDICTION_OUTPUT_DIR" \
#     --device cuda
# 2. Run the multi-GPU batch transcription
python /scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/transcribe_all_multiGPU.py \
    --maestro_dir "$MAESTRO_DATASET_PREPROCESSED" \
    --output_dir "$PREDICTION_OUTPUT_DIR"


# 3. Run the holistic evaluation (CPU-bound, highly parallel)
python /scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/compute_comparison_metrics.py \
    --maestro_dir "$MAESTRO_DATASET_PREPROCESSED" \
    --est_dir "$PREDICTION_OUTPUT_DIR" \
    --workers $CPU_PER_TASK