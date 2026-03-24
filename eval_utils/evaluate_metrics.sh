

MAESTRO_DATASET_PREPROCESSED=/scratch/gilbreth/li5042/datasets/maestro_dataset
PREDICTION_OUTPUT_DIR=/scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/output/predicted_midis

# 1. Activate your environment
#source transkun/transkun_fork/eval_utils/environment_setup/setup_environment.sh

# 2. Run the batch transcription (GPU-bound, sequential)
python /scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/transcribe_all.py \
    --maestro_dir "$MAESTRO_DATASET_PREPROCESSED" \
    --output_dir "$PREDICTION_OUTPUT_DIR" \
    --device cuda

# 3. Run the holistic evaluation (CPU-bound, highly parallel)
python /scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/evaluate_metrics.py \
    --maestro_dir "$MAESTRO_DATASET_PREPROCESSED" \
    --est_dir "$PREDICTION_OUTPUT_DIR" \
    --workers 16