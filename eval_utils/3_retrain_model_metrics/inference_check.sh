# source /scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/0_environment_setup/setup_environment.sh
# export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# MAESTRO_METADATA_DIR="/scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/3_retrain_model_metrics/output/MAESTRO_METADATA"
# MAESTRO_DIR="$SCRATCH/datasets/MAESTRO"
# SAVE_DIR="/scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/3_retrain_model_metrics/transkun_checkpoints"
# mkdir -p "$SAVE_DIR"

# echo "[$(timestamp)] Starting inference check"

# python -m transkun.transcribe \
#     "sample_audio.wav" \
#     "sample_output_pred.mid" \
#     --device "cpu" \
#     --weight "$SAVE_DIR/checkpoint_baseline.pt"
#     --conf "/scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/3_retrain_model_metrics/output/MAESTRO_METADATA/transkun_base.json"


