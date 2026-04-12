#!/bin/bash
#SBATCH --account yunglu
#SBATCH --partition=a100-80gb
#SBATCH --qos=normal
#SBATCH --ntasks=1 --cpus-per-task=32
#SBATCH --nodes=1 --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --job-name prepare_transkun_pickles
#SBATCH --output=/scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/3_retrain_model_metrics/augmented_retrain/preprocess_datasets/output/prepare_transkun_pickles.out
#SBATCH --error=/scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/3_retrain_model_metrics/augmented_retrain/preprocess_datasets/output/prepare_transkun_pickles.err

# set -euo pipefail

SCRIPT_DIR="/scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/3_retrain_model_metrics/augmented_retrain/preprocess_datasets"

source /scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/0_environment_setup/setup_environment.sh

cd "$SCRIPT_DIR"

python -u prepare_transkun_pickles.py --workers 24
