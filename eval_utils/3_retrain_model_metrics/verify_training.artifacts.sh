#!/bin/bash
#SBATCH --account yunglu
#SBATCH --partition=a100-80gb
#SBATCH --qos=normal
#SBATCH --ntasks=1 --cpus-per-task=8
#SBATCH --nodes=1 --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=00:30:00   
#SBATCH --job-name transkun_dryrun
#SBATCH --output=/scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/3_retrain_model_metrics/output/verify_training_artifacts.out
#SBATCH --error=/scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/3_retrain_model_metrics/output/verify_training_artifacts.err


source /scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/0_environment_setup/setup_environment.sh

echo "[$(timestamp)] Starting verification of training artifacts..."


python /scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/3_retrain_model_metrics/verify_training_artifacts.py

echo "[$(timestamp)] Verification completed."