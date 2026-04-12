#!/bin/bash
#SBATCH --account yunglu
#SBATCH --partition=a100-80gb
#SBATCH --qos=normal
#SBATCH --ntasks=1 --cpus-per-task=32
#SBATCH --nodes=1 --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --job-name transkun_finetune_verify
#SBATCH --output=/scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/4_fine_tuning/output/run_verify.out
#SBATCH --error=/scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/4_fine_tuning/output/run_verify.err
#SBATCH --chdir=/scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/4_fine_tuning

