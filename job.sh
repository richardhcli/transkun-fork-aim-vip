#!/bin/bash
#SBATCH --account yunglu
#SBATCH --partition=a100-80gb
#SBATCH --qos=standby
#SBATCH --ntasks=1 --cpus-per-task=16
#SBATCH --nodes=1 --gpus-per-node=2
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --job-name transkun_job
#SBATCH --output=/scratch/gilbreth/li5042/transkun/transkun_fork/myjob.out
#SBATCH --error=/scratch/gilbreth/li5042/transkun/transkun_fork/myjob.err

#===========================================
#run command: 
# sbatch /scratch/gilbreth/li5042/transkun/transkun_fork/job.sh

#===========================================
# README: 
# This is the main job script for running the transcription and evaluation pipeline on the HPC cluster.

#1) run /scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/environment_setup/one_time_init.sh

echo "Starting the transcription and evaluation pipeline..."
source /scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/evaluate_metrics.sh