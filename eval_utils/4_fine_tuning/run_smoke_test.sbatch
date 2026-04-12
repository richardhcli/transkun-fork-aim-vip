#!/bin/bash
#SBATCH --account yunglu
#SBATCH --partition=a100-80gb
#SBATCH --qos=normal
#SBATCH --ntasks=1 --cpus-per-task=8
#SBATCH --nodes=1 --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --job-name transkun_smoke
#SBATCH --output=/scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/4_fine_tuning/output/smoke/sbatch_smoke.stdout.log
#SBATCH --error=/scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/4_fine_tuning/output/smoke/sbatch_smoke.stderr.log

set -euo pipefail

FINE_TUNING_DIR="/scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/4_fine_tuning"
SETUP_SCRIPT="/scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/0_environment_setup/setup_environment.sh"

# Ensure SBATCH output paths exist before the job writes to them.
mkdir -p "$FINE_TUNING_DIR/output/smoke"

source "$SETUP_SCRIPT"
cd "$FINE_TUNING_DIR"

# Allow each smoke step to run for up to 2 hours.
export SMOKE_TIMEOUT_SECONDS=7200

bash "$FINE_TUNING_DIR/run_smoke_test.sh"