#!/bin/bash
#SBATCH --account yunglu
#SBATCH --partition=a100-80gb
#SBATCH --qos=normal
#SBATCH --ntasks=1 --cpus-per-task=32
#SBATCH --nodes=1 --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --job-name transkun_finetune_verify
#SBATCH --output=/scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/5_whole_pipeline/output/main.sh.out
#SBATCH --error=/scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/5_whole_pipeline/output/main.sh.err

#====================================================================================================================================

MAIN_SCRIPT_DIR="/scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/5_whole_pipeline"
export MAIN_SCRIPT_DIR
MAIN_SCRIPT_NAME="main.sh"
export MAIN_SCRIPT_NAME
WORKING_DIR="/scratch/gilbreth/li5042/transkun/transkun_fork"
export WORKING_DIR

#====================================================================================================================================
RUNNING_SCRIPT="$MAIN_SCRIPT_DIR/$MAIN_SCRIPT_NAME"
export RUNNING_SCRIPT
RUNNING_SCRIPT_NAME=$MAIN_SCRIPT_NAME
export RUNNING_SCRIPT_NAME
#====================================================================================================================================


WHOLE_PIPELINE_DIR="$MAIN_SCRIPT_DIR"
export WHOLE_PIPELINE_DIR
MODEL_PREP_DIR="$WHOLE_PIPELINE_DIR/model_files_preparation"
export MODEL_PREP_DIR
CSV_DIR="$WHOLE_PIPELINE_DIR/csv_generation/CSVs"
export CSV_DIR
OUTPUT_DIR="$WHOLE_PIPELINE_DIR/output"
export OUTPUT_DIR
GENERATE_PICKLES_SCRIPT="$MODEL_PREP_DIR/generate_pickles.py"
export GENERATE_PICKLES_SCRIPT
TRAIN_SCRIPT="$WHOLE_PIPELINE_DIR/train.py"
export TRAIN_SCRIPT

DATASET_ROOT="${DATASET_ROOT:-/scratch/gilbreth/li5042/datasets}"
export DATASET_ROOT
#====================================================================================================================================


if [[ ! -f "$GENERATE_PICKLES_SCRIPT" ]]; then
	echo "[main.sh] ERROR: Missing generate_pickles.py: $GENERATE_PICKLES_SCRIPT"
	exit 1
fi

if [[ ! -f "$TRAIN_SCRIPT" ]]; then
	echo "[main.sh] ERROR: Missing train.py: $TRAIN_SCRIPT"
	exit 1
fi

if [[ ! -d "$CSV_DIR" ]]; then
	echo "[main.sh] ERROR: Missing CSV directory: $CSV_DIR"
	exit 1
fi

source $MAIN_SCRIPT_DIR/prologue.sh
cd $WORKING_DIR

#====================================================================================================================================
#main.sh code: 
echo "[main.sh] Starting Transkun fine-tuning evaluation pipeline at $(timestamp)"

python $MAIN_SCRIPT_DIR/main.py --mode smoke

#====================================================================================================================================
source $MAIN_SCRIPT_DIR/epilogue.sh
#====================================================================================================================================
