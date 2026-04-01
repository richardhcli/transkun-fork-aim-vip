#!/bin/bash
#SBATCH --account yunglu
#SBATCH --partition=a100-80gb
#SBATCH --qos=standby
#SBATCH --ntasks=1 --cpus-per-task=8
#SBATCH --nodes=1 --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=00:20:00   
#SBATCH --job-name transkun_dryrun
#SBATCH --output=/scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/3_retrain_model_metrics/output/generate_prereq_files.out
#SBATCH --error=/scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/3_retrain_model_metrics/output/generate_prereq_files.err

#run this ONCE

# Ensure your Conda environment is active first!
source /scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/0_environment_setup/setup_environment.sh

#1. Generate the Dataset Metadata (.pt files)
# 1. Define your new custom metadata directory
MAESTRO_METADATA_DIR="/scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/3_retrain_model_metrics/output/MAESTRO_METADATA"
#shove into 3_retrain_model_metrics/output/ so that it is ignored-- pickle files are too large. 

# 2. Create the directory
mkdir -p "$MAESTRO_METADATA_DIR"

#  3. Run the generator with the new output path
# # python -m transkun.createDatasetMaestro --help
# # usage: createDatasetMaestro.py [-h] [--noPedalExtension] datasetPath metadataCSVPath outputPath

# # positional arguments:
# #   datasetPath         folder path to the maestro dataset
# #   metadataCSVPath     path to the metadata file of the maestro dataset (csv)
# #   outputPath          path to the output folder

# # options:
# #   -h, --help          show this help message and exit
# #   --noPedalExtension  Do not perform pedal extension according to the sustain pedal
python -m transkun.createDatasetMaestro \
    "$SCRATCH/datasets/MAESTRO/" \
    "$SCRATCH/datasets/MAESTRO/maestro-v3.0.0.csv" \
    "$MAESTRO_METADATA_DIR"


python -m moduleconf.generate Model:transkun.ModelTransformer > "$MAESTRO_METADATA_DIR/transkun_base.json"
