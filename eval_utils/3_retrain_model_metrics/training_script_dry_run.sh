#!/bin/bash
#SBATCH --account yunglu
#SBATCH --partition=a100-80gb
#SBATCH --qos=normal
#SBATCH --ntasks=1 --cpus-per-task=8
#SBATCH --nodes=1 --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=05:00:00   
#SBATCH --job-name transkun_dryrun
#SBATCH --output=/scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/3_retrain_model_metrics/output/train_dry.out
#SBATCH --error=/scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/3_retrain_model_metrics/output/train_dry.err

# 5 hours is plenty for a dry run

#maestro csv 
# only important fields;
# "split" (train/validation/test)
# midi_filename (relative path to dataset dir)
# audio_filename (relative path to dataset dir)
# 



# Ensure your Conda environment is active first!
source /scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/0_environment_setup/setup_environment.sh
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

MAESTRO_METADATA_DIR="/scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/3_retrain_model_metrics/output/MAESTRO_METADATA"
MAESTRO_DIR="$SCRATCH/datasets/MAESTRO"
SAVE_DIR="/scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/3_retrain_model_metrics/transkun_checkpoints"
mkdir -p "$SAVE_DIR"

echo "[$(timestamp)] Starting Transkun Training Dry-Run..."

#  The Training Command
# # (transkun_aim) li5042@gilbreth-fe01:[transkun_fork] $ python -m transkun.train --help
# # usage: Perform Training [-h] --nProcess NPROCESS [--master_addr MASTER_ADDR] [--master_port MASTER_PORT] [--allow_tf32] --datasetPath DATASETPATH --datasetMetaFile_train
# #                         DATASETMETAFILE_TRAIN --datasetMetaFile_val DATASETMETAFILE_VAL [--batchSize BATCHSIZE] [--hopSize HOPSIZE] [--chunkSize CHUNKSIZE]
# #                         [--dataLoaderWorkers DATALOADERWORKERS] [--gradClippingQuantile GRADCLIPPINGQUANTILE] [--max_lr MAX_LR] [--weight_decay WEIGHT_DECAY] [--nIter NITER] --modelConf
# #                         MODELCONF [--augment] [--noiseFolder NOISEFOLDER] [--irFolder IRFOLDER]
# #                         saved_filename

# # positional arguments:
# #   saved_filename

# # options:
# #   -h, --help            show this help message and exit
# #   --nProcess NPROCESS   # of processes for parallel training
# #   --master_addr MASTER_ADDR
# #                         master address for distributed training
# #   --master_port MASTER_PORT
# #                         master port number for distributed training
# #   --allow_tf32
# #   --datasetPath DATASETPATH
# #   --datasetMetaFile_train DATASETMETAFILE_TRAIN
# #   --datasetMetaFile_val DATASETMETAFILE_VAL
# #   --batchSize BATCHSIZE
# #   --hopSize HOPSIZE
# #   --chunkSize CHUNKSIZE
# #   --dataLoaderWorkers DATALOADERWORKERS
# #   --gradClippingQuantile GRADCLIPPINGQUANTILE
# #   --max_lr MAX_LR
# #   --weight_decay WEIGHT_DECAY
# #   --nIter NITER
# #   --modelConf MODELCONF
# #                         the path to the model conf file
# #   --augment             do data augmentation
# #   --noiseFolder NOISEFOLDER
# #   --irFolder IRFOLDER
# The Corrected Training Dry Run Command
python -m transkun.train \
    --nProcess 1 \
    --datasetPath "$MAESTRO_DIR" \
    --datasetMetaFile_train "$MAESTRO_METADATA_DIR/train.pickle" \
    --datasetMetaFile_val "$MAESTRO_METADATA_DIR/val.pickle" \
    --modelConf "$MAESTRO_METADATA_DIR/transkun_base.json" \
    --nIter 500 \
    "$SAVE_DIR/checkpoint_baseline.pt"

echo "[$(timestamp)] Transkun Training Dry-Run Completed!"