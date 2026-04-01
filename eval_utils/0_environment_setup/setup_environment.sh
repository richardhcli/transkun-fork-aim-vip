#!/bin/bash
CONDA_ENV="/scratch/gilbreth/li5042/.conda/envs/transkun_aim"

module load cuda anaconda
conda activate $CONDA_ENV

#change to the transkun codebase directory
cd /scratch/gilbreth/li5042/transkun/transkun_fork