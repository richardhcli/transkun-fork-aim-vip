#!/bin/bash
#====================================================================================================================================
#README: 
# This script sets up the environment for running Transkun fine-tuning evaluations.
# It loads necessary modules, activates the Conda environment,
#Usage: 
# Called by other scripts to ensure the environment is correctly configured before executing fine-tuning related tasks.
#====================================================================================================================================

CONDA_ENV="/scratch/gilbreth/li5042/.conda/envs/transkun_aim"

#====================================================================================================================================
START_TIME=$(date +%s) # Record the start time of the pipeline



export CONDA_ENV
export START_TIME

RUNNING_SCRIPT="${RUNNING_SCRIPT:-$MAIN_SCRIPT_DIR/main.sh}"


echo "[prologue.sh] Starting $RUNNING_SCRIPT at $(date +"%Y-%m-%d %H:%M:%S") at START_TIME $START_TIME"

#do not use strict mode
set +u

module load cuda anaconda #disables that strict behavior
conda activate $CONDA_ENV

#change to the transkun codebase directory
#cd /scratch/gilbreth/li5042/transkun/transkun_fork
#done later

# Gemini said
# Let's tackle these one at a time. The GLIBCXX error is a notorious HPC cluster bug, and the checkpoint question is a crucial aspect of managing PyTorch training states.
# 1. Fixing the GLIBCXX_3.4.31 not found Error
# This crash has nothing to do with your code or Transkun. It is a classic conflict between Conda and your cluster's operating system.
# Why it happens: A modern Python package you installed (optree) was compiled using a very new C++ standard library. However, Gilbreth's underlying operating system is likely running an older enterprise Linux version with outdated system libraries (/lib64/libstdc++.so.6). When Python tries to load the package, it defaults to the system's old C++ library, realizes it's missing the required modern features, and crashes.
# The Fix: Conda actually downloads the modern C++ libraries into your environment, but you have to explicitly tell the system to look there first before falling back to the OS system libraries.
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH


timestamp() {
date +"%Y-%m-%d %I:%M:%S %p" # Custom format with AM/PM (e.g., 2024-05-20 02:30:05 PM)
  # Usage
  #echo "$(timestamp) - Script started"
}