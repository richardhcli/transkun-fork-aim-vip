#!/bin/bash
CONDA_ENV="/scratch/gilbreth/li5042/.conda/envs/transkun_aim"

# Some callers use `set -u`; conda's activate/deactivate hooks may reference
# unset backup variables and crash under nounset. Temporarily relax nounset,
# then restore the previous shell state.
RESTORE_NOUNSET=0
if [[ "$-" == *u* ]]; then
  RESTORE_NOUNSET=1
  set +u
fi

module load cuda anaconda
conda activate "$CONDA_ENV"

if [[ "$RESTORE_NOUNSET" -eq 1 ]]; then
  set -u
fi

#change to the transkun codebase directory
cd /scratch/gilbreth/li5042/transkun/transkun_fork

# Gemini said
# Let's tackle these one at a time. The GLIBCXX error is a notorious HPC cluster bug, and the checkpoint question is a crucial aspect of managing PyTorch training states.
# 1. Fixing the GLIBCXX_3.4.31 not found Error
# This crash has nothing to do with your code or Transkun. It is a classic conflict between Conda and your cluster's operating system.
# Why it happens: A modern Python package you installed (optree) was compiled using a very new C++ standard library. However, Gilbreth's underlying operating system is likely running an older enterprise Linux version with outdated system libraries (/lib64/libstdc++.so.6). When Python tries to load the package, it defaults to the system's old C++ library, realizes it's missing the required modern features, and crashes.
# The Fix: Conda actually downloads the modern C++ libraries into your environment, but you have to explicitly tell the system to look there first before falling back to the OS system libraries.
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
else
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib"
fi

timestamp() {
  date +"%Y-%m-%d %H:%M:%S" # Custom format (e.g., 2024-05-20 14:30:05)
  # Usage
  #echo "$(timestamp) - Script started"
}

