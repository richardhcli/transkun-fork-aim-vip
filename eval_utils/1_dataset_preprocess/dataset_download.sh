#!/bin/bash
#SBATCH --account yunglu
#SBATCH --partition=a100-80gb
#SBATCH --qos=standby
#SBATCH --ntasks=1 --cpus-per-task=8
#SBATCH --nodes=1 --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --job-name download_datasets
#SBATCH --output=/scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/1_dataset_preprocess/dataset_download.out
#SBATCH --error=/scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/1_dataset_preprocess/dataset_download.err

#run once
TARGET_DIR="$SCRATCH/datasets"
mkdir -p "$TARGET_DIR"

# ---------------------------------------------------------
# 0. Conda Environment Setup
# ---------------------------------------------------------
echo "Setting up Conda environment..."

# Load the Conda module (Name varies by university, e.g., anaconda/2023, miniconda)
module load conda 

# Initialize Conda for this non-interactive shell script
source $(conda info --base)/etc/profile.d/conda.sh

# Create an isolated environment for data tools (if it doesn't already exist)
conda create -y -n transkun_data_fetch_env python=3.10

# Activate the environment
conda activate transkun_data_fetch_env

# Install zenodo-get securely inside the Conda sandbox
pip install zenodo-get

# ---------------------------------------------------------
# 1. MAESTRO v3.0.0
# ---------------------------------------------------------
echo "Downloading and streaming extraction for MAESTRO v3..."
wget -qO- https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.zip | bsdtar -xf- -C "$TARGET_DIR"

mv "$TARGET_DIR/maestro-v3.0.0" "$TARGET_DIR/MAESTRO"

# ---------------------------------------------------------
# 2. A-MAPS (Zenodo 2590657)
# ---------------------------------------------------------
echo "Downloading A-MAPS..."
mkdir -p "$TARGET_DIR/A-MAPS"
zenodo_get 2590657 -o "$TARGET_DIR/A-MAPS"

unzip -q -o "$TARGET_DIR/A-MAPS/*.zip" -d "$TARGET_DIR/A-MAPS/" 2>/dev/null || true
rm -f "$TARGET_DIR/A-MAPS/"*.zip

# ---------------------------------------------------------
# 3. SMD v2 (Zenodo 13753319)
# ---------------------------------------------------------
echo "Downloading SMD v2..."
mkdir -p "$TARGET_DIR/SMD_v2"
zenodo_get 13753319 -o "$TARGET_DIR/SMD_v2"

unzip -q -o "$TARGET_DIR/SMD_v2/*.zip" -d "$TARGET_DIR/SMD_v2/" 2>/dev/null || true
rm -f "$TARGET_DIR/SMD_v2/"*.zip

# ---------------------------------------------------------
# 4. Isolate Testing Sample
# ---------------------------------------------------------
echo "Creating isolated testing folder..."
TEST_DIR="$TARGET_DIR/TESTING_SAMPLE"
mkdir -p "$TEST_DIR"

SAMPLE_WAV=$(find "$TARGET_DIR/MAESTRO" -type f -name "*.wav" | head -n 1)
SAMPLE_MIDI="${SAMPLE_WAV%.wav}.midi"

cp "$SAMPLE_WAV" "$TEST_DIR/"
cp "$SAMPLE_MIDI" "$TEST_DIR/"

echo "Data acquisition and setup complete."