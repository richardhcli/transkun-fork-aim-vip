#!/bin/bash
#SBATCH --account yunglu
#SBATCH --partition=a100-80gb
#SBATCH --qos=normal
#SBATCH --ntasks=1 --cpus-per-task=32
#SBATCH --nodes=1 --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --job-name amaps_mid_to_wav
#SBATCH --output=/scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/5_whole_pipeline/output/mid_to_wav.out
#SBATCH --error=/scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/5_whole_pipeline/output/mid_to_wav.err

#run: source /scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/5_whole_pipeline/pickle_generation/mid_to_wav.sh

START_TIME=$(date +%s) # Record the start time of the pipeline
echo "[mid_to_wav.sh] Starting MIDI to WAV conversion at $(date +"%Y-%m-%d %H:%M:%S") at START_TIME $START_TIME"
RUNNING_SCRIPT="mid_to_wav.sh"
timestamp() {
    date +"%Y-%m-%d %I:%M:%S %p" # Custom format with AM/PM (e.g., 2024-05-20 02:30:05 PM)
}

module load ffmpeg
module load parallel

# Plug-and-play dataset root: change this path to swap datasets.
DATASET_DIR="/scratch/gilbreth/li5042/datasets/A-MAPS"
DATASET_NAME="$(basename "$DATASET_DIR")"

if [ ! -d "$DATASET_DIR" ]; then
    echo "Dataset directory not found: $DATASET_DIR"
    exit 1
fi


# Clone a dataset repository and keep relevant files
# git clone --depth=1 <dataset_repo_url> "$DATASET_DIR"
# find "$DATASET_DIR" -mindepth 1 -maxdepth 1 ! -name "bimmuda_dataset" -exec rm -rf {} +
# if [ -d "$DATASET_DIR/bimmuda_dataset" ]; then
#     mv "$DATASET_DIR"/bimmuda_dataset/* "$DATASET_DIR"/
#     rmdir "$DATASET_DIR"/bimmuda_dataset
# fi

# # Remove songs with no main melody
# NO_MELODY_DIRS=(
#     "1992/2"
#     "1992/3"
#     "1993/2"
#     "1997/5"
#     "2017/4"
# )
# for bad_dir in "${NO_MELODY_DIRS[@]}"; do
#     rm -rf "$DATASET_DIR/$bad_dir"
# done

# Delete all non-MIDI essential files
# find "$DATASET_DIR" -type f -name "*.mid" ! -name "*_full.mid" -delete
# find "$DATASET_DIR" -type f -name "*.mscz" -delete
# find "$DATASET_DIR" -type f -name "*.txt" -delete
# find "$DATASET_DIR" -name .DS_Store -type f -delete
# find "$DATASET_DIR" -type d -name ".mscbackup" -exec rm -r {} +

# Create a Singularity container for FluidSynth
FS_CONTAINER="fluidsynth.sif"
FS_DEFINITION="fluidsynth.def"
cat <<EOF >$FS_DEFINITION
BootStrap: docker
From: ubuntu:22.04

%post
    apt-get update && apt-get install -y \
        fluidsynth \
        wget \
        ffmpeg \
        curl \
        ca-certificates

    # Download the FluidR3 GM soundfont
    mkdir -p /usr/share/sounds/sf2
    wget -O /usr/share/sounds/sf2/FluidR3_GM.sf2 https://github.com/pianobooster/fluid-soundfont/releases/download/v3.1/FluidR3_GM.sf2

%environment
    export SOUND_FONT=/usr/share/sounds/sf2/FluidR3_GM.sf2

%runscript
    exec fluidsynth -ni "\$SOUND_FONT" "\$@"
EOF
singularity build --force "$FS_CONTAINER" "$FS_DEFINITION" >/dev/null
SF_PATH="/usr/share/sounds/sf2/FluidR3_GM.sf2"

# Function to convert a single MIDI file
convert_midi() {
    local midi_path="$1"
    local out="${midi_path%.mid}.wav"
    local tmp="${out%.wav}_tmp.wav"

    echo "Processing: $midi_path"

    # Convert MIDI to WAV using FluidSynth
    if singularity exec "$FS_CONTAINER" fluidsynth -ni "$SF_PATH" "$midi_path" -F "$tmp" -r 44100 2>/dev/null; then
        # Convert to mono 44.1kHz with ffmpeg
        if ffmpeg -loglevel error -y -i "$tmp" -ac 1 -ar 44100 "$out" 2>/dev/null; then
            rm "$tmp"
            echo "Converted: $out"
        else
            rm -f "$tmp"
            echo "FFmpeg failed on: $midi_path"
        fi
    else
        echo "FluidSynth failed on: $midi_path"
    fi
}

# Export the function and variables for parallel
export -f convert_midi
export FS_CONTAINER
export SF_PATH

# Run conversion in parallel (using all available CPU cores)
find "$DATASET_DIR" -type f -name "*.mid" | parallel --jobs 32 convert_midi {}

# Clean up
rm -f "$FS_DEFINITION" "$FS_CONTAINER"

# Generate a sorted list of all input files
find "$(realpath "$DATASET_DIR")" -type f -name "*.wav" | sort >"${DATASET_NAME}.txt"

# Print the number of .MID files and then .WAV files
echo "Number of .MID files: $(find "$DATASET_DIR" -type f -name "*.mid" | wc -l)"
echo "Number of .WAV files: $(find "$DATASET_DIR" -type f -name "*.wav" | wc -l)"


#ping user
echo "MIDI to WAV conversion completed at $(timestamp)"
source /scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/5_whole_pipeline/epilogue.sh "mid_to_wav.sh"