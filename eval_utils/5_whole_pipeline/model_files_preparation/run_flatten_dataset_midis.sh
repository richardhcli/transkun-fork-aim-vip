#!/bin/bash
#SBATCH --account yunglu
#SBATCH --partition=a100-80gb
#SBATCH --qos=normal
#SBATCH --ntasks=1 --cpus-per-task=32
#SBATCH --nodes=1 --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name flatten_maestro
#SBATCH --output=/scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/5_whole_pipeline/output/flatten_dataset.out
#SBATCH --error=/scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/5_whole_pipeline/output/flatten_dataset.err

# Configure paths
SCRIPT_DIR="/scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/5_whole_pipeline/model_files_preparation"
DATASETS_DIR="/scratch/gilbreth/li5042/datasets"
LOG_DIR="/scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/5_whole_pipeline/output/flatten_logs"
mkdir -p "$LOG_DIR"

# Source normal prologue for environment
source "/scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/5_whole_pipeline/prologue.sh"

echo "Starting one-time dataset MIDI flattening..."

echo "Granting read and write permissions to datasets directory..."
chmod -R u+rw "$DATASETS_DIR"

# Flatten script
python "$SCRIPT_DIR/flatten_dataset_midis.py" \
  --dataset-dir "$DATASETS_DIR" \
  --output-json "$LOG_DIR/flatten_log.json" \
  --output-log "$LOG_DIR/flatten_script.log"

EXIT_CODE=$?

echo "Reverting permissions on datasets directory (removing write access)..."
chmod -R a-w "$DATASETS_DIR"

echo "Flattening process finished with code $EXIT_CODE"

if [ $EXIT_CODE -eq 0 ]; then
    echo "Running quick summary over the JSON log outputs..."
    
    python - <<'EOF'
import json
from pathlib import Path

log_file = Path("/scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/5_whole_pipeline/output/flatten_logs/flatten_log.json")

if not log_file.exists():
    print(f"Log file not found: {log_file}")
    exit(1)

with open(log_file, "r") as f:
    data = json.load(f)

modified = 0
skipped = 0
errors = 0

for item in data:
    status = item.get("status")
    if status == "modified":
        modified += 1
    elif status == "skipped":
        skipped += 1
    else:
        errors += 1

print("=== Flatten Summary ===")
print(f"Total checked: {len(data)}")
print(f"Modified: {modified}")
print(f"Skipped (already single-track): {skipped}")
print(f"Errors: {errors}")

if errors > 0:
    print("\nError sampling:")
    for item in data:
        if "error" in item.get("status", ""):
            print(f"File: {item.get('path')} => Error: {item.get('error')}")

print("\nTesting validation complete.")
EOF

else
    echo "Error detected during Python script execution. Check flatten_script.log."
fi

RUNNING_SCRIPT="$SCRIPT_DIR/flatten_dataset_midis.sh"
export RUNNING_SCRIPT
source "/scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/5_whole_pipeline/epilogue.sh"