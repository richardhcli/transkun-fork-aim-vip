#!/bin/bash
#record of my one-time setup needed for the environment. 
#only run this script once. 

CONDA_ENV="/scratch/gilbreth/li5042/.conda/envs/transkun_aim"

module load cuda anaconda

conda create -p $CONDA_ENV

conda activate $CONDA_ENV

#--prune: When you run a standard conda env update, Conda will only add or update packages. If you deleted a package from your environment.yml file, Conda will leave it installed in your environment. Adding the --prune flag forces Conda to uninstall any packages that are no longer listed in your .yml file, keeping your environment perfectly synced:
conda env update --prune --prefix=$CONDA_ENV --file=/scratch/gilbreth/li5042/transkun/transkun_fork/environment.yml

pip install -r requirements.txt

# #!/bin/bash
# # ==============================================================================
# # Script Name: setup_env.sh
# # Description: One-time environment creation with robust network retries and 
# #              HPC quota protection.
# # Usage: source setup_env.sh <MODEL_NAME> <SCRATCH_PATH>
# # ==============================================================================

# if [ -z "$1" ] || [ -z "$2" ]; then
#     echo "[ERROR] Usage: source setup_env.sh <MODEL_NAME> <SCRATCH_PATH>"
#     return 1 2>/dev/null || exit 1
# fi

# MODEL_NAME_RAW="$1"
# MODEL_NAME=${MODEL_NAME_RAW// /_}
# USER_SCRATCH_ROOT=$(echo "$2" | cut -d'/' -f1-4)

# WORKING_DIR=$(pwd)
# ENV_NAME="running-env-${MODEL_NAME}"
# ENV_PATH="${WORKING_DIR}/.venv-${ENV_NAME}"
# ENVIRONMENT_FILE="${WORKING_DIR}/environment.yml"

# echo "[INFO] Initializing setup for $MODEL_NAME in $WORKING_DIR..."

# # --- 1. HPC Quota Protection (The Home Hijack) ---
# export ORIGINAL_HOME="$HOME"
# export HOME="$USER_SCRATCH_ROOT"

# export XDG_CACHE_HOME="$USER_SCRATCH_ROOT/.cache"
# export XDG_CONFIG_HOME="$USER_SCRATCH_ROOT/.config"
# export CONDARC="$USER_SCRATCH_ROOT/.condarc"
# export CONDA_PKGS_DIRS="$USER_SCRATCH_ROOT/.conda/pkgs"
# export PIP_CACHE_DIR="$XDG_CACHE_HOME/pip"

# mkdir -p "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR"

# # --- 2. Conda/Mamba Initialization ---
# module load anaconda 2>/dev/null || true
# conda config --set solver libmamba
# eval "$(conda shell.bash hook)"

# # Remove existing environment if rebuilding
# if [ -d "$ENV_PATH" ]; then
#     echo "[INFO] Removing existing environment at $ENV_PATH..."
#     conda env remove -y --prefix "$ENV_PATH" >/dev/null 2>&1
#     rm -rf "$ENV_PATH"
# fi

# # --- 3. Environment Creation (With Retry Logic) ---
# if [ ! -f "$ENVIRONMENT_FILE" ]; then
#     echo "[ERROR] Missing $ENVIRONMENT_FILE"
#     export HOME="$ORIGINAL_HOME"
#     return 1 2>/dev/null || exit 1
# fi

# MAX_ATTEMPTS=3
# ENV_SUCCESS=false

# for ((i=1; i<=MAX_ATTEMPTS; i++)); do
#     echo "[INFO] Creating conda env (Attempt $i/$MAX_ATTEMPTS)..."
#     if conda env create -y -q -f "$ENVIRONMENT_FILE" --prefix "$ENV_PATH" >/dev/null 2>"$WORKING_DIR/env-create.log"; then
#         ENV_SUCCESS=true
#         break
#     elif [[ $i -lt $MAX_ATTEMPTS ]]; then
#         echo "[WARN] Env creation failed — retrying in a few seconds..."
#         sleep $((RANDOM % 10 + 5))
#     fi
# done

# if [ "$ENV_SUCCESS" = false ]; then
#     echo "[ERROR] Failed to create environment after $MAX_ATTEMPTS attempts."
#     cat "$WORKING_DIR/env-create.log"
#     export HOME="$ORIGINAL_HOME"
#     return 1 2>/dev/null || exit 1
# fi

# # --- 4. Validation ---
# conda activate "$ENV_PATH"
# PYTHON_CMD="$ENV_PATH/bin/python"
# PY_VER=$($PYTHON_CMD -c 'import sys; print(sys.version.split()[0])' 2>/dev/null)

# if command -v nvidia-smi &>/dev/null && nvidia-smi -L | grep -q "GPU"; then
#     echo "[INFO] GPU detected on node via nvidia-smi."
# else
#     echo "[WARN] No GPU detected — GPU libraries may not be usable."
# fi

# if $PYTHON_CMD -c "import torch" 2>/dev/null; then
#     PT_VER=$($PYTHON_CMD -c "import torch; print(torch.__version__)" 2>/dev/null)
#     CUDA_AVAIL=$($PYTHON_CMD -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)
#     echo "[RESULT] $MODEL_NAME_RAW | Python: $PY_VER | PyTorch: $PT_VER | CUDA: $CUDA_AVAIL"
# else
#     echo "[RESULT] $MODEL_NAME_RAW | Python: $PY_VER | PyTorch not detected."
# fi

# conda deactivate
# export HOME="$ORIGINAL_HOME"
# echo "[INFO] Setup complete. Use activate_env.sh to load this environment."