# Transkun Model Files and Execution Environment Guidelines

This note documents the current, verified behavior in `eval_utils/5_whole_pipeline`.

## 1) Transkun-compatible required model files and formats

### Required files by stage

| Stage | Required file(s) | Notes |
|---|---|---|
| Checkpoint preparation | `source-checkpoint` (`.pt`), `source-model-conf` (`model.conf`) | Used by `model_files_preparation/prepare_train_checkpoint.py` |
| Training launch (`train.py`) | `checkpoint.pt` (training format), `model.conf`, `train.pickle`, `val.pickle`, `test.pickle` | `train.py` validates checkpoint schema before launching `python -m transkun.train` |
| Full evaluation (`main.py` full mode) | Trained `checkpoint.pt` + `model.conf` | Used by `transcribe_maestro_test.py` and metrics step |

### `.pt` checkpoint formats

There are two relevant `.pt` formats:

1. Inference-style checkpoint (not directly train-resumable)
- Usually only weights (`state_dict`) or a payload centered on inference weights.
- Missing optimizer/scheduler/counters.

2. Training-format checkpoint (required by `transkun.train` resume path)
- Must include all keys below:
  - `state_dict`
  - `best_state_dict`
  - `epoch`
  - `nIter`
  - `loss_tracker`
  - `optimizer_state_dict`
  - `lr_scheduler_state_dict`

These keys are produced by `transkun/TrainUtil.py` via `save_checkpoint(...)`.

### `epoch` and `nIter` ("epcho" means `epoch`)

- `epoch`:
  - The epoch index loaded at resume time.
  - In `transkun.train`, checkpoints are saved with `epoc + 1`, so resume starts from the next epoch.

- `nIter`:
  - Global optimizer-step counter loaded at resume time.
  - In `transkun.train`, `globalStep` starts from `startIter`, increments every batch, and checkpoints store `globalStep + 1`.

- How they are tracked:
  - Load path: `load_checkpoint(...)` reads `checkpoint['epoch']` and `checkpoint['nIter']`.
  - Save path: periodic and end-of-epoch `save_checkpoint(...)` updates both counters.
  - `loss_tracker` stores train/validation history across epochs.

Important detail: the CLI arg `--nIter` in `transkun.train` controls the OneCycleLR schedule horizon, not a hard training stop. In this pipeline, hard stop is enforced by `eval_utils/5_whole_pipeline/train.py --max-train-seconds`.

### How the current pipeline enforces compatibility

- `main.py` calls `ensure_training_checkpoint(...)` before launching `train.py`.
- If output checkpoint is missing, `main.py` calls `model_files_preparation/prepare_train_checkpoint.py`.
- `prepare_train_checkpoint.py` behavior:
  - If source is already training format: copy-through.
  - Else: initialize a training checkpoint structure, load pretrained weights tolerantly, and write training-format `.pt`.
  - Always ensures `model.conf` is next to output `checkpoint.pt`.

### Verified status (latest run)

`output/checkpoints/full/checkpoint.pt` was verified to contain all required training keys, with:
- `epoch = 0`
- `nIter = 0`
- Neighbor file exists: `output/checkpoints/full/model.conf`


## 2) AI execution-environment guidelines (srun + prologue)

These are the recommended operational rules for agents and scripts on Gilbreth.

### Required execution order

1. Start an interactive allocation (or submit `main.sh` via `sbatch`).
2. Enter allocated shell.
3. Source `prologue.sh` to load modules + activate conda env.
4. Run checkpoint preparation and/or pipeline commands.
5. Validate outputs/logs.

### Canonical command templates

Interactive allocation:

```bash
srun -A yunglu -p a100-80gb --qos=normal \
  --nodes=1 --ntasks-per-node=1 --gres=gpu:1 \
  --cpus-per-task=32 --mem=160G --time=02:00:00 --pty /bin/bash
```

Environment setup in that shell:

```bash
source /scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/5_whole_pipeline/prologue.sh
cd /scratch/gilbreth/li5042/transkun/transkun_fork
```

Prepare training checkpoint explicitly:

```bash
python /scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/5_whole_pipeline/model_files_preparation/prepare_train_checkpoint.py \
  --source-checkpoint /scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/5_whole_pipeline/model_files_preparation/transkunV2/checkpointMSimpler/checkpoint.pt \
  --source-model-conf /scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/5_whole_pipeline/model_files_preparation/transkunV2/checkpointMSimpler/model.conf \
  --output-checkpoint /scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/5_whole_pipeline/output/checkpoints/full/checkpoint.pt \
  --n-iter 180000 --force
```

Submit full batch pipeline:

```bash
sbatch /scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/5_whole_pipeline/main.sh
```

### AI guardrails and best practices

- **Audio Data Homogeneity**: The dataset batch collator requires all `audioSlices` tensors to have the same shape. Because raw datasets (like A-MAPS, MAESTRO, etc.) can contain a mixture of mono and stereo datasets (`[N, 1]` vs `[N, 2]`), you should pre-flatten the datasets into Mono `.wav` files using the `run_flatten_dataset_wavs.sh` script to avoid PyTorch `torch.stack` RuntimeErrors on DataLoader processes.
- Always run `prologue.sh` before Python GPU workflows.
- Keep `WORKING_DIR` pointed at repo root (`transkun_fork`).
- Prefer `--dry-run` for orchestration checks before long runs.
- For quick iteration, start with smoke mode first.
- Confirm checkpoint schema before training if there is any resume-format uncertainty.
- Read `output/main.sh.err` first when diagnosing failures.

### Cluster/module stability notes

- Avoid mixing ad-hoc module state with pipeline startup. If the shell has prior module modifications, reset to a clean state before sourcing `prologue.sh`.
- If torch import behavior is abnormal in an interactive shell, validate module state and environment consistency (`module list`, `which python`, `echo $CONDA_PREFIX`) before re-running.

### Quick checkpoint schema validation snippet

```bash
python - <<'PY'
from pathlib import Path
import torch

p = Path('/scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/5_whole_pipeline/output/checkpoints/full/checkpoint.pt')
keys = ['state_dict','best_state_dict','epoch','nIter','loss_tracker','optimizer_state_dict','lr_scheduler_state_dict']

print('exists', p.exists())
if p.exists():
    ck = torch.load(p, map_location='cpu')
    print('is_dict', isinstance(ck, dict))
    print('has_training_keys', isinstance(ck, dict) and all(k in ck for k in keys))
    if isinstance(ck, dict):
        print('epoch', ck.get('epoch'))
        print('nIter', ck.get('nIter'))
PY
```
