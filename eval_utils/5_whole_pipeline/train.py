#!/usr/bin/env python3
"""Universal training launcher for transkun.train.

This script expects train/val pickle files and a model-conf folder/path,
then calls `python -m transkun.train` with configurable training arguments.

Checkpoint behavior:
- This launcher expects a training-format `checkpoint.pt`.
- Inference-only checkpoints must be converted first with
    `model_files_preparation/prepare_train_checkpoint.py`.
"""

from __future__ import annotations

import argparse
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path


TRAINING_CHECKPOINT_KEYS = {
    "state_dict",
    "best_state_dict",
    "epoch",
    "nIter",
    "loss_tracker",
    "optimizer_state_dict",
    "lr_scheduler_state_dict",
}


# Resolve environment-provided paths in one helper to keep script defaults readable.
def env_path(
    name: str,
    default: Path | str | None = None,
    *,
    must_exist: bool = False,
    expect_dir: bool | None = None,
) -> Path:
    raw = os.environ.get(name, "").strip()
    if not raw:
        if default is None:
            raise EnvironmentError(f"{name} is not set")
        raw = str(default)

    path = Path(raw).expanduser().resolve()
    if must_exist and not path.exists():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    if expect_dir is True and path.exists() and not path.is_dir():
        raise NotADirectoryError(f"{name} is not a directory: {path}")
    if expect_dir is False and path.exists() and not path.is_file():
        raise FileNotFoundError(f"{name} is not a file: {path}")

    return path


WORKING_DIR = env_path("WORKING_DIR", must_exist=True, expect_dir=True)
if not (WORKING_DIR / "transkun" / "train.py").exists():
    raise FileNotFoundError(
        f"WORKING_DIR must point to transkun_fork repo root: {WORKING_DIR}"
    )

WHOLE_PIPELINE_DIR = env_path(
    "WHOLE_PIPELINE_DIR",
    default=WORKING_DIR / "eval_utils" / "5_whole_pipeline",
    must_exist=True,
    expect_dir=True,
)
MODEL_PREP_DIR = env_path(
    "MODEL_PREP_DIR",
    default=WHOLE_PIPELINE_DIR / "model_files_preparation",
    must_exist=True,
    expect_dir=True,
)
OUTPUT_DIR = env_path(
    "OUTPUT_DIR",
    default=WHOLE_PIPELINE_DIR / "output",
    must_exist=False,
)

DEFAULT_DATASET_ROOT = env_path(
    "DATASET_ROOT",
    default="/scratch/gilbreth/li5042/datasets",
    must_exist=False,
)
DEFAULT_PICKLE_DIR = env_path(
    "PICKLE_DIR_FULL",
    default=OUTPUT_DIR / "pickles" / "full",
    must_exist=False,
)
DEFAULT_CHECKPOINT_DIR = env_path(
    "CHECKPOINT_DIR",
    default=MODEL_PREP_DIR / "transkunV2" / "checkpointMSimpler",
    must_exist=False,
)
DEFAULT_MODEL_CONF_DIR = env_path(
    "MODEL_CONF_DIR_DEFAULT",
    default=DEFAULT_CHECKPOINT_DIR,
    must_exist=False,
)
DEFAULT_CHECKPOINT_OUT = env_path(
    "CHECKPOINT_OUT_FULL",
    default=OUTPUT_DIR / "checkpoints" / "full" / "checkpoint.pt",
    must_exist=False,
)


def log(message: str) -> None:
    print(message, flush=True)
    print(message, file=sys.stderr, flush=True)


def run_training_with_timeout(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout_seconds: float,
) -> tuple[int, bool]:
    """Run command and force-stop process group when timeout is reached."""
    if timeout_seconds <= 0:
        completed = subprocess.run(cmd, cwd=str(cwd), env=env, check=False)
        return completed.returncode, False

    process = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        start_new_session=True,
    )

    try:
        return process.wait(timeout=timeout_seconds), False
    except subprocess.TimeoutExpired:
        log(
            "[train.py] Time limit reached "
            f"({timeout_seconds:.2f}s). Stopping transkun.train process group."
        )

        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()

        return 0, True


def detect_n_process() -> int:
    """Infer process count from SLURM, CUDA visibility, or torch device count."""
    slurm_gpus = os.environ.get("SLURM_GPUS_ON_NODE", "").strip()
    if slurm_gpus:
        try:
            return max(1, int(slurm_gpus))
        except ValueError:
            pass

    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible_devices:
        devices = [dev for dev in visible_devices.split(",") if dev.strip()]
        if devices:
            return len(devices)

    try:
        import torch  # pylint: disable=import-outside-toplevel

        cuda_mod = getattr(torch, "cuda", None)
        if cuda_mod is not None and hasattr(cuda_mod, "device_count"):
            count = int(cuda_mod.device_count())
            if count > 0:
                return count
    except Exception:
        pass

    return 1


def resolve_model_conf(model_conf: Path | None, model_conf_dir: Path | None) -> Path:
    """Resolve model conf path from explicit file or from a model-conf directory."""
    if model_conf is not None:
        resolved = model_conf.resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Model conf does not exist: {resolved}")
        return resolved

    if model_conf_dir is None:
        raise ValueError("Either --model-conf or --model-conf-dir must be provided")

    model_conf_dir = model_conf_dir.resolve()
    candidates = [
        model_conf_dir / "transkun_base.json",
        model_conf_dir / "model.conf",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Could not find model conf in model-conf-dir. "
        f"Tried: {', '.join(str(c) for c in candidates)}"
    )


def resolve_source_checkpoint(model_conf: Path, model_conf_dir: Path | None) -> Path | None:
    """Find a source checkpoint.pt near model conf if one exists."""
    candidates: list[Path] = []
    if model_conf_dir is not None:
        candidates.append(model_conf_dir.resolve() / "checkpoint.pt")
    candidates.append(model_conf.resolve().parent / "checkpoint.pt")

    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate.resolve())
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return candidate.resolve()

    return None


def load_checkpoint_payload(checkpoint_path: Path) -> object:
    import torch  # pylint: disable=import-outside-toplevel

    torch_load = getattr(torch, "load")
    return torch_load(str(checkpoint_path), map_location="cpu")


def checkpoint_has_training_state(checkpoint_path: Path) -> bool:
    payload = load_checkpoint_payload(checkpoint_path)
    return isinstance(payload, dict) and TRAINING_CHECKPOINT_KEYS.issubset(payload.keys())


def prepare_training_artifacts(
    checkpoint_out: Path,
    model_conf: Path,
    model_conf_dir: Path | None,
) -> tuple[Path, Path]:
    """Return validated (model_conf, checkpoint) paths for transkun.train."""
    source_conf = model_conf.resolve()
    source_checkpoint = resolve_source_checkpoint(source_conf, model_conf_dir)

    if source_checkpoint is None:
        canonical = checkpoint_out.resolve()
        raise FileNotFoundError(
            "Training checkpoint not found near model conf. "
            f"Expected checkpoint at {canonical}. "
            "Run model_files_preparation/prepare_train_checkpoint.py first."
        )

    if not checkpoint_has_training_state(source_checkpoint):
        raise RuntimeError(
            f"Checkpoint is not in training format: {source_checkpoint}. "
            "Run model_files_preparation/prepare_train_checkpoint.py first."
        )

    return source_conf, source_checkpoint


def main() -> int:
    # main.sh exports WORKING_DIR as the repository root.
    repo_root = WORKING_DIR

    parser = argparse.ArgumentParser(description="Launch transkun.train with pickle metadata artifacts")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument(
        "--pickle-dir",
        type=Path,
        default=DEFAULT_PICKLE_DIR,
        help="Directory containing train.pickle and val.pickle",
    )

    parser.add_argument("--model-conf", type=Path, default=None, help="Optional direct path to model conf file")
    parser.add_argument(
        "--model-conf-dir",
        type=Path,
        default=DEFAULT_MODEL_CONF_DIR,
        help="Folder containing transkun_base.json or model.conf",
    )

    parser.add_argument("--checkpoint-out", type=Path, default=DEFAULT_CHECKPOINT_OUT)

    parser.add_argument("--n-process", type=int, default=0, help="If <=0, auto-detect GPU process count")
    parser.add_argument("--n-iter", type=int, default=180000)
    parser.add_argument("--max-train-seconds", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--data-loader-workers", type=int, default=8)
    parser.add_argument("--max-lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--allow-tf32", action="store_true")

    parser.add_argument("--hop-size", type=float, default=None)
    parser.add_argument("--chunk-size", type=float, default=None)

    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--noise-folder", type=Path, default=None)
    parser.add_argument("--ir-folder", type=Path, default=None)

    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    pickle_dir = args.pickle_dir.resolve()
    checkpoint_out = args.checkpoint_out.resolve()

    train_pickle = pickle_dir / "train.pickle"
    val_pickle = pickle_dir / "val.pickle"
    test_pickle = pickle_dir / "test.pickle"

    for path in (dataset_root, pickle_dir, train_pickle, val_pickle, test_pickle):
        if not path.exists():
            raise FileNotFoundError(f"Required path does not exist: {path}")

    # model_conf is read-only input; checkpoint path may be copied or reused per policy.
    model_conf = resolve_model_conf(args.model_conf, args.model_conf_dir)
    model_conf_effective, checkpoint_effective = prepare_training_artifacts(
        checkpoint_out=checkpoint_out,
        model_conf=model_conf,
        model_conf_dir=args.model_conf_dir,
    )

    n_process = args.n_process if args.n_process and args.n_process > 0 else detect_n_process()

    cmd = [
        sys.executable,
        "-m",
        "transkun.train",
        "--nProcess",
        str(n_process),
        "--datasetPath",
        str(dataset_root),
        "--datasetMetaFile_train",
        str(train_pickle),
        "--datasetMetaFile_val",
        str(val_pickle),
        "--batchSize",
        str(args.batch_size),
        "--dataLoaderWorkers",
        str(args.data_loader_workers),
        "--max_lr",
        str(args.max_lr),
        "--weight_decay",
        str(args.weight_decay),
        "--nIter",
        str(args.n_iter),
        "--modelConf",
        str(model_conf_effective),
    ]

    if args.allow_tf32:
        cmd.append("--allow_tf32")
    if args.hop_size is not None:
        cmd.extend(["--hopSize", str(args.hop_size)])
    if args.chunk_size is not None:
        cmd.extend(["--chunkSize", str(args.chunk_size)])
    if args.augment:
        cmd.append("--augment")
        if args.noise_folder is not None:
            cmd.extend(["--noiseFolder", str(args.noise_folder.resolve())])
        if args.ir_folder is not None:
            cmd.extend(["--irFolder", str(args.ir_folder.resolve())])

    cmd.append(str(checkpoint_effective))

    env = os.environ.copy()
    env["WORKING_DIR"] = str(WORKING_DIR)
    old_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{repo_root}:{old_pythonpath}" if old_pythonpath else str(repo_root)

    log(f"Using model conf: {model_conf_effective}")
    log(f"Using checkpoint path: {checkpoint_effective}")
    log(f"Using nProcess: {n_process}")
    log("Running command:")
    log(shlex.join(cmd))

    if args.dry_run:
        log("Dry run enabled; command not executed.")
        return 0

    returncode, timed_out = run_training_with_timeout(
        cmd,
        cwd=repo_root,
        env=env,
        timeout_seconds=max(0.0, args.max_train_seconds),
    )

    if timed_out:
        log("[train.py] Training force-stopped by time limit; continuing pipeline with latest available checkpoint file.")
        return 0

    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, cmd)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
