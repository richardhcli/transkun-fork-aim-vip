#!/usr/bin/env python3
"""Universal training launcher for transkun.train.

This script expects train/val pickle files and a model-conf folder/path,
then calls `python -m transkun.train` with configurable training arguments.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

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
DEFAULT_MODEL_CONF_DIR = env_path(
    "MODEL_CONF_DIR_DEFAULT",
    default=MODEL_PREP_DIR / "transkunV2" / "checkpointMSimpler",
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


def detect_n_process() -> int:
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

        count = int(torch.cuda.device_count())
        if count > 0:
            return count
    except Exception:
        pass

    return 1


def resolve_model_conf(model_conf: Path | None, model_conf_dir: Path | None) -> Path:
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


def main() -> int:
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

    model_conf = resolve_model_conf(args.model_conf, args.model_conf_dir)

    checkpoint_out.parent.mkdir(parents=True, exist_ok=True)

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
        str(model_conf),
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

    cmd.append(str(checkpoint_out))

    env = os.environ.copy()
    env["WORKING_DIR"] = str(WORKING_DIR)
    old_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{repo_root}:{old_pythonpath}" if old_pythonpath else str(repo_root)

    log(f"Using model conf: {model_conf}")
    log(f"Using nProcess: {n_process}")
    log("Running command:")
    log(shlex.join(cmd))

    if args.dry_run:
        log("Dry run enabled; command not executed.")
        return 0

    subprocess.run(cmd, cwd=str(repo_root), env=env, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
