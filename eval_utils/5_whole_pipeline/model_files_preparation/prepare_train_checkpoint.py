#!/usr/bin/env python3
"""Prepare a transkun.train-compatible checkpoint from an inference checkpoint."""

from __future__ import annotations

import argparse
import copy
import os
import shutil
import sys
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
CHECKPOINT_DIR = env_path(
    "CHECKPOINT_DIR",
    default=MODEL_PREP_DIR / "transkunV2" / "checkpointMSimpler",
    must_exist=False,
)
OUTPUT_DIR = env_path(
    "OUTPUT_DIR",
    default=WHOLE_PIPELINE_DIR / "output",
    must_exist=False,
)

DEFAULT_SOURCE_CHECKPOINT = env_path(
    "SOURCE_CHECKPOINT",
    default=CHECKPOINT_DIR / "checkpoint.pt",
    must_exist=False,
)
DEFAULT_SOURCE_MODEL_CONF = env_path(
    "SOURCE_MODEL_CONF",
    default=CHECKPOINT_DIR / "model.conf",
    must_exist=False,
)
DEFAULT_OUTPUT_CHECKPOINT = env_path(
    "CHECKPOINT_OUT_FULL",
    default=OUTPUT_DIR / "checkpoints" / "full" / "checkpoint.pt",
    must_exist=False,
)


def log(message: str) -> None:
    print(message, flush=True)


def load_payload(checkpoint_path: Path) -> object:
    import torch  # pylint: disable=import-outside-toplevel

    return torch.load(str(checkpoint_path), map_location="cpu")


def has_training_state(payload: object) -> bool:
    return isinstance(payload, dict) and TRAINING_CHECKPOINT_KEYS.issubset(payload.keys())


def extract_pretrained_state_dict(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise RuntimeError("Unsupported checkpoint payload type; expected dict")

    state_dict = payload.get("state_dict", payload)
    if not isinstance(state_dict, dict):
        raise RuntimeError("Checkpoint payload does not contain a usable state_dict")

    return state_dict


def ensure_output_model_conf(source_model_conf: Path, output_checkpoint: Path, dry_run: bool) -> Path:
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    output_model_conf = output_checkpoint.parent / "model.conf"

    if source_model_conf.resolve() == output_model_conf.resolve():
        return output_model_conf

    if dry_run:
        log(f"[prepare_train_checkpoint.py] DRY RUN copy conf: {source_model_conf} -> {output_model_conf}")
    else:
        shutil.copy2(source_model_conf, output_model_conf)

    return output_model_conf


def bootstrap_training_checkpoint(
    *,
    source_checkpoint: Path,
    output_checkpoint: Path,
    model_conf: Path,
    max_lr: float,
    weight_decay: float,
    n_iter: int,
    dry_run: bool,
) -> None:
    payload = load_payload(source_checkpoint)

    if has_training_state(payload):
        if dry_run:
            log(
                "[prepare_train_checkpoint.py] DRY RUN source checkpoint is already "
                f"training format; copy {source_checkpoint} -> {output_checkpoint}"
            )
            return
        shutil.copy2(source_checkpoint, output_checkpoint)
        return

    if dry_run:
        log(
            "[prepare_train_checkpoint.py] DRY RUN bootstrap training checkpoint "
            f"from {source_checkpoint} -> {output_checkpoint}"
        )
        return

    repo_root = WORKING_DIR
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    import moduleconf  # pylint: disable=import-outside-toplevel
    import torch  # pylint: disable=import-outside-toplevel
    from transkun.TrainUtil import (  # pylint: disable=import-outside-toplevel
        initializeCheckpoint,
        load_state_dict_tolerant,
        save_checkpoint,
    )

    pretrained_state = extract_pretrained_state_dict(payload)

    conf_manager = moduleconf.parseFromFile(str(model_conf))
    transkun_cls = conf_manager["Model"].module.TransKun
    conf = conf_manager["Model"].config

    start_epoch, start_iter, model, loss_tracker, _best_state_dict, optimizer, lr_scheduler = (
        initializeCheckpoint(
            transkun_cls,
            device=torch.device("cpu"),
            max_lr=max_lr,
            weight_decay=weight_decay,
            nIter=n_iter,
            conf=conf,
        )
    )

    load_state_dict_tolerant(model, pretrained_state)
    best_state_dict = copy.deepcopy(model.state_dict())

    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        str(output_checkpoint),
        start_epoch,
        start_iter,
        model,
        loss_tracker,
        best_state_dict,
        optimizer,
        lr_scheduler,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare a training-format checkpoint from inference weights"
    )
    parser.add_argument("--source-checkpoint", type=Path, default=DEFAULT_SOURCE_CHECKPOINT)
    parser.add_argument("--source-model-conf", type=Path, default=DEFAULT_SOURCE_MODEL_CONF)
    parser.add_argument("--output-checkpoint", type=Path, default=DEFAULT_OUTPUT_CHECKPOINT)
    parser.add_argument("--n-iter", type=int, default=180000)
    parser.add_argument("--max-lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    source_checkpoint = args.source_checkpoint.expanduser().resolve()
    source_model_conf = args.source_model_conf.expanduser().resolve()
    output_checkpoint = args.output_checkpoint.expanduser().resolve()

    if not source_checkpoint.exists():
        raise FileNotFoundError(f"Source checkpoint does not exist: {source_checkpoint}")
    if not source_model_conf.exists():
        raise FileNotFoundError(f"Source model conf does not exist: {source_model_conf}")

    output_model_conf = output_checkpoint.parent / "model.conf"
    if output_checkpoint.exists() and not args.force:
        payload = load_payload(output_checkpoint)
        if has_training_state(payload):
            if not output_model_conf.exists():
                ensure_output_model_conf(source_model_conf, output_checkpoint, args.dry_run)
            log(
                "[prepare_train_checkpoint.py] Existing training checkpoint is valid; "
                "skipping regeneration."
            )
            return 0

        log(
            "[prepare_train_checkpoint.py] Existing output checkpoint is not in training "
            "format; regenerating."
        )

    target_model_conf = ensure_output_model_conf(source_model_conf, output_checkpoint, args.dry_run)
    bootstrap_training_checkpoint(
        source_checkpoint=source_checkpoint,
        output_checkpoint=output_checkpoint,
        model_conf=target_model_conf,
        max_lr=args.max_lr,
        weight_decay=args.weight_decay,
        n_iter=args.n_iter,
        dry_run=args.dry_run,
    )

    if args.dry_run:
        log("[prepare_train_checkpoint.py] DRY RUN complete.")
    else:
        log(f"[prepare_train_checkpoint.py] Wrote training checkpoint: {output_checkpoint}")
        log(f"[prepare_train_checkpoint.py] Wrote model conf: {target_model_conf}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())