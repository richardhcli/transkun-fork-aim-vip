#!/usr/bin/env python3
"""Entry point for whole-pipeline smoke, user, and full fine-tune runs."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from time import time

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
CSV_DIR = env_path(
    "CSV_DIR",
    default=WHOLE_PIPELINE_DIR / "csv_generation" / "CSVs",
    must_exist=True,
    expect_dir=True,
)
OUTPUT_DIR = env_path(
    "OUTPUT_DIR",
    default=WHOLE_PIPELINE_DIR / "output",
    must_exist=False,
)
GENERATE_PICKLES_SCRIPT = env_path(
    "GENERATE_PICKLES_SCRIPT",
    default=MODEL_PREP_DIR / "generate_pickles.py",
    must_exist=True,
    expect_dir=False,
)
TRAIN_SCRIPT = env_path(
    "TRAIN_SCRIPT",
    default=WHOLE_PIPELINE_DIR / "train.py",
    must_exist=True,
    expect_dir=False,
)
DEFAULT_DATASET_ROOT = env_path(
    "DATASET_ROOT",
    default="/scratch/gilbreth/li5042/datasets",
    must_exist=False,
)


@dataclass(frozen=True)
class CaseConfig:
    name: str
    csv_path: Path
    pickle_dir: Path
    checkpoint_out: Path
    n_iter: int


def log(message: str) -> None:
    print(message, flush=True)
    print(message, file=sys.stderr, flush=True)


def run_command(cmd: list[str], dry_run: bool) -> None:
    log(shlex.join(cmd))
    if dry_run:
        return

    child_env = os.environ.copy()
    child_env["WORKING_DIR"] = str(WORKING_DIR)
    child_env["WHOLE_PIPELINE_DIR"] = str(WHOLE_PIPELINE_DIR)
    child_env["MODEL_PREP_DIR"] = str(MODEL_PREP_DIR)
    child_env["CSV_DIR"] = str(CSV_DIR)
    child_env["OUTPUT_DIR"] = str(OUTPUT_DIR)
    child_env["GENERATE_PICKLES_SCRIPT"] = str(GENERATE_PICKLES_SCRIPT)
    child_env["TRAIN_SCRIPT"] = str(TRAIN_SCRIPT)
    child_env["DATASET_ROOT"] = str(DEFAULT_DATASET_ROOT)
    subprocess.run(cmd, check=True, cwd=str(WORKING_DIR), env=child_env)


def ensure_scripts_exist() -> None:
    for path in (GENERATE_PICKLES_SCRIPT, TRAIN_SCRIPT):
        if not path.exists():
            raise FileNotFoundError(f"Required script not found: {path}")


def build_case_configs(smoke_iter: int, user_iter: int, full_iter: int) -> dict[str, CaseConfig]:
    return {
        "smoke": CaseConfig(
            name="smoke",
            csv_path=CSV_DIR / "smoketest.csv",
            pickle_dir=OUTPUT_DIR / "pickles" / "smoke",
            checkpoint_out=OUTPUT_DIR / "checkpoints" / "smoke" / "checkpoint.pt",
            n_iter=smoke_iter,
        ),
        "user": CaseConfig(
            name="user",
            csv_path=CSV_DIR / "user_testing.csv",
            pickle_dir=OUTPUT_DIR / "pickles" / "user",
            checkpoint_out=OUTPUT_DIR / "checkpoints" / "user" / "checkpoint.pt",
            n_iter=user_iter,
        ),
        "full": CaseConfig(
            name="full",
            csv_path=CSV_DIR / "entire.csv",
            pickle_dir=OUTPUT_DIR / "pickles" / "full",
            checkpoint_out=OUTPUT_DIR / "checkpoints" / "full" / "checkpoint.pt",
            n_iter=full_iter,
        ),
    }


def run_case(
    case: CaseConfig,
    dataset_root: Path,
    dry_run: bool,
    generate_only: bool,
    train_only: bool,
    train_dry_run: bool,
) -> None:
    log(f"[main.py] Starting case: {case.name}")

    if not train_only:
        if not case.csv_path.exists():
            raise FileNotFoundError(f"Missing CSV for case {case.name}: {case.csv_path}")

        generate_cmd = [
            sys.executable,
            str(GENERATE_PICKLES_SCRIPT),
            "--dataset-root",
            str(dataset_root),
            "--input-csv",
            str(case.csv_path),
            "--output-dir",
            str(case.pickle_dir),
        ]
        log(f"[main.py] Generating pickles for case: {case.name}")
        run_command(generate_cmd, dry_run=dry_run)

    if not generate_only:
        train_cmd = [
            sys.executable,
            str(TRAIN_SCRIPT),
            "--dataset-root",
            str(dataset_root),
            "--pickle-dir",
            str(case.pickle_dir),
            "--model-conf-dir",
            str(case.pickle_dir),
            "--checkpoint-out",
            str(case.checkpoint_out),
            "--n-iter",
            str(case.n_iter),
        ]
        if train_dry_run:
            train_cmd.append("--dry-run")

        log(f"[main.py] Launching train.py for case: {case.name}")
        run_command(train_cmd, dry_run=dry_run)

    log(f"[main.py] Completed case: {case.name}")


def main() -> int:

    #parse args
    parser = argparse.ArgumentParser(description="Whole pipeline runner for smoke/user/full fine-tune flows")
    parser.add_argument(
        "--mode",
        choices=["smoke", "user", "full", "all"],
        default="all",
        help="Select one case or run all cases in sequence",
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--smoke-n-iter", type=int, default=300)
    parser.add_argument("--user-n-iter", type=int, default=3000)
    parser.add_argument("--full-n-iter", type=int, default=180000)
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--train-dry-run", action="store_true", help="Pass --dry-run to train.py")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only")
    args = parser.parse_args()

    selected: list[str]
    if args.mode == "all":
        selected = ["smoke", "user", "full"]
    else:
        selected = [args.mode]

    #checks
    if args.generate_only and args.train_only:
        raise ValueError("--generate-only and --train-only cannot be used together")

    ensure_scripts_exist()

    #configs
    cases = build_case_configs(
        smoke_iter=args.smoke_n_iter,
        user_iter=args.user_n_iter,
        full_iter=args.full_n_iter,
    )


    #execution
    log("[main.py] Starting execution")
    time_start = time.time()

    for name in selected:
        mode_time_start = time.time()
        run_case(
            case=cases[name],
            dataset_root=args.dataset_root.resolve(),
            dry_run=args.dry_run,
            generate_only=args.generate_only,
            train_only=args.train_only,
            train_dry_run=args.train_dry_run,
        )
        mode_time_end = time.time()
        log(f"[main.py] Completed case: {name} in {mode_time_end - mode_time_start} seconds")

    final_time_end = time.time()
    log(f"[main.py] Finished execution in {final_time_end - time_start} seconds")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
