#!/usr/bin/env python3
"""Run original and delayed-reward DT across every checked-in DT config."""

import argparse
import concurrent.futures
import os
import queue
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs" / "offline" / "dt"
DT_PROGRAM = ROOT / "algorithms" / "offline" / "dt.py"
VARIANT_NAMES = {
    "original": "DT",
    "delayed": "DelayedRewardDT",
}


@dataclass(frozen=True)
class Job:
    config_path: Path
    env_name: str
    reward_mode: str
    seed: int

    @property
    def key(self) -> str:
        return f"{self.reward_mode}__{self.env_name}__seed{self.seed}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Launch the complete DT vs. delayed-reward DT experiment matrix. "
            "By default this is 30 datasets x 2 variants x 3 seeds = 180 runs."
        )
    )
    parser.add_argument("--project", default="CORL-DDR")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=tuple(VARIANT_NAMES),
        default=list(VARIANT_NAMES),
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        help="Optional exact D4RL environment names; omit to run all DT configs.",
    )
    parser.add_argument(
        "--gpus",
        nargs="+",
        help="GPU IDs, with at most one concurrent run per ID (for example: 0 1).",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        help="Concurrent runs. Defaults to the number of --gpus, otherwise 1.",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--wandb-mode", choices=("online", "offline", "disabled"), default="online"
    )
    parser.add_argument(
        "--run-dir", type=Path, default=ROOT / ".dt_runs", help="Logs and markers."
    )
    parser.add_argument(
        "--rerun-completed",
        action="store_true",
        help="Run jobs again even when a successful completion marker exists.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue scheduling jobs after a run fails.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print commands without starting runs."
    )
    parser.add_argument(
        "--extra-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Arguments appended to every dt.py command (must be the last option).",
    )
    return parser.parse_args()


def load_configs(dataset_filter: Optional[Sequence[str]]) -> List[Tuple[Path, str]]:
    configs = []
    requested = set(dataset_filter or [])
    found = set()
    for config_path in sorted(CONFIG_ROOT.rglob("*.yaml")):
        with config_path.open() as config_file:
            config = yaml.safe_load(config_file)
        env_name = config["env_name"]
        if requested and env_name not in requested:
            continue
        configs.append((config_path, env_name))
        found.add(env_name)

    missing = requested - found
    if missing:
        raise ValueError(f"Unknown dataset(s): {', '.join(sorted(missing))}")
    if not configs:
        raise ValueError(f"No DT configs found under {CONFIG_ROOT}")
    return configs


def make_jobs(args: argparse.Namespace) -> List[Job]:
    configs = load_configs(args.datasets)
    return [
        Job(config_path, env_name, reward_mode, seed)
        for config_path, env_name in configs
        for reward_mode in args.variants
        for seed in args.seeds
    ]


def build_command(job: Job, args: argparse.Namespace) -> List[str]:
    run_name = f"{VARIANT_NAMES[job.reward_mode]}-seed{job.seed}"
    group = f"DT-{job.reward_mode}-{job.env_name}-{len(args.seeds)}seed-v0"
    return [
        args.python,
        str(DT_PROGRAM),
        "--config_path",
        str(job.config_path),
        "--project",
        args.project,
        "--group",
        group,
        "--name",
        run_name,
        "--reward_mode",
        job.reward_mode,
        "--train_seed",
        str(job.seed),
        *args.extra_args,
    ]


def resolve_slots(args: argparse.Namespace) -> List[Optional[str]]:
    if args.max_parallel is not None and args.max_parallel < 1:
        raise ValueError("--max-parallel must be at least 1")
    if args.gpus:
        max_parallel = args.max_parallel or len(args.gpus)
        if max_parallel > len(args.gpus):
            raise ValueError("--max-parallel cannot exceed the number of --gpus")
        return list(args.gpus[:max_parallel])
    return [None] * (args.max_parallel or 1)


def run_job(
    job: Job,
    args: argparse.Namespace,
    available_slots: "queue.Queue[Optional[str]]",
) -> Tuple[Job, int]:
    gpu = available_slots.get()
    try:
        args.run_dir.mkdir(parents=True, exist_ok=True)
        log_path = args.run_dir / f"{job.key}.log"
        done_path = args.run_dir / f"{job.key}.done"
        command = build_command(job, args)
        env = os.environ.copy()
        env["WANDB_MODE"] = args.wandb_mode
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = gpu

        slot_label = f"GPU {gpu}" if gpu is not None else "default device"
        print(f"[start] {job.key} on {slot_label}; log={log_path}", flush=True)
        with log_path.open("w") as log_file:
            log_file.write(f"command: {shlex.join(command)}\n\n")
            log_file.flush()
            result = subprocess.run(
                command,
                cwd=ROOT,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=False,
            )

        if result.returncode == 0:
            done_path.write_text("success\n")
            print(f"[done]  {job.key}", flush=True)
        else:
            print(
                f"[fail]  {job.key} exited {result.returncode}; see {log_path}",
                flush=True,
            )
        return job, result.returncode
    finally:
        available_slots.put(gpu)


def main() -> int:
    args = parse_args()
    jobs = make_jobs(args)
    slots = resolve_slots(args)
    pending_jobs = [
        job
        for job in jobs
        if args.rerun_completed or not (args.run_dir / f"{job.key}.done").exists()
    ]

    print(
        f"Selected {len(jobs)} runs; {len(pending_jobs)} pending; "
        f"parallelism={len(slots)}."
    )
    if args.dry_run:
        for job in pending_jobs:
            print(shlex.join(build_command(job, args)))
        return 0
    if not pending_jobs:
        return 0

    available_slots: "queue.Queue[Optional[str]]" = queue.Queue()
    for slot in slots:
        available_slots.put(slot)

    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(slots)) as executor:
        future_to_job = {
            executor.submit(run_job, job, args, available_slots): job
            for job in pending_jobs
        }
        for future in concurrent.futures.as_completed(future_to_job):
            job, returncode = future.result()
            if returncode:
                failures.append(job)
                if not args.keep_going:
                    for pending in future_to_job:
                        pending.cancel()
                    break

    if failures:
        print(f"{len(failures)} run(s) failed.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
