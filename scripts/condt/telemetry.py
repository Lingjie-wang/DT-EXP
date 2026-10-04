"""Observation hooks for a frozen copy of the author's ConDT entry point.

These hooks seed a run and record existing updates/evaluations. They never select
batches, calculate losses, step optimizers, or run replacement evaluations.
"""

import importlib.metadata
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

_state = {}


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def initialize(variant):
    directory = Path(os.environ["CONDT_RECORD_DIR"])
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "runtime.json").exists():
        raise RuntimeError("Refusing to replace an existing ConDT attempt")
    seed = int(os.environ.get("CONDT_TRAIN_SEED", "0"))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    _state.update(
        directory=directory,
        start=time.monotonic(),
        pretrain_updates=0,
        main_updates=0,
        seed=seed,
    )
    packages = {}
    for name in (
        "torch",
        "numpy",
        "gym",
        "mujoco-py",
        "d4rl",
        "transformers",
        "ray",
        "pytorch-metric-learning",
        "wandb",
        "pandas",
        "h5py",
        "protobuf",
    ):
        packages[name] = importlib.metadata.version(name)
    write(
        directory / "runtime.json",
        dict(
            packages=packages,
            train_seed=seed,
            argv_variant=variant,
            cuda_available=torch.cuda.is_available(),
            gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            slurm_job_id=os.environ.get("SLURM_JOB_ID"),
            hostname=os.uname().nodename,
            torch_threads=torch.get_num_threads(),
        ),
    )
    status("training")


def status(value):
    write(
        _state["directory"] / "status.json",
        dict(
            status=value,
            main_updates=_state["main_updates"],
            pretrain_updates=_state["pretrain_updates"],
            wall_seconds=time.monotonic() - _state["start"],
        ),
    )


def ready(trainer):
    write(
        _state["directory"] / "optimizer_initial.json",
        dict(
            learning_rates=[group["lr"] for group in trainer.optimizer.param_groups],
            scheduler_base_lrs=trainer.scheduler.base_lrs,
            state_mean=trainer.data_class.state_mean.tolist(),
            state_std=trainer.data_class.state_std.tolist(),
            retained_trajectories=len(trainer.data_class.traj_lens),
            retained_transitions=int(sum(trainer.data_class.traj_lens)),
        ),
    )


def training_update(iteration, index, num_steps, losses, used_lrs, optimizer):
    phase = "pretrain" if iteration == 0 else "main"
    step = index + 1 if iteration == 0 else (iteration - 1) * num_steps + index + 1
    _state["pretrain_updates" if iteration == 0 else "main_updates"] = step
    if step % 100 and index != 0 and index != num_steps - 1:
        return
    record = dict(
        phase=phase,
        phase_updates=step,
        main_updates=_state["main_updates"],
        total_updates=_state["main_updates"] + _state["pretrain_updates"],
        learning_rates=used_lrs,
        next_learning_rates=[group["lr"] for group in optimizer.param_groups],
        elapsed_seconds=time.monotonic() - _state["start"],
        **{key: float(value) for key, value in losses.items()},
    )
    with open(_state["directory"] / "metrics.jsonl", "a") as stream:
        stream.write(json.dumps(record, allow_nan=False) + "\n")
    status("training")


def evaluation(iteration, num_steps, outputs, trainer, logs):
    directory = _state["directory"]
    main_updates = 0 if iteration == 0 else iteration * num_steps
    assert main_updates == _state["main_updates"]
    phase = (
        "main"
        if main_updates
        else ("post_pretrain" if _state["pretrain_updates"] else "initial")
    )
    evaluations = directory / "evaluations"
    evaluations.mkdir(exist_ok=True)
    destination = evaluations / f"eval_main_{main_updates:06d}.json"
    if destination.exists():
        raise RuntimeError(f"Refusing to replace evaluation {destination}")
    record = dict(
        phase=phase,
        main_updates=main_updates,
        pretrain_updates=_state["pretrain_updates"],
        outputs={
            str(seed): {
                "returns": [float(v) for v in data["returns"]],
                "lengths": [int(v) for v in data["lengths"]],
            }
            for seed, data in outputs.items()
        },
        epoch_metrics={key: float(value) for key, value in logs.items()},
        wall_seconds=time.monotonic() - _state["start"],
    )
    checkpoints = directory / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    checkpoint = checkpoints / f"main_{main_updates:06d}.pt"
    temporary = checkpoint.with_suffix(".tmp")
    torch.save(
        dict(
            model=trainer.model.state_dict(),
            optimizer=trainer.optimizer.state_dict(),
            scheduler=trainer.scheduler.state_dict(),
            python_rng=random.getstate(),
            numpy_rng=np.random.get_state(),
            torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            main_updates=main_updates,
            pretrain_updates=_state["pretrain_updates"],
            state_mean=trainer.data_class.state_mean,
            state_std=trainer.data_class.state_std,
        ),
        temporary,
    )
    temporary.replace(checkpoint)
    write(destination, record)
    status("training")
