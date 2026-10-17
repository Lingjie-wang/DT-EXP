"""CORL CQL with explicit terminal-return data; no changes to CQL updates."""

import argparse
import importlib.metadata
import math
import random
import time
from dataclasses import asdict
from pathlib import Path

# D4RL's top-level import can skip this after optional Adroit/mjrl fails.
import d4rl.locomotion  # noqa: F401 -- register official AntMaze environments
import gym
import numpy as np
import torch
import yaml

from algorithms.offline.cql import (
    compute_mean_std,
    eval_actor,
    modify_reward,
    normalize_states,
    ReplayBuffer,
    TrainConfig,
    wrap_env,
)
from scripts.cql_delayed.common import append, digest, read, verify, write
from scripts.cql_delayed.trainer import build_trainer

def run(root, arm, smoke=False, device=None):
    p = verify(root)
    spec = p["runs"][arm]
    work = root / arm / ("preflight" if smoke else "training")
    work.mkdir(parents=True, exist_ok=False)
    status = work / "status.json"
    write(status, dict(status="loading", completed_updates=0))
    try:
        for name in ["dataset.npz", "audit.json"]:
            if digest(root / arm / name) != spec["data_sha256"][name]:
                raise RuntimeError(f"Prepared dataset changed: {arm}/{name}")
        config_file = root / "source" / spec["config"]
        config = TrainConfig(**yaml.safe_load(config_file.read_text()))
        config.name = spec["wandb_name"]
        if device:
            config.device = device
        if smoke:
            config.max_timesteps, config.eval_freq, config.n_episodes = 100, 100, 1
        if config.load_model:
            raise ValueError("Expected training from scratch")
        if arm == "shapley" and not p.get("shapley_gate", {}).get("passed"):
            raise ValueError("Held-out prediction gate must pass before CQL")
        env = gym.make(config.env)
        with np.load(root / arm / "dataset.npz") as stored:
            dataset = {key: stored[key] for key in stored.files}
        if config.normalize_reward:
            modify_reward(dataset, config.env, reward_scale=config.reward_scale,
                          reward_bias=config.reward_bias)
        state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)
        if not config.normalize:
            state_mean, state_std = 0, 1
        for key in ["observations", "next_observations"]:
            dataset[key] = normalize_states(dataset[key], state_mean, state_std)
        env = wrap_env(env, state_mean=state_mean, state_std=state_std)
        replay = ReplayBuffer(
            env.observation_space.shape[0], env.action_space.shape[0],
            config.buffer_size, config.device,
        )
        replay.load_d4rl_dataset(dataset)
        del dataset
        trainer = build_trainer(config, env)
        write(work / "config.json", {
            k: v if not isinstance(v, float) or math.isfinite(v) else str(v)
            for k, v in asdict(config).items()
        })
        write(work / "runtime.json", dict(
            device=config.device,
            gpu=(torch.cuda.get_device_name()
                 if config.device.startswith("cuda") else None),
            versions={name: importlib.metadata.version(name) for name in
                      ["torch", "numpy", "gym", "d4rl", "pyrallis", "h5py"]},
        ))
        started = time.monotonic()
        for step in range(1, int(config.max_timesteps) + 1):
            batch = [b.to(config.device) for b in replay.sample(config.batch_size)]
            metrics = trainer.train(batch)
            if not all(math.isfinite(float(v)) for v in metrics.values()):
                raise FloatingPointError(f"Non-finite CQL metric at step {step}")
            if step % 100 == 0:
                row = dict(completed_updates=step,
                           elapsed_seconds=time.monotonic() - started,
                           **metrics)
                append(work / "metrics.jsonl", row)
                write(status, dict(status="training", completed_updates=step))
                print(f"CQL updates={step} elapsed={row['elapsed_seconds']:.1f}",
                      flush=True)
            if step % config.eval_freq == 0:
                scores = eval_actor(env, trainer.actor, config.device,
                                    config.n_episodes, config.seed)
                score = float(env.get_normalized_score(scores.mean()) * 100)
                row = dict(completed_updates=step, raw_return_mean=float(scores.mean()),
                           normalized_score=score, episodes=scores.tolist())
                write(work / f"eval_{step:07d}.json", row)
                print(f"CQL EVAL updates={step} score={score:.6f}", flush=True)
            if step % 100000 == 0 or step == config.max_timesteps:
                torch.save(dict(
                    trainer=trainer.state_dict(),
                    state_mean=state_mean, state_std=state_std,
                    completed_updates=step, numpy_rng=np.random.get_state(),
                    python_rng=random.getstate(), torch_rng=torch.get_rng_state(),
                    cuda_rng=(torch.cuda.get_rng_state_all()
                              if torch.cuda.is_available() else []),
                ), work / f"checkpoint_{step:07d}.pt")
        env.close()
        write(status, dict(status="completed", completed_updates=step,
                           final_normalized_score=score))
    except BaseException as error:
        previous = read(status)
        write(status, dict(status="failed",
                           completed_updates=previous["completed_updates"],
                           error=repr(error)))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--arm", choices=["delayed", "shapley", "uniform", "dense"],
                        required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device")
    args = parser.parse_args()
    run(args.root.resolve(), args.arm, args.smoke, args.device)


if __name__ == "__main__":
    main()
