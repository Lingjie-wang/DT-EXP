"""Ordinary delayed-reward DT parent for corrected staged experiments."""

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Tuple

import gym
import numpy as np
import pyrallis
import torch
import torch.nn.functional as F
import wandb
from delayed_staged import (
    audit_batch_rtg,
    audit_delayed_dataset,
    TerminalReward,
)
from dt import (
    DecisionTransformer,
    eval_rollout,
    SequenceDataset,
    set_seed,
    TrainConfig,
    wrap_env,
)
from state_only_preference import (
    mine_pairs,
    preference_batch,
    single_sided_loss,
    validate_resume_config,
)
from top_return_weighted_dt import model_hash, restore_rng, rng_state
from torch.utils.data import DataLoader

@dataclass
class DelayedParentConfig(TrainConfig):
    env_name: str = "halfcheetah-medium-replay-v2"
    reward_mode: str = "delayed"
    train_seed: int = 0
    batch_size: int = 4096
    learning_rate: float = 0.0008
    target_returns: Tuple[float, ...] = (12000.0,)
    eval_every: int = 10000
    output_dir: str = ""
    preference_batch_size: int = 256
    preference_weight: float = 0.0
    variant: str = "parent_dt"
    validation_mode: bool = False
    preference_margin: float = 0.05
    state_max_rmse: float = 0.5
    log_every: int = 100
    checkpoint_every: int = 5000
    wandb_mode: str = "online"
    wandb_entity: str = "2820402607-shandong-university"
    group: str = "DelayedStage-HCMR-5090-20261005"
    name: str = "DelayedStage-parent-dt"
    resume_checkpoint: str = ""
    attention_backend: str = "math"


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@pyrallis.wrap()
def train(config: DelayedParentConfig):
    if not config.output_dir:
        raise ValueError("Choose a NEW output_dir; historical results cannot be reused")
    if (config.env_name != "halfcheetah-medium-replay-v2"
            or config.reward_mode != "delayed"):
        raise ValueError("This version is defined for delayed HCMR only")
    if (config.variant != "parent_dt" or config.preference_weight != 0
            or config.resume_checkpoint):
        raise ValueError("Parent must be ordinary delayed DT from random initialization")
    if not config.validation_mode and (
            config.update_steps != 100000 or config.eval_every != 10000
            or config.eval_episodes != 100 or config.learning_rate != 0.0008
            or config.batch_size != 4096 or config.preference_batch_size != 256
            or config.train_seed != 0 or config.warmup_steps != 10000):
        raise ValueError("Formal parent protocol: 100k ordinary delayed DT, seed 0")
    if tuple(config.target_returns) != (12000.0,):
        raise ValueError("C uses the predeclared RTG 12000 only")
    if min(config.update_steps, config.eval_every, config.eval_episodes,
           config.log_every, config.checkpoint_every) <= 0:
        raise ValueError("Training, evaluation and saving intervals must be positive")
    if config.attention_backend not in {"auto", "math"}:
        raise ValueError("attention_backend must be auto or math")
    if config.attention_backend == "math":
        # Public PyTorch backend controls; no modifications to the DT architecture.
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    root = Path(config.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / "checkpoints").mkdir()
    (root / "evaluations").mkdir()
    write_json(root / "config.json", asdict(config))
    write_json(root / "status.json", {"state": "preparing", "completed_updates": 0})
    run = None
    env = None
    step = 0
    completed_updates = 0
    try:
        set_seed(config.train_seed, deterministic_torch=config.deterministic_torch)
        dataset = SequenceDataset(
            config.env_name, config.seq_len, config.reward_scale, config.reward_mode
        )
        delayed_audit = audit_delayed_dataset(dataset)
        write_json(root / "delayed_reward_audit.json", delayed_audit)
        pairs, distances, pair_stats = mine_pairs(
            dataset.dataset, dataset.state_mean, dataset.state_std, config.state_max_rmse
        )
        if len(pairs) != 6768:
            raise ValueError("Delayed HCMR pair pool must contain 6768 pairs")
        np.savez_compressed(root / "pairs.npz", pairs=pairs, state_rmse=distances)
        write_json(root / "pair_stats.json", pair_stats)
        print("PAIR_STATS " + json.dumps(pair_stats), flush=True)
        env = wrap_env(
            TerminalReward(gym.make(config.env_name)),
            dataset.state_mean, dataset.state_std,
            config.reward_scale,
        )
        loader_generator = torch.Generator().manual_seed(config.train_seed)
        loader = DataLoader(
            dataset, batch_size=config.batch_size, num_workers=config.num_workers,
            pin_memory=True, generator=loader_generator,
        )
        pair_rng = np.random.RandomState(config.train_seed + 10000)
        set_seed(config.train_seed, deterministic_torch=config.deterministic_torch)
        model = DecisionTransformer(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0], embedding_dim=config.embedding_dim,
            seq_len=config.seq_len, episode_len=config.episode_len,
            num_layers=config.num_layers, num_heads=config.num_heads,
            attention_dropout=config.attention_dropout,
            residual_dropout=config.residual_dropout,
            embedding_dropout=config.embedding_dropout, max_action=config.max_action,
        ).to(config.device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate, betas=config.betas,
            weight_decay=config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda updates: min((updates + 1) / config.warmup_steps, 1)
        )
        resumed = None
        if config.resume_checkpoint:
            resumed = torch.load(config.resume_checkpoint, map_location="cpu",
                                 weights_only=False)
            completed_updates = resumed["completed_updates"]
            validate_resume_config(asdict(config), resumed["config"], completed_updates)
            if not (np.array_equal(dataset.state_mean, resumed["state_mean"])
                    and np.array_equal(dataset.state_std, resumed["state_std"])):
                raise ValueError("Resume dataset normalization differs")
            if resumed["provenance"]["pairs_sha256"] != file_hash(root / "pairs.npz"):
                raise ValueError("Resume pair pool differs")
            model.load_state_dict(resumed["model_state"])
            optimizer.load_state_dict(resumed["optimizer_state"])
            scheduler.load_state_dict(resumed["scheduler_state"])
            pair_rng.set_state(resumed["pair_rng_state"])
            loader_generator.set_state(resumed["loader_generator_state"])
            if not all(torch.isfinite(p).all() for p in model.parameters()):
                raise ValueError("Resume model contains nonfinite parameters")
        project = Path(__file__).resolve().parents[2]
        source_files = [Path(__file__),
                        Path(__file__).with_name("delayed_staged.py"),
                        Path(__file__).with_name("late_preference.py"),
                        Path(__file__).with_name("state_only_preference.py"),
                        Path(__file__).with_name("dt.py"),
                        Path(__file__).with_name("top_return_weighted_dt.py")]
        provenance = {
            "git_commit": subprocess.check_output(
                ["git", "-C", str(project), "rev-parse", "HEAD"], text=True
            ).strip(),
            "source_sha256": {str(p.relative_to(project)): file_hash(p)
                              for p in source_files},
            "dataset_sha256": file_hash(env.dataset_filepath),
            "delayed_reward_audit": delayed_audit,
            "pairs_sha256": file_hash(root / "pairs.npz"),
            "initial_model_sha256": model_hash(model),
            "python": platform.python_version(),
            "packages": {name: importlib.metadata.version(name) for name in
                         ("torch", "numpy", "gym", "d4rl", "mujoco-py", "wandb")},
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            "cuda": torch.version.cuda,
            "attention_backend": config.attention_backend,
        }
        if resumed is not None:
            provenance["recovery"] = {
                "checkpoint_sha256": file_hash(config.resume_checkpoint),
                "completed_updates": completed_updates,
                "parent_run_id": resumed["wandb_run_id"],
                "parent_git_commit": resumed["provenance"]["git_commit"],
                "worker_sampling": "New worker streams; prefetch state was not saved. "
                                   "Not an exact replay of the interrupted run.",
            }
        write_json(root / "provenance.json", provenance)
        run = wandb.init(
            entity=config.wandb_entity, project=config.project, group=config.group,
            name=config.name, config={
                **asdict(config), "provenance": provenance,
                "pair_stats": pair_stats, "dataset": dataset.stats,
            },
            dir=str(root), mode=config.wandb_mode,
            settings=wandb.Settings(disable_git=True, save_code=False),
        )
        write_json(root / "wandb_run.json", {"id": run.id, "url": run.url})
        print("WANDB_RUN " + str(run.url), flush=True)
        if config.wandb_mode != "disabled":
            artifact = wandb.Artifact(
                f"delayed-stage-pairs-{run.id}", type="preference-pairs")
            for filename in ("pairs.npz", "pair_stats.json", "provenance.json",
                             "delayed_reward_audit.json"):
                artifact.add_file(str(root / filename))
            run.log_artifact(artifact)
        start_time = time.monotonic()

        def checkpoint(completed):
            saved = {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "completed_updates": completed, "config": asdict(config),
                "state_mean": dataset.state_mean, "state_std": dataset.state_std,
                "rng_state": rng_state(), "pair_rng_state": pair_rng.get_state(),
                "loader_generator_state": loader_generator.get_state(),
                "provenance": provenance, "wandb_run_id": run.id,
                "resume_note": "Worker prefetch queues are not serialized. "
                               "Exact batch continuation / auto-resume is not claimed.",
            }
            path = root / "checkpoints" / f"step{completed:06d}.pt"
            torch.save(saved, str(path) + ".tmp")
            os.replace(str(path) + ".tmp", path)

        if resumed is not None:
            restore_rng(resumed["rng_state"])
        checkpoint(completed_updates)
        iterator = iter(loader)
        evaluations = []
        with (root / "metrics.jsonl").open("a", buffering=1) as metrics_file:
            for step in range(completed_updates + 1, config.update_steps + 1):
                states, actions, returns, times, mask = [
                    item.to(config.device) for item in next(iterator)
                ]
                before_forward_rng = rng_state()
                prediction = model(states, actions, returns, times, ~mask.bool())
                dt_loss = (F.mse_loss(prediction, actions.detach(), reduction="none")
                           * mask.unsqueeze(-1)).mean()
                indices = pair_rng.randint(len(pairs), size=config.preference_batch_size)
                preference = preference_batch(dataset, pairs, indices, 12000.0)
                ps, pa, pr, pt, pm, target_index, negative = [
                    torch.from_numpy(item).to(config.device) for item in preference
                ]
                if step == 1:
                    write_json(root / "batch_rtg_verified.json", audit_batch_rtg(
                        returns, mask, pr, pm, 12000.0 * config.reward_scale))
                output = model(ps, pa, pr, pt, ~pm.bool())
                row = torch.arange(len(indices), device=config.device)
                pref_loss, dpos, dneg, active = single_sided_loss(
                    output[row, target_index], pa[row, target_index], negative,
                    config.preference_margin,
                )
                loss = dt_loss + config.preference_weight * pref_loss
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite loss at update {step}")
                optimizer.zero_grad()
                try:
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.clip_grad, error_if_nonfinite=True
                    )
                except RuntimeError:
                    torch.save({
                        "step": step, "model_state": model.state_dict(),
                        "batch": [states, actions, returns, times, mask],
                        "auxiliary": [ps, pa, pr, pt, pm, target_index, negative],
                        "rng_state": before_forward_rng, "config": asdict(config),
                    }, root / "failure_batch.pt")
                    # Gradients have not been applied; retain the last valid state.
                    checkpoint(completed_updates)
                    raise
                optimizer.step()
                scheduler.step()
                completed_updates = step
                if step == 1 or step % config.log_every == 0:
                    metrics = {
                        "train/dt_loss": dt_loss.item(),
                        "train/preference_loss": pref_loss.item(),
                        "train/total_loss": loss.item(),
                        "train/positive_mse": dpos.mean().item(),
                        "train/negative_mse": dneg.mean().item(),
                        "train/active_fraction": active.float().mean().item(),
                        "train/grad_norm": grad_norm.item(),
                        "train/next_learning_rate": scheduler.get_last_lr()[0],
                        "time/elapsed_seconds": time.monotonic() - start_time,
                    }
                    run.log(metrics, step=step, commit=False)
                    metrics_file.write(json.dumps({"step": step, **metrics}) + "\n")
                    write_json(root / "status.json", {
                        "state": "training", "completed_updates": step,
                        "pid": os.getpid(), "wandb_url": run.url, **metrics,
                    })
                    print(f"UPDATE {step} " + json.dumps(metrics), flush=True)
                if step % config.eval_every == 0 or step == config.update_steps:
                    saved_rng = rng_state()
                    model.eval()
                    env.seed(config.eval_seed)
                    episode_returns, lengths = [], []
                    for _ in range(config.eval_episodes):
                        score, length = eval_rollout(
                            model, env, 12000.0 * config.reward_scale,
                            config.device, config.reward_mode,
                        )
                        episode_returns.append(score / config.reward_scale)
                        lengths.append(length)
                    scores = env.get_normalized_score(np.array(episode_returns)) * 100
                    evaluation = {
                        "step": step, "return_mean": float(np.mean(episode_returns)),
                        "normalized_score_mean": float(np.mean(scores)),
                        "normalized_score_std": float(np.std(scores)),
                        "returns": episode_returns, "lengths": lengths,
                    }
                    evaluations.append(evaluation)
                    write_json(root / "evaluations" / f"step{step:06d}.json", evaluation)
                    metrics = {
                        "eval/12000_normalized_score_mean": float(np.mean(scores)),
                        "eval/12000_normalized_score_std": float(np.std(scores)),
                        "eval/12000_return_mean": float(np.mean(episode_returns)),
                        "eval/12000_return_std": float(np.std(episode_returns)),
                    }
                    run.log(metrics, step=step, commit=False)
                    run.log({f"episodes/step_{step:06d}": wandb.Table(
                        columns=["episode", "return", "normalized_score", "length"],
                        data=[[i, float(r), float(s), float(n)] for i, (r, s, n)
                              in enumerate(zip(episode_returns, scores, lengths))],
                    )}, step=step, commit=False)
                    metrics_file.write(json.dumps({"step": step, **metrics}) + "\n")
                    print(f"EVAL {step} " + json.dumps(metrics), flush=True)
                    restore_rng(saved_rng)
                    model.train()
                if step % config.checkpoint_every == 0 or step == config.update_steps:
                    checkpoint(step)
                # Flush once per completed update. Separate committed log() calls
                # at the same step would discard evaluation values and tables.
                if (step == 1 or step % config.log_every == 0
                        or step % config.eval_every == 0 or step == config.update_steps):
                    run.log({}, step=step)
        summary = {
            "completed_updates": step,
            "last_score": evaluations[-1]["normalized_score_mean"],
            "last_three_eval_mean": float(np.mean([
                e["normalized_score_mean"] for e in evaluations[-3:]
            ])),
            "elapsed_seconds": time.monotonic() - start_time,
        }
        late = [e["normalized_score_mean"] for e in evaluations
                if e["step"] in (60000, 80000, 100000)]
        if len(late) == 3:
            summary["late_60_80_100k_mean"] = float(np.mean(late))
        write_json(root / "summary.json", summary)
        run.summary.update(summary)
        run.finish()
        write_json(root / "status.json", {"state": "completed", **summary})
    except BaseException as error:
        failure = {
            "state": "failed", "last_loop_step": step,
            "completed_updates": completed_updates,
            "error_type": type(error).__name__,
            "error": str(error),
        }
        write_json(root / "status.json", failure)
        if run is not None:
            run.summary["failure"] = failure
            run.finish(exit_code=1)
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    train()
