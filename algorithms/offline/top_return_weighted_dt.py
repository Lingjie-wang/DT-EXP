"""From-scratch CORL DT with trajectory-level top-return MSE weighting.

Both arms use this entry point: top_weight=1 is the DT control; top_weight=2
reinforces the selected trajectories. Recorded RTGs and data sampling are
identical. No reference, preference pairs, extra forward pass, or RTG relabeling.
Steps count completed optimizer updates (unlike legacy zero-based DT logs).
"""

import hashlib
import json
import math
import os
import random
import subprocess
from dataclasses import asdict, dataclass
from typing import Tuple

import gym
import numpy as np
import pyrallis
import torch
import torch.nn.functional as F
import wandb
from dt import (
    DecisionTransformer,
    eval_rollout,
    SequenceDataset,
    set_seed,
    TrainConfig,
    wandb_init,
    wrap_env,
)
from torch.utils.data import DataLoader
from tqdm.auto import trange

@dataclass
class TopReturnTrainConfig(TrainConfig):
    top_fraction: float = 0.05
    top_weight: float = 2.0
    checkpoint_steps: Tuple[int, ...] = (50_000, 75_000, 100_000)
    log_every: int = 100


def select_top_trajectories(returns, fraction):
    """Exact ceil(fraction * N) trajectories; index breaks return ties stably."""
    returns = np.asarray(returns, dtype=np.float64)
    if not len(returns) or not np.isfinite(returns).all():
        raise ValueError("Expected a nonempty vector of finite trajectory returns")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("top_fraction must be in (0, 1]")
    count = max(1, math.ceil(fraction * len(returns)))
    indices = np.argsort(-returns, kind="stable")[:count]
    selected = np.zeros(len(returns), dtype=bool)
    selected[indices] = True
    return selected


class TopReturnDataset(SequenceDataset):
    def __init__(self, config):
        super().__init__(
            config.env_name, config.seq_len, config.reward_scale, config.reward_mode
        )
        self.trajectory_returns = np.array(
            [float(trajectory["returns"][0]) for trajectory in self.dataset]
        )
        self.top_selected = select_top_trajectories(
            self.trajectory_returns, config.top_fraction
        )

    def __iter__(self):
        # Reuse CORL's exact preparation, including its padding and recorded RTG.
        # Adding a label makes no additional random draw in the original sampler.
        while True:
            trajectory_id = np.random.choice(len(self.dataset), p=self.sample_prob)
            start = random.randint(0, len(self.dataset[trajectory_id]["rewards"]) - 1)
            sample = self._SequenceDataset__prepare_sample(trajectory_id, start)
            yield (*sample, np.float32(self.top_selected[trajectory_id]))


def weighted_action_loss(prediction, actions, mask, selected, top_weight):
    """Keep CORL's padded-token mean, with mean-one weights on valid tokens."""
    elementwise = F.mse_loss(prediction, actions.detach(), reduction="none")
    masked = elementwise * mask.unsqueeze(-1)
    ordinary_loss = masked.mean()
    weights = 1.0 + (top_weight - 1.0) * selected.to(mask.dtype)
    normalizer = (weights[:, None] * mask).sum() / mask.sum().clamp_min(1)
    # This reduction is exactly the original DT loss when top_weight is one.
    loss = (masked * (weights / normalizer)[:, None, None]).mean()
    return loss, ordinary_loss, elementwise, normalizer


def model_hash(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def batch_hash(batch):
    digest = hashlib.sha256()
    for tensor in batch:
        digest.update(tensor.contiguous().numpy().tobytes())
    return digest.hexdigest()


def rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


@pyrallis.wrap()
def train(config: TopReturnTrainConfig):
    if config.reward_mode != "delayed":
        raise ValueError("This experiment is defined for delayed rewards")
    if not np.isfinite(config.top_weight) or config.top_weight < 1.0:
        raise ValueError("top_weight must be finite and >= 1")
    if min(config.update_steps, config.eval_every, config.log_every) <= 0:
        raise ValueError("Training and logging intervals must be positive")
    if not config.checkpoints_path:
        raise ValueError("checkpoints_path is required for this experiment")
    set_seed(config.train_seed, deterministic_torch=config.deterministic_torch)
    wandb_init(asdict(config))
    dataset = TopReturnDataset(config)
    selection = {
        "fraction_requested": config.top_fraction,
        "num_trajectories": len(dataset.dataset),
        "num_selected": int(dataset.top_selected.sum()),
        "fraction_actual": float(dataset.top_selected.mean()),
        "selected_sampling_probability": float(
            dataset.sample_prob[dataset.top_selected].sum()
        ),
        "threshold_return": float(
            dataset.trajectory_returns[dataset.top_selected].min()
        ),
        "selected_trajectory_ids": np.flatnonzero(dataset.top_selected).tolist(),
        "selected_trajectory_returns": dataset.trajectory_returns[
            dataset.top_selected
        ].tolist(),
        "rule": "ceil(fraction * trajectory_count), stable descending return rank",
    }
    os.makedirs(config.checkpoints_path, exist_ok=True)
    with open(os.path.join(config.checkpoints_path, "selection.json"), "w") as file:
        json.dump(selection, file, indent=2)
    with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as file:
        pyrallis.dump(config, file)
    print(f"Trajectory selection: {json.dumps(selection)}", flush=True)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    wandb.run.summary["provenance/git_commit"] = revision
    wandb.run.summary["provenance/selection"] = selection
    wandb.run.summary["provenance/checkpoints_path"] = config.checkpoints_path
    wandb.log({f"dataset/{k}": v for k, v in dataset.stats.items()}, step=0)

    # Dedicated loader generator separates the training sample stream from
    # model/dropout RNG. Both arms receive the same worker seeds and minibatches.
    loader_generator = torch.Generator().manual_seed(config.train_seed)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=True,
        generator=loader_generator,
    )
    env = wrap_env(
        gym.make(config.env_name), dataset.state_mean, dataset.state_std,
        config.reward_scale,
    )
    # Explicit reset guarantees identical initialization even if logging or
    # environment construction consumed randomness.
    set_seed(config.train_seed, deterministic_torch=config.deterministic_torch)
    model = DecisionTransformer(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        embedding_dim=config.embedding_dim,
        seq_len=config.seq_len,
        episode_len=config.episode_len,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        attention_dropout=config.attention_dropout,
        residual_dropout=config.residual_dropout,
        embedding_dropout=config.embedding_dropout,
        max_action=config.max_action,
    ).to(config.device)
    initial_hash = model_hash(model)
    wandb.run.summary["provenance/initial_model_sha256"] = initial_hash
    print(f"Initial model SHA256: {initial_hash}", flush=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, betas=config.betas,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda steps: min((steps + 1) / config.warmup_steps, 1)
    )
    best_scores = {target: -float("inf") for target in config.target_returns}

    def save_checkpoint(step, filename):
        checkpoint = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "state_mean": dataset.state_mean,
            "state_std": dataset.state_std,
            "config": asdict(config),
            "next_step": step,
            "completed_updates": step,
            "rng_state": rng_state(),
            "loader_generator_state": loader_generator.get_state(),
            "selection": selection,
            "git_commit": revision,
            "initial_model_sha256": initial_hash,
            "wandb_run_id": wandb.run.id,
            "best_scores": best_scores,
            "resume_note": "Worker prefetch queues are not serialized; resuming "
            "restores model/optimizer but not an exact continuation of data batches.",
        }
        destination = os.path.join(config.checkpoints_path, filename)
        torch.save(checkpoint, destination + ".tmp")
        os.replace(destination + ".tmp", destination)

    save_checkpoint(0, "step000000.pt")
    iterator = iter(loader)
    for step in trange(1, config.update_steps + 1, desc="Training"):
        batch = next(iterator)
        if step == 1:
            first_hash = batch_hash(batch)
            wandb.run.summary["provenance/first_batch_sha256"] = first_hash
            print(f"First batch SHA256: {first_hash}", flush=True)
        states, actions, returns, times, mask, selected = [
            item.to(config.device) for item in batch
        ]
        prediction = model(states, actions, returns, times, ~mask.bool())
        loss, dt_loss, elementwise, normalizer = weighted_action_loss(
            prediction, actions, mask, selected, config.top_weight
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite training loss at update {step}")
        optimizer.zero_grad()
        loss.backward()
        if config.clip_grad is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.clip_grad, error_if_nonfinite=True
            )
        else:
            grad_norm = torch.zeros((), device=config.device)
        optimizer.step()
        scheduler.step()
        if step == 1 or step % config.log_every == 0:
            top_mask = mask * selected[:, None]
            other_mask = mask * (1 - selected[:, None])
            action_error = elementwise.detach().mean(dim=-1)
            metrics = {
                "train_loss": dt_loss.item(),
                "train/weighted_loss": loss.item(),
                "train/top_action_mse": (
                    (action_error * top_mask).sum() / top_mask.sum().clamp_min(1)
                ).item(),
                "train/other_action_mse": (
                    (action_error * other_mask).sum() / other_mask.sum().clamp_min(1)
                ).item(),
                "train/top_token_fraction": (top_mask.sum() / mask.sum()).item(),
                "train/weight_normalizer": normalizer.item(),
                "train/grad_norm": grad_norm.item(),
                "learning_rate": scheduler.get_last_lr()[0],
            }
            wandb.log(metrics, step=step)
            if step == 1 or step % 5000 == 0:
                print(f"Update {step}: {json.dumps(metrics)}", flush=True)
        if step % config.eval_every == 0 or step == config.update_steps:
            saved_rng = rng_state()
            model.eval()
            metrics = {}
            for target in config.target_returns:
                env.seed(config.eval_seed)
                episode_returns, lengths = [], []
                for _ in range(config.eval_episodes):
                    score, length = eval_rollout(
                        model, env, target * config.reward_scale,
                        config.device, config.reward_mode,
                    )
                    episode_returns.append(score / config.reward_scale)
                    lengths.append(length)
                normalized = env.get_normalized_score(np.array(episode_returns)) * 100
                mean_score = float(np.mean(normalized))
                metrics.update({
                    f"eval/{target}_normalized_score_mean": mean_score,
                    f"eval/{target}_normalized_score_std": float(np.std(normalized)),
                    f"eval/{target}_return_mean": float(np.mean(episode_returns)),
                    f"eval/{target}_return_std": float(np.std(episode_returns)),
                    f"eval/{target}_length_mean": float(np.mean(lengths)),
                })
                if mean_score > best_scores[target]:
                    best_scores[target] = mean_score
                    wandb.run.summary[f"best/{target}_score"] = mean_score
                    wandb.run.summary[f"best/{target}_step"] = step
                    save_checkpoint(step, f"best_target_{int(target)}.pt")
            wandb.log(metrics, step=step)
            print(f"Evaluation {step}: {json.dumps(metrics)}", flush=True)
            restore_rng(saved_rng)
            model.train()
        if step in config.checkpoint_steps or step == config.update_steps:
            save_checkpoint(step, f"step{step:06d}.pt")
    env.close()
    wandb.finish()


if __name__ == "__main__":
    train()
