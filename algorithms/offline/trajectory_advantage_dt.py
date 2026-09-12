"""Trajectory-relative advantage weighted Decision Transformer.

This experiment keeps CORL DT's delayed-reward data, architecture, optimizer,
and evaluation protocol.  A separate supervised state/time return baseline is
fit from the offline trajectories before continuation from a frozen DT 50k
checkpoint.  The baseline is then frozen.  For every valid DT token, its
behavior-cloning loss is weighted by the clipped, self-normalized residual

    R(tau) - E_dataset[R(tau) | s_t, t].

The method therefore uses all transitions instead of a small, static set of
trajectory pairs.  It contains no temporal-difference target, Bellman backup,
or reward redistribution: every value target is the observed trajectory's
terminal return.
"""

import os
import random
from dataclasses import asdict, dataclass
from typing import Dict, Tuple

import gym
import numpy as np
import pyrallis
import torch
import torch.nn as nn
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
class TrajectoryAdvantageTrainConfig(TrainConfig):
    """Configuration for frozen trajectory-return baselines and weighted DT."""

    pretrained_checkpoint_path: str = ""
    value_hidden_dim: int = 256
    value_learning_rate: float = 3e-4
    value_weight_decay: float = 1e-4
    value_pretrain_steps: int = 5_000
    value_batch_size: int = 8_192
    advantage_temperature: float = 0.5
    advantage_weight_min: float = 0.25
    advantage_weight_max: float = 4.0


class StateTimeReturnBaseline(nn.Module):
    """Predict a normalized terminal trajectory return from state and time."""

    def __init__(self, state_dim: int, hidden_dim: int, episode_len: int):
        super().__init__()
        self.episode_len = float(episode_len)
        self.network = nn.Sequential(
            nn.Linear(state_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, states: torch.Tensor, time_steps: torch.Tensor
    ) -> torch.Tensor:
        time_feature = time_steps.to(dtype=states.dtype).unsqueeze(-1)
        time_feature = time_feature / self.episode_len
        return self.network(torch.cat([states, time_feature], dim=-1)).squeeze(-1)


def collect_transition_return_targets(
    dataset: SequenceDataset,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flatten state/time/terminal-return supervision for the frozen baseline."""

    state_parts = []
    time_parts = []
    return_parts = []
    for trajectory in dataset.dataset:
        length = trajectory["observations"].shape[0]
        states = (trajectory["observations"] - dataset.state_mean) / dataset.state_std
        # With delayed reward, returns[0] is the terminal trajectory reward.  We
        # use that observed trajectory outcome at every transition, not a target
        # bootstrapped from other time steps.
        terminal_return = trajectory["returns"][0] * dataset.reward_scale
        state_parts.append(states.astype(np.float32))
        time_parts.append(np.arange(length, dtype=np.int64))
        return_parts.append(np.full(length, terminal_return, dtype=np.float32))

    return (
        np.concatenate(state_parts, axis=0),
        np.concatenate(time_parts, axis=0),
        np.concatenate(return_parts, axis=0),
    )


def save_rng_state() -> Dict[str, object]:
    """Preserve the resumed DT's data and dropout RNG stream exactly."""

    state: Dict[str, object] = {
        "numpy": np.random.get_state(),
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Dict[str, object]) -> None:
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def pretrain_return_baseline(
    config: TrajectoryAdvantageTrainConfig,
    dataset: SequenceDataset,
    state_dim: int,
) -> Tuple[StateTimeReturnBaseline, float, float, Dict[str, float]]:
    """Fit the supervised baseline using an isolated RNG stream, then freeze it."""

    states_np, time_steps_np, target_returns_np = collect_transition_return_targets(
        dataset
    )
    target_mean = float(target_returns_np.mean())
    target_std = float(target_returns_np.std() + 1e-6)
    normalized_targets_np = (target_returns_np - target_mean) / target_std

    states = torch.as_tensor(states_np, dtype=torch.float32, device=config.device)
    time_steps = torch.as_tensor(time_steps_np, dtype=torch.long, device=config.device)
    targets = torch.as_tensor(
        normalized_targets_np, dtype=torch.float32, device=config.device
    )
    baseline = StateTimeReturnBaseline(
        state_dim=state_dim,
        hidden_dim=config.value_hidden_dim,
        episode_len=config.episode_len,
    ).to(config.device)
    optimizer = torch.optim.AdamW(
        baseline.parameters(),
        lr=config.value_learning_rate,
        weight_decay=config.value_weight_decay,
    )
    sample_generator = torch.Generator(device=config.device)
    sample_generator.manual_seed(config.train_seed + 94_003)

    initial_loss = 0.0
    final_loss = 0.0
    baseline.train()
    for pretrain_step in trange(
        config.value_pretrain_steps, desc="Pretraining return baseline", leave=False
    ):
        indices = torch.randint(
            len(targets),
            (config.value_batch_size,),
            generator=sample_generator,
            device=config.device,
        )
        predictions = baseline(states[indices], time_steps[indices])
        loss = F.mse_loss(predictions, targets[indices])
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(baseline.parameters(), config.clip_grad)
        optimizer.step()
        if pretrain_step == 0:
            initial_loss = loss.item()
        final_loss = loss.item()

    baseline.eval()
    with torch.no_grad():
        predicted_returns = baseline(states, time_steps) * target_std + target_mean
        residual = target_returns_np - predicted_returns.cpu().numpy()
    residual_std = float(np.std(residual))
    total_variance = float(np.var(target_returns_np))
    r_squared = 1.0 - float(np.mean(np.square(residual))) / max(total_variance, 1e-12)
    for parameter in baseline.parameters():
        parameter.requires_grad_(False)

    stats = {
        "value/num_transitions": float(len(target_returns_np)),
        "value/target_return_mean": target_mean,
        "value/target_return_std": target_std,
        "value/pretrain_initial_mse": initial_loss,
        "value/pretrain_final_mse": final_loss,
        "value/pretrain_r2": r_squared,
        "value/pretrain_residual_std": residual_std,
    }
    return baseline, target_mean, target_std, stats


def advantage_weights(
    returns: torch.Tensor,
    value_predictions: torch.Tensor,
    mask: torch.Tensor,
    config: TrajectoryAdvantageTrainConfig,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Clipped exponential weights normalized to mean one over valid tokens."""

    valid_mask = mask.to(torch.bool)
    advantages = returns - value_predictions
    valid_advantages = advantages[valid_mask]
    advantage_mean = valid_advantages.mean()
    advantage_std = valid_advantages.std(unbiased=False).clamp_min(1e-6)
    standardized = (advantages - advantage_mean) / advantage_std
    raw_weights = torch.exp(config.advantage_temperature * standardized)
    raw_weights = raw_weights.clamp(
        min=config.advantage_weight_min, max=config.advantage_weight_max
    )
    normalization = raw_weights[valid_mask].mean().clamp_min(1e-6)
    weights = torch.ones_like(raw_weights)
    weights[valid_mask] = raw_weights[valid_mask] / normalization
    return weights, advantages, standardized


@pyrallis.wrap()
def train(config: TrajectoryAdvantageTrainConfig):
    if not config.pretrained_checkpoint_path:
        raise ValueError("pretrained_checkpoint_path is required for fair continuation")
    if config.value_pretrain_steps < 1:
        raise ValueError("value_pretrain_steps must be positive")
    if config.value_batch_size < 1:
        raise ValueError("value_batch_size must be positive")
    if config.advantage_temperature <= 0.0:
        raise ValueError("advantage_temperature must be positive")
    if not 0.0 < config.advantage_weight_min <= config.advantage_weight_max:
        raise ValueError("advantage weights must satisfy 0 < min <= max")

    set_seed(config.train_seed, deterministic_torch=config.deterministic_torch)
    wandb_init(asdict(config))

    dataset = SequenceDataset(
        config.env_name,
        seq_len=config.seq_len,
        reward_scale=config.reward_scale,
        reward_mode=config.reward_mode,
    )
    wandb.log(
        {f"dataset/{key}": value for key, value in dataset.stats.items()}, step=0
    )
    trainloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        pin_memory=True,
        num_workers=config.num_workers,
    )
    eval_env = wrap_env(
        env=gym.make(config.env_name),
        state_mean=dataset.state_mean,
        state_std=dataset.state_std,
        reward_scale=config.reward_scale,
    )

    config.state_dim = eval_env.observation_space.shape[0]
    config.action_dim = eval_env.action_space.shape[0]
    model = DecisionTransformer(
        state_dim=config.state_dim,
        action_dim=config.action_dim,
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
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=config.betas,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda steps: min((steps + 1) / config.warmup_steps, 1),
    )

    checkpoint = torch.load(
        config.pretrained_checkpoint_path, map_location=config.device
    )
    model.load_state_dict(checkpoint["model_state"])
    if "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    if "scheduler_state" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    start_step = int(checkpoint["next_step"])
    print(f"Loaded pretrained DT checkpoint at step {start_step}")
    wandb.log({"checkpoint/loaded_step": start_step}, step=start_step)

    # The auxiliary pretraining must not alter the resumed DT's random data or
    # dropout sequence.  It uses its own generator and we restore all global
    # states before the first continuation update.
    continuation_rng_state = save_rng_state()
    baseline, target_mean, target_std, baseline_stats = pretrain_return_baseline(
        config=config, dataset=dataset, state_dim=config.state_dim
    )
    restore_rng_state(continuation_rng_state)
    baseline.eval()
    wandb.log(baseline_stats, step=start_step)
    wandb.log({"value/frozen": 1.0}, step=start_step)

    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as file:
            pyrallis.dump(config, file)
        torch.save(
            {
                "value_model_state": baseline.state_dict(),
                "target_mean": target_mean,
                "target_std": target_std,
                "value_stats": baseline_stats,
                "config": asdict(config),
            },
            os.path.join(config.checkpoints_path, "trajectory_return_baseline.pt"),
        )

    print(f"Total DT parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Frozen baseline parameters: {sum(p.numel() for p in baseline.parameters())}")
    trainloader_iter = iter(trainloader)
    for step in trange(start_step, config.update_steps, desc="Training"):
        batch = next(trainloader_iter)
        states, actions, returns, time_steps, mask = [
            item.to(config.device) for item in batch
        ]
        predicted_actions = model(
            states=states,
            actions=actions,
            returns_to_go=returns,
            time_steps=time_steps,
            padding_mask=~mask.to(torch.bool),
        )
        elementwise_loss = F.mse_loss(
            predicted_actions, actions.detach(), reduction="none"
        )
        unweighted_dt_loss = (elementwise_loss * mask.unsqueeze(-1)).mean()
        with torch.no_grad():
            value_predictions = (
                baseline(states, time_steps) * target_std + target_mean
            )
            weights, advantages, standardized_advantages = advantage_weights(
                returns=returns,
                value_predictions=value_predictions,
                mask=mask,
                config=config,
            )
        weighted_dt_loss = (
            elementwise_loss * mask.unsqueeze(-1) * weights.unsqueeze(-1)
        ).mean()

        optimizer.zero_grad()
        weighted_dt_loss.backward()
        if config.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
        optimizer.step()
        scheduler.step()

        valid_mask = mask.to(torch.bool)
        valid_weights = weights[valid_mask]
        weight_second_moment = valid_weights.square().mean().clamp_min(1e-6)
        metrics = {
            # Keep this field directly comparable to the original CORL DT
            # behavior-cloning loss, while logging the optimized loss separately.
            "train_loss": unweighted_dt_loss.item(),
            "train/weighted_dt_loss": weighted_dt_loss.item(),
            "train/advantage_mean": advantages[valid_mask].mean().item(),
            "train/advantage_std": advantages[valid_mask].std(unbiased=False).item(),
            "train/standardized_advantage_mean": (
                standardized_advantages[valid_mask].mean().item()
            ),
            "train/weight_mean": valid_weights.mean().item(),
            "train/weight_min": valid_weights.min().item(),
            "train/weight_max": valid_weights.max().item(),
            "train/weight_effective_fraction": (1.0 / weight_second_moment).item(),
            "learning_rate": scheduler.get_last_lr()[0],
        }
        wandb.log(metrics, step=step)

        if step % config.eval_every == 0 or step == config.update_steps - 1:
            model.eval()
            for target_return in config.target_returns:
                eval_env.seed(config.eval_seed)
                eval_returns = []
                eval_lengths = []
                for _ in trange(config.eval_episodes, desc="Evaluation", leave=False):
                    eval_return, eval_len = eval_rollout(
                        model=model,
                        env=eval_env,
                        target_return=target_return * config.reward_scale,
                        device=config.device,
                        reward_mode=config.reward_mode,
                    )
                    eval_returns.append(eval_return / config.reward_scale)
                    eval_lengths.append(eval_len)
                normalized_scores = (
                    eval_env.get_normalized_score(np.asarray(eval_returns)) * 100
                )
                wandb.log(
                    {
                        f"eval/{target_return}_return_mean": np.mean(eval_returns),
                        f"eval/{target_return}_return_std": np.std(eval_returns),
                        f"eval/{target_return}_length_mean": np.mean(eval_lengths),
                        f"eval/{target_return}_length_std": np.std(eval_lengths),
                        f"eval/{target_return}_normalized_score_mean": np.mean(
                            normalized_scores
                        ),
                        f"eval/{target_return}_normalized_score_std": np.std(
                            normalized_scores
                        ),
                    },
                    step=step,
                )
            model.train()

    if config.checkpoints_path is not None:
        torch.save(
            {
                "model_state": model.state_dict(),
                "state_mean": dataset.state_mean,
                "state_std": dataset.state_std,
                "value_model_state": baseline.state_dict(),
                "value_target_mean": target_mean,
                "value_target_std": target_std,
                "next_step": config.update_steps,
            },
            os.path.join(
                config.checkpoints_path, "trajectory_advantage_dt_checkpoint.pt"
            ),
        )
    wandb.finish()


if __name__ == "__main__":
    train()
