"""State-relative outcome weighted Decision Transformer.

This file contains two deliberately matched delayed-reward DT continuations:

* ``global_return`` uses actions from the globally highest-return trajectories.
* ``state_relative`` uses actions whose observed trajectory outcome is high
  relative to actions available near the same normalized state.

Both retain the original CORL DT behavior-cloning loss on recorded delayed
RTGs.  A second, high-RTG action loss only reinforces selected transitions,
and a frozen 50k DT reference lightly anchors that high-RTG branch.  There is
no Q-function, temporal-difference target, Bellman backup, or reward
redistribution: each quality label comes only from the recorded terminal
trajectory outcome in the offline data.
"""

import copy
import os
import random
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import gym
import numpy as np
import pyrallis
import torch
import torch.nn.functional as F
import wandb
from dt import (
    DecisionTransformer,
    eval_rollout,
    pad_along_axis,
    SequenceDataset,
    set_seed,
    TrainConfig,
    wandb_init,
    wrap_env,
)
from torch.utils.data import DataLoader, IterableDataset
from tqdm.auto import trange

@dataclass
class StateRelativeTrainConfig(TrainConfig):
    """Configuration shared by the global and state-relative controls."""

    pretrained_checkpoint_path: str = ""
    quality_mode: str = "state_relative"
    quality_fraction: float = 0.30
    quality_aux_weight: float = 0.10
    reference_weight: float = 0.10
    quality_target_return: float = 12_000.0
    quality_aux_batch_size: int = 1_024
    state_neighbor_count: int = 64
    state_projection_bits: int = 8
    state_neighbor_query_chunk_size: int = 256
    state_neighbor_seed: int = 91_731


class QualitySequenceDataset(IterableDataset):
    """CORL DT sequences plus a precomputed selected-transition mask.

    The sampler is intentionally the same as ``SequenceDataset``: trajectories
    are drawn proportional to their length and start positions are uniform.
    The extra mask only decides which tokens receive the high-RTG auxiliary
    loss; it never removes data from the ordinary delayed-DT loss.
    """

    def __init__(
        self,
        sequence_dataset: SequenceDataset,
        quality_masks: List[np.ndarray],
    ):
        self.trajectories = sequence_dataset.dataset
        self.quality_masks = quality_masks
        self.state_mean = sequence_dataset.state_mean
        self.state_std = sequence_dataset.state_std
        self.reward_scale = sequence_dataset.reward_scale
        self.seq_len = sequence_dataset.seq_len
        self.sample_prob = sequence_dataset.sample_prob

        if len(self.trajectories) != len(self.quality_masks):
            raise ValueError("quality masks must match the number of trajectories")
        for trajectory, quality_mask in zip(self.trajectories, self.quality_masks):
            if trajectory["actions"].shape[0] != quality_mask.shape[0]:
                raise ValueError("each quality mask must match its trajectory length")

    def _prepare_sample(
        self, traj_idx: int, start_idx: int
    ) -> Tuple[np.ndarray, ...]:
        trajectory = self.trajectories[traj_idx]
        states = trajectory["observations"][start_idx : start_idx + self.seq_len]
        actions = trajectory["actions"][start_idx : start_idx + self.seq_len]
        returns = trajectory["returns"][start_idx : start_idx + self.seq_len]
        quality_mask = self.quality_masks[traj_idx][
            start_idx : start_idx + self.seq_len
        ]
        time_steps = np.arange(start_idx, start_idx + self.seq_len)

        states = (states - self.state_mean) / self.state_std
        returns = returns * self.reward_scale
        mask = np.hstack(
            [
                np.ones(states.shape[0], dtype=np.float32),
                np.zeros(self.seq_len - states.shape[0], dtype=np.float32),
            ]
        )
        if states.shape[0] < self.seq_len:
            states = pad_along_axis(states, pad_to=self.seq_len)
            actions = pad_along_axis(actions, pad_to=self.seq_len)
            returns = pad_along_axis(returns, pad_to=self.seq_len)
            quality_mask = pad_along_axis(quality_mask, pad_to=self.seq_len)

        return states, actions, returns, time_steps, mask, quality_mask

    def __iter__(self):
        while True:
            traj_idx = np.random.choice(len(self.trajectories), p=self.sample_prob)
            start_idx = random.randint(
                0, self.trajectories[traj_idx]["rewards"].shape[0] - 1
            )
            yield self._prepare_sample(traj_idx, start_idx)


def flatten_transition_metadata(
    dataset: SequenceDataset,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int]]:
    """Flatten normalized states and their recorded delayed trajectory outcomes."""

    states_parts = []
    outcome_parts = []
    trajectory_id_parts = []
    trajectory_lengths = []

    for trajectory_id, trajectory in enumerate(dataset.dataset):
        observations = trajectory["observations"]
        length = observations.shape[0]
        normalized_states = (observations - dataset.state_mean) / dataset.state_std
        # In delayed mode, returns[0] equals the observed terminal outcome and
        # is constant across the trajectory.  We use no predicted value here.
        outcome = float(trajectory["returns"][0] * dataset.reward_scale)
        states_parts.append(normalized_states.astype(np.float32))
        outcome_parts.append(np.full(length, outcome, dtype=np.float32))
        trajectory_id_parts.append(np.full(length, trajectory_id, dtype=np.int32))
        trajectory_lengths.append(length)

    return (
        np.concatenate(states_parts, axis=0),
        np.concatenate(outcome_parts, axis=0),
        np.concatenate(trajectory_id_parts, axis=0),
        trajectory_lengths,
    )


def split_transition_mask(
    flat_mask: np.ndarray, trajectory_lengths: List[int]
) -> List[np.ndarray]:
    """Restore a flattened transition mask to the dataset trajectory layout."""

    masks = []
    offset = 0
    for length in trajectory_lengths:
        masks.append(flat_mask[offset : offset + length].astype(np.float32))
        offset += length
    if offset != len(flat_mask):
        raise ValueError("transition mask length does not match trajectory lengths")
    return masks


def build_global_quality_mask(
    outcomes: np.ndarray, quality_fraction: float
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Select the same transition fraction by globally observed outcome."""

    threshold = float(np.quantile(outcomes, 1.0 - quality_fraction))
    selected = outcomes >= threshold
    selected_outcomes = outcomes[selected]
    return selected.astype(np.float32), {
        "quality/global_threshold": threshold,
        "quality/selected_fraction": float(selected.mean()),
        "quality/outcome_mean": float(outcomes.mean()),
        "quality/selected_outcome_mean": float(selected_outcomes.mean()),
        "quality/selected_outcome_std": float(selected_outcomes.std()),
    }


def build_state_relative_quality_mask(
    states: np.ndarray,
    outcomes: np.ndarray,
    trajectory_ids: np.ndarray,
    config: StateRelativeTrainConfig,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Select transitions with top local outcome advantages.

    Random-projection buckets provide local candidate sets without adding a
    dependency to the experiment environment.  Exact squared distances choose
    the nearest cross-trajectory states inside each bucket.  Neighbor states
    from the same trajectory are excluded so a trajectory cannot label its own
    adjacent states as evidence.
    """

    num_transitions = len(states)
    neighbor_count = config.state_neighbor_count
    if config.state_projection_bits < 1:
        raise ValueError("state_projection_bits must be positive")
    if config.state_projection_bits > 16:
        raise ValueError("state_projection_bits must be at most 16")

    projection_rng = np.random.default_rng(config.state_neighbor_seed)
    projections = projection_rng.standard_normal(
        (states.shape[1], config.state_projection_bits)
    ).astype(np.float32)
    projection_signs = (states @ projections) >= 0.0
    bit_values = 1 << np.arange(config.state_projection_bits, dtype=np.int64)
    bucket_codes = projection_signs.astype(np.int64) @ bit_values
    sorted_indices = np.argsort(bucket_codes, kind="stable")
    sorted_codes = bucket_codes[sorted_indices]
    bucket_starts = np.r_[0, np.flatnonzero(np.diff(sorted_codes)) + 1]
    bucket_ends = np.r_[bucket_starts[1:], num_transitions]

    # Only rare, undersized buckets use this fallback.  It keeps the method
    # defined for every transition while the recorded fallback fraction tells
    # us whether the local-bucket approximation was sufficiently selective.
    fallback_size = min(num_transitions, max(4_096, neighbor_count * 8))
    fallback_indices = np.linspace(
        0, num_transitions - 1, num=fallback_size, dtype=np.int64
    )
    local_advantages = np.empty(num_transitions, dtype=np.float32)
    neighbor_counts = np.empty(num_transitions, dtype=np.int32)
    neighbor_distances = np.empty(num_transitions, dtype=np.float32)
    used_fallback = np.zeros(num_transitions, dtype=bool)

    for bucket_start, bucket_end in zip(bucket_starts, bucket_ends):
        bucket_indices = sorted_indices[bucket_start:bucket_end]
        bucket_trajectory_ids = trajectory_ids[bucket_indices]
        _, inverse_trajectory_ids, trajectory_counts = np.unique(
            bucket_trajectory_ids, return_inverse=True, return_counts=True
        )
        cross_trajectory_counts = (
            len(bucket_indices) - trajectory_counts[inverse_trajectory_ids]
        )
        needs_fallback = cross_trajectory_counts < neighbor_count
        if np.any(needs_fallback):
            candidate_indices = np.unique(
                np.concatenate([bucket_indices, fallback_indices])
            )
            used_fallback[bucket_indices[needs_fallback]] = True
        else:
            candidate_indices = bucket_indices

        candidate_states = states[candidate_indices]
        candidate_norms = np.sum(candidate_states * candidate_states, axis=1)
        candidate_trajectory_ids = trajectory_ids[candidate_indices]
        candidate_outcomes = outcomes[candidate_indices]

        for query_start in range(
            0, len(bucket_indices), config.state_neighbor_query_chunk_size
        ):
            query_end = min(
                query_start + config.state_neighbor_query_chunk_size,
                len(bucket_indices),
            )
            query_indices = bucket_indices[query_start:query_end]
            query_states = states[query_indices]
            query_norms = np.sum(query_states * query_states, axis=1, keepdims=True)
            squared_distances = (
                query_norms
                + candidate_norms[None, :]
                - 2.0 * query_states @ candidate_states.T
            )
            squared_distances = np.maximum(squared_distances, 0.0)
            same_trajectory = candidate_trajectory_ids[None, :] == trajectory_ids[
                query_indices, None
            ]
            squared_distances[same_trajectory] = np.inf
            available_counts = (~same_trajectory).sum(axis=1)
            if np.any(available_counts < neighbor_count):
                raise RuntimeError(
                    "a transition has too few cross-trajectory neighbors"
                )

            nearest_columns = np.argpartition(
                squared_distances, kth=neighbor_count - 1, axis=1
            )[:, :neighbor_count]
            nearest_distances = np.take_along_axis(
                squared_distances, nearest_columns, axis=1
            )
            nearest_outcomes = candidate_outcomes[nearest_columns]
            local_means = nearest_outcomes.mean(axis=1)
            local_stds = np.sqrt(
                np.maximum(nearest_outcomes.var(axis=1), 1e-6)
            )
            local_advantages[query_indices] = (
                outcomes[query_indices] - local_means
            ) / local_stds
            neighbor_counts[query_indices] = neighbor_count
            neighbor_distances[query_indices] = np.sqrt(nearest_distances).mean(axis=1)

    threshold = float(np.quantile(local_advantages, 1.0 - config.quality_fraction))
    selected = local_advantages >= threshold
    selected_outcomes = outcomes[selected]
    return selected.astype(np.float32), {
        "quality/local_advantage_threshold": threshold,
        "quality/local_advantage_mean": float(local_advantages.mean()),
        "quality/local_advantage_std": float(local_advantages.std()),
        "quality/selected_fraction": float(selected.mean()),
        "quality/outcome_mean": float(outcomes.mean()),
        "quality/selected_outcome_mean": float(selected_outcomes.mean()),
        "quality/selected_outcome_std": float(selected_outcomes.std()),
        "quality/neighbor_count_mean": float(neighbor_counts.mean()),
        "quality/neighbor_count_min": float(neighbor_counts.min()),
        "quality/neighbor_distance_mean": float(neighbor_distances.mean()),
        "quality/projection_bits": float(config.state_projection_bits),
        "quality/projection_bucket_count": float(len(bucket_starts)),
        "quality/projection_bucket_max_size": float(
            np.diff(np.r_[bucket_starts, num_transitions]).max()
        ),
        "quality/fallback_transition_fraction": float(used_fallback.mean()),
    }


def build_quality_masks(
    dataset: SequenceDataset, config: StateRelativeTrainConfig
) -> Tuple[List[np.ndarray], Dict[str, float]]:
    """Build either matched global or state-relative selected-token labels."""

    states, outcomes, trajectory_ids, trajectory_lengths = flatten_transition_metadata(
        dataset
    )
    if config.quality_mode == "global_return":
        flat_mask, stats = build_global_quality_mask(
            outcomes=outcomes, quality_fraction=config.quality_fraction
        )
    elif config.quality_mode == "state_relative":
        flat_mask, stats = build_state_relative_quality_mask(
            states=states,
            outcomes=outcomes,
            trajectory_ids=trajectory_ids,
            config=config,
        )
    else:
        raise ValueError(
            "quality_mode must be one of {'global_return', 'state_relative'}"
        )

    stats.update(
        {
            "quality/num_transitions": float(len(outcomes)),
            "quality/num_trajectories": float(len(trajectory_lengths)),
        }
    )
    return split_transition_mask(flat_mask, trajectory_lengths), stats


def masked_action_mse(
    prediction: torch.Tensor, action: torch.Tensor, token_weight: torch.Tensor
) -> torch.Tensor:
    """Mean MSE over selected tokens, normalized independently of selection rate."""

    elementwise_loss = F.mse_loss(prediction, action.detach(), reduction="none")
    denominator = token_weight.sum().clamp_min(1.0) * prediction.shape[-1]
    return (elementwise_loss * token_weight.unsqueeze(-1)).sum() / denominator


def corl_dt_action_mse(
    prediction: torch.Tensor, action: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Reproduce CORL DT's recorded-RTG behavior-cloning loss exactly."""

    elementwise_loss = F.mse_loss(prediction, action.detach(), reduction="none")
    return (elementwise_loss * mask.unsqueeze(-1)).mean()


@pyrallis.wrap()
def train(config: StateRelativeTrainConfig):
    if config.reward_mode != "delayed":
        raise ValueError("this sparse-reward prototype requires --reward_mode delayed")
    if not config.pretrained_checkpoint_path:
        raise ValueError("pretrained_checkpoint_path is required for fair continuation")
    if not 0.0 < config.quality_fraction < 1.0:
        raise ValueError("quality_fraction must be in (0, 1)")
    if config.quality_aux_weight <= 0.0:
        raise ValueError("quality_aux_weight must be positive")
    if config.reference_weight < 0.0:
        raise ValueError("reference_weight must be non-negative")
    if not 0 < config.quality_aux_batch_size <= config.batch_size:
        raise ValueError("quality_aux_batch_size must be in [1, batch_size]")
    if config.state_neighbor_count < 1:
        raise ValueError("state_neighbor_count must be positive")
    if not 1 <= config.state_projection_bits <= 16:
        raise ValueError("state_projection_bits must be in [1, 16]")
    if config.state_neighbor_query_chunk_size < 1:
        raise ValueError("state_neighbor_query_chunk_size must be positive")

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
    quality_masks, quality_stats = build_quality_masks(dataset, config)
    wandb.log(quality_stats, step=0)
    quality_dataset = QualitySequenceDataset(dataset, quality_masks)
    trainloader = DataLoader(
        quality_dataset,
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

    reference_model = copy.deepcopy(model).to(config.device)
    reference_model.eval()
    for parameter in reference_model.parameters():
        parameter.requires_grad_(False)
    wandb.log({"train/reference_initialized": 1.0}, step=start_step)

    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as file:
            pyrallis.dump(config, file)

    print(f"Total DT parameters: {sum(p.numel() for p in model.parameters())}")
    trainloader_iter = iter(trainloader)
    for step in trange(start_step, config.update_steps, desc="Training"):
        batch = next(trainloader_iter)
        states, actions, returns, time_steps, mask, quality_mask = [
            item.to(config.device) for item in batch
        ]
        padding_mask = ~mask.to(torch.bool)
        predicted_actions = model(
            states=states,
            actions=actions,
            returns_to_go=returns,
            time_steps=time_steps,
            padding_mask=padding_mask,
        )
        dt_loss = corl_dt_action_mse(predicted_actions, actions, mask)

        # Retain CORL's full 4096-sequence DT batch, but bound the additional
        # high-RTG activation memory so this auxiliary remains usable on the
        # same GPUNorm hardware as the previous continuations.  The main
        # sampler shuffles every sequence independently, so this prefix is an
        # unbiased auxiliary minibatch.
        aux_batch_size = config.quality_aux_batch_size
        aux_states = states[:aux_batch_size]
        aux_actions = actions[:aux_batch_size]
        aux_returns = returns[:aux_batch_size]
        aux_time_steps = time_steps[:aux_batch_size]
        aux_mask = mask[:aux_batch_size]
        aux_padding_mask = padding_mask[:aux_batch_size]
        high_returns = torch.full_like(
            aux_returns, config.quality_target_return * config.reward_scale
        ) * aux_mask
        high_prediction = model(
            states=aux_states,
            actions=aux_actions,
            returns_to_go=high_returns,
            time_steps=aux_time_steps,
            padding_mask=aux_padding_mask,
        )
        selected_tokens = quality_mask[:aux_batch_size] * aux_mask
        quality_loss = masked_action_mse(
            high_prediction, aux_actions, selected_tokens
        )
        with torch.no_grad():
            reference_prediction = reference_model(
                states=aux_states,
                actions=aux_actions,
                returns_to_go=high_returns,
                time_steps=aux_time_steps,
                padding_mask=aux_padding_mask,
            )
        reference_loss = masked_action_mse(
            high_prediction, reference_prediction, aux_mask
        )
        total_loss = (
            dt_loss
            + config.quality_aux_weight * quality_loss
            + config.reference_weight * reference_loss
        )

        optimizer.zero_grad()
        total_loss.backward()
        if config.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
        optimizer.step()
        scheduler.step()

        metrics = {
            "train_loss": dt_loss.item(),
            "train/total_loss": total_loss.item(),
            "train/quality_loss": quality_loss.item(),
            "train/reference_loss": reference_loss.item(),
            "train/selected_token_fraction": selected_tokens[aux_mask.to(torch.bool)]
            .mean()
            .item(),
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
                "next_step": config.update_steps,
                "config": asdict(config),
            },
            os.path.join(config.checkpoints_path, "state_relative_dt_checkpoint.pt"),
        )
    wandb.finish()


if __name__ == "__main__":
    train()
