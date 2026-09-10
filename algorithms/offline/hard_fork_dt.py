"""Reference-anchored hard-positive and hard-fork DT experiments.

Both variants resume from the same frozen CORL DT checkpoint and the same set of
locally aligned trajectory forks.  Hard-positive DT always reduces error on the
preferred action.  Hard-fork DT uses the non-preferred action only as a detached
stopping boundary, so it can decide when a correction is still necessary but
can never push the policy away without bound.  A frozen reference model anchors
predictions at the CORL evaluation return.
"""

import copy
import os
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple

import gym
import numpy as np
import pyrallis
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from tqdm.auto import trange

import wandb
from dt import (
    DecisionTransformer,
    SequenceDataset,
    TrainConfig,
    eval_rollout,
    set_seed,
    wandb_init,
    wrap_env,
)


@dataclass
class SAPTrainConfig(TrainConfig):
    preference_start_step: int = 50_000
    preference_weight: float = 0.001
    reference_weight: float = 0.1
    preference_margin: float = 0.05
    preference_batch_size: int = 256
    preference_good_quantile: float = 0.70
    preference_bad_quantile: float = 0.30
    preference_timestep_bucket: int = 25
    preference_num_pairs: int = 100_000
    preference_num_candidates: int = 64
    preference_state_keep_fraction: float = 0.25
    reference_target_return: float = 12_000.0
    preference_pair_seed: int = 0
    preference_mode: str = "state_aligned"
    preference_arrays_path: Optional[str] = None
    preference_hardness_temperature: float = 0.05
    reference_checkpoint_path: Optional[str] = None
    pretrained_checkpoint_path: Optional[str] = None


class StateAlignedPreferenceDataset(IterableDataset):
    """Strict approximate state matching between good and bad trajectories.

    For each sampled good transition, we draw several bad candidates from the
    same timestep bucket and retain the closest state after dataset
    normalization.  Only the closest fraction of all candidate matches is kept.
    The model is conditioned on the preferred trajectory's recorded RTG, rather
    than an artificial evaluation RTG.
    """

    def __init__(
        self,
        sequence_dataset: SequenceDataset,
        state_keep_fraction: float,
        good_quantile: float,
        bad_quantile: float,
        timestep_bucket: int,
        num_pairs: int,
        num_candidates: int,
        pair_seed: int,
        pair_arrays_path: Optional[str] = None,
        hardness_temperature: float = 0.05,
    ):
        self.trajectories = sequence_dataset.dataset
        self.seq_len = sequence_dataset.seq_len
        self.state_mean = sequence_dataset.state_mean.astype(np.float32)
        self.state_std = sequence_dataset.state_std.astype(np.float32)
        self.reward_scale = sequence_dataset.reward_scale
        if not 0.0 < state_keep_fraction <= 1.0:
            raise ValueError("preference_state_keep_fraction must be in (0, 1]")
        self.state_keep_fraction = float(state_keep_fraction)

        trajectory_returns = np.asarray(
            [float(trajectory["returns"][0]) for trajectory in self.trajectories],
            dtype=np.float32,
        )
        good_threshold = float(np.quantile(trajectory_returns, good_quantile))
        bad_threshold = float(np.quantile(trajectory_returns, bad_quantile))

        if pair_arrays_path is not None:
            if hardness_temperature <= 0.0:
                raise ValueError("preference_hardness_temperature must be positive")
            pair_arrays = np.load(pair_arrays_path)
            valid_branch = pair_arrays["valid_branch"].astype(bool)
            self.pairs = pair_arrays["pairs"][valid_branch].astype(np.int32)
            frozen_positive_error = pair_arrays["positive_error"][valid_branch]
            frozen_negative_error = pair_arrays["negative_error"][valid_branch]
            frozen_signed_margin = frozen_negative_error - frozen_positive_error
            hardness = 1.0 / (
                1.0
                + np.exp(
                    np.clip(
                        (frozen_signed_margin - 0.05) / hardness_temperature,
                        -60.0,
                        60.0,
                    )
                )
            )
            base_confidence = pair_arrays["pair_confidence"][valid_branch]
            raw_weights = np.maximum(base_confidence * hardness, 1e-6)
            self.pair_confidences = (
                raw_weights / raw_weights.mean()
            ).astype(np.float32)
            if len(self.pairs) == 0:
                raise RuntimeError("The diagnostic contains no valid branch pairs")

            hard_pair = pair_arrays["hard_pair"][valid_branch].astype(bool)
            good_pair_returns = trajectory_returns[self.pairs[:, 0]]
            bad_pair_returns = trajectory_returns[self.pairs[:, 2]]
            return_gaps = good_pair_returns - bad_pair_returns
            effective_pairs = float(
                np.square(self.pair_confidences.sum())
                / np.square(self.pair_confidences).sum()
            )
            self.stats = {
                "num_pairs": float(len(self.pairs)),
                "valid_branch_pairs": float(valid_branch.sum()),
                "hard_pairs": float(hard_pair.sum()),
                "good_return_threshold": good_threshold,
                "bad_return_threshold": bad_threshold,
                "hardness_temperature": float(hardness_temperature),
                "hardness_mean": float(hardness.mean()),
                "hardness_p50": float(np.quantile(hardness, 0.50)),
                "hardness_p90": float(np.quantile(hardness, 0.90)),
                "pair_weight_mean": float(self.pair_confidences.mean()),
                "pair_weight_max": float(self.pair_confidences.max()),
                "effective_pair_count": effective_pairs,
                "return_gap_mean": float(return_gaps.mean()),
                "return_gap_min": float(return_gaps.min()),
            }
            return

        good_trajectories = np.flatnonzero(trajectory_returns >= good_threshold)
        bad_trajectories = np.flatnonzero(trajectory_returns <= bad_threshold)

        good_buckets = self._make_buckets(good_trajectories, timestep_bucket)
        bad_buckets = self._make_buckets(bad_trajectories, timestep_bucket)
        common_buckets = sorted(set(good_buckets).intersection(bad_buckets))
        if not common_buckets:
            raise RuntimeError("No timestep bucket contains both good and bad data")

        rng = np.random.default_rng(pair_seed)
        pairs_per_bucket = int(np.ceil(num_pairs / len(common_buckets)))
        pair_parts = []
        distance_parts = []

        state_mean = self.state_mean.reshape(-1)
        state_std = self.state_std.reshape(-1)
        for bucket in common_buckets:
            good_entries, good_states = good_buckets[bucket]
            bad_entries, bad_states = bad_buckets[bucket]
            sample_count = min(pairs_per_bucket, num_pairs)

            good_choice = rng.integers(0, len(good_entries), size=sample_count)
            candidate_choice = rng.integers(
                0,
                len(bad_entries),
                size=(sample_count, num_candidates),
            )

            normalized_good = (
                good_states[good_choice] - state_mean
            ) / state_std
            normalized_bad = (
                bad_states[candidate_choice] - state_mean
            ) / state_std
            squared_distances = np.square(
                normalized_bad - normalized_good[:, None, :]
            ).mean(axis=-1)
            nearest_column = squared_distances.argmin(axis=1)
            nearest_choice = candidate_choice[
                np.arange(sample_count), nearest_column
            ]

            pair_parts.append(
                np.concatenate(
                    [good_entries[good_choice], bad_entries[nearest_choice]], axis=1
                )
            )
            distance_parts.append(
                np.sqrt(
                    squared_distances[np.arange(sample_count), nearest_column]
                )
            )

        candidate_pairs = np.concatenate(pair_parts, axis=0).astype(np.int32)
        candidate_distances = np.concatenate(distance_parts, axis=0)

        # Remove duplicate sampled pairs, then retain only the closest matches.
        self.pairs, unique_indices = np.unique(
            candidate_pairs, axis=0, return_index=True
        )
        match_distances = candidate_distances[unique_indices]
        state_distance_threshold = float(
            np.quantile(match_distances, self.state_keep_fraction)
        )
        keep_mask = match_distances <= state_distance_threshold
        self.pairs = self.pairs[keep_mask].astype(np.int32)
        match_distances = match_distances[keep_mask]
        if len(self.pairs) > num_pairs:
            selected = rng.choice(len(self.pairs), size=num_pairs, replace=False)
            self.pairs = self.pairs[selected]
            match_distances = match_distances[selected]
        if len(self.pairs) == 0:
            raise RuntimeError("Strict state filtering removed every preference pair")

        confidence_scale = max(float(np.median(match_distances)), 1e-6)
        self.pair_confidences = np.exp(
            -match_distances / confidence_scale
        ).astype(np.float32)
        good_pair_returns = trajectory_returns[self.pairs[:, 0]]
        bad_pair_returns = trajectory_returns[self.pairs[:, 2]]
        return_gaps = good_pair_returns - bad_pair_returns

        self.stats: Dict[str, float] = {
            "num_candidate_pairs": float(len(candidate_pairs)),
            "num_pairs": float(len(self.pairs)),
            "state_keep_fraction": self.state_keep_fraction,
            "state_distance_threshold": state_distance_threshold,
            "good_return_threshold": good_threshold,
            "bad_return_threshold": bad_threshold,
            "match_distance_mean": float(match_distances.mean()),
            "match_distance_median": float(np.median(match_distances)),
            "match_distance_p90": float(np.quantile(match_distances, 0.90)),
            "pair_confidence_mean": float(self.pair_confidences.mean()),
            "return_gap_mean": float(return_gaps.mean()),
            "return_gap_min": float(return_gaps.min()),
        }

    def _make_buckets(
        self, trajectory_indices: np.ndarray, bucket_width: int
    ) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        entry_buckets = {}
        state_buckets = {}
        for trajectory_index in trajectory_indices:
            observations = self.trajectories[int(trajectory_index)]["observations"]
            for step, state in enumerate(observations):
                bucket = step // bucket_width
                entry_buckets.setdefault(bucket, []).append(
                    (int(trajectory_index), step)
                )
                state_buckets.setdefault(bucket, []).append(state)

        return {
            bucket: (
                np.asarray(entry_buckets[bucket], dtype=np.int32),
                np.asarray(state_buckets[bucket], dtype=np.float32),
            )
            for bucket in entry_buckets
        }

    def _context(self, trajectory_index: int, step: int):
        trajectory = self.trajectories[trajectory_index]
        start = max(0, step - self.seq_len + 1)
        stop = step + 1

        states = trajectory["observations"][start:stop].copy()
        actions = trajectory["actions"][start:stop].copy()
        # Match CORL's padding convention: padded positions receive increasing
        # timestep ids but are ignored by the attention padding mask.
        time_steps = np.arange(start, start + self.seq_len, dtype=np.int64)
        valid_length = stop - start
        returns = (
            trajectory["returns"][start:stop].copy() * self.reward_scale
        ).astype(np.float32)
        mask = np.concatenate(
            [
                np.ones(valid_length, dtype=np.float32),
                np.zeros(self.seq_len - valid_length, dtype=np.float32),
            ]
        )

        states = (states - self.state_mean) / self.state_std
        if valid_length < self.seq_len:
            state_padding = np.zeros(
                (self.seq_len - valid_length, states.shape[-1]), dtype=np.float32
            )
            action_padding = np.zeros(
                (self.seq_len - valid_length, actions.shape[-1]), dtype=np.float32
            )
            scalar_padding = np.zeros(
                self.seq_len - valid_length, dtype=np.float32
            )
            states = np.concatenate([states, state_padding], axis=0)
            actions = np.concatenate([actions, action_padding], axis=0)
            returns = np.concatenate([returns, scalar_padding], axis=0)

        return states, actions, returns, time_steps, mask, valid_length - 1

    def __iter__(self):
        while True:
            pair_index = np.random.randint(len(self.pairs))
            good_traj, good_step, bad_traj, bad_step = self.pairs[pair_index]
            negative_action = self.trajectories[int(bad_traj)]["actions"][
                int(bad_step)
            ].copy()
            yield self._context(int(good_traj), int(good_step)) + (
                negative_action,
                self.pair_confidences[pair_index],
            )


def predict_action(
    model: DecisionTransformer,
    states: torch.Tensor,
    actions: torch.Tensor,
    returns: torch.Tensor,
    time_steps: torch.Tensor,
    mask: torch.Tensor,
    target_index: torch.Tensor,
) -> torch.Tensor:
    predictions = model(
        states=states,
        actions=actions,
        returns_to_go=returns,
        time_steps=time_steps,
        padding_mask=~mask.to(torch.bool),
    )
    batch_index = torch.arange(predictions.shape[0], device=predictions.device)
    return predictions[batch_index, target_index]


@pyrallis.wrap()
def train(config: SAPTrainConfig):
    if config.preference_mode not in {
        "state_aligned",
        "hard_positive",
        "hard_fork",
    }:
        raise ValueError(f"Unknown preference_mode: {config.preference_mode}")
    if config.preference_mode != "state_aligned" and not config.preference_arrays_path:
        raise ValueError(
            "hard_positive and hard_fork require preference_arrays_path"
        )
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

    preference_dataset = StateAlignedPreferenceDataset(
        sequence_dataset=dataset,
        state_keep_fraction=config.preference_state_keep_fraction,
        good_quantile=config.preference_good_quantile,
        bad_quantile=config.preference_bad_quantile,
        timestep_bucket=config.preference_timestep_bucket,
        num_pairs=config.preference_num_pairs,
        num_candidates=config.preference_num_candidates,
        pair_seed=config.preference_pair_seed,
        pair_arrays_path=config.preference_arrays_path,
        hardness_temperature=config.preference_hardness_temperature,
    )
    wandb.log(
        {
            f"preference_data/{key}": value
            for key, value in preference_dataset.stats.items()
        },
        step=0,
    )
    preference_generator = torch.Generator()
    preference_generator.manual_seed(config.preference_pair_seed)
    preference_loader = DataLoader(
        preference_dataset,
        batch_size=config.preference_batch_size,
        pin_memory=True,
        num_workers=0,
        generator=preference_generator,
    )

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=config.betas,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optim,
        lambda steps: min((steps + 1) / config.warmup_steps, 1),
    )

    start_step = 0
    if config.pretrained_checkpoint_path is not None:
        checkpoint = torch.load(config.pretrained_checkpoint_path, map_location=config.device)
        model.load_state_dict(checkpoint["model_state"])
        if "optimizer_state" in checkpoint:
            optim.load_state_dict(checkpoint["optimizer_state"])
        if "scheduler_state" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_step = int(checkpoint["next_step"])
        print(f"Loaded pretrained DT checkpoint at step {start_step}")
        wandb.log({"checkpoint/loaded_step": start_step}, step=start_step)

    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as file:
            pyrallis.dump(config, file)

    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Preference statistics: {preference_dataset.stats}")
    trainloader_iter = iter(trainloader)
    preference_iter = iter(preference_loader)
    reference_model = None

    for step in trange(start_step, config.update_steps, desc="Training"):
        if step == config.preference_start_step:
            if config.reference_checkpoint_path is not None:
                checkpoint_dir = os.path.dirname(config.reference_checkpoint_path) or "."
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint = {
                    "next_step": step,
                    "model_state": model.state_dict(),
                    "optimizer_state": optim.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "state_mean": dataset.state_mean,
                    "state_std": dataset.state_std,
                    "config": asdict(config),
                }
                temp_path = f"{config.reference_checkpoint_path}.tmp-{os.getpid()}"
                torch.save(checkpoint, temp_path)
                os.replace(temp_path, config.reference_checkpoint_path)
                print(f"Saved reusable DT checkpoint: {config.reference_checkpoint_path}")
                wandb.log({"checkpoint/saved_step": step}, step=step)
                wandb.save(config.reference_checkpoint_path, base_path=checkpoint_dir)
            reference_model = copy.deepcopy(model).to(config.device)
            reference_model.eval()
            for parameter in reference_model.parameters():
                parameter.requires_grad_(False)
            wandb.log({"train/reference_initialized": 1.0}, step=step)

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
        dt_loss = (elementwise_loss * mask.unsqueeze(-1)).mean()
        total_loss = dt_loss
        preference_loss = None
        reference_loss = None
        preference_accuracy = None
        preference_active_ratio = None
        pair_confidence_mean = None
        positive_error = None
        negative_error = None

        if step >= config.preference_start_step:
            preference_batch = [
                item.to(config.device) for item in next(preference_iter)
            ]
            assert reference_model is not None
            (
                pref_states,
                pref_actions,
                pref_returns,
                pref_time_steps,
                pref_mask,
                target_index,
                negative_actions,
                pair_confidence,
            ) = preference_batch
            positive_prediction = predict_action(
                model,
                pref_states,
                pref_actions,
                pref_returns,
                pref_time_steps,
                pref_mask,
                target_index,
            )
            batch_index = torch.arange(
                positive_prediction.shape[0], device=config.device
            )
            positive_actions = pref_actions[batch_index, target_index]
            positive_error = F.mse_loss(
                positive_prediction, positive_actions.detach(), reduction="none"
            ).mean(dim=-1)
            negative_error = F.mse_loss(
                positive_prediction, negative_actions.detach(), reduction="none"
            ).mean(dim=-1)
            margin_violation = (
                positive_error - negative_error.detach() + config.preference_margin
            )
            if config.preference_mode == "hard_positive":
                weighted_preference = pair_confidence * positive_error
                preference_active_ratio = torch.ones_like(positive_error).mean()
            else:
                weighted_preference = pair_confidence * F.relu(margin_violation)
                preference_active_ratio = (margin_violation > 0).float().mean()
            preference_loss = (
                weighted_preference.sum() / pair_confidence.sum().clamp_min(1e-6)
            )
            preference_accuracy = (positive_error < negative_error).float().mean()
            pair_confidence_mean = pair_confidence.mean()

            anchor_returns = torch.full_like(
                pref_returns,
                config.reference_target_return * config.reward_scale,
            )
            anchor_returns = anchor_returns * pref_mask
            anchor_prediction = predict_action(
                model,
                pref_states,
                pref_actions,
                anchor_returns,
                pref_time_steps,
                pref_mask,
                target_index,
            )
            with torch.no_grad():
                reference_prediction = predict_action(
                    reference_model,
                    pref_states,
                    pref_actions,
                    anchor_returns,
                    pref_time_steps,
                    pref_mask,
                    target_index,
                )
            reference_loss = F.mse_loss(anchor_prediction, reference_prediction)
            total_loss = (
                dt_loss
                + config.preference_weight * preference_loss
                + config.reference_weight * reference_loss
            )

        optim.zero_grad()
        total_loss.backward()
        if config.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
        optim.step()
        scheduler.step()

        metrics = {
            "train_loss": dt_loss.item(),
            "train/total_loss": total_loss.item(),
            "learning_rate": scheduler.get_last_lr()[0],
            "train/preference_active": float(
                step >= config.preference_start_step
            ),
        }
        if preference_loss is not None:
            metrics.update(
                {
                    "train/preference_loss": preference_loss.item(),
                    "train/reference_loss": reference_loss.item(),
                    "train/preference_active_ratio": preference_active_ratio.item(),
                    "train/preference_margin_violation": margin_violation.mean().item(),
                    "train/pair_confidence_mean": pair_confidence_mean.item(),
                    "train/preference_accuracy": preference_accuracy.item(),
                    "train/preference_positive_error": positive_error.mean().item(),
                    "train/preference_negative_error": negative_error.mean().item(),
                }
            )
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
        checkpoint = {
            "model_state": model.state_dict(),
            "state_mean": dataset.state_mean,
            "state_std": dataset.state_std,
        }
        torch.save(
            checkpoint,
            os.path.join(config.checkpoints_path, "sap_dt_checkpoint.pt"),
        )
    wandb.finish()


if __name__ == "__main__":
    train()
