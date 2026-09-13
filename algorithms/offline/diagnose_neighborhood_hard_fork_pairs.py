"""Read-only diagnostic for expanding v3-high hard-fork preference pairs.

The existing v3-high run learns from a fixed set of strict state-aligned
pairs.  This diagnostic does not train a policy.  It asks whether a broader,
but still local, construction can yield enough additional *unresolved* action
comparisons to justify an ablation:

1. retrieve nearby states from other trajectories at nearby timesteps;
2. retain pairs with a large observed outcome gap and different actions;
3. score them with the frozen 50k delayed DT at RTG=12000;
4. count pairs that still violate v3-high's action margin.

For a follow-up ablation, the script also exports only the margin-active pairs
in the exact four-column format consumed by ``hard_fork_dt.py``.  The export is
data preparation only; this script never updates a model.

The nearest-neighbor search uses deterministic random-projection buckets and
exact distances within each bucket.  It requires only NumPy, matching the
server environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from dt import DecisionTransformer, SequenceDataset, set_seed

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--env_name", default="halfcheetah-medium-replay-v2")
    parser.add_argument("--reward_mode", default="delayed")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--neighbor_count", type=int, default=64)
    parser.add_argument("--projection_bits", type=int, default=8)
    parser.add_argument("--projection_seed", type=int, default=91_731)
    parser.add_argument("--query_chunk_size", type=int, default=256)
    parser.add_argument("--max_timestep_gap", type=int, default=5)
    parser.add_argument("--max_state_rmse", type=float, default=0.75)
    parser.add_argument("--min_return_gap", type=float, default=2_000.0)
    parser.add_argument("--min_action_rmse", type=float, default=0.25)
    parser.add_argument("--preference_margin", type=float, default=0.05)
    parser.add_argument("--target_return", type=float, default=12_000.0)
    parser.add_argument(
        "--result_dir", default="results/neighborhood_pair_diagnostics"
    )
    parser.add_argument("--wandb_project", default="corl-ddr")
    parser.add_argument(
        "--wandb_group", default="Neighborhood-HardFork-pair-diagnostic"
    )
    parser.add_argument(
        "--wandb_name", default="NeighborhoodHardForkDiagnostic-HCMR-delayed-seed0"
    )
    parser.add_argument(
        "--wandb_mode", default="online", choices=("online", "offline", "disabled")
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def batched(indices: np.ndarray, batch_size: int) -> Iterable[np.ndarray]:
    for start in range(0, len(indices), batch_size):
        yield indices[start : start + batch_size]


def percentiles(values: np.ndarray) -> Dict[str, float]:
    if len(values) == 0:
        return {f"p{quantile}": float("nan") for quantile in (10, 50, 90)}
    return {
        f"p{quantile}": float(np.percentile(values, quantile))
        for quantile in (10, 50, 90)
    }


def finite_flatten(prefix: str, value: object) -> Dict[str, float]:
    flattened: Dict[str, float] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}/{key}" if prefix else key
            flattened.update(finite_flatten(child_prefix, child))
    elif isinstance(value, (bool, int, float)) and math.isfinite(float(value)):
        flattened[prefix] = float(value)
    return flattened


def build_model(
    checkpoint: dict, dataset: SequenceDataset, device: torch.device
) -> DecisionTransformer:
    config = checkpoint["config"]
    model = DecisionTransformer(
        state_dim=int(np.asarray(dataset.state_mean).reshape(-1).shape[0]),
        action_dim=int(dataset.dataset[0]["actions"].shape[-1]),
        embedding_dim=int(config["embedding_dim"]),
        seq_len=int(config["seq_len"]),
        episode_len=int(config["episode_len"]),
        num_layers=int(config["num_layers"]),
        num_heads=int(config["num_heads"]),
        attention_dropout=float(config["attention_dropout"]),
        residual_dropout=float(config["residual_dropout"]),
        embedding_dropout=float(config["embedding_dropout"]),
        max_action=float(config["max_action"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def flatten_dataset(
    dataset: SequenceDataset,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Flatten normalized states and metadata while retaining trajectory identity."""

    state_parts = []
    action_parts = []
    outcome_parts = []
    trajectory_parts = []
    timestep_parts = []
    for trajectory_id, trajectory in enumerate(dataset.dataset):
        length = trajectory["observations"].shape[0]
        state_parts.append(
            ((trajectory["observations"] - dataset.state_mean) / dataset.state_std).astype(
                np.float32
            )
        )
        action_parts.append(trajectory["actions"].astype(np.float32))
        outcome_parts.append(
            np.full(length, trajectory["returns"][0], dtype=np.float32)
        )
        trajectory_parts.append(np.full(length, trajectory_id, dtype=np.int32))
        timestep_parts.append(np.arange(length, dtype=np.int32))

    return (
        np.concatenate(state_parts, axis=0),
        np.concatenate(action_parts, axis=0),
        np.concatenate(outcome_parts, axis=0),
        np.concatenate(trajectory_parts, axis=0),
        np.concatenate(timestep_parts, axis=0),
    )


def build_neighborhood_pairs(
    states: np.ndarray,
    actions: np.ndarray,
    outcomes: np.ndarray,
    trajectory_ids: np.ndarray,
    timesteps: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, Dict[str, float], Dict[str, np.ndarray]]:
    """Build positive/negative local action pairs without using a learned value."""

    projection_rng = np.random.default_rng(args.projection_seed)
    projections = projection_rng.standard_normal(
        (states.shape[1], args.projection_bits)
    ).astype(np.float32)
    signs = (states @ projections) >= 0.0
    bit_values = 1 << np.arange(args.projection_bits, dtype=np.int64)
    bucket_codes = signs.astype(np.int64) @ bit_values
    sorted_indices = np.argsort(bucket_codes, kind="stable")
    sorted_codes = bucket_codes[sorted_indices]
    bucket_starts = np.r_[0, np.flatnonzero(np.diff(sorted_codes)) + 1]
    bucket_ends = np.r_[bucket_starts[1:], len(states)]

    pair_keys: List[np.ndarray] = []
    distance_parts: List[np.ndarray] = []
    return_gap_parts: List[np.ndarray] = []
    action_gap_parts: List[np.ndarray] = []
    timestep_gap_parts: List[np.ndarray] = []
    anchors_with_time_neighbor = 0
    raw_time_neighbors = 0
    retrieved_time_neighbors = 0
    state_close_neighbors = 0
    state_and_return_neighbors = 0
    state_return_and_action_neighbors = 0

    for bucket_start, bucket_end in zip(bucket_starts, bucket_ends):
        bucket_indices = sorted_indices[bucket_start:bucket_end]
        candidate_states = states[bucket_indices]
        candidate_norms = np.sum(candidate_states * candidate_states, axis=1)
        candidate_trajectories = trajectory_ids[bucket_indices]
        candidate_timesteps = timesteps[bucket_indices]

        for query_start in range(0, len(bucket_indices), args.query_chunk_size):
            query_end = min(
                query_start + args.query_chunk_size, len(bucket_indices)
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
            invalid = (
                candidate_trajectories[None, :] == trajectory_ids[query_indices, None]
            ) | (
                np.abs(candidate_timesteps[None, :] - timesteps[query_indices, None])
                > args.max_timestep_gap
            )
            squared_distances[invalid] = np.inf
            available = np.isfinite(squared_distances).sum(axis=1)
            anchors_with_time_neighbor += int((available > 0).sum())
            raw_time_neighbors += int(available.sum())
            keep_count = min(args.neighbor_count, squared_distances.shape[1])
            nearest_columns = np.argpartition(
                squared_distances, kth=keep_count - 1, axis=1
            )[:, :keep_count]
            nearest_distances = np.take_along_axis(
                squared_distances, nearest_columns, axis=1
            )
            valid_neighbor = np.isfinite(nearest_distances)
            retrieved_time_neighbors += int(valid_neighbor.sum())
            row_indices = np.broadcast_to(
                query_indices[:, None], nearest_columns.shape
            )[valid_neighbor]
            neighbor_indices = bucket_indices[nearest_columns[valid_neighbor]]
            state_rmse = np.sqrt(
                nearest_distances[valid_neighbor] / states.shape[1]
            )
            return_gap = np.abs(outcomes[row_indices] - outcomes[neighbor_indices])
            action_rmse = np.sqrt(
                np.square(actions[row_indices] - actions[neighbor_indices]).mean(axis=1)
            )
            state_close = state_rmse <= args.max_state_rmse
            return_different = return_gap >= args.min_return_gap
            action_different = action_rmse >= args.min_action_rmse
            state_close_neighbors += int(state_close.sum())
            state_and_return_neighbors += int(
                (state_close & return_different).sum()
            )
            keep = state_close & return_different & action_different
            state_return_and_action_neighbors += int(keep.sum())
            if not np.any(keep):
                continue

            first = row_indices[keep]
            second = neighbor_indices[keep]
            positive_is_first = outcomes[first] > outcomes[second]
            positive = np.where(positive_is_first, first, second)
            negative = np.where(positive_is_first, second, first)
            pair_keys.append(np.stack([positive, negative], axis=1).astype(np.int32))
            distance_parts.append(state_rmse[keep].astype(np.float32))
            return_gap_parts.append(return_gap[keep].astype(np.float32))
            action_gap_parts.append(action_rmse[keep].astype(np.float32))
            timestep_gap_parts.append(
                np.abs(timesteps[first] - timesteps[second]).astype(np.float32)
            )

    if not pair_keys:
        raise RuntimeError("No pair passed the state, return, and action filters")
    pairs = np.concatenate(pair_keys, axis=0)
    state_rmse = np.concatenate(distance_parts, axis=0)
    return_gap = np.concatenate(return_gap_parts, axis=0)
    action_rmse = np.concatenate(action_gap_parts, axis=0)
    timestep_gap = np.concatenate(timestep_gap_parts, axis=0)

    pair_hash = pairs[:, 0].astype(np.int64) * len(states) + pairs[:, 1]
    _, unique_indices = np.unique(pair_hash, return_index=True)
    unique_indices.sort()
    pairs = pairs[unique_indices]
    diagnostics = {
        "state_rmse": state_rmse[unique_indices],
        "return_gap": return_gap[unique_indices],
        "action_rmse": action_rmse[unique_indices],
        "timestep_gap": timestep_gap[unique_indices],
    }
    stats = {
        "num_transitions": float(len(states)),
        "projection_bucket_count": float(len(bucket_starts)),
        "projection_bucket_max_size": float(
            np.diff(np.r_[bucket_starts, len(states)]).max()
        ),
        "anchors_with_time_neighbor": float(anchors_with_time_neighbor),
        "anchors_with_time_neighbor_fraction": float(
            anchors_with_time_neighbor / len(states)
        ),
        "raw_cross_trajectory_time_neighbors": float(raw_time_neighbors),
        "retrieved_time_neighbors": float(retrieved_time_neighbors),
        "state_close_neighbors": float(state_close_neighbors),
        "state_and_return_neighbors": float(state_and_return_neighbors),
        "state_return_and_action_neighbors": float(
            state_return_and_action_neighbors
        ),
        "raw_filtered_pairs_before_dedup": float(len(pair_hash)),
        "unique_pairs": float(len(pairs)),
    }
    return pairs, stats, diagnostics


def make_context(
    dataset: SequenceDataset, trajectory_id: int, timestep: int, target_return: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Build a DT context from the preferred trajectory at one decision point."""

    trajectory = dataset.dataset[trajectory_id]
    start = max(0, timestep - dataset.seq_len + 1)
    stop = timestep + 1
    states = trajectory["observations"][start:stop].copy()
    actions = trajectory["actions"][start:stop].copy()
    valid_length = stop - start
    time_steps = np.arange(start, start + dataset.seq_len, dtype=np.int64)
    mask = np.concatenate(
        [
            np.ones(valid_length, dtype=np.float32),
            np.zeros(dataset.seq_len - valid_length, dtype=np.float32),
        ]
    )
    states = (states - dataset.state_mean) / dataset.state_std
    returns = np.full(valid_length, target_return * dataset.reward_scale, dtype=np.float32)
    if valid_length < dataset.seq_len:
        states = np.concatenate(
            [
                states,
                np.zeros(
                    (dataset.seq_len - valid_length, states.shape[-1]), dtype=np.float32
                ),
            ],
            axis=0,
        )
        actions = np.concatenate(
            [
                actions,
                np.zeros(
                    (dataset.seq_len - valid_length, actions.shape[-1]), dtype=np.float32
                ),
            ],
            axis=0,
        )
        returns = np.concatenate(
            [returns, np.zeros(dataset.seq_len - valid_length, dtype=np.float32)]
        )
    return states, actions, returns, time_steps, mask, valid_length - 1


@torch.no_grad()
def score_pairs(
    model: DecisionTransformer,
    dataset: SequenceDataset,
    pairs: np.ndarray,
    trajectory_ids: np.ndarray,
    timesteps: np.ndarray,
    actions: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, np.ndarray]:
    """Score positive versus negative actions under v3-high's RTG=12000."""

    positive_errors = np.empty(len(pairs), dtype=np.float32)
    negative_errors = np.empty(len(pairs), dtype=np.float32)
    pair_indices = np.arange(len(pairs))
    for batch_indices in batched(pair_indices, args.batch_size):
        contexts = [
            make_context(
                dataset=dataset,
                trajectory_id=int(trajectory_ids[pairs[index, 0]]),
                timestep=int(timesteps[pairs[index, 0]]),
                target_return=args.target_return,
            )
            for index in batch_indices
        ]
        states = torch.as_tensor(
            np.stack([context[0] for context in contexts]),
            dtype=torch.float32,
            device=args.device,
        )
        action_context = torch.as_tensor(
            np.stack([context[1] for context in contexts]),
            dtype=torch.float32,
            device=args.device,
        )
        returns = torch.as_tensor(
            np.stack([context[2] for context in contexts]),
            dtype=torch.float32,
            device=args.device,
        )
        time_steps = torch.as_tensor(
            np.stack([context[3] for context in contexts]),
            dtype=torch.long,
            device=args.device,
        )
        mask = torch.as_tensor(
            np.stack([context[4] for context in contexts]),
            dtype=torch.float32,
            device=args.device,
        )
        target_index = torch.as_tensor(
            [context[5] for context in contexts], dtype=torch.long, device=args.device
        )
        predictions = model(
            states=states,
            actions=action_context,
            returns_to_go=returns,
            time_steps=time_steps,
            padding_mask=~mask.to(torch.bool),
        )
        row = torch.arange(len(batch_indices), device=args.device)
        prediction = predictions[row, target_index]
        positive_actions = torch.as_tensor(
            actions[pairs[batch_indices, 0]], dtype=torch.float32, device=args.device
        )
        negative_actions = torch.as_tensor(
            actions[pairs[batch_indices, 1]], dtype=torch.float32, device=args.device
        )
        positive_errors[batch_indices] = (
            F.mse_loss(prediction, positive_actions, reduction="none")
            .mean(dim=-1)
            .cpu()
            .numpy()
        )
        negative_errors[batch_indices] = (
            F.mse_loss(prediction, negative_actions, reduction="none")
            .mean(dim=-1)
            .cpu()
            .numpy()
        )
    return positive_errors, negative_errors


def export_margin_active_pairs(
    result_dir: Path,
    result_stem: str,
    pairs: np.ndarray,
    active: np.ndarray,
    trajectory_ids: np.ndarray,
    timesteps: np.ndarray,
    positive_error: np.ndarray,
    negative_error: np.ndarray,
    margin_violation: np.ndarray,
    pair_diagnostics: Dict[str, np.ndarray],
    checkpoint_sha256: str,
    args: argparse.Namespace,
) -> Tuple[Path, Dict[str, float]]:
    """Export frozen-margin-active pairs in hard_fork_dt's legacy NPZ schema."""

    active_indices = np.flatnonzero(active)
    if len(active_indices) == 0:
        raise RuntimeError("No frozen-margin-active pair is available for export")

    active_flat_pairs = pairs[active_indices]
    hard_fork_pairs = np.stack(
        [
            trajectory_ids[active_flat_pairs[:, 0]],
            timesteps[active_flat_pairs[:, 0]],
            trajectory_ids[active_flat_pairs[:, 1]],
            timesteps[active_flat_pairs[:, 1]],
        ],
        axis=1,
    ).astype(np.int32)
    active_state_rmse = pair_diagnostics["state_rmse"][active_indices]
    confidence_scale = max(float(np.median(active_state_rmse)), 1e-6)
    pair_confidence = np.exp(-active_state_rmse / confidence_scale).astype(
        np.float32
    )
    output_path = result_dir / f"{result_stem}_margin_active_pairs.npz"
    np.savez_compressed(
        output_path,
        pairs=hard_fork_pairs,
        valid_branch=np.ones(len(active_indices), dtype=bool),
        hard_pair=np.ones(len(active_indices), dtype=bool),
        pair_confidence=pair_confidence,
        positive_error=positive_error[active_indices],
        negative_error=negative_error[active_indices],
        source_margin_violation=margin_violation[active_indices],
        source_flat_pair_indices=active_indices.astype(np.int32),
        source_state_rmse=active_state_rmse,
        source_return_gap=pair_diagnostics["return_gap"][active_indices],
        source_action_rmse=pair_diagnostics["action_rmse"][active_indices],
        source_timestep_gap=pair_diagnostics["timestep_gap"][active_indices],
        source_checkpoint_sha256=np.asarray(checkpoint_sha256),
        source_target_return=np.asarray(args.target_return, dtype=np.float32),
        source_preference_margin=np.asarray(
            args.preference_margin, dtype=np.float32
        ),
    )
    stats = {
        "exported_margin_active_pairs": float(len(active_indices)),
        "confidence_scale_state_rmse": confidence_scale,
        "pair_confidence_mean": float(pair_confidence.mean()),
        "pair_confidence_min": float(pair_confidence.min()),
        "pair_confidence_max": float(pair_confidence.max()),
    }
    return output_path, stats


def main() -> None:
    args = parse_args()
    if args.reward_mode != "delayed":
        raise ValueError("this diagnostic is defined for delayed-reward DT")
    if args.neighbor_count < 1:
        raise ValueError("neighbor_count must be positive")
    if not 1 <= args.projection_bits <= 16:
        raise ValueError("projection_bits must be in [1, 16]")
    if args.query_chunk_size < 1:
        raise ValueError("query_chunk_size must be positive")
    if args.max_timestep_gap < 0:
        raise ValueError("max_timestep_gap must be non-negative")
    if args.max_state_rmse <= 0.0:
        raise ValueError("max_state_rmse must be positive")
    if args.min_return_gap <= 0.0 or args.min_action_rmse <= 0.0:
        raise ValueError("return and action gaps must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {args.device}")

    set_seed(args.seed)
    device = torch.device(args.device)
    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    dataset = SequenceDataset(
        args.env_name,
        seq_len=int(checkpoint["config"]["seq_len"]),
        reward_scale=float(checkpoint["config"]["reward_scale"]),
        reward_mode=args.reward_mode,
    )
    model = build_model(checkpoint, dataset, device)
    states, actions, outcomes, trajectory_ids, timesteps = flatten_dataset(dataset)
    pairs, pair_stats, pair_diagnostics = build_neighborhood_pairs(
        states=states,
        actions=actions,
        outcomes=outcomes,
        trajectory_ids=trajectory_ids,
        timesteps=timesteps,
        args=args,
    )
    positive_error, negative_error = score_pairs(
        model=model,
        dataset=dataset,
        pairs=pairs,
        trajectory_ids=trajectory_ids,
        timesteps=timesteps,
        actions=actions,
        args=args,
    )
    margin_violation = args.preference_margin + positive_error - negative_error
    active = margin_violation > 0.0
    wrong = positive_error >= negative_error
    clear_correct = margin_violation <= 0.0

    result_dir = Path(args.result_dir).resolve()
    result_dir.mkdir(parents=True, exist_ok=True)
    result_stem = "neighborhood_hard_fork_diagnostic_hcmr_delayed_seed" f"{args.seed}"
    checkpoint_sha256 = sha256(checkpoint_path)
    active_pairs_path, active_pair_stats = export_margin_active_pairs(
        result_dir=result_dir,
        result_stem=result_stem,
        pairs=pairs,
        active=active,
        trajectory_ids=trajectory_ids,
        timesteps=timesteps,
        positive_error=positive_error,
        negative_error=negative_error,
        margin_violation=margin_violation,
        pair_diagnostics=pair_diagnostics,
        checkpoint_sha256=checkpoint_sha256,
        args=args,
    )

    summary = {
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_sha256,
            "next_step": int(checkpoint.get("next_step", -1)),
        },
        "dataset": {
            "env_name": args.env_name,
            "reward_mode": args.reward_mode,
            "num_trajectories": len(dataset.dataset),
            "num_transitions": len(states),
        },
        "thresholds": {
            "neighbor_count": args.neighbor_count,
            "max_timestep_gap": args.max_timestep_gap,
            "max_state_rmse": args.max_state_rmse,
            "min_return_gap": args.min_return_gap,
            "min_action_rmse": args.min_action_rmse,
            "preference_margin": args.preference_margin,
            "target_return": args.target_return,
        },
        "candidate_generation": pair_stats,
        "pair_distribution": {
            "state_rmse": percentiles(pair_diagnostics["state_rmse"]),
            "return_gap": percentiles(pair_diagnostics["return_gap"]),
            "action_rmse": percentiles(pair_diagnostics["action_rmse"]),
            "timestep_gap": percentiles(pair_diagnostics["timestep_gap"]),
        },
        "frozen_dt_high_target": {
            "total_unique_pairs": int(len(pairs)),
            "margin_active_count": int(active.sum()),
            "margin_active_ratio": float(active.mean()),
            "wrong_count": int(wrong.sum()),
            "wrong_ratio": float(wrong.mean()),
            "clear_correct_count": int(clear_correct.sum()),
            "clear_correct_ratio": float(clear_correct.mean()),
            "positive_error": percentiles(positive_error),
            "negative_error": percentiles(negative_error),
            "margin_violation": percentiles(margin_violation),
        },
        "hard_fork_margin_active_export": {
            "path": str(active_pairs_path),
            **active_pair_stats,
        },
    }

    summary_path = result_dir / f"{result_stem}.json"
    arrays_path = result_dir / f"{result_stem}.npz"
    with summary_path.open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2, sort_keys=True)
    np.savez_compressed(
        arrays_path,
        pairs=pairs,
        positive_error=positive_error,
        negative_error=negative_error,
        margin_violation=margin_violation,
        active=active,
        wrong=wrong,
        **pair_diagnostics,
    )

    run = wandb.init(
        project=args.wandb_project,
        group=args.wandb_group,
        name=args.wandb_name,
        config={**vars(args), "checkpoint_sha256": summary["checkpoint"]["sha256"]},
        mode=args.wandb_mode,
    )
    wandb.log(finite_flatten("diagnostic", summary), step=0)
    if run is not None:
        run.summary["result_json"] = str(summary_path)
        run.summary["result_npz"] = str(arrays_path)
    wandb.finish()
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
