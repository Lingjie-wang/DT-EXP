"""Diagnose whether frozen DT has useful unresolved state-aligned preferences.

This script never updates model parameters.  It builds the same good/bad pairs
as the one-sided SAP-DT experiment, measures whether a frozen DT already ranks
the preferred action ahead of the negative action, and then counts the smaller
set of pairs that resemble a local trajectory fork: similar recent history,
different current actions, different outcomes, and an uncertain or wrong DT.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from dt import DecisionTransformer, SequenceDataset, set_seed
from sap_dt_one_sided import predict_action, StateAlignedPreferenceDataset

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--env_name", default="halfcheetah-medium-replay-v2")
    parser.add_argument("--reward_mode", default="delayed")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_pairs", type=int, default=100_000)
    parser.add_argument("--num_candidates", type=int, default=64)
    parser.add_argument("--state_keep_fraction", type=float, default=0.25)
    parser.add_argument("--good_quantile", type=float, default=0.70)
    parser.add_argument("--bad_quantile", type=float, default=0.30)
    parser.add_argument("--timestep_bucket", type=int, default=25)
    parser.add_argument("--prefix_length", type=int, default=5)
    parser.add_argument("--min_prefix_steps", type=int, default=5)
    parser.add_argument("--prefix_state_threshold", type=float, default=0.75)
    parser.add_argument("--prefix_action_threshold", type=float, default=0.50)
    parser.add_argument("--max_timestep_gap", type=int, default=5)
    parser.add_argument("--action_difference_threshold", type=float, default=0.25)
    parser.add_argument("--min_return_gap", type=float, default=2_000.0)
    parser.add_argument("--preference_margin", type=float, default=0.05)
    parser.add_argument("--result_dir", default="results/pair_diagnostics")
    parser.add_argument("--wandb_project", default="corl-ddr")
    parser.add_argument("--wandb_group", default="SAP-DT-pair-diagnostic")
    parser.add_argument("--wandb_name", default="DTPairDiagnostic-HCMR-delayed-seed0")
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


def percentiles(values: np.ndarray) -> Dict[str, float]:
    return {
        f"p{quantile}": float(np.percentile(values, quantile))
        for quantile in (10, 25, 50, 75, 90)
    }


def ratio(mask: np.ndarray) -> float:
    return float(mask.mean()) if len(mask) else float("nan")


def count(mask: np.ndarray) -> int:
    return int(mask.sum())


def flatten(prefix: str, value: object) -> Dict[str, float]:
    flattened: Dict[str, float] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            flattened.update(flatten(f"{prefix}/{key}" if prefix else key, child))
    elif isinstance(value, (bool, int, float)) and math.isfinite(float(value)):
        flattened[prefix] = float(value)
    return flattened


def build_model(checkpoint: dict, dataset: SequenceDataset, device: torch.device):
    config = checkpoint["config"]
    state_dim = int(np.asarray(dataset.state_mean).reshape(-1).shape[0])
    action_dim = int(dataset.dataset[0]["actions"].shape[-1])
    model = DecisionTransformer(
        state_dim=state_dim,
        action_dim=action_dim,
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


def batched(indices: np.ndarray, batch_size: int) -> Iterable[np.ndarray]:
    for start in range(0, len(indices), batch_size):
        yield indices[start : start + batch_size]


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {args.device}")
    device = torch.device(args.device)
    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_config = checkpoint["config"]

    dataset = SequenceDataset(
        args.env_name,
        seq_len=int(checkpoint_config["seq_len"]),
        reward_scale=float(checkpoint_config["reward_scale"]),
        reward_mode=args.reward_mode,
    )
    pair_dataset = StateAlignedPreferenceDataset(
        sequence_dataset=dataset,
        state_keep_fraction=args.state_keep_fraction,
        good_quantile=args.good_quantile,
        bad_quantile=args.bad_quantile,
        timestep_bucket=args.timestep_bucket,
        num_pairs=args.num_pairs,
        num_candidates=args.num_candidates,
        pair_seed=args.seed,
    )
    model = build_model(checkpoint, dataset, device)

    pairs = pair_dataset.pairs
    trajectories = pair_dataset.trajectories
    state_mean = np.asarray(pair_dataset.state_mean).reshape(-1)
    state_std = np.asarray(pair_dataset.state_std).reshape(-1)
    trajectory_returns = np.asarray(
        [float(trajectory["returns"][0]) for trajectory in trajectories]
    )

    n_pairs = len(pairs)
    current_state_rmse = np.empty(n_pairs, dtype=np.float32)
    prefix_state_rmse = np.empty(n_pairs, dtype=np.float32)
    prefix_action_rmse = np.full(n_pairs, np.inf, dtype=np.float32)
    target_action_rmse = np.empty(n_pairs, dtype=np.float32)
    return_gap = np.empty(n_pairs, dtype=np.float32)
    timestep_gap = np.empty(n_pairs, dtype=np.int32)
    prefix_steps = np.empty(n_pairs, dtype=np.int32)

    for pair_index, (good_traj, good_step, bad_traj, bad_step) in enumerate(pairs):
        good_trajectory = trajectories[int(good_traj)]
        bad_trajectory = trajectories[int(bad_traj)]
        good_step, bad_step = int(good_step), int(bad_step)

        good_state = (
            good_trajectory["observations"][good_step] - state_mean
        ) / state_std
        bad_state = (
            bad_trajectory["observations"][bad_step] - state_mean
        ) / state_std
        current_state_rmse[pair_index] = np.sqrt(
            np.square(good_state - bad_state).mean()
        )

        state_steps = min(args.prefix_length, good_step + 1, bad_step + 1)
        good_states = good_trajectory["observations"][
            good_step - state_steps + 1 : good_step + 1
        ]
        bad_states = bad_trajectory["observations"][
            bad_step - state_steps + 1 : bad_step + 1
        ]
        good_states = (good_states - state_mean) / state_std
        bad_states = (bad_states - state_mean) / state_std
        prefix_state_rmse[pair_index] = np.sqrt(
            np.square(good_states - bad_states).mean()
        )
        prefix_steps[pair_index] = state_steps

        action_steps = min(max(args.prefix_length - 1, 0), good_step, bad_step)
        if action_steps:
            good_actions = good_trajectory["actions"][
                good_step - action_steps : good_step
            ]
            bad_actions = bad_trajectory["actions"][
                bad_step - action_steps : bad_step
            ]
            prefix_action_rmse[pair_index] = np.sqrt(
                np.square(good_actions - bad_actions).mean()
            )

        good_action = good_trajectory["actions"][good_step]
        bad_action = bad_trajectory["actions"][bad_step]
        target_action_rmse[pair_index] = np.sqrt(
            np.square(good_action - bad_action).mean()
        )
        return_gap[pair_index] = (
            trajectory_returns[int(good_traj)] - trajectory_returns[int(bad_traj)]
        )
        timestep_gap[pair_index] = abs(good_step - bad_step)

    positive_error = np.empty(n_pairs, dtype=np.float32)
    negative_error = np.empty(n_pairs, dtype=np.float32)
    all_indices = np.arange(n_pairs)
    with torch.no_grad():
        for batch_indices in batched(all_indices, args.batch_size):
            contexts: List[tuple] = []
            negative_actions: List[np.ndarray] = []
            for pair_index in batch_indices:
                good_traj, good_step, bad_traj, bad_step = pairs[int(pair_index)]
                contexts.append(pair_dataset._context(int(good_traj), int(good_step)))
                negative_actions.append(
                    trajectories[int(bad_traj)]["actions"][int(bad_step)].copy()
                )

            states = torch.as_tensor(
                np.stack([context[0] for context in contexts]),
                dtype=torch.float32,
                device=device,
            )
            actions = torch.as_tensor(
                np.stack([context[1] for context in contexts]),
                dtype=torch.float32,
                device=device,
            )
            returns = torch.as_tensor(
                np.stack([context[2] for context in contexts]),
                dtype=torch.float32,
                device=device,
            )
            time_steps = torch.as_tensor(
                np.stack([context[3] for context in contexts]),
                dtype=torch.long,
                device=device,
            )
            mask = torch.as_tensor(
                np.stack([context[4] for context in contexts]),
                dtype=torch.float32,
                device=device,
            )
            target_index = torch.as_tensor(
                [context[5] for context in contexts], dtype=torch.long, device=device
            )
            negative_action = torch.as_tensor(
                np.stack(negative_actions), dtype=torch.float32, device=device
            )

            prediction = predict_action(
                model, states, actions, returns, time_steps, mask, target_index
            )
            row = torch.arange(len(batch_indices), device=device)
            positive_action = actions[row, target_index]
            positive_error[batch_indices] = (
                F.mse_loss(prediction, positive_action, reduction="none")
                .mean(dim=-1)
                .cpu()
                .numpy()
            )
            negative_error[batch_indices] = (
                F.mse_loss(prediction, negative_action, reduction="none")
                .mean(dim=-1)
                .cpu()
                .numpy()
            )

    clear_correct = positive_error + args.preference_margin < negative_error
    uncertain_correct = (positive_error < negative_error) & ~clear_correct
    wrong = positive_error >= negative_error

    prefix_close = (
        (prefix_steps >= args.min_prefix_steps)
        & (prefix_state_rmse <= args.prefix_state_threshold)
        & (prefix_action_rmse <= args.prefix_action_threshold)
        & (timestep_gap <= args.max_timestep_gap)
    )
    action_different = target_action_rmse >= args.action_difference_threshold
    outcome_different = return_gap >= args.min_return_gap
    valid_branch = prefix_close & action_different & outcome_different
    hard_pair = valid_branch & ~clear_correct
    hard_uncertain = valid_branch & uncertain_correct
    hard_wrong = valid_branch & wrong

    sensitivity = {}
    for state_threshold in (0.50, 0.75, 1.00):
        close = (
            (prefix_steps >= args.min_prefix_steps)
            & (prefix_state_rmse <= state_threshold)
            & (prefix_action_rmse <= args.prefix_action_threshold)
            & (timestep_gap <= args.max_timestep_gap)
        )
        branch = close & action_different & outcome_different
        sensitivity[f"prefix_state_le_{state_threshold:.2f}"] = {
            "valid_branch_pairs": count(branch),
            "hard_pairs": count(branch & ~clear_correct),
            "wrong_pairs": count(branch & wrong),
        }

    summary = {
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256(checkpoint_path),
            "next_step": int(checkpoint.get("next_step", -1)),
        },
        "dataset": {
            "env_name": args.env_name,
            "reward_mode": args.reward_mode,
            "trajectory_count": len(trajectories),
        },
        "pair_generation": pair_dataset.stats,
        "thresholds": {
            "preference_margin": args.preference_margin,
            "prefix_length": args.prefix_length,
            "min_prefix_steps": args.min_prefix_steps,
            "prefix_state_rmse_max": args.prefix_state_threshold,
            "prefix_action_rmse_max": args.prefix_action_threshold,
            "max_timestep_gap": args.max_timestep_gap,
            "target_action_rmse_min": args.action_difference_threshold,
            "return_gap_min": args.min_return_gap,
        },
        "dt_preference": {
            "total_pairs": n_pairs,
            "satisfied_count": count(positive_error < negative_error),
            "satisfied_ratio": ratio(positive_error < negative_error),
            "clear_correct_count": count(clear_correct),
            "clear_correct_ratio": ratio(clear_correct),
            "uncertain_correct_count": count(uncertain_correct),
            "uncertain_correct_ratio": ratio(uncertain_correct),
            "wrong_count": count(wrong),
            "wrong_ratio": ratio(wrong),
            "positive_error": percentiles(positive_error),
            "negative_error": percentiles(negative_error),
            "signed_margin_negative_minus_positive": percentiles(
                negative_error - positive_error
            ),
        },
        "fork_filter": {
            "prefix_close_count": count(prefix_close),
            "prefix_close_ratio": ratio(prefix_close),
            "action_different_count": count(action_different),
            "outcome_different_count": count(outcome_different),
            "valid_branch_count": count(valid_branch),
            "valid_branch_ratio": ratio(valid_branch),
            "hard_pair_count": count(hard_pair),
            "hard_pair_ratio_all": ratio(hard_pair),
            "hard_pair_ratio_within_valid": (
                float(hard_pair.sum() / valid_branch.sum())
                if valid_branch.any()
                else float("nan")
            ),
            "hard_uncertain_count": count(hard_uncertain),
            "hard_wrong_count": count(hard_wrong),
        },
        "distributions": {
            "current_state_rmse": percentiles(current_state_rmse),
            "prefix_state_rmse": percentiles(prefix_state_rmse),
            "prefix_action_rmse": percentiles(
                prefix_action_rmse[np.isfinite(prefix_action_rmse)]
            ),
            "target_action_rmse": percentiles(target_action_rmse),
            "return_gap": percentiles(return_gap),
            "timestep_gap": percentiles(timestep_gap.astype(np.float32)),
        },
        "sensitivity": sensitivity,
    }

    result_dir = Path(args.result_dir).resolve()
    result_dir.mkdir(parents=True, exist_ok=True)
    result_stem = (
        "dt_pair_diagnostic_halfcheetah_medium_replay_delayed_"
        f"seed{args.seed}"
    )
    result_path = result_dir / f"{result_stem}.json"
    arrays_path = result_dir / f"{result_stem}.npz"
    with result_path.open("w", encoding="utf-8") as result_file:
        json.dump(summary, result_file, indent=2, sort_keys=True)
    np.savez_compressed(
        arrays_path,
        pairs=pairs,
        pair_confidence=pair_dataset.pair_confidences,
        current_state_rmse=current_state_rmse,
        prefix_state_rmse=prefix_state_rmse,
        prefix_action_rmse=prefix_action_rmse,
        target_action_rmse=target_action_rmse,
        return_gap=return_gap,
        timestep_gap=timestep_gap,
        prefix_steps=prefix_steps,
        positive_error=positive_error,
        negative_error=negative_error,
        clear_correct=clear_correct,
        uncertain_correct=uncertain_correct,
        wrong=wrong,
        valid_branch=valid_branch,
        hard_pair=hard_pair,
    )

    run = wandb.init(
        project=args.wandb_project,
        group=args.wandb_group,
        name=args.wandb_name,
        config={**vars(args), "checkpoint_sha256": summary["checkpoint"]["sha256"]},
        mode=args.wandb_mode,
    )
    flat_summary = flatten("diagnostic", summary)
    wandb.log(flat_summary, step=0)
    if run is not None:
        for key, value in flat_summary.items():
            run.summary[key] = value
        wandb.save(str(result_path), base_path=str(result_dir))
        wandb.save(str(arrays_path), base_path=str(result_dir))
        run.finish()

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Results: {result_path}")
    print(f"Pair arrays: {arrays_path}")


if __name__ == "__main__":
    main()
