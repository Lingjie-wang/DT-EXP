"""Oracle one-action branching diagnostic for delayed-reward Decision Transformer.

This is a privileged diagnostic, not an offline-RL algorithm.  At selected states
from a normal DT rollout, it restores the exact MuJoCo/policy state, executes one
candidate action, and then lets the same frozen DT act until the episode ends.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import d4rl  # noqa: F401
import gym
import numpy as np
import torch
from dt import DecisionTransformer, get_d4rl_dataset, validate_reward_mode, wrap_env
from tqdm.auto import tqdm

@dataclass
class EnvSnapshot:
    sim_state: Any
    sim_data_state: Dict[str, np.ndarray]
    wrapper_elapsed_steps: List[Tuple[int, int]]
    env_rng_state: Any
    action_rng_state: Any


@dataclass
class PolicySnapshot:
    states: torch.Tensor
    actions: torch.Tensor
    returns: torch.Tensor
    prefix_scaled_return: float
    step: int


@dataclass
class BranchResult:
    full_raw_return: float
    suffix_raw_return: float
    normalized_score: float
    episode_length: int
    first_next_state: np.ndarray
    first_reward_scaled: float
    first_done: bool


SIM_DATA_STATE_FIELDS = (
    # MjSimState omits solver warm-start and applied-force arrays.  Leaving the
    # terminal branch values in these buffers can make two restored rollouts
    # diverge after hundreds of chaotic MuJoCo steps.
    "qacc_warmstart",
    "qacc",
    "act_dot",
    "ctrl",
    "qfrc_applied",
    "xfrc_applied",
    "mocap_pos",
    "mocap_quat",
    "userdata",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--target-return", type=float, default=12000.0)
    parser.add_argument(
        "--reward-mode", choices=("original", "delayed"), default="delayed"
    )
    parser.add_argument("--eval-seed", type=int, default=42)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--anchors-per-episode", type=int, default=10)
    parser.add_argument("--exclude-final-steps", type=int, default=50)
    parser.add_argument("--num-pairs", type=int, default=8)
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=(0.05, 0.10, 0.20, 0.30),
    )
    parser.add_argument("--knn-neighbors", type=int, default=64)
    parser.add_argument("--action-std-floor-frac", type=float, default=0.05)
    parser.add_argument("--candidate-seed", type=int, default=20260916)
    parser.add_argument("--dataset-chunk-size", type=int, default=65536)
    parser.add_argument("--bootstrap-reps", type=int, default=5000)
    parser.add_argument("--restore-state-tol", type=float, default=1e-8)
    parser.add_argument("--restore-return-tol", type=float, default=1e-5)
    parser.add_argument(
        "--repeat-baseline-check",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def iter_env_chain(env: gym.Env) -> Iterable[Tuple[int, gym.Env]]:
    depth = 0
    current = env
    while True:
        yield depth, current
        if not hasattr(current, "env"):
            break
        current = current.env
        depth += 1


def copy_rng_state(rng: Any) -> Any:
    if rng is None:
        return None
    if hasattr(rng, "get_state"):
        return ("random_state", copy.deepcopy(rng.get_state()))
    if hasattr(rng, "bit_generator"):
        return ("generator", copy.deepcopy(rng.bit_generator.state))
    return None


def restore_rng_state(rng: Any, state: Any) -> None:
    if rng is None or state is None:
        return
    kind, payload = state
    if kind == "random_state":
        rng.set_state(copy.deepcopy(payload))
    elif kind == "generator":
        rng.bit_generator.state = copy.deepcopy(payload)
    else:
        raise ValueError(f"Unknown RNG state kind: {kind}")


def capture_env_snapshot(env: gym.Env) -> EnvSnapshot:
    unwrapped = env.unwrapped
    if not hasattr(unwrapped, "sim"):
        raise RuntimeError("Expected a mujoco_py environment exposing env.unwrapped.sim")

    elapsed = []
    for depth, wrapper in iter_env_chain(env):
        if hasattr(wrapper, "_elapsed_steps"):
            elapsed.append((depth, int(wrapper._elapsed_steps)))

    sim_data_state = {}
    for name in SIM_DATA_STATE_FIELDS:
        if not hasattr(unwrapped.sim.data, name):
            continue
        value = getattr(unwrapped.sim.data, name)
        if value is not None:
            sim_data_state[name] = np.asarray(value).copy()
    return EnvSnapshot(
        sim_state=copy.deepcopy(unwrapped.sim.get_state()),
        sim_data_state=sim_data_state,
        wrapper_elapsed_steps=elapsed,
        env_rng_state=copy_rng_state(getattr(unwrapped, "np_random", None)),
        action_rng_state=copy_rng_state(getattr(env.action_space, "np_random", None)),
    )


def restore_env_snapshot(env: gym.Env, snapshot: EnvSnapshot) -> None:
    unwrapped = env.unwrapped
    unwrapped.sim.set_state(copy.deepcopy(snapshot.sim_state))
    unwrapped.sim.forward()
    # Restore after forward(): mj_forward recomputes several buffers and can
    # otherwise overwrite the anchor's solver warm-start state.
    for name, value in snapshot.sim_data_state.items():
        target = getattr(unwrapped.sim.data, name)
        if target is not None:
            target[...] = value

    wrapper_by_depth = dict(iter_env_chain(env))
    for depth, elapsed_steps in snapshot.wrapper_elapsed_steps:
        wrapper_by_depth[depth]._elapsed_steps = int(elapsed_steps)

    restore_rng_state(getattr(unwrapped, "np_random", None), snapshot.env_rng_state)
    restore_rng_state(
        getattr(env.action_space, "np_random", None), snapshot.action_rng_state
    )


def build_model(
    checkpoint: Dict[str, Any], device: str
) -> Tuple[DecisionTransformer, Dict[str, Any]]:
    config = dict(checkpoint["config"])
    state_mean = np.asarray(checkpoint["state_mean"], dtype=np.float32)
    np.asarray(checkpoint["state_std"], dtype=np.float32)
    state_dim = int(state_mean.shape[-1])

    action_dim = checkpoint["model_state"]["action_head.0.weight"].shape[0]
    model = DecisionTransformer(
        state_dim=state_dim,
        action_dim=int(action_dim),
        seq_len=int(config["seq_len"]),
        episode_len=int(config["episode_len"]),
        embedding_dim=int(config["embedding_dim"]),
        num_layers=int(config["num_layers"]),
        num_heads=int(config["num_heads"]),
        attention_dropout=float(config["attention_dropout"]),
        residual_dropout=float(config["residual_dropout"]),
        embedding_dropout=float(config["embedding_dropout"]),
        max_action=float(config["max_action"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, config


def allocate_policy_tensors(
    model: DecisionTransformer,
    target_return_scaled: float,
    first_state: np.ndarray,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    states = torch.zeros(
        1,
        model.episode_len + 1,
        model.state_dim,
        dtype=torch.float32,
        device=device,
    )
    actions = torch.zeros(
        1,
        model.episode_len,
        model.action_dim,
        dtype=torch.float32,
        device=device,
    )
    returns = torch.zeros(
        1,
        model.episode_len + 1,
        dtype=torch.float32,
        device=device,
    )
    time_steps = torch.arange(model.episode_len, dtype=torch.long, device=device)
    time_steps = time_steps.view(1, -1)
    states[:, 0] = torch.as_tensor(first_state, dtype=torch.float32, device=device)
    returns[:, 0] = float(target_return_scaled)
    return states, actions, returns, time_steps


@torch.inference_mode()
def predict_action(
    model: DecisionTransformer,
    states: torch.Tensor,
    actions: torch.Tensor,
    returns: torch.Tensor,
    time_steps: torch.Tensor,
    step: int,
) -> np.ndarray:
    predicted = model(
        states[:, : step + 1][:, -model.seq_len :],
        actions[:, : step + 1][:, -model.seq_len :],
        returns[:, : step + 1][:, -model.seq_len :],
        time_steps[:, : step + 1][:, -model.seq_len :],
    )
    return predicted[0, -1].detach().cpu().numpy().astype(np.float32, copy=True)


def update_policy_tensors(
    states: torch.Tensor,
    actions: torch.Tensor,
    returns: torch.Tensor,
    step: int,
    action: np.ndarray,
    next_state: np.ndarray,
    reward_scaled: float,
    reward_mode: str,
) -> None:
    actions[:, step] = torch.as_tensor(
        action, dtype=torch.float32, device=actions.device
    )
    states[:, step + 1] = torch.as_tensor(
        next_state, dtype=torch.float32, device=states.device
    )
    conditioning_reward = reward_scaled if reward_mode == "original" else 0.0
    returns[:, step + 1] = returns[:, step] - float(conditioning_reward)


def normalized_score(env: gym.Env, raw_return: float) -> float:
    score = env.get_normalized_score(float(raw_return)) * 100.0
    return float(np.asarray(score))


@torch.inference_mode()
def branch_rollout(
    *,
    env: gym.Env,
    env_snapshot: EnvSnapshot,
    policy_snapshot: PolicySnapshot,
    first_action: np.ndarray,
    model: DecisionTransformer,
    time_steps: torch.Tensor,
    reward_scale: float,
    reward_mode: str,
) -> BranchResult:
    restore_env_snapshot(env, env_snapshot)
    states = policy_snapshot.states.clone()
    actions = policy_snapshot.actions.clone()
    returns = policy_snapshot.returns.clone()
    step = policy_snapshot.step

    next_state, reward_scaled, done, _ = env.step(first_action)
    first_next_state = np.asarray(next_state, dtype=np.float64).copy()
    first_reward_scaled = float(reward_scaled)
    first_done = bool(done)
    update_policy_tensors(
        states,
        actions,
        returns,
        step,
        first_action,
        next_state,
        reward_scaled,
        reward_mode,
    )
    suffix_scaled_return = float(reward_scaled)
    final_step = step + 1

    if not done:
        for continuation_step in range(step + 1, model.episode_len):
            action = predict_action(
                model, states, actions, returns, time_steps, continuation_step
            )
            next_state, reward_scaled, done, _ = env.step(action)
            update_policy_tensors(
                states,
                actions,
                returns,
                continuation_step,
                action,
                next_state,
                reward_scaled,
                reward_mode,
            )
            suffix_scaled_return += float(reward_scaled)
            final_step = continuation_step + 1
            if done:
                break

    full_raw_return = (
        policy_snapshot.prefix_scaled_return + suffix_scaled_return
    ) / reward_scale
    return BranchResult(
        full_raw_return=float(full_raw_return),
        suffix_raw_return=float(suffix_scaled_return / reward_scale),
        normalized_score=normalized_score(env, full_raw_return),
        episode_length=int(final_step),
        first_next_state=first_next_state,
        first_reward_scaled=first_reward_scaled,
        first_done=first_done,
    )


def stratified_anchor_times(
    *,
    episode_len: int,
    num_anchors: int,
    exclude_final_steps: int,
    seed_components: Sequence[int],
) -> List[int]:
    upper = episode_len - exclude_final_steps
    if num_anchors <= 0 or upper <= 0:
        raise ValueError("Invalid anchor configuration")
    edges = np.linspace(0, upper, num_anchors + 1, dtype=np.int64)
    rng = np.random.default_rng(np.random.SeedSequence(seed_components))
    anchors = []
    for left, right in zip(edges[:-1], edges[1:]):
        right = max(int(right), int(left) + 1)
        anchors.append(int(rng.integers(int(left), right)))
    return sorted(set(anchors))


@torch.inference_mode()
def knn_indices(
    query_normalized: np.ndarray,
    offline_states_normalized: torch.Tensor,
    k: int,
    chunk_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    query = torch.as_tensor(
        query_normalized,
        dtype=torch.float32,
        device=offline_states_normalized.device,
    ).view(1, -1)
    best_distances = None
    best_indices = None
    total = int(offline_states_normalized.shape[0])

    for start in range(0, total, chunk_size):
        stop = min(start + chunk_size, total)
        chunk = offline_states_normalized[start:stop]
        squared = torch.sum((chunk - query) ** 2, dim=1)
        local_k = min(k, squared.numel())
        distances, indices = torch.topk(squared, k=local_k, largest=False)
        indices = indices + start

        if best_distances is None:
            best_distances = distances
            best_indices = indices
        else:
            merged_distances = torch.cat((best_distances, distances), dim=0)
            merged_indices = torch.cat((best_indices, indices), dim=0)
            keep_k = min(k, merged_distances.numel())
            best_distances, positions = torch.topk(
                merged_distances, k=keep_k, largest=False
            )
            best_indices = merged_indices[positions]

    return (
        best_indices.detach().cpu().numpy(),
        torch.sqrt(best_distances).detach().cpu().numpy(),
    )


def make_antithetic_directions(
    *,
    action_dim: int,
    num_pairs: int,
    seed_components: Sequence[int],
) -> np.ndarray:
    rng = np.random.default_rng(np.random.SeedSequence(seed_components))
    positive = rng.normal(size=(num_pairs, action_dim)).astype(np.float32)
    return np.concatenate((positive, -positive), axis=0)


def support_distance(
    candidate: np.ndarray,
    local_actions: np.ndarray,
    local_std: np.ndarray,
) -> float:
    scaled = (local_actions - candidate[None, :]) / (local_std[None, :] + 1e-8)
    return float(np.sqrt(np.sum(np.square(scaled), axis=1)).min())


def write_rows_atomic(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def episode_bootstrap_ci(
    episode_to_values: Dict[int, List[float]], reps: int, seed: int
) -> Tuple[float, float]:
    episode_ids = np.asarray(sorted(episode_to_values), dtype=np.int64)
    if episode_ids.size == 0:
        return float("nan"), float("nan")
    episode_means = {
        int(key): float(np.mean(values)) for key, values in episode_to_values.items()
    }
    rng = np.random.default_rng(seed)
    estimates = np.empty(reps, dtype=np.float64)
    for rep in range(reps):
        sampled = rng.choice(episode_ids, size=episode_ids.size, replace=True)
        estimates[rep] = np.mean([episode_means[int(key)] for key in sampled])
    return tuple(float(x) for x in np.quantile(estimates, [0.025, 0.975]))


def summarize_rows(
    rows: List[Dict[str, Any]], alphas: Sequence[float], bootstrap_reps: int
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for alpha in alphas:
        alpha_rows = [row for row in rows if np.isclose(float(row["alpha"]), alpha)]
        grouped: Dict[Tuple[int, int], List[Dict[str, Any]]] = defaultdict(list)
        for row in alpha_rows:
            grouped[(int(row["episode_id"]), int(row["anchor_timestep"]))].append(row)

        headrooms = []
        random_means = []
        densities_01 = []
        densities_05 = []
        oracle_support = []
        oracle_clipped = []
        episode_to_headroom: Dict[int, List[float]] = defaultdict(list)

        for (episode_id, _), group in grouped.items():
            candidates = [row for row in group if int(row["candidate_id"]) > 0]
            if not candidates:
                continue
            deltas = np.asarray(
                [float(row["delta_normalized_return"]) for row in candidates]
            )
            best_position = int(np.argmax(deltas))
            best = candidates[best_position]
            headroom = float(deltas[best_position])
            headrooms.append(headroom)
            random_means.append(float(np.mean(deltas)))
            densities_01.append(float(np.mean(deltas > 0.1)))
            densities_05.append(float(np.mean(deltas > 0.5)))
            oracle_support.append(float(best["action_support_distance"]))
            oracle_clipped.append(float(int(best["was_action_clipped"])))
            episode_to_headroom[episode_id].append(headroom)

        if not headrooms:
            continue
        values = np.asarray(headrooms, dtype=np.float64)
        ci_low, ci_high = episode_bootstrap_ci(
            episode_to_headroom,
            reps=bootstrap_reps,
            seed=12345 + int(round(alpha * 10000)),
        )
        summary[f"{alpha:.6g}"] = {
            "num_anchors": int(values.size),
            "mean_oracle_uplift_normalized": float(np.mean(values)),
            "median_oracle_uplift_normalized": float(np.median(values)),
            "bootstrap_95_ci": [ci_low, ci_high],
            "headroom_rate_gt_0": float(np.mean(values > 0.0)),
            "headroom_rate_gt_0.1": float(np.mean(values > 0.1)),
            "headroom_rate_gt_0.5": float(np.mean(values > 0.5)),
            "mean_positive_candidate_density_gt_0.1": float(np.mean(densities_01)),
            "mean_positive_candidate_density_gt_0.5": float(np.mean(densities_05)),
            "mean_random_candidate_delta_normalized": float(np.mean(random_means)),
            "mean_oracle_support_distance": float(np.mean(oracle_support)),
            "oracle_clipping_rate": float(np.mean(oracle_clipped)),
        }
    return summary


def main() -> None:
    args = parse_args()
    validate_reward_mode(args.reward_mode)
    if args.num_pairs < 1:
        raise ValueError("--num-pairs must be at least 1")
    if args.eval_episodes < 1 or args.anchors_per_episode < 1:
        raise ValueError("Evaluation and anchor counts must be positive")
    if any(alpha < 0 for alpha in args.alphas):
        raise ValueError("Candidate radii must be non-negative")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "candidate_branches.csv"
    summary_path = output_dir / "summary.json"
    config_path = output_dir / "run_config.json"

    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
    model, checkpoint_config = build_model(checkpoint, args.device)
    state_mean = np.asarray(checkpoint["state_mean"], dtype=np.float32)
    state_std = np.asarray(checkpoint["state_std"], dtype=np.float32)
    reward_scale = float(checkpoint_config["reward_scale"])
    train_seed = int(checkpoint_config["train_seed"])
    env_name = str(checkpoint_config["env_name"])

    run_config = vars(args).copy()
    run_config.update(
        {
            "checkpoint_train_seed": train_seed,
            "checkpoint_step": checkpoint.get("next_step"),
            "env_name": env_name,
            "reward_scale": reward_scale,
            "checkpoint_config": checkpoint_config,
        }
    )
    config_path.write_text(json.dumps(run_config, indent=2, default=str) + "\n")

    print("Loading offline dataset for local action scales...", flush=True)
    dataset_env = gym.make(env_name)
    offline_dataset = get_d4rl_dataset(dataset_env)
    dataset_env.close()
    offline_observations = np.asarray(
        offline_dataset["observations"], dtype=np.float32
    )
    offline_actions = np.asarray(offline_dataset["actions"], dtype=np.float32)
    offline_normalized = (offline_observations - state_mean) / state_std
    offline_states_device = torch.as_tensor(
        offline_normalized, dtype=torch.float32, device=args.device
    )
    global_action_std = offline_actions.std(axis=0).astype(np.float32) + 1e-6

    eval_env = wrap_env(
        env=gym.make(env_name),
        state_mean=state_mean,
        state_std=state_std,
        reward_scale=reward_scale,
    )
    eval_env.seed(args.eval_seed)
    eval_env.action_space.seed(args.eval_seed)

    time_steps = torch.arange(
        model.episode_len, dtype=torch.long, device=args.device
    ).view(1, -1)
    target_return_scaled = args.target_return * reward_scale
    action_low = np.asarray(eval_env.action_space.low, dtype=np.float32)
    action_high = np.asarray(eval_env.action_space.high, dtype=np.float32)

    all_rows: List[Dict[str, Any]] = []
    baseline_episode_returns: List[float] = []
    restore_return_errors: List[float] = []
    restore_state_errors: List[float] = []
    outer_baseline_errors: List[float] = []
    start_time = time.time()

    for episode_id in range(args.eval_episodes):
        first_state = np.asarray(eval_env.reset(), dtype=np.float32)
        states, actions, returns, _ = allocate_policy_tensors(
            model, target_return_scaled, first_state, args.device
        )
        prefix_scaled_return = 0.0
        anchor_times = stratified_anchor_times(
            episode_len=model.episode_len,
            num_anchors=args.anchors_per_episode,
            exclude_final_steps=args.exclude_final_steps,
            seed_components=(
                args.candidate_seed,
                train_seed,
                args.eval_seed,
                episode_id,
                991,
            ),
        )
        anchor_set = set(anchor_times)
        episode_base_branches: List[Tuple[int, float]] = []

        progress = tqdm(
            range(model.episode_len),
            desc=f"Baseline episode {episode_id + 1}/{args.eval_episodes}",
        )
        for step in progress:
            action_base = predict_action(
                model, states, actions, returns, time_steps, step
            )

            if step in anchor_set:
                env_snapshot = capture_env_snapshot(eval_env)
                policy_snapshot = PolicySnapshot(
                    states=states.clone(),
                    actions=actions.clone(),
                    returns=returns.clone(),
                    prefix_scaled_return=float(prefix_scaled_return),
                    step=int(step),
                )

                query_normalized = states[0, step].detach().cpu().numpy()
                neighbor_indices, neighbor_distances = knn_indices(
                    query_normalized=query_normalized,
                    offline_states_normalized=offline_states_device,
                    k=args.knn_neighbors,
                    chunk_size=args.dataset_chunk_size,
                )
                local_actions = offline_actions[neighbor_indices]
                local_std = local_actions.std(axis=0).astype(np.float32)
                local_std = np.maximum(
                    local_std,
                    args.action_std_floor_frac * global_action_std,
                )
                directions = make_antithetic_directions(
                    action_dim=model.action_dim,
                    num_pairs=args.num_pairs,
                    seed_components=(
                        args.candidate_seed,
                        train_seed,
                        args.eval_seed,
                        episode_id,
                        step,
                    ),
                )

                baseline_branch = branch_rollout(
                    env=eval_env,
                    env_snapshot=env_snapshot,
                    policy_snapshot=policy_snapshot,
                    first_action=action_base,
                    model=model,
                    time_steps=time_steps,
                    reward_scale=reward_scale,
                    reward_mode=args.reward_mode,
                )
                episode_base_branches.append((step, baseline_branch.full_raw_return))

                if args.repeat_baseline_check:
                    repeated = branch_rollout(
                        env=eval_env,
                        env_snapshot=env_snapshot,
                        policy_snapshot=policy_snapshot,
                        first_action=action_base,
                        model=model,
                        time_steps=time_steps,
                        reward_scale=reward_scale,
                        reward_mode=args.reward_mode,
                    )
                    return_error = abs(
                        repeated.full_raw_return - baseline_branch.full_raw_return
                    )
                    state_error = float(
                        np.max(
                            np.abs(
                                repeated.first_next_state
                                - baseline_branch.first_next_state
                            )
                        )
                    )
                    restore_return_errors.append(return_error)
                    restore_state_errors.append(state_error)
                    if return_error > args.restore_return_tol:
                        raise RuntimeError(
                            f"Restore return mismatch at episode={episode_id}, "
                            f"step={step}: {return_error}"
                        )
                    if state_error > args.restore_state_tol:
                        raise RuntimeError(
                            f"Restore next-state mismatch at episode={episode_id}, "
                            f"step={step}: {state_error}"
                        )

                for alpha in args.alphas:
                    base_support = support_distance(
                        action_base, local_actions, local_std
                    )
                    common = {
                        "experiment_version": "oracle_branching_v0.1",
                        "checkpoint_path": str(args.checkpoint_path),
                        "checkpoint_seed": train_seed,
                        "checkpoint_step": checkpoint.get("next_step"),
                        "target_return": float(args.target_return),
                        "reward_mode": args.reward_mode,
                        "eval_seed": int(args.eval_seed),
                        "episode_id": int(episode_id),
                        "anchor_timestep": int(step),
                        "alpha": float(alpha),
                        "state_knn_distance": float(neighbor_distances[-1]),
                        "prefix_raw_return": float(prefix_scaled_return / reward_scale),
                    }
                    base_row = {
                        **common,
                        "candidate_id": 0,
                        "is_baseline_candidate": 1,
                        "action_base": json.dumps(action_base.tolist()),
                        "action_candidate_raw": json.dumps(action_base.tolist()),
                        "action_candidate_clipped": json.dumps(action_base.tolist()),
                        "action_l2_distance": 0.0,
                        "action_mahalanobis_distance": 0.0,
                        "action_support_distance": base_support,
                        "was_action_clipped": 0,
                        "branch_suffix_return": baseline_branch.suffix_raw_return,
                        "branch_full_return": baseline_branch.full_raw_return,
                        "branch_normalized_return": baseline_branch.normalized_score,
                        "delta_raw_return": 0.0,
                        "delta_normalized_return": 0.0,
                        "termination_timestep": baseline_branch.episode_length,
                        "restore_check_passed": 1,
                    }
                    all_rows.append(base_row)

                    for candidate_id, direction in enumerate(directions, start=1):
                        candidate_raw = (
                            action_base + float(alpha) * local_std * direction
                        )
                        candidate = np.clip(
                            candidate_raw, action_low, action_high
                        ).astype(np.float32)
                        was_clipped = int(
                            not np.allclose(
                                candidate_raw, candidate, rtol=0.0, atol=1e-8
                            )
                        )
                        result = branch_rollout(
                            env=eval_env,
                            env_snapshot=env_snapshot,
                            policy_snapshot=policy_snapshot,
                            first_action=candidate,
                            model=model,
                            time_steps=time_steps,
                            reward_scale=reward_scale,
                            reward_mode=args.reward_mode,
                        )
                        delta_raw = (
                            result.full_raw_return - baseline_branch.full_raw_return
                        )
                        delta_normalized = (
                            result.normalized_score - baseline_branch.normalized_score
                        )
                        mahalanobis = float(
                            np.sqrt(
                                np.sum(
                                    np.square(
                                        (candidate - action_base) / (local_std + 1e-8)
                                    )
                                )
                            )
                        )
                        all_rows.append(
                            {
                                **common,
                                "candidate_id": int(candidate_id),
                                "is_baseline_candidate": 0,
                                "action_base": json.dumps(action_base.tolist()),
                                "action_candidate_raw": json.dumps(
                                    candidate_raw.tolist()
                                ),
                                "action_candidate_clipped": json.dumps(
                                    candidate.tolist()
                                ),
                                "action_l2_distance": float(
                                    np.linalg.norm(candidate - action_base)
                                ),
                                "action_mahalanobis_distance": mahalanobis,
                                "action_support_distance": support_distance(
                                    candidate, local_actions, local_std
                                ),
                                "was_action_clipped": was_clipped,
                                "branch_suffix_return": result.suffix_raw_return,
                                "branch_full_return": result.full_raw_return,
                                "branch_normalized_return": result.normalized_score,
                                "delta_raw_return": float(delta_raw),
                                "delta_normalized_return": float(delta_normalized),
                                "termination_timestep": result.episode_length,
                                "restore_check_passed": 1,
                            }
                        )

                restore_env_snapshot(eval_env, env_snapshot)

            next_state, reward_scaled, done, _ = eval_env.step(action_base)
            update_policy_tensors(
                states,
                actions,
                returns,
                step,
                action_base,
                next_state,
                reward_scaled,
                args.reward_mode,
            )
            prefix_scaled_return += float(reward_scaled)
            if done:
                break

        outer_raw_return = float(prefix_scaled_return / reward_scale)
        baseline_episode_returns.append(outer_raw_return)
        for anchor_step, base_branch_return in episode_base_branches:
            error = abs(base_branch_return - outer_raw_return)
            outer_baseline_errors.append(error)
            if error > args.restore_return_tol:
                raise RuntimeError(
                    f"Candidate-0 branch at episode={episode_id}, step={anchor_step} "
                    f"does not reproduce outer baseline: error={error}"
                )
        write_rows_atomic(rows_path, all_rows)
        print(
            f"episode={episode_id} raw_return={outer_raw_return:.6f} "
            f"normalized={normalized_score(eval_env, outer_raw_return):.6f} "
            f"rows={len(all_rows)}",
            flush=True,
        )

    alpha_summary = summarize_rows(all_rows, args.alphas, args.bootstrap_reps)
    final_summary = {
        "experiment_version": "oracle_branching_v0.1",
        "checkpoint_path": str(args.checkpoint_path),
        "checkpoint_seed": train_seed,
        "checkpoint_step": checkpoint.get("next_step"),
        "env_name": env_name,
        "target_return": float(args.target_return),
        "reward_mode": args.reward_mode,
        "eval_seed": int(args.eval_seed),
        "eval_episodes": int(args.eval_episodes),
        "anchors_per_episode": int(args.anchors_per_episode),
        "num_candidates_per_alpha": int(1 + 2 * args.num_pairs),
        "alphas": [float(alpha) for alpha in args.alphas],
        "baseline_raw_return_mean": float(np.mean(baseline_episode_returns)),
        "baseline_raw_return_std": float(np.std(baseline_episode_returns)),
        "baseline_normalized_score_mean": float(
            np.mean(
                [normalized_score(eval_env, value) for value in baseline_episode_returns]
            )
        ),
        "baseline_normalized_score_std": float(
            np.std(
                [normalized_score(eval_env, value) for value in baseline_episode_returns]
            )
        ),
        "max_repeated_restore_state_error": float(
            max(restore_state_errors, default=0.0)
        ),
        "max_repeated_restore_return_error": float(
            max(restore_return_errors, default=0.0)
        ),
        "max_outer_baseline_return_error": float(
            max(outer_baseline_errors, default=0.0)
        ),
        "wall_time_seconds": float(time.time() - start_time),
        "by_alpha": alpha_summary,
    }
    summary_path.write_text(json.dumps(final_summary, indent=2) + "\n")
    write_rows_atomic(rows_path, all_rows)
    eval_env.close()

    print(json.dumps(final_summary, indent=2), flush=True)
    print(f"Saved candidate rows to {rows_path}", flush=True)
    print(f"Saved summary to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
