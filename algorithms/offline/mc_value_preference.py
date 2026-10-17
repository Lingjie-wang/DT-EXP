"""Frozen Monte Carlo state values and one-step action preference scores."""

import copy

import numpy as np
import torch
from torch import nn

VALUE_SETTINGS = {"hidden_dim": 64, "batch_size": 1024, "learning_rate": 0.001,
                  "weight_decay": 0.0001, "validation_fraction": 0.2}


class StateTimeValue(nn.Module):
    """Input: standardized current state and t/T. Output: scaled remaining return."""

    def __init__(self, state_dim):
        super().__init__()
        width = VALUE_SETTINGS["hidden_dim"]
        self.network = nn.Sequential(nn.Linear(state_dim + 1, width), nn.ReLU(),
                                     nn.Linear(width, width), nn.ReLU(),
                                     nn.Linear(width, 1))

    def forward(self, inputs):
        return self.network(inputs).squeeze(-1)


def value_data(trajectories, state_mean, state_std):
    """Keep trajectory boundaries; MC labels use observed original rewards only."""
    if len(trajectories) < 2:
        raise ValueError("Need at least two complete trajectories")
    lengths = [len(t["actions"]) for t in trajectories]
    if min(lengths) <= 0 or len(set(lengths)) != 1:
        raise ValueError("MC-value HCMR requires equal, nonempty horizons")
    if (not np.isfinite(state_mean).all() or not np.isfinite(state_std).all()
            or not np.all(state_std > 0)):
        raise ValueError("Invalid state normalization")
    states = np.stack([(t["observations"] - state_mean) / state_std
                       for t in trajectories]).astype(np.float32)
    rewards = np.stack([t["rewards"] for t in trajectories]).astype(np.float32)
    if states.shape[:2] != rewards.shape or rewards.shape[1] != lengths[0]:
        raise ValueError("State/reward/action lengths differ")
    if not np.isfinite(states).all() or not np.isfinite(rewards).all():
        raise ValueError("Nonfinite state or original reward")
    times = np.broadcast_to(np.arange(lengths[0], dtype=np.float32)[None, :, None]
                            / lengths[0], (*rewards.shape, 1))
    inputs = np.concatenate((states, times), axis=-1)
    targets = np.cumsum(rewards[:, ::-1], axis=1, dtype=np.float64)[:, ::-1].copy()
    return inputs, targets, rewards


def trajectory_split(count, seed):
    if count < 2:
        raise ValueError("Need two trajectories to hold out complete episodes")
    order = np.random.RandomState(seed).permutation(count)
    fraction = VALUE_SETTINGS["validation_fraction"]
    validation_count = max(1, int(np.ceil(count * fraction)))
    return np.sort(order[validation_count:]), np.sort(order[:validation_count])


def one_step_scores(rewards, values):
    """Gamma=1, finite-horizon score r[t] + V(s[t+1], t+1) - V(s[t], t).

    Within each complete episode, s[t+1] is its next recorded observation.
    At the final step V(next)=0, including the 1000-step HCMR time limit. Never
    bootstrap from another episode or the model's unconstrained terminal output.
    """
    rewards, values = np.asarray(rewards, dtype=np.float64), np.asarray(
        values, dtype=np.float64)
    if rewards.shape != values.shape or rewards.ndim != 2 or not rewards.shape[1]:
        raise ValueError("Expected matching [trajectory, time] rewards and values")
    if not np.isfinite(rewards).all() or not np.isfinite(values).all():
        raise ValueError("Nonfinite reward or frozen value")
    following = np.zeros_like(values)
    following[:, :-1] = values[:, 1:]
    return (rewards + following - values).astype(np.float32)


def fit_frozen_value(trajectories, state_mean, state_std, reward_scale=0.001,
                     epochs=50, seed=1729, device="cpu", on_epoch=None):
    """Plain supervised MSE, whole-episode validation, then freeze the best epoch.

    No Q function, Bellman target, target network, expectile or policy feedback.
    A private NumPy stream and restored CPU RNG keep DT's sampling unaffected.
    """
    if epochs <= 0 or not np.isfinite(reward_scale) or reward_scale <= 0:
        raise ValueError("Positive epochs and finite positive reward scale required")
    features, raw_targets, rewards = value_data(trajectories, state_mean, state_std)
    train_ids, val_ids = trajectory_split(len(trajectories), seed)
    rng = np.random.RandomState(seed + 1)
    batch_size = VALUE_SETTINGS["batch_size"]
    # Initialize on CPU without seeding or drawing from any CUDA RNG stream.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model = StateTimeValue(features.shape[-1] - 1).to(device)
        x = torch.as_tensor(features, device=device)
        y = torch.as_tensor(raw_targets * reward_scale, dtype=torch.float32,
                            device=device)
        train_x = x[train_ids].reshape(-1, features.shape[-1])
        train_y = y[train_ids].reshape(-1)
        val_x = x[val_ids].reshape(-1, features.shape[-1])
        val_y = y[val_ids].reshape(-1)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=VALUE_SETTINGS["learning_rate"],
            weight_decay=VALUE_SETTINGS["weight_decay"])

        @torch.no_grad()
        def evaluate(inputs, labels):
            errors = []
            for start in range(0, len(inputs), batch_size):
                errors.append((model(inputs[start:start + batch_size])
                               - labels[start:start + batch_size]).cpu().numpy())
            errors = np.concatenate(errors).astype(np.float64)
            mse = float(np.mean(errors ** 2))
            return mse, float(np.mean(np.abs(errors)))

        history = []
        best_mse, best_epoch, best_state = float("inf"), None, None
        for epoch in range(1, epochs + 1):
            model.train()
            for ids in np.array_split(rng.permutation(len(train_x)),
                                       int(np.ceil(len(train_x) / batch_size))):
                index = torch.as_tensor(ids, dtype=torch.long, device=device)
                loss = (model(train_x[index]) - train_y[index]).square().mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite MC value regression loss")
                optimizer.zero_grad()
                loss.backward()
                if not all(torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise FloatingPointError("Nonfinite MC value gradient")
                optimizer.step()
            model.eval()
            train_mse, _ = evaluate(train_x, train_y)
            val_mse, val_mae = evaluate(val_x, val_y)
            if not np.isfinite([train_mse, val_mse, val_mae]).all():
                raise FloatingPointError("Nonfinite MC value validation")
            row = {"epoch": epoch, "train_mse_scaled": train_mse,
                   "validation_mse_scaled": val_mse,
                   "validation_rmse_raw": float(np.sqrt(val_mse) / reward_scale),
                   "validation_mae_raw": val_mae / reward_scale}
            history.append(row)
            if val_mse < best_mse:
                best_mse, best_epoch = val_mse, epoch
                best_state = copy.deepcopy(model.state_dict())
            if on_epoch is not None:
                on_epoch(dict(row))
        model.load_state_dict(best_state)
        model.eval().requires_grad_(False)
        flat = x.reshape(-1, features.shape[-1])
        with torch.no_grad():
            values = np.concatenate([model(flat[i:i + batch_size]).cpu().numpy()
                                     for i in range(0, len(flat), batch_size)])
        values = values.reshape(raw_targets.shape).astype(np.float64) / reward_scale
    if not np.isfinite(values).all():
        raise FloatingPointError("Nonfinite frozen value predictions")
    scores = one_step_scores(rewards, values)
    val_variance = float(np.var(raw_targets[val_ids]))
    raw_mse = best_mse / reward_scale ** 2
    train_mean = raw_targets[train_ids].mean()
    baseline_mse = float(np.mean((raw_targets[val_ids] - train_mean) ** 2))
    report = {
        "method": "supervised_monte_carlo_value_then_freeze",
        "target": "sum_of_observed_original_rewards_from_t_to_episode_end",
        "seed": seed, "epochs": epochs, "best_epoch": best_epoch,
        "settings": VALUE_SETTINGS, "reward_scale": reward_scale,
        "train_trajectory_ids": train_ids.tolist(),
        "validation_trajectory_ids": val_ids.tolist(),
        "validation_rmse_raw": float(np.sqrt(raw_mse)),
        "validation_r2": 1 - raw_mse / val_variance if val_variance > 0 else None,
        "validation_train_mean_baseline_rmse_raw": float(np.sqrt(baseline_mse)),
        "selection": "minimum whole-trajectory validation MSE; no policy scores",
        "mining_includes_value_training_trajectories": True,
        "score_mean": float(scores.mean()), "score_std": float(scores.std()),
        "reward_mean": float(rewards.mean()), "reward_std": float(rewards.std()),
        "value_change_std": float((scores - rewards).std()),
    }
    checkpoint = {"model_state": {k: v.cpu() for k, v in model.state_dict().items()},
                  "state_mean": state_mean, "state_std": state_std,
                  "episode_len": rewards.shape[1], "report": report}
    return scores, values.astype(np.float32), report, history, checkpoint


def mine_value_pairs(trajectories, scores, state_mean, state_std, max_distance=0.5):
    """Same-time nearest-state pairing, ranked by frozen one-step value scores."""
    features, returns, rewards = value_data(trajectories, state_mean, state_std)
    scores = np.asarray(scores)
    if scores.shape != rewards.shape or not np.isfinite(scores).all():
        raise ValueError("Scores must be finite and match complete trajectories")
    if not np.isfinite(max_distance) or max_distance <= 0:
        raise ValueError("Positive finite state cutoff required")
    states = features[..., :-1]
    pairs, distances, gaps = [], [], []
    candidates = skipped = 0
    for step in range(scores.shape[1]):
        low, high = np.quantile(scores[:, step], [0.3, 0.7])
        if low >= high:
            skipped += 1
            continue
        good, bad = np.flatnonzero(scores[:, step] >= high), np.flatnonzero(
            scores[:, step] <= low)
        delta = states[good, step, None] - states[bad, step]
        distance = np.sqrt((delta ** 2).mean(-1))
        nearest = distance.argmin(axis=1)
        candidates += len(good)
        for row, column in enumerate(nearest):
            if distance[row, column] <= max_distance:
                pos, neg = good[row], bad[column]
                pairs.append((pos, step, neg, step))
                distances.append(distance[row, column])
                gaps.append(scores[pos, step] - scores[neg, step])
    if not pairs:
        raise ValueError("No supported frozen-value pairs; do not relax cutoff silently")
    pairs = np.asarray(pairs, dtype=np.int64)
    distances, gaps = np.asarray(distances, dtype=np.float32), np.asarray(gaps)
    pos, pt, neg, nt = pairs.T
    stats = {"label": "original_reward_plus_frozen_next_value_minus_current_value",
             "group_quantiles": [0.3, 0.7], "num_pairs": len(pairs),
             "candidate_high_states": candidates, "coverage": len(pairs) / candidates,
             "state_rmse_mean": float(distances.mean()),
             "state_rmse_max": float(distances.max()),
             "score_gap_mean": float(gaps.mean()), "score_gap_min": float(gaps.min()),
             "score_gap_max": float(gaps.max()), "skipped_tied_timesteps": skipped,
             "covered_good_trajectories": len(np.unique(pos)),
             "covered_bad_trajectories": len(np.unique(neg)),
             "remaining_return_order_disagreement_fraction": float(np.mean(
                 returns[pos, pt] <= returns[neg, nt])),
             "immediate_reward_order_disagreement_fraction": float(np.mean(
                 rewards[pos, pt] <= rewards[neg, nt]))}
    return pairs, distances, stats
