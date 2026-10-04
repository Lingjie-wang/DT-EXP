"""Current-state mining and the C-only, detached-negative action objective."""

import numpy as np
import torch.nn.functional as F

def mine_pairs(trajectories, state_mean, state_std, max_distance=0.5):
    """One exact nearest low-return state at the SAME timestep per high state.

    Return groups use trajectory quantiles, never dense per-step rewards. No
    prefix comparison, action-distance filter, difficulty score or random miner.
    Ties are resolved by the original trajectory order.
    """
    if not trajectories or not np.isfinite(max_distance) or max_distance <= 0:
        raise ValueError("Expected trajectories and a positive finite state cutoff")
    returns = np.array([float(t["returns"][0]) for t in trajectories])
    if not np.isfinite(returns).all() or not np.all(state_std > 0):
        raise ValueError("Nonfinite returns or invalid state normalization")
    low, high = np.quantile(returns, [0.3, 0.7])
    if low >= high:
        raise ValueError("High and low return groups must be disjoint")
    good = np.flatnonzero(returns >= high)
    bad = np.flatnonzero(returns <= low)
    lengths = np.array([len(t["actions"]) for t in trajectories])
    states = [
        (t["observations"] - state_mean) / state_std for t in trajectories
    ]
    if not all(np.isfinite(s).all() for s in states):
        raise ValueError("Nonfinite normalized state")
    pairs, distances = [], []
    candidates = 0
    for step in range(int(lengths.max())):
        good_ids = good[lengths[good] > step]
        bad_ids = bad[lengths[bad] > step]
        if not len(good_ids) or not len(bad_ids):
            continue
        positive = np.stack([states[i][step] for i in good_ids])
        negative = np.stack([states[i][step] for i in bad_ids])
        distance = np.sqrt(((positive[:, None] - negative[None]) ** 2).mean(-1))
        nearest = distance.argmin(axis=1)
        candidates += len(good_ids)
        for row, column in enumerate(nearest):
            if distance[row, column] <= max_distance:
                pairs.append((good_ids[row], step, bad_ids[column], step))
                distances.append(distance[row, column])
    if not pairs:
        raise ValueError("No supported state-only pairs; do not relax cutoff silently")
    pairs = np.asarray(pairs, dtype=np.int64)
    distances = np.asarray(distances, dtype=np.float32)
    stats = {
        "num_pairs": len(pairs),
        "candidate_high_states": candidates,
        "coverage": len(pairs) / candidates,
        "good_return_threshold": float(high),
        "bad_return_threshold": float(low),
        "good_trajectories": len(good),
        "bad_trajectories": len(bad),
        "covered_good_trajectories": len(np.unique(pairs[:, 0])),
        "covered_bad_trajectories": len(np.unique(pairs[:, 2])),
        "state_rmse_mean": float(distances.mean()),
        "state_rmse_max": float(distances.max()),
    }
    return pairs, distances, stats


def preference_batch(dataset, pairs, indices, target_return):
    """Use the positive DT context; both actions are scored in this one context.

    Prefixes are model inputs only, never mining criteria. Right padding keeps
    valid causal queries from attending to padding. Current action labels are
    causally downstream of the state token used for action prediction.
    """
    samples = []
    for pair_id in indices:
        good, step, bad, negative_step = pairs[pair_id]
        trajectory = dataset.dataset[good]
        start = max(0, step + 1 - dataset.seq_len)
        length = step + 1 - start
        states = np.zeros(
            (dataset.seq_len, trajectory["observations"].shape[1]), dtype=np.float32
        )
        actions = np.zeros(
            (dataset.seq_len, trajectory["actions"].shape[1]), dtype=np.float32
        )
        returns = np.zeros(dataset.seq_len, dtype=np.float32)
        mask = np.zeros(dataset.seq_len, dtype=np.float32)
        states[:length] = (
            trajectory["observations"][start : step + 1] - dataset.state_mean
        ) / dataset.state_std
        actions[:length] = trajectory["actions"][start : step + 1]
        returns[:length] = target_return * dataset.reward_scale
        mask[:length] = 1
        samples.append((
            states, actions, returns,
            np.arange(start, start + dataset.seq_len, dtype=np.int64), mask,
            np.int64(length - 1),
            dataset.dataset[bad]["actions"][negative_step].astype(np.float32),
        ))
    return tuple(np.stack(items) for items in zip(*samples))


def single_sided_loss(prediction, positive, negative, margin=0.05):
    """Whole-batch mean; negative distance is ONLY a detached stopping gate."""
    d_positive = F.mse_loss(prediction, positive.detach(), reduction="none").mean(-1)
    d_negative = F.mse_loss(prediction, negative.detach(), reduction="none").mean(-1)
    violation = d_positive - d_negative.detach() + margin
    return F.relu(violation).mean(), d_positive, d_negative, violation > 0
