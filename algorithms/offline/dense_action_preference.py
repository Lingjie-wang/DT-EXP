"""Original-reward RTG labels for an independent state-only C experiment."""

import numpy as np
from state_only_preference import preference_batch as delayed_preference_batch

def mine_dense_pairs(trajectories, state_mean, state_std, max_distance=0.5):
    """Nearest low-RTG state per high-RTG state, with SAME time and horizon.

    Recompute 30/70-percentile groups at each timestep using original-reward
    return-to-go. Earlier rewards never enter the preference label. All episodes
    must have the same horizon (the HCMR dataset used here has 1000-step episodes).
    No action/prefix filter, critic, fitted reward or extra gap threshold.
    """
    if not trajectories or not np.isfinite(max_distance) or max_distance <= 0:
        raise ValueError("Expected trajectories and a positive finite state cutoff")
    if not np.isfinite(state_mean).all() or not np.isfinite(state_std).all():
        raise ValueError("Nonfinite state normalization")
    if not np.all(state_std > 0):
        raise ValueError("State standard deviation must be positive")
    lengths = [len(t["actions"]) for t in trajectories]
    if min(lengths) <= 0 or len(set(lengths)) != 1:
        raise ValueError("Dense HCMR matching requires equal, nonempty horizons")
    states = np.stack([(t["observations"] - state_mean) / state_std
                       for t in trajectories])
    returns = np.stack([t["returns"] for t in trajectories])
    if returns.shape != (len(trajectories), lengths[0]):
        raise ValueError("RTG and action lengths differ")
    if not np.isfinite(states).all() or not np.isfinite(returns).all():
        raise ValueError("Nonfinite state or RTG")
    pairs, distances, gaps = [], [], []
    candidates = 0
    skipped_times = 0
    for step in range(lengths[0]):
        low, high = np.quantile(returns[:, step], [0.3, 0.7])
        if low >= high:
            skipped_times += 1
            continue
        good = np.flatnonzero(returns[:, step] >= high)
        bad = np.flatnonzero(returns[:, step] <= low)
        distance = np.sqrt(((states[good, step, None] - states[bad, step]) ** 2)
                           .mean(-1))
        nearest = distance.argmin(axis=1)
        candidates += len(good)
        for row, column in enumerate(nearest):
            if distance[row, column] <= max_distance:
                pos, neg = good[row], bad[column]
                pairs.append((pos, step, neg, step))
                distances.append(distance[row, column])
                gaps.append(returns[pos, step] - returns[neg, step])
    if not pairs:
        raise ValueError("No supported dense RTG pairs; do not relax cutoff silently")
    pairs = np.asarray(pairs, dtype=np.int64)
    distances = np.asarray(distances, dtype=np.float32)
    gaps = np.asarray(gaps, dtype=np.float32)
    totals = returns[:, 0]
    stats = {
        "label": "original_reward_undiscounted_return_to_go",
        "group_quantiles": [0.3, 0.7],
        "num_pairs": len(pairs),
        "candidate_high_states": candidates,
        "coverage": len(pairs) / candidates,
        "covered_good_trajectories": len(np.unique(pairs[:, 0])),
        "covered_bad_trajectories": len(np.unique(pairs[:, 2])),
        "state_rmse_mean": float(distances.mean()),
        "state_rmse_max": float(distances.max()),
        "rtg_gap_mean": float(gaps.mean()),
        "rtg_gap_min": float(gaps.min()),
        "rtg_gap_max": float(gaps.max()),
        "total_return_order_disagreement_fraction": float(np.mean(
            totals[pairs[:, 0]] <= totals[pairs[:, 2]])),
        "skipped_tied_timesteps": skipped_times,
    }
    return pairs, distances, stats


def dense_preference_batch(dataset, pairs, indices, target_return):
    """Condition on desired initial return minus actual positive-prefix rewards.

    Both candidate actions use the SAME positive state/context and RTG tokens.
    Prefix rewards are used only to update DT's budget, never to match states or
    to label actions. No future reward is used to build this target budget.
    """
    batch = delayed_preference_batch(dataset, pairs, indices, target_return)
    returns = batch[2]
    for row, pair_id in enumerate(indices):
        good, step, _, _ = pairs[pair_id]
        trajectory = dataset.dataset[good]
        start = max(0, step + 1 - dataset.seq_len)
        # Accumulate in float64; at time t, only rewards before t are observed.
        past = np.concatenate(([0.0], np.cumsum(
            trajectory["rewards"][:step], dtype=np.float64)))
        length = step + 1 - start
        returns[row, :length] = ((target_return - past[start:step + 1])
                                * dataset.reward_scale)
    return batch
