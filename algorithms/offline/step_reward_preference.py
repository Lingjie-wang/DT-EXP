"""Single-step reward labels; no learned value or return-to-go ranking."""

import numpy as np

def mine_step_reward_pairs(trajectories, state_mean, state_std, max_distance=0.5):
    """Rank original r[t] at each time, then match nearest current states.

    Upper/lower 30% groups, same timestep and equal episode horizon, RMSE <= 0.5.
    Future/past rewards and actions never enter the ranking or distance metric.
    Return statistics are diagnostics only and cannot affect selected pairs.
    """
    if not trajectories or not np.isfinite(max_distance) or max_distance <= 0:
        raise ValueError("Expected trajectories and a positive finite state cutoff")
    if not np.isfinite(state_mean).all() or not np.isfinite(state_std).all():
        raise ValueError("Nonfinite state normalization")
    if not np.all(state_std > 0):
        raise ValueError("State standard deviation must be positive")
    lengths = [len(t["actions"]) for t in trajectories]
    if min(lengths) <= 0 or len(set(lengths)) != 1:
        raise ValueError("Step-reward HCMR matching requires equal, nonempty horizons")
    states = np.stack([(t["observations"] - state_mean) / state_std
                       for t in trajectories])
    rewards = np.stack([t["rewards"] for t in trajectories])
    if rewards.shape != (len(trajectories), lengths[0]):
        raise ValueError("Reward and action lengths differ")
    if not np.isfinite(states).all() or not np.isfinite(rewards).all():
        raise ValueError("Nonfinite state or reward")
    pairs, distances, gaps = [], [], []
    candidates = 0
    skipped_times = 0
    for step in range(lengths[0]):
        low, high = np.quantile(rewards[:, step], [0.3, 0.7])
        if low >= high:
            skipped_times += 1
            continue
        good = np.flatnonzero(rewards[:, step] >= high)
        bad = np.flatnonzero(rewards[:, step] <= low)
        distance = np.sqrt(((states[good, step, None] - states[bad, step]) ** 2)
                           .mean(-1))
        nearest = distance.argmin(axis=1)
        candidates += len(good)
        for row, column in enumerate(nearest):
            if distance[row, column] <= max_distance:
                pos, neg = good[row], bad[column]
                pairs.append((pos, step, neg, step))
                distances.append(distance[row, column])
                gaps.append(rewards[pos, step] - rewards[neg, step])
    if not pairs:
        raise ValueError("No supported step-reward pairs; do not relax cutoff silently")
    pairs = np.asarray(pairs, dtype=np.int64)
    distances = np.asarray(distances, dtype=np.float32)
    gaps = np.asarray(gaps, dtype=np.float32)
    rtgs = np.cumsum(rewards[:, ::-1], axis=1, dtype=np.float64)[:, ::-1]
    stats = {
        "label": "original_single_step_reward",
        "group_quantiles": [0.3, 0.7],
        "num_pairs": len(pairs),
        "candidate_high_states": candidates,
        "coverage": len(pairs) / candidates,
        "covered_good_trajectories": len(np.unique(pairs[:, 0])),
        "covered_bad_trajectories": len(np.unique(pairs[:, 2])),
        "state_rmse_mean": float(distances.mean()),
        "state_rmse_max": float(distances.max()),
        "reward_gap_mean": float(gaps.mean()),
        "reward_gap_min": float(gaps.min()),
        "reward_gap_max": float(gaps.max()),
        "remaining_return_order_disagreement_fraction": float(np.mean(
            rtgs[pairs[:, 0], pairs[:, 1]] <= rtgs[pairs[:, 2], pairs[:, 3]])),
        "skipped_tied_timesteps": skipped_times,
    }
    return pairs, distances, stats
