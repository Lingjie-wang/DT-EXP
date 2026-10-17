"""Timeout-only AntMaze trajectories and total-only segment redistribution."""

import numpy as np

def segment_lengths(horizon, count=20):
    if horizon < count:
        raise ValueError("Need at least one transition per segment")
    return np.array([len(part) for part in np.array_split(np.arange(horizon), count)])


def segment_inputs(normalized):
    n, horizon, width = normalized.shape
    lengths = segment_lengths(horizon)
    padded = np.zeros((n, 20, int(lengths.max()), width), dtype=np.float32)
    valid = np.zeros((n, 20, int(lengths.max())), dtype=np.float32)
    offset = 0
    for i, length in enumerate(lengths):
        padded[:, i, :length] = normalized[:, offset:offset + length]
        valid[:, i, :length] = 1
        offset += length
    position = np.broadcast_to(np.arange(20)[None, :, None] / 19, (n, 20, 1))
    return np.concatenate([padded.reshape(n, 20, -1), valid, position], -1).astype(
        np.float32)


def conserved_rewards(contributions, total, horizon):
    phi = np.asarray(contributions, dtype=np.float64)
    lengths = segment_lengths(horizon, len(phi))
    correction = float(total - phi.sum())
    values = np.repeat(phi / lengths, lengths) + correction / horizon
    result = values.astype(np.float32)
    rounding = float(total - result.astype(np.float64).sum())
    tolerance = 2 * np.finfo(np.float32).eps * max(np.abs(values).sum(), 1.0)
    if not np.isfinite(values).all() or abs(rounding) > tolerance:
        raise ValueError("Invalid reward conservation")
    result[-1] = np.float32(float(result[-1]) + rounding)
    if abs(result.astype(np.float64).sum() - total) >= 1e-3:
        raise ValueError("Return conservation exceeds tolerance")
    return result, correction, rounding


def prepare_arrays(raw, horizon):
    """Success terminals are deliberately absent from all prepared learner inputs."""
    rewards = np.asarray(raw["rewards"], dtype=np.float64).reshape(-1)
    timeouts = np.asarray(raw["timeouts"]).reshape(-1)
    states = np.asarray(raw["observations"], dtype=np.float32)
    actions = np.asarray(raw["actions"], dtype=np.float32)
    if (not len(rewards) or any(len(x) != len(rewards)
                               for x in [timeouts, states, actions])
            or not np.isin(timeouts, [0, 1]).all()
            or not all(np.isfinite(x).all() for x in [rewards, states, actions])):
        raise ValueError("Invalid AntMaze arrays")
    ends = np.flatnonzero(timeouts)
    starts = np.r_[0, ends[:-1] + 1]
    if not len(ends) or not np.all(ends - starts + 1 == horizon):
        raise ValueError("Unexpected timeout trajectory lengths")
    stop = int(ends[-1]) + 1
    terminals = timeouts[:stop].astype(np.float32)
    successors = np.zeros_like(states[:stop])
    live = np.flatnonzero(terminals == 0)
    if "next_observations" in raw:
        recorded = np.asarray(raw["next_observations"], dtype=np.float32)
        if recorded.shape != states.shape:
            raise ValueError("Invalid recorded successor shape")
        successors[live] = recorded[live]
    else:
        successors[live] = states[live + 1]
    if not np.isfinite(successors).all():
        raise ValueError("Invalid next states")
    dense = rewards[:stop].astype(np.float32)
    totals = rewards[:stop].reshape(-1, horizon).sum(1)
    delayed = np.zeros(stop, dtype=np.float32)
    delayed[ends] = totals.astype(np.float32)
    uniform = np.concatenate([conserved_rewards(np.zeros(20), r, horizon)[0]
                              for r in totals])
    base = dict(observations=states[:stop].copy(), actions=actions[:stop].copy(),
                next_observations=successors, terminals=terminals)
    inputs = dict(features=np.concatenate([base["observations"], base["actions"]], 1)
                  .reshape(-1, horizon, states.shape[1] + actions.shape[1]),
                  returns=totals, starts=starts, ends=ends)
    audit = dict(episodes=len(ends), transitions=stop, trajectory_length=horizon,
                 raw_transitions=len(rewards), dropped_incomplete_tail=len(rewards)-stop,
                 boundary="timeouts only; retain boundary transition; stop bootstrap",
                 original_success_terminals_used=False,
                 next_states="recorded/shifted within timeout fragment; zero at end",
                 predictor_receives_dense_reward_timing=False,
                 predictor_inputs="state/action sequences and scalar trajectory totals",
                 return_min=float(totals.min()), return_max=float(totals.max()),
                 nonzero_return_trajectories=int(np.count_nonzero(totals)))
    for values in [dense, delayed, uniform]:
        np.testing.assert_allclose(values.reshape(-1, horizon).astype(np.float64).sum(1),
                                   totals, rtol=1e-6, atol=1e-4)
    return base, inputs, dict(dense=dense, delayed=delayed, uniform=uniform), audit
