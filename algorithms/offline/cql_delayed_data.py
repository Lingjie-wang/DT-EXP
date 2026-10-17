"""Terminal-return data for CQL; preserve timeout rewards and block reset bootstrap."""

from typing import Dict, Tuple

import numpy as np

def terminal_return_dataset(raw: Dict[str, np.ndarray]) -> Tuple[Dict, Dict]:
    rewards = np.asarray(raw["rewards"], dtype=np.float64).reshape(-1)
    terminals = np.asarray(raw["terminals"], dtype=bool).reshape(-1)
    timeouts = np.asarray(raw["timeouts"], dtype=bool).reshape(-1)
    observations = np.asarray(raw["observations"], dtype=np.float32)
    actions = np.asarray(raw["actions"], dtype=np.float32)
    n = len(rewards)
    if not n or any(len(x) != n for x in [terminals, timeouts, observations, actions]):
        raise ValueError("Dataset fields must have equal nonzero lengths")
    ends = terminals | timeouts
    if not ends[-1]:
        raise ValueError("Incomplete final trajectory; refusing to invent its return")
    if not all(np.isfinite(x).all() for x in [rewards, observations, actions]):
        raise ValueError("Non-finite dataset values")
    end_indices = np.flatnonzero(ends)
    starts = np.r_[0, end_indices[:-1] + 1]
    totals = np.array([rewards[a:b + 1].sum() for a, b in zip(starts, end_indices)])
    delayed = np.zeros(n, dtype=np.float32)
    delayed[end_indices] = totals.astype(np.float32)
    # Use recorded successors where available. Terminal placeholders are never
    # bootstrapped from; do not join the end of one episode to the next reset.
    next_states = np.zeros_like(observations)
    if "next_observations" in raw:
        recorded = np.asarray(raw["next_observations"], dtype=np.float32)
        if recorded.shape != observations.shape:
            raise ValueError("Recorded next observations have the wrong shape")
        next_states[~ends] = recorded[~ends]
    else:
        live = np.flatnonzero(~ends)
        next_states[live] = observations[live + 1]
    if not np.isfinite(next_states).all():
        raise ValueError("Non-finite next observations")
    np.testing.assert_allclose(delayed[end_indices], totals, rtol=1e-6, atol=1e-5)
    data = dict(
        observations=observations.copy(), actions=actions.copy(), rewards=delayed,
        next_observations=next_states, terminals=ends.astype(np.float32),
    )
    audit = dict(
        transitions=n, episodes=len(end_indices),
        original_terminals=int(terminals.sum()), timeouts=int(timeouts.sum()),
        bootstrap_stops=int(ends.sum()),
        nonzero_rewards=int(np.count_nonzero(delayed)),
        nonterminal_nonzero_rewards=int(np.count_nonzero(delayed[~ends])),
        return_min=float(totals.min()), return_max=float(totals.max()),
        return_mean=float(totals.mean()),
        max_episode_return_rounding_error=float(
            np.max(np.abs(delayed[end_indices] - totals))),
        trajectory_length_min=int((end_indices - starts + 1).min()),
        trajectory_length_max=int((end_indices - starts + 1).max()),
        timeout_policy="retain transition and terminal return; stop bootstrap",
        next_state_policy="recorded/shifted within episode; zero at all episode ends",
    )
    return data, audit
