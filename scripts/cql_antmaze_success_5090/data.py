"""Success/timeout fragments over exactly the default D4RL retained transitions."""

import numpy as np

MAX_SEGMENTS = 20


def partition(raw):
    rewards = np.asarray(raw["rewards"], dtype=np.float64).reshape(-1)
    terminals = np.asarray(raw["terminals"]).reshape(-1)
    timeouts = np.asarray(raw["timeouts"]).reshape(-1)
    n = len(rewards)
    if (n < 2 or len(terminals) != n or len(timeouts) != n
            or not np.isfinite(rewards).all()
            or not np.isin(terminals, [0, 1]).all()
            or not np.isin(timeouts, [0, 1]).all()):
        raise ValueError("Invalid raw boundary arrays")
    # The official loader discards the last raw row and all timeout rows.
    index = np.flatnonzero(~timeouts[:-1].astype(bool))
    boundaries = terminals.astype(bool) | timeouts.astype(bool)
    ends = np.unique(np.r_[np.flatnonzero(boundaries), n - 1])
    offsets, groups, raw_bounds = [0], [], []
    start, group, empty = 0, 0, 0
    for end in ends:
        stop = int(np.searchsorted(index, end, side="right"))
        if stop > offsets[-1]:
            offsets.append(stop)
            groups.append(group)
            raw_bounds.append([start, int(end)])
        else:
            empty += 1
        if timeouts[end]:
            group += 1
        start = int(end) + 1
    if offsets[-1] != len(index) or len(offsets) < 2:
        raise ValueError("Fragments do not cover official transitions")
    offsets = np.asarray(offsets, dtype=np.int64)
    returns = np.add.reduceat(rewards[index], offsets[:-1])
    return dict(index=index, offsets=offsets, returns=returns,
                groups=np.asarray(groups, dtype=np.int64),
                raw_bounds=np.asarray(raw_bounds, dtype=np.int64),
                skipped_empty_fragments=empty)


def lengths(horizon):
    if horizon < 1:
        raise ValueError("Empty fragments must be omitted")
    count = min(MAX_SEGMENTS, horizon)
    return np.asarray([len(x) for x in np.array_split(np.arange(horizon), count)])


def conserved(contributions, total, horizon):
    parts = lengths(horizon)
    phi = np.asarray(contributions, dtype=np.float64)
    if phi.shape != parts.shape:
        raise ValueError("One contribution per valid segment is required")
    correction = float(total - phi.sum())
    values = np.repeat(phi / parts, parts) + correction / horizon
    result = values.astype(np.float32)
    rounding = float(total - result.astype(np.float64).sum())
    tolerance = 2 * np.finfo(np.float32).eps * max(np.abs(values).sum(), 1.0)
    if not np.isfinite(values).all() or abs(rounding) > tolerance:
        raise ValueError("Invalid redistribution")
    result[-1] = np.float32(float(result[-1]) + rounding)
    if abs(result.astype(np.float64).sum() - total) >= 1e-4:
        raise ValueError("Fragment return is not conserved")
    return result, correction, rounding


def uniform(part):
    result = np.empty(len(part["index"]), dtype=np.float32)
    for start, end, total in zip(part["offsets"][:-1], part["offsets"][1:],
                                 part["returns"]):
        result[start:end] = conserved(np.zeros(len(lengths(end - start))),
                                      total, end - start)[0]
    return result


def grouped_folds(groups, count=5):
    """Keep all success fragments from a parent timeout rollout in one fold."""
    unique = np.unique(groups)
    if len(unique) < count + 2:
        raise ValueError("Too few parent groups for independent cross fitting")
    for fold, held_groups in enumerate(np.array_split(unique, count)):
        remaining = np.setdiff1d(unique, held_groups)
        remaining = np.random.default_rng(100 + fold).permutation(remaining)
        nv = max(1, int(np.ceil(len(remaining) * 0.2)))
        split = dict(fold=fold, training_seed=1000 + fold,
                     validation_seed=2000 + fold, split_seed=100 + fold)
        for name, selected in [("held", held_groups),
                               ("validation", remaining[:nv]),
                               ("train", remaining[nv:])]:
            split[name] = np.flatnonzero(np.isin(groups, selected)).tolist()
            split[name + "_groups"] = selected.tolist()
            if not split[name]:
                raise ValueError("Empty cross-fitting split")
        yield split


def inputs(features, offsets, returns, training):
    """Training-only scaling; padded zeros are never used to estimate statistics."""
    selected = np.zeros(len(features), dtype=bool)
    for i in training:
        selected[offsets[i]:offsets[i + 1]] = True
    frames = features[selected].astype(np.float64)
    mean, std = frames.mean(0), frames.std(0).clip(1e-3)
    del frames
    stats = dict(mean=mean, std=std, target_mean=float(returns[training].mean()),
                 target_std=float(returns[training].std().clip(1e-3)))
    # A common width is necessary, but padding is never a coalition player.
    padded_steps = max(int(lengths(t).max()) for t in np.diff(offsets))
    width = features.shape[1]
    x = np.zeros((len(returns), MAX_SEGMENTS,
                  padded_steps * (width + 1) + 1), dtype=np.float32)
    valid = np.zeros((len(returns), MAX_SEGMENTS), dtype=np.float32)
    for i, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
        parts = lengths(end - start)
        valid[i, :len(parts)] = 1
        cursor = start
        for j, length in enumerate(parts):
            x[i, j, :length * width] = (
                (features[cursor:cursor + length] - mean) / std).ravel()
            x[i, j, padded_steps * width:padded_steps * width + length] = 1
            x[i, j, -1] = j / max(len(parts) - 1, 1)
            cursor += length
    return x, valid, stats


def masks(rng, valid):
    """For a one-player game, the only nonempty coalition is the full coalition."""
    result = np.zeros_like(valid)
    for i, count in enumerate(valid.sum(1).astype(int)):
        size = 1 if count == 1 else int(rng.integers(1, count))
        result[i, rng.choice(count, size, replace=False)] = 1
    return result
