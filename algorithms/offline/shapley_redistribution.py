"""Predictive segment Shapley; no causal or discounted-policy invariance claim."""

import numpy as np
import torch
from torch import nn

def folds(n, count=5):
    """Contiguous outer folds; validation never overlaps training or attribution."""
    for fold, held in enumerate(np.array_split(np.arange(n), count)):
        remaining = np.setdiff1d(np.arange(n), held)
        rng = np.random.default_rng(100 + fold)
        remaining = rng.permutation(remaining)
        nv = int(np.ceil(len(remaining) * 0.2))
        yield dict(fold=fold, train=remaining[nv:].tolist(),
                   validation=remaining[:nv].tolist(), held=held.tolist(),
                   split_seed=100 + fold, training_seed=1000 + fold,
                   validation_seed=2000 + fold)


def standardize(features, returns, training):
    """Fit only on the inner training trajectories, never validation/outer folds."""
    values = features[training].reshape(-1, features.shape[-1]).astype(np.float64)
    mean, std = values.mean(0), values.std(0).clip(1e-3)
    target_mean = float(returns[training].mean())
    target_std = float(returns[training].std().clip(1e-3))
    stats = dict(mean=mean, std=std, target_mean=target_mean, target_std=target_std)
    return ((features - mean) / std).astype(np.float32), stats


def segment_inputs(normalized, length=50):
    n, horizon, width = normalized.shape
    if horizon % length:
        raise ValueError("Only complete fixed-length segments are supported")
    k = horizon // length
    x = normalized.reshape(n, k, length * width)
    position = np.broadcast_to(np.arange(k)[None, :, None] / max(k - 1, 1),
                               (n, k, 1))
    return np.concatenate([x, position], axis=-1).astype(np.float32)


class SegmentReturnModel(nn.Module):
    def __init__(self, width, segments=20):
        super().__init__()
        self.segments = segments
        self.encoder = nn.Sequential(nn.Linear(width, 64), nn.ReLU(),
                                     nn.Linear(64, 64), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(65, 64), nn.ReLU(), nn.Linear(64, 1))

    def value(self, encoded, mask):
        # No cross-segment processing occurs before masking.
        pooled = (encoded * mask[..., None]).sum(-2) / self.segments
        count = mask.sum(-1, keepdim=True) / self.segments
        result = self.head(torch.cat([pooled, count], dim=-1)).squeeze(-1)
        # In standardized units, empty-set value is the inner-training mean.
        return torch.where(count.squeeze(-1) == 0, torch.zeros_like(result), result)

    def forward(self, x, mask):
        return self.value(self.encoder(x), mask)


def random_masks(rng, n, k):
    counts = rng.integers(1, k, size=n)
    masks = np.zeros((n, k), dtype=np.float32)
    for i, count in enumerate(counts):
        masks[i, rng.choice(k, count, replace=False)] = 1
    return masks


def permutation_shapley(value, k, permutations):
    """value accepts a batch of coalition masks; return per-permutation estimates."""
    permutations = np.asarray(permutations)
    if not np.all(np.sort(permutations, axis=1) == np.arange(k)):
        raise ValueError("Each row must be a complete permutation")
    m = len(permutations)
    masks = np.zeros((m, k + 1, k), dtype=np.float32)
    for j in range(k):
        masks[:, j + 1] = masks[:, j]
        masks[np.arange(m), j + 1, permutations[:, j]] = 1
    values = np.asarray(value(masks.reshape(-1, k)), dtype=np.float64).reshape(m, k + 1)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite coalition value")
    estimates = np.zeros((m, k), dtype=np.float64)
    np.put_along_axis(estimates, permutations, np.diff(values, axis=1), axis=1)
    return estimates, values[0, 0], values[0, -1]


def conserved_rewards(contributions, total, segment_length=50):
    contributions = np.asarray(contributions, dtype=np.float64)
    horizon = len(contributions) * segment_length
    correction = float(total - contributions.sum())
    result64 = np.repeat(contributions / segment_length, segment_length)
    result64 += correction / horizon
    result = result64.astype(np.float32)
    # Only fix float32 rounding; never put the model residual on the final step.
    rounding = float(total - result.astype(np.float64).sum())
    tolerance = 2 * np.finfo(np.float32).eps * max(np.abs(result64).sum(), 1.0)
    if abs(rounding) > tolerance:
        raise ValueError("Non-rounding conservation error")
    result[-1] = np.float32(float(result[-1]) + rounding)
    if not np.isfinite(result).all():
        raise ValueError("Non-finite redistributed reward")
    return result, correction, rounding
