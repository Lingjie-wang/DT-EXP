# CQL predictive Shapley pilot v1

## Purpose and limits

User-authorized pilot on HalfCheetah-medium-replay-v2, seed 0. Compare the existing
terminal-delayed CQL run with uniform redistribution, predictive segment Shapley,
and a matched original-dense-reward reference. This implements the supplied survey's
framework, not a reproduction of a fully specified published algorithm. There is
no separate learned reward model g_theta in v1. Attribution is predictive information
credit, not a causal estimate of executing subsets of actions. One policy seed does
not establish reliable superiority. Preserving undiscounted return does not preserve
the discounted CQL objective at gamma=0.99.

Historical implementations, datasets, checkpoints and results must be preserved.
New code is isolated under scripts/cql_shapley and new campaign output directories.
Git publication must build on latest remote main, preserving the 5090 server's work.

## Fixed design

The 202 complete 1000-step trajectories are divided into 20 consecutive segments
of 50 steps. Each flattened segment contains normalized states and actions plus
position k/19. An MLP (input -> 64 ReLU -> 64 ReLU) encodes each segment independently.
For a coalition, selected embeddings are summed and divided by 20; selected fraction
is concatenated, then a 65 -> 64 ReLU -> 1 MLP predicts standardized total return.
Empty value is explicitly zero in standardized units (inner-training mean in raw units).
No attention or other cross-segment computation occurs before masking.

Five outer folds use contiguous trajectory indices in original data order. For each
fold, the remaining indices are permuted with seed 100+fold; ceil(20%) are validation.
Normalization uses only inner training data. Training seed is 1000+fold, fixed
validation-mask seed 2000+fold, and attribution seed 3000+trajectory. Adam lr=1e-3,
weight_decay=1e-4, batch=32, maximum 300 epochs, patience=30. Best validation checkpoint
is restored. Loss is half full-input MSE, half masked-input MSE. Each training example
gets one random mask: coalition size uniform from 1..19, subset uniform conditional
on size. Validation averages eight fixed masks per trajectory and full-input MSE.
Masked-input targets remain the observed total return, defining a predictive
conditional-value surrogate; they are not observed counterfactual returns.

Every trajectory is attributed by the model trained without its entire outer fold.
Use 128 uniform random permutations and all their prefix coalitions. Cache segment
embeddings; batch coalition prediction in chunks of 512. Preserve signed contributions.
Reward at each step in segment k is phi_k/50 + (R - sum(phi))/1000. Empty-set baseline
and full-input prediction residual are therefore spread uniformly, not hidden at
the final transition. A final-step correction is allowed only for float32 rounding.
Save all permutation contributions, fold checkpoints, split-half stability,
Monte Carlo standard errors, predictions, residuals and reward distributions.

Attribution input contains only state/action features, trajectory totals and boundaries.
The predictor does not load per-step rewards or dense-reference data. The original HDF5
is only read during preparation, where totals and the separate dense arm are produced.
All arms have identical non-reward transition fields and terminal-or-timeout handling,
including retained final transitions and no bootstrap across resets. Their arrays are
checked against the frozen delayed baseline. Dense reference is not claimed as an upper
bound and differs from default CORL timeout filtering to make the comparison matched.

## Prediction gate and CQL

Aggregate outer-fold prediction MSE must be strictly below the MSE of corresponding
inner-training means. If it fails, keep diagnostics but do not create/submit the Shapley
training dataset. Uniform and dense controls can continue. Never tune on CQL eval scores.
Split-half relative L1 attribution disagreement is reported, not used for hidden tuning.

Unchanged CORL CQL code/config scientific settings: 1M updates, batch 256, gamma=.99,
CQL alpha=10, actor lr=3e-5, critic lr=3e-4, state normalization, no reward normalization,
buffer capacity 10M, eval every 5000 updates over 10 episodes with seed 0. The independent
runner copies the previous runner and construction order, with explicit arm selection
and a Shapley gate check. Model, losses, updates and evaluation RNG behavior are unchanged.
Each job: 1 GPU, 2 CPUs, 16GB, 72h. Run 100 full-batch updates and one eval episode in an
independent preflight process, then formal training from a fresh process. Existing jobs
are not cancelled or reprioritized. GPU preflight is pending while jobs are queued.

Reuse .runtime/lpt-official-env (Torch2.0.1+cu118, NumPy1.24.4). GPU MuJoCo setup is copied
from the prior CQL job; login CPU MuJoCo compilation lacks GL/osmesa.h. Prediction fitting
uses CPU PyTorch only, two threads, and never imports MuJoCo or changes shared packages.

## Execution and records

From the new worktree, with PYTHONPATH=. and the above Python:

```bash
python scripts/cql_shapley/prepare.py --root CAMPAIGN --baseline DELAYED_CAMPAIGN --hdf5 DATA
PYTHONPATH=CAMPAIGN/source python CAMPAIGN/source/scripts/cql_shapley/fit_redistribute.py \
  --root CAMPAIGN --device cpu
python CAMPAIGN/source/scripts/cql_shapley/submit.py --root CAMPAIGN \
  --bridge-python /path/to/adt-delayed/bin/python
```

Preparation freezes source and raw/prepared data hashes and rejects existing campaigns.
Fit rejects an existing fit directory. Submission persists intent/job IDs and refuses
ambiguous retries; repeated successful submissions reuse job IDs, avoiding duplicates.
Training verifies all frozen source/data hashes. Local W&B bridges run on the login node,
read back startup configuration and final results, and distinguish Slurm queue status
from W&B's online process state. Axes use completed_updates.

- CQL-CORL-HCMR-uniform-v1-seed0-1M
- CQL-CORL-HCMR-shapley20x128-v1-seed0-1M
- CQL-CORL-HCMR-dense-matched-v1-seed0-1M

Primary outcome: evaluation at 1M updates. Secondary: learning curves and mean of last
ten evaluations, never the peak as a replacement for final performance. Compare Shapley
against uniform first; improvement over delayed alone does not establish attribution value.
Dataset/model/generated artifacts remain ignored; publish only scoped code, tests and
documentation, validate clean-export CI, and inspect GitHub Actions after normal publication.

Tests: python -m unittest discover -s tests -p 'test_shapley_redistribution.py' -v
