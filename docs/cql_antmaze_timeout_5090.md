# AntMaze CQL: four rewards, two datasets, one paired seed, 300k

Requested on 2026-10-09: queue delayed, predictive-Shapley, uniform and original
reward CQL on `antmaze-umaze-v2` and `antmaze-umaze-diverse-v2`. Eight independent
runs, policy/evaluation **seed 21**, **300,000 updates each**. Neither server had
an AntMaze protocol and Slurm had no queued jobs at planning. Seed 21 also avoids
previous CQL policy seeds 0/1/11/12. Predictor cross-fit seeds remain the previous
fixed 100+/1000+/2000+ scheme, and attribution seeds remain 3000+trajectory index.

## Explicitly selected trajectory definition

The user selected **timeouts-only long fragments**, rather than splitting on
success terminals. Official HDF5 boundaries give:

| Dataset | Complete fragments | Length | Retained transitions | Discarded final tail |
| --- | ---: | ---: | ---: | ---: |
| antmaze-umaze-v2 | 1426 | 701 | 999626 | 374 |
| antmaze-umaze-diverse-v2 | 999 | 1001 | 999999 | 1 |

These are actual recorded lengths, not the Gym evaluation horizon. Evaluation
still uses the official environments and their original success/reward behavior.
Download URLs and SHA-256 hashes are recorded in each input audit. Verify the
expected population and lengths before preparing any rewards.

All four arms retain the same complete fragments and all timeout transitions.
Stop CQL bootstrap at timeouts only; use recorded successors where available,
otherwise the following observation within a fragment; use a zero placeholder
at the masked fragment end. **Do not read or pass original success terminals**:
they expose per-step success information, even after delaying the reward array.
This differs from the unmodified official `qlearning_dataset` terminal handling;
these are matched timeout-fragment reward controls, not an unchanged official
AntMaze reproduction. Preserve the original algorithm/config files and all
historical runs. Do not silently mix results from other boundary definitions.

## Rewards and information isolation

For a fragment of length T with its original scalar sum R:

- Original (`dense`): retain the original 0/1 per-step rewards.
- Delayed: zero before the final transition, then R.
- Uniform: R/T at each step, with only final-step float32 rounding correction.
- Shapley: cross-fit the same segment-return model on states/actions and scalar
  totals, estimate 128 permutation contributions for each of 20 segments, spread
  each contribution across its segment, then spread the residual uniformly
  across all T transitions. Only rounding correction is applied at the last step.

The predictor artifact contains only features, scalar totals and timeout
boundaries. It contains no dense reward sequence, success flags or goal metadata.
Tests replace the entire dense sequence with terminal-only totals and flip/remove
all success terminals: predictor inputs and the non-original reward arms remain
unchanged. Separate tests verify held-fold features/labels cannot change model
weights or checkpoint selection. Shared non-reward transition arrays and trajectory
return conservation are checked for all four arms.

Apply the official AntMaze CQL affine transform **10*r - 5 after redistribution**
to every arm, through the unchanged official `modify_reward` function. Thus
all four learner-return sums equal 10*R - 5*T. In the delayed arm the learner sees
a fixed -5 before the endpoint; that known constant reveals no dense success
location. Predictor labels and saved reward arrays use the original raw scale.

## Adaptation of the existing predictive-Shapley method

The old implementation supports 1000 steps = 20x50. Preserve it unchanged.
The new independent implementation splits 701/1001 into 20 contiguous, near-equal
segments (35/36 or 50/51 steps). Pad shorter segments with zeros **after**
standardization, add explicit valid-step indicators, and keep segment position.
There is no cross-segment processing before coalition masking. Keep the existing
64-unit ReLU encoder/head, five outer folds, training-only standardization,
inner validation, 300-epoch cap, patience30, Adam lr1e-3/weight decay1e-4,
batch32, eight validation masks and 128 attribution permutations. Contiguous
outer folds avoid random interleaving of adjacent collection fragments.

The held-out full-return MSE must beat its fold's training-mean predictor in the
aggregate before Shapley CQL is admitted. Record all diagnostics even if this gate
fails; mark that arm blocked by the prediction gate and continue the other arms.
Do not silently substitute uniform rewards or tune the predictor after viewing
policy scores. This is especially relevant to the very sparse diverse dataset.

## CQL configuration and evaluation

Copy `algorithms/offline/cql.py` and the existing CQL constructor byte for byte.
Use official AntMaze YAMLs: batch256, discount.99, no state normalization,
reward normalization with scale10/bias-5, actor lr1e-4, critic lr3e-4,
CQL alpha5 with Lagrange target gap.8, max target backup, orthogonal initialization,
and buffer10M. Preserve even upstream constructor details such as critic-depth
asymmetry; do not insert an unrelated algorithm fix into the baseline.

Configuration changes are seed21, budget300k, logging identifiers, and evaluation
frequency10k instead of50k for a more informative learning curve. Keep official
100 evaluation episodes, giving30 evaluations per run. Report final at300k,
best within300k and mean of the last10 evaluations (210k..300k) separately.
Single-seed results are exploratory; there is no across-seed uncertainty estimate.
Save checkpoints at100k,200k,300k and stream original-environment D4RL scores.

## Persistent shared-server queue

Campaign: `results/cql-antmaze-timeout-four-rewards-seed21-300k-5090-20261009`.
Tmux session: `cql-antmaze-four-rewards-300k-20261009`.
Predecessor is the completed HCM uniform500k campaign. Require its training and
W&B receipts before admission. First fit the two predictors sequentially on GPU,
then admit CQL jobs in dataset/reward order, at most two own jobs and eight total
compute PIDs. Planning observed six foreign GPU processes and about15GiB free.
Reserve5GiB per new/pending GPU job plus4GiB free, and6GiB host RAM per new/pending
job plus8GiB free. Admission requires <=85% GPU utilization. Poll every30s and
recheck after each separate100-update CUDA/one-episode preflight. Leave foreign
jobs untouched. The manager refuses to overwrite/replay an existing launch.

`initial_plan.json` is immutable. Before training, materialize successful Shapley
reward files, update their protocol hashes in `plan.json`, and freeze a
`queue/materialized_plan.json` copy. Preserve failed predictor artifacts.
Observers can restart independently. Verify all3000 training markers,30 evals,
final/best/last10 summaries, final checkpoint size and W&B finished state before
marking successful jobs fully complete.

W&B: `2820402607-shandong-university/CORL-DDR`.
Names: `CQL-CORL-{env}-{reward}-timeout-v1-seed21-300k-5090`, where `{env}` is
one of the two full environment names and `{reward}` is `delayed`,
`shapley20x128`, `uniform` or `dense`.

Run from the published immutable archive:

```bash
python -m scripts.cql_antmaze_5090.prepare --project "$PROJECT" \
  --inputs "$PROJECT/.runtime/cql-antmaze-inputs-20261009" \
  --code-revision PUBLISHED_COMMIT
bash scripts/cql_antmaze_5090/launch.sh
```

Use an isolated worktree based on freshly fetched main. Validate a clean tracked
export before publishing; update GitHub with an expected-head, normal fast-forward
push so the other server's commits are preserved. Keep all data, models, generated
run files, environments and secrets out of Git. Back up nonsecret launch/protocol
records locally. Future work must preserve all historical experiment code,
configurations, checkpoints and results.
