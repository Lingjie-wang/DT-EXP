# HCMR uniform redistribution control, three paired seeds

Requested on 2026-10-08 to test whether making terminal rewards dense accounts
for the predictive-Shapley improvement. Run **only HalfCheetah-medium-replay-v2**,
policy/evaluation seeds **1, 11, 12**, **100,000 CQL updates each**, from scratch.
These seeds intentionally match the existing delayed, Shapley and original-reward
arms. Slurm's policy seed 0 is separate. The cancelled historical seed-0 uniform
arm remains cancelled; this is an independent, newly authorized campaign.

## Scientific definition

For each of the 202 complete 1000-step trajectories, set every reward to `R/1000`.
Read `R` from the **same frozen predictor input artifact used by Shapley** (hash
`8419eb1e4c845d9ed6bc9e6a1805cee23be45d04f88cd317645ccb98909c670d`). No predictor
is fitted, no Shapley contributions are computed, and no original within-trajectory
dense rewards are read. The existing `conserved_rewards` helper with zero segment
contributions gives this exact definition, including the same float32 rounding
correction at the final transition. That correction contains no model residual.

Verify predictor features and boundaries against the frozen delayed dataset, and
verify that totals cast to float32 equal its terminal labels. Use float64 totals
to match Shapley exactly; their maximum difference from the stored delayed float32
labels is already documented in the information audit. All non-reward transition
arrays, including terminal/timeout masks and next observations, remain identical.
Copy the frozen CQL algorithm, trainer and training entrypoint byte for byte. Only
reward arrays, policy/evaluation seed, update budget and logging identifiers change.

Budget follows the user's latest 100k setting. Existing Shapley seeds 11/12 reach
44.99/43.54 at 100k; seed 1 reaches 43.16 at 100k. These observations motivate a
matched initial 100k control, not early stopping based on a new control's score.
For existing seed-1 500k runs, use **only evaluations through 100k** in this comparison.
Primary metric: per-run mean of evaluations from 55k to 100k, then aggregate across
the three paired seeds. Also report final-at-100k and best-within-100k separately.
Keep batch=256, discount=.99, CQL alpha=10, actor lr=3e-5, critic lr=3e-4, state
normalization, no reward normalization, and 10 original-environment evaluation
episodes every 5000 updates. Undiscounted return conservation does not imply
discounted policy invariance. The comparison tests the full Shapley pipeline
against uniform redistribution; it does not prove causal segment contributions.

## Shared server scheduling and logging

At planning, the 5090 had six external compute processes (five using about 0.7GB
each and one using about 12.4GB), 15.2GB free VRAM and about 33GB available host RAM;
four GPU utilization samples were 55%, 68%, 77%, 58%. The previous all-CQL process
cap is unsuitable for this different background workload. This queue admits each
new job separately with a maximum of three own jobs and nine total compute PIDs.
Reserve 3GiB VRAM per new/pending job plus 4GiB free; reserve 4GiB host RAM per
new/pending job plus 8GiB free. Admission requires GPU utilization <=85%. Count
foreign GPU processes and own children that have not yet created a CUDA context.
Do not stop or modify other users' jobs. Check resources every 30s and again after
each independent 100-update CUDA preflight. When limits are reached, wait.

Training uses independent outputs and immutable source archives. The manager
remains alive until all successful runs have finished W&B verification, restarting
an interrupted observer without restarting training. The uniform-specific bridge
uses the existing observer, then checks the complete logged training-step markers,
all 20 evaluations, final summary, finished state and final checkpoint byte size.
It repairs only missing or mismatched online rows/files from the preserved local
data. This addresses the interrupted-upload behavior seen in the prior dense runs.

Output: `results/cql-uniform-hcmr-seeds1-11-12-100k-5090-20261008`.
W&B project: `2820402607-shandong-university/CORL-DDR`.

- `CQL-CORL-HCMR-uniform-matched-v1-seed1-100k-5090`
- `CQL-CORL-HCMR-uniform-matched-v1-seed11-100k-5090`
- `CQL-CORL-HCMR-uniform-matched-v1-seed12-100k-5090`

Prepare using the uploaded **unchanged** predictor input and the published archive:

```bash
python -m scripts.cql_uniform_5090.prepare --project "$PROJECT" \
  --predictor-input "$PROJECT/.runtime/cql-uniform-inputs-20261008/predictor_input.npz" \
  --code-revision PUBLISHED_COMMIT
bash scripts/cql_uniform_5090/launch.sh
```

Run in persistent tmux `cql-uniform-hcmr-100k-20261008`. The queue refuses to reuse
an existing launch; inspect failures before preparing an independent recovery.
Source, configuration and prepared data hashes are checked before running.
Preserve historical code, configurations, checkpoints and results in all future
work. Keep datasets, runtime state and secrets out of Git. Publish scoped changes
with a normal fast-forward update after tests and CI on a clean tracked export.
