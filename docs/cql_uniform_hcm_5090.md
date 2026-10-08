# HCM uniform redistribution control, three paired seeds at 500k

User-requested on 2026-10-08: add **HalfCheetah-medium-v2** uniform-return CQL
controls with policy/evaluation seeds **1, 11, 12**, **500,000 updates each**.
The explicit 500k request supersedes the proposed 100k budget for this new HCM
campaign only. Seeds intentionally pair with the existing 5090 delayed, Shapley
and original-reward controls; the Slurm policy seed 0 remains separate.

## Reward and implementation

For each of the **1000 complete, 1000-step trajectories**, distribute its total
return equally: `r[t] = R / 1000`. Use the same float64 totals as the existing HCM
Shapley fit, from `cql-shapley-hcm-v1-seed1-5090-20261007/predictor/input.npz`:
SHA-256 `49ec75e6889a4fca67269dc55fb43794dac9742215ee8cd0295a5ccec2aaa6a9`.
The input is already on the 5090. Verify its hash against that run's
`preparation_sha256` and verify its features, boundaries and float32 terminal
labels against the frozen delayed baseline `cql-corl-delayed-hc-seed1-5090-20261007`.

Reuse the audited uniform conversion from `scripts/cql_uniform_5090/prepare.py`:
zero segment contributions to `conserved_rewards`, with only the same float32
rounding correction at the final step. Fit no predictor and read no original
within-trajectory dense rewards. Preserve all non-reward transition arrays.
The HCM-specific preparation, queue and logging live in independent entrypoints;
the active HCMR campaign and all historical code and results remain untouched.
Copy the frozen CQL algorithm and trainer byte for byte. Only the selected YAML's
seed, update budget and logging identifiers change, along with the reward data.

Keep batch=256, discount=.99, CQL alpha=10, actor lr=3e-5, critic lr=3e-4,
state normalization, no reward normalization, and 10 original-environment
evaluation episodes every 5000 updates. Formal runs start from scratch after
separate 100-update CUDA preflights. No early stopping based on observed scores.

## Matched comparisons and reporting

Report the mean of the last ten evaluations (455k..500k), final at 500k, and
best through 500k separately. Preserve the full learning curves and checkpoints.
Also record separate 100k summaries: last-ten mean over 55k..100k, final at 100k,
and best through 100k. W&B summary keys use `result/100k/` and `result/500k/`;
the original `result/` keys refer to the full 500k run. These budget summaries
are finalized after the full run; individual evaluations stream during training.

Existing Shapley/delayed seeds 11/12 and dense controls have 100k budgets. Compare
all three paired seeds **only through 100k**; do not compare this control's 500k
best to another method's 100k best. Existing seed 1 delayed/Shapley runs additionally
support a 500k comparison. These three policy seeds share a fixed reward dataset;
they do not quantify variation across newly fitted Shapley predictors. Return
conservation is undiscounted and does not imply discounted policy invariance.

## Queue and W&B

Predecessor: `cql-uniform-hcmr-seeds1-11-12-100k-5090-20261008`.
Wait for all three predecessor runs to complete successfully and for matching,
finished W&B verification receipts. A failed or unfinished predecessor blocks
admission; a changed predecessor plan/protocol is an error. This gate also applies
to HCM CUDA preflights. Freeze the predecessor plan hash before queue launch.

After that gate opens, admit each job individually subject to fresh resource
checks: at most three own jobs and nine total compute PIDs, <=85% GPU utilization,
3GiB VRAM per new/pending job plus 4GiB reserve, and 4GiB host RAM per new/pending
job plus 8GiB reserve. Poll every 30s and recheck after each preflight. Count foreign
GPU processes and own pending CUDA contexts. Never alter other users' processes.
At preparation the HCMR queue still had two running jobs and GPU utilization
was about 92%; HCM should wait instead of adding to that workload.

Use persistent tmux `cql-uniform-hcm-500k-20261008`. Preserve attempts; refuse to
reuse an existing queue launch. Source, configuration and data hashes are checked
before running. Independent observers stream W&B telemetry and can be restarted
without restarting training. Verify all 5000 training markers, 100 evaluations,
100k/500k summaries, finished state and final checkpoint size before marking the
campaign complete; repair only missing or mismatched uploads from local originals.

Output: `results/cql-uniform-hcm-seeds1-11-12-500k-5090-20261008`.
W&B project: `2820402607-shandong-university/CORL-DDR`.

- `CQL-CORL-HCM-uniform-matched-v1-seed1-500k-5090`
- `CQL-CORL-HCM-uniform-matched-v1-seed11-500k-5090`
- `CQL-CORL-HCM-uniform-matched-v1-seed12-500k-5090`

Run from the immutable archive of the published revision:

```bash
python -m scripts.cql_uniform_medium_5090.prepare --project "$PROJECT" \
  --predictor-input "$PROJECT/results/cql-shapley-hcm-v1-seed1-5090-20261007/predictor/input.npz" \
  --code-revision PUBLISHED_COMMIT
bash scripts/cql_uniform_medium_5090/launch.sh
```

Preserve historical experiment code, configurations, checkpoints and results in
future work. Keep runtime artifacts, datasets, environments and secrets out of
Git. Publish scoped changes after tests and CI on a clean tracked export; use
normal fast-forward updates so concurrent server work is preserved.
