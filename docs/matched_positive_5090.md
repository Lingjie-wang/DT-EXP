# Matched positive-imitation B versus original-reward RTG C

This new, independently named experiment tests whether the negative stopping gate
adds value beyond imitating the same positive actions. Historical code, configs,
checkpoints, results and queues remain unchanged. This is our ablation, not a
paper reproduction. Only one formal seed-0 B is scheduled; reuse the completed
RTG C and ordinary DT instead of retraining them or adding seeds automatically.

## Exactly matched comparison

- Dataset: HalfCheetah-medium-replay-v2, ORIGINAL per-step rewards, gamma 1.
- Reuse `mine_dense_pairs` and `dense_preference_batch` unchanged. The exact same
  7,306 ordered pairs, distances and RTG gaps must hash identically to RTG C.
  Positives/negatives are the upper/lower 30% of observed RTG at each timestep;
  nearest-state standardized RMSE <= 0.5 at the same timestep. No prefix mining.
- B: `L_DT + 0.05 * mean(d_positive)` over ALL sampled pairs.
- C reference: `L_DT + 0.05 * mean(relu(d_positive - stopgrad(d_negative) + 0.05))`.
  The only algorithmic change is removing the negative gate. B has no negative
  gradient, margin effect, adaptive weighting or active-pair normalization.
  The nominal coefficient is identical; B generally activates more positive
  gradients. That difference is the gate ablation, not a matched-gradient-norm
  comparison. Negatives remain in the batch solely for diagnostics and unchanged
  data/forward structure.
- Keep the ordinary and auxiliary forwards, dropout RNG draws, private pair RNG
  (seed 10000), data-loader seed, worker count and sampling with replacement.
  `train/active_fraction` is 1 for B. `diagnostic/c_gate_active_fraction` reports
  what C's gate would do on B's CURRENT predictions, without controlling B.
- Seed 0 from random initialization. Ordinary batch 4096, auxiliary batch 256,
  context 20, 128-wide / 3-layer / 1-head DT, LR 0.0008, warmup 10k, math SDPA.
  Architecture, optimizer, clipping, dropout and normalization match RTG C.
- The ordinary DT branch uses recorded RTGs; auxiliary desired return is
  `12000 - observed_positive_prefix_rewards`. Evaluate initial RTG 12000 with
  original-reward updates, seed 42, 100 episodes every 10k updates, train to 100k.
  Primary endpoint is final 100k score; secondary is mean of 60k/80k/100k.
  No best-checkpoint selection or early stopping based on policy performance.
- Log training every 100 updates; checkpoint every 5k. This entry point requires
  a new directory and a fresh run. Recovery would require a separate audit.

Before any policy update, require a completed reference and compare ALL shared
training settings, initial model hash, dataset and pair-pool hashes, package
versions, attention backend and shared-code hashes. Only arm identity, weight
and output/logging identity may differ. Write and upload `reference_verified.json`.
The zero-weight `dt` option exists only for small execution-equivalence checks.

References in W&B project `2820402607-shandong-university/CORL-DDR`:

- RTG C: [e9hi2xt0](https://wandb.ai/2820402607-shandong-university/CORL-DDR/runs/e9hi2xt0).
- Ordinary DT: [8xhoq8o5](https://wandb.ai/2820402607-shandong-university/CORL-DDR/runs/8xhoq8o5).
- New B group: `MatchedPositive-HCMR-5090-20261005`; run prefix
  `MatchedPositive-b-seed0`. Upload metrics, each 100-episode evaluation table,
  pair artifact, provenance and matched-reference audit using the existing
  ignored server credential file. Data, artifacts, checkpoints and secrets are
  excluded from Git.

## Server execution

Project base `/home/sckd02/workspace/DT-EXP`; use the existing isolated
`.runtime/state-only-c-5090-env` with Python 3.10 / Torch 2.7.1+cu128.

- Frozen source: `.runtime/matched-positive-source-20261005`.
- Campaign: `results/matched-positive-5090-seed0-20261005`.
- Tmux: `matched-positive-queue-20261005`.
- Log: `logs/matched-positive-queue-20261005.log`.
- Entry: `algorithms/offline/matched_positive_dt.py`.
- Config: `configs/offline/dt/halfcheetah/matched_positive_5090.yaml`.

```bash
STATE_ONLY_RUNTIME_BASE=/home/sckd02/workspace/DT-EXP \
python3 scripts/matched_positive/queue.py \
  --dependency /home/sckd02/workspace/DT-EXP/results/mc-value-preference-5090-seed0-20261005 \
  --dependency-commit cb0191f1a4b89c6ce539a75d7eda24bc64a61d7f \
  --dependency-queue-pid 602774 \
  --output-root /home/sckd02/workspace/DT-EXP/results/matched-positive-5090-seed0-20261005
```

Wait for the entire MC-value campaign to complete 100k, final evaluation and
comparison, and for its process to exit. Also verify every prior campaign.
Failure or an incomplete exited predecessor stops B. Exclusive locking and
new output directories prevent duplicate launches; prior queues are not edited.
Once ready, first run 3 full-batch CUDA updates and one episode (offline W&B),
using the old dense CUDA smoke as the exact reference. Then run formal B using
the completed RTG C. Save final and late-window differences B-minus-DT and
C-minus-B. SSH disconnects do not stop tmux; reboot recovery needs inspection.

## Validation and interpretation

Tests check all-positive gradients even when C would deactivate them, no effect
of negatives/margin on B, equality of B/C gradients when every C gate is open,
strict reference consistency and predecessor-chain completion. Run the existing
RTG/context/C/queue tests and Ruff 0.0.278 on a clean tracked export before
publishing; inspect GitHub Actions separately afterward.

Small CPU runs must preserve initialization and first DT loss, and zero-weight
B execution must reproduce old DT parameters after the same updates. Verify the
first positive/negative distances and ordered pair hash against RTG C, plus
W&B evaluation and artifact upload. These are execution checks, not policy results.

This is a single-training-seed pilot. Similar B/C performance provides no strong
evidence for keeping the gate; an apparent C advantage requires additional seeds
before claiming a stable effect. Do not infer action-label correctness or causal
advantages from one run or from episode-level error bars.
