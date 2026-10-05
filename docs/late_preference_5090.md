# Short preference fine-tuning after a completed DT

Test whether positive imitation or negative-gated positive imitation improves an
already-trained DT. Preserve every historical source, checkpoint and output.
This is a single-training-seed pilot, not an equal-total-budget test of activation
timing against the earlier from-scratch C.

## Predeclared protocol

- Parent: completed original-reward DT `8xhoq8o5`, its FINAL 100,000-update
  checkpoint. Never select the best checkpoint retrospectively.
- Dataset: HalfCheetah-medium-replay-v2, 202 x 1000 transitions, original rewards.
- Three children: DT MSE only; B = DT MSE + 0.05 * positive action MSE;
  C = DT MSE + 0.05 * mean(relu(d_pos - stopgrad(d_neg) + 0.05)).
- Reuse EXACTLY the dense RTG miner's 7,306 pairs and positive 20-step contexts,
  desired RTG 12000 minus observed prefix rewards. Negative actions have no
  direct repulsion gradient. B includes every sampled positive.
- Restore identical model, AdamW moments and step counters, scheduler state,
  global RNG, pair RNG and loader generator from the same parent checkpoint.
  After verifying restoration, lower LR from 8e-4 to a constant 1e-4 in ALL
  children. Retain scheduler step count beyond warmup; do not warm up again.
- New identical loader worker streams across children: the parent did not save
  prefetch queues. This is not an exact continuation of the parent data stream.
- Keep architecture 128/3/1, sequence 20, batch 4096, auxiliary batch 256, dropout,
  clipping, optimizer betas/weight decay, seed 0 and math SDPA unchanged.
  All arms execute both forwards and sample the same way, including DT.
- Each child adds 5,000 updates. Evaluate at additional steps 0/1000/3000/5000,
  100 episodes, seed 42, initial RTG 12000 with ordinary original-reward updates.
  The 0-step returns must reproduce the saved parent evaluation within 1e-6.
- Primary endpoint is FINAL additional step 5000, not the best intermediate
  checkpoint. Compare B-minus-DT, C-minus-DT and C-minus-B. Episode dispersion
  does not measure uncertainty across training seeds. Save every 1k updates.

## Integrity and execution

Independent entry `algorithms/offline/late_preference_dt.py`, helper
`late_preference.py`, config `late_preference_5090.yaml`, launcher and queue under
`scripts/late_preference`. Parent checks reject changed data, pair pool, shared
model/miner source, runtime packages and training configuration outside the
explicit changes above. Save exact model/optimizer/scheduler fingerprints before
and after the common LR change, plus first batch, pair indices and dropout RNG
fingerprints. Compare these across all three children.

Runtime base is `/home/sckd02/workspace/DT-EXP`; reuse the existing isolated
`.runtime/state-only-c-5090-env`. The launcher reads existing ignored W&B
credentials without printing them. W&B project `2820402607-shandong-university/CORL-DDR`,
group `LatePreference-HCMR-5090-20261005`, names `LatePreference-{dt,b,c}-seed0`.
W&B steps are ADDITIONAL updates; `progress/total_updates` includes the parent's
100k. Log training every 100, each evaluation with per-episode tables, and a pair
artifact containing statistics, provenance and parent restoration audit.

The queue refuses a busy GPU, dirty source, duplicate campaign or existing child
output. It first runs all three full-batch CUDA smokes (3 updates, one episode at
0 and 3, offline W&B), then the three formal children sequentially. Failure
halts the queue. Use tmux to survive SSH disconnection; do not silently restart
or auto-resume after failure/reboot. An exclusive file lock prevents duplicates.

```bash
export STATE_ONLY_RUNTIME_BASE=/home/sckd02/workspace/DT-EXP
python3 scripts/late_preference/queue.py \
  --parent-checkpoint "$STATE_ONLY_RUNTIME_BASE/results/dense-preference-5090-seed0-20261005/dt-seed0/checkpoints/step100000.pt" \
  --output-root "$STATE_ONLY_RUNTIME_BASE/results/late-preference-5090-seed0-20261005"
```

Run late-preference, matched-positive, dense-preference and state-only tests, then
Ruff 0.0.278 against a clean tracked export before publication. Verify GitHub
Actions and all three GPU smokes before interpreting any formal results.
