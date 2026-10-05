# Preference-only continuation of the completed DT

Test whether removing ordinary DT action regression improves the existing late
preference fine-tuning. This is an independent C-only arm, not a paper reproduction.
Historical implementations, checkpoints and experiments remain unchanged.

## Fixed protocol

- Start from the same 100k original-reward HalfCheetah-medium-replay-v2 DT parent
  (W&B `8xhoq8o5`); reuse all 7,306 original RTG-mined ordered pairs.
- Objective: `0.05 * mean(relu(d_pos - stopgrad(d_neg) + 0.05))`.
  Ordinary DT MSE is evaluated under `torch.no_grad()` only for diagnostics and
  the matched forward/dropout schedule. It contributes no new gradient.
- This C objective pulls towards positive actions when its gate is open. It
  does not directly repel negative actions. Dropping DT does not change that.
- Restore the same model, optimizer moments, scheduler counters and RNG as the
  completed late-preference C; use its identical LR reduction 8e-4 to 1e-4.
  Historical AdamW moments (learned from DT) and weight decay remain. Resetting
  them would change a second factor, so this is NOT a fresh-optimizer ablation.
- Keep ordinary diagnostic batch 4096, auxiliary batch 256, seed 0, context 20,
  architecture, normalization, dropout, clipping and math SDPA unchanged.
  Use the same new worker streams as the previous late-preference children.
- Train exactly 5,000 additional updates. Evaluate steps 0/1000/3000/5000 with
  100 episodes, seed 42, initial target return 12000. The primary endpoint is
  final step 5000; intermediate scores are secondary, without checkpoint picking.
- Reuse finished controls from `results/late-preference-5090-seed0-20261005`:
  ordinary DT `g9fe4dp1`, mixed DT+C `zie5w3hn`, and contextual DT+B `c1amwyrf`.
  The frozen common parent scores 38.5277; these controls end at 36.9562,
  36.1141, and 35.8897, respectively.

Exact restored states, shared config/source/data/pairs, first ordinary batch,
pair indices, dropout RNG and all initial losses must match the completed C.
All baseline episode returns must exactly match C and reproduce the parent
within the existing 1e-6 tolerance. Reject changes before the first policy update.

## Execution

Independent runner `algorithms/offline/preference_only_dt.py`, helper
`preference_only.py`, config `configs/offline/dt/halfcheetah/preference_only_5090.yaml`,
and scripts `scripts/preference_only/`. Use the existing isolated 5090 runtime.
The queue first runs three full-batch CUDA smoke updates with one-episode
evaluations at 0 and 3, then one formal C-only child. It refuses a busy GPU,
dirty source, incomplete reference, or an existing campaign/trial directory.

```bash
export STATE_ONLY_RUNTIME_BASE=/home/sckd02/workspace/DT-EXP
python3 scripts/preference_only/queue.py \
  --parent-checkpoint "$STATE_ONLY_RUNTIME_BASE/results/dense-preference-5090-seed0-20261005/dt-seed0/checkpoints/step100000.pt" \
  --reference-campaign "$STATE_ONLY_RUNTIME_BASE/results/late-preference-5090-seed0-20261005" \
  --output-root "$STATE_ONLY_RUNTIME_BASE/results/preference-only-5090-seed0-20261005"
```

W&B project `2820402607-shandong-university/CORL-DDR`, group
`PreferenceOnly-HCMR-5090-20261005`, name prefix `PreferenceOnly-c-only-seed0`.
Upload all metrics, episode tables and provenance audits. Save comparisons to
DT, mixed C and the frozen parent. Credentials, data and checkpoints stay out of Git.

Run preference-only and existing late/matched/dense/state-only unit tests and
Ruff 0.0.278 on a clean tracked export before publication. Verify CI, GPU smoke,
first-update integrity and W&B uploads before interpreting final results.

This one-seed ablation tests the effect of including DT loss, not whether
overfitting caused the earlier decline. Preference-only fine-tuning can also
overfit its limited pair pool or alter behavior outside covered states. Even an
improvement needs independent evaluation/training seeds before a stable claim.
