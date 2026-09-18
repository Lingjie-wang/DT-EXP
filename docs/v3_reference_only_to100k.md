# DT + reference only: matched v3-high loss ablation

Question: does the preference gradient improve on DT + the existing reference
anchor? This is the third arm of the seed-0 50k-to-100k reference ablation.

| Arm | DT coefficient | Preference coefficient | Reference coefficient |
| --- | ---: | ---: | ---: |
| Full v3-high (already complete) | 1 | 0.05 | 0.1 |
| v3-high without reference (already complete) | 1 | 0.05 | 0 |
| DT + reference only (new) | 1 | 0 | 0.1 |

The new arm uses the unchanged `algorithms/offline/hard_fork_dt.py` trainer.
Do not set `preference_mode=control`: that disables the reference branch too.
Instead set `preference_weight=0.0`, retaining `preference_mode=hard_fork`.
Its optimized loss is `L_DT(recorded delayed RTG) + 0.1 * L_reference(12000)`.
The frozen reference is the original 50k checkpoint, in eval mode. Ordinary
DT sampling, dropout, optimizer/scheduler restoration and all other settings
match the completed full v3-high arm.

## What is and is not removed

Keep the existing preference forwards, diagnostics, active-pair calculation
and online priority updates. Zero times the preference loss contributes no
gradient, while retaining the same forward-call order and dropout RNG use.
A positive raw `train/preference_loss` or `train/preference_active=1` describes
the diagnostic branch; `train/weighted_preference_loss` must be exactly zero.
`train/total_loss` must equal `train_loss + 0.1 * train/reference_loss` up to
floating-point precision. The weighted reference loss should be positive.

The reference still uses the same 2,124 good-pair contexts and the same
confidence/hardness/online-priority sampling rule. Thus preference information
remains in context selection even though its direct optimization term is off.
The sampling rule is matched, but actual sampled pairs can diverge as policies
and their dynamic priorities diverge. This experiment isolates the preference
loss coefficient in the existing algorithm; it is NOT a preference-free
baseline anchored on uniformly sampled states.

## Fixed protocol and existing comparison runs

- Dataset: `halfcheetah-medium-replay-v2`, delayed reward; train seed 0.
- Shared checkpoint SHA256:
  `b36430ef5e9e29a090514bf934bca2d5b66625ee6ae18da93ed60aea9e38cbbf`.
- Shared pair file SHA256:
  `8ec016c96d81f0cc5b73e0e32521616ee1364f143abc0efccf07dd8f72c8b52a`.
- Original CORL DT config, batch 4096, learning rate 0.0008, pair batch 256;
  reference MSE at RTG 12000, coefficient 0.1; dynamic priority mix 0.5, EMA 0.9.
- Evaluate both RTG 12000 and 6000, 100 episodes each, seed 42, every 5k.
  Delayed evaluation keeps the policy's RTG constant during the rollout;
  reported D4RL scores still use the environment's total return.
- Preserve legacy indexing: logged steps 50000 through 100000 inclusive,
  50,001 continuation updates; the 50k evaluation follows the first update.
- Primary: 100k last and mean of 80/85/90/95/100k; secondary: post-50k best.
- Save separate 75k/100k model/optimizer/scheduler snapshots without overwriting
  earlier versions. The existing trainer does not restore all RNG/loader/priority
  state; these files do not promise bitwise-identical continuation.
- Prefer the same RTX 3090 node (`gn4`) as the completed full v3-high run when
  available; record actual hardware. One seed does not establish significance.

W&B project: `2820402607-shandong-university/CORL-DDR`.
Group: `V3High-ReferenceAblation-50kTo100k-HCMR-delayed-seed0`.
New name prefix: `DT-RefOnly-50kTo100k-HCMR-delayed-seed0`.

- Full v3-high: [Ref010](https://wandb.ai/2820402607-shandong-university/CORL-DDR/runs/ce8301c1-f293-4c88-a6a5-accc34b42639).
- No reference: [NoRef](https://wandb.ai/2820402607-shandong-university/CORL-DDR/runs/631f5ffd-f7fd-4e9f-bcbe-7aa77e1f51d8).

## Launch and verification

```bash
python -m unittest discover -s tests -p test_v3_reference_only_protocol.py
sbatch --time=00:15:00 scripts/dt_experiments/run_v3_reference_only_to100k_hcmr_seed0.sbatch smoke
# Inspect smoke config, finite losses, zero weighted preference and snapshots.
sbatch --nodelist=gn4 scripts/dt_experiments/run_v3_reference_only_to100k_hcmr_seed0.sbatch
```

Smoke uses the full training batch, two updates, one evaluation episode per
target and offline W&B. Full training logs online; no recurring monitor is set.
