# New-checkpoint paired DT vs v3-high, seed 0

Question: does the original v3-high objective improve a different, stronger DT
starting point, rather than only the historical 50k checkpoint?

Branch the completed from-scratch DT control's `step050000.pt` into two runs
of the SAME `hard_fork_dt.py` entry point, with the opt-in `paired_resume=true`:

- DT control: `preference_weight=0`, `reference_weight=0`.
- v3-high: `preference_weight=0.05`, `reference_weight=0.1`.

The control still performs the auxiliary forward calls and diagnostics, but
both coefficients are zero: its gradient is exactly ordinary CORL DT's masked
action MSE gradient. Auxiliary contexts do not change its ordinary data sampler.
Keeping forwards in both arms matches dropout RNG consumption. Do not interpret
positive raw preference/reference losses as enabled objectives in the control.
Dynamic priorities can diverge as policies diverge; the ordinary DT worker
sample stream remains independent and matched.

## What is held fixed

Dataset `halfcheetah-medium-replay-v2`, delayed terminal-sum reward; seed 0;
batch 4096, context 20, architecture, dropout 0.1, LR 0.0008, AdamW and clipping
as in the CORL config. Restore model, optimizer, scheduler and Python/NumPy/
Torch CPU/CUDA RNG from the new 50k checkpoint, after creating loaders/models.
Restore its dedicated DataLoader generator state to both fresh worker pools.
Protect all training RNG state around evaluation in both arms.

The old loader's worker states/prefetch queues were NOT saved. Therefore this
is a matched fresh continuation, NOT bitwise resumption of the uninterrupted
from-scratch curve. Even with identical seed numbers, that distinction matters.
Both arms use the same GPU model when scheduled; nondeterministic kernels remain
enabled as in the original experiment. Audit initial model hash, restored RNG,
optimizer LR/scheduler epoch, first three DT batch/RNG hashes and first DT loss.

Keep the original v3 pair NPZ unchanged: 2,124 strict valid pairs, margin 0.05,
batch 256, active normalization minimum 16, dynamic mix 0.5, EMA 0.9, hardness
temperature 0.05. Its stored confidence/frozen-hardness scores still come from
the original diagnostic checkpoint; there is NO new pair mining/reweighting.
The reference MODEL is newly frozen from the NEW 50k checkpoint, not the old one.
Preference and reference RTG stay at 12000; ordinary DT loss uses recorded RTG.
The v3 objective and dynamic-priority formula are unchanged.

## Step and evaluation protocol

Opt-in completed-update indexing: perform exactly 25,000 further updates,
logged at 50001 through 75000. Evaluate/save at 55/60/65/70/75k. This matches
the from-scratch trainer's step meaning; historical v3 used zero-based indices
and had one extra update at each identically named evaluation point.
No pre-update 50k evaluation is repeated: input model hashes establish the shared
start, and the original checkpoint's 50k evaluation is already available.

Evaluate RTG 12000 and 6000, each 100 episodes, eval seed 42; delayed-policy
evaluation keeps RTG constant and reports original environment total returns.
Primary: 75k last and mean across the five 55k-to-75k evaluation points; secondary:
best among those same five points. Also show the existing uninterrupted DT curve.
This is a seed-0 screening experiment, not a multi-seed significance claim.

Save model, optimizer/scheduler, RNG, loader-generator state, frozen reference
and dynamic priorities in separate output directories. Worker queues are still
not serialized; exact later resumption is not implemented. Legacy defaults and
all older experiment launch scripts remain unchanged.

Source run: [from-scratch DT](https://wandb.ai/2820402607-shandong-university/CORL-DDR/runs/40316f55-aceb-446d-81ce-e5528033e3e4).
Input model tensor SHA256: `47f657d4f99ba610ba950d0132e968639c462f224677b04bde6c163ab912d5a8`.
Input checkpoint file SHA256: `ca2d5a63f89d8fa5953e34b73d543c30976f98e73b618ae0edb456fc55f836d3`.
Pair file SHA256: `8ec016c96d81f0cc5b73e0e32521616ee1364f143abc0efccf07dd8f72c8b52a`.
W&B project `corl-ddr`; group `New50k-Paired-DT-v3-50kTo75k-HCMR-delayed-seed0`.

## Verification and launch

```bash
PYTHONPATH=algorithms/offline python -m unittest discover -s tests -p test_paired_hard_fork_resume.py
sbatch --array=0-1 --time=00:20:00 scripts/dt_experiments/run_new50k_paired_dt_v3_to75k_hcmr_seed0.sbatch smoke
# Compare the two offline smoke summaries before submitting production.
sbatch --array=0-1 scripts/dt_experiments/run_new50k_paired_dt_v3_to75k_hcmr_seed0.sbatch
```

Smoke runs three full-batch updates with evaluation after EACH update, one
episode per target, to audit pairing even across intervening evaluations.
No new dependencies or recurring monitors are required.

## Launch record and pairing audit (2026-09-18)

Production code: `3266fde65cf869d2a8d189dcdeb4afc17c305989` (GitHub codestyle
passed). The feature commit is `487b0cd`; follow-up commits place tests in the
configured MuJoCo environment and make the source-integrity test Python-version
independent. Initial smoke array `9182` stopped at that test, before training.

Successful smoke: `9184_0` (DT) and `9184_1` (v3), both completed on RTX 4090
with exit 0. Each passed four unit tests and three full-batch updates, including
intervening evaluations. Cross-run audit of their W&B binary records verified:

- Both initial model hashes match the NEW input checkpoint tensor hash above.
- Restored LR is 0.0008 and scheduler epoch is 50000 in both arms.
- All first-three DT batch hashes and Torch CPU/CUDA RNG hashes match exactly.
- First DT loss is exactly `0.039531702088585516` in both arms.
- DT's total loss equals ordinary DT loss on all three steps; both weighted
  auxiliary terms are zero. V3 has nonzero preference and reference terms.
- Both `step050003.pt` snapshots contain exactly 50,003 completed updates and
  a frozen reference whose tensor hash matches the NEW input checkpoint.

Production array: `9186_0` is DT control; `9186_1` is v3-high. Both use the
same script with only the two objective coefficients and run/output names changed.
Submission restricts eligible nodes to the known RTX 4090 nodes (`gn7`, `gn8`,
`gn12`) by excluding other current GPUNorm nodes. The account's two-running-job
limit also includes the existing reference-only job `9140`; no old job is stopped.
Group: [paired runs](https://wandb.ai/2820402607-shandong-university/CORL-DDR/groups/New50k-Paired-DT-v3-50kTo75k-HCMR-delayed-seed0).
