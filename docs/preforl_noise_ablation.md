# PREFORL action-noise ablation: delayed HalfCheetah-medium-replay-v2, seed 0

Question: does the action-corruption mechanism in the existing PREFORL port
improve deterministic policy return on our dataset, before attempting another
DT adaptation? This is not another DT variant and not a pure-BC comparison.

## Controlled difference

Two new from-scratch runs of the same entry point:

| Arm | `shadow_noise` | Other settings |
| --- | ---: | --- |
| Zero noise | 0 | Shared |
| Original noise | .01 | Shared |

Both still draw exactly the same uniform action-space noise and time-step masks
(mask probability .4); zero noise multiplies the draw by zero, not by skipping
random calls. No clipping of synthetic negative labels. Positive/negative
positions are independently sampled as in the original port. Preserve the
source's first-segment auxiliary BC convention, including after preference
label flips. Consequently this tests **action corruption in the original full
implementation**, not exclusively the negative term's gradient in isolation.

Reuse (without editing) `third_party/PREFORL/train_mujoco_delayed.py`, SHA256
`762cd5196866782e5046dfda325a773fe3735bafb43b91541fb1c8816462b934`.
The loader refuses another source version. This is our earlier dependency-light
medium-replay port, not a claim of bitwise reproduction of upstream code.
The old source, old runs, DT variants and old checkpoints are untouched.

## Shared settings

- Terminal-sum delayed rewards; Top 25% episode-return selection, identical
  data hash and threshold in both arms. Full original state observations.
- Gaussian policy: 3x1024 ReLU MLP, learned global log standard deviation per
  action coordinate. Neither RTG input nor reference model is introduced.
- Original log probabilities sum over action dimensions and 100 sampled time
  positions. Alpha .1, contrastive bias .5, BC coefficient .5, Adam LR .0003,
  global gradient clipping at 1000 (unchanged port convention).
- Requested episode batch 64; actual count is min(64, number of selected
  trajectories), as in the original sampler. No replacement of episode IDs.
- Seed 0, train from scratch to **15,000 completed updates**. Each arm has a
  private but identically initialized NumPy/Python sampler state, protected
  from logging, probing, and evaluation. Model initialization is explicitly
  seeded after environment/data construction.
- Both production arms request two CPU cores and use two PyTorch CPU threads.
  The initial four-core 4090 smoke submission queued briefly, then both tasks
  completed on gn7. Production is planned on the same RTX 3090 node sequentially
  to use the currently available GPU rather than mixing GPU architectures.
- Every 500 updates, evaluate 100 episodes with identical per-step reset seeds
  across arms: `seed + 10000 + step*100 + episode_index`. Seeds vary across
  evaluation points, as in the old port; the corresponding arms remain paired.
- Evaluation uses the deterministic policy mean, clipped to action bounds.
  This state-only policy receives no rewards/RTG, so delaying evaluation rewards
  cannot change its actions; summed original rewards report the same return.
- Primary metrics: last and mean of final five evaluation means (13k, 13.5k,
  14k, 14.5k, 15k). Best evaluation mean is secondary. Distinguish these from
  the best single episode and from threshold-based success rate.

This removes neither learned variance nor original scoring conventions. It
does not assume the previous 50k DT diagnostic applies to this policy.

## Audit and preservation

Five unit tests validate sampler isolation/pairing, unchanged loss expression,
learnable variance, initial weights and BC conventions. Two-step smoke uses
the **full production architecture and batch shape**, with two two-episode
evaluations. Check matching initial model hash, selected data/probe hashes,
GPU type, batch structure/positive actions and post-sampling RNG state, including
after evaluation. GPU kernels retain the original non-deterministic setting;
we audit matched inputs, not guaranteed bitwise training reproducibility.

Every production batch contributes to a cumulative hash of states, labels and
positive actions; periodic audits record that hash plus the sampler RNG hash.
End-of-run equality therefore checks the entire 15k sampling stream. The raw
negative actions are intentionally different.

Separate per-arm, per-job output directories hold configs, selected trajectory
IDs, loss JSONL, evaluation history, pairing audit, fixed-probe action means,
and checkpoints. Save initial, 5k, 10k, 15k, latest and best checkpoints, including
optimizer and RNG states. No existing experiment directory is reused.

The fixed 1024-state probe records mean-action error/magnitude and each learned
log_std to distinguish policy-mean changes from mere variance changes. The raw
probe actions and checkpoints are local only; W&B uploads scalar metrics and
configuration, not checkpoint weights or raw state/action data.

## Launch and inspection

```bash
sbatch --array=0-1 --time=00:15:00 \
  scripts/dt_experiments/run_preforl_noise_ablation.sbatch smoke
# After successful smoke and explicit W&B upload approval:
sbatch --array=0-1 scripts/dt_experiments/run_preforl_noise_ablation.sbatch online
python scripts/dt_experiments/audit_preforl_noise_ablation.py \
  results/preforl_noise_ablation/job-JOBID --require-complete
```

Without `online`, W&B stores offline records. The user explicitly approved
uploading these two runs' loss, normalized score, policy variance and configuration
to their existing CORL-DDR project on 2026-09-18. This approval does not upload
the previous gradient-diagnostic artifacts.

W&B group: `PREFORL-NoiseAblation-HCMR-delayed-seed0`.
Run names: `PREFORL-Noise0-Paired-HCMR-delayed-seed0` and
`PREFORL-Noise0.01-Paired-HCMR-delayed-seed0`.
No recurring monitor is created. Seed 0 is an exploratory ablation, not a
multi-seed significance claim; compare final/late means, not cherry-picked best.

## Launch record (2026-09-18)

Smoke array **9322**: both tasks completed with exit 0 on gn7 (RTX 4090),
20 and 19 seconds. All five unit tests passed. The audit verified identical
initial model (`d30f88c446c0f6e4a200f74eaaacbc549dab959d24c73cd1bfbe14177ac39305`),
selected data (`77f28d7f627bf640471bdf3913f77dce629c98645d288e25f102c33974b470f8`),
probe, batch signatures, complete two-step sampler stream and final RNG state.
Both evaluations ran; their scores are smoke checks only, not performance evidence.
The attempted pending-only cancellation guard correctly refused to cancel these
already completed tasks; no experiment was interrupted.

Production array **9324**, submitted with `--array=0-1%1 --nodelist=gn5` and the
`online` option after user approval. Task 0 is noise=0; task 1 is noise=.01.
Both use two CPU threads and a two-hour limit per task. The array concurrency
limit of one intentionally runs them sequentially on the available 3090 node.
Runner SHA256:
`5b39b60682cb18d78c5e19f0575dbcd98398aacff024bb6cad38695c9cae1719`.
The CPU-thread change applies to both production arms; do not compare their
initialization hash to the four-thread smoke as a requirement for pairing.

Output root: `results/preforl_noise_ablation/job-9324/`, with separate
`noise-0/` and `noise-0.01/` subdirectories. Logs:
`logs/preforl-noise-ablation-9324_0.out` and `...-9324_1.out`.
The old source, source checkpoints, and all previous DT variants remain unchanged.

At the final submission check, both production tasks were pending for Resources;
Slurm estimated 2026-09-19 01:06:14 on gn5 (Asia/Shanghai; estimate may change).
An unallocated GPU in a node summary does not guarantee immediate schedulability
of this complete resource request. W&B run URLs are not yet available because
the processes have not started. The `online` launch option will create/upload
the approved metrics automatically when the scheduler starts each task.
