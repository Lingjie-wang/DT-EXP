# Grouped DT experiments

The 30 DT datasets are split into four independently runnable benchmark groups:

| Entry point | Datasets | Runs with default seeds |
|---|---:|---:|
| `gym_mujoco.py` | 9 | 54 |
| `maze2d.py` | 3 | 18 |
| `antmaze.py` | 6 | 36 |
| `adroit.py` | 12 | 72 |

Each dataset contributes six runs: original DT seeds 0/1/2 followed by
delayed-reward DT seeds 0/1/2.

Run one group on one GPU:

```bash
python scripts/dt_experiments/gym_mujoco.py --gpus 0 --keep-going
```

Run different groups on different servers by choosing a different entry point
on each server. Every group entry accepts the same options as
`scripts/run_dt_experiments.py`, including `--dry-run`, `--seeds`,
`--variants`, `--wandb-mode`, and `--rerun-completed`.

## Target-aligned hard-fork experiment

The target-aligned hard-fork comparison resumes every method from the same
delayed-reward, seed-zero DT checkpoint at step 50,000 and stops at step 75,000.
All runs use the CORL `halfcheetah-medium-replay-v2` config and evaluate 100
episodes at RTG 6,000 and 12,000 every 5,000 updates.

- `run_dt50k_control_to75k_delayed_hcmr_seed0.sbatch` is the plain-DT control.
- `run_target_aligned_hard_fork_v2_l005_delayed_hcmr_seed0.sbatch` uses
  preference weight 0.05.
- `run_target_aligned_hard_fork_v2_l010_delayed_hcmr_seed0.sbatch` uses
  preference weight 0.10.

The target-aligned runs sample recorded/6,000/12,000 RTG conditions in a
50%/25%/25% mixture. They normalize the hinge loss over currently active pairs,
falling back to the full batch when fewer than 16 pairs are active. Pair
priorities are updated online with an EMA and mixed equally with frozen-DT
hardness weights. The reference anchor is applied at both evaluation RTGs.

`run_single_target_hard_fork_v3_high_delayed_hcmr_seed0.sbatch` is the
single-target ablation. It leaves the base DT loss, pair data, preference
weight, and 50k-to-75k budget unchanged, but applies every preference pair and
the frozen-reference anchor only at RTG 12,000. Evaluation still reports both
RTG 6,000 and 12,000, so the run directly tests whether the v2 target mixture
caused a conditioning trade-off.

`run_single_target_hard_fork_v3_high_delayed_hcmr_seed1.sbatch` reuses the
completed seed-one 50k checkpoint and seed-one pair diagnostic with the frozen
v3-high configuration. Its existing DT-control and v2 counterparts provide the
paired seed-one baselines.

`run_single_target_hard_fork_v3_high_delayed_hcmr_seed2.sbatch` uses the same
frozen v3-high configuration with the completed seed-two checkpoint and pair
diagnostic, completing the three-seed comparison.

Seed 1 uses the same staged protocol. Run
`prepare_dt50k_pair_diagnostic_seed1.sbatch`, followed by
`run_dt_pair_diagnostic_seed1.sbatch`. The control can start as soon as the
checkpoint preparation succeeds; the target-aligned run starts after the pair
diagnostic succeeds. Their entry points are
`run_dt50k_control_to75k_delayed_hcmr_seed1.sbatch` and
`run_target_aligned_hard_fork_v2_l005_delayed_hcmr_seed1.sbatch`.

Seed 2 uses the same checkpoint preparation, diagnostic, control, and
target-aligned entry points with `_seed2` suffixes.
