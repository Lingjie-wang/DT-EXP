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
