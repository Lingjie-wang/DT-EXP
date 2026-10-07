# Paired CQL policy repeats on RTX 5090

User-requested on 2026-10-07: add two seeds to each of the four active arms,
choose a shorter common budget based on recent curves, and schedule within
available resources. Historical campaigns, code, rewards, and results remain
unchanged. No uniform or dense control is restarted.

## Protocol fixed before new training

| Dataset | Reward settings | New policy/evaluation seeds | Updates per run |
| --- | --- | --- | --- |
| HalfCheetah-medium-v2 | terminal-delayed, predictive Shapley 20x128 v1 | 11, 12 | 300,000 |
| HalfCheetah-medium-replay-v2 | terminal-delayed, predictive Shapley 20x128 v1 | 11, 12 | 300,000 |

Eight new training runs. CQL Slurm seed 0 and 5090 pilot seed 1 are reserved;
11/12 also avoid the recent Slurm LPT 0--4 seeds. Compare both reward methods on
the same new seeds. These are independent **policy/evaluation** runs conditional
on the existing reward fits: no new Shapley predictor/attribution seeds. Copy
the audited frozen reward NPZs, source and passing predictor gates byte for byte.
Change only YAML seed, max_timesteps, name and group, plus protocol metadata.
All other CQL settings, batch 256, gamma .99, evaluation every 5k over 10 episodes,
and checkpoints every 100k remain unchanged. Existing 1M pilots retain their budget.

Budget evidence was read at 2026-10-07 18:32 Asia/Shanghai. Three pilots had reached
267.3k and medium Shapley 187.3k. Shapley scores had largely stabilized by 30--50k
(roughly 44--45 later); delayed baselines still changed through 100--200k and
subsequently deteriorated/oscillated. A common 300k horizon reduces new training
by 70%, leaves room beyond the early plateaus and does not pick a different
stopping point for each method. It does not prove convergence or exclude later
recovery. Treat seed 1 as the budget-selection pilot; seeds 11/12 are confirmation.

Primary summary: mean of the last 10 evaluations through 300k (255k--300k), with
per-seed values and cross-seed uncertainty. Report the 300k score and best score
within the **same 300k window** separately; never mix an old 1M best with a 300k
best. Any combined historical comparison must truncate old curves to this budget.
No original dense reward information is newly introduced. The prior audit's
float precision, discount-objective and extra-preprocessing caveats still apply.

## Resource plan and recovery

At inspection the four existing CQL processes used about 10GB of 32GB VRAM,
90% GPU utilization and approximately 34 updates/sec each. Spare VRAM alone
does not justify adding eight more competing training processes. The durable
queue counts **all** compute processes on GPU 0 and allows at most four total,
including jobs whose CUDA initialization has not yet appeared in nvidia-smi.
Each admission reserves two slots, 3GiB per new job plus 2GiB GPU headroom, and
at least 8GiB available host memory. Existing jobs are never signalled or altered.

Admit delayed/Shapley pairs together: seed 11 medium, seed 11 medium-replay,
seed 12 medium, seed 12 medium-replay. Up to two pairs train concurrently when
the GPU is free. Every pair first passes separate 100-update full-batch CUDA
preflights with an evaluation episode; formal runs start from scratch. A failed
preflight blocks its pair while allowing other pairs to proceed. No failed run
is silently resumed or overwritten. Resource query failures leave the queue
waiting. A second manager is rejected by a lock and existing status file.

Queue state is in `CAMPAIGN/queue/status.json`, including per-run state, process
IDs and resource readings. Each run retains independent logs, configs, frozen
source, dataset hashes, preflight, evaluations and checkpoints. W&B observers
start with training, retry network failures independently and upload checkpoints.
Queued names/IDs are reserved in `plan.json`; W&B pages may not exist until start.
Inspect saved status and live PIDs before manually recovering a manager failure;
never start a replacement manager over surviving training children.

## Reproduce and identify runs

```bash
python -m scripts.cql_repeats_5090.prepare --project /home/sckd02/workspace/DT-EXP
bash scripts/cql_repeats_5090/launch.sh
```

Use an immutable, validated Git commit archive on the server, the existing
`.runtime/state-only-c-5090-env`, and the ignored existing W&B credentials.
Run the launcher inside its own tmux session. The campaign root is
`results/cql-repeats-seeds11-12-300k-5090-20261007`.

W&B project: `2820402607-shandong-university/CORL-DDR`. For each N in 11,12:

- `CQL-CORL-HCM-delayed-seedN-300k-5090`
- `CQL-CORL-HCM-shapley20x128-v1-seedN-300k-5090`
- `CQL-CORL-HCMR-delayed-seedN-300k-5090`
- `CQL-CORL-HCMR-shapley20x128-v1-seedN-300k-5090`

Publish changes from a clean worktree on the latest GitHub main, validate a clean
tracked export, and use normal fast-forward pushes. Reconcile concurrent commits
without rewriting shared history. Server deployment does not update either dirty
working tree. Keep datasets, checkpoints, logs, environments and secrets out of Git.
