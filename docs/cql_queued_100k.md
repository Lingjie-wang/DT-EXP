# User-requested 100k replacement for both pending CQL queues

On 2026-10-07 the user changed **all queued experiments to 100,000 updates**.
At 19:03 Asia/Shanghai, every job in both queues remained `queued`; none had
started GPU preflight, training or a W&B observer. Scope:

| Dataset(s) | Reward(s) | Policy/eval seeds | Runs | New updates |
| --- | --- | --- | --- | --- |
| HalfCheetah medium and medium-replay | delayed, predictive Shapley | 11, 12 | 8 | 100,000 |
| HalfCheetah medium and medium-replay | original dense | 1, 11, 12 | 6 | 100,000 |

The four already-running seed-1 pilots are outside the queued-only request and
retain their 1M budget. No Slurm job changes. Historical 300k protocols, frozen
source, configurations and status files are retained; their two idle manager
processes are stopped after checking identity, no children, all-queued state and
absence of preflight/training directories. A `budget_change_stop.json` receipt
records this action. Never signal existing training processes for this change.

The replacement entrypoint requires both old manager locks, rechecks all jobs
are untouched, and copies each run into a separate `...-100k-...` campaign. Data,
scientific Python, seeds, evaluation frequency and episode count are unchanged.
Only max_timesteps, run/group names and IDs plus budget metadata change. Old
campaigns receive `superseded_by_100k.json` receipts and cannot be relaunched over
their existing queue state. New W&B names use `-100k-` and new stable IDs.

The shared queue validator now checks the budget against the plan **and** frozen
YAML rather than a hard-coded 300k. This is an explicitly requested orchestration
change; it does not change the CQL algorithm, trainer or evaluation implementation.
The dense dependency points to the new 100k repeat plan and its hash. Maintain
the same priority and max four total GPU compute processes, paired admission,
memory thresholds, independent CUDA preflights and independent W&B bridges.

Each completed run has 20 evaluations (every 5k, 10 episodes each) and its normal
final checkpoint at 100k. Report the final score, last-10 mean (55k--100k), and best
within 100k separately. Use this same window for comparisons with historical pilots.
The 100k horizon is the user's budget choice, not a convergence claim.

```bash
# After safely stopping only the two entirely pending 300k managers:
python -m scripts.cql_budget_100k.prepare \
  --project /home/sckd02/workspace/DT-EXP --code-revision PUBLISHED_COMMIT
bash scripts/cql_budget_100k/launch.sh repeats
bash scripts/cql_budget_100k/launch.sh dense
```

Use separate tmux sessions and a validated immutable published source archive.
Queue status remains `queue/status.json` for repeats, and `dependency/status.json`
then `queue/status.json` for dense. Back up new plans, frozen sources and receipts
to the other server. Preserve unrelated dirty trees and publish scoped changes
using normal fast-forward GitHub updates with clean-export CI and Actions checks.
