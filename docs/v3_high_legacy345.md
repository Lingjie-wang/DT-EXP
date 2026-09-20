# Archived v3-high protocol, seeds 3/4/5, extended to 100k

Requested 2026-09-20: rerun the **original seed-0/1/2 protocol**, not the modern
paired-resume experiment. The user subsequently explicitly extended the stop
from 75k to **100k** for both DT control and v3-high.

## Historical evidence and source selection

Frozen revision: `25626cefd9465dadbd0a6ef95756ab943ca7fab6` (original seed-2
v3 launcher). Core `dt.py`, `sap_dt_one_sided.py`, `hard_fork_dt.py`, and
`diagnose_sap_pairs.py` are byte-identical between original v3 revisions
`76896b6`, `29a0b6b`, and `25626ce`. The warmup `sap_dt_one_sided.py` and `dt.py`
are also unchanged from diagnostic revision `c062d81`.

DT control uses its separately frozen, earlier revision
`e8126627628ebf2e928796e8a5db3f81fbcd3aa3`, under `control_source/`.
Actual old control configs predate the `preference_target_mode` field added
for v3. Although that field is inactive in control mode, use the older complete
entrypoint too rather than retaining even this unused configuration difference.

The three historical 50k checkpoint configs were read directly from server:
all came from `DT50kPairDiagnosticPrep`, with `update_steps=50001`,
`preference_start_step=50000`, `eval_every=50000`, `eval_episodes=10`.
The checkpoint is saved BEFORE the first auxiliary update, at exactly 50,000
ordinary updates. The warmup process subsequently performs its historical extra
auxiliary update, but that updated model is NOT the fork checkpoint.

Old W&B metadata is not always sufficient provenance: the server previously
received file-by-file code sync while its Git HEAD lagged. Source selection is
cross-checked against tracked historical launcher/core content and actual configs,
not only old metadata's Git field. Byte-for-byte old *runtime* files were not
independently archived by every old run; bitwise historical reproduction is not
promised. Current Torch 1.11.0+cu113, NumPy 1.23.1, Gym 0.23.0, mujoco-py 2.1.2.14,
D4RL 1.1, W&B 0.17.4, and Pyrallis 0.3.1 match recorded core requirements.
Some auxiliary packaging libraries differ; no existing environment is modified.

## What is preserved

The orchestrator reads command arguments mechanically from the frozen old
seed-1 sbatch files. Only seed, output paths, reporting names/group, and the
user-requested final stop are changed. All seeds use the same shared recipe.

- Recreate a fresh old-entrypoint 50k checkpoint per seed; do NOT reuse the
  modern campaign's 50k checkpoint or `top_return_weighted_dt.py`.
- Original strict pair mining per seed, using the archived diagnostic code.
- Original v3 preference/reference weights 0.05/0.1, margin 0.05, high RTG
  12000, fixed 50k reference, dynamic mix 0.5, EMA 0.9, active floor 16.
- Original control `preference_mode=control`: skips auxiliary forwards. No
  zero-weight auxiliary-forward matching and no `paired_resume` extension.
- Do NOT add training/loader RNG restoration or evaluation RNG isolation.
- Preserve CORL config, delayed training/evaluation, 12000/6000 targets,
  eval seed 42, 100 episodes every 5k during continuation.
- Preserve legacy indexing: `update_steps=100001`, logged steps 50000..100000
  inclusive. Thus 50,001 continuation updates, and the labeled 75k/100k points
  follow 75,001/100,001 total updates, exactly extending the old convention.
  Keep both 75k and 100k results; do not silently compare differing best windows.
- Final model save follows the original implementation. No extra in-loop
  checkpoint saves or modified losses are introduced.

Match historical GPU assignments through old->new seed mapping 0->3, 1->4,
2->5. GPU names are verified at runtime:

| Seed | Old warmup type | Old DT continuation type | Old v3 continuation type |
| --- | --- | --- | --- |
| 3 | RTX 3090 | RTX 4090 | RTX 3090 |
| 4 | RTX 3090 | RTX 3090 | RTX 3090 |
| 5 | RTX 3090 | RTX 3090 | RTX 4090 |

These intentionally retain historical hardware differences. This experiment
tests legacy-protocol replication, NOT the stricter same-GPU/RNG paired design.
Nondeterministic Torch kernels remain enabled as historically.

## Isolation, logging and checks

Campaign: `results/v3-high-legacy345-20260920/`. The full historical repository
is a detached Git worktree under its `source/` (v3/warmup/miner) and
`control_source/` (DT); current algorithms, old launchers
and all earlier results are untouched. Verify source hashes before each worker.
New checkpoint, pair and journal paths are under `production/seedN/`.

A logging-only observer records original W&B calls locally, without hooking
tensors, RNG or training. It suppresses explicit `.pt/.npz/.npy` W&B uploads
while preserving server-local saves. This is a reporting/privacy exception to
the original launcher, not a learning change. Diagnostics have W&B disabled;
production training metrics/configuration upload to the existing CORL-DDR project.

CPU tests check original launcher fidelity, seed limits, hardware mapping,
legacy control behavior, smoke isolation and W&B init rebinding. A short GPU
smoke runs archived warmup, full pair miner, control and v3, before production.
Smoke model/pairs are isolated and never initialize production. Each full
warmup is checked for legacy checkpoint format, seed, step and valid pairs.
Branches verify shared input hashes, complete update/evaluation counts,
ordinary control loss and v3 high-only target. No recurring monitor is created.

W&B group: `Legacy012-V3High-Seeds345-To100k-HCMR-delayed`.
Names (trainer appends environment and unique suffix):

- `Legacy012-prepare-seed{3,4,5}-HCMR-delayed`
- `Legacy012-dt-seed{3,4,5}-HCMR-delayed`
- `Legacy012-v3-seed{3,4,5}-HCMR-delayed`

## Launch sequence

Initialize the detached source with `legacy_v3_high.py init`. Submit a short
`run_v3_high_legacy345.sbatch smoke` job on an available supported GPU. After
it passes, submit three `prepare --seed N` jobs on RTX 3090 nodes (gn4/gn5),
then two `branch --seed N --stage dt|v3` jobs with `afterok` dependencies on
their own warmup. Use the GPU mapping above, original 12 CPUs and GPUNorm.
The initial 48 GB RAM reservation is now 32 GB with explicit user approval
(see the resource-only amendment below).
Existing jobs are never cancelled. GPU and CPU/RAM allocation can cause queues;
the account allows only two concurrent jobs.

## Validation and submission record

Implementation commits: `acc2df3` and `354e772` (exact earlier DT entrypoint).
Eight CPU protocol tests pass locally and on the server. Full-repository local
Ruff check is clean. [GitHub codestyle for the initial implementation passed](https://github.com/Lingjie-wang/DT-EXP/actions/runs/35494115809).

GPU smoke 9562 completed on gn7 / RTX 4090 in 1m10s. A follow-up full smoke,
**9594**, includes the exact earlier control revision; it completed on gn7 in
**1m16s**, exit 0. It used isolated smoke seed 4, ran the original warmup and
full 100k-candidate pair miner, then both legacy continuation branches. Each
branch passed its update-count, loss and evaluation audits. The persisted gate
includes BOTH historical source revisions. Smoke outputs never initialize long
training. Initial smoke artifacts are retained, not overwritten.

| Seed | Old-entrypoint warmup + mining | DT to 100k | v3-high to 100k |
| --- | --- | --- | --- |
| 3 | 9563 | 9595 | 9596 |
| 4 | 9564 | 9597 | 9598 |
| 5 | 9565 | 9599 | 9600 |

Each continuation depends on successful completion of its own warmup AND full
smoke 9594. The original CPU/RAM requests are preserved. At submission the
warmups are pending RTX 3090 resources; continuation jobs wait on dependencies.
The known 3090 nodes are gn4/gn5. A free GPU alone is insufficient when the
required 12 CPUs and 48 GB RAM cannot be allocated together. No old job was
cancelled, no existing experiment data was removed, and no monitor was created.

## User-approved resource-only amendment

The user explicitly approved reducing the Slurm RAM reservation from 48 GB to
32 GB to use an available RTX 3090. At the check, gn4 had one unallocated GPU
but only about 34 GB of schedulable RAM, so the original request could not fit.
The scheduler's provisional start estimate was September 23. Recorded MaxRSS
for the full recent warmups and archived smoke was approximately 5 GB; this
is an observed process-memory statistic, not a guarantee of future peak usage.

Applied `MinMemoryNode=32768` to the nine existing production jobs: 9563, 9564,
9565, 9595, 9596, 9597, 9598, 9599, 9600. Before/after scheduler records verify
32 GB and unchanged 12 CPUs, GPU-node filters, time limits and dependencies.
Job IDs are retained; no job was cancelled or resubmitted. The launcher default
is updated to 32 GB as well.
The warmup scheduler time limits were restored to the historical 12 hours;
continuation limits remain 18 hours.

No learning setting changes: same historical sources, GPU-type mapping,
12 CPUs, four DataLoader workers, batch 4096, context 20, optimizer, loss
coefficients, pair miner/sampler, RNG behavior, 50k fork, 100k stop and evaluation.
In particular, this is a host RAM reservation change, NOT a smaller batch,
lower GPU memory limit, mixed precision, gradient accumulation or model change.

## User-approved parallel single-GPU dispatch

The user subsequently approved allowing available RTX 3090 **or** RTX 4090
GPUs for pending stages, to seek two concurrent independent experiments.
This supersedes the historical per-seed GPU mapping as a scheduling constraint;
the old mapping remains a provenance reference, not an enforced requirement.
The approved node pool is gn4/gn5 (3090) and gn7/gn8/gn12 (4090).

Each job still requests exactly ONE GPU, 12 CPUs and 32 GB RAM. This is NOT
DDP/DataParallel or multi-GPU training of a single model. The account's two-job
limit remains in force; no scheduler priority or account limit is bypassed.
Only still-pending jobs have their excluded-node lists broadened. Running and
completed jobs are left untouched, and existing dependencies are preserved.

The orchestration-only GPU guard now accepts either supported model. It writes
the actual GPU, historical GPU, match flag, policy, Slurm job ID and node to
`records/<stage>/hardware.json` and `run.json`, and to W&B summary fields under
`hardware/`. This does not alter the archived training configuration or source.
Earlier running/completed stages retain their original W&B hardware metadata.

All historical learner files, batch size, optimizer, preference/reference losses,
sampling, RNG handling, 50k fork, 100k stop and evaluations remain unchanged.
Different GPU models can still lead to numerical differences; hardware identity
with the old cohort is no longer claimed. No running experiment is restarted.
