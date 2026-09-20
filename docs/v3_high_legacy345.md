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
is a detached Git worktree under its `source/`; current algorithms, old launchers
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
their own warmup. Use the GPU mapping above, original 12 CPUs/48 GB and GPUNorm.
Existing jobs are never cancelled. GPU and CPU/RAM allocation can cause queues;
the account allows only two concurrent jobs.

Submission IDs and observed validation results are recorded after launch.
