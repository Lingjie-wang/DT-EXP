# CQL startup recovery (2026-10-07)

Jobs 12978, 12979 and 13041 failed before their first CQL update. D4RL 1.1's
top-level `__init__.py` imports locomotion, hand_manipulation_suite, pointmaze,
gym_minigrid and gym_mujoco inside one try/except. The optional Adroit hand suite
requires mjrl; its missing import skips the later gym_mujoco registration.
Consequently `gym.make('halfcheetah-medium[-replay]-v2')` raises NameNotFound.

The independent CQL experiment entrypoints now explicitly import the official
`d4rl.gym_mujoco` registration module. HalfCheetah's implementation does not need
mjrl. No stubs, monkey patches, simulator substitutions, upstream algorithm edits,
dependency installations or shared-environment changes are introduced. This is
a minimal environment-registration compatibility change. Network construction,
losses, optimizers, rewards, episode boundaries, training seeds and schedules remain
identical; GPU validation tests actual training and evaluation, not just registration.

`scripts/cql_runtime/prepare_retry.py` creates fresh campaigns, checks historical
source/data hashes, copies the exact prepared datasets, replaces only the fixed
entrypoint, and freezes new source hashes. It requires explicitly selected arms
and rejects arms marked user-cancelled. Shapley rewards and the successful
cross-fitting gate are reused, never regenerated. Old failure records and all
model/attribution artifacts stay in their original directories.

Example preparation (from the fixed worktree):

```bash
python3 scripts/cql_runtime/prepare_retry.py --previous OLD_DELAYED --root NEW_DELAYED \
  --runner cql_delayed --arms medium medium_replay
python3 scripts/cql_runtime/prepare_retry.py --previous OLD_SHAPLEY --root NEW_SHAPLEY \
  --runner cql_shapley --arms shapley
sbatch NEW_DELAYED/source/scripts/cql_runtime/validate.sbatch NEW_DELAYED NEW_SHAPLEY
```

The validation job uses 1 GPU/2 CPU/16GB and runs each arm's 100-update/full-batch
preflight plus one full evaluation episode, in separate processes and separate
validation directories. When GPU resources are queued, an additional compute-node
CPU validation uses the same frozen entrypoints with --device cpu and separate
validation_cpu output directories. It does not substitute for CUDA validation.
Formal jobs can be queued with --dependency=afterok:GPU_VALIDATION_JOB and
--kill-on-invalid-dep=yes, so they can start only after all three GPU checks pass;
they repeat their own isolated preflight before training from scratch for 1M steps.
The copied per-campaign run.sbatch entrypoints remain usable without modification.

Fresh W&B runs append `-retry1-env` and link their previous run IDs in configuration.
Sync processes use the existing metric bridge on the login node, with Slurm status
and completed_updates as the authoritative progress. Only medium delayed,
medium-replay delayed and Shapley are restarted. Uniform and dense reference
remain cancelled. Publication uses latest GitHub main, clean-export CI and normal
fast-forward updates; remote 5090 work and unrelated local files are preserved.
