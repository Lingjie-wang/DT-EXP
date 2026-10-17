# Maze2D four-arm campaign: move from 5090 to Slurm

On 2026-10-17 the user requested stopping the 5090 campaign and returning to
Slurm. The 5090 process group 1721973 (pipeline, comparison uploader and its
children) was stopped and verified empty before migration. All four arms were
still queued: no formal training had started. The original directory and all
preparation evidence remain intact, with an added `migration_stop.json`.

## Scientific settings and provenance

Continue the same campaign with seed 23, four arms, 1M updates, evaluation every
5k updates over 100 episodes, native sparse evaluation rewards, and best observed
score selection. The official CQL file, Maze2D YAML, pinned D4RL tree, original
HDF5, delayed/uniform/Shapley HDF5 files and already fitted Shapley outputs are
transferred without modification. Verify every frozen SHA256 before migration,
every worker start and completion. Do not refit the predictor on the new server.
The original plan and audits are retained in `provenance_5090/`; rerun the real
D4RL loader audit on Slurm. This migration does not introduce a new seed.

The frozen scientific protocol is documented in
`cql-maze2d-umaze-four-rewards-20261017.md`. Its Python 3.10 / Torch 2.7.1 cu128
runtime was required for the 5090. Slurm instead uses the existing Python 3.9 /
Torch 1.11.0 cu113 environment, with a dedicated virtual environment containing
W&B 0.12.21 and protobuf 3.20.3. NumPy remains 1.23.1. Torch 1.11 CUDA execution
was verified on an allocated RTX 2080 Ti, and the compute node can reach W&B.
Record exact runtime versions and GPU identity in each job's `runtime.json`.
There is no training source modification or algorithm monkey patch.

## Scheduling and durable status

Use independent scripts in `scripts/cql_maze2d_slurm/`; preserve the original
5090 entrypoints. A Slurm array has four tasks, ordered original, delayed,
uniform, Shapley, at most two concurrently. Each task requests one GPU, two CPU
cores, 8 GiB RAM and a 72-hour wall-clock limit in GPULite. Slurm assigns
`CUDA_VISIBLE_DEVICES`; the launcher never overrides it. The 72-hour limit is
an allocation limit, not a change to the 1M-update training budget.

Each task validates the frozen inputs and performs a 100-update CUDA preflight
with online W&B. Its one-episode evaluation is excluded from all scientific
comparisons. The full job starts only after its preflight's native training
telemetry and score have been verified remotely. Failures are recorded and do
not silently restart or overwrite attempts. Automatic Slurm requeue is disabled.
After all 200 evaluations and the final checkpoint exist, allow 5 minutes for
native W&B shutdown before stopping only that worker's finished training child.

The lightweight observer tracks per-worker state and Slurm accounting, including
cancellation, timeout and OOM. It resumes the original comparison run:
https://wandb.ai/2820402607-shandong-university/CORL-DDR/runs/626e7ba3-836b-4a21-bd4c-1605587077d2
The original comparison logic still excludes all preflight evaluations and
publishes four curves, a common evaluation table and best-score table.

Slurm campaign directory:
`/labmount/users/202615385/code/DT-EXP/results/cql-maze2d-umaze-four-rewards-seed23-1m-slurm-20261017`.
Submission IDs are recorded in `slurm/submission.json`; status is in
`queue/status.json`, `<arm>/slurm_status.json`, and `comparison/status.json`.

## Validation

Three focused tests verify migration preserves frozen inputs and provenance,
rejects active/trained source campaigns and repeated adoption, and treats a
cancelled Slurm task as terminal instead of waiting forever. Bash syntax checks
cover the environment and array launchers. Run Ruff against a clean Git export
before pushing the new entrypoint to main. Existing unrelated changes in the
Slurm working tree are preserved; deploy an isolated committed source checkout.
