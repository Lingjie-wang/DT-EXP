# Original v3-high replay: seeds 0/1/2, new runs to 100k

Requested 2026-09-21 after the archived seed-3/4/5 experiment: repeat the old
seed-0/1/2 protocol on those SAME seed numbers. The user explicitly extended
the endpoint to 100k. Compare the 75k evaluations to the original 75k results;
report 100k as a separate extension, not as an equal-budget historical result.

## Fixed learning implementation

- v3, warmup and miner: detached full repository at
  `25626cefd9465dadbd0a6ef95756ab943ca7fab6`.
- DT control: detached full repository at
  `e8126627628ebf2e928796e8a5db3f81fbcd3aa3`.
- No edits to archived learners. Actual file hashes are checked and journaled
  before every training worker. Original seed-0/1/2 launcher arguments are
  compared in CPU tests; the shared seed-1 recipe has the same learning flags.
- Re-run the original `sap_dt_one_sided.py` warmup from scratch for each seed,
  save a NEW checkpoint before the first auxiliary update at 50k, and mine NEW
  strict pairs using the original per-seed diagnostic settings.
- Fork both ordinary DT and v3-high from that seed's same new 50k checkpoint.
  Restore model, optimizer and scheduler as the original code does. Do not add
  RNG/loader restoration, paired auxiliary forwards or isolated evaluation RNG.
- v3: preference 0.05, reference 0.1, margin 0.05, high-only RTG 12000,
  pair batch 256, active normalization with floor 16, priority mix 0.5, EMA 0.9.
- Ordinary DT batch 4096, context 20, original optimizer/model/dropout settings;
  delayed train/evaluation, targets 12000 and 6000, 100 evaluation episodes,
  evaluation seed 42, every 5k during continuation.
- Preserve legacy indexing: continuation logged 50000..100000 inclusive,
  `update_steps=100001`. The recorded 50k evaluation follows one continuation
  update. At 75k and 100k there have been 75001 and 100001 total updates.

## Hardware and scheduling

Match the original GPU MODEL per stage, with a runtime guard. Do not silently
relax this new replay profile using the broader legacy345 GPU permission.

| Seed | Warmup/mining | DT continuation | v3-high continuation |
| --- | --- | --- | --- |
| 0 | RTX 3090 | RTX 4090 | RTX 3090 |
| 1 | RTX 3090 | RTX 3090 | RTX 3090 |
| 2 | RTX 3090 | RTX 3090 | RTX 4090 |

GPUNorm, one GPU/job, 12 CPUs, 32 GB RAM (the previously approved reservation,
not a batch-size reduction). The historical runs requested 48 GB. Jobs may
overlap up to the account's two-job limit; preserve Slurm resource isolation.
Same GPU model is not a guarantee of bitwise determinism. Historical
`deterministic_torch=false` and the existing library environment are preserved.

## Isolation and provenance

Campaign: `results/v3-high-replay012-20260921/`.
Profile: `replay012`; legacy345 remains the default, with unchanged behavior.
An explicit campaign manifest rejects accidentally using another profile's
directory. Existing result directories are never overwritten.

Each seed has `production/seedN/dt-step50000.pt`, `pairs/`, `records/prepare/`,
`records/dt/`, `records/v3/` and final branch checkpoints. Existing historical
checkpoints and pairs are READ ONLY. Their hashes are recorded at initialization.
After each new warmup/miner subprocess exits, a separate CPU comparison records
model-tensor equality, maximum tensor difference, normalization, and pair-array
equality/counts in `original_comparison.json`. File hashes alone cannot establish
model equality because saved configs/paths and serialization metadata differ.
This comparison cannot change the training subprocess's RNG or parameters.

W&B project: existing `CORL-DDR` (historical CLI `corl-ddr`).
Group: `Replay012-OriginalV3High-To100k-HCMR-delayed`.
Names: `Replay012-{prepare,dt,v3}-seed{0,1,2}-HCMR-delayed`, followed by the
original trainer's environment/UUID suffix. Metrics/config/hardware are uploaded;
checkpoints and pair arrays remain server-local. No recurring monitor is added.

## Validation and launch

Initialize with `legacy_v3_high.py init --profile replay012`. Use
`run_v3_high_replay012.sbatch smoke --seed 1` on a 3090: seed 1 matches that GPU
in all three stages. Smoke is offline, uses an isolated directory, and never
initializes production. Run full CPU protocol tests in the training environment.

After successful smoke, run three `prepare --seed N` jobs (12-hour limit), then
two `branch --seed N --stage dt|v3` jobs per seed (18-hour limit), each with
`afterok` on its own preparation. For DT seed0 and v3 seed2, override exclusions
to allow only the known 4090 nodes gn7/gn8/gn12. All other stages use gn4/gn5.
No completed or running legacy345 job is restarted, changed or removed.

This is a test of historical reproducibility, not a guarantee of reproducing
the old improvement. Old full runtime source snapshots were not archived for
every run; the saved Git sources and actual old W&B configs are the evidence.

## Submission record (2026-09-21)

Implementation: `36586fa`. All 13 protocol tests pass in the server's actual
PyTorch environment (12 pass locally, one PyTorch-dependent test skipped there).
Repository-wide local Ruff and shell syntax checks pass. The old checkpoint/
pair self-comparison passes for all three seeds; original strict pair counts
are 2124, 2122 and 2212 respectively. Original inputs remain unchanged.

GPU smoke **9681** completed on gn4 / RTX 3090, exit 0, elapsed **1m40s**.
Both archived DT and v3 audits passed; the persisted smoke gate records the
two source revisions and `profile=replay012`. Smoke data are never reused in
production, and smoke runs remain offline.

| Seed | New 0-50k warmup + mining | DT 50k-100k | v3-high 50k-100k |
| --- | --- | --- | --- |
| 0 | 9682 | 9685 | 9686 |
| 1 | 9683 | 9687 | 9688 |
| 2 | 9684 | 9689 | 9690 |

The two seed-0/1 warmups started concurrently on gn4/gn5 (3090). Seed 2 awaits
an account slot. Each continuation has an `afterok` dependency on its own
warmup/mining job; historical-model node filters and runtime guards remain in
effect. This records submission/startup, not completed training results.
