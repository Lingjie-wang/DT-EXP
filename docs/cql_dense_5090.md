# Original-reward CQL reference, HalfCheetah, RTX 5090

Requested on 2026-10-07: queue original-reward CQL on HalfCheetah-medium-v2 and
HalfCheetah-medium-replay-v2, three seeds per dataset. Six independent runs use
policy/evaluation seeds **1, 11, 12** and **300,000 updates** each. Seeds are
deliberately paired with the delayed/Shapley controls, not additional replicates
of an existing dense run. The previously cancelled dense seed-0 experiment stays
cancelled; this campaign has separate preparation, outputs, IDs and queue entry.

## Reward and comparison protocol

Read original per-transition rewards directly from the same HDF5 files used by
the existing pilots. No return redistribution, reward normalization, total-return
correction or Shapley predictor is applied to these rewards. Require source hashes:

| Dataset | Original HDF5 SHA256 | Samples / episodes |
| --- | --- | --- |
| medium | `41324e8487a9556d8cfa4538827f8937f4d63dae3030825ebc9f3a726ed08ca0` | 1,000,000 / 1000 |
| medium-replay | `48d494a4770c11f48260736dec090b78bdca8375647ed4d24bfa5b6610c7f683` | 202,000 / 202 |

First reconstruct and compare the entire delayed dataset against its frozen
reference, then replace only `rewards` with the original float32 step rewards.
Observations, actions, next observations, sample selection, terminal/timeout
masks and normalization remain matched. The finite-episode protocol retains
timeout transitions and stops bootstrapping there. **This is a matched dense
reference, not an unchanged reproduction of CORL's default data-loading path**,
which discards timeout transitions. That difference is retained deliberately to
isolate reward-information changes in comparisons with delayed/Shapley CQL.

Copy existing CQL algorithm, trainer and evaluation Python files byte for byte.
Only selected YAML seed, max_timesteps and logging identifiers change. Keep batch
256, gamma .99, CQL alpha 10, policy lr 3e-5, critic lr 3e-4, state normalization,
10 evaluation episodes every 5000 updates and checkpoints every 100k. The dense
reference intentionally has more reward information than delayed-feedback methods.
No new dense artifacts are added to their frozen inputs.

Use the same 300k budget chosen from the seed-1 pilot for the recent repeat queue.
Report last-10 evaluation mean (255k--300k), final 300k score and best within 300k
separately, with all three seeds. Truncate the historical seed-1 delayed/Shapley
curves to the same budget when comparing. More original reward information does
not guarantee higher measured performance with fixed hyperparameters.

## Queue and reproducibility

The predecessor is `cql-repeats-seeds11-12-300k-5090-20261007`. Give its eight jobs
priority. The new dependency watcher starts admitting dense jobs only after all
predecessor jobs have been launched or terminated, and all still-running predecessor
training PIDs are visible on the GPU. This prevents racing two managers for the
last slots during CUDA initialization. Failed/stale managers block admission for
inspection. GPU query failures wait and retry; no historical task is signalled.

Then reuse the published resource queue without changing it: at most **four total
GPU compute processes**, counting existing users; two jobs admitted together,
3GiB per job plus 2GiB GPU headroom, at least 8GiB free host memory. Dispatch the
two datasets for seed 1, then seed 11, then seed 12. Before each pair trains, each
run must pass its separate 100-update CUDA preflight and one evaluation episode.
Training begins from scratch. W&B observers run independently and upload progress
and checkpoints; queued W&B IDs become live when training starts.

```bash
python -m scripts.cql_dense_5090.prepare \
  --project /home/sckd02/workspace/DT-EXP \
  --medium /path/to/halfcheetah_medium-v2.hdf5 \
  --medium-replay /path/to/halfcheetah_medium_replay-v2.hdf5 \
  --code-revision PUBLISHED_COMMIT
bash scripts/cql_dense_5090/launch.sh
```

Deploy an immutable archive of the published commit and run the launcher in its
own tmux session. Reuse the existing read-only 5090 environment and ignored W&B
credentials. Root: `results/cql-dense-seeds1-11-12-300k-5090-20261007`. Dependency
state is in `dependency/status.json`; training queue state in `queue/status.json`.
Inspect live PIDs and saved state before any recovery; existing attempts cannot
be overwritten or silently restarted.

W&B project: `2820402607-shandong-university/CORL-DDR`. For N in 1,11,12:

- `CQL-CORL-HCM-dense-matched-v1-seedN-300k-5090`
- `CQL-CORL-HCMR-dense-matched-v1-seedN-300k-5090`

Publish scoped commits on current GitHub main with normal fast-forward updates,
clean-export CI and Actions verification. Preserve both dirty working trees and
all historical experiments. Back up launch metadata across servers. Datasets,
checkpoints, environments, generated results and secrets remain outside Git.
