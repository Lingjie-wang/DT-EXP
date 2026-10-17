# Maze2D-Umaze: four reward settings with official CORL CQL

Requested on 2026-10-17: one seed, `maze2d-umaze-v1`, original / delayed /
uniform / predictive Shapley. Use the first proposed protocol: retain collection
fragments and delay their cumulative reward; do not terminate at first success.
Preserve all historical code, campaigns and results. Publish this independent
entrypoint to main.

## Fixed scientific settings

CORL revision `6afec90484bbf47dee05fdf525e26a3ebe028e9b`, D4RL revision
`db6e4b34bb5ce2a51dd3879177c0a0223208a614`. Copy the official CQL source and
`configs/offline/cql/maze2d/umaze_v1.yaml` byte for byte. The latter SHA256 is
`be51a3197b81f40a8858a7e268d74576698dc663f499fd34d3021dc981dac51e`.
Use the previously validated isolated Python 3.10 / Torch 2.7.1 cu128 /
W&B 0.12.21 runtime. No training monkey patches, algorithm edits or optimization
shortcuts. Only seed, device and output/logging locations are CLI overrides.

| Setting | All four arms |
| --- | --- |
| Paired seed | 23 |
| Gradient updates | 1,000,000 |
| Evaluation interval / episodes | 5,000 / 100 |
| Batch / discount | 256 / 0.99 |
| Actor / critic learning rates | 3e-5 / 3e-4 |
| State normalization | true |
| Reward normalization | false; scale 1, bias 0 |
| CQL Lagrange / target gap / alpha | true / 5 / 10 |
| Replay capacity / Q hidden layers | 10,000,000 / 3 |
| Evaluation task | Original native sparse Maze2D-Umaze for every arm |

Both servers were checked: 5090 saved plans reserve seeds 1, 11, 12, 21, 22;
Slurm has no saved plan reservation or running user job, and neither server had a
Maze2D campaign. Seed 23 is new and paired. Existing Medium AntMaze jobs all have
1M completion artifacts; previous best evaluations occurred as late as 900k.
Keep the official 1M budget rather than shortening it. This is a one-seed pilot.
Report best observed evaluations, retain all 200 points and final checkpoints.

## Data and information boundary

Official source:
`https://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-umaze-sparse-v1.hdf5`.
Full file SHA256:
`f2d327d95067d53e274ac2fe876b740ed7883d0a9738bd5b447635cdb786bfb7`.
It contains 1M records, four state dimensions, two actions, all-false terminals
and 12,459 timeout markers. These are collection-fragment markers, not success
terminals. Rewards can contain multiple ones within a fragment.

Partition at raw timeouts; keep exactly default `d4rl.qlearning_dataset` indices
0..N-2 excluding timeout rows, skip empty fragments, and retain the official
unfinished tail. Expected: 987,540 transitions, 10,686 nonempty fragments,
3,132 positive fragments, total retained reward 79,445, lengths 21-239.
The total of a fragment is computed over retained transitions only, so the
original arm and all modified arms have the same undiscounted return. This
explicitly excludes rewards on rows the official loader discards.

| Arm | Training reward |
| --- | --- |
| Original | Unmodified HDF5 and native sparse rewards |
| Delayed | All retained rewards zero except last retained step = fragment total |
| Uniform | Same fragment total divided by retained length |
| Shapley | Cross-fitted predictive segment Shapley plus uniform residual |

Only HDF5 rewards at retained indices change. All other arrays, next observations,
terminal flags, timeout flags and excluded rows remain unchanged. Real-loader
audits compare every non-reward array and check each fragment's return. Evaluation
still uses the original environment reward, so all four scores measure the same
task. Conservation is undiscounted only: with gamma .99 the discounted training
objective may change. No optimal-policy invariance is claimed.

Reuse the established AntMaze Shapley design unchanged: 20 near-equal segments,
128 permutations, five contiguous outer folds grouped by timeout fragment;
20% of remaining groups for inner validation; 64-unit segment encoder and head;
Adam lr .001, weight decay .0001, batch 32, up to 300 epochs, patience 30,
eight validation masks. Signed rewards and uniform residual remain unchanged.
Zero-return fragments may receive signed contributions summing to zero.
Prediction-vs-training-mean MSE is report-only, with no tuning using CQL scores.
The CPU fitting subprocess reads only features (state/action), offsets, groups,
and scalar returns. Original per-step reward positions never enter its inputs.
Boundary and length information remain available; adjacent fragments in this
continuous collection need not be statistically independent.

## Execution and telemetry

Independent entrypoint: `scripts/cql_maze2d_four_5090/`. Freeze source, dependencies,
configs and data before use. Fit on CPU (two threads) while the shared GPU is busy.
Then run four loader audits, online CPU preflights (100 updates / one episode,
excluded from scientific comparison), and per-arm GPU preflights before admission.
The durable tmux pipeline refuses to overwrite attempts or silently relaunch jobs.

At resource inspection, seven foreign GPU processes occupied about 28 GiB.
Never interrupt them. Admission permits at most two own trainers and eight total
GPU processes, reserves 4 GiB GPU / 8 GiB host memory, and budgets 5 GiB GPU /
6 GiB host per job. Admit only at GPU utilization <=85%; recheck after preflight.
Wait if resources are insufficient; do not reduce batch or replay capacity.
After all 200 evaluations and the final checkpoint exist, a stuck native W&B
shutdown may be terminated after a five-minute grace period; retain its exit
code and evidence. Never terminate incomplete learning.

Native runs and an independent comparison observer use
`2820402607-shandong-university/CORL-DDR`. The comparison contains all four curves,
one evaluation table and a best-score table. It publishes a pending state before
training without fabricated scores, then adds completed evaluations. Native W&B
has full precision; the comparison parser uses the official CLI's three decimals.
Preflight remote score validation allows exactly this rounding tolerance.

```bash
PROJECT=/home/sckd02/workspace/DT-EXP
ROOT="$PROJECT/results/cql-maze2d-umaze-four-rewards-seed23-1m-5090-20261017"
PYTHON="$PROJECT/.runtime/cql-antmaze-official-wandb012-full-env/bin/python"
PYTHONPATH="$PROJECT" "$PYTHON" -m scripts.cql_maze2d_four_5090.prepare \
  --root "$ROOT" --project "$PROJECT" \
  --reference "$PROJECT/results/cql-antmaze-medium-rewards-seed22-1m-5090-20261014" \
  --inputs "$PROJECT/.runtime/maze2d-umaze-inputs-20261017" \
  --audit "$PROJECT/.runtime/maze2d-umaze-inputs-20261017/seed_audit.json" \
  --revision "$(git -C "$PROJECT" rev-parse HEAD)"
tmux new-session -d -s cql-maze2d-umaze-20261017 \
  "bash '$ROOT/source/scripts/cql_maze2d_four_5090/launch.sh' '$ROOT'"
```
