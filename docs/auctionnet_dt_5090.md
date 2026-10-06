# PRGS ordinary DT on AuctionNet, RTX 5090

The requested run uses the PRGS authors' ordinary DT (`is_stitch: false`) on
AuctionNet. Historical experiments and environments are preserved. The new entry
points are under `scripts/auctionnet_dt`; no existing learner is replaced.

## Source and fidelity

- PRGS: <https://github.com/deligentfool/PRGS>, commit
  `33bad7bc3e3f7cbfbefb68294b8e61069a6e58d4`, `AuctionNet/`.
- AuctionNet: <https://github.com/alimama-tech/AuctionNet>, commit
  `c20de631a40e24669d1d6a8e2acefd8f492f172d`, official
  `strategy_train_env/bidding_train_env/train_data_generator/train_data_generator.py`.
- `upstream_manifest.json` pins all 17 PRGS Python/YAML files and the generator.
  The launcher checks the Python files before and after execution. Training runs
  the original `python -u main.py --algo dt --env AuctionNet` in a copied directory.
- No model, loss, optimizer, sampler, training loop, or evaluator source changes;
  no monkey patches. Logging/W&B observation happens in the parent process.
- PRGS does **not** publish its processed training pickle or normalization file.
  We reconstruct these from public raw data through the unchanged AuctionNet
  generator, then convert RL rows to trajectory dictionaries. This reconstruction
  has not been verified against the authors' private data. Do not call the result
  an exact reproduction of their published scores.

## Data protocol

The URLs come from AuctionNet's `pre_generated_dataset/readme_dataset.md`:
`https://alimama-bidding-competition.oss-cn-beijing.aliyuncs.com/share/final/`
`autoBidding_general_track_final_data_period_<range>.zip`.

Use final/general data, not the different AIGB or extended-trajectory datasets.
Download groups 7-8, 9-10, 11-12, 13, 14-15, 16-17, 18-19, 20-21; extract only
P7-P20. ZIP downloads total about 12.8 GB, extracted CSVs about 54 GB. Preserve
source files, ZIP CRCs, measured SHA256s, sizes and URL/ETag records. These SHA256s
identify the downloaded bytes; they are not publisher-provided checksums.

Training is P7-P13 only. P14-P20 are held out. Run the official generator for each
training period, preserving its 16 state features, scalar bid multiplier, exposed
conversion-count reward, and done flags. Split at done and exclude singleton
episodes/incomplete tails, following the official DT loader's convention; never
combine advertisers. Store vector rewards (PRGS format). Compute state mean and
`std + 1e-6` from retained training states only. No delayed reward conversion.

The official PRGS environment configuration has `max_ep_len: 96`, while the
official AuctionNet generator uses 48 steps. This discrepancy is recorded and
left unchanged; any investigation/correction belongs in a separate comparison.

## Training and evaluation

Keep original DT/default hyperparameters: 3 layers, width 128, 1 head, context 20,
batch 64, AdamW learning rate 1e-4, weight decay 1e-4, 10k warmup, dropout 0.1,
gradient clipping 0.25. Train 10 iterations of 10k updates (100k total), evaluating
all original targets `[1.0, 0.8, 0.6, 0.4, 0.2]` on P14-P20 each iteration.
The primary result is the final 100k target-1.0 score, not a selected best target
or checkpoint. Preserve the official model serialization/checkpoint schedule.
The original main script does not set RNG seeds; no seed injection is added.
This is one unseeded run, not a three-seed paper result. A saved model contains
no optimizer/scheduler resume state; exact interrupted continuation is not claimed.

Only the copied configuration's three input path fields change for the full run.
The separate smoke run uses P7-P8, 10 updates, target 1.0 and all P14 opportunities
for one advertiser. Smoke results validate execution only and are not benchmark
scores. Synthetic CUDA validation is a separate process and does not seed training.

## Isolated runtime and server paths

Server project: `/home/sckd02/workspace/DT-EXP` on `100.89.227.18`.

- Environment: `.runtime/auctionnet-dt-5090-env`, Python 3.10.21,
  PyTorch 2.7.1+cu128 for RTX 5090; other pins in `requirements.txt`.
- Frozen upstream: `.runtime/auctionnet-upstream-20261005`.
- Download/preprocessing cache: `.runtime/auctionnet-dt-data-20261005`.
- Smoke: `results/prgs-dt-auctionnet-5090-smoke-20261005`.
- Training: `results/prgs-dt-auctionnet-5090-20261005`.
- W&B: project `CORL-DDR`, entity `2820402607-shandong-university`,
  group `AuctionNet-DT-PRGS-5090-20261005`.

Create a new environment with Python 3.10, install Torch from
`https://download.pytorch.org/whl/cu128`, then install
`scripts/auctionnet_dt/requirements.txt`. The initial direct Torch download was
stopped in favor of an already verified local official wheel and cached CUDA
dependencies; direct PyPI access stalled, so installation used a temporary SSH
proxy. No credentials are embedded in scripts or committed.

```bash
python3 scripts/auctionnet_dt/download_data.py \
  --root .runtime/auctionnet-dt-data-20261005 --workers 8
.runtime/auctionnet-dt-5090-env/bin/python scripts/auctionnet_dt/verify_runtime.py \
  --upstream .runtime/auctionnet-upstream-20261005/prgs/AuctionNet \
  --output .runtime/auctionnet-dt-data-20261005/runtime.json
bash scripts/auctionnet_dt/pipeline.sh
```

`pipeline.sh` waits for complete CSVs, preprocesses P7-P8, completes the smoke,
then prepares the remaining training periods and launches the full run. Each
training attempt requires a new output directory. It reads the existing ignored
`.dt_runs/wandb.env`; credentials never enter Git. CPU math thread count is 4.
W&B receives observed TensorBoard scalars and model files; TensorBoard/console,
protocol, data audits, status and parsed evaluations are retained locally too.

Commit and publish each completed code change promptly and inspect Actions.
Keep only this task's scripts, tests, pins and documentation in its commit;
preserve unrelated dirty files. If direct server GitHub access is unavailable,
transfer the changes back to the primary workspace and publish from there.

## Validation and current execution

On 2026-10-05, the isolated server environment passed `pip check`, all five
data-boundary/split tests, and an actual official DT CUDA training step at batch
64/context 20. Gradients and loss were finite, and the MMD parameters had no
gradients. `requirements-lock.txt` records this environment without local wheel
paths. This synthetic check is not a benchmark result.

The public P7 and P8 data have been processed by the unchanged official generator:
48 trajectories each, respectively 1,910 and 1,949 retained transitions. The
combined 96 trajectories / 3,859 transitions are for the separate smoke only.
The full pipeline requires all periods and cannot silently substitute the smoke
training set for P7-P13. Download and pipeline tmux sessions are
`auctionnet-dt-download-parallel-20261005` and `auctionnet-dt-pipeline-20261005`.

The real-data smoke completed on 2026-10-05 at about 22:06 Asia/Shanghai:
10 updates plus one held-out P14 advertiser evaluation, with the official Python
source unchanged. Its score of 0.0 is an execution check after only 10 updates,
not a useful performance measurement. The checkpoint and metrics were synced to
[W&B run 0xdy3ncr](https://wandb.ai/2820402607-shandong-university/CORL-DDR/runs/0xdy3ncr).
Status, protocol, logs, metrics and checkpoint were also copied to the primary
workspace's ignored `.runtime/auctionnet-dt-records-20261005/smoke/` directory.

Full P7-P13 preparation produced 336 trajectories / 13,972 retained transitions.
Each period has 48 trajectories; transition counts for P7 through P13 are
1,910, 1,949, 2,050, 2,005, 2,066, 1,994 and 1,998. All seven periods' observations,
actions, rewards and done flags matched the actual official AuctionNet
`EpisodeReplayBuffer` output at `rtol=atol=1e-12`. The reference loader was
`strategy_train_env/bidding_train_env/baseline/dt/utils.py` at the pinned AuctionNet
commit, SHA256 `0c37581500b22c42ec78267ee4dc00be0cd57aa24278e5f3cd83fd4ed9a4dc9d`.
This validates the format conversion, not equivalence to PRGS's unpublished data.

Full prepared-data SHA256s:

- `training_data_small.pkl`: `83281032a1c48807069fd365a74070822a16dc272a5dec07618f9014c0c172fd`
- `normalize_dict.pkl`: `0cd59199cb9640a0eb61907e67badfb87014a713dd6aa8dfdc9ce0f38b35e8a7`

All required public raw data finished downloading and extracting on 2026-10-05.
The full 100k-update job started at about 22:17 Asia/Shanghai. By 22:20, it had
completed its first 10,000 updates with finite logged loss and entered evaluation;
P14 and P15 had returned scores. Training ran at about 94 updates/second. The job is
running in tmux and can continue after the interactive SSH connection closes.
Follow [W&B run 5sm7osbe](https://wandb.ai/2820402607-shandong-university/CORL-DDR/runs/5sm7osbe).
Training throughput excludes the full official bidding evaluations; no completed
100k result is available at this launch check.

Prepared training pickles, normalization, per-period audits and reference-loader
comparison records are backed up under the primary workspace's ignored
`.runtime/auctionnet-dt-records-20261005/data/`. An initial snapshot of the full
run's frozen source, protocol and logs is under `full/` in that same backup root.
The running observer uploads metrics and each official checkpoint to W&B.

Implementation commit `804bdbd90694caf8babd7758c74781011cbb6f5f` was published to
GitHub; [Actions run 37321428963](https://github.com/Lingjie-wang/DT-EXP/actions/runs/37321428963)
passed. The data, environment, W&B credentials and generated results are not in Git.

## Sequential non-final/general DT run (2026-10-06)

The user requested ordinary DT on the newly identified non-final/general dataset
after the existing final/general experiment finishes. The independent entry point
is `scripts/auctionnet_dt_nonfinal/launch.sh`. It downloads ahead of time, but
does not preprocess or start training until the predecessor reports `completed`,
100,000 updates and unchanged upstream Python. Reaching 100,000 updates while
still evaluating is insufficient. A failed predecessor blocks the queue.

The source is the Alibaba competition's original
[general-track download list](https://github.com/alimama-tech/NeurIPS_Auto_Bidding_General_Track_Baseline#dataset-link):
`https://alimama-bidding-competition.oss-cn-beijing.aliyuncs.com/share/`
`autoBidding_general_track_data_period_<range>.zip`.
Unlike the first run, these URLs contain neither `/final/` nor `_final_`.
The eight groups cover P7-P20 (P21 is excluded) and total 12,696,781,121 compressed
bytes. `dataset_manifest.json` records the measured sizes and ETags; the downloader
checks them, supports byte-range resume, retries transport failures up to five
times, and verifies extracted file CRCs and SHA256s. ETags are identity checks,
not claimed to be cryptographic whole-file checksums.

Range samples from P7 (120,067 rows) and P14 (115,409 rows) each included all
48 advertisers and CPA values 6-12. In contrast, the existing final/general data
has CPA 60-130 and mean retained training return 26.7. This supports a different
conversion regime; it does not establish equivalence to PRGS's unpublished data
or guarantee its reported DT mean score of 267.3. The similarly named AIGB files
are not interchangeable: the sampled non-final/general and AIGB P7 rows differed
in `pValueSigma` and `conversionAction`. Use general-track raw data only, with no
extended or generated trajectory additions.

The only experimental change is the dataset version. Retain ordinary DT,
`is_stitch: false`, P7-P13 training, P14-P20 evaluation, original step rewards,
100,000 updates, all five targets, and the final target-1.0 primary result. The
official 48-step-generator versus 96-step-evaluator discrepancy and target-return
settings remain unchanged in this comparison. Their effects require separately
named controls. No existing source/configuration/checkpoint/result is modified.

- Cache and queue state: `.runtime/auctionnet-dt-nonfinal-data-20261006/`.
- New run: `results/prgs-dt-auctionnet-nonfinal-5090-20261006/`.
- Predecessor: `results/prgs-dt-auctionnet-5090-20261005/`.
- tmux session: `auctionnet-dt-nonfinal-queue-20261006`.
- Logs: `pipeline.log`, `download.log` and `queue_status.json` in the new cache.
- W&B: existing `CORL-DDR` project and `AuctionNet-DT-PRGS-5090-20261005` comparison
  group; the new run has a distinct `nonfinal` name and dataset-version metadata.

The new pipeline reuses the validated preprocessing, run preparation and observer
scripts from `scripts/auctionnet_dt` without modifying them. After preparation it
corrects provenance text in the new run's `protocol.json` to identify non-final
data; it does not change upstream code or effective configuration. Normalization
is recomputed from the new training data only. W&B uploads metrics/checkpoints.

```bash
mkdir -p .runtime/auctionnet-dt-nonfinal-data-20261006
tmux new-session -d -s auctionnet-dt-nonfinal-queue-20261006 \
  'bash scripts/auctionnet_dt_nonfinal/launch.sh > .runtime/auctionnet-dt-nonfinal-data-20261006/pipeline.log 2>&1'
```

Run this from the server project directory. The launch script uses the existing
dedicated environment and ignored W&B credentials. An advisory lock prevents
duplicate queues. Prior run directories are never reused; failures stop visibly
in `queue_status.json` and need inspection before a new attempt. Six queue-gate
tests cover pending/running/failed/incomplete/verified completion, in addition
to the five existing data-conversion tests.

## Sequential delayed-reward DT on non-final/general (2026-10-06)

The user additionally requested ordinary DT with delayed rewards on the same new
dataset. Execution order is final/general step-reward DT, non-final/general
step-reward DT, then non-final/general delayed-reward DT. The independent launcher
is `scripts/auctionnet_dt_delayed/launch.sh`. It waits for successful completion
of the second experiment, including final evaluation, 100,000 updates and source
verification. Missing/running predecessors wait; failed predecessors stop the
queue. No GPU process starts while waiting.

Training uses the official `SequenceDataset` implementation with
`delayed_reward: true`: for each trajectory, set all earlier rewards to zero and
place that trajectory's original reward sum at its final transition. Returns,
observations, actions, terminal flags, normalization and trajectory selection
stay unchanged. Prepared data are hash-verified against the ordinary run's audit
and copied to an independent snapshot; reward conversion occurs in memory and
does not rewrite either pickle. The ordinary predecessor must identify itself as
non-final/general, `model_type: dt`, `is_stitch: false`, and non-delayed.

This is an explicitly modified task variant, not an unchanged official baseline.
The official evaluator does not consult `delayed_reward` and normally supplies
each intermediate conversion reward to `model.take_actions`. In the independent
runtime copy only, one line in `evaluation_bidding.py` changes from
`pre_reward=pre_reward` to `pre_reward=0.0`. This keeps evaluation RTG constant
within each episode, consistent with terminal-only reward feedback. No action
is requested after the terminal reward. This is a requested reward-protocol
change, not a compatibility fix; there was no upstream runtime error motivating
it. The patch is saved as `delayed_reward.patch`, and the protocol records both
the original and modified hashes. No monkey patch is used in the experiment.
The observer is independently copied to report `upstream_python_unchanged: false`
and verify `runtime_source_unchanged: true` on successful completion.

Auction dynamics, CPA-penalized total-conversion scoring and state features,
including historical conversion statistics, remain unchanged. Thus this is a
change to explicit reward timing, not a delayed-observation environment. Equal
actions retain equal scores; a differently trained policy can have different
scores. The same P7-P13/P14-P20 split, 100,000 updates, five targets, architecture,
optimizer, unseeded official training flow and 48/96-step discrepancy are retained.
Primary comparison remains the final target-1.0 score. Matching the paper's
unpublished prepared data remains unverified.

- Queue: `.runtime/auctionnet-dt-nonfinal-delayed-queue-20261006/`.
- Independent data snapshot: `prepared-data/` inside the queue directory.
- Results: `results/prgs-dt-auctionnet-nonfinal-delayed-5090-20261006/`.
- tmux: `auctionnet-dt-nonfinal-delayed-queue-20261006`.
- W&B: existing `CORL-DDR` project/comparison group, a distinct delayed run name,
  and explicit `reward_protocol`/`dataset_version` metadata. The run is created
  when training starts, not while the queue is waiting.

```bash
mkdir -p .runtime/auctionnet-dt-nonfinal-delayed-queue-20261006
tmux new-session -d -s auctionnet-dt-nonfinal-delayed-queue-20261006 \
  'bash scripts/auctionnet_dt_delayed/launch.sh > .runtime/auctionnet-dt-nonfinal-delayed-queue-20261006/pipeline.log 2>&1'
PRGS_AUCTIONNET_SOURCE="$PWD/.runtime/auctionnet-upstream-20261005/prgs/AuctionNet" \
  .runtime/auctionnet-dt-5090-env/bin/python -m unittest discover \
  -s tests -p 'test_auctionnet*.py'
```

The additional tests exercise queue failure/wait/release, data provenance and
snapshot independence, actual official loader reward/RTG behavior, and the actual
original/modified evaluator with deterministic test doubles. They check unchanged
states and scoring for identical actions and changed intermediate reward
feedback. Source-dependent tests require `PRGS_AUCTIONNET_SOURCE`; release
validation must set it and run those tests without skips. Existing implementations
and results must continue to be preserved in subsequent work.

## Fidelity audit and three independent ordinary-DT runs (2026-10-06)

The user clarified that the objective is reproducing the paper's results. The
single non-final/general run is an unchanged-official-code execution on
reconstructed public data, **not a completed reproduction of Table 3**. Its final
100k/target-1.0 score is 269.179216; closeness to 267.3 does not prove matching data
or evaluation protocol. In particular, its P20 score is 205.504014 versus the
paper's 241.0 ± 9.9, so average-score agreement must not hide period differences.

Verified alignment: all official ordinary-DT Python hashes, all effective model
and training hyperparameters, 100k updates, non-delayed rewards, no stitching,
P7-P13 training and P14-P20 evaluation. Only three data-path fields differ in the
ordinary run's YAML. The delayed experiment is a separately requested extension,
with its documented reward/evaluator changes; it is not a Table 3 baseline.

Unresolved paper/code differences and provenance limits:

- Appendix B.2 (p18) says all results average three independent runs with standard
  deviations. The official AuctionNet CLI has no seed argument or RNG seeding.
  Neither the paper nor this code specifies the three actual seed values.
- Table 5 (p17) says 32 eval episodes. `config/default.yaml` also says 32, but
  `config/env/AuctionNet.yaml` overrides it to 1. Existing experiments preserve
  the effective official AuctionNet value 1. Within-run std=0 from one evaluation
  is not the across-run standard deviation printed in Table 3.
- The paper does not specify which of the five target returns or which checkpoint
  supplies Table 3. Final 100k/target1.0 is our explicit reporting convention, not
  a verified author convention. Retain every target; do not select per-period
  maxima or tune to the published test results.
- Author trajectory/normalization pickles are not published. Our official
  AuctionNet preprocessing plus format adapter yields 336 trajectories and
  13,972 transitions from non-final/general P7-P13. Exact author-data equivalence
  remains unverified, including trajectory splitting/filtering and normalization.
- The paper uses RTX 3090; the requested 5090 uses Python3.10/Torch2.7.1+cu128.
  This avoids source compatibility patches but is not an identical environment.
- Preserve the previously recorded 48-step preprocessing / 96-step evaluator
  discrepancy. Do not silently change it, eval repeats or seed injection in an
  alleged unchanged official baseline.

Sources: [paper](https://proceedings.iclr.cc/paper_files/paper/2026/file/439bf902de1807088d8b731ca20b0777-Paper-Conference.pdf),
[official AuctionNet configuration](https://github.com/deligentfool/PRGS/blob/33bad7bc3e3f7cbfbefb68294b8e61069a6e58d4/AuctionNet/config/env/AuctionNet.yaml),
[official entrypoint](https://github.com/deligentfool/PRGS/blob/33bad7bc3e3f7cbfbefb68294b8e61069a6e58d4/AuctionNet/main.py).

To address the missing repetitions without altering official behavior,
`scripts/auctionnet_dt_reproduction/launch.sh` launches two additional fresh
processes sequentially, using unchanged official source/config and exactly the
same prepared-data hashes as the completed baseline. They are independent
unseeded runs, **not three explicitly seeded/replayable runs**. The first run
remains untouched and is included regardless of its score; there is no run
selection. This completes the count of independent runs but does not settle the
other reproduction limitations above. The already completed delayed experiment
is preserved and not included in ordinary-DT statistics.

- Queue: `.runtime/auctionnet-dt-three-run-20261006/`.
- Outputs: `results/prgs-dt-auctionnet-nonfinal-repeat2-5090-20261006/` and
  `results/prgs-dt-auctionnet-nonfinal-repeat3-5090-20261006/`.
- W&B: two separate runs in the existing comparison project/group.
- Aggregation: `summary.json` in the queue, generated only after all three runs
  pass 100k/final-evaluation/source/config/data checks. It reports P14-P20 mean
  and sample std (ddof=1), plus population std because the paper's convention
  is unspecified, for all five targets. Primary remains final 100k/target1.0.
- Training never restarts into an existing result directory. Failures stop visibly.

```bash
mkdir -p .runtime/auctionnet-dt-three-run-20261006
tmux new-session -d -s auctionnet-dt-three-run-20261006 \
  'bash scripts/auctionnet_dt_reproduction/launch.sh > .runtime/auctionnet-dt-three-run-20261006/pipeline.log 2>&1'
```

Six additional tests cover final-checkpoint selection, all-target retention,
between-run statistics, incomplete evaluation, data/config/source mismatches,
and rejection of delayed variants as ordinary baselines. Source-dependent tests
from the earlier suites must also run without skips before publication.
