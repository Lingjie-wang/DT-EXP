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
The full pipeline waits for all required periods; it cannot silently substitute
the smoke training set for P7-P13. Download and pipeline tmux sessions are
`auctionnet-dt-download-parallel-20261005` and `auctionnet-dt-pipeline-20261005`.
