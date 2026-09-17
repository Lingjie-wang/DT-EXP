# Top-5% trajectory imitation: seed-0 experiment

Dataset: halfcheetah-medium-replay-v2, delayed terminal-sum rewards.
Train from random initialization for exactly 100,000 optimizer updates.
Two arms share initialization, sampling, architecture, optimizer, recorded RTG,
and evaluation. Only the relative top-trajectory loss weight differs (1 vs 2).

Rank **trajectories**, not transitions, by their recorded total return. Select
ceil(0.05 * N), with trajectory index breaking ties. With 202 episodes this is
11 episodes (5.45%). Log IDs, returns, threshold and actual sampling mass.
Do not filter out the remaining trajectories or oversample the selected group.

Let m be the padding mask and w be 2 on selected trajectories and 1 elsewhere.
Use c = sum(m*w)/sum(m) and L = mean(m*(w/c)*(predicted_action-action)^2).
The control uses w=1. This preserves the original CORL padded-token reduction
and makes valid-token weights average one on each minibatch. There is no
reference model, fixed-high-RTG auxiliary, or negative sample.

CORL config: batch 4096, context 20, hidden 128, 3 layers, 1 head, dropout 0.1,
AdamW LR 0.0008, warmup 10k, weight decay 0.0001, clip 0.25.
Evaluate every 5k at RTG 12000/6000, 100 episodes each, eval seed 42. Delayed
evaluation keeps the RTG constant and reports original-return D4RL scores.
W&B steps denote completed updates; legacy runs used a zero-based loop index.

Predefined primary results: last at 100k and mean over 80/85/90/95/100k,
reported separately for each target. Best over all evaluations is secondary.
This is one seed and cannot establish statistical significance.

Save initial, 50k, 75k, 100k and best-per-target checkpoints with model,
optimizer, scheduler, normalization, config, selection and RNG state. Loader
prefetch queues are not serialized, so later reuse is possible but exact batch
stream resumption is not promised. The trainer intentionally has no resume flag.
Initial-model and first-batch SHA256 verify the paired experiment.

Launch from DT-EXP:

```bash
sbatch --array=0-1 scripts/dt_experiments/run_top5_return_dt_delayed_hcmr_seed0.sbatch smoke
sbatch --array=0-1 scripts/dt_experiments/run_top5_return_dt_delayed_hcmr_seed0.sbatch
```

Smoke uses the full 4096 batch, two updates and one episode per target, with
offline W&B and separate checkpoint names. Production uses GPUNorm and corl-ddr.
