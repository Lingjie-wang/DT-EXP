# v3-high reference ablation, seed 0, 50k to 100k

Question: does the reference penalty help or constrain v3-high?
Run the original v3-high with reference_weight=0.1 and a matched arm with
reference_weight=0.0. Both resume the original seed-0 delayed DT checkpoint,
with identical optimizer/scheduler state, strict pair pool, batch sizes,
dropout, active normalization, dynamic sampling, and high-only RTG 12000.
The standard DT loss continues to use recorded delayed RTGs.

The zero-weight arm still computes the reference distance, preserving the
forward-call order and its dropout random-number consumption. The weighted
reference term and its gradients are zero; a positive raw reference_loss is
only a diagnostic, not an enabled constraint. No model.eval() change is made.

Inputs, checked before launch:

- checkpoint: checkpoints/dt-halfcheetah-medium-replay-v2-delayed-seed0-step50000.pt
  SHA256 b36430ef5e9e29a090514bf934bca2d5b66625ee6ae18da93ed60aea9e38cbbf
- pairs: results/pair_diagnostics/dt_pair_diagnostic_halfcheetah_medium_replay_delayed_seed0.npz
  SHA256 8ec016c96d81f0cc5b73e0e32521616ee1364f143abc0efccf07dd8f72c8b52a

Keep preference_weight=0.05, margin=0.05, preference batch 256,
dynamic_priority_mix=0.5, dynamic_priority_ema=0.9, and minimum active count 16.
Use the existing CORL medium-replay config (DT batch 4096, LR 0.0008).
Two RTGs (12000, 6000), 100 evaluation episodes each, eval seed 42, every 5k.

Use legacy v3 step indexing: update_steps=100001 runs updates at logged steps
50000 through 100000 inclusive (50,001 additional updates); the 50k evaluation
is after the first continuation update. Both arms follow exactly this scheme.
Primary report: each target's last score and mean over 80/85/90/95/100k.
Also report best over post-50k evaluations and the 75k value for old-run context.
Only the newly matched 100k arms isolate the reference coefficient.

Save separate 75k/100k snapshots with optimizer/scheduler state in each arm's
unique output directory; no previous checkpoint is overwritten. These snapshots
support later reuse, but the legacy trainer does not restore RNG/loader queues
or evolving sampling priorities and does not promise exact resumed trajectories.

Launch:

```bash
sbatch --array=0-1 --time=00:15:00 scripts/dt_experiments/run_v3_reference_ablation_to100k_hcmr_seed0.sbatch smoke
sbatch --array=0-1 scripts/dt_experiments/run_v3_reference_ablation_to100k_hcmr_seed0.sbatch
```

Array 0 is NoRef; array 1 is Ref010. Smoke retains the full training batch and
runs two continuation updates with one episode per target and offline W&B.
The 100k runs log online to corl-ddr. No recurring monitor is installed.
