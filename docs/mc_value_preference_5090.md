# Frozen Monte Carlo value preferences, queued last

This independently named experiment uses original rewards and a frozen state
value model to score actions. It preserves all historical source, configurations,
checkpoints, results and active queues. It is our method, not a paper reproduction.

## Value regression and action labels

1. Use HalfCheetah-medium-replay-v2's original rewards to compute
   `G[t] = sum(rewards[t:])`, gamma 1. There are 202 complete 1000-step episodes.
   Dataset SHA256: `48d494a4770c11f48260736dec090b78bdca8375647ed4d24bfa5b6610c7f683`.
2. Train one MLP `V(s,t)`: standardized current state plus `t/1000`, two 64-wide
   ReLU hidden layers, one scalar output. Predict `0.001 * G[t]` using ordinary
   MSE. AdamW, LR 0.001, weight decay 0.0001, batch 1024, 50 epochs, seed 1729.
   No Q network, bootstrapped training targets, target network or joint DT update.
3. Split by WHOLE trajectories: 161 training episodes, 41 validation episodes,
   fixed shuffled split from seed 1729. Select the epoch with minimum held-out
   MSE, then freeze. DT state normalization is shared with existing controls;
   it uses all observations but no reward targets. Value fitting uses a private
   shuffle stream and restores CPU RNG; DT initialization is reset identically.
4. Convert V back to raw return units. Score each observed action using
   `r[t] + V(s[t+1],t+1) - V(s[t],t)`. Within an episode, the next observation is
   its next recorded state. The HDF5 audit confirmed exact equality with
   `next_observations` at every non-final transition. At the 1000-step endpoint
   set next V to ZERO: this is the finite-horizon task, not continuing-task
   time-limit bootstrapping. Never join adjacent episodes. The original dataset
   has 202 timeouts and no earlier true terminations.
5. At each timestep, take upper/lower 30% by this score, then match each upper
   state to the nearest lower state at the SAME timestep with standardized state
   RMSE <= 0.5. Same distance rule, quantiles and uniform sampling as other C runs.
   No learned thresholds, prefix/action-distance filters, or extra pair weighting.

This predicts average continuation in the offline replay distribution, not an
optimal-policy value or a proven causal action advantage. Trajectory labels are
correlated. Validation checks generalization to unseen trajectories for V; it
does not prove local advantages are correct. Mining uses all trajectories,
including V's training trajectories: this simple version is NOT cross-fitted.
Overfitting V to sample returns could collapse the scores, so retain validation
curves, score spread and pair coverage. Do not silently tune on policy scores or
relax the pair cutoff if no supported pairs exist.

A local CPU preprocessing audit (Torch 1.11, not the server training runtime)
completed all 50 value epochs: selected epoch 50, held-out R2 0.87969, raw RMSE
479.96 versus 1386.76 for the training-mean predictor. It yielded 15,217 pairs
(24.95% coverage), mean state RMSE 0.40630. Ordering disagreed with observed RTG
on 44.08% of pairs and immediate reward on 48.91%. Score standard deviation was
254.62 raw units: a good long-horizon fit does not guarantee precise one-step
differences. Log reward, value-change and score dispersion explicitly. This is
preprocessing validation, not a policy result; the server fits its own frozen V
with the same declared protocol after all earlier experiments complete.

## Unchanged policy experiment

- Ordinary DT uses recorded original-reward RTGs. Auxiliary conditioning is the
  same dense budget `12000 - sum(positive_prefix_rewards)`. Candidate actions
  share the positive state's context; no future reward enters its target budget.
- Keep `DT_MSE + 0.05 * mean(relu(d_pos - stopgrad(d_neg) + 0.05))`, with all
  pairs in the denominator. Negative actions only gate positive imitation.
- Seed 0 from scratch, ordinary batch 4096, auxiliary batch 256, context 20,
  embedding 128 / 3 layers / 1 head, LR 0.0008, warmup 10k, math SDPA, 100k updates.
- Every 10k: evaluate initial target 12000 with original-reward budget updates,
  100 episodes, seed 42. Final 100k is primary; mean of 60k/80k/100k is secondary.
  Save checkpoints every 5k. This is a single-training-seed pilot.
- Reuse the existing dense DT baseline. Check configuration, initial model,
  dataset, package versions and shared-code hashes. A zero-weight branch remains
  available for execution validation, not an additional scheduled baseline.
- The frozen value checkpoint is saved separately and inside policy checkpoints.
  Explicit recovery in a new directory repeats the deterministic preprocessing
  and validates the pair hash and method settings before restoring DT state.
  Worker prefetch state is not saved; exact interrupted-batch replay is not claimed.

## Outputs and W&B

Entry point: `algorithms/offline/mc_value_preference_dt.py`.
Config: `configs/offline/dt/halfcheetah/mc_value_preference_5090.yaml`.
Reuse the isolated Python 3.10.21 / Torch 2.7.1+cu128 server runtime, through
`STATE_ONLY_RUNTIME_BASE=/home/sckd02/workspace/DT-EXP`. The launcher reads the
existing ignored W&B credential file without printing it.

W&B project `2820402607-shandong-university/CORL-DDR`, group
`MCValuePreference-HCMR-5090-20261005`. Record every value epoch locally, then upload
the complete fit-history table, held-out RMSE curve, R2, selected epoch, split
IDs, frozen model, per-state values/scores and pair audit at policy step 0.
Stream policy losses every 100 updates and evaluation plus episode tables every
10k. Value pretraining epochs are separate from policy optimizer-step numbering.
The fit uses validation MSE, never policy evaluation, to select its checkpoint.
Datasets, generated artifacts, model checkpoints and secrets stay out of Git.

## Queue ordering

Append this independent queue after the ENTIRE single-step reward campaign.
It verifies that single-step C and its preceding dense RTG C/DT campaign all
completed 100k and final evaluations, then waits for the prior queue process to
exit. A failed or disappeared predecessor halts the queue. No prior queue is
edited. Exclusive locking prevents duplicate launches; outputs must be new.

Server paths under `/home/sckd02/workspace/DT-EXP`:

- Source: `.runtime/mc-value-preference-source-20261005` (frozen published commit).
- Campaign: `results/mc-value-preference-5090-seed0-20261005`.
- Tmux: `mc-value-preference-queue-20261005`.
- Log: `logs/mc-value-preference-queue-20261005.log`.

```bash
python3 scripts/mc_value_preference/queue.py \
  --dependency /home/sckd02/workspace/DT-EXP/results/step-reward-preference-5090-seed0-20261005 \
  --dependency-commit 2775c67e5cd15befbc0109bf745295db816f808d \
  --dependency-queue-pid 602164 \
  --output-root /home/sckd02/workspace/DT-EXP/results/mc-value-preference-5090-seed0-20261005
```

Once ready, run a full-batch CUDA smoke with two value epochs, three DT updates
and one real episode (offline W&B). Verify baseline compatibility, then train
formal C with the full 50-epoch value fit. Save the final comparison against
single-step C, RTG C and DT. The queue survives SSH disconnects; server reboots
need explicit inspection. Manifests pin source, dependency and process identity.

## Validation

Tests cover MC target construction, whole-episode split isolation, terminal-zero
values, no episode-boundary leakage, score-based matching, reconstruction from
the saved value model, isolated RNG, and full predecessor-chain completion.
Retain the existing C gradient and dense-context tests. Run Ruff 0.0.278 against
a clean tracked export before each publication and inspect GitHub Actions after.
