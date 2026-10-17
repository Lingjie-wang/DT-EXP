# Information and fairness audit of the running Shapley CQL experiments

Audited on 2026-10-07. Scope: the RTX 5090 seed 1 delayed and predictive-Shapley
pairs on HalfCheetah-medium-v2 and HalfCheetah-medium-replay-v2. The audit adds
read-only validation and regression tests; it does not alter ongoing experiments,
their frozen source, rewards, checkpoints, configurations or evaluation schedules.

## Finding

No use of the original within-trajectory dense reward pattern was found in the
predictor, attribution or CQL training path. The method's information budget is
offline state/action trajectories and a terminal total return. It transforms
that information into dense **learned** rewards before CQL training. Thus this
is delayed-feedback CQL with reward redistribution, not CQL retaining sparse
terminal-only rewards throughout its own optimization.

Original HDF5 rewards are read during preparation to construct each trajectory's
total. The replay preparer also historically created separate dense/uniform
control datasets; those controls were cancelled. The Shapley fit does not read
them. The standalone CQL runner loads the selected frozen NPZ, not
`d4rl.qlearning_dataset` or the original HDF5.

## Checked against actual artifacts

The audit covered all 202 replay trajectories and all 1000 medium trajectories:

- Predictor NPZ keys are exactly `features`, `returns`, `starts`, `ends`.
  Features equal the baseline's concatenated state and action arrays exactly.
  The state/action history is available to both offline methods.
- Baseline rewards are zero at all nonterminal steps. Predictor returns cast
  to float32 equal the baseline terminal rewards exactly.
- Every non-reward CQL field matches exactly: observations, actions, next
  observations and terminal/timeout masks. CQL source, trainer construction,
  hyperparameters and policy/evaluation seed match; only logging names differ.
- Every trajectory is attributed by its held outer-fold model. Training,
  validation and held indices are disjoint and cover the entire dataset.
  Saved checkpoint split metadata matches the input folds. Input and target
  normalization statistics match calculations on inner-training data only.
- Each checkpoint's selected epoch is the minimum inner-validation loss.
  Replayed full-input predictions match saved predictions; maximum absolute
  differences were 0.0000674 (replay) and 0.0003564 (medium), consistent with
  CPU runtime/batch rounding in the audit.
- All saved permutation seeds/orders and contribution means were checked.
  Reconstructing rewards from saved contributions and totals reproduced every
  actual CQL reward exactly. This is not a claim that every coalition prediction
  was separately recomputed during the audit.
- Evaluation uses the same original Gym return, evaluation seeds and episode
  count. No evaluation reward or evaluation transition is inserted into replay,
  fed into predictor fitting or used to select its checkpoint.

The running server's source/data hashes were rechecked. Its protocol hashes
match the audited local provenance/backup byte for byte:

| Campaign | Protocol SHA256 |
| --- | --- |
| Delayed pair | `4221049d08ac12bfb85a3efaec34eb2604757f0dc0f956cd3392facbc7e077cd` |
| Replay Shapley | `797c03796a57e55ad94a40071ae94119be1ea58dd0b7f7af61d888c483761623` |
| Medium Shapley | `dfe65e02fe28dbdf1c844d7c7a84560285a7c8aabed1176ec2d7090b21a4e441` |

## Counterfactual regression tests

`tests/test_cql_shapley_information.py` exercises the existing preparation and
fitting computations without modifying scientific behavior:

1. Remove the entire within-trajectory dense reward pattern, retaining only its
   exact total at the end. The actual replay preparation entrypoint and medium
   preparation function yield identical predictor inputs and base transitions.
   Only Git revision metadata is stubbed for test exports without `.git`.
2. Change all held-fold states/actions and returns while keeping inner training
   and validation fixed. The actual fitting function produces bit-identical
   model weights. This covers both target and feature/normalization leakage.

These tests complement artifact verification; they cannot prove absence of
every conceivable implementation or provenance error.

## Fairness qualifications

There is a small precision mismatch: predictors use float64 sums from the raw
file, while delayed CQL stores terminal totals in float32. Maximum differences
are 0.0002359785 (replay) and 0.0002439935 (medium). This preserves no information
about *where* reward occurred, but is not bit-identical terminal feedback.
Future strictly matched versions should derive both methods' targets from the
same stored terminal reward values. Historical running campaigns are preserved.

The held trajectory's own total is used to enforce return conservation after
attribution. That total is an available offline label, not an evaluation reward.
The baseline and model residual are distributed uniformly across the trajectory.
This mechanism itself may help even without accurate segment credit; the cancelled
uniform control is not automatically restarted by this audit.

Only the undiscounted trajectory return is conserved. With gamma 0.99, moving
rewards earlier changes the discounted learning objective and reward/penalty
balance. Do not claim discounted objective or policy invariance. The experiment
compares the complete redistribution pipeline to terminal-delayed CQL; without
a uniform control, it cannot isolate the benefit of Shapley-specific attribution.
The extra predictor preprocessing also means equal CQL update counts do not mean
equal total computational budgets. Final superiority still needs completed runs
and multiple policy seeds.

## Repeating the artifact audit

Use a CPU environment with Torch >= 1.13 (the audit used the existing
`.runtime/lpt-official-env`, Torch 2.0.1/NumPy 1.24.4). Load only trusted project
checkpoints. No package changes to the training environment are needed.

```bash
python -m scripts.cql_shapley_audit.check \
  --delayed DELAYED_CAMPAIGN --campaign SHAPLEY_CAMPAIGN \
  --prepared ORIGINAL_PREPARED_CAMPAIGN --fit ORIGINAL_PREDICTOR_FIT \
  --arm medium_replay --output NEW_AUDIT_JSON
python -m unittest discover -s tests -p 'test_cql_shapley_information.py'
```

For medium, select `--arm medium`; its saved server backup supplies the finalized
campaign and fit while the original preparation directory supplies predictor inputs.
JSON receipts are ignored run artifacts under
`results/cql-shapley-information-audit-20261007/`. The audit refuses to overwrite
receipts. Commit only audit code, tests and this report, using a normal fast-forward
update on the latest shared GitHub head after clean-export CI validation.
