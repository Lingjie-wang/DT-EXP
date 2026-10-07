# RTX 5090 CQL seed 1

Run the three active arms from the Slurm Shapley pilot on one RTX 5090:

| Dataset | Reward | Policy and evaluation seed |
| --- | --- | --- |
| HalfCheetah-medium-v2 | Terminal-delayed | 1 |
| HalfCheetah-medium-replay-v2 | Terminal-delayed | 1 |
| HalfCheetah-medium-replay-v2 | Predictive Shapley, 20 segments / 128 permutations | 1 |

Slurm seed 0 jobs remain unchanged. Uniform redistribution and dense reference
were cancelled and are excluded. Shapley attribution is **not refitted**: the
prepared rewards, attribution seeds, and successful prediction gate are reused
byte for byte. Seed 1 is an additional CQL policy run, not a new attribution seed.

## Preserved protocol

Reference code: `5df676531a6b2a3b920f18dd6375e5b2d7dfc562`.
The independent campaign copies every frozen Python source file from the Slurm
retry campaigns. Only YAML seed, run name and group change. No CQL optimization,
architecture, reward scaling, batching or evaluation logic is changed.
Each arm uses 1,000,000 updates, batch 256, discount 0.99, and evaluates every
5,000 updates over 10 episodes. The existing dataset boundary/terminal handling
and reward transformations are documented in `cql_corl_delayed.md` and
`cql_shapley_pilot.md`; these are reward-setting variants of CORL CQL.

Frozen `algorithms/offline/cql.py` SHA256:
`76e0db2667c7f1a51c77840fb2aeaa97fcefa165fe1556e54a5981d925ca3b81`.

Dataset NPZ SHA256:

| Arm | SHA256 |
| --- | --- |
| medium | `669814b10ceae9bab93d84602b206d80b9b03361ed7bd3f3e87b844b039ce69c` |
| medium_replay | `ddea9aabac3fb196b204f18adce811ae288e183ae65c708055c24985f37bffc2` |
| shapley | `d62df061f297738424452087ef6768c4c23bc33c6ed853888a6d760cdeb31bfa` |

The 5090 runtime differs from Slurm: existing isolated Python 3.10.21,
Torch 2.7.1+cu128, NumPy 1.23.5, Gym 0.23.0, D4RL 1.1, mujoco-py 2.1.2.14,
MuJoCo 2.1.0. It is reused without installation or source patches. Explicit
`d4rl.gym_mujoco` registration is already part of the reference retry entrypoints.
Hardware/runtime differences mean bitwise equality with Slurm is not expected.

## Preparation and launch

From a checkout containing this version, with `PROJECT` pointing to the project:

```bash
python -m scripts.cql_5090.prepare \
  --previous "$PROJECT/results/cql-corl-delayed-hc-seed0-20261007-retry1-env" \
  --root "$PROJECT/results/cql-corl-delayed-hc-seed1-5090-20261007" \
  --arms medium medium_replay
python -m scripts.cql_5090.prepare \
  --previous "$PROJECT/results/cql-shapley-hcmr-v1-seed0-20261007-retry1-env" \
  --root "$PROJECT/results/cql-shapley-hcmr-v1-seed1-5090-20261007" \
  --arms shapley
bash scripts/cql_5090/launch.sh
```

Transfer the prepared directories with symlinks preserved. Launch in tmux on the
5090. `launch.sh` defaults to `/home/sckd02/workspace/DT-EXP`; override with
`CQL_5090_PROJECT`. It uses `.runtime/state-only-c-5090-env` and loads existing
ignored W&B credentials. Credentials are never recorded in this repository.

The runner first performs separate 100-update CUDA preflights at the full batch
size, each including a complete evaluation episode. All three must pass before
three fresh formal processes start concurrently. Preflight checkpoints are never
loaded into formal training. Each process has two CPU threads. All status and
logs are under `.runtime/cql-5090-seed1-20261007`, and each campaign retains its
frozen source, config, dataset, audit, validation and training directories.
Existing output directories are refused, preserving all historical attempts.

Independent W&B observers read training files and retry service failures without
blocking training. Observers survive manager exit and retain their local W&B
spools. They upload completed checkpoints and verify the final score by API.
If a bridge is interrupted, restart `python -m scripts.cql_5090.observe --root
CAMPAIGN --arm ARM` under the same runtime and credential environment; the stable
run ID is reused. Local JSONL, evaluation JSON and checkpoints remain the source
of truth. Preserve the observer spool when recovering an interrupted upload.

W&B project: `2820402607-shandong-university/CORL-DDR`. Names:

- `CQL-CORL-HCM-delayed-seed1-1M-5090`
- `CQL-CORL-HCMR-delayed-seed1-1M-5090`
- `CQL-CORL-HCMR-shapley20x128-v1-seed1-1M-5090`

## Concurrent Git publishing

Prepare changes in an isolated checkout based on the latest remote main. Run
CI on a clean tracked export, fetch again before publishing, and fast-forward
only. If another server publishes first, rebase this scoped addition on its
commit, repeat validation, and retry. Never force-push or reset either server's
working directory. Deploy an immutable archive of the published commit to a
new `.runtime/cql-5090-source-COMMIT` directory; training uses its own frozen
source snapshot. Datasets, runtime files and credentials stay ignored.
