# DT-EXP Environment Setup

This document describes a reproducible setup for running the DT and
delayed-reward DT experiments in this repository on another Linux server.

The project uses an older offline-RL stack. A successful `pip install` is
necessary, but it is not sufficient: MuJoCo system libraries, the MuJoCo 2.1
binary, D4RL dataset files, W&B configuration, and the active Conda
environment in `screen` must also be checked.

## Recommended Baseline

The working environment used for the experiments has the following important
versions:

| Component | Version |
| --- | --- |
| Python | 3.9 |
| PyTorch | 1.11.0+cu113 |
| NumPy | 1.23.1 |
| Gym | 0.23.0 |
| D4RL | `tinkoff-ai/d4rl` GitHub master archive |
| mujoco-py | 2.1.2.14 |
| MuJoCo Python package | 2.3.7 |
| h5py | 3.14.0 |
| W&B | 0.12.21 |

The requirements files pin the main dependencies, but they do not fully pin
all transitive dependencies. After creating a verified environment, export a
lock snapshot:

```bash
conda env export -n corl-ddr --no-builds > environment-corl-ddr.yml
conda run -n corl-ddr pip freeze > requirements-corl-ddr-lock.txt
```

## Option A: Docker

When Docker and the NVIDIA Container Toolkit are available, the repository
`Dockerfile` is the most reproducible option. It fixes the Ubuntu and CUDA
base image, installs the MuJoCo system dependencies, downloads MuJoCo 2.1,
installs the requirements, and verifies `mujoco_py` during the image build.

```bash
docker build -t dt-exp:corl-ddr .
docker run --gpus=all -it --rm \
  --name dt-exp \
  -v "$PWD:/workspace/DT-EXP" \
  dt-exp:corl-ddr
```

## Option B: Conda and pip

The README installation command is a good starting point. Use
`requirements.txt` for experiments; `requirements_dev.txt` additionally
installs development tools such as `pre-commit` and `ruff`.

### 1. Install system packages

On Ubuntu, install the build and OpenGL dependencies required by
`mujoco-py`:

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential patchelf git wget curl \
  libgl1-mesa-dev libgl1-mesa-glx libglew-dev \
  libosmesa6-dev xserver-xorg-dev
```

### 2. Install MuJoCo 2.1

`mujoco-py==2.1.2.14` expects the MuJoCo 2.1 distribution. Install it before
installing the Python requirements:

```bash
mkdir -p "$HOME/.mujoco"
cd /tmp
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
tar -xzf mujoco210-linux-x86_64.tar.gz -C "$HOME/.mujoco"

export MUJOCO_PY_MUJOCO_PATH="$HOME/.mujoco/mujoco210"
export LD_LIBRARY_PATH="$MUJOCO_PY_MUJOCO_PATH/bin:${LD_LIBRARY_PATH:-}"
```

Add the two `export` lines to the shell startup file used on the server if
they should persist across sessions.

### 3. Create the Conda environment

Use Python 3.9. The pinned PyTorch wheel and the old D4RL/Gym stack are not
intended for current Python versions such as Python 3.11 or 3.12.

```bash
conda create -n corl-ddr python=3.9 -y
conda activate corl-ddr

python -m pip install -r requirements/requirements.txt
```

For code development instead:

```bash
python -m pip install -r requirements/requirements_dev.txt
```

### 4. Verify the Python environment

Run these checks before starting any experiment:

```bash
python -m pip check

python - <<'PY'
import gym
import d4rl
import h5py
import mujoco_py
import pyrallis
import torch
import wandb

print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA devices:", torch.cuda.device_count())
print("imports: OK")
PY
```

Warnings about the optional D4RL `flow` and `CARLA` environments can be
ignored for the MuJoCo, Maze2D, AntMaze, and Adroit experiments.

## D4RL Datasets

D4RL stores datasets under:

```text
~/.d4rl/datasets/
```

The first call to `env.get_dataset()` may download a file automatically.
Do not start multiple seeds that are all downloading the same dataset for the
first time. Concurrent first downloads can leave a truncated HDF5 file that
still exists at the expected path, causing later jobs to fail.

Use one of these approaches:

1. Copy a verified cache from another server into `~/.d4rl/datasets/`.
2. Download datasets serially, using a temporary file for each download.
3. Before training, open every required file with `h5py` and load it through
   `gym.make(env_name).get_dataset(h5path=...)`.

A minimal validation example is:

```bash
python - <<'PY'
from pathlib import Path
import gym
import d4rl

datasets = [
    "pen-human-v1",
    "pen-cloned-v1",
    "pen-expert-v1",
    "door-human-v1",
    "door-cloned-v1",
    "door-expert-v1",
    "hammer-human-v1",
    "hammer-cloned-v1",
    "hammer-expert-v1",
    "relocate-human-v1",
    "relocate-cloned-v1",
    "relocate-expert-v1",
]

for name in datasets:
    env = gym.make(name)
    path = Path(env.dataset_filepath)
    data = env.get_dataset(h5path=str(path))
    assert data["observations"].shape[1:] == env.observation_space.shape
    assert data["actions"].shape[1:] == env.action_space.shape
    assert data["rewards"].shape[0] == data["observations"].shape[0]
    print("OK:", name, data["observations"].shape)
    env.close()
PY
```

For large datasets, this validation loads the data into memory. Run it
serially and ensure the server has enough RAM.

## W&B Configuration

The experiment runner reads a project-local file:

```text
.dt_runs/wandb.env
```

Create it inside this repository, not globally, and protect it from other
users:

```bash
mkdir -p .dt_runs
read -rsp "W&B API key: " WANDB_API_KEY
echo
printf 'WANDB_API_KEY=%s\n' "$WANDB_API_KEY" > .dt_runs/wandb.env
unset WANDB_API_KEY
chmod 600 .dt_runs/wandb.env
```

Do not commit `.dt_runs/wandb.env` or place the key in a README, shell
history, or shared environment file.

## Running in screen

Activate the environment inside the `screen` shell before running Python:

```bash
screen -S corl -dm bash
screen -S corl -X stuff $'cd /path/to/DT-EXP\nconda activate corl-ddr\n'
```

For example, to run all delayed-reward Adroit DT experiments on GPUs 4, 5,
and 6:

```bash
screen -S corl -X stuff $'python scripts/dt_experiments/adroit.py \
  --variants delayed \
  --seeds 0 1 2 \
  --gpus 4 5 6 \
  --keep-going \
  --wandb-mode online\n'
```

Inspect the window with:

```bash
screen -r corl
```

The runner assigns one process to each GPU by setting
`CUDA_VISIBLE_DEVICES`. The GPU number passed to the runner is the host GPU
number; inside each individual process, its assigned GPU is visible as device
0.

## Preflight Checklist

Before launching a long experiment:

- `conda activate corl-ddr` succeeds.
- `python -m pip check` reports no broken requirements.
- `import mujoco_py` succeeds.
- `torch.cuda.is_available()` is `True`.
- The required D4RL HDF5 files exist and pass D4RL validation.
- `.dt_runs/wandb.env` exists with mode `600` when online logging is needed.
- The `screen` shell has the correct project directory and active environment.
- No other process is downloading the same D4RL dataset.
