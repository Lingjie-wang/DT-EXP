"""Paired dense / episode-delayed reproduction of the released Decision Diffuser.

The upstream neural-network, loss, CDF normalization and sampler are unmodified.
This adapter replaces the obsolete cloud/renderer stack with local HDF5, audited
reward conversion, resumable training, seeded D4RL evaluation and local JSONL.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import pickle
import random
import signal
import sys
import time
import types
from pathlib import Path

import h5py
import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[2]
UPSTREAM_COMMIT = "01ce528c30b4733dc59aa6203e46ec165561158d"
DATA = "/labmount/users/202615385/.d4rl/datasets/halfcheetah_medium_replay-v2.hdf5"
UPSTREAM_FILES = [
    "models/temporal.py",
    "models/diffusion.py",
    "models/helpers.py",
    "datasets/normalization.py",
    "utils/progress.py",
]


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def append(path, value):
    with open(path, "a") as f:
        f.write(json.dumps(value, allow_nan=False) + "\n")
        f.flush()


def load_upstream(base):
    """Load official files without executing renderer / cloud package __init__.

    Only utils.Progress, utils.Silent and utils.to_np are used by these files.
    Progress classes are upstream; to_np is the same detach/cpu/numpy conversion.
    """
    base = Path(base)
    existing = sys.modules.get("diffuser")
    if existing is not None:
        if getattr(existing, "_repro_source", None) != str(base):
            raise RuntimeError("A different diffuser package is already loaded")
        return (
            sys.modules["diffuser.models.temporal"],
            sys.modules["diffuser.models.diffusion"],
            sys.modules["diffuser.datasets.normalization"],
        )
    for name, sub in [
        ("diffuser", ""),
        ("diffuser.models", "models"),
        ("diffuser.utils", "utils"),
        ("diffuser.datasets", "datasets"),
    ]:
        module = types.ModuleType(name)
        module.__path__ = [str(base / sub)]
        module._repro_source = str(base)
        sys.modules[name] = module
        if "." in name:
            parent, child = name.rsplit(".", 1)
            setattr(sys.modules[parent], child, module)

    def load(name, relative):
        spec = importlib.util.spec_from_file_location(name, base / relative)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        parent, child = name.rsplit(".", 1)
        setattr(sys.modules[parent], child, module)
        return module

    progress = load("diffuser.utils.progress", "utils/progress.py")
    utils = sys.modules["diffuser.utils"]
    utils.Progress, utils.Silent = progress.Progress, progress.Silent
    utils.to_np = lambda x: x.detach().cpu().numpy() if torch.is_tensor(x) else x
    normalizer = load("diffuser.datasets.normalization", "datasets/normalization.py")
    load("diffuser.models.helpers", "models/helpers.py")
    temporal = load("diffuser.models.temporal", "models/temporal.py")
    diffusion = load("diffuser.models.diffusion", "models/diffusion.py")
    return temporal, diffusion, normalizer


def reward_transform(rewards, mode):
    rewards = np.asarray(rewards, dtype=np.float64)
    if mode == "dense":
        return rewards.copy()
    if mode != "delayed":
        raise ValueError(mode)
    result = np.zeros_like(rewards)
    result[:, -1] = rewards.sum(axis=1)
    return result


def discounted_returns(rewards, discount):
    result = np.zeros_like(rewards, dtype=np.float64)
    running = np.zeros(len(rewards), dtype=np.float64)
    for t in range(rewards.shape[1] - 1, -1, -1):
        running = rewards[:, t] + discount * running
        result[:, t] = running
    return result


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all([v.cpu() for v in state["cuda"]])


def state_hash(model):
    h = hashlib.sha256()
    for key, value in model.state_dict().items():
        h.update(key.encode())
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def prepare(root):
    root.mkdir(parents=True, exist_ok=True)
    if (root / "protocol.json").exists():
        raise FileExistsError("Refusing to replace a frozen campaign")
    source = PROJECT / "third_party/decision-diffuser/code/diffuser"
    _, _, norm = load_upstream(source)
    with h5py.File(DATA, "r") as f:
        states, actions = f["observations"][:], f["actions"][:]
        rewards = f["rewards"][:].reshape(-1)
        terminals, timeouts = f["terminals"][:].astype(bool), f["timeouts"][:].astype(
            bool
        )
    ends = np.flatnonzero(terminals | timeouts) + 1
    lengths = np.diff(np.r_[0, ends])
    assert len(lengths) == 202 and np.all(lengths == 1000)
    assert ends[-1] == len(rewards) and not terminals.any()
    p = dict(
        method="Decision Diffuser",
        recipe="released-code-defaults-transferred-to-HCMR",
        upstream_commit=UPSTREAM_COMMIT,
        env_name="halfcheetah-medium-replay-v2",
        seed=0,
        arms=["dense", "delayed"],
        data_path=DATA,
        data_sha256=sha(DATA),
        horizon=100,
        n_diffusion_steps=200,
        dim=128,
        dim_mults=[1, 4, 8],
        inverse_hidden_dim=256,
        condition_dropout=0.25,
        guidance=1.2,
        discount=0.99,
        returns_scale=400.0,
        test_return=0.9,
        normalizer="CDFNormalizer",
        learning_rate=2e-4,
        batch_size=32,
        gradient_accumulate_every=2,
        total_updates=1000000,
        ema_decay=0.995,
        ema_every=10,
        ema_start=2000,
        log_every=100,
        save_every=10000,
        eval_updates=[100000, 500000, 1000000],
        eval_episodes=10,
        eval_seed_start=271828,
        eval_torch_seed=8675309,
        sample_seed=0,
        max_episode_steps=1000,
        valid_starts_per_episode=900,
        wandb_entity="2820402607-shandong-university",
        wandb_project="CORL-DDR",
        wandb_group="DecisionDiffuser-HCMR-dense-vs-delayed-seed0-20261002",
        upstream_files={name: sha(source / name) for name in UPSTREAM_FILES},
        changes=[
            "HDF5 adapter and local logging, no renderer",
            "Explicit independent paired shuffle stream and seeded evaluation",
            "Optimizer/RNG/sampler checkpointed; steps mean completed updates",
            "Dense versus episode-terminal reward timing only",
        ],
        limitations=[
            "Released example targets Hopper; "
            "no published HalfCheetah-specific scale supplied",
            "Paper: K=100, inverse hidden=512, context=20, 2M steps; "
            "released code: K=200, hidden=256, current state only, "
            "1M updates with accumulation 2",
            "Released returns_scale=400 and test_ret=.9 "
            "transferred unchanged; no tuning",
            "Delayed gamma=.99 attenuates early return labels; target is constant .9",
            "One training seed cannot establish reproducibility "
            "or match a five-seed paper mean",
        ],
    )
    rewards = rewards.reshape(202, 1000)
    normalizers = {
        "observations": norm.CDFNormalizer(states),
        "actions": norm.CDFNormalizer(actions),
    }
    normalized = {
        key: normalizers[key].normalize(value).astype(np.float32).reshape(202, 1000, -1)
        for key, value in [("observations", states), ("actions", actions)]
    }
    labels, stats = {}, {}
    for arm in p["arms"]:
        transformed = reward_transform(rewards, arm)
        np.testing.assert_allclose(
            transformed.sum(1), rewards.sum(1, dtype=np.float64), atol=1e-10
        )
        rtg = discounted_returns(transformed, p["discount"]) / p["returns_scale"]
        labels[arm] = rtg.astype(np.float32)
        values = rtg[:, :900]
        stats[arm] = {
            "condition_quantiles": dict(
                zip(
                    ["min", "q10", "q50", "q90", "q99", "max"],
                    np.quantile(values, [0, 0.1, 0.5, 0.9, 0.99, 1]).tolist(),
                )
            ),
            "episode_start_condition_quantiles": np.quantile(
                rtg[:, 0], [0, 0.5, 1]
            ).tolist(),
            "fraction_labels_below_test_return": float(
                np.mean(values < p["test_return"])
            ),
        }
    np.savez(root / "dataset.npz", **normalized, **labels)
    with open(root / "normalizers.pkl", "wb") as f:
        pickle.dump(normalizers, f)
    p["dataset_sha256"] = sha(root / "dataset.npz")
    p["normalizers_sha256"] = sha(root / "normalizers.pkl")
    write_json(
        root / "data_audit.json",
        dict(
            trajectories=202,
            lengths=lengths.tolist(),
            timeout_count=int(timeouts.sum()),
            terminal_count=int(terminals.sum()),
            training_windows=181800,
            reward_total_preserved=True,
            original_return_quantiles=np.quantile(
                rewards.sum(1, dtype=np.float64), [0, 0.5, 0.9, 1]
            ).tolist(),
            condition_statistics=stats,
            official_last_start_899_preserved=True,
        ),
    )
    import shutil

    frozen = root / "source"
    for name in UPSTREAM_FILES:
        destination = frozen / "diffuser" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / name, destination)
    shutil.copy2(__file__, frozen / "decision_diffuser_repro.py")
    p["entry_sha256"] = sha(frozen / "decision_diffuser_repro.py")
    write_json(root / "protocol.json", p)
    write_json(root / "status.json", {"status": "prepared", "arms": p["arms"]})
    print(json.dumps(stats, indent=2), flush=True)


def verify(root, p):
    assert sha(root / "dataset.npz") == p["dataset_sha256"]
    assert sha(root / "normalizers.pkl") == p["normalizers_sha256"]
    assert sha(root / "source/decision_diffuser_repro.py") == p["entry_sha256"]
    for name, expected in p["upstream_files"].items():
        assert sha(root / "source/diffuser" / name) == expected


def build_model(p, source):
    temporal, diffusion, _ = load_upstream(source)
    model = temporal.TemporalUnet(
        horizon=p["horizon"],
        transition_dim=17,
        cond_dim=17,
        dim=p["dim"],
        dim_mults=tuple(p["dim_mults"]),
        returns_condition=True,
        condition_dropout=p["condition_dropout"],
        calc_energy=False,
    )
    return diffusion.GaussianInvDynDiffusion(
        model,
        horizon=p["horizon"],
        observation_dim=17,
        action_dim=6,
        n_timesteps=p["n_diffusion_steps"],
        loss_type="l2",
        clip_denoised=True,
        predict_epsilon=True,
        hidden_dim=p["inverse_hidden_dim"],
        action_weight=10,
        loss_discount=1,
        loss_weights=None,
        returns_condition=True,
        condition_guidance_w=p["guidance"],
        ar_inv=False,
        train_only_inv=False,
    )


class WindowStream:
    """Uniform random-permutation epochs, including a short final microbatch."""

    def __init__(self, size, seed):
        self.size = size
        self.rng = np.random.default_rng(seed)
        self.order = self.rng.permutation(size)
        self.cursor = 0
        self.chain = "0" * 64

    def next(self, count):
        if self.cursor == self.size:
            self.order = self.rng.permutation(self.size)
            self.cursor = 0
        result = self.order[self.cursor : min(self.size, self.cursor + count)]
        self.cursor += len(result)
        self.chain = hashlib.sha256(
            bytes.fromhex(self.chain) + result.tobytes()
        ).hexdigest()
        return result

    def state(self):
        return dict(
            order=self.order.copy(),
            cursor=self.cursor,
            rng=self.rng.bit_generator.state,
            chain=self.chain,
        )

    def restore(self, state):
        self.order, self.cursor, self.chain = (
            state["order"],
            state["cursor"],
            state["chain"],
        )
        self.rng.bit_generator.state = state["rng"]


def batch(data, ids, arm, p, device):
    trajectories, starts = (
        ids // p["valid_starts_per_episode"],
        ids % p["valid_starts_per_episode"],
    )
    steps = starts[:, None] + np.arange(p["horizon"])[None]
    observations = data["observations"][trajectories[:, None], steps]
    actions = data["actions"][trajectories[:, None], steps]
    x = torch.as_tensor(np.concatenate([actions, observations], -1), device=device)
    cond = {0: x[:, 0, 6:].clone()}
    returns = torch.as_tensor(data[arm][trajectories, starts, None], device=device)
    return x, cond, returns


@torch.no_grad()
def update_ema(ema, model, zero_based_step, p):
    if zero_based_step % p["ema_every"]:
        return
    if zero_based_step < p["ema_start"]:
        ema.load_state_dict(model.state_dict())
    else:
        for old, current in zip(ema.parameters(), model.parameters()):
            old.copy_(old * p["ema_decay"] + current * (1 - p["ema_decay"]))


def evaluate(model, normalizers, p, device, completed, directory, smoke_steps=None):
    # Save/restore RNG even on errors: evaluation never changes the training stream.
    saved_rng, was_training = rng_state(), model.training
    environments = []
    started = time.monotonic()
    try:
        import d4rl  # noqa: F401 -- environment registration
        import gym

        seed_all(p["eval_torch_seed"])
        model.eval()
        count = p["eval_episodes"] if smoke_steps is None else 1
        environments = [gym.make(p["env_name"]) for _ in range(count)]
        observations = []
        for i, env in enumerate(environments):
            env.seed(p["eval_seed_start"] + i)
            env.action_space.seed(p["eval_seed_start"] + i)
            observations.append(env.reset())
        observations = np.asarray(observations)
        episode_returns, lengths, done = (
            np.zeros(count),
            np.zeros(count, dtype=int),
            np.zeros(count, dtype=bool),
        )
        target = torch.full((count, 1), p["test_return"], device=device)
        for t in range(p["max_episode_steps"] if smoke_steps is None else smoke_steps):
            conditions = {
                0: torch.as_tensor(
                    normalizers["observations"].normalize(observations),
                    dtype=torch.float32,
                    device=device,
                )
            }
            with torch.no_grad():
                samples = model.conditional_sample(
                    conditions, returns=target, verbose=False
                )
                torch.testing.assert_close(samples[:, 0], conditions[0], rtol=0, atol=0)
                paired = torch.cat([samples[:, 0], samples[:, 1]], -1)
                actions = normalizers["actions"].unnormalize(
                    model.inv_model(paired).cpu().numpy()
                )
            assert np.isfinite(actions).all()
            for i, env in enumerate(environments):
                if done[i]:
                    continue
                assert np.all(actions[i] <= env.action_space.high + 1e-5)
                assert np.all(actions[i] >= env.action_space.low - 1e-5)
                obs, reward, finished, _ = env.step(actions[i])
                observations[i] = obs
                episode_returns[i] += reward
                lengths[i] += 1
                done[i] = finished
            if (t + 1) % 100 == 0:
                print(
                    f"EVAL update={completed} step={t + 1} "
                    f"elapsed={time.monotonic() - started:.1f}s",
                    flush=True,
                )
            if done.all():
                break
        if smoke_steps is None:
            assert done.all(), "Evaluation did not reach all episode boundaries"
        rows = [
            dict(
                episode_seed=p["eval_seed_start"] + i,
                raw_return=float(episode_returns[i]),
                normalized_score=float(
                    environments[i].get_normalized_score(episode_returns[i]) * 100
                ),
                length=int(lengths[i]),
            )
            for i in range(count)
        ]
        result = dict(
            completed_updates=completed,
            episodes=rows,
            test_return=p["test_return"],
            mean_score=float(np.mean([row["normalized_score"] for row in rows])),
            std_score=float(np.std([row["normalized_score"] for row in rows])),
            elapsed_seconds=time.monotonic() - started,
            smoke=smoke_steps is not None,
        )
        write_json(directory / f"eval_{completed:07d}.json", result)
        return result
    finally:
        for env in environments:
            env.close()
        model.train(was_training)
        restore_rng(saved_rng)


def save_checkpoint(path, model, ema, optimizer, stream, completed, elapsed, p):
    tmp = path.with_suffix(".tmp")
    torch.save(
        dict(
            model=model.state_dict(),
            ema=ema.state_dict(),
            optimizer=optimizer.state_dict(),
            stream=stream.state(),
            rng=rng_state(),
            completed_updates=completed,
            elapsed_seconds=elapsed,
            protocol=p,
        ),
        tmp,
    )
    tmp.replace(path)


def smoke(root, device):
    p = read_json(root / "protocol.json")
    verify(root, p)
    directory = root / ("gpu_smoke" if device.startswith("cuda") else "cpu_smoke")
    directory.mkdir(exist_ok=True)
    data = dict(np.load(root / "dataset.npz"))
    seed_all(p["seed"])
    model = build_model(p, root / "source/diffuser").to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=p["learning_rate"])
    size = p["batch_size"] if device.startswith("cuda") else 1
    stream = WindowStream(181800, p["sample_seed"])
    ids = stream.next(size)
    dense = batch(data, ids, "dense", p, device)
    delayed = batch(data, ids, "delayed", p, device)
    assert torch.equal(dense[0], delayed[0]) and torch.equal(dense[1][0], delayed[1][0])
    before = state_hash(model)
    losses, durations = [], []
    for step in range(3):
        start = time.monotonic()
        optimizer.zero_grad()
        for _ in range(p["gradient_accumulate_every"]):
            loss, _ = model.loss(*batch(data, stream.next(size), "dense", p, device))
            assert torch.isfinite(loss)
            (loss / p["gradient_accumulate_every"]).backward()
        optimizer.step()
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        durations.append(time.monotonic() - start)
        losses.append(float(loss.detach()))
        print("SMOKE", step + 1, losses[-1], durations[-1], flush=True)
    assert before != state_hash(model)
    with open(root / "normalizers.pkl", "rb") as f:
        normalizers = pickle.load(f)
    before_eval = state_hash(model)
    random_before = rng_state()
    # Full 200-step sampler even in smoke; CPU checks one action, GPU checks three.
    first = evaluate(
        model,
        normalizers,
        p,
        device,
        0,
        directory,
        3 if device.startswith("cuda") else 1,
    )
    second = evaluate(
        model,
        normalizers,
        p,
        device,
        1,
        directory,
        3 if device.startswith("cuda") else 1,
    )
    assert first["episodes"] == second["episodes"]
    assert before_eval == state_hash(model)
    assert torch.equal(random_before["torch"], torch.get_rng_state())
    if random_before["cuda"] is not None:
        assert all(
            torch.equal(a, b)
            for a, b in zip(random_before["cuda"], torch.cuda.get_rng_state_all())
        )
    write_json(
        directory / "audit.json",
        dict(
            passed=True,
            batch_size=size,
            full_official_model=True,
            full_diffusion_sampler=True,
            paired_state_action_batches=True,
            repeated_eval_exact=True,
            eval_rng_preserved=True,
            eval_weights_preserved=True,
            losses=losses,
            update_seconds=durations,
            estimated_training_hours=float(
                np.mean(durations[1:]) * p["total_updates"] / 3600
            )
            if size == 32
            else None,
            cuda_peak_mib=torch.cuda.max_memory_allocated() / 2**20
            if device.startswith("cuda")
            else None,
        ),
    )


def train(root, arm, device):
    p = read_json(root / "protocol.json")
    verify(root, p)
    directory = root / arm
    directory.mkdir(exist_ok=True)
    # An exclusive lock prevents accidentally running the same arm twice.
    import fcntl

    lock = open(directory / "train.lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    seed_all(p["seed"])
    torch.backends.cudnn.benchmark = True  # released training script
    model = build_model(p, root / "source/diffuser").to(device)
    initial_hash = state_hash(model)
    ema = copy.deepcopy(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=p["learning_rate"])
    data = dict(np.load(root / "dataset.npz"))
    with open(root / "normalizers.pkl", "rb") as f:
        normalizers = pickle.load(f)
    stream = WindowStream(181800, p["sample_seed"])
    completed, previous_seconds = 0, 0.0
    checkpoint = directory / "latest.pt"
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location="cpu")
        assert state["protocol"] == p
        model.load_state_dict(state["model"])
        ema.load_state_dict(state["ema"])
        optimizer.load_state_dict(state["optimizer"])
        stream.restore(state["stream"])
        completed, previous_seconds = (
            state["completed_updates"],
            state["elapsed_seconds"],
        )
        restore_rng(state["rng"])
        del state
        print(f"RESUME {arm} completed_updates={completed}", flush=True)
    else:
        write_json(
            directory / "initialization.json",
            dict(
                initial_model_sha256=initial_hash,
                seed=p["seed"],
                parameters=sum(t.numel() for t in model.parameters()),
                torch_version=torch.__version__,
                numpy_version=np.__version__,
                gpu=torch.cuda.get_device_name() if device.startswith("cuda") else "cpu",
            ),
        )
    stop = []
    signal.signal(signal.SIGUSR1, lambda *_: stop.append("walltime_warning"))
    signal.signal(signal.SIGTERM, lambda *_: stop.append("terminated"))
    start = time.monotonic()

    def elapsed():
        return previous_seconds + time.monotonic() - start

    def checkpoint_now():
        save_checkpoint(
            checkpoint, model, ema, optimizer, stream, completed, elapsed(), p
        )

    def eval_if_due():
        if (
            completed in p["eval_updates"]
            and not (directory / f"eval_{completed:07d}.json").exists()
        ):
            ema.eval()
            result = evaluate(ema, normalizers, p, device, completed, directory)
            append(directory / "evaluations.jsonl", result)

    write_json(
        directory / "status.json",
        dict(
            status="running",
            completed_updates=completed,
            job_id=os.environ.get("SLURM_JOB_ID"),
        ),
    )
    eval_if_due()
    while completed < p["total_updates"] and not stop:
        optimizer.zero_grad()
        losses, diffusion_losses = [], []
        for _ in range(p["gradient_accumulate_every"]):
            ids = stream.next(p["batch_size"])
            loss, info = model.loss(*batch(data, ids, arm, p, device))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at {completed}")
            (loss / p["gradient_accumulate_every"]).backward()
            losses.append(float(loss.detach()))
            diffusion_losses.append(float(info["a0_loss"].detach()))
        optimizer.step()
        update_ema(ema, model, completed, p)
        completed += 1
        if completed == 1 or completed % p["log_every"] == 0:
            row = dict(
                completed_updates=completed,
                loss=float(np.mean(losses)),
                diffusion_loss=float(np.mean(diffusion_losses)),
                elapsed_seconds=elapsed(),
                sample_stream_sha256=stream.chain,
                cuda_peak_mib=torch.cuda.max_memory_allocated() / 2**20
                if device.startswith("cuda")
                else None,
            )
            append(directory / "metrics.jsonl", row)
            write_json(
                directory / "status.json",
                dict(
                    status="running",
                    completed_updates=completed,
                    job_id=os.environ.get("SLURM_JOB_ID"),
                    elapsed_seconds=elapsed(),
                ),
            )
            print(json.dumps(row), flush=True)
        if completed % p["save_every"] == 0:
            checkpoint_now()
        if completed in p["eval_updates"]:
            # Separate immutable model/EMA weights for independent later evaluation.
            torch.save(
                dict(
                    model=model.state_dict(),
                    ema=ema.state_dict(),
                    completed_updates=completed,
                    protocol=p,
                ),
                directory / f"step{completed:07d}.pt",
            )
            eval_if_due()
    checkpoint_now()
    finished = completed == p["total_updates"]
    assert sha(p["data_path"]) == p["data_sha256"]
    write_json(
        directory / "status.json",
        dict(
            status="completed" if finished else "interrupted",
            completed_updates=completed,
            elapsed_seconds=elapsed(),
            stop_reason=stop,
            sample_stream_sha256=stream.chain,
            original_dataset_unchanged=True,
            initial_model_sha256=initial_hash,
            job_id=os.environ.get("SLURM_JOB_ID"),
        ),
    )
    if not finished:
        raise SystemExit(75)


def summarize(root):
    p = read_json(root / "protocol.json")
    result = {"recipe": p["recipe"], "arms": {}}
    for arm in p["arms"]:
        directory = root / arm
        result["arms"][arm] = dict(
            status=read_json(directory / "status.json")
            if (directory / "status.json").exists()
            else {"status": "pending"},
            evaluations=[read_json(x) for x in sorted(directory.glob("eval_*.json"))],
        )
    inits = [root / arm / "initialization.json" for arm in p["arms"]]
    if all(x.exists() for x in inits):
        result["paired_initialization"] = (
            len({read_json(x)["initial_model_sha256"] for x in inits}) == 1
        )
        assert result["paired_initialization"]
    write_json(root / "summary.json", result)
    lines = [
        "# Decision Diffuser 原始 / 整局延迟奖励复现",
        "",
        "官方发布代码默认配置迁移到 HalfCheetah-medium-replay-v2，"
        "seed 0；不是论文五种子精确复现。",
        "",
        "两组仅奖励时序不同，终局总回报保持不变。γ=0.99、scale=400、固定条件=0.9，未调参。",
        "",
        "| 奖励设置 | 已完成更新 | 状态 | 最后评测 normalized score |",
        "|---|---:|---|---:|",
    ]
    for arm, value in result["arms"].items():
        last = value["evaluations"][-1] if value["evaluations"] else None
        lines.append(
            f"| {arm} | {value['status'].get('completed_updates', 0)} "
            f"| {value['status']['status']} "
            f"| {last['mean_score'] if last else '尚无结果'} |"
        )
    lines += [
        "",
        "发布代码预算：1,000,000 次优化更新，每次累积两个 batch 32；"
        "并非声称等价于论文 2M 次优化。",
        "论文 HCMR 原始奖励 DD 报告 39.3±4.1（5 seeds，标准误），仅供参考。",
        "延迟条件早期被折扣显著压缩；若表现差，不能仅归因为生成模型能力。",
        "此前普通 DT 仅作为历史背景；训练预算、模型、条件和评测均不构成严格配对。",
    ]
    (root / "report_zh.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["prepare", "smoke", "train", "summarize"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--arm", choices=["dense", "delayed"])
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.mode == "prepare":
        prepare(args.root.resolve())
    elif args.mode == "smoke":
        smoke(args.root.resolve(), args.device)
    elif args.mode == "train":
        if not args.arm:
            parser.error("--arm is required for train")
        train(args.root.resolve(), args.arm, args.device)
    else:
        summarize(args.root.resolve())


if __name__ == "__main__":
    main()
