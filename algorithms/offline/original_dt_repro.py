"""Instrument the pinned original DT, retaining its sampler and optimizer update.

The recipe follows QDT (Yamagata et al., 2023), Appendix C.3. Runtime versions,
evaluation seeds and instrumentation are recorded; this is not a recovered copy
of the authors' historical five-seed runs.
"""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
import pickle
import random
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import h5py
import numpy as np
import torch

UPSTREAM_COMMIT = "e2d82e68f330c00f763507b3b01d774740bee53f"
ENV_NAME = "halfcheetah-medium-replay-v2"


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def append(path, value):
    with open(path, "a") as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_hash(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def rng_state():
    return dict(
        python=random.getstate(),
        numpy=np.random.get_state(),
        torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    )


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@contextmanager
def isolated_evaluation(seed):
    state = rng_state()
    seed_all(seed)
    try:
        yield
    finally:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        if state["cuda"]:
            torch.cuda.set_rng_state_all(state["cuda"])


def audit_dataset(pickle_path, hdf5_path):
    with open(pickle_path, "rb") as stream:
        trajectories = pickle.load(stream)
    with h5py.File(hdf5_path, "r") as data:
        boundaries = np.flatnonzero(data["terminals"][:] | data["timeouts"][:]) + 1
        lengths = np.diff(np.r_[0, boundaries])
        np.testing.assert_array_equal(lengths, [len(t["rewards"]) for t in trajectories])
        assert boundaries[-1] == len(data["rewards"])
        for key in ("observations", "actions", "rewards", "terminals"):
            np.testing.assert_array_equal(
                np.concatenate([t[key] for t in trajectories]), data[key][:]
            )
    return dict(
        trajectories=len(trajectories),
        transitions=int(lengths.sum()),
        lengths=np.unique(lengths).tolist(),
        pickle_sha256=file_hash(pickle_path),
        hdf5_sha256=file_hash(hdf5_path),
        arrays_and_boundaries_equal=True,
    )


def prepare(args):
    root, source = args.root.resolve(), args.source.resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain"], text=True
    )
    if revision != UPSTREAM_COMMIT or dirty:
        raise RuntimeError("Expected a clean checkout of the pinned original DT")
    audit = audit_dataset(args.dataset, args.hdf5)
    assert audit["trajectories"] == 202 and audit["transitions"] == 202000
    root.mkdir(parents=True, exist_ok=False)
    (root / "source").mkdir()
    shutil.copytree(
        source / "gym",
        root / "source" / "upstream",
        ignore=shutil.ignore_patterns("data", "__pycache__", "wandb"),
    )
    shutil.copy2(__file__, root / "source" / "original_dt_repro.py")
    scripts = Path(__file__).resolve().parents[2] / "scripts" / "original_dt"
    for name in ("run.sbatch", "sync_results.py"):
        shutil.copy2(scripts / name, root / "source" / name)
    (root / "data").mkdir()
    shutil.copy2(args.dataset, root / "data" / f"{ENV_NAME}.pkl")
    for arm in ("dense", "delayed"):
        (root / arm).mkdir()
        (root / arm / "data").symlink_to(root / "data", target_is_directory=True)
    protocol = dict(
        method="OriginalDT",
        recipe="QDT Appendix C.3; original DT source/defaults",
        upstream_commit=revision,
        env_name=ENV_NAME,
        eval_env="HalfCheetah-v3",
        seed=args.seed,
        batch_size=64,
        context_length=20,
        hidden_size=128,
        num_layers=3,
        num_heads=1,
        activation="relu",
        dropout=0.1,
        learning_rate=1e-4,
        optimizer="AdamW",
        weight_decay=1e-4,
        warmup_steps=10000,
        grad_clip=0.25,
        reward_scale=0.001,
        total_updates=100000,
        eval_every=10000,
        eval_episodes=100,
        eval_seed_start=42,
        target_returns=[12000, 6000],
        sampling="upstream length-weighted trajectories, uniform starts; all data",
        loss="upstream action MSE over valid tokens only",
        pct_traj=1.0,
        main_result="final 100k checkpoint; report both RTGs, no best selection",
        wandb_entity="2820402607-shandong-university",
        wandb_project="CORL-DDR",
        wandb_group=root.name,
        dataset=audit,
        limitations="one training seed; modern compatible runtime; historical seeds "
        "and per-run RTG selection are not available",
        source_sha256={
            str(p.relative_to(root)): file_hash(p)
            for p in sorted((root / "source").rglob("*"))
            if p.is_file()
        },
    )
    write(root / "protocol.json", protocol)
    print(json.dumps(dict(root=str(root), **audit)), flush=True)


def load_upstream(source):
    source = Path(source).resolve()
    sys.path.insert(0, str(source))
    spec = importlib.util.spec_from_file_location(
        "original_dt_experiment", source / "experiment.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_trainer(upstream, protocol, arm, work, device, episodes):
    original = upstream.SequenceTrainer

    class RecordedTrainer(original):
        def __init__(self, **kwargs):
            closure = inspect.getclosurevars(kwargs["get_batch"]).nonlocals
            self.mean, self.std = closure["state_mean"], closure["state_std"]
            self.env = inspect.getclosurevars(kwargs["eval_fns"][0]).nonlocals["env"]
            self.completed_updates = 0
            self.losses = []
            self.sample_digest = hashlib.sha256()
            self.last_sample_hash = None
            self.started = time.monotonic()
            trajectories = closure["trajectories"]
            with open(work / "data" / f"{ENV_NAME}.pkl", "rb") as stream:
                originals = pickle.load(stream)
            for trajectory, original_trajectory in zip(trajectories, originals):
                reward = trajectory["rewards"]
                if arm == "delayed":
                    np.testing.assert_array_equal(reward[:-1], 0)
                    assert reward[-1] == original_trajectory["rewards"].sum()
                else:
                    np.testing.assert_array_equal(reward, original_trajectory["rewards"])
            write(
                work / "reward_audit.json",
                dict(
                    passed=True,
                    arm=arm,
                    trajectories=len(trajectories),
                    nonzero_rewards=sum(
                        int(np.count_nonzero(t["rewards"])) for t in trajectories
                    ),
                ),
            )
            batch_fn = kwargs["get_batch"]

            def recorded_batch(size):
                batch = batch_fn(size)
                # Sampling audit excludes reward/RTG, which legitimately differ.
                if self.completed_updates < 16:
                    for index in (0, 1, 5, 6):
                        self.sample_digest.update(
                            batch[index].detach().cpu().numpy().tobytes()
                        )
                    self.last_sample_hash = self.sample_digest.hexdigest()
                    append(
                        work / "sampling_audit.jsonl",
                        dict(
                            batch=self.completed_updates + 1,
                            sha256=self.last_sample_hash,
                        ),
                    )
                return batch

            kwargs["get_batch"] = recorded_batch
            kwargs["eval_fns"] = [self.evaluate]
            super().__init__(**kwargs)
            self.initial_hash = state_hash(self.model.state_dict())
            write(work / "initialization.json", dict(model_sha256=self.initial_hash))
            self.save()

        def save(self):
            checkpoint = dict(
                completed_updates=self.completed_updates,
                model=self.model.state_dict(),
                optimizer=self.optimizer.state_dict(),
                scheduler=self.scheduler.state_dict(),
                rng=rng_state(),
                state_mean=self.mean,
                state_std=self.std,
                protocol=protocol,
                reward_mode=arm,
            )
            path = work / f"checkpoint_{self.completed_updates:06d}.pt"
            temporary = path.with_suffix(".tmp")
            torch.save(checkpoint, temporary)
            temporary.replace(path)

        def train_step(self):
            loss = super().train_step()
            if not np.isfinite(loss):
                raise FloatingPointError("Non-finite original DT loss")
            self.completed_updates += 1
            self.losses.append(loss)
            if self.completed_updates % 100 == 0:
                self.log_training()
            return loss

        def log_training(self):
            if not self.losses:
                return
            row = dict(
                completed_updates=self.completed_updates,
                loss=float(np.mean(self.losses)),
                learning_rate=self.optimizer.param_groups[0]["lr"],
                elapsed_seconds=time.monotonic() - self.started,
            )
            append(work / "metrics.jsonl", row)
            write(work / "status.json", dict(status="training", **row))
            self.losses = []
            print(json.dumps(row), flush=True)

        def evaluate(self, model):
            from d4rl.infos import REF_MAX_SCORE, REF_MIN_SCORE

            results = []
            before = state_hash(model.state_dict())
            write(
                work / "status.json",
                dict(
                    status="evaluating",
                    completed_updates=self.completed_updates,
                ),
            )
            for target in protocol["target_returns"]:
                records = []
                with isolated_evaluation(protocol["eval_seed_start"]):
                    for index in range(episodes):
                        seed = protocol["eval_seed_start"] + index
                        self.env.seed(seed)
                        with torch.no_grad():
                            value, length = upstream.evaluate_episode_rtg(
                                self.env,
                                model.state_dim,
                                model.act_dim,
                                model,
                                max_ep_len=1000,
                                scale=1000.0,
                                target_return=target / 1000.0,
                                mode="normal" if arm == "dense" else "delayed",
                                state_mean=self.mean,
                                state_std=self.std,
                                device=device,
                            )
                        score = (
                            100
                            * (value - REF_MIN_SCORE[ENV_NAME])
                            / (REF_MAX_SCORE[ENV_NAME] - REF_MIN_SCORE[ENV_NAME])
                        )
                        records.append(
                            dict(
                                episode_seed=seed,
                                raw_return=float(value),
                                normalized_score=float(score),
                                length=int(length),
                            )
                        )
                scores = [r["normalized_score"] for r in records]
                results.append(
                    dict(
                        target_return=target,
                        mean_score=float(np.mean(scores)),
                        std_score=float(np.std(scores)),
                        episodes=records,
                    )
                )
            assert state_hash(model.state_dict()) == before
            write(
                work / f"eval_{self.completed_updates:06d}.json",
                dict(
                    completed_updates=self.completed_updates,
                    targets=results,
                    weights_unchanged=True,
                ),
            )
            return {
                f"target_{r['target_return']}_normalized_score": r["mean_score"]
                for r in results
            }

        def train_iteration(self, *args, **kwargs):
            result = super().train_iteration(*args, **kwargs)
            self.log_training()
            self.save()
            return result

    return RecordedTrainer


def train(args):
    root = args.root.resolve()
    protocol = read(root / "protocol.json")
    for name, expected in protocol["source_sha256"].items():
        assert file_hash(root / name) == expected, f"Changed source: {name}"
    assert (
        file_hash(root / "data" / f"{ENV_NAME}.pkl")
        == protocol["dataset"]["pickle_sha256"]
    )
    work = root / ("smoke" if args.smoke else "") / args.arm
    work.mkdir(parents=True, exist_ok=True)
    if not (work / "data").exists():
        (work / "data").symlink_to(root / "data", target_is_directory=True)
    if (work / "status.json").exists():
        raise RuntimeError("Refusing to overwrite an existing run")
    write(work / "status.json", dict(status="starting", completed_updates=0))
    try:
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        upstream = load_upstream(root / "source" / "upstream")
        # Import simulator/reference registration before seeding training. Lazy
        # imports during the first evaluation must not consume the training RNG.
        importlib.import_module("d4rl.infos")
        updates = 2 if args.smoke else protocol["total_updates"]
        interval = 2 if args.smoke else protocol["eval_every"]
        episodes = 1 if args.smoke else protocol["eval_episodes"]
        variant = dict(
            env="halfcheetah",
            dataset="medium-replay",
            model_type="dt",
            mode="normal" if args.arm == "dense" else "delayed",
            K=20,
            pct_traj=1.0,
            batch_size=64,
            embed_dim=128,
            n_layer=3,
            n_head=1,
            activation_function="relu",
            dropout=0.1,
            learning_rate=1e-4,
            weight_decay=1e-4,
            warmup_steps=10000,
            num_eval_episodes=episodes,
            max_iters=updates // interval,
            num_steps_per_iter=interval,
            device=args.device,
            log_to_wandb=False,
        )
        write(
            work / "runtime.json",
            dict(
                variant=variant,
                python=sys.version,
                slurm_job_id=os.getenv("SLURM_JOB_ID"),
                gpu=torch.cuda.get_device_name(0)
                if args.device.startswith("cuda")
                else None,
                versions={
                    name: importlib.metadata.version(name)
                    for name in (
                        "torch",
                        "numpy",
                        "transformers",
                        "gym",
                        "mujoco-py",
                        "d4rl",
                    )
                },
            ),
        )
        upstream.SequenceTrainer = make_trainer(
            upstream,
            protocol,
            args.arm,
            work,
            args.device,
            episodes,
        )
        seed_all(protocol["seed"])
        old_directory = Path.cwd()
        try:
            os.chdir(work)
            upstream.experiment("OriginalDT", variant)
        finally:
            os.chdir(old_directory)
        write(work / "status.json", dict(status="completed", completed_updates=updates))
    except BaseException as error:
        previous = read(work / "status.json")
        write(
            work / "status.json",
            dict(
                status="failed",
                completed_updates=previous.get("completed_updates", 0),
                error=repr(error),
            ),
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--root", type=Path, required=True)
    prepare_parser.add_argument("--source", type=Path, required=True)
    prepare_parser.add_argument("--dataset", type=Path, required=True)
    prepare_parser.add_argument("--hdf5", type=Path, required=True)
    prepare_parser.add_argument("--seed", type=int, default=0)
    train_parser = commands.add_parser("train")
    train_parser.add_argument("--root", type=Path, required=True)
    train_parser.add_argument("--arm", choices=("dense", "delayed"), required=True)
    train_parser.add_argument("--device", default="cuda:0")
    train_parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
