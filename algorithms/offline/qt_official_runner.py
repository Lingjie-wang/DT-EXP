"""Run pinned upstream QT, preserving its training implementation verbatim.

The user explicitly chose the uncorrected upstream delayed-reward implementation.
This adapter adds logging, snapshots and separately named evaluation protocols.
It is not a reproduction claim for the paper's delayed-reward table.
"""

import argparse
import ast
import hashlib
import importlib.metadata
import inspect
import json
import os
import pickle
import random
import subprocess
import sys
import time
import types
from contextlib import contextmanager
from pathlib import Path

import gym
import numpy as np
import torch
import wandb

UPSTREAM_COMMIT = "cb9e1a4873449b3467f6bf5586e010180d90614c"
ENV_NAME = "halfcheetah-medium-replay-v2"
METRICS = ("bc_loss", "ql_loss", "actor_loss", "critic_loss", "target_q_mean")


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def append_json(path, value):
    with open(path, "a") as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")


def rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


@contextmanager
def isolated_evaluation(seed):
    saved = rng_state()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        yield
    finally:
        random.setstate(saved["python"])
        np.random.set_state(saved["numpy"])
        torch.set_rng_state(saved["torch"])
        if saved["cuda"]:
            torch.cuda.set_rng_state_all(saved["cuda"])


class TerminalReward(gym.Wrapper):
    """Expose only the episode-total reward, including at a time-limit boundary."""

    def reset(self, **kwargs):
        self.episode_return = 0.0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.episode_return += float(reward)
        return obs, self.episode_return if done else 0.0, done, info


def load_upstream(source):
    source = Path(source).resolve()
    commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(source), "diff", "HEAD", "--"], text=True
    )
    if commit != UPSTREAM_COMMIT or dirty:
        raise RuntimeError("QT must be the clean, pinned upstream checkout")
    sys.path.insert(0, str(source))
    # Both are unused imports in the HalfCheetah execution path. Do not install
    # unrelated robotics or TensorBoard stacks or edit any upstream source file.
    tree = ast.parse((source / "experiment.py").read_text())
    removed = []
    retained = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module in (
            "mjrl.utils.gym_env", "torch.utils.tensorboard"
        ):
            removed.append(node.module)
        else:
            retained.append(node)
    if sorted(removed) != ["mjrl.utils.gym_env", "torch.utils.tensorboard"]:
        raise RuntimeError("Unexpected upstream entry-point imports")
    tree.body = retained
    module = types.ModuleType("qt_official_experiment")
    module.__file__ = str(source / "experiment.py")
    exec(compile(tree, module.__file__, "exec"), module.__dict__)
    manifest = {
        str(path.relative_to(source)): file_hash(path)
        for path in sorted(source.rglob("*.py"))
    }
    return module, manifest


@torch.no_grad()
def fixed_rtg_rollout(env, model, state_mean, state_std, target, device):
    """Single forward action, fixed RTG, no critic selection, terminal-only reward.

    This is an additional actor-only evaluation, not upstream QT inference.
    Padding and normalization match upstream get_action.
    """
    k = model.max_length
    state = env.reset()
    states, actions = [], []
    for step in range(1000):
        states.append((state - state_mean) / state_std)
        actions.append(np.zeros(model.act_dim, dtype=np.float32))
        n = min(len(states), k)
        s = np.zeros((1, k, model.state_dim), dtype=np.float32)
        a = np.zeros((1, k, model.act_dim), dtype=np.float32)
        s[0, -n:] = states[-n:]
        a[0, -n:] = actions[-n:]
        r = np.zeros((1, k, 1), dtype=np.float32)
        rtg = np.zeros((1, k, 1), dtype=np.float32)
        rtg[0, -n:, 0] = target / 1000.0
        times = np.zeros((1, k), dtype=np.int64)
        times[0, -n:] = np.arange(step - n + 1, step + 1)
        mask = np.zeros((1, k), dtype=np.int64)
        mask[0, -n:] = 1
        tensors = [
            torch.as_tensor(x, device=device) for x in (s, a, r, rtg, times, mask)
        ]
        _, prediction, _ = model(
            tensors[0], tensors[1], tensors[2], None, tensors[3], tensors[4],
            attention_mask=tensors[5],
        )
        action = prediction[0, -1].cpu().numpy()
        actions[-1] = action
        state, _, done, _ = env.step(action)
        if done:
            return env.episode_return, step + 1
    raise RuntimeError("HalfCheetah must terminate at the 1000-step time limit")


class ScalarRecorder:
    def __init__(self):
        self.values = {}

    def add_scalar(self, name, value, step):
        self.values[name] = float(value)


def make_instrumented_trainer(upstream, args, output, run):
    original_trainer = upstream.Trainer

    class InstrumentedTrainer(original_trainer):
        def __init__(self, **kwargs):
            closure = inspect.getclosurevars(kwargs["eval_fns"][0]).nonlocals
            self.eval_env = closure["env"]
            self.strict_env = TerminalReward(self.eval_env)
            self.state_mean = closure["state_mean"]
            self.state_std = closure["state_std"]
            original_batch = kwargs["get_batch"]
            self.pre_reward_count = 0
            self.post_reward_count = 0
            self.samples_seen = 0

            def observed_batch(batch_size):
                batch = original_batch(batch_size)
                self.last_rewards = batch[2]
                self.pre_reward_count += int(torch.count_nonzero(batch[2]).item())
                self.samples_seen += batch_size
                return batch

            kwargs["get_batch"] = observed_batch
            kwargs["eval_fns"] = [self.evaluate]
            super().__init__(**kwargs)
            self.recorder = ScalarRecorder()
            self.window = {key: [] for key in METRICS}
            self.evaluations = []
            self.best = {}
            self.started = time.time()
            self.save_full(0)

        def save_full(self, step):
            state = {
                "completed_updates": step,
                "upstream_commit": UPSTREAM_COMMIT,
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "ema_model": self.ema_model.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "actor_scheduler": self.actor_lr_scheduler.state_dict(),
                "critic_scheduler": self.critic_lr_scheduler.state_dict(),
                "rng": rng_state(),
                "state_mean": self.state_mean,
                "state_std": self.state_std,
                "adapter_arguments": vars(args),
                "eta": self.eta,
                "eta2": self.eta2,
            }
            torch.save(state, output / f"checkpoint_{step}.pt")

        def train_step(self, log_writer=None, loss_metric=None):
            if loss_metric is None:
                loss_metric = {key: [] for key in METRICS}
            result = super().train_step(self.recorder, loss_metric)
            self.post_reward_count += int(torch.count_nonzero(self.last_rewards).item())
            for key in METRICS:
                value = float(result[key][-1])
                if not np.isfinite(value):
                    raise FloatingPointError(f"Non-finite {key} at {self.step}")
                self.window[key].append(value)
            if self.step % args.log_every == 0:
                record = {
                    "update_step": self.step,
                    **{f"train/{k}": float(np.mean(v)) for k, v in self.window.items()},
                    "train/actor_grad_norm": self.recorder.values["Actor Grad Norm"],
                    "train/critic_grad_norm": self.recorder.values["Critic Grad Norm"],
                    "train/learning_rate": self.actor_optimizer.param_groups[0]["lr"],
                    "audit/nonzero_rewards_before_cumulative": self.pre_reward_count,
                    "audit/nonzero_rewards_after_cumulative": self.post_reward_count,
                    "train/sequences_seen": self.samples_seen,
                    "time/elapsed_seconds": time.time() - self.started,
                }
                append_json(output / "metrics.jsonl", record)
                run.log(record)
                self.window = {key: [] for key in METRICS}
            return result

        def train_iteration(self, *positional, **kwargs):
            result = super().train_iteration(*positional, **kwargs)
            if self.step in (10000, 20000, 50000, 75000, 100000, args.updates):
                self.save_full(self.step)
            return result

        def evaluate(self, model, critic):
            if self.step % args.eval_every:
                return {}
            model.eval()
            critic.eval()
            record = {"update_step": self.step}
            protocols = ["qt_strict_delayed", "actor_12000", "actor_6000"]
            # Upstream feedback includes dense rewards even in mode='delayed'.
            # Keep this only as an explicitly separate diagnostic, not the main score.
            if self.step == args.updates or args.smoke:
                protocols.append("qt_upstream_feedback")
            for protocol in protocols:
                returns, lengths = [], []
                count = args.eval_episodes
                if protocol == "qt_upstream_feedback":
                    count = args.feedback_eval_episodes
                with isolated_evaluation(args.eval_seed):
                    self.eval_env.seed(args.eval_seed)
                    for _ in range(count):
                        if protocol.startswith("actor_"):
                            ret, length = fixed_rtg_rollout(
                                self.strict_env, model, self.state_mean, self.state_std,
                                int(protocol.split("_")[1]), args.device,
                            )
                        else:
                            env = (
                                self.strict_env if protocol == "qt_strict_delayed"
                                else self.eval_env
                            )
                            ret, length = upstream.evaluate_episode_rtg(
                                env, model.state_dim, model.act_dim, model, critic,
                                max_ep_len=1000, scale=1000.0,
                                state_mean=self.state_mean, state_std=self.state_std,
                                device=args.device, target_return=[12.0, 9.0, 6.0],
                                mode="delayed",
                            )
                        returns.append(float(ret))
                        lengths.append(int(length))
                score = float(self.eval_env.get_normalized_score(np.mean(returns)) * 100)
                prefix = f"eval/{protocol}"
                record[f"{prefix}/normalized_score"] = score
                record[f"{prefix}/return_mean"] = float(np.mean(returns))
                record[f"{prefix}/return_std"] = float(np.std(returns))
                record[f"{prefix}/episodes"] = count
                record[f"{prefix}/length_mean"] = float(np.mean(lengths))
                if protocol not in self.best or score > self.best[protocol]["score"]:
                    self.best[protocol] = {"score": score, "step": self.step}
                print(f"QT EVAL {self.step} {protocol}: {score:.4f}", flush=True)
            append_json(output / "evaluations.jsonl", record)
            self.evaluations.append(record)
            json_write(output / "summary.json", {
                "completed_updates": self.step,
                "complete": self.step == args.updates,
                "best": self.best,
                "last": record,
                "known_issue": "upstream_delayed_terminal_reward_excluded_uncorrected",
            })
            run.log(record)
            # Retain upstream best-checkpoint selection, using the main strict score.
            return {
                "strict_return_mean": record["eval/qt_strict_delayed/return_mean"],
                "strict_normalized_score": (
                    record["eval/qt_strict_delayed/normalized_score"] / 100.0
                ),
            }

    return InstrumentedTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--updates", type=int, default=100000)
    parser.add_argument("--eval-every", type=int, default=5000)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--feedback-eval-episodes", type=int, default=10)
    parser.add_argument("--eval-seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.source = str(Path(args.source).resolve())
    args.output = str(Path(args.output).resolve())
    if args.smoke:
        args.updates = 20
        args.eval_every = 20
        args.eval_episodes = 1
        args.feedback_eval_episodes = 1
        args.log_every = 5
    block_size = 20 if args.smoke else 1000
    if args.updates % block_size or args.eval_every % block_size:
        raise ValueError("Budgets must be multiples of upstream iteration length")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    upstream, manifest = load_upstream(args.source)
    variant = {
        "exp_name": "qt-official-uncorrected", "seed": args.seed,
        "env": "halfcheetah", "dataset": "medium-replay", "mode": "delayed",
        "K": 5, "pct_traj": 1.0, "batch_size": 256, "embed_dim": 256,
        "n_layer": 4, "n_head": 4, "activation_function": "relu", "dropout": 0.1,
        "learning_rate": 3e-4, "lr_min": 0.0, "weight_decay": 1e-4,
        "warmup_steps": 10000, "num_eval_episodes": args.eval_episodes,
        "max_iters": 500, "num_steps_per_iter": block_size, "device": args.device,
        "save_path": str(output) + "/", "discount": 0.99, "tau": 0.005,
        "eta": 5.0, "eta2": 1.0, "lambda": 1.0, "max_q_backup": False,
        "lr_decay": True, "grad_norm": 15.0, "early_stop": True,
        "early_epoch": args.updates // block_size - 1,
        "k_rewards": True, "use_discount": True, "sar": False,
        "reward_tune": "no", "scale": None, "test_scale": None,
        "rtg_no_q": False, "infer_no_q": False,
    }
    config = {
        "upstream_repository": "https://github.com/charleshsc/QT",
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_source_sha256": manifest,
        "adapter_sha256": file_hash(__file__),
        "adapter": vars(args), "variant": variant,
        "known_issue": "delayed_terminal_reward_excluded_uncorrected",
        "parameter_provenance": "released HCMR run.sh (not delayed-tuned)",
        "reward_mode": "delayed", "env_name": ENV_NAME,
        "comparison_warning": "different model/batch/context/inference from CORL DT",
        "package_versions": {
            key: importlib.metadata.version(key) for key in (
                "torch", "numpy", "gym", "transformers", "tokenizers",
                "huggingface-hub", "wandb", "mujoco-py", "d4rl"
            )
        },
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    dataset_path = Path(args.source) / "D4RL" / f"{ENV_NAME}.pkl"
    config["dataset_sha256"] = file_hash(dataset_path)
    with open(dataset_path, "rb") as stream:
        trajectories = pickle.load(stream)
    config["num_trajectories"] = len(trajectories)
    config["num_transitions"] = sum(len(x["rewards"]) for x in trajectories)
    json_write(output / "config.json", config)
    run = wandb.init(
        project="CORL-DDR", entity="2820402607-shandong-university",
        group="QT-Official-Uncorrected-HCMR-delayed",
        name=f"QT-Official-Uncorrected-HCMR-delayed-seed{args.seed}-100k",
        config=config, dir=str(output),
        mode="offline" if args.smoke else "online",
        settings=wandb.Settings(disable_code=True),
        tags=["QT", "official-uncorrected", "delayed", "terminal-reward-issue"],
    )
    run.define_metric("update_step")
    run.define_metric("*", step_metric="update_step")
    json_write(output / "wandb_run.json", {"id": run.id, "url": run.url})
    upstream.Trainer = make_instrumented_trainer(upstream, args, output, run)
    upstream.args = types.SimpleNamespace(save_path=variant["save_path"])
    os.chdir(args.source)
    try:
        upstream.experiment(variant["exp_name"], variant)
    except BaseException:
        run.finish(exit_code=1)
        raise
    run.finish()


if __name__ == "__main__":
    main()
