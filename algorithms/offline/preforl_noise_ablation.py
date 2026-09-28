"""Matched on/off action-corruption ablation of the existing PREFORL HCMR port.

Imports the unchanged, SHA256-checked legacy port. Outputs never share a run
directory with that port or with the other arm. Online W&B is opt-in only.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import pickle
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

SOURCE_SHA256 = "762cd5196866782e5046dfda325a773fe3735bafb43b91541fb1c8816462b934"


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_hash(arrays):
    digest = hashlib.sha256()
    for value in arrays:
        value = np.ascontiguousarray(value)
        digest.update(str((value.shape, str(value.dtype))).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def model_hash(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_source(path):
    if file_hash(path) != SOURCE_SHA256:
        raise ValueError("Legacy PREFORL source changed; review before running")
    spec = importlib.util.spec_from_file_location("preforl_legacy_port", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PairedSampler:
    """Isolate the original global-NumPy/Python sampler from logging/evaluation."""

    def __init__(self, seed):
        self.numpy_state = np.random.RandomState(seed).get_state()
        self.python_state = random.Random(seed).getstate()

    def sample(self, source, *args):
        outside_numpy = np.random.get_state()
        outside_python = random.getstate()
        try:
            np.random.set_state(self.numpy_state)
            random.setstate(self.python_state)
            arrays = source.make_training_batch(*args)
            self.numpy_state = np.random.get_state()
            self.python_state = random.getstate()
        finally:
            np.random.set_state(outside_numpy)
            random.setstate(outside_python)
        return arrays

    def state_hash(self):
        return hashlib.sha256(
            pickle.dumps((self.numpy_state, self.python_state), protocol=4)
        ).hexdigest()


def pairing_signature(arrays):
    first_obs, first_act, second_obs, second_act, bc_obs, _, labels = arrays
    positive_actions = np.where(labels[:, None, None] == 0, first_act, second_act)
    return array_hash([first_obs, second_obs, bc_obs, labels, positive_actions])


def losses(source, policy, arrays, device, alpha, bias, bc_coeff):
    first_obs, first_act, second_obs, second_act, bc_obs, bc_act, labels = [
        torch.as_tensor(value, dtype=torch.float32, device=device) for value in arrays
    ]
    count, length, obs_dim = first_obs.shape
    action_dim = first_act.shape[-1]
    first_logp = policy.log_prob(
        first_obs.reshape(-1, obs_dim), first_act.reshape(-1, action_dim)
    ).reshape(count, length)
    second_logp = policy.log_prob(
        second_obs.reshape(-1, obs_dim), second_act.reshape(-1, action_dim)
    ).reshape(count, length)
    preference = source.biased_preference_loss(
        alpha * first_logp.sum(dim=1), alpha * second_logp.sum(dim=1), labels, bias
    )
    bc = -policy.log_prob(
        bc_obs.reshape(-1, obs_dim), bc_act.reshape(-1, action_dim)
    ).mean()
    return preference + bc_coeff * bc, preference, bc


def write_json(path, value):
    temporary = str(path) + ".tmp"
    with open(temporary, "w") as destination:
        json.dump(value, destination, indent=2, allow_nan=False)
    os.replace(temporary, path)


@torch.no_grad()
def probe_policy(policy, observations, actions):
    mean = policy.action_mean(policy.features(observations))
    metrics = {
        "probe/action_mean_mse": float((mean - actions).square().mean()),
        "probe/action_mean_rms": float(mean.square().mean().sqrt()),
        "probe/action_mean_out_of_bounds": float((mean.abs() > 1).float().mean()),
        "probe/std_mean": float(policy.log_std.exp().mean()),
        "probe/std_min": float(policy.log_std.exp().min()),
        "probe/std_max": float(policy.log_std.exp().max()),
    }
    for index, value in enumerate(policy.log_std.detach().cpu().tolist()):
        metrics[f"probe/log_std_{index}"] = value
    return metrics, mean.detach().cpu().numpy()


def run(args):
    source = load_source(args.source)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    checkpoints = output / "checkpoints"
    checkpoints.mkdir()
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)
    source.set_seed(args.seed)
    env = source.gym.make("halfcheetah-medium-replay-v2")
    data = env.get_dataset()
    episodes = source.split_episodes(data, args.segment_length)
    returns, delay_error = source.delay_rewards(episodes)
    threshold = float(np.quantile(returns, 0.75))
    selected_ids = [i for i, ret in enumerate(returns) if ret > threshold]
    successful = [episodes[i] for i in selected_ids]
    if delay_error > 1e-4 or not successful:
        raise AssertionError("Invalid delayed data or empty selected set")
    if any(np.count_nonzero(episode["rewards"][:-1]) for episode in episodes):
        raise AssertionError("Nonterminal reward survived the delayed transform")
    obs_dim = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.shape[0])
    # Reset after env/data construction so model initialization is arm-independent.
    source.set_seed(args.seed)
    policy = source.GaussianPolicy(
        obs_dim, action_dim, args.hidden_dim, args.hidden_layers
    ).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    sampler = PairedSampler(args.seed)
    initial_hash = model_hash(policy)
    selected_hash = array_hash([
        value for ep in successful for value in (ep["observations"], ep["actions"])
    ])
    probe_rng = np.random.default_rng(777)
    probe_pairs = [
        (int(probe_rng.integers(len(successful))), int(probe_rng.integers(1000)))
        for _ in range(1024)
    ]
    # This dataset consists of full 1000-transition episodes. Validate, don't assume.
    if any(len(ep["actions"]) != 1000 for ep in successful):
        raise AssertionError("Expected full HCMR trajectories for fixed probe indexing")
    probe_obs_np = np.stack([successful[i]["observations"][t] for i, t in probe_pairs])
    probe_act_np = np.stack([successful[i]["actions"][t] for i, t in probe_pairs])
    probe_obs = torch.as_tensor(probe_obs_np, device=device)
    probe_act = torch.as_tensor(probe_act_np, device=device)
    source_config = {
        **vars(args), "algorithm": "PREFORL/PNP-noise-ablation",
        "env_name": "halfcheetah-medium-replay-v2", "reward_mode": "delayed",
        "source_sha256": file_hash(args.source), "runner_sha256": file_hash(__file__),
        "git_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "initial_model_sha256": initial_hash, "selected_data_sha256": selected_hash,
        "probe_sha256": array_hash([probe_obs_np, probe_act_np]),
        "dataset_episodes": len(episodes), "successful_episodes": len(successful),
        "return_threshold": threshold, "delayed_max_return_error": delay_error,
        "effective_batch_size": min(args.batch_size, len(successful)),
        "eval_seed_rule": "train_seed + 10000 + step * eval_episodes + episode",
        "evaluation": "deterministic mean action, clipped; normalized total return",
        "primary_metrics": (
            "last and mean of last five evaluation means; best secondary"
        ),
        "bc_convention": (
            "unchanged source: first segment, possibly corrupted after flip"
        ),
        "torch_version": torch.__version__, "numpy_version": np.__version__,
        "gpu": torch.cuda.get_device_name() if device.type == "cuda" else "cpu",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }
    write_json(output / "config.json", source_config)
    write_json(output / "selection.json", {
        "trajectory_ids": selected_ids, "returns": returns[selected_ids].tolist(),
        "probe_indices_within_selected": probe_pairs,
    })
    log_config = {
        k: v for k, v in source_config.items() if k not in ("source", "output")
    }
    wandb_run = source.wandb.init(
        project="corl-ddr", entity="2820402607-shandong-university",
        group="PREFORL-NoiseAblation-HCMR-delayed-seed0",
        name=f"PREFORL-Noise{args.shadow_noise:g}-Paired-HCMR-delayed-seed0",
        mode=args.wandb_mode, config=log_config, dir=str(output),
        tags=["paired-noise-ablation", "delayed", "seed0"],
    )
    source_config["wandb_id"] = wandb_run.id
    source_config["wandb_mode"] = args.wandb_mode
    if args.wandb_mode == "online":
        source_config["wandb_url"] = wandb_run.url
    write_json(output / "config.json", source_config)
    print("CONFIG " + json.dumps(source_config), flush=True)
    audit = []
    history = []
    stream_digest = hashlib.sha256()
    best = None

    def save(step, name):
        path = checkpoints / name
        torch.save({
            "policy": policy.state_dict(), "optimizer": optimizer.state_dict(),
            "gradient_step": step, "best_normalized_score": best,
            "config": source_config, "sampler_numpy_state": sampler.numpy_state,
            "sampler_python_state": sampler.python_state,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
        }, str(path) + ".tmp")
        os.replace(str(path) + ".tmp", path)

    save(0, "step000000.pt")
    _, initial_actions = probe_policy(policy, probe_obs, probe_act)
    np.save(output / "probe_actions_step000000.npy", initial_actions)
    action_low = np.asarray(env.action_space.low, dtype=np.float32)
    action_high = np.asarray(env.action_space.high, dtype=np.float32)
    started = time.time()
    for step in range(1, args.gradient_steps + 1):
        arrays = sampler.sample(
            source, successful, args.batch_size, args.segment_length,
            args.shadow_noise, 0.4, action_low, action_high,
        )
        signature = pairing_signature(arrays)
        stream_digest.update(signature.encode())
        if step <= 3 or step % args.eval_interval in (0, 1):
            audit.append({
                "step": step, "structural_batch_sha256": signature,
                "sampler_state_sha256": sampler.state_hash(),
                "cumulative_structural_sha256": stream_digest.hexdigest(),
            })
            write_json(output / "pairing_audit.json", audit)
        loss, preference, bc = losses(
            source, policy, arrays, device, args.alpha, args.bias, args.bc_coeff
        )
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"Nonfinite loss at step {step}")
        optimizer.zero_grad()
        loss.backward()
        std_grad_norm = float(policy.log_std.grad.norm())
        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(), max_norm=1000.0, error_if_nonfinite=True
        )
        optimizer.step()
        metrics = {
            "gradient_step": step, "train/loss": float(loss.detach()),
            "train/preference_loss": float(preference.detach()),
            "train/bc_loss": float(bc.detach()), "train/grad_norm": float(grad_norm),
            "train/log_std_mean": float(policy.log_std.detach().mean()),
            "train/log_std_grad_norm": std_grad_norm,
            "train/elapsed_seconds": time.time() - started,
        }
        should_eval = step % args.eval_interval == 0 or step == args.gradient_steps
        if should_eval:
            metrics.update(source.evaluate(
                policy, env, device, args.seed + 10000 + step * args.eval_episodes,
                args.eval_episodes, threshold,
            ))
            probe_metrics, probe_actions = probe_policy(policy, probe_obs, probe_act)
            metrics.update(probe_metrics)
            np.save(output / f"probe_actions_step{step:06d}.npy", probe_actions)
            score = metrics["eval/normalized_score_mean"]
            is_best = best is None or score > best
            best = score if is_best else best
            metrics["eval/best_normalized_score_mean"] = best
            history.append(dict(metrics))
            write_json(output / "history.json", history)
            save(step, "latest.pt")
            if is_best:
                save(step, "best.pt")
            if step % 5000 == 0 or step == args.gradient_steps:
                save(step, f"step{step:06d}.pt")
            print("EVAL " + json.dumps(metrics), flush=True)
        if step == 1 or step % 25 == 0 or should_eval:
            wandb_run.log(metrics, step=step)
            with open(output / "train_metrics.jsonl", "a") as destination:
                destination.write(json.dumps(metrics, allow_nan=False) + "\n")
            if not should_eval:
                print("TRAIN " + json.dumps(metrics), flush=True)
    summary = {
        "completed_steps": args.gradient_steps,
        "last_normalized_score": history[-1]["eval/normalized_score_mean"],
        "best_normalized_score": best,
        "last_five_mean_normalized_score": float(np.mean([
            row["eval/normalized_score_mean"] for row in history[-5:]
        ])),
        "last_five_steps": [row["gradient_step"] for row in history[-5:]],
        "cumulative_structural_sha256": stream_digest.hexdigest(),
        "final_sampler_state_sha256": sampler.state_hash(),
        "final_model_sha256": model_hash(policy),
        "elapsed_seconds": time.time() - started,
    }
    write_json(output / "summary.json", summary)
    wandb_run.summary.update(summary)
    wandb_run.finish()
    env.close()
    print("SUMMARY " + json.dumps(summary), flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=str(
        Path(__file__).resolve().parents[2]
        / "third_party/PREFORL/train_mujoco_delayed.py"
    ))
    parser.add_argument("--output", required=True)
    parser.add_argument("--shadow-noise", type=float, required=True, choices=(0.0, 0.01))
    parser.add_argument("--seed", type=int, default=0, choices=(0,))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--gradient-steps", type=int, default=15000)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--segment-length", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--hidden-layers", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--bias", type=float, default=0.5)
    parser.add_argument("--bc-coeff", type=float, default=0.5)
    parser.add_argument("--wandb-mode", choices=("offline", "disabled", "online"),
                        default="offline")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
