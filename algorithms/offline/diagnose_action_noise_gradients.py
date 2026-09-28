"""Read-only, matched gradient diagnostic for a proposed PREFORL-inspired DT loss.

No optimizer, environment rollout, checkpoint write, or training-mode dropout.
Run via scripts/dt_experiments/run_action_noise_gradient_diagnostic.sbatch.
"""

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

def scores(prediction, labels, mask, score_std):
    # FP64 scalar objective avoids cancellation of the tiny noise correction;
    # model parameters/forward/backward remain FP32, as in the source DT.
    error = labels.detach().double() - prediction.double()
    logp = -error.square() / (2 * score_std**2)
    logp = logp - math.log(score_std * math.sqrt(2 * math.pi))
    denominator = mask.sum(1) * prediction.shape[-1]
    if not bool((denominator > 0).all()):
        raise ValueError("Each segment must have at least one valid token")
    return (logp * mask.unsqueeze(-1)).sum((1, 2)) / denominator


def objectives(prediction, actions, negatives, mask, std=0.1, alpha=0.1, bias=0.5):
    positive = scores(prediction, actions, mask, std)
    negative = scores(prediction, negatives, mask, std)
    control = F.softplus(-alpha * (1 - bias) * positive).mean()
    contrastive = F.softplus(-alpha * (positive - bias * negative)).mean()
    return control, contrastive


def sample_negative(actions, mask, rng, amplitude=0.01, probability=0.4):
    # One mask per time step, shared by all action coordinates; no clipping.
    selected = rng.random((*actions.shape[:2], 1)) < probability
    noise = rng.uniform(-amplitude, amplitude, size=tuple(actions.shape))
    perturbation = torch.as_tensor(noise * selected, device=actions.device)
    perturbation = perturbation * mask.unsqueeze(-1)
    # Keep synthetic labels in FP64 so roundoff is not confused with noise.
    negatives = actions.double() + perturbation
    valid = mask.bool().unsqueeze(-1).expand_as(actions)
    audit = {
        "selected_valid_timestep_fraction": float(
            (selected[..., 0] * mask.cpu().numpy()).sum() / mask.sum().item()
        ),
        "mean_squared_perturbation": float(perturbation[valid].square().mean()),
        "out_of_bounds_fraction": float((negatives[valid].abs() > 1).double().mean()),
        "maximum_absolute_perturbation": float(perturbation[valid].abs().max()),
    }
    return negatives, audit


def gradient_vector(loss, parameters, retain_graph=False):
    gradients = torch.autograd.grad(loss, parameters, retain_graph=retain_graph)
    result = torch.cat([item.detach().reshape(-1) for item in gradients])
    if not bool(torch.isfinite(result).all()):
        raise FloatingPointError("Nonfinite gradient")
    return result.cpu().double().numpy()


def cosine(first, second):
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if denominator == 0:
        return None
    return float(np.clip(np.dot(first, second) / denominator, -1, 1))


def compare_gradients(base, dt_gradient, deltas, weight=0.05):
    """Deltas are directly differentiated (L_C-L_B), not subtracted FP32 grads."""
    mean_delta = deltas.mean(axis=0)
    base_norm = float(np.linalg.norm(base))
    total_base = dt_gradient + weight * base
    total_norm = float(np.linalg.norm(total_base))
    mean_norm = float(np.linalg.norm(mean_delta))
    variance_trace = float(np.square(deltas - mean_delta).sum() / (len(deltas) - 1))
    single_noise = math.sqrt(variance_trace)
    mean_error = single_noise / math.sqrt(len(deltas))
    signal_squared = mean_norm**2 - mean_error**2
    def ratio(value, norm):
        return float(value / norm) if norm > 1e-15 else None
    return {
        "base_gradient_norm": base_norm,
        "dt_gradient_norm": float(np.linalg.norm(dt_gradient)),
        "weighted_aux_to_dt_norm": ratio(
            weight * base_norm, np.linalg.norm(dt_gradient)
        ),
        "mean_gradient_cosine": cosine(base, base + mean_delta),
        "mean_relative_difference": ratio(mean_norm, base_norm),
        "single_draw_noise_rms_relative": ratio(single_noise, base_norm),
        "mc_mean_error_norm_relative": ratio(mean_error, base_norm),
        "mean_difference_to_mc_error": ratio(mean_norm, mean_error),
        "noise_corrected_squared_signal_relative": ratio(signal_squared, base_norm**2),
        "noise_corrected_signal_relative": ratio(
            math.sqrt(max(signal_squared, 0)), base_norm
        ),
        "split_half_difference_cosine": cosine(
            deltas[::2].mean(0), deltas[1::2].mean(0)
        ),
        "total_gradient_cosine": cosine(total_base, total_base + weight * mean_delta),
        "total_mean_relative_difference": ratio(weight * mean_norm, total_norm),
        "total_mc_mean_error_norm_relative": ratio(weight * mean_error, total_norm),
        "single_draw_relative_difference_mean": float(
            np.mean([np.linalg.norm(delta) / base_norm for delta in deltas])
        ) if base_norm > 1e-15 else None,
    }


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_hash(items):
    digest = hashlib.sha256()
    for name, tensor in items:
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def prepare_batch(dataset, rng, size, probabilities, device):
    ids = rng.choice(len(dataset.dataset), size=size, p=probabilities)
    starts = [int(rng.integers(len(dataset.dataset[index]["actions"]))) for index in ids]
    samples = [
        dataset._SequenceDataset__prepare_sample(int(i), t) for i, t in zip(ids, starts)
    ]
    batch = tuple(
        torch.as_tensor(np.stack(values), device=device) for values in zip(*samples)
    )
    return batch, {"trajectory_ids": ids.tolist(), "starts": starts}


def forward(model, batch):
    states, actions, returns, times, mask = batch
    return model(states, actions, returns, times, ~mask.bool())


def main_gradient(model, parameters, batch, microbatch):
    # Same CORL padded-token MSE; only split the batch to cap peak GPU memory.
    count = len(batch[0])
    result = None
    loss_value = 0.0
    for start in range(0, count, microbatch):
        part = tuple(item[start : start + microbatch] for item in batch)
        prediction = forward(model, part)
        loss = ((prediction - part[1]).square() * part[4].unsqueeze(-1)).mean()
        fraction = len(part[0]) / count
        grad = gradient_vector(loss * fraction, parameters)
        result = grad if result is None else result + grad
        loss_value += float(loss.detach()) * fraction
    return result, loss_value


def write_json(path, value):
    with open(path, "w") as target:
        json.dump(value, target, indent=2, allow_nan=False)


def run(args):
    # These imports need the existing D4RL/MuJoCo environment. Unit tests of
    # the objective/statistics themselves require only NumPy and PyTorch.
    from dt import DecisionTransformer, SequenceDataset

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    checkpoint_hash_before = sha256_file(args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint["config"]
    if config["env_name"] != "halfcheetah-medium-replay-v2":
        raise ValueError("Wrong dataset")
    if config["reward_mode"] != "delayed" or config["train_seed"] != 0:
        raise ValueError("Expected seed-0 delayed DT")
    if config.get("top_weight") != 1.0:
        raise ValueError("Expected the ordinary from-scratch DT control")
    dataset = SequenceDataset(
        config["env_name"], config["seq_len"], config["reward_scale"], "delayed"
    )
    np.testing.assert_array_equal(dataset.state_mean, checkpoint["state_mean"])
    np.testing.assert_array_equal(dataset.state_std, checkpoint["state_std"])
    returns = np.array([trajectory["returns"][0] for trajectory in dataset.dataset])
    selected = np.argsort(-returns, kind="stable")[: math.ceil(0.25 * len(returns))]
    aux_probabilities = dataset.sample_prob.copy()
    aux_probabilities[~np.isin(np.arange(len(returns)), selected)] = 0
    aux_probabilities /= aux_probabilities.sum()
    for trajectory in dataset.dataset:
        if np.count_nonzero(trajectory["rewards"][:-1]):
            raise AssertionError("Nonterminal reward in delayed dataset")
        np.testing.assert_array_equal(
            trajectory["returns"],
            np.full_like(trajectory["returns"], trajectory["returns"][0]),
        )
    model_keys = (
        "seq_len", "episode_len", "embedding_dim", "num_layers", "num_heads",
        "attention_dropout", "residual_dropout", "embedding_dropout", "max_action",
    )
    model = DecisionTransformer(
        state_dim=dataset.state_mean.shape[-1],
        action_dim=dataset.dataset[0]["actions"].shape[-1],
        **{key: config[key] for key in model_keys},
    ).to(args.device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    del checkpoint
    initial_model_hash = tensor_hash(model.state_dict().items())
    parameters = tuple(model.parameters())
    groups = {
        "all": [], "action_head": [], "transformer_blocks": [], "embeddings_norms": []
    }
    offset = 0
    for name, parameter in model.named_parameters():
        positions = np.arange(offset, offset + parameter.numel())
        groups["all"].append(positions)
        group = "action_head" if name.startswith("action_head.") else (
            "transformer_blocks" if name.startswith("blocks.") else "embeddings_norms"
        )
        groups[group].append(positions)
        offset += parameter.numel()
    groups = {name: np.concatenate(indices) for name, indices in groups.items()}
    provenance = {
        "arguments": vars(args), "source_config": config,
        "checkpoint_sha256": checkpoint_hash_before, "model_sha256": initial_model_hash,
        "git_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "diagnostic_source_sha256": sha256_file(__file__),
        "dt_source_sha256": sha256_file(Path(__file__).with_name("dt.py")),
        "torch_version": torch.__version__, "numpy_version": np.__version__,
        "device": (
            torch.cuda.get_device_name() if args.device.startswith("cuda") else "cpu"
        ),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "parameter_count": offset, "dataset_stats": dataset.stats,
        "selected_trajectory_ids": selected.tolist(),
        "selected_returns": returns[selected].tolist(),
        "selected_threshold": float(returns[selected].min()),
        "sampling": (
            "length-weighted trajectories then uniform starts; recorded delayed RTG"
        ),
        "reference": (
            "Identical frozen copy: exact zero loss/gradient in eval mode; "
            "no teacher training"
        ),
        "precision": (
            "FP32 model; FP64 auxiliary scores and CPU gradient statistics; TF32 off"
        ),
        "main_batch_size": args.main_batch_size,
        "limitations": (
            "Local gradients, not AdamW updates or return improvement; "
            "no environment steps"
        ),
    }
    write_json(output / "provenance.json", provenance)
    print("PROVENANCE " + json.dumps(provenance), flush=True)
    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(
            project="corl-ddr", entity="2820402607-shandong-university",
            group="ActionNoise-GradientDiagnostic-HCMR-delayed-seed0",
            name="ActionNoise-GradientDiagnostic-DT50k-HCMR-delayed-seed0",
            job_type="gradient-diagnostic", config=provenance,
            tags=["diagnostic", "no-training", "delayed", "seed0"],
        )
        provenance["wandb_url"] = wandb_run.url
        write_json(output / "provenance.json", provenance)
    records = []
    batch_indices = []
    global_sums = {name: None for name in ("base", "dt", "delta_mean")}
    global_variance_trace = 0.0
    for batch_id in range(args.batches):
        main_batch, main_indices = prepare_batch(
            dataset, np.random.default_rng(args.seed + batch_id * 10 + 1),
            args.main_batch_size, dataset.sample_prob, args.device,
        )
        auxiliary, auxiliary_indices = prepare_batch(
            dataset, np.random.default_rng(args.seed + batch_id * 10 + 2),
            args.aux_batch_size, aux_probabilities, args.device,
        )
        batch_indices.append({"main": main_indices, "auxiliary": auxiliary_indices})
        main_hash = tensor_hash((str(i), tensor) for i, tensor in enumerate(main_batch))
        aux_hash = tensor_hash((str(i), tensor) for i, tensor in enumerate(auxiliary))
        dt_grad, dt_loss = main_gradient(model, parameters, main_batch, args.microbatch)
        del main_batch
        prediction = forward(model, auxiliary)
        actions, mask = auxiliary[1], auxiliary[4]
        base_loss, zero_loss = objectives(prediction, actions, actions, mask)
        base_grad = gradient_vector(base_loss, parameters, retain_graph=True)
        zero_delta = gradient_vector(
            zero_loss - base_loss, parameters, retain_graph=True
        )
        zero_max = float(np.abs(zero_delta).max())
        if zero_max > 1e-10 or float((zero_loss - base_loss).abs()) > 1e-12:
            raise AssertionError("Zero-noise objective/gradient mismatch")
        deltas = []
        repetition_records = []
        for repeat in range(args.repeats):
            noise_seed = args.seed + 100000 + batch_id * 1000 + repeat
            negatives, audit = sample_negative(
                actions, mask, np.random.default_rng(noise_seed)
            )
            control, contrastive = objectives(prediction, actions, negatives, mask)
            delta = gradient_vector(
                contrastive - control, parameters, retain_graph=repeat < args.repeats - 1
            )
            deltas.append(delta)
            repetition_records.append({
                "repeat": repeat, "noise_seed": noise_seed, **audit,
                "contrastive_loss": float(contrastive.detach()),
                "loss_difference": float((contrastive - control).detach()),
                "relative_gradient_difference": float(
                    np.linalg.norm(delta) / np.linalg.norm(base_grad)
                ),
                "gradient_cosine": cosine(base_grad, base_grad + delta),
            })
        deltas = np.stack(deltas)
        record = {
            "batch": batch_id, "dt_loss": dt_loss,
            "base_aux_loss": float(base_loss.detach()),
            "main_batch_sha256": main_hash, "aux_batch_sha256": aux_hash,
            "zero_noise_max_absolute_gradient_difference": zero_max,
            "groups": {
                name: compare_gradients(
                    base_grad[index], dt_grad[index], deltas[:, index]
                )
                for name, index in groups.items()
            },
            "repeats": repetition_records,
        }
        records.append(record)
        mean_delta = deltas.mean(0)
        global_variance_trace += float(
            np.square(deltas - mean_delta).sum() / (args.repeats - 1)
        )
        for name, value in (
            ("base", base_grad), ("dt", dt_grad), ("delta_mean", mean_delta)
        ):
            previous = global_sums[name]
            global_sums[name] = value.copy() if previous is None else previous + value
        write_json(output / "batches.json", records)
        write_json(output / "sample_indices.json", batch_indices)
        print(
            "BATCH " + json.dumps({"batch": batch_id, **record["groups"]["all"]}),
            flush=True,
        )
        if wandb_run:
            wandb_run.log({
                "batch": batch_id,
                **{f"{group}/{key}": value for group, metrics in record["groups"].items()
                   for key, value in metrics.items() if value is not None},
            })
        del prediction, base_loss, zero_loss, control, contrastive, auxiliary, deltas
    summary = {}
    for group in groups:
        summary[group] = {}
        for metric in records[0]["groups"][group]:
            values = [record["groups"][group][metric] for record in records]
            values = [value for value in values if value is not None]
            if values:
                summary[group][metric] = {
                    "mean": float(np.mean(values)), "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }
    summed_base = global_sums["base"]
    summed_delta = global_sums["delta_mean"]
    summed_total = global_sums["dt"] + 0.05 * summed_base
    summed_mc_error = math.sqrt(global_variance_trace / args.repeats)
    pooled = {
        "mean_gradient_cosine": cosine(summed_base, summed_base + summed_delta),
        "mean_relative_difference": float(
            np.linalg.norm(summed_delta) / np.linalg.norm(summed_base)
        ),
        "mc_mean_error_norm_relative": float(
            summed_mc_error / np.linalg.norm(summed_base)
        ),
        "total_mean_relative_difference": float(
            0.05 * np.linalg.norm(summed_delta) / np.linalg.norm(summed_total)
        ),
        "total_gradient_cosine": cosine(
            summed_total, summed_total + 0.05 * summed_delta
        ),
        "note": (
            "Pooled fixed-batch mean; MC error is conditional on these batches, "
            "not dataset uncertainty"
        ),
    }
    final_hash = tensor_hash(model.state_dict().items())
    file_hash_after = sha256_file(args.checkpoint)
    if final_hash != initial_model_hash or file_hash_after != checkpoint_hash_before:
        raise AssertionError("Input model/checkpoint changed")
    result = {
        "summary": summary, "pooled": pooled, "model_unchanged": True,
        "checkpoint_file_unchanged": True, "optimizer_steps": 0, "environment_steps": 0,
        "reference_loss_and_gradient": 0.0,
        "elapsed_seconds": time.time() - started,
        "zero_noise_checks_passed": args.batches,
    }
    write_json(output / "summary.json", result)
    print("SUMMARY " + json.dumps(result), flush=True)
    if wandb_run:
        wandb_run.summary.update(result)
        for filename in (
            "provenance.json", "batches.json", "sample_indices.json", "summary.json"
        ):
            wandb_run.save(str(output / filename), base_path=str(output), policy="now")
        wandb_run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=16)
    parser.add_argument("--main-batch-size", type=int, default=4096)
    parser.add_argument("--aux-batch-size", type=int, default=256)
    parser.add_argument("--microbatch", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wandb", action="store_true")
    arguments = parser.parse_args()
    if arguments.repeats < 2 or min(arguments.batches, arguments.main_batch_size,
                                    arguments.aux_batch_size, arguments.microbatch) < 1:
        parser.error("Positive batch sizes and at least two repetitions are required")
    run(arguments)
