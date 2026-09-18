"""From-scratch, current-policy pair scoring with a periodically frozen anchor.

Independent entry point: historical v3 and all old launchers remain unchanged.
The reference sampler is fixed and shared across arms, NOT preference-prioritized.
"""

import copy
import hashlib
import json
import os
import random
import subprocess
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Tuple

import gym
import numpy as np
import pyrallis
import torch
import torch.nn.functional as F
import wandb
from dt import (
    DecisionTransformer,
    eval_rollout,
    SequenceDataset,
    set_seed,
    TrainConfig,
    wandb_init,
    wrap_env,
)
from hard_fork_dt import file_sha256, predict_action, StateAlignedPreferenceDataset
from top_return_weighted_dt import batch_hash, model_hash, restore_rng, rng_state
from torch.utils.data import DataLoader

@dataclass
class AdaptiveConfig(TrainConfig):
    arm: str = "adaptive_pref"
    pair_arrays_path: str = ""
    auxiliary_start: int = 10_000
    auxiliary_full: int = 20_000
    reference_every: int = 5_000
    rescore_every: int = 1_000
    preference_weight: float = 0.05
    reference_weight: float = 0.1
    preference_margin: float = 0.05
    preference_batch_size: int = 256
    min_active_pairs: int = 16
    target_return: float = 12_000.0
    coverage_fraction: float = 0.2
    violation_cap: float = 5.0
    pair_seed: int = 0
    score_batch_size: int = 512
    log_every: int = 100
    checkpoint_steps: Tuple[int, ...] = (0, 10_000, 20_000, 50_000, 75_000, 100_000)


class AuditedSequenceDataset(SequenceDataset):
    def __iter__(self):
        while True:
            trajectory = np.random.choice(len(self.dataset), p=self.sample_prob)
            start = random.randint(0, len(self.dataset[trajectory]["rewards"]) - 1)
            sample = self._SequenceDataset__prepare_sample(trajectory, start)
            yield (*sample, np.asarray([trajectory, start], dtype=np.int64))


def ramp(completed, start, full):
    return float(np.clip((completed - start) / (full - start), 0.0, 1.0))


def refresh_due(completed, start, interval):
    return completed >= start and (completed - start) % interval == 0


def sampling_probabilities(confidence, violations, margin, coverage, cap):
    """No historical model errors are inputs; retain a confidence-only floor."""
    confidence = np.maximum(np.asarray(confidence, dtype=np.float64), 1e-8)
    violations = np.maximum(np.asarray(violations, dtype=np.float64), 0.0)
    if (not np.isfinite(confidence).all() or not np.isfinite(violations).all()
            or confidence.shape != violations.shape or not len(confidence)):
        raise ValueError("Invalid pair confidences/violations")
    base = confidence / confidence.sum()
    weighted = confidence * np.minimum(violations / margin, cap)
    if weighted.sum() == 0:
        return base
    return coverage * base + (1.0 - coverage) * weighted / weighted.sum()


@contextmanager
def evaluating(model):
    """Disable dropout without disabling student autograd; restore mode safely."""
    previous = model.training
    model.eval()
    try:
        yield
    finally:
        model.train(previous)


def frozen_copy(model):
    reference = copy.deepcopy(model).eval()
    reference.requires_grad_(False)
    return reference


def preference_loss(prediction, positive, negative, margin, minimum):
    positive_error = F.mse_loss(prediction, positive, reduction="none").mean(-1)
    negative_error = F.mse_loss(prediction, negative, reduction="none").mean(-1)
    violation = positive_error - negative_error.detach() + margin
    active = (violation > 0).sum()
    denominator = active.clamp_min(1) if active.item() >= minimum else len(violation)
    loss = F.relu(violation).sum() / denominator
    return loss, {
        "preference_loss": loss.detach().item(),
        "preference_active_count": active.item(),
        "preference_active_ratio": active.item() / len(violation),
        "active_normalization_fallback": float(active.item() < minimum),
        "preference_positive_error": positive_error.mean().detach().item(),
        "preference_negative_error": negative_error.mean().detach().item(),
        "preference_accuracy": (positive_error < negative_error).float().mean().item(),
    }


def combined_loss(dt_loss, pref_loss, ref_loss, arm, factor, config):
    # Directly return the original objective in A, without zero-weight graphs.
    if arm == "dt":
        return dt_loss
    total = dt_loss + factor * config.reference_weight * ref_loss
    if arm == "adaptive_pref":
        total = total + factor * config.preference_weight * pref_loss
    return total


class PairPool:
    """Reuse v3 contexts/strict indices, explicitly ignoring old model difficulty."""
    def __init__(self, dataset, path, target_return):
        with np.load(path, allow_pickle=False) as arrays:
            valid = arrays["valid_branch"].astype(bool)
            self.pairs = arrays["pairs"][valid].astype(np.int64)
            self.confidence = arrays["pair_confidence"][valid].astype(np.float64)
        if not len(self.pairs) or not np.isfinite(self.confidence).all():
            raise ValueError("Empty or nonfinite strict pair pool")
        self.base = np.maximum(self.confidence, 1e-8)
        self.base /= self.base.sum()
        # Only _context is reused: no old difficulty, hard_pair mask or priorities.
        self.trajectories = dataset.dataset
        self.seq_len = dataset.seq_len
        self.state_mean, self.state_std = dataset.state_mean, dataset.state_std
        self.reward_scale = dataset.reward_scale
        contexts, negatives = [], []
        for good, good_step, bad, bad_step in self.pairs:
            context = StateAlignedPreferenceDataset._context(self, good, good_step)
            contexts.append(context)
            negatives.append(self.trajectories[bad]["actions"][bad_step])
        self.tensors = [torch.from_numpy(np.stack(items)) for items in zip(*contexts)]
        self.tensors[2] = torch.full_like(
            self.tensors[2], target_return * self.reward_scale
        ) * self.tensors[4]
        self.tensors.append(torch.from_numpy(np.stack(negatives)))

    def batch(self, indices, device):
        indices = torch.as_tensor(indices, dtype=torch.long)
        return [value[indices].to(device) for value in self.tensors]

    @torch.no_grad()
    def score(self, model, config):
        violations, signed_margins, positive_errors = [], [], []
        with evaluating(model):
            for start in range(0, len(self.pairs), config.score_batch_size):
                indices = np.arange(start, min(start + config.score_batch_size,
                                              len(self.pairs)))
                states, actions, returns, times, mask, target, negative = self.batch(
                    indices, config.device
                )
                prediction = predict_action(model, states, actions, returns, times,
                                            mask, target)
                rows = torch.arange(len(target), device=config.device)
                positive = actions[rows, target]
                dplus = (prediction - positive).square().mean(-1)
                dminus = (prediction - negative).square().mean(-1)
                signed_margins.append((dminus - dplus).cpu().numpy())
                positive_errors.append(dplus.cpu().numpy())
                violations.append(F.relu(dplus - dminus + config.preference_margin)
                                  .cpu().numpy())
        violations = np.concatenate(violations)
        probabilities = sampling_probabilities(
            self.confidence, violations, config.preference_margin,
            config.coverage_fraction, config.violation_cap,
        )
        metrics = {
            "pool/active_ratio": float((violations > 0).mean()),
            "pool/active_count": int((violations > 0).sum()),
            "pool/mean_violation": float(violations.mean()),
            "pool/mean_signed_margin": float(np.concatenate(signed_margins).mean()),
            "pool/positive_mse": float(np.concatenate(positive_errors).mean()),
            "pool/sampling_ess": float(1.0 / np.square(probabilities).sum()),
            "pool/max_sampling_probability": float(probabilities.max()),
        }
        return probabilities, metrics


def tensor_rng_hash():
    digest = hashlib.sha256(torch.get_rng_state().numpy().tobytes())
    if torch.cuda.is_available():
        for state in torch.cuda.get_rng_state_all():
            digest.update(state.cpu().numpy().tobytes())
    return digest.hexdigest()


def write_json(path, value):
    temporary = str(path) + ".tmp"
    with open(temporary, "w") as stream:
        json.dump(value, stream, indent=2)
    os.replace(temporary, path)


def validate_config(config):
    if config.arm not in {"dt", "recent_ref", "adaptive_pref"}:
        raise ValueError("Unknown experiment arm")
    if config.reward_mode != "delayed" or not config.checkpoints_path:
        raise ValueError("Delayed rewards and a fresh output directory are required")
    if config.num_workers < 1:
        raise ValueError("Use dedicated loader workers to isolate DT sampling RNG")
    if not 0 < config.auxiliary_start < config.auxiliary_full < config.update_steps:
        raise ValueError("Require 0 < auxiliary_start < auxiliary_full < update_steps")
    intervals = (config.reference_every, config.rescore_every, config.log_every,
                 config.eval_every, config.score_batch_size,
                 config.preference_batch_size)
    if min(intervals) < 1 or config.preference_margin <= 0 or config.violation_cap <= 0:
        raise ValueError("Intervals, batch sizes, margin and cap must be positive")
    if not 0 < config.coverage_fraction <= 1 or config.min_active_pairs < 1:
        raise ValueError("Invalid coverage/normalization configuration")


@pyrallis.wrap()
def train(config: AdaptiveConfig):
    validate_config(config)
    output = Path(config.checkpoints_path)
    output.mkdir(parents=True, exist_ok=False)
    set_seed(config.train_seed, deterministic_torch=config.deterministic_torch)
    dataset = AuditedSequenceDataset(
        config.env_name, config.seq_len, config.reward_scale, config.reward_mode
    )
    pool = PairPool(dataset, config.pair_arrays_path, config.target_return)
    generator = torch.Generator().manual_seed(config.train_seed)
    loader = DataLoader(
        dataset, batch_size=config.batch_size, num_workers=config.num_workers,
        pin_memory=True, generator=generator,
    )
    environment = wrap_env(
        gym.make(config.env_name), dataset.state_mean, dataset.state_std,
        config.reward_scale,
    )
    wandb_init(asdict(config))
    set_seed(config.train_seed, deterministic_torch=config.deterministic_torch)
    model = DecisionTransformer(
        state_dim=environment.observation_space.shape[0],
        action_dim=environment.action_space.shape[0], embedding_dim=config.embedding_dim,
        seq_len=config.seq_len, episode_len=config.episode_len,
        num_layers=config.num_layers,
        num_heads=config.num_heads, attention_dropout=config.attention_dropout,
        residual_dropout=config.residual_dropout,
        embedding_dropout=config.embedding_dropout,
        max_action=config.max_action,
    ).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                 betas=config.betas, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min((step + 1) / config.warmup_steps, 1)
    )
    initial_hash = model_hash(model)
    reference = None
    reference_step = None
    probabilities = pool.base.copy()
    preference_rng = np.random.default_rng(config.pair_seed + 100_000)
    reference_rng = np.random.default_rng(config.pair_seed + 200_000)
    data_digest, reference_digest = hashlib.sha256(), hashlib.sha256()
    iterator = iter(loader)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    provenance = {
        **asdict(config), "initial_model_sha256": initial_hash,
        "pair_file_sha256": file_sha256(config.pair_arrays_path),
        "strict_pair_count": len(pool.pairs), "git_commit": revision,
        "source_sha256": file_sha256(__file__),
        "dt_source_sha256": file_sha256(str(Path(__file__).with_name("dt.py"))),
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
        "wandb_id": wandb.run.id, "wandb_url": wandb.run.url,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "dataset_statistics": dataset.stats,
    }
    write_json(output / "config.json", provenance)
    wandb.run.summary.update({"provenance/initial_model_sha256": initial_hash,
                              "provenance/pair_sha256": provenance["pair_file_sha256"],
                              "provenance/source_sha256": provenance["source_sha256"],
                              "provenance/git_commit": revision,
                              "provenance/strict_pair_count": len(pool.pairs)})
    print(f"RUN {json.dumps(provenance)}", flush=True)
    best = {str(target): {"score": -float("inf"), "step": None}
            for target in config.target_returns}
    evaluations, audit_rows, reference_events = [], [], []
    metric_file = open(output / "metrics.jsonl", "a", buffering=1)
    device = torch.device(config.device)
    cuda_devices = []
    if device.type == "cuda":
        cuda_devices = [device.index if device.index is not None
                        else torch.cuda.current_device()]

    def record(step, metrics):
        row = {"step": step, **metrics}
        metric_file.write(json.dumps(row, allow_nan=False) + "\n")
        wandb.log(metrics, step=step)

    def save(step):
        checkpoint = {
            "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(), "completed_updates": step,
            "next_step": step, "config": asdict(config), "rng_state": rng_state(),
            "loader_generator_state": generator.get_state(),
            "state_mean": dataset.state_mean, "state_std": dataset.state_std,
            "reference_model_state": (
                None if reference is None else reference.state_dict()
            ),
            "reference_step": reference_step, "sampling_probabilities": probabilities,
            "preference_rng_state": preference_rng.bit_generator.state,
            "reference_rng_state": reference_rng.bit_generator.state,
            "initial_model_sha256": initial_hash, "wandb_run_id": wandb.run.id,
            "dt_sample_stream_sha256": data_digest.hexdigest(),
            "reference_sample_stream_sha256": reference_digest.hexdigest(),
            "resume_note": "From-scratch-only entry point. Snapshots for reuse; "
            "loader worker states/prefetch queues are not saved. Not exact resume.",
        }
        path = output / f"step{step:06d}.pt"
        torch.save(checkpoint, str(path) + ".tmp")
        os.replace(str(path) + ".tmp", path)

    if 0 in config.checkpoint_steps:
        save(0)
    for step in range(1, config.update_steps + 1):
        completed = step - 1
        if refresh_due(completed, config.auxiliary_start, config.reference_every):
            reference = frozen_copy(model)
            reference_step = completed
            reference_events.append({"completed_updates": completed,
                                     "reference_sha256": model_hash(reference)})
            write_json(output / "reference_events.json", reference_events)
            print(f"Reference refreshed after {completed} updates", flush=True)
        if refresh_due(completed, config.auxiliary_start, config.rescore_every):
            probabilities, pool_metrics = pool.score(model, config)
            record(step, {**pool_metrics, "pool/scored_at_completed_updates": completed})
        batch = next(iterator)
        data_digest.update(batch[-1].numpy().tobytes())
        audit = (step <= 3 or step in {
            config.auxiliary_start, config.auxiliary_start + 1,
            config.auxiliary_start + 2, config.auxiliary_full,
            config.auxiliary_full + 1,
        } or step % config.eval_every == 0 or step == config.update_steps)
        audit_row = None
        if audit:
            audit_row = {"step": step, "dt_batch_sha256": batch_hash(batch[:-1]),
                         "torch_rng_sha256": tensor_rng_hash(),
                         "dt_sample_stream_sha256": data_digest.hexdigest()}
        states, actions, returns, times, mask = [
            item.to(config.device) for item in batch[:-1]
        ]
        prediction = model(states, actions, returns, times, ~mask.bool())
        errors = F.mse_loss(prediction, actions, reduction="none")
        dt_loss = (errors * mask.unsqueeze(-1)).mean()
        factor = ramp(completed, config.auxiliary_start, config.auxiliary_full)
        pref_loss = ref_loss = dt_loss.new_zeros(())
        metrics = {}
        if completed >= config.auxiliary_start:
            pref_indices = preference_rng.choice(
                len(pool.pairs), config.preference_batch_size, p=probabilities
            )
            ref_indices = reference_rng.choice(
                len(pool.pairs), config.preference_batch_size, p=pool.base
            )
            reference_digest.update(ref_indices.astype(np.int64).tobytes())
            # Auxiliary dropout cannot perturb ordinary DT's future RNG stream.
            with torch.random.fork_rng(devices=cuda_devices):
                preference_batch = pool.batch(pref_indices, config.device)
                ps, pa, pr, pt, pm, pi, negative = preference_batch
                with torch.set_grad_enabled(config.arm == "adaptive_pref"):
                    pred = predict_action(model, ps, pa, pr, pt, pm, pi)
                    positive = pa[torch.arange(len(pi), device=config.device), pi]
                    pref_loss, metrics = preference_loss(
                        pred, positive, negative, config.preference_margin,
                        config.min_active_pairs,
                    )
                rs, ra, rr, rt, rm, ri, _ = pool.batch(ref_indices, config.device)
                with evaluating(model), torch.set_grad_enabled(config.arm != "dt"):
                    anchor = predict_action(model, rs, ra, rr, rt, rm, ri)
                with torch.no_grad():
                    target = predict_action(reference, rs, ra, rr, rt, rm, ri)
                ref_loss = F.mse_loss(anchor, target)
            metrics.update({"reference_loss": ref_loss.detach().item(),
                            "reference_step": reference_step,
                            "reference_age": completed - reference_step,
                            "preference_unique_in_batch": len(np.unique(pref_indices))})
        loss = combined_loss(dt_loss, pref_loss, ref_loss, config.arm, factor, config)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite loss at {step}")
        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad,
                                                  error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        if audit:
            audit_row.update({"dt_loss": dt_loss.item(), "total_loss": loss.item(),
                              "reference_sample_stream_sha256":
                              reference_digest.hexdigest(),
                              "model_sha256": model_hash(model)})
            audit_rows.append(audit_row)
            write_json(output / "pairing_audit.json", audit_rows)
        if step == 1 or step % config.log_every == 0 or audit:
            metrics = {f"train/{key}": value for key, value in metrics.items()}
            metrics.update({
                "train_loss": dt_loss.item(), "train/total_loss": loss.item(),
                "train/completed_updates": step, "train/auxiliary_ramp": factor,
                "train/weighted_preference_loss": factor * config.preference_weight
                * pref_loss.detach().item() * (config.arm == "adaptive_pref"),
                "train/weighted_reference_loss": factor * config.reference_weight
                * ref_loss.detach().item() * (config.arm != "dt"),
                "train/grad_norm": grad_norm.item(),
                "learning_rate": scheduler.get_last_lr()[0],
            })
            record(step, metrics)
            if step == 1 or step % config.eval_every == 0:
                print(f"UPDATE {step} {json.dumps(metrics)}", flush=True)
        if step % config.eval_every == 0 or step == config.update_steps:
            saved_rng = rng_state()
            eval_metrics = {}
            with evaluating(model), torch.no_grad():
                for target_return in config.target_returns:
                    environment.seed(config.eval_seed)
                    values, lengths = [], []
                    for _ in range(config.eval_episodes):
                        value, length = eval_rollout(model, environment,
                                                    target_return * config.reward_scale,
                                                    config.device, config.reward_mode)
                        values.append(value / config.reward_scale)
                        lengths.append(length)
                    normalized = environment.get_normalized_score(
                        np.asarray(values)
                    ) * 100
                    score = float(normalized.mean())
                    prefix = f"eval/{target_return}"
                    eval_metrics.update({f"{prefix}_normalized_score_mean": score,
                                         f"{prefix}_normalized_score_std":
                                         float(normalized.std()),
                                         f"{prefix}_return_mean": float(np.mean(values)),
                                         f"{prefix}_length_mean":
                                         float(np.mean(lengths))})
                    if score > best[str(target_return)]["score"]:
                        best[str(target_return)] = {"score": score, "step": step}
                        wandb.run.summary[f"best/{target_return}_score"] = score
                        wandb.run.summary[f"best/{target_return}_step"] = step
            restore_rng(saved_rng)
            evaluations.append({"step": step, **eval_metrics})
            write_json(output / "evaluations.json", evaluations)
            record(step, eval_metrics)
            print(f"EVAL {step} {json.dumps(eval_metrics)}", flush=True)
        if step in config.checkpoint_steps or step == config.update_steps:
            save(step)
    result = {
        "completed_updates": config.update_steps, "best": best,
        "last": evaluations[-1],
        "last_five_steps": [x["step"] for x in evaluations[-5:]],
        "last_five_mean": {str(target): float(np.mean([
            row[f"eval/{target}_normalized_score_mean"] for row in evaluations[-5:]
        ])) for target in config.target_returns},
        "dt_sample_stream_sha256": data_digest.hexdigest(),
        "reference_sample_stream_sha256": reference_digest.hexdigest(),
        "final_model_sha256": model_hash(model), "wandb_url": wandb.run.url,
    }
    write_json(output / "summary.json", result)
    wandb.run.summary.update({"completed_updates": config.update_steps,
                              "pairing/dt_sample_stream_sha256": data_digest.hexdigest(),
                              "pairing/reference_sample_stream_sha256":
                              reference_digest.hexdigest()})
    print(f"COMPLETE {json.dumps(result)}", flush=True)
    metric_file.close()
    environment.close()
    wandb.finish()


if __name__ == "__main__":
    train()
