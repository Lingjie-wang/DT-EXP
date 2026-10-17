"""Audited parent-state transfer and matched late-preference objectives."""

import hashlib

import numpy as np
import torch
from matched_positive import positive_imitation_loss
from state_only_preference import single_sided_loss

def fingerprint(value):
    """Device-independent exact hash of nested tensor/array state."""
    digest = hashlib.sha256()

    def add(item):
        if isinstance(item, torch.Tensor):
            item = item.detach().cpu().contiguous().numpy()
        if isinstance(item, np.ndarray):
            digest.update(str((str(item.dtype), item.shape)).encode())
            digest.update(item.tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                add(key)
                add(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode())
            for child in item:
                add(child)
        else:
            digest.update((type(item).__name__ + ':' + repr(item) + ';').encode())

    add(value)
    return digest.hexdigest()


def auxiliary_loss(variant, prediction, positive, negative, margin):
    if variant not in {"dt", "b", "c"}:
        raise ValueError("Unknown arm")
    objective = positive_imitation_loss if variant == "b" else single_sided_loss
    return objective(prediction, positive, negative, margin)


def validate_parent_config(current, saved, completed):
    if (completed != 100000 or saved.get("update_steps") != 100000
            or saved.get("variant") != "dt" or saved.get("preference_weight") != 0
            or saved.get("reward_mode") != "original"
            or saved.get("resume_checkpoint")):
        raise ValueError("Expected a completed from-scratch 100k original-reward DT")
    allowed = {"name", "group", "project", "wandb_entity", "wandb_mode", "output_dir",
               "checkpoints_path", "variant", "preference_weight", "learning_rate",
               "update_steps", "eval_every", "log_every", "checkpoint_every"}
    if current.get("validation_mode"):
        allowed |= {"device", "batch_size", "preference_batch_size", "num_workers",
                    "eval_episodes"}
    for key, expected in saved.items():
        actual = current.get(key)
        if isinstance(expected, (tuple, list)) and isinstance(actual, (tuple, list)):
            expected, actual = tuple(expected), tuple(actual)
        if key not in allowed and actual != expected:
            raise ValueError(f"Parent setting differs: {key}")


def restore_parent(model, optimizer, scheduler, saved, learning_rate):
    """Restore moments and step counts, then apply the same lower LR to every arm."""
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Invalid fine-tuning learning rate")
    model.load_state_dict(saved["model_state"])
    optimizer.load_state_dict(saved["optimizer_state"])
    scheduler.load_state_dict(saved["scheduler_state"])
    result = {}
    for label, actual, expected in (
        ("model", model.state_dict(), saved["model_state"]),
        ("optimizer", optimizer.state_dict(), saved["optimizer_state"]),
        ("scheduler", scheduler.state_dict(), saved["scheduler_state"]),
    ):
        if fingerprint(actual) != fingerprint(expected):
            raise ValueError(f"Parent {label} state was not exactly restored")
        result["restored_" + label + "_sha256"] = fingerprint(actual)
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError("Nonfinite parent model")
    for state in optimizer.state.values():
        if not all(torch.isfinite(v).all() for v in state.values()
                   if isinstance(v, torch.Tensor)):
            raise ValueError("Nonfinite parent optimizer")
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
        group["initial_lr"] = learning_rate
    # LambdaLR is already beyond the parent's warmup; retain its 100k step count.
    scheduler.base_lrs = [learning_rate] * len(optimizer.param_groups)
    scheduler._last_lr = [learning_rate] * len(optimizer.param_groups)
    result["child_optimizer_sha256"] = fingerprint(optimizer.state_dict())
    result["child_scheduler_sha256"] = fingerprint(scheduler.state_dict())
    return result


def verify_parent_provenance(parent, child):
    for key in ("dataset_sha256", "pairs_sha256", "packages", "attention_backend"):
        if parent[key] != child[key]:
            raise ValueError(f"Parent provenance differs: {key}")
    for filename in ("dt.py", "dense_action_preference.py",
                     "state_only_preference.py", "top_return_weighted_dt.py"):
        key = "algorithms/offline/" + filename
        if parent["source_sha256"][key] != child["source_sha256"][key]:
            raise ValueError(f"Parent shared source differs: {filename}")
