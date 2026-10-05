"""Positive-only objective and strict matching to a completed dense reference."""

import json
from pathlib import Path

import torch
import torch.nn.functional as F

def positive_imitation_loss(prediction, positive, negative, margin=0.05):
    """All sampled positives contribute; negative actions are diagnostic only.

    Keep the C call signature and its one auxiliary forward/RNG schedule.
    The margin is intentionally unused by B; it is retained for C diagnostics.
    """
    d_positive = F.mse_loss(prediction, positive.detach(), reduction="none").mean(-1)
    d_negative = F.mse_loss(prediction, negative.detach(), reduction="none").mean(-1)
    active = torch.ones_like(d_positive, dtype=torch.bool)
    return d_positive.mean(), d_positive, d_negative, active


def verify_reference(config, provenance, reference_dir):
    """Abort before policy updates if B differs from its declared dense control."""
    root = Path(reference_dir).resolve()
    reference = json.loads((root / "config.json").read_text())
    previous = json.loads((root / "provenance.json").read_text())
    status = json.loads((root / "status.json").read_text())
    if (status["state"] != "completed"
            or status["completed_updates"] != reference["update_steps"]):
        raise ValueError("Reference training is incomplete")
    if reference.get("variant") not in {"c", "dt"}:
        raise ValueError("Expected an original dense C or DT reference")
    operational = {"name", "group", "project", "wandb_entity", "wandb_mode",
                   "output_dir", "checkpoints_path", "variant", "preference_weight"}
    for key, expected in reference.items():
        actual = config.get(key)
        if isinstance(expected, (list, tuple)) and isinstance(actual, (list, tuple)):
            expected, actual = tuple(expected), tuple(actual)
        if key not in operational and actual != expected:
            raise ValueError(f"Reference training setting differs: {key}")
    for key in ("initial_model_sha256", "dataset_sha256", "pairs_sha256",
                "packages", "attention_backend"):
        if previous[key] != provenance[key]:
            raise ValueError(f"Reference provenance differs: {key}")
    for filename in ("dt.py", "state_only_preference.py", "dense_action_preference.py",
                     "top_return_weighted_dt.py"):
        key = "algorithms/offline/" + filename
        if previous["source_sha256"][key] != provenance["source_sha256"][key]:
            raise ValueError(f"Reference shared source differs: {filename}")
    return {"reference_dir": str(root), "reference_git_commit": previous["git_commit"],
            "reference_run": json.loads((root / "wandb_run.json").read_text()),
            "initial_model_sha256": provenance["initial_model_sha256"],
            "dataset_sha256": provenance["dataset_sha256"],
            "pairs_sha256": provenance["pairs_sha256"],
            "training_settings_match": True, "shared_sources_match": True,
            "objective": (f"DT_MSE + {config['preference_weight']} "
                          "* mean(positive_action_MSE)"),
            "negative_action_gradient": False, "all_sampled_positives_included": True}
