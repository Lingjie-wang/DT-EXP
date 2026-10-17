"""Capture and replay a C nonfinite gradient without updating a bad model."""

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from dt import DecisionTransformer, SequenceDataset, set_seed
from state_only_preference import mine_pairs, preference_batch, single_sided_loss
from top_return_weighted_dt import restore_rng, rng_state
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.utils.data import DataLoader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--updates", type=int, default=10000)
    args = parser.parse_args()
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=False)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]
    set_seed(cfg["train_seed"])
    dataset = SequenceDataset(cfg["env_name"], cfg["seq_len"], cfg["reward_scale"],
                              cfg["reward_mode"])
    pairs, _, _ = mine_pairs(dataset.dataset, dataset.state_mean, dataset.state_std,
                             cfg["state_max_rmse"])
    generator = torch.Generator()
    generator.set_state(checkpoint["loader_generator_state"])
    loader = DataLoader(dataset, batch_size=cfg["batch_size"],
                        num_workers=cfg["num_workers"], pin_memory=True,
                        generator=generator)
    pair_rng = np.random.RandomState()
    pair_rng.set_state(checkpoint["pair_rng_state"])
    model = DecisionTransformer(
        state_dim=dataset.state_mean.shape[-1],
        action_dim=dataset.dataset[0]["actions"].shape[-1],
        **{k: cfg[k] for k in ("embedding_dim", "seq_len", "episode_len", "num_layers",
                              "num_heads", "attention_dropout", "residual_dropout",
                              "embedding_dropout", "max_action")},
    ).cuda()
    model.load_state_dict(checkpoint["model_state"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"],
                                 betas=cfg["betas"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda n: min((n + 1) / cfg["warmup_steps"], 1))
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scheduler.load_state_dict(checkpoint["scheduler_state"])
    restore_rng(checkpoint["rng_state"])
    iterator = iter(loader)

    def backward(batch, auxiliary, part="both"):
        optimizer.zero_grad(set_to_none=True)
        losses = []
        if part in ("both", "dt"):
            states, actions, returns, times, mask = batch
            prediction = model(states, actions, returns, times, ~mask.bool())
            losses.append((F.mse_loss(prediction, actions, reduction="none")
                           * mask.unsqueeze(-1)).mean())
        if part in ("both", "preference"):
            ps, pa, pr, pt, pm, index, negative = auxiliary
            output = model(ps, pa, pr, pt, ~pm.bool())
            row = torch.arange(len(index), device="cuda")
            pref, _, _, _ = single_sided_loss(output[row, index], pa[row, index],
                                             negative, cfg["preference_margin"])
            losses.append(cfg["preference_weight"] * pref)
        loss = sum(losses)
        loss.backward()
        return float(loss.detach())

    def report():
        bad = [name for name, p in model.named_parameters()
               if p.grad is not None and not torch.isfinite(p.grad).all()]
        return {"nonfinite_parameters": bad,
                "gradient_norm_float64": float(torch.stack([
                    p.grad.double().norm() for p in model.parameters()
                    if p.grad is not None]).norm())}

    for offset in range(1, args.updates + 1):
        step = checkpoint["completed_updates"] + offset
        batch = [t.cuda() for t in next(iterator)]
        indices = pair_rng.randint(len(pairs), size=cfg["preference_batch_size"])
        auxiliary = [torch.from_numpy(t).cuda() for t in
                     preference_batch(dataset, pairs, indices, 12000.0)]
        before = rng_state()
        loss = backward(batch, auxiliary)
        try:
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip_grad"],
                                                  error_if_nonfinite=True)
        except RuntimeError:
            result = {"step": step, "loss": loss, "initial": report()}
            torch.save({"model_state": model.state_dict(), "batch": batch,
                        "auxiliary": auxiliary, "rng_state": before,
                        "config": cfg, "step": step}, root / "failure.pt")
            for backend in ("auto", "math"):
                for part in ("both", "dt", "preference"):
                    restore_rng(before)
                    context = (sdpa_kernel(SDPBackend.MATH) if backend == "math"
                               else nullcontext())
                    with context:
                        value = backward(batch, auxiliary, part)
                    result[f"{backend}/{part}"] = {"loss": value, **report()}
            # Isolate backend arithmetic from different SDPA dropout draws.
            # This modifies only the diagnostic model, after saving failure.pt.
            for module in model.modules():
                if isinstance(module, torch.nn.Dropout):
                    module.p = 0
                if isinstance(module, torch.nn.MultiheadAttention):
                    module.dropout = 0
            for label, backend in (("efficient", SDPBackend.EFFICIENT_ATTENTION),
                                   ("math", SDPBackend.MATH)):
                with sdpa_kernel(backend):
                    value = backward(batch, auxiliary, "dt")
                result[f"dropout_zero/{label}"] = {"loss": value, **report()}
            (root / "diagnosis.json").write_text(json.dumps(result, indent=2) + "\n")
            print("DIAGNOSIS " + json.dumps(result), flush=True)
            return
        optimizer.step()
        scheduler.step()
        if offset == 1 or offset % 100 == 0:
            print(f"UPDATE {step} loss={loss} grad_norm={float(norm)}", flush=True)
    print("NO_FAILURE_REPRODUCED", flush=True)


if __name__ == "__main__":
    main()
