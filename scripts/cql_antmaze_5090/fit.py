"""Cross-fit on trajectory totals, audit prediction, then generate frozen rewards."""

import argparse
import copy
import time
from pathlib import Path

import numpy as np
import torch

from algorithms.offline.shapley_redistribution import (
    permutation_shapley,
    random_masks,
    SegmentReturnModel,
    standardize,
)
from scripts.cql_antmaze_5090.data import conserved_rewards, segment_inputs
from scripts.cql_delayed.common import append, digest, read, verify, write

def fit_one(features, returns, split, settings, out, device):
    torch.manual_seed(split["training_seed"])
    rng = np.random.default_rng(split["training_seed"])
    vrng = np.random.default_rng(split["validation_seed"])
    normalized, stats = standardize(features, returns, split["train"])
    x = torch.from_numpy(segment_inputs(normalized)).to(device)
    y = torch.tensor((returns - stats["target_mean"]) / stats["target_std"],
                     dtype=torch.float32, device=device)
    model = SegmentReturnModel(x.shape[-1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"],
                                 weight_decay=settings["weight_decay"])
    valid = split["validation"]
    vm = torch.from_numpy(random_masks(vrng, len(valid) * settings["validation_masks"],
                                      20)).to(device)
    vx = x[valid].repeat_interleave(settings["validation_masks"], dim=0)
    vy = y[valid].repeat_interleave(settings["validation_masks"])
    best, best_state, best_epoch = float("inf"), None, 0
    for epoch in range(1, settings["epochs"] + 1):
        model.train()
        training = rng.permutation(split["train"])
        for start in range(0, len(training), settings["batch"]):
            indices = training[start:start + settings["batch"]]
            mask = torch.from_numpy(random_masks(rng, len(indices), 20)).to(device)
            encoded = model.encoder(x[indices])
            full = model.value(encoded, torch.ones_like(mask))
            partial = model.value(encoded, mask)
            loss = 0.5 * ((full - y[indices]) ** 2).mean()
            loss += 0.5 * ((partial - y[indices]) ** 2).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite predictor training loss")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            full_loss = ((model(x[valid], torch.ones(len(valid), 20, device=device))
                          - y[valid]) ** 2).mean()
            masked_loss = ((model(vx, vm) - vy) ** 2).mean()
            validation = float((full_loss + masked_loss) * 0.5)
        if not np.isfinite(validation):
            raise FloatingPointError("Non-finite validation loss")
        append(out / "epochs.jsonl", dict(epoch=epoch, validation=validation,
                                          validation_full=float(full_loss),
                                          validation_masked=float(masked_loss)))
        if validation < best:
            best, best_epoch = validation, epoch
            best_state = copy.deepcopy(model.state_dict())
        if epoch - best_epoch >= settings["patience"]:
            break
    model.load_state_dict(best_state)
    model.eval()
    torch.save(dict(model=best_state, stats=stats, split=split,
                    best_epoch=best_epoch, validation=best, settings=settings),
               out / "model.pt")
    write(out / "fit.json", dict(best_epoch=best_epoch, epochs=epoch,
                                 validation=best, target_mean=stats["target_mean"],
                                 target_std=stats["target_std"]))
    return model, x, stats


def run(root, device):
    p = verify(root)
    for name, expected in p["preparation_sha256"].items():
        if digest(root / name) != expected:
            raise ValueError(f"Prepared input changed: {name}")
    out = root / "predictor" / "fit"
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    write(out / "status.json", dict(status="fitting", device=device))
    with np.load(root / "predictor/input.npz") as data:
        features, returns = data["features"], data["returns"]
    n, horizon = features.shape[:2]
    predictions, baselines = np.zeros(n), np.zeros(n)
    contributions = np.zeros((n, 20))
    rewards = np.zeros((n, horizon), dtype=np.float32)
    diagnostics = []
    for split in read(root / "predictor/folds.json"):
        work = out / f"fold_{split['fold']}"
        work.mkdir()
        model, x, stats = fit_one(
            features, returns, split, p["attribution"], work, device)
        with torch.no_grad():
            for index in split["held"]:
                encoded = model.encoder(x[index:index + 1])

                def value(masks):
                    result = []
                    for offset in range(0, len(masks), 512):
                        mask = torch.from_numpy(masks[offset:offset + 512]).to(device)
                        v = model.value(encoded, mask).cpu().numpy()
                        result.append(v.astype(np.float64) * stats["target_std"]
                                      + stats["target_mean"])
                    return np.concatenate(result)

                seed = p["attribution"]["attribution_seed_base"] + index
                rng = np.random.default_rng(seed)
                permutations = np.stack([rng.permutation(20) for _ in range(128)])
                samples, empty, full = permutation_shapley(value, 20, permutations)
                phi = samples.mean(0)
                np.testing.assert_allclose(phi.sum(), full - empty, atol=1e-7, rtol=1e-7)
                redistributed, correction, rounding = conserved_rewards(
                    phi, returns[index], horizon)
                predictions[index], baselines[index] = full, empty
                contributions[index], rewards[index] = phi, redistributed
                disagreement = np.abs(samples[:64].mean(0) - samples[64:].mean(0)).sum()
                stability = float(disagreement / max(np.abs(phi).sum(), 1e-8))
                standard_error = float(samples.std(0, ddof=1).max() / np.sqrt(128))
                diagnostics.append(dict(trajectory=index, fold=split["fold"],
                                        attribution_seed=seed, empty=empty, full=full,
                                        prediction_error=float(full - returns[index]),
                                        uniform_correction=correction,
                                        rounding_correction=rounding,
                                        split_half_relative_l1=stability,
                                        max_mc_standard_error=standard_error))
                np.savez(work / f"trajectory_{index:03d}.npz",
                         permutation_contributions=samples, permutations=permutations)
        elapsed = time.monotonic() - started
        print(f"fold {split['fold']} completed, elapsed={elapsed:.1f}s",
              flush=True)
    mse = float(np.mean((predictions - returns) ** 2))
    baseline_mse = float(np.mean((baselines - returns) ** 2))
    improvement = 1 - mse / baseline_mse if baseline_mse else None
    gate = dict(passed=bool(np.isfinite(mse) and mse < baseline_mse),
                model_mse=mse, training_mean_mse=baseline_mse,
                relative_mse_improvement=improvement,
                elapsed_seconds=time.monotonic() - started)
    write(out / "gate.json", gate)
    diagnostics.sort(key=lambda row: row["trajectory"])
    write(out / "diagnostics.json", diagnostics)
    np.savez(out / "attribution.npz",
             contributions=contributions, predictions=predictions,
             baselines=baselines, returns=returns, rewards=rewards)
    errors = rewards.astype(np.float64).sum(1) - returns
    if np.max(np.abs(errors)) >= 1e-3:
        raise ValueError("Shapley total conservation failed")
    write(out / "audit.json", dict(
        trajectories=n, transitions=n * horizon, trajectory_length=horizon,
        reward_mode="shapley20x128-timeout-v1", gate=gate,
        original_dense_rewards_used=False, original_success_terminals_used=False,
        max_return_error=float(np.abs(errors).max()),
        negative_reward_fraction=float((rewards < 0).mean()),
        reward_min=float(rewards.min()), reward_max=float(rewards.max())))
    status = "completed" if gate["passed"] else "gate_failed"
    write(out / "status.json", dict(status=status, **gate))
    print(gate, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(2)
    run(args.root.resolve(), args.device)
