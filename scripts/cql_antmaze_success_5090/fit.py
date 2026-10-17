"""Ragged, group-isolated cross fitting for predictive segment Shapley rewards."""

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from algorithms.offline.shapley_redistribution import (
    permutation_shapley,
    SegmentReturnModel,
)
from scripts.cql_antmaze_official_5090.common import read, write
from scripts.cql_antmaze_success_5090.data import (
    conserved,
    grouped_folds,
    inputs,
    lengths,
    masks,
)

SETTINGS = dict(segments=20, permutations=128, folds=5, learning_rate=1e-3,
                weight_decay=1e-4, batch=32, epochs=300, patience=30,
                validation_masks=8, attribution_seed_base=3000)


def fit_one(features, part, split, settings, out, device):
    torch.manual_seed(split["training_seed"])
    rng = np.random.default_rng(split["training_seed"])
    vrng = np.random.default_rng(split["validation_seed"])
    x, valid, stats = inputs(features, part["offsets"], part["returns"], split["train"])
    targets = ((part["returns"] - stats["target_mean"]) / stats["target_std"]
               ).astype(np.float32)
    model = SegmentReturnModel(x.shape[-1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"],
                                 weight_decay=settings["weight_decay"])
    validation = np.asarray(split["validation"])
    vm = masks(vrng, np.repeat(valid[validation], settings["validation_masks"], 0))
    vm = vm.reshape(len(validation), settings["validation_masks"], 20)
    best, best_epoch, best_state = float("inf"), 0, None
    batch = settings["batch"]

    def tensors(ids):
        return [torch.from_numpy(a[ids]).to(device) for a in [x, valid, targets]]

    for epoch in range(1, settings["epochs"] + 1):
        model.train()
        training = rng.permutation(split["train"])
        for start in range(0, len(training), batch):
            ids = training[start:start + batch]
            bx, full_mask, by = tensors(ids)
            mask = torch.from_numpy(masks(rng, valid[ids])).to(device)
            encoded = model.encoder(bx)
            full = model.value(encoded, full_mask)
            partial = model.value(encoded, mask)
            loss = 0.5 * ((full - by) ** 2 + (partial - by) ** 2).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        full_sum, partial_sum = 0.0, 0.0
        with torch.no_grad():
            for start in range(0, len(validation), batch):
                ids = validation[start:start + batch]
                bx, full_mask, by = tensors(ids)
                encoded = model.encoder(bx)
                full = model.value(encoded, full_mask)
                mask = torch.from_numpy(vm[start:start + batch]).to(device)
                partial = model.value(encoded[:, None], mask)
                full_sum += float(((full - by) ** 2).sum())
                partial_sum += float(((partial - by[:, None]) ** 2).mean(1).sum())
        score = 0.5 * (full_sum + partial_sum) / len(validation)
        if not np.isfinite(score):
            raise FloatingPointError("Non-finite validation loss")
        with (out / "epochs.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(epoch=epoch, validation=score,
                         full=full_sum / len(validation),
                         masked=partial_sum / len(validation))) + "\n")
        if score < best:
            best, best_epoch = score, epoch
            best_state = copy.deepcopy(model.state_dict())
        if epoch % 10 == 0:
            print(f"fold={split['fold']} epoch={epoch} validation={score:.6f}",
                  flush=True)
        if epoch - best_epoch >= settings["patience"]:
            break
    model.load_state_dict(best_state)
    model.eval()
    torch.save(dict(model=best_state, stats=stats, split=split, settings=settings,
                    best_epoch=best_epoch, validation=best), out / "model.pt")
    write(out / "fit.json", dict(best_epoch=best_epoch, epochs=epoch,
                                 validation=best, target_mean=stats["target_mean"],
                                 target_std=stats["target_std"]))
    return model, x, valid, stats


def fit(features, part, out, device="cuda", settings=None):
    settings = dict(SETTINGS if settings is None else settings)
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    splits = list(grouped_folds(part["groups"], settings["folds"]))
    write(out / "folds.json", splits)
    n = len(part["returns"])
    predictions, baselines = np.zeros(n), np.zeros(n)
    contributions = np.zeros((n, 20))
    rewards = np.empty(len(features), dtype=np.float32)
    diagnostics = []
    count = settings["permutations"]
    for split in splits:
        work = out / f"fold_{split['fold']}"
        work.mkdir()
        write(out / "status.json", dict(stage="fitting", fold=split["fold"]))
        model, x, valid, stats = fit_one(features, part, split, settings, work, device)
        held = split["held"]
        samples_all = np.zeros((len(held), count, 20), dtype=np.float32)
        perms_all = np.full((len(held), count, 20), 255, dtype=np.uint8)
        with torch.no_grad():
            for row, index in enumerate(held):
                start, end = part["offsets"][index:index + 2]
                k = len(lengths(end - start))
                encoded = model.encoder(torch.from_numpy(x[index:index + 1]).to(device))

                def value(coalitions):
                    result = []
                    for offset in range(0, len(coalitions), 512):
                        mask = np.zeros((len(coalitions[offset:offset + 512]), 20),
                                        dtype=np.float32)
                        mask[:, :k] = coalitions[offset:offset + 512]
                        v = model.value(encoded, torch.from_numpy(mask).to(device))
                        result.append(v.cpu().numpy().astype(np.float64)
                                      * stats["target_std"] + stats["target_mean"])
                    return np.concatenate(result)

                rng = np.random.default_rng(settings["attribution_seed_base"] + index)
                # All one-player permutations are identical; evaluate just once.
                permutations = np.stack([rng.permutation(k)
                                         for _ in range(1 if k == 1 else count)])
                samples, empty, full = permutation_shapley(value, k, permutations)
                phi = samples.mean(0)
                np.testing.assert_allclose(phi.sum(), full - empty, atol=1e-7, rtol=1e-7)
                redistributed, correction, rounding = conserved(
                    phi, part["returns"][index], end - start)
                predictions[index], baselines[index] = full, empty
                contributions[index, :k], rewards[start:end] = phi, redistributed
                samples_all[row, :, :k] = samples
                perms_all[row, :, :k] = permutations
                half = len(samples) // 2
                disagreement = (float(np.abs(samples[:half].mean(0)
                                             - samples[half:].mean(0)).sum())
                                if half else 0.0)
                diagnostics.append(dict(fragment=index, fold=split["fold"],
                    players=k, empty=empty, full=full, uniform_correction=correction,
                    rounding_correction=rounding,
                    split_half_relative_l1=disagreement / max(np.abs(phi).sum(), 1e-8)))
        np.savez_compressed(work / "permutations.npz", held=held,
                            permutations=perms_all, contributions=samples_all)
        model = None
        del x, valid, samples_all, perms_all
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        print(f"fold={split['fold']} complete elapsed={time.monotonic()-started:.1f}s",
              flush=True)
    errors = np.add.reduceat(rewards.astype(np.float64), part["offsets"][:-1])
    errors -= part["returns"]
    if not np.isfinite(rewards).all() or np.max(np.abs(errors)) >= 1e-4:
        raise ValueError("Shapley conservation failed")
    mse = float(np.mean((predictions - part["returns"]) ** 2))
    baseline_mse = float(np.mean((baselines - part["returns"]) ** 2))
    gate = dict(passed=mse < baseline_mse, model_mse=mse,
                training_mean_mse=baseline_mse, policy="report_only",
                elapsed_seconds=time.monotonic() - started)
    write(out / "gate.json", gate)
    write(out / "diagnostics.json", sorted(diagnostics, key=lambda r: r["fragment"]))
    np.savez_compressed(out / "attribution.npz", contributions=contributions,
                        predictions=predictions, baselines=baselines,
                        returns=part["returns"], rewards=rewards)
    write(out / "status.json", dict(stage="completed", gate=gate,
          max_return_error=float(np.abs(errors).max()),
          reward_min=float(rewards.min()), reward_max=float(rewards.max()),
          negative_reward_fraction=float((rewards < 0).mean())))
    return rewards


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    plan = read(root / "plan.json")
    job = next(j for j in plan["jobs"] if j["dataset_key"] == args.dataset)
    # Run in a child process so its CUDA context exits before CQL admission.
    from scripts.cql_antmaze_success_5090.materialize import load_raw
    raw = load_raw(root / "datasets" / job["dataset"])
    work = root / "predictor" / args.dataset
    with np.load(work / "partition.npz") as stored:
        part = {key: stored[key] for key in stored.files}
    features = np.concatenate([raw["observations"][part["index"]],
                               raw["actions"][part["index"]]], 1).astype(np.float32)
    del raw
    torch.set_num_threads(2)
    fit(features, part, work / "fit", settings=plan["attribution"])
