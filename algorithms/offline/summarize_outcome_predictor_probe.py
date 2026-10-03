"""Postprocess the fixed probe; never fit models or select hyperparameters."""
import argparse
import json
from pathlib import Path

import numpy as np
import outcome_predictor_probe as probe
import torch

@torch.no_grad()
def offline(root):
    models, norm = probe.load_predictors(root / "models")
    manifest = json.loads((root / "models/training_manifest.json").read_text())
    episodes, split, _ = probe.load_data(manifest["config"]["dataset"], 731)
    for ep in episodes:
        ep["x"] = probe.tokens(ep["states"], ep["actions"], norm)
    report = {}
    for split_name in ["validation", "test"]:
        ids = split[split_name]
        ys = np.asarray([episodes[i]["return"] for i in ids])
        part = {"episodes": len(ids), "return_variance": float(ys.var()),
                "constant_rmse": float(np.sqrt(np.mean((ys - norm["return_mean"]) ** 2)))}
        for kind, members in models.items():
            errors, phases = [], [[], [], []]
            for start in range(0, len(ids), 8):
                batch = ids[start:start + 8]
                x, y, mask = probe.padded_batch(episodes, batch, norm, "cpu")
                pred = torch.stack([m.mean(m(x)[0]) for m in members]).mean(0)
                err = (pred - y).square() * norm["return_std"] ** 2
                errors.extend(((err * mask).sum(1) / mask.sum(1)).tolist())
                for b, index in enumerate(batch):
                    n = len(episodes[index]["x"])
                    for phase in range(3):
                        phases[phase].append(float(err[b, n * phase // 3:n * (phase + 1) // 3].mean()))
            part[kind] = {"ensemble_rmse_raw": float(np.sqrt(np.mean(errors))),
                          "r2": float(1 - np.mean(errors) / ys.var()),
                          "phase_rmse_raw": [float(np.sqrt(np.mean(p))) for p in phases]}
        report[split_name] = part
    probe.dump(root / "offline_ensemble_summary.json", report)
    print(json.dumps(report, indent=2))


def merge(root):
    sources = [("heldout_ranking", 0, 5), ("heldout_shard_5_10", 5, 10),
               ("heldout_shard_10_15", 10, 15), ("heldout_shard_15_20", 15, 20)]
    records, baselines, hashes = [], [], None
    methods = {"baseline", "random", "mse_greedy", "mse_gated", "twohot_greedy", "twohot_gated"}
    for folder, start, stop in sources:
        path = root / folder
        protocol = json.loads((path / "protocol.json").read_text())
        if hashes is None:
            hashes = protocol["model_sha256"]
        assert hashes == protocol["model_sha256"], "Model hashes changed across shards"
        assert protocol["primary"] == "twohot_gated"
        assert protocol["gating_beta"] == 1 and protocol["gating_delta"] == 0
        assert protocol["args"]["eval_seed"] == 20261001
        assert protocol["args"]["target_return"] == 12000
        rows = [r for r in json.loads((path / "records.json").read_text())
                if start <= r["episode"] < stop]
        assert len(rows) == (stop - start) * 10 * len(methods), (folder, len(rows))
        bs = json.loads((path / "baseline_scores.json").read_text())[:stop - start]
        assert len(bs) == stop - start
        baselines.extend(bs)
        for ep in range(start, stop):
            local = [r for r in rows if r["episode"] == ep]
            assert len({r["step"] for r in local}) == 10
            for step in {r["step"] for r in local}:
                assert {r["method"] for r in local if r["step"] == step} == methods
            assert max(abs(r["baseline_score"] - bs[ep - start]) for r in local) < 1e-6
        records.extend(rows)
    keys = [(r["episode"], r["step"], r["method"]) for r in records]
    assert len(set(keys)) == len(keys) == 1200
    assert {r["episode"] for r in records} == set(range(20))
    summary = {}
    for method in sorted(methods):
        rows = [r for r in records if r["method"] == method]
        ds = np.asarray([r["delta"] for r in rows])
        vs_random = np.asarray([r["delta"] - r["random_delta"] for r in rows])
        summary[method] = {
            "mean_delta": float(ds.mean()),
            "ci95_delta": probe.ci_episode([(r["episode"], r["delta"]) for r in rows]),
            "mean_vs_random": float(vs_random.mean()),
            "ci95_vs_random": probe.ci_episode([(r["episode"], r["delta"] - r["random_delta"]) for r in rows]),
            "harm_rate": float(np.mean(ds < -.1)), "benefit_rate": float(np.mean(ds > .1)),
            "replacement_rate": float(np.mean([r["selected"] != 0 for r in rows])),
            "n_episodes": 20, "n_anchors": len(rows),
            "per_episode": {str(ep): float(np.mean([r["delta"] for r in rows if r["episode"] == ep]))
                            for ep in range(20)}}
    out = root / "heldout_merged"
    out.mkdir(exist_ok=True)
    probe.dump(out / "records.json", records)
    probe.dump(out / "summary.json", summary)
    replay_checks, restore_errors = [], []
    log_dir = root.parents[2] / "logs"
    for job, start, stop in [(8980, 0, 5), (8981, 5, 10), (8982, 10, 15), (8983, 15, 20)]:
        for line in (log_dir / f"outcome-probe-{job}.out").read_text().splitlines():
            try:
                row = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            if not isinstance(row, dict):
                continue
            if "replayed_prefix_episode" in row:
                ep = row["replayed_prefix_episode"]
                error = abs(row["baseline_score"] - baselines[ep])
                assert error < 1e-9, (job, ep, error)
                replay_checks.append(error)
            if "restore_error" in row and start <= row["episode"] < stop:
                restore_errors.append(row["restore_error"])
    assert len(replay_checks) == 30 and len(restore_errors) == 20
    assert max(restore_errors) < 1e-6
    # The original job finished one extra episode before cancellation. Its
    # independently repeated shard is an exact replay check, never extra data.
    originals = json.loads((root / "heldout_ranking/records.json").read_text())
    merged_by_key = {(r["episode"], r["step"], r["method"]): r for r in records}
    duplicates = [r for r in originals if r["episode"] >= 5]
    for row in duplicates:
        other = merged_by_key[row["episode"], row["step"], row["method"]]
        assert row["selected"] == other["selected"]
        assert abs(row["delta"] - other["delta"]) < 1e-9
    probe.dump(out / "audit.json", {"sources": sources, "model_sha256": hashes,
                                    "baseline_mean": float(np.mean(baselines)),
                                    "baseline_std": float(np.std(baselines)),
                                    "baseline_scores": baselines,
                                    "prefix_replay_checks": len(replay_checks),
                                    "prefix_replay_max_error": max(replay_checks),
                                    "branch_restore_max_error": max(restore_errors),
                                    "duplicate_record_replay_checks": len(duplicates),
                                    "all_200_anchors_complete": True,
                                    "record_count": len(records)})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["offline", "merge"])
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    (offline if args.mode == "offline" else merge)(args.root)
