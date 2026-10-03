"""Offline-only return prediction and privileged DT branch ranking diagnostic.

No branch outcomes, intermediate rewards, or RTGs enter predictor training.
The recurrent input at time t is (s[t-1], a[t-1], s[t], t/1000, valid).
At time zero both states are s[0], action is zero, and valid is zero.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn

def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str) + "\n")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Predictor(nn.Module):
    def __init__(self, input_dim, hidden=128, kind="twohot", bins=101):
        super().__init__()
        self.kind = kind
        self.embed = nn.Sequential(nn.Linear(input_dim, hidden), nn.SiLU())
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(),
                                  nn.Linear(hidden, bins if kind == "twohot" else 1))
        self.register_buffer("bins", torch.linspace(-4.0, 4.0, bins))

    def forward(self, x, hidden=None):
        z, hidden = self.gru(self.embed(x), hidden)
        return self.head(z), hidden

    def mean(self, logits):
        if self.kind == "mse":
            return logits.squeeze(-1)
        return (logits.softmax(-1) * self.bins).sum(-1)

    def loss(self, logits, target):
        if self.kind == "mse":
            return (self.mean(logits) - target).square()
        position = ((target - self.bins[0]) / (self.bins[-1] - self.bins[0])
                    * (len(self.bins) - 1)).clamp(0, len(self.bins) - 1)
        lo = position.floor().long()
        hi = (lo + 1).clamp(max=len(self.bins) - 1)
        weight = position - lo
        logp = logits.log_softmax(-1)
        return -(1 - weight) * logp.gather(-1, lo.unsqueeze(-1)).squeeze(-1) - (
            weight * logp.gather(-1, hi.unsqueeze(-1)).squeeze(-1))


def tokens(states, actions, norm):
    """s0 token followed by completed transitions; rewards never used here."""
    s = (np.asarray(states, np.float32) - norm["state_mean"]) / norm["state_std"]
    n, sd = s.shape
    ad = len(norm["action_mean"])
    result = np.zeros((n, 2 * sd + ad + 2), np.float32)
    result[:, :sd] = np.concatenate((s[:1], s[:-1]))
    result[:, sd + ad:2 * sd + ad] = s
    if n > 1:
        result[1:, sd:sd + ad] = (np.asarray(actions[:n - 1], np.float32)
                                 - norm["action_mean"]) / norm["action_std"]
        result[1:, -1] = 1
    result[:, -2] = np.arange(n, dtype=np.float32) / 1000.0
    return result


def candidate_tokens(state, actions, next_states, step, norm):
    s = (np.asarray(state) - norm["state_mean"]) / norm["state_std"]
    sn = (np.asarray(next_states) - norm["state_mean"]) / norm["state_std"]
    a = (np.asarray(actions) - norm["action_mean"]) / norm["action_std"]
    return np.concatenate((np.broadcast_to(s, sn.shape), a, sn,
                           np.full((len(a), 1), (step + 1) / 1000.0),
                           np.ones((len(a), 1))), -1).astype(np.float32)


def load_data(path, split_seed=731):
    with h5py.File(path, "r") as f:
        obs, act = f["observations"][:], f["actions"][:]
        rew = f["rewards"][:]
        ends = np.flatnonzero(f["terminals"][:] | f["timeouts"][:])
    episodes, start = [], 0
    for end in ends:
        # Last observed s is before final reward is revealed; omit terminal transition.
        episodes.append({"states": obs[start:end + 1].copy(),
                         "actions": act[start:end].copy(),
                         "return": float(rew[start:end + 1].astype(np.float64).sum()),
                         "start": int(start), "end": int(end)})
        start = end + 1
    order = np.random.default_rng(split_seed).permutation(len(episodes))
    n = len(order)
    split = {"train": order[:int(.7 * n)].tolist(),
             "validation": order[int(.7 * n):int(.85 * n)].tolist(),
             "test": order[int(.85 * n):].tolist()}
    train_obs = np.concatenate([episodes[i]["states"] for i in split["train"]])
    train_act = np.concatenate([episodes[i]["actions"] for i in split["train"]])
    train_y = np.asarray([episodes[i]["return"] for i in split["train"]])
    norm = {"state_mean": train_obs.mean(0), "state_std": train_obs.std(0) + 1e-6,
            "action_mean": train_act.mean(0), "action_std": train_act.std(0) + 1e-6,
            "return_mean": float(train_y.mean()), "return_std": float(train_y.std())}
    return episodes, split, norm


def padded_batch(episodes, indices, norm, device):
    seqs = [episodes[i]["x"] for i in indices]
    max_len = max(len(x) for x in seqs)
    x = np.zeros((len(seqs), max_len, seqs[0].shape[-1]), np.float32)
    mask = np.zeros((len(seqs), max_len), np.float32)
    y = np.empty((len(seqs), max_len), np.float32)
    for b, (seq, i) in enumerate(zip(seqs, indices)):
        x[b, :len(seq)] = seq
        mask[b, :len(seq)] = 1.0
        y[b] = (episodes[i]["return"] - norm["return_mean"]) / norm["return_std"]
    return tuple(torch.as_tensor(z, device=device) for z in (x, y, mask))


@torch.no_grad()
def validate(model, episodes, indices, norm, device, batch_size):
    model.eval()
    errors, phase = [], [[], [], []]
    for start in range(0, len(indices), batch_size):
        ids = indices[start:start + batch_size]
        x, y, mask = padded_batch(episodes, ids, norm, device)
        logits, _ = model(x)
        err = (model.mean(logits) - y).square()
        errors.extend(((err * mask).sum(1) / mask.sum(1)).cpu().tolist())
        for b, i in enumerate(ids):
            for p in range(3):
                left = int(len(episodes[i]["x"]) * p / 3)
                right = int(len(episodes[i]["x"]) * (p + 1) / 3)
                phase[p].append(float(err[b, left:right].mean()))
    return {"rmse_raw": float(np.sqrt(np.mean(errors)) * norm["return_std"]),
            "phase_rmse_raw": [float(np.sqrt(np.mean(e)) * norm["return_std"])
                               for e in phase]}


def train(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    seed_all(args.split_seed)
    episodes, split, norm = load_data(args.dataset, args.split_seed)
    for ep in episodes:
        ep["x"] = tokens(ep["states"], ep["actions"], norm)
    manifest = {"split_seed": args.split_seed, "split": split,
                "trajectory_bounds": [[e["start"], e["end"]] for e in episodes],
                "return_stats": {k: {"n": len(ids),
                                     "min": min(episodes[i]["return"] for i in ids),
                                     "max": max(episodes[i]["return"] for i in ids)}
                                 for k, ids in split.items()},
                "train_return_mean": norm["return_mean"],
                "train_return_std": norm["return_std"],
                "config": vars(args), "no_branch_data_in_training": True}
    dump(out / "training_manifest.json", manifest)
    torch.save({"norm": norm, "split": split}, out / "preprocessing.pt")
    print(json.dumps(manifest["return_stats"]), flush=True)
    reports = []
    for kind in ("mse", "twohot"):
        for member in range(args.members):
            path = out / f"{kind}_{member}.pt"
            if path.exists():
                print(f"Existing checkpoint: {path}; not overwriting", flush=True)
                continue
            member_seed = 10000 + member
            seed_all(member_seed)
            model = Predictor(episodes[0]["x"].shape[-1], args.hidden, kind).to(args.device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
            rng = np.random.default_rng(member_seed)
            best, best_epoch, best_state = float("inf"), -1, None
            history = []
            started = time.time()
            for epoch in range(args.epochs):
                model.train()
                shuffled = rng.permutation(split["train"]).tolist()
                losses = []
                for start in range(0, len(shuffled), args.batch_size):
                    x, y, mask = padded_batch(episodes, shuffled[start:start + args.batch_size],
                                             norm, args.device)
                    optimizer.zero_grad(set_to_none=True)
                    logits, _ = model(x)
                    loss = ((model.loss(logits, y) * mask).sum(1) / mask.sum(1)).mean()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    losses.append(float(loss.detach()))
                val = validate(model, episodes, split["validation"], norm,
                               args.device, args.batch_size)
                row = {"kind": kind, "member": member, "epoch": epoch + 1,
                       "train_loss": float(np.mean(losses)), "validation": val,
                       "elapsed_seconds": time.time() - started}
                history.append(row)
                if val["rmse_raw"] < best:
                    best, best_epoch = val["rmse_raw"], epoch
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                if epoch % 5 == 0:
                    print(json.dumps(row), flush=True)
                if epoch >= args.min_epochs - 1 and epoch - best_epoch >= args.patience:
                    break
            model.load_state_dict(best_state)
            test = validate(model, episodes, split["test"], norm, args.device, args.batch_size)
            report = {"kind": kind, "member": member, "best_epoch": best_epoch + 1,
                      "validation_rmse_raw": best, "test": test,
                      "elapsed_seconds": time.time() - started}
            ckpt = {"state_dict": best_state, "kind": kind, "hidden": args.hidden,
                    "input_dim": episodes[0]["x"].shape[-1], "norm": norm,
                    "report": report, "split": split, "member_seed": member_seed}
            torch.save(ckpt, path)
            dump(out / f"{kind}_{member}_history.json", history)
            reports.append(report)
            dump(out / "training_results.json", reports)
            print("FINISHED " + json.dumps(report), flush=True)
    # Constant train-mean baseline; same episode-balanced weighting as learned models.
    constant = {}
    for name, ids in split.items():
        constant[name] = float(np.sqrt(np.mean([(episodes[i]["return"] - norm["return_mean"])**2
                                               for i in ids])))
    dump(out / "constant_baseline.json", constant)
    print("TRAINING_COMPLETE", flush=True)


def load_predictors(directory, device="cpu"):
    models, norm = defaultdict(list), None
    for path in sorted(Path(directory).glob("*.pt")):
        if path.name == "preprocessing.pt":
            continue
        ckpt = torch.load(path, map_location="cpu")
        if "kind" not in ckpt:
            continue
        model = Predictor(ckpt["input_dim"], ckpt["hidden"], ckpt["kind"]).to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        models[ckpt["kind"]].append(model)
        norm = ckpt["norm"]
    if any(len(models[k]) != 3 for k in ("mse", "twohot")):
        raise RuntimeError("Require 3 completed members per head before branch evaluation")
    return models, norm


@torch.no_grad()
def score_anchor(anchor, models, norm):
    prefix = torch.from_numpy(tokens(anchor["states"], anchor["past_actions"], norm))[None]
    candidates = torch.from_numpy(candidate_tokens(
        anchor["states"][-1], anchor["candidate_actions"], anchor["next_states"],
        anchor["step"], norm))[:, None]
    predictions = {}
    for kind, ensemble in models.items():
        member_predictions = []
        for model in ensemble:
            _, h = model(prefix)
            logits, _ = model(candidates, h.expand(-1, len(candidates), -1).contiguous())
            value = model.mean(logits)[:, 0].numpy() * norm["return_std"] + norm["return_mean"]
            member_predictions.append(value)
        predictions[kind] = np.asarray(member_predictions)
    return predictions


def choose(predictions):
    choices = {}
    for kind, values in predictions.items():
        delta = values - values[:, :1]
        mean = delta.mean(0)
        choices[kind + "_greedy"] = int(np.argmax(mean))
        # Fixed before looking at branch labels. Std is a heuristic, not a CI.
        conservative = mean - delta.std(0)
        choices[kind + "_gated"] = int(np.argmax(conservative))
    return choices


def ci_episode(values, reps=10000):
    grouped = defaultdict(list)
    for ep, value in values:
        grouped[ep].append(value)
    means = np.asarray([np.mean(v) for _, v in sorted(grouped.items())])
    rng = np.random.default_rng(731)
    boot = means[rng.integers(len(means), size=(reps, len(means)))].mean(1)
    return [float(v) for v in np.quantile(boot, [.025, .975])]


def analyze_anchors(anchors, models, norm, out):
    def ranks(x):
        x = np.asarray(x)
        order = np.argsort(x, kind="stable")
        result = np.empty(len(x), np.float64)
        start = 0
        while start < len(x):
            end = start + 1
            while end < len(x) and x[order[end]] == x[order[start]]:
                end += 1
            result[order[start:end]] = (start + end - 1) / 2.0
            start = end
        return result
    records, groups = [], defaultdict(list)
    for anchor in anchors:
        pred = score_anchor(anchor, models, norm)
        choices = choose(pred)
        truth = np.asarray(anchor["delta_normalized"])
        random_mean = float(truth[1:].mean())
        for method, index in choices.items():
            kind = method.split("_")[0]
            r1, r2 = ranks(pred[kind].mean(0)[1:]), ranks(truth[1:])
            rho = float(np.corrcoef(r1, r2)[0, 1]) if r1.std() and r2.std() else float("nan")
            row = {"episode": anchor["episode"], "step": anchor["step"],
                   "method": method, "selected": index, "delta": float(truth[index]),
                   "random_mean": random_mean, "oracle": float(max(0, truth.max())),
                   "rank_spearman": rho}
            records.append(row)
            groups[method].append(row)
    summaries = {}
    for name, rows in groups.items():
        delta = np.asarray([r["delta"] for r in rows])
        summaries[name] = {
            "n_episodes": len(set(r["episode"] for r in rows)), "n_anchors": len(rows),
            "mean_delta": float(delta.mean()),
            "ci95_delta": ci_episode([(r["episode"], r["delta"]) for r in rows]),
            "mean_vs_random": float(np.mean([r["delta"] - r["random_mean"] for r in rows])),
            "ci95_vs_random": ci_episode([(r["episode"], r["delta"] - r["random_mean"])
                                           for r in rows]),
            "harm_rate": float(np.mean(delta < -.1)),
            "benefit_rate": float(np.mean(delta > .1)),
            "replacement_rate": float(np.mean([r["selected"] != 0 for r in rows])),
            "mean_spearman": float(np.nanmean([r["rank_spearman"] for r in rows])),
            "oracle_mean": float(np.mean([r["oracle"] for r in rows])),
            "random_mean": float(np.mean([r["random_mean"] for r in rows])),
            "per_episode": {str(ep): float(np.mean([r["delta"] for r in rows if r["episode"] == ep]))
                            for ep in sorted(set(r["episode"] for r in rows))}}
    dump(out / "ranking_summary.json", summaries)
    dump(out / "ranking_records.json", records)
    print(json.dumps(summaries, indent=2), flush=True)


def environment(args):
    import gym
    import oracle_branching_dt as ob
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    dt, config = ob.build_model(ckpt, "cpu")
    env = ob.wrap_env(gym.make(config["env_name"]), ckpt["state_mean"],
                      ckpt["state_std"], config["reward_scale"])
    env.seed(args.eval_seed)
    env.action_space.seed(args.eval_seed)
    return ob, ckpt, dt, config, env


def replay(args):
    """Recover prefix/one-step outcomes only; retain frozen old branch returns."""
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ob, ckpt, dt, config, env = environment(args)
    with open(args.branch_csv) as f:
        rows = [r for r in csv.DictReader(f) if abs(float(r["alpha"]) - .1) < 1e-8]
    by_episode = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_episode[int(r["episode_id"])][int(r["anchor_timestep"])].append(r)
    cache, audits = [], []
    for ep in sorted(by_episode):
        first = np.asarray(env.reset(), np.float32)
        states, actions, returns, times = ob.allocate_policy_tensors(
            dt, args.target_return * config["reward_scale"], first, "cpu")
        raw_states = [env.unwrapped._get_obs().astype(np.float32)]
        past_actions = []
        total = 0.0
        for step in range(dt.episode_len):
            base = ob.predict_action(dt, states, actions, returns, times, step)
            if step in by_episode[ep]:
                group = sorted(by_episode[ep][step], key=lambda r: int(r["candidate_id"]))
                candidates = np.asarray([json.loads(r["action_candidate_clipped"]) for r in group],
                                        np.float32)
                error = float(np.max(np.abs(base - candidates[0])))
                if error > 1e-6:
                    raise RuntimeError(f"Baseline action replay mismatch: {ep} {step} {error}")
                snapshot = ob.capture_env_snapshot(env)
                next_states = []
                for a in candidates:
                    ob.restore_env_snapshot(env, snapshot)
                    env.step(a)
                    next_states.append(env.unwrapped._get_obs().astype(np.float32))
                ob.restore_env_snapshot(env, snapshot)
                cache.append({"episode": ep, "step": step,
                              "states": np.asarray(raw_states), "past_actions": np.asarray(past_actions),
                              "candidate_actions": candidates, "next_states": np.asarray(next_states),
                              "delta_normalized": [float(r["delta_normalized_return"]) for r in group],
                              "base_raw_return": float(group[0]["branch_full_return"])})
            nxt, reward, done, _ = env.step(base)
            ob.update_policy_tensors(states, actions, returns, step, base, nxt, reward, "delayed")
            raw_states.append(env.unwrapped._get_obs().astype(np.float32))
            past_actions.append(base.copy())
            total += float(reward) / config["reward_scale"]
            if done:
                break
        # All rows have a branch-specific return; get candidate zero explicitly.
        group0 = next(iter(by_episode[ep].values()))
        expected_return = float(next(r for r in group0 if int(r["candidate_id"]) == 0)["branch_full_return"])
        error = abs(total - expected_return)
        audits.append({"episode": ep, "base_raw_return": total, "replay_error": error})
        if error > 1e-4:
            raise RuntimeError(f"Baseline return replay mismatch: {audits[-1]}")
        print("REPLAY " + json.dumps(audits[-1]), flush=True)
    torch.save(cache, out / "anchors.pt")
    dump(out / "replay_audit.json", audits)
    env.close()


def heldout(args):
    """Run only preselected candidate branches; no costly oracle search is needed."""
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    models, norm = load_predictors(args.model_dir)
    ob, ckpt, dt, config, env = environment(args)
    import gym
    data_env = gym.make(config["env_name"])
    ds = ob.get_d4rl_dataset(data_env)
    data_env.close()
    offline_states = torch.as_tensor((ds["observations"] - ckpt["state_mean"]) / ckpt["state_std"])
    offline_actions = ds["actions"].astype(np.float32)
    global_std = offline_actions.std(0) + 1e-6
    all_records, baselines = [], []
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in Path(args.model_dir).glob("*.pt")}
    dump(out / "protocol.json", {"args": vars(args), "model_sha256": source_hashes,
                                  "gating_beta": 1, "gating_delta": 0,
                                  "primary": "twohot_gated", "true_next_states": True,
                                  "random_control": "one presampled perturbation per anchor"})
    stop_episode = args.eval_episodes if args.eval_stop is None else args.eval_stop
    if not 0 <= args.eval_start < stop_episode <= args.eval_episodes:
        raise ValueError("Require 0 <= eval_start < eval_stop <= eval_episodes")
    # Execute preceding baseline episodes to preserve exactly the sequential-reset
    # stream of the unsharded protocol; only skip their diagnostic branches.
    for ep in range(stop_episode):
        first = np.asarray(env.reset(), np.float32)
        states, actions, returns, times = ob.allocate_policy_tensors(
            dt, args.target_return * config["reward_scale"], first, "cpu")
        raw_states = [env.unwrapped._get_obs().astype(np.float32)]
        past_actions = []
        total = 0.0
        anchors = set(ob.stratified_anchor_times(
            episode_len=dt.episode_len, num_anchors=10, exclude_final_steps=50,
            seed_components=(20260916, config["train_seed"], args.eval_seed, ep, 991)))
        if ep < args.eval_start:
            anchors = set()
        base_branch_scores = []
        for step in range(dt.episode_len):
            base = ob.predict_action(dt, states, actions, returns, times, step)
            if step in anchors:
                snapshot = ob.capture_env_snapshot(env)
                neighbors, _ = ob.knn_indices(query_normalized=states[0, step].numpy(),
                                              offline_states_normalized=offline_states,
                                              k=64, chunk_size=65536)
                local_std = np.maximum(offline_actions[neighbors].std(0), .05 * global_std)
                directions = ob.make_antithetic_directions(
                    action_dim=dt.action_dim, num_pairs=8,
                    seed_components=(20260916, config["train_seed"], args.eval_seed, ep, step))
                candidates = np.concatenate((base[None], np.clip(base + .1 * local_std * directions,
                                                                 env.action_space.low, env.action_space.high)))
                next_states = []
                for a in candidates:
                    ob.restore_env_snapshot(env, snapshot)
                    env.step(a)
                    next_states.append(env.unwrapped._get_obs().astype(np.float32))
                anchor = {"episode": ep, "step": step, "states": np.asarray(raw_states),
                          "past_actions": np.asarray(past_actions), "candidate_actions": candidates,
                          "next_states": np.asarray(next_states)}
                predictions = score_anchor(anchor, models, norm)
                choices = choose(predictions)
                rng = np.random.default_rng(np.random.SeedSequence([9182, args.eval_seed, ep, step]))
                choices["random"] = int(rng.integers(1, len(candidates)))
                choices["baseline"] = 0
                ps = ob.PolicySnapshot(states.clone(), actions.clone(), returns.clone(),
                                       total * config["reward_scale"], step)
                scores = {}
                for index in sorted(set(choices.values())):
                    result = ob.branch_rollout(env=env, env_snapshot=snapshot, policy_snapshot=ps,
                                               first_action=candidates[index], model=dt, time_steps=times,
                                               reward_scale=config["reward_scale"], reward_mode="delayed")
                    scores[index] = result.normalized_score
                base_branch_scores.append(scores[0])
                for method, index in choices.items():
                    all_records.append({"episode": ep, "step": step, "method": method,
                                        "selected": index, "delta": scores[index] - scores[0],
                                        "random_delta": scores[choices["random"]] - scores[0],
                                        "baseline_score": scores[0]})
                ob.restore_env_snapshot(env, snapshot)
            nxt, reward, done, _ = env.step(base)
            ob.update_policy_tensors(states, actions, returns, step, base, nxt, reward, "delayed")
            raw_states.append(env.unwrapped._get_obs().astype(np.float32))
            past_actions.append(base.copy())
            total += float(reward) / config["reward_scale"]
            if done:
                break
        base_score = ob.normalized_score(env, total)
        if ep < args.eval_start:
            print(json.dumps({"replayed_prefix_episode": ep, "baseline_score": base_score}), flush=True)
            continue
        error = max(abs(v - base_score) for v in base_branch_scores)
        if error > 1e-6:
            raise RuntimeError(f"Heldout restore mismatch {error}")
        baselines.append(base_score)
        dump(out / "records.json", all_records)
        dump(out / "baseline_scores.json", baselines)
        print(json.dumps({"episode": ep, "baseline_score": base_score,
                          "restore_error": error, "records": len(all_records)}), flush=True)
    summary = {}
    for method in sorted(set(r["method"] for r in all_records)):
        rows = [r for r in all_records if r["method"] == method]
        delta = np.asarray([r["delta"] for r in rows])
        summary[method] = {"mean_delta": float(delta.mean()),
                           "ci95_delta": ci_episode([(r["episode"], r["delta"]) for r in rows]),
                           "mean_vs_random": float(np.mean([r["delta"] - r["random_delta"] for r in rows])),
                           "ci95_vs_random": ci_episode([(r["episode"], r["delta"] - r["random_delta"])
                                                        for r in rows]),
                           "harm_rate": float(np.mean(delta < -.1)),
                           "benefit_rate": float(np.mean(delta > .1)),
                           "replacement_rate": float(np.mean([r["selected"] != 0 for r in rows])),
                           "n_episodes": len(baselines), "n_anchors": len(rows),
                           "per_episode": {str(ep): float(np.mean([r["delta"] for r in rows if r["episode"] == ep]))
                                           for ep in range(args.eval_start, stop_episode)}}
    dump(out / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    env.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["train", "replay", "analyze", "heldout", "selftest"])
    p.add_argument("--output-dir", required=True)
    p.add_argument("--dataset", default="/labmount/users/202615385/.d4rl/datasets/halfcheetah_medium_replay-v2.hdf5")
    p.add_argument("--checkpoint", default="/labmount/users/202615385/code/DT-EXP/checkpoints/dt-halfcheetah-medium-replay-v2-delayed-seed0-step50000.pt")
    p.add_argument("--branch-csv", default="/labmount/users/202615385/code/DT-EXP/results/oracle_branching/pilot_seed0_rtg12000_cpu_v3/candidate_branches.csv")
    p.add_argument("--model-dir")
    p.add_argument("--anchor-cache")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--min-epochs", type=int, default=25)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--members", type=int, default=3)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--split-seed", type=int, default=731)
    p.add_argument("--eval-seed", type=int, default=42)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--eval-start", type=int, default=0)
    p.add_argument("--eval-stop", type=int)
    p.add_argument("--target-return", type=float, default=12000)
    args = p.parse_args()
    torch.set_num_threads(1 if args.mode in ("replay", "analyze", "heldout") else 4)
    if args.mode == "train":
        train(args)
    elif args.mode == "replay":
        replay(args)
    elif args.mode == "analyze":
        models, norm = load_predictors(args.model_dir)
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        analyze_anchors(torch.load(args.anchor_cache), models, norm, out)
    elif args.mode == "heldout":
        heldout(args)
    else:
        selftest()


def selftest():
    seed_all(4)
    for kind in ["mse", "twohot"]:
        model = Predictor(42, 16, kind).eval()
        x = torch.randn(2, 12, 42)
        whole, _ = model(x)
        _, h = model(x[:, :11])
        last, _ = model(x[:, 11:], h)
        assert torch.allclose(whole[:, -1:], last, atol=1e-6)
        modified = x.clone()
        modified[:, 8:] += 10
        other, _ = model(modified)
        assert torch.allclose(whole[:, :8], other[:, :8], atol=1e-6)
        target = torch.linspace(-5, 5, 24).reshape(2, 12)
        loss = model.loss(whole, target).mean()
        loss.backward()
        assert torch.isfinite(loss)
    print("SELFTEST_PASS: causal prefix, cached hidden state, finite loss and gradients", flush=True)


if __name__ == "__main__":
    main()
