"""Freeze the author release and add explicit, observation-only hooks."""

import argparse
import difflib
import hashlib
import io
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

COMMIT = "ac85be4877016724d648ee055854fd14482b5dad"


def digest(path):
    value = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def replace_once(text, before, after):
    if text.count(before) != 1:
        raise ValueError(f"Expected one exact upstream block: {before!r}")
    return text.replace(before, after, 1)


def observed_source(relative, text):
    if relative == "gym/experiment_clean.py":
        replacements = [
            ("import json\n", "import json\nimport condt_observe as observer\n"),
            (
                "    data_class, env = prep_data(variant)\n",
                "    observer.initialize(variant)\n"
                "    data_class, env = prep_data(variant)\n",
            ),
            (
                "    print(f'Starting iter')\n",
                "    observer.ready(trainer)\n    print(f'Starting iter')\n",
            ),
            (
                "trainer.train_iteration(num_steps=100000, iter_num = 0, "
                "print_logs=True)",
                "trainer.train_iteration(num_steps=int(os.environ.get("
                "'CONDT_PREFLIGHT_PRETRAIN_STEPS', '100000')), iter_num = 0, "
                "print_logs=True)",
            ),
            (
                "        seeded_outputs = trainer.eval_method()\n",
                "        seeded_outputs = trainer.eval_method()\n"
                "        observer.evaluation(0, 0, seeded_outputs, trainer, {})\n",
            ),
        ]
    elif relative == "gym/decision_transformer/training/trainer_clean.py":
        replacements = [
            (
                "import itertools\n",
                "import itertools\nimport condt_observe as observer\n",
            ),
            (
                "        for ix in range(num_steps):\n",
                "        for ix in range(num_steps):\n"
                "            observed_lrs = "
                "[g['lr'] for g in self.optimizer.param_groups]\n",
            ),
            (
                "                self.scheduler.step()\n",
                "                self.scheduler.step()\n"
                "            observer.training_update(iter_num, ix, num_steps, "
                "train_loss, observed_lrs, self.optimizer)\n",
            ),
            (
                "        return json_outputs\n",
                "        observer.evaluation(iter_num, num_steps, "
                "json_outputs, self, logs)\n"
                "        return json_outputs\n",
            ),
        ]
    else:
        return text
    for before, after in replacements:
        text = replace_once(text, before, after)
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    upstream = args.upstream.resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True
    ).strip()
    assert revision == COMMIT
    assert not subprocess.check_output(
        ["git", "-C", str(upstream), "diff", "HEAD", "--"], text=True
    )
    data = args.data_dir.resolve()
    assert (data / "hopper-medium-v2.pkl").is_file()
    audit = json.loads((data / "data_audit.json").read_text())
    assert digest(data / "hopper-medium-v2.pkl") == audit["pickle_sha256"]
    assert digest(data / "hopper_medium-v2.hdf5") == audit["source_sha256"]
    assert audit["author_commit"] == COMMIT
    root.mkdir(parents=True, exist_ok=False)
    source = root / "source"
    source.mkdir()
    for path in Path(__file__).parent.iterdir():
        if path.is_file():
            shutil.copy2(path, source / path.name)
    official = source / "upstream_original"
    official.mkdir()
    archive = subprocess.check_output(["git", "-C", str(upstream), "archive", COMMIT])
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(official)
    original_hashes = {
        str(p.relative_to(official)): digest(p)
        for p in official.rglob("*")
        if p.is_file()
    }
    diffs = []
    for arm in ("dt", "condt"):
        for phase in ("preflight", "training"):
            checkout = root / arm / phase / "upstream"
            shutil.copytree(official, checkout)
            for relative in (
                "gym/experiment_clean.py",
                "gym/decision_transformer/training/trainer_clean.py",
            ):
                path = checkout / relative
                before = path.read_text()
                after = observed_source(relative, before)
                path.write_text(after)
                if arm == "dt" and phase == "preflight":
                    diffs.extend(
                        difflib.unified_diff(
                            before.splitlines(True),
                            after.splitlines(True),
                            fromfile="a/" + relative,
                            tofile="b/" + relative,
                        )
                    )
            shutil.copy2(source / "telemetry.py", checkout / "gym" / "condt_observe.py")
            (checkout / "gym" / "data" / "hopper-medium-v2.pkl").symlink_to(
                data / "hopper-medium-v2.pkl"
            )
    (root / "source_patch.diff").write_text("".join(diffs))
    shutil.copy2(data / "data_audit.json", root / "data_audit.json")
    runtime = Path(__file__).resolve().parents[2] / ".runtime" / "condt-author-env"
    runtime_differences = json.loads(
        (runtime / "environment-differences.json").read_text()
    )
    for name in (
        "environment-differences.json",
        "installed-packages.txt",
        "verified-versions.json",
        "dependency-provenance.json",
    ):
        shutil.copy2(runtime / name, root / name)
    patched = root / "dt" / "preflight" / "upstream"
    patched_hashes = {
        str(p.relative_to(patched)): digest(p)
        for p in patched.rglob("*")
        if p.is_file() and not p.is_symlink()
    }
    protocol = dict(
        method="ConDT-Author-Code",
        upstream_url="https://github.com/SachinKonan/Contrastive-Decision-Transformers",
        upstream_commit=COMMIT,
        paper_url="https://proceedings.mlr.press/v205/konan23a.html",
        dataset="hopper-medium-v2",
        dataset_path=str(data / "hopper-medium-v2.pkl"),
        dataset_sha256=digest(data / "hopper-medium-v2.pkl"),
        hdf5_sha256=digest(data / "hopper_medium-v2.hdf5"),
        seed=0,
        training_seed_count=1,
        evaluation_seeds=[1, 5, 10],
        eval_episodes_per_seed=100,
        eval_environment="Hopper-v3",
        target_return=3600,
        reward_mode="normal (original dense rewards)",
        reward_scale=1000.0,
        main_updates=100000,
        eval_every=10000,
        eval_updates=list(range(0, 100001, 10000)),
        batch_size=64,
        context_length=20,
        hidden_size=128,
        layers=3,
        heads=1,
        dropout=0.1,
        normalized_score_min=-20.272305,
        normalized_score_max=3234.3,
        arms={
            "dt": dict(model_type="dt", pretrain=False, pretrain_updates=0,
                       total_updates=100000),
            "condt": dict(
                model_type="dt_contrast_simclr_product",
                pretrain=True,
                pretrain_updates=100000,
                total_updates=200000,
                beta=0.1,
            ),
        },
        primary_result="Final 100000-main-update checkpoint, not the best checkpoint",
        source_changes=[
            "Observation hooks, training seed 0, JSON records and full checkpoints",
            "Two-update preflight override only; full pretraining unchanged",
            "Native W&B disabled; online observer records actual epoch outputs",
        ],
        preserved_author_behaviors=[
            "100000 contrastive pretraining updates; original optimizer and LambdaLR",
            "ConDT main training continues MSE + 0.1 NTXent after pretraining",
            "Trajectory-window positives sampled with replacement; padding possible",
            "Metric-learning NTXentLoss 1.3.0 rather than paper printed equation",
            "Hard-coded compression dimension 128 instead of paper Hopper-medium 50",
            "All models filter trajectories of length <=3 due to original condition",
            "Original normalization, RTG updates, evaluation seeds and Ray rollout code",
        ],
        limitations=[
            "One training seed; the three evaluation seeds are not training seeds",
            "Author release differs from paper; no algorithm corrections applied",
            "Runtime uses independently recorded compatibility versions",
        ],
        runtime_differences=runtime_differences,
        execution_device="cuda; Slurm GPU allocation; GPU preflight before full run",
        ray_gpu_visibility="RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1",
        data_audit=audit,
        upstream_sha256=original_hashes,
        execution_sha256=patched_hashes,
        harness_sha256={p.name: digest(p) for p in source.iterdir() if p.is_file()},
        wandb_entity="2820402607-shandong-university",
        wandb_project="CORL-DDR",
        wandb_group=root.name,
    )
    (root / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print(json.dumps({"root": str(root), "upstream_commit": COMMIT}))


if __name__ == "__main__":
    main()
