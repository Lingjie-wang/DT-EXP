"""CPU-only tests for archived-command fidelity, not algorithm rewrites."""

import importlib.util
import json
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "legacy", ROOT / "scripts/dt_experiments/legacy_v3_high.py"
)
legacy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(legacy)


class LegacyProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.source = Path(self.temporary.name)
        scripts = self.source / "scripts/dt_experiments"
        scripts.mkdir(parents=True)
        for name in legacy.TEMPLATES.values():
            data = subprocess.check_output([
                "git", "show", f"{legacy.REVISION}:scripts/dt_experiments/{name}",
            ], cwd=ROOT)
            (scripts / name).write_bytes(data)

    def flags(self, stage, seed=3, smoke=False):
        args = legacy.command(self.source, stage, seed, self.source / "out", smoke)
        return dict(zip(args[1::2], args[2::2]))

    def test_original_warmup_not_new_trainer(self):
        args = legacy.command(self.source, "prepare", 4, self.source / "out")
        self.assertTrue(args[0].endswith("sap_dt_one_sided.py"))
        flags = self.flags("prepare", 4)
        for key, value in {"update_steps": "50001", "eval_every": "50000",
                           "eval_episodes": "10", "preference_start_step": "50000",
                           "train_seed": "4", "preference_pair_seed": "4"}.items():
            self.assertEqual(flags["--" + key], value)

    def test_original_v3_not_modern_paired_resume(self):
        for seed in (3, 4, 5):
            flags = self.flags("v3", seed)
            for key, value in {"update_steps": "100001", "eval_every": "5000",
                               "eval_episodes": "100", "preference_weight": "0.05",
                               "reference_weight": "0.1", "dynamic_priority_mix": "0.5",
                               "preference_target_mode": "high_only"}.items():
                self.assertEqual(flags["--" + key], value)
            self.assertNotIn("--paired_resume", flags)
            self.assertNotIn("--reference_anchor_mode", flags)

    def test_control_really_skips_auxiliary_forwards(self):
        flags = self.flags("dt")
        self.assertEqual(flags["--preference_mode"], "control")
        self.assertEqual(flags["--update_steps"], "100001")
        self.assertNotIn("--preference_arrays_path", flags)
        self.assertNotIn("--preference_pair_seed", flags)

    def test_control_uses_its_own_historical_revision(self):
        source, revision = legacy.source_for(self.source, "dt")
        self.assertEqual(source, self.source / "control_source")
        self.assertEqual(revision, legacy.CONTROL_REVISION)
        text = subprocess.check_output([
            "git", "show", f"{revision}:algorithms/offline/hard_fork_dt.py",
        ], cwd=ROOT, text=True)
        self.assertNotIn("preference_target_mode", text)
        self.assertNotIn("paired_resume", text)

    def test_original_pair_thresholds(self):
        flags = self.flags("diagnose", 5)
        for key, value in {"seed": "5", "num_pairs": "100000",
                           "state_keep_fraction": "0.25", "num_candidates": "64",
                           "prefix_length": "5", "prefix_state_threshold": "0.75",
                           "prefix_action_threshold": "0.50", "min_return_gap": "2000",
                           "wandb_mode": "disabled"}.items():
            self.assertEqual(flags["--" + key], value)

    def test_smoke_cannot_change_production_arguments(self):
        self.assertEqual(self.flags("v3", smoke=True)["--update_steps"], "5")
        self.assertEqual(self.flags("v3")["--update_steps"], "100001")
        with self.assertRaises(ValueError):
            self.flags("v3", seed=0)

    def test_historical_gpu_mapping(self):
        for seed in (3, 4, 5):
            self.assertEqual(legacy.expected_gpu(seed, "prepare"), "RTX 3090")
        self.assertEqual(legacy.expected_gpu(3, "dt"), "RTX 4090")
        self.assertEqual(legacy.expected_gpu(4, "dt"), "RTX 3090")
        self.assertEqual(legacy.expected_gpu(5, "dt"), "RTX 3090")
        self.assertEqual(legacy.expected_gpu(3, "v3"), "RTX 3090")
        self.assertEqual(legacy.expected_gpu(4, "v3"), "RTX 3090")
        self.assertEqual(legacy.expected_gpu(5, "v3"), "RTX 4090")

    def test_user_approved_mixed_gpu_dispatch(self):
        for seed in (3, 4, 5):
            for stage in ("prepare", "dt", "v3"):
                for model in ("RTX 3090", "RTX 4090"):
                    data = legacy.hardware_record("NVIDIA GeForce " + model, seed, stage)
                    self.assertEqual(data["historical_gpu"], legacy.expected_gpu(seed, stage))
                    self.assertEqual(data["matches_historical_gpu"],
                                     model == legacy.expected_gpu(seed, stage))
        with self.assertRaises(RuntimeError):
            legacy.hardware_record("NVIDIA A100", 3, "prepare")

    def test_replay_original_seed_commands(self):
        ignored = {"--config_path", "--group", "--name", "--update_steps",
                   "--pretrained_checkpoint_path", "--preference_arrays_path",
                   "--reference_checkpoint_path", "--checkpoints_path",
                   "--checkpoint", "--result_dir", "--wandb_mode",
                   "--wandb_group", "--wandb_name"}
        for seed in (0, 1, 2):
            for stage in legacy.TEMPLATES:
                args = legacy.command(self.source, stage, seed, self.source / "out",
                                      profile="replay012")
                actual = dict(zip(args[1::2], args[2::2]))
                suffix = "" if seed == 0 and stage in ("prepare", "diagnose") else f"_seed{seed}"
                filename = legacy.TEMPLATES[stage].replace("_seed1", suffix)
                body = subprocess.check_output([
                    "git", "show", f"{legacy.REVISION}:scripts/dt_experiments/{filename}",
                ], cwd=ROOT, text=True)
                argv = shlex.split(body.split("\npython ", 1)[1].replace("\\\n", " "))
                original = dict(zip(argv[1::2], argv[2::2]))
                for key in set(original) - ignored:
                    self.assertEqual(actual[key], original[key], (seed, stage, key))
                if stage in ("dt", "v3"):
                    self.assertEqual(actual["--update_steps"], "100001")
                elif stage == "prepare":
                    self.assertEqual(actual["--update_steps"], "50001")
                self.assertNotIn("--paired_resume", actual)
                self.assertIn("Replay012", str(args))
        with self.assertRaises(ValueError):
            legacy.command(self.source, "v3", 3, self.source / "out", profile="replay012")

    def test_replay_hardware_is_not_relaxed(self):
        for seed in (0, 1, 2):
            for stage in ("prepare", "dt", "v3"):
                wanted = "RTX 4090" if (seed, stage) in ((0, "dt"), (2, "v3")) else "RTX 3090"
                data = legacy.hardware_record("NVIDIA GeForce " + wanted, seed,
                                              stage, "replay012")
                self.assertTrue(data["matches_historical_gpu"])
                other = "RTX 3090" if wanted == "RTX 4090" else "RTX 4090"
                with self.assertRaises(RuntimeError):
                    legacy.hardware_record(other, seed, stage, "replay012")

    def test_replay_cannot_use_existing_legacy_directory(self):
        result = subprocess.run([
            sys.executable, str(ROOT / "scripts/dt_experiments/legacy_v3_high.py"),
            "prepare", "--profile", "replay012", "--seed", "0", "--root", str(self.source),
        ], text=True, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("different protocol", result.stderr)

    def test_original_comparison_is_read_only(self):
        try:
            import numpy as np
            import torch
        except ImportError:
            self.skipTest("Checkpoint-comparison test also runs in server PyTorch environment")
        old = self.source / "old.pt"
        new = self.source / "new.pt"
        pairs = self.source / "pairs.npz"
        checkpoint = {"model_state": {"weight": torch.ones(3)}, "next_step": 50000,
                      "state_mean": np.zeros(2), "state_std": np.ones(2)}
        torch.save(checkpoint, old)
        checkpoint["model_state"]["weight"][0] = 2
        torch.save(checkpoint, new)
        np.savez(pairs, valid_branch=np.array([True, False]))
        legacy.write_json(self.source / "original_inputs.json", {"0": {
            "checkpoint": str(old), "checkpoint_sha256": legacy.digest(old),
            "pairs": str(pairs), "pairs_sha256": legacy.digest(pairs),
        }})
        before = old.read_bytes(), pairs.read_bytes()
        report = legacy.compare_original(self.source, 0, new, pairs)
        self.assertFalse(report["model_tensors_equal"])
        self.assertEqual(report["max_abs_tensor_difference"], 1.0)
        self.assertTrue(report["pair_arrays_equal"])
        self.assertEqual(report["valid_pairs_old_new"], [1, 1])
        self.assertEqual(before, (old.read_bytes(), pairs.read_bytes()))

    def test_logger_survives_init_rebinding_and_blocks_binary_upload(self):
        logged, saved = [], []
        run = SimpleNamespace(
            id="test", url="offline", name="test", config={}, summary={},
            log=lambda *a, **kw: logged.append((a, kw)),
            save=lambda *a, **kw: saved.append((a, kw)),
        )
        wandb = SimpleNamespace(finish=lambda: None)

        def init():
            wandb.log, wandb.save = run.log, run.save
            return run

        wandb.init = init
        record = self.source / "record"
        record.mkdir()
        hardware = legacy.hardware_record("NVIDIA GeForce RTX 4090", 4, "prepare")
        legacy.install_observer(wandb, record, {}, hardware=hardware)
        wandb.init()
        self.assertEqual(run.summary["hardware/gpu"], "NVIDIA GeForce RTX 4090")
        self.assertFalse(run.summary["hardware/matches_historical_gpu"])
        self.assertEqual(run.config, {})
        self.assertEqual(json.loads((record / "run.json").read_text())["hardware"],
                         hardware)
        wandb.log({"train_loss": 0.5}, step=50000)
        self.assertEqual(wandb.save("weights.pt"), [])
        wandb.save("report.json")
        self.assertEqual(len(logged), 1)
        self.assertEqual(len(saved), 1)
        row = json.loads((record / "metrics.jsonl").read_text())
        self.assertEqual(row, {"step": 50000, "train_loss": 0.5})
        with self.assertRaises(ValueError):
            wandb.log({"train_loss": float("nan")}, step=50001)


if __name__ == "__main__":
    unittest.main()
