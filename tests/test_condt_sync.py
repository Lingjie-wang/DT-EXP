"""Offline checks of metric levels, event ordering, and W&B readback checks."""

import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.condt.sync_results import (
    evaluation_metrics,
    expected_points,
    initialize_run,
    monitor,
    pending_events,
    rows,
    run_identity,
    sanitized_config,
    sync_pending,
    verify_remote,
)

def evaluation(step=0):
    return dict(
        phase="main",
        main_updates=step,
        pretrain_updates=10000,
        wall_seconds=2.5,
        outputs={
            "1": dict(returns=[0.0, 10.0], lengths=[10, 20]),
            "5": dict(returns=[20.0, 30.0], lengths=[30, 40]),
            "10": dict(returns=[40.0, 50.0], lengths=[50, 60]),
        },
    )


def training(phase, main, total):
    return dict(
        phase=phase,
        main_updates=main,
        total_updates=total,
        phase_updates=main if phase == "main" else total,
        train_loss=0.5,
        learning_rates=[0.0001],
        elapsed_seconds=total / 100,
    )


class ConDTSyncTests(unittest.TestCase):
    def setUp(self):
        self.protocol = dict(
            normalized_score_min=0,
            normalized_score_max=100,
            upstream_commit="pinned-author-code",
            wandb_entity="test",
            wandb_project="test",
            wandb_group="campaign1",
            eval_episodes=2,
        )

    def test_normalization_and_uncertainty_use_correct_hierarchy(self):
        self.protocol.update(normalized_score_min=-100, normalized_score_max=100)
        metrics, table = evaluation_metrics(evaluation(), self.protocol)
        self.assertEqual(metrics["eval/raw_return_mean"], 25)
        self.assertEqual(metrics["eval/normalized_score_mean"], 62.5)
        self.assertEqual(metrics["eval/raw_return_eval_seed_std"], 20)
        self.assertAlmostEqual(
            metrics["eval/raw_return_eval_seed_se"], 20 / math.sqrt(3)
        )
        self.assertEqual(metrics["eval/normalized_score_eval_seed_std"], 10)
        self.assertAlmostEqual(metrics["eval/raw_return_episode_std"], math.sqrt(350))
        self.assertEqual(metrics["eval/evaluation_seed_count"], 3)
        self.assertEqual(metrics["eval/episode_count"], 6)
        self.assertEqual(table[-1], [0, 10, 1, 50.0, 75.0, 60])
        self.assertEqual(metrics["total_updates"], 10000)

    def test_unequal_episode_counts_do_not_reweight_seed_uncertainty(self):
        value = evaluation()
        value["outputs"]["10"] = dict(returns=[40.0], lengths=[50])
        metrics, _ = evaluation_metrics(value, self.protocol)
        self.assertEqual(metrics["eval/raw_return_mean"], 20)
        self.assertAlmostEqual(metrics["eval/raw_return_eval_seed_mean"], 70 / 3)

    def test_invalid_results_fail_instead_of_uploading_nan(self):
        for bad in (float("nan"), float("inf")):
            value = evaluation()
            value["outputs"]["1"]["returns"][0] = bad
            with self.assertRaisesRegex(ValueError, "Non-finite"):
                evaluation_metrics(value, self.protocol)
        value = evaluation()
        value["outputs"]["1"]["lengths"] = []
        with self.assertRaisesRegex(ValueError, "Mismatched"):
            evaluation_metrics(value, self.protocol)

    def test_event_order_resumption_and_zero_point(self):
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            (work / "evaluations").mkdir()
            records = [training("pretrain", 0, 10000), training("main", 10000, 20000)]
            (work / "metrics.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in records)
            )
            for step in (0, 10000):
                (work / "evaluations" / f"eval_main_{step:06d}.json").write_text(
                    json.dumps(evaluation(step))
                )
            cursor = dict(metrics_rows=0, evaluated=[], latest_main_updates=0)
            logged = []
            sdk = SimpleNamespace(Table=lambda **kwargs: kwargs)
            run = SimpleNamespace(log=lambda record: logged.append(record))
            sync_pending(run, sdk, work, cursor, self.protocol)
            self.assertEqual([record["total_updates"] for record in logged],
                             [10000, 10000, 20000, 20000])
            self.assertEqual(logged[0]["train/phase"], "pretrain")
            self.assertEqual(logged[1]["eval/raw_return_mean"], 25)
            self.assertEqual(logged[2]["train/phase"], "main")
            self.assertEqual(cursor["evaluated"], [0, 10000])
            self.assertEqual(cursor["metrics_rows"], 2)
            sync_pending(run, sdk, work, cursor, self.protocol)
            self.assertEqual(len(logged), 4)
            (work / "metrics.jsonl").write_text("")
            with self.assertRaisesRegex(RuntimeError, "truncated"):
                pending_events(work, cursor)

    def test_incomplete_last_jsonl_record_is_not_consumed(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "metrics.jsonl"
            path.write_text('{"total_updates": 10}\n{"total_updates":')
            self.assertEqual(rows(path), [dict(total_updates=10)])

    def test_configuration_allowlist_excludes_private_and_unknown_nested_values(self):
        self.protocol.update(
            dataset="hopper-medium-v2",
            dataset_path="PRIVATE_PATH",
            upstream_sha256={"private_source.py": "PRIVATE_HASH"},
            execution_sha256={"private_source.py": "PRIVATE_HASH"},
            harness_sha256={"private_source.py": "PRIVATE_HASH"},
            arms={"condt": dict(beta=0.1, source="PRIVATE_SOURCE")},
            runtime_differences={
                "torch": dict(author="1.10.2", runtime="1.11.0", log="PRIVATE_LOG"),
                "private-package": dict(runtime="PRIVATE_VERSION"),
            },
            data_audit={
                "raw_transitions": 100,
                "hdf5_inventory": {"PRIVATE_DATA": [1, 2, 3]},
                "raw_reward_statistics": dict(mean=1.0, samples="PRIVATE_SAMPLES"),
            },
        )
        config = sanitized_config(self.protocol, "condt")
        self.assertNotIn("PRIVATE", json.dumps(config))
        self.assertNotIn("wandb_entity", config)
        self.assertEqual(config["data_statistics"]["raw_transitions"], 100)
        self.assertEqual(config["runtime_differences"]["torch"]["runtime"], "1.11.0")
        self.assertEqual(config["arms"]["condt"], dict(beta=0.1))

    def test_complete_monitor_only_sends_safe_config_metrics_and_eval_tables(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            work = root / "condt"
            (work / "evaluations").mkdir(parents=True)
            directory = root / "wandb_sync"
            directory.mkdir()
            self.protocol.update(eval_updates=[0, 10000], dataset_path="PRIVATE_PATH")
            (root / "protocol.json").write_text(json.dumps(self.protocol))
            (root / "source_patch.diff").write_text("PRIVATE_PATCH")
            (root / "installed-packages.txt").write_text("PRIVATE_INSTALL_LOG")
            (work / "status.json").write_text(json.dumps(dict(status="completed")))
            records = [training("pretrain", 0, 10000), training("main", 10000, 20000)]
            (work / "metrics.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in records)
            )
            for step in (0, 10000):
                (work / "evaluations" / f"eval_main_{step:06d}.json").write_text(
                    json.dumps(evaluation(step))
                )
            logs, initialized, finishes = [], [], []
            run = SimpleNamespace(
                url="https://example.invalid/run", summary={},
                log=lambda value: logs.append(value),
                define_metric=lambda *args, **kwargs: None,
                finish=lambda **kwargs: finishes.append(kwargs),
            )

            def initialize(**kwargs):
                initialized.append(kwargs)
                run.config = kwargs["config"]
                return run

            run.scan_history = lambda keys: [row for row in logs if all(
                key in row for key in keys
            )]
            # Deliberately provide no Artifact/log_artifact/add_file/save APIs.
            sdk = SimpleNamespace(
                init=initialize, Settings=lambda **kwargs: kwargs,
                Table=lambda **kwargs: kwargs,
                Api=lambda **kwargs: SimpleNamespace(run=lambda _: run),
            )
            args = SimpleNamespace(
                root=root, arm="condt", job="123", max_days=7, poll_seconds=60,
            )
            with patch("scripts.condt.sync_results.slurm", return_value="COMPLETED"):
                monitor(args, sdk, directory)
            self.assertEqual(finishes, [{}])
            self.assertEqual(len(logs), 4)
            self.assertNotIn("PRIVATE", json.dumps(initialized))
            self.assertNotIn("PRIVATE", json.dumps(run.summary))
            self.assertEqual(run.summary["experiment_config"], run.config)
            self.assertTrue((directory / "completion_verification.json").exists())
            scope = json.loads((directory / "upload_scope.json").read_text())
            self.assertEqual(scope["schema"], "condt-metrics-v2")
            self.assertFalse(scope["file_artifacts"])

    def test_final_readback_requires_every_evaluation_point(self):
        key = "eval/normalized_score_mean"
        history = [dict(main_updates=0, **{key: 25})]
        remote = SimpleNamespace(
            config=dict(arm="condt", upstream_commit="pinned-author-code"),
            summary={"result/final_normalized_score": 25},
            scan_history=lambda keys: history,
        )
        sdk = SimpleNamespace(Api=lambda **kwargs: SimpleNamespace(run=lambda _: remote))
        verify_remote(sdk, "fake/path", "condt", self.protocol, [evaluation(0)])
        with self.assertRaisesRegex(RuntimeError, "update 10000"):
            verify_remote(sdk, "fake/path", "condt", self.protocol,
                          [evaluation(0), evaluation(10000)])

    def test_run_ids_and_complete_schedule(self):
        self.assertEqual(expected_points(self.protocol), list(range(0, 100001, 10000)))
        self.assertEqual(run_identity(self.protocol, "condt"),
                         run_identity(self.protocol, "condt"))
        self.assertNotEqual(run_identity(self.protocol, "condt"),
                            run_identity(self.protocol, "dt"))
        self.protocol["eval_updates"] = [0, 100, 1000]
        self.assertEqual(expected_points(self.protocol), [0, 100, 1000])

    def test_initialization_retries_the_same_run_identity(self):
        identities = []

        def initialize(**kwargs):
            identities.append(kwargs["id"])
            if len(identities) == 1:
                raise OSError("Temporary connection failure")
            return "fake-run"

        with tempfile.TemporaryDirectory() as temporary:
            sdk = SimpleNamespace(init=initialize)
            with patch("scripts.condt.sync_results.time.sleep"):
                result = initialize_run(
                    sdk, Path(temporary), id="stable", resume="allow"
                )
            self.assertEqual(result, "fake-run")
            self.assertEqual(identities, ["stable", "stable"])


if __name__ == "__main__":
    unittest.main()
