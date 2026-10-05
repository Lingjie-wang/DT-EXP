"""Guard seed isolation, upstream integrity, and extended-budget observation."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from observe import parse_line
from prepare import COMMIT, ENV_NAME, prepare
from run import digest, official_command

class ExtendedCampaignTests(unittest.TestCase):
    def test_independent_seeds_preserve_source_and_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "baseline"
            upstream = base / "source/upstream/scripts/train.py"
            upstream.parent.mkdir(parents=True)
            upstream.write_text("unchanged upstream\n")
            data = base / "delayed/dataset" / ENV_NAME / "train/data.arrow"
            data.parent.mkdir(parents=True)
            data.write_bytes(b"same prepared dataset")
            original = dict(
                upstream_commit=COMMIT, source_modifications=[], trainer_seed=42,
                upstream_sha256={"scripts/train.py": digest(upstream)},
                dataset_sha256={str(data.relative_to(base)): digest(data)},
            )
            (base / "protocol.json").write_text(json.dumps(original))
            root = Path(directory) / "extended"
            with patch("prepare.subprocess.check_output", return_value="test==1\n"):
                campaign = prepare(root, base, 8000, [0, 3], "delayed")
            self.assertEqual(len({r["run_id"] for r in campaign["runs"]}), 2)
            for seed in [0, 3]:
                run_root = root / f"seed{seed}"
                p = json.loads((run_root / "protocol.json").read_text())
                self.assertEqual(p["trainer_seed"], 42)
                self.assertEqual(p["eval_updates"], list(range(500, 8001, 500)))
                command = official_command(p, "training")
                self.assertEqual(command[command.index("--seed") + 1], str(seed))
                self.assertEqual(command[command.index("--epochs") + 1], "8000")
                smoke = official_command(p, "preflight")
                self.assertEqual(smoke[smoke.index("--epochs") + 1], "1")
                for phase in ["preflight", "training"]:
                    copy = run_root / "delayed" / phase / "upstream"
                    self.assertEqual(digest(copy / "scripts/train.py"), digest(upstream))
                    link = copy / "data/dataset" / ENV_NAME
                    self.assertEqual(
                        link.resolve(), run_root / "delayed/dataset" / ENV_NAME
                    )
                for relative, expected in p["dataset_sha256"].items():
                    self.assertEqual(digest(run_root / relative), expected)
            with self.assertRaises(FileExistsError):
                prepare(root, base, 8000, [0, 3], "delayed")
            data.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "Baseline dataset changed"):
                prepare(Path(directory) / "other", base, 8000, [0], "delayed")

    def test_progress_uses_extended_budget(self):
        self.assertEqual(
            parse_line("75%|xxx| 6000/8000 [01:00<00:30]", 8000),
            ("progress", {"completed_updates": 6000}),
        )
        self.assertIsNone(parse_line("100%|xxx| 1/1 [00:10<00:00]", 8000))
        kind, row = parse_line(
            "Evaluation metrics at step 8000: {'eval/norm_score': 39.0}", 8000
        )
        self.assertEqual(kind, "evaluation")
        self.assertEqual(row["completed_updates"], 8000)


if __name__ == "__main__":
    unittest.main()
