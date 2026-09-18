"""Check that reference-only changes just one v3-high loss coefficient."""

import pathlib
import shlex
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "dt_experiments"


def training_flags(script_name):
    text = (SCRIPTS / script_name).read_text()
    command = text.split("python algorithms/offline/hard_fork_dt.py", 1)[1]
    command = command.split('"${EXTRA_ARGS[@]}"', 1)[0]
    tokens = shlex.split(command.replace("\\\n", " "))
    if len(tokens) % 2:
        raise AssertionError("Expected flag/value pairs")
    return dict(zip(tokens[::2], tokens[1::2]))


class ReferenceOnlyProtocolTest(unittest.TestCase):
    def test_only_preference_coefficient_and_output_path_change(self):
        full = training_flags(
            "run_v3_reference_ablation_to100k_hcmr_seed0.sbatch"
        )
        ref_only = training_flags(
            "run_v3_reference_only_to100k_hcmr_seed0.sbatch"
        )
        # Resolve the reference=0.1 arm of the existing two-arm script.
        full["--reference_weight"] = "0.1"
        self.assertEqual(full["--preference_weight"], "0.05")
        self.assertEqual(ref_only["--preference_weight"], "0.0")
        self.assertNotEqual(
            full["--checkpoints_path"], ref_only["--checkpoints_path"]
        )
        full.pop("--checkpoints_path")
        ref_only.pop("--checkpoints_path")
        full["--preference_weight"] = "0.0"
        self.assertEqual(full, ref_only)

    def test_reference_branch_and_100k_protocol_remain_enabled(self):
        flags = training_flags(
            "run_v3_reference_only_to100k_hcmr_seed0.sbatch"
        )
        self.assertEqual(flags["--preference_mode"], "hard_fork")
        self.assertEqual(flags["--reference_weight"], "0.1")
        self.assertEqual(flags["--preference_target_mode"], "high_only")
        self.assertEqual(flags["--update_steps"], "100001")
        self.assertEqual(flags["--checkpoint_steps"], "[75000,100000]")
        self.assertEqual(flags["--eval_episodes"], "100")
        self.assertEqual(flags["--reward_mode"], "delayed")


if __name__ == "__main__":
    unittest.main()
