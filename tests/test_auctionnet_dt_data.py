"""Check trajectory boundaries, data quality, and the held-out split."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

SPEC = importlib.util.spec_from_file_location(
    "auctionnet_prepare_data",
    Path(__file__).resolve().parents[1] / "scripts/auctionnet_dt/prepare_data.py",
)
DATA = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DATA)


def row(advertiser, timestep, done, reward):
    return {"deliveryPeriodIndex": 7, "advertiserNumber": advertiser,
            "timeStepIndex": timestep, "state": tuple([float(timestep)] * 16),
            "action": 2.0, "reward": reward, "done": done}


class AuctionNetConversionTests(unittest.TestCase):
    def test_done_splitting_preserves_step_rewards_and_excludes_singletons(self):
        frame = pd.DataFrame([row(1, 2, 1, 99), row(1, 1, 1, 3),
                              row(1, 0, 0, 2), row(2, 0, 0, 5), row(2, 1, 1, 7)])
        trajectories, discarded = DATA.convert_frame(frame)
        self.assertEqual(len(trajectories), 2)
        self.assertEqual(discarded, 1)
        np.testing.assert_array_equal(trajectories[0]["rewards"], [2, 3])
        np.testing.assert_array_equal(trajectories[1]["rewards"], [5, 7])
        self.assertEqual(trajectories[0]["actions"].shape, (2, 1))
        self.assertEqual(trajectories[0]["observations"].shape, (2, 16))

    def test_never_stitches_advertisers_across_incomplete_episodes(self):
        frame = pd.DataFrame([row(1, 0, 0, 100), row(2, 0, 0, 2), row(2, 1, 1, 3)])
        trajectories, discarded = DATA.convert_frame(frame)
        self.assertEqual(discarded, 1)
        np.testing.assert_array_equal(trajectories[0]["rewards"], [2, 3])

    def test_nonfinite_data_rejected(self):
        frame = pd.DataFrame([row(1, 0, 0, np.nan), row(1, 1, 1, 3)])
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            DATA.convert_frame(frame)

    def test_test_periods_cannot_become_training_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "held out"):
                DATA.assemble(root, root / "output", [7, 14])
            self.assertFalse((root / "output").exists())

    def test_duplicate_periods_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "unique"):
                DATA.assemble(root, root / "output", [7, 7])


if __name__ == "__main__":
    unittest.main()
