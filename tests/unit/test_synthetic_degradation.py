import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from trajflow_vsr.data.synthetic import SyntheticVideoSpec, make_controlled_motion_batch, make_stage_a_batch  # noqa: E402


class SyntheticDegradationTest(unittest.TestCase):
    def test_stage_a_batch_shapes(self):
        batch = make_stage_a_batch(
            SyntheticVideoSpec(batch_size=2, frames=3, channels=3, height=8, width=8, scale=2.0),
            device="cpu",
        )
        self.assertEqual(tuple(batch["lr"].shape), (2, 3, 3, 8, 8))
        self.assertEqual(tuple(batch["clean_lr"].shape), (2, 3, 3, 8, 8))
        self.assertEqual(tuple(batch["hr"].shape), (2, 3, 3, 16, 16))
        self.assertEqual(tuple(batch["artifact"].shape), (2, 3, 1, 8, 8))
        self.assertEqual(tuple(batch["reliability"].shape), (2, 3, 1, 8, 8))
        self.assertEqual(tuple(batch["degradation"].shape), (2, 8))

    def test_controlled_motion_batch_has_known_integer_shift(self):
        torch_batch = make_controlled_motion_batch(
            SyntheticVideoSpec(batch_size=1, frames=3, channels=1, height=8, width=8, scale=2.0),
            motion={"shift_x": 2, "shift_y": 1, "wrap": True},
            device="cpu",
        )

        self.assertEqual(tuple(torch_batch["lr"].shape), (1, 3, 1, 8, 8))
        self.assertEqual(tuple(torch_batch["hr"].shape), (1, 3, 1, 16, 16))
        self.assertEqual(torch_batch["controlled_motion"]["shift_x"], 2)
        self.assertEqual(torch_batch["controlled_motion"]["shift_y"], 1)
        self.assertTrue(torch_batch["metadata"][0]["controlled_motion"])


if __name__ == "__main__":
    unittest.main()
