import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from trajflow_vsr.training.runner import TrainingRunner  # noqa: E402
from trajflow_vsr.utils.config import apply_overrides, load_config  # noqa: E402


class StageBCpuTrainingTest(unittest.TestCase):
    def test_one_step_stage_b_training_on_cpu(self):
        config = apply_overrides(
            load_config("configs/train/stage_b_deterministic.yaml"),
            [
                "runtime.dry_run=false",
                "runtime.device=cpu",
                "data.batch_size=1",
                "data.frames=2",
                "data.height=6",
                "data.width=6",
                "model.hidden_channels=8",
                "model.transport.sinkhorn_iterations=4",
                "model.transport.bridge_steps=3",
                "optimizer.max_steps=1",
            ],
        )
        result = TrainingRunner(config).run()
        self.assertEqual(len(result["history"]), 1)
        self.assertGreater(result["history"][0]["total"], 0.0)
        self.assertIn("schrodinger_bridge", result["history"][0])
        self.assertIn("temporal", result["history"][0])
        self.assertIn("trajectory", result["history"][0])


if __name__ == "__main__":
    unittest.main()
