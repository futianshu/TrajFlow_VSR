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


class StageACpuTrainingTest(unittest.TestCase):
    def test_one_step_stage_a_training_on_cpu(self):
        config = apply_overrides(
            load_config("configs/train/stage_a_tokenizer.yaml"),
            [
                "runtime.dry_run=false",
                "runtime.device=cpu",
                "data.batch_size=1",
                "data.frames=2",
                "data.height=8",
                "data.width=8",
                "model.hidden_channels=8",
                "optimizer.max_steps=1",
            ],
        )
        result = TrainingRunner(config).run()
        self.assertEqual(len(result["history"]), 1)
        self.assertGreater(result["history"][0]["total"], 0.0)


if __name__ == "__main__":
    unittest.main()
