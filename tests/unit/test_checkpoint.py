import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from trajflow_vsr.training import load_training_checkpoint, save_training_checkpoint  # noqa: E402
from trajflow_vsr.utils.torch_utils import require_torch  # noqa: E402


class CheckpointTest(unittest.TestCase):
    def setUp(self):
        self.torch = require_torch()

    def test_save_and_load_training_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = self.torch.nn.Linear(3, 2)
            optimizer = self.torch.optim.AdamW(model.parameters(), lr=1e-3)
            path = Path(tmpdir) / "checkpoint.pt"
            saved = save_training_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                step=3,
                summary={"stage": {"name": "unit"}},
                history=[{"step": 3, "total": 1.0}],
            )

            restored = self.torch.nn.Linear(3, 2)
            restored_optimizer = self.torch.optim.AdamW(restored.parameters(), lr=1e-3)
            info = load_training_checkpoint(saved, restored, restored_optimizer, device="cpu")

            self.assertEqual(info["step"], 3)
            self.assertEqual(info["next_step"], 4)
            self.assertTrue(info["optimizer_loaded"])
            self.assertEqual(info["history"][0]["total"], 1.0)
            for original, loaded in zip(model.parameters(), restored.parameters(), strict=True):
                self.assertTrue(self.torch.allclose(original, loaded))


if __name__ == "__main__":
    unittest.main()
