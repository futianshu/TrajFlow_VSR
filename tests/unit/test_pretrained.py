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

from trajflow_vsr.models.factory import build_model, build_stage_model  # noqa: E402
from trajflow_vsr.training import load_pretrained_components, save_training_checkpoint, trainable_parameters  # noqa: E402
from trajflow_vsr.utils.torch_utils import require_torch  # noqa: E402


class PretrainedComponentsTest(unittest.TestCase):
    def setUp(self):
        self.torch = require_torch()

    def test_load_and_freeze_stage_a_components(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            stage_a = build_stage_model(
                {"name": "stage_a_tokenizer"},
                {"hidden_channels": 8, "uncertainty": {"degradation_dim": 8}},
            )
            optimizer = self.torch.optim.AdamW(stage_a.parameters(), lr=1e-4)
            checkpoint = Path(tmpdir) / "stage_a.pt"
            save_training_checkpoint(
                checkpoint,
                model=stage_a,
                optimizer=optimizer,
                step=0,
                summary={"stage": {"name": "stage_a_tokenizer"}},
                history=[{"step": 0, "total": 1.0}],
            )

            model = build_model(
                {
                    "hidden_channels": 8,
                    "transport": {"sinkhorn_iterations": 2, "bridge_steps": 2},
                }
            )
            first_key = next(key for key in stage_a.state_dict() if key.startswith("tokenizer."))
            before = model.state_dict()[first_key].clone()
            info = load_pretrained_components(
                model,
                checkpoint,
                components=["tokenizer", "uncertainty"],
                freeze_components=["tokenizer"],
                device="cpu",
            )

            self.assertTrue(info["loaded"])
            self.assertGreater(info["loaded_key_count"], 0)
            self.assertFalse(self.torch.allclose(before, model.state_dict()[first_key]))
            self.assertFalse(next(model.tokenizer.parameters()).requires_grad)
            self.assertTrue(next(model.uncertainty.parameters()).requires_grad)
            self.assertTrue(trainable_parameters(model))


if __name__ == "__main__":
    unittest.main()
