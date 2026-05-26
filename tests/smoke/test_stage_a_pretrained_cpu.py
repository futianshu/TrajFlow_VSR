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

from trajflow_vsr.training.runner import TrainingRunner  # noqa: E402
from trajflow_vsr.utils.config import apply_overrides, load_config  # noqa: E402


class StageAPretrainedCpuTest(unittest.TestCase):
    def test_stage_b_loads_stage_a_tokenizer_uncertainty_on_cpu(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            stage_a_config = apply_overrides(
                load_config("configs/train/stage_a_tokenizer.yaml"),
                [
                    "runtime.dry_run=false",
                    "runtime.device=cpu",
                    "data.batch_size=1",
                    "data.frames=2",
                    "data.height=6",
                    "data.width=6",
                    "model.hidden_channels=8",
                    "optimizer.max_steps=1",
                    "checkpoint.save_final=true",
                    f"project.output_dir={root / 'stage_a_outputs'}",
                    f"checkpoint.output_dir={root / 'stage_a_checkpoints'}",
                ],
            )
            stage_a_result = TrainingRunner(stage_a_config).run()
            stage_a_checkpoint = stage_a_result["artifacts"]["final_checkpoint"]
            self.assertTrue(Path(stage_a_checkpoint).exists())

            stage_b_config = apply_overrides(
                load_config("configs/train/stage_b_deterministic.yaml"),
                [
                    "runtime.dry_run=false",
                    "runtime.device=cpu",
                    "data.batch_size=1",
                    "data.frames=2",
                    "data.height=4",
                    "data.width=4",
                    "model.hidden_channels=8",
                    "model.transport.sinkhorn_iterations=3",
                    "model.transport.bridge_steps=2",
                    f"model.pretrained.path={stage_a_checkpoint}",
                    "model.pretrained.freeze_components=[\"tokenizer\", \"uncertainty\"]",
                    "optimizer.max_steps=1",
                    f"project.output_dir={root / 'stage_b_outputs'}",
                    f"checkpoint.output_dir={root / 'stage_b_checkpoints'}",
                ],
            )
            stage_b_result = TrainingRunner(stage_b_config).run()

            self.assertEqual(len(stage_b_result["history"]), 1)
            self.assertTrue(stage_b_result["pretrained"]["loaded"])
            self.assertGreater(stage_b_result["pretrained"]["loaded_key_count"], 0)
            self.assertTrue(stage_b_result["pretrained"]["frozen_parameters"])


if __name__ == "__main__":
    unittest.main()
