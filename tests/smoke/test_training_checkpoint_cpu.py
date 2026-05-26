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


class TrainingCheckpointCpuTest(unittest.TestCase):
    def test_checkpoint_save_and_resume_on_cpu(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            first = self._small_stage_b_config(
                root,
                [
                    "optimizer.max_steps=1",
                    "checkpoint.save_every_steps=1",
                    "checkpoint.save_final=true",
                ],
            )
            first_result = TrainingRunner(first).run()
            final_checkpoint = first_result["artifacts"]["final_checkpoint"]
            self.assertTrue(Path(final_checkpoint).exists())
            self.assertTrue(Path(first_result["artifacts"]["history"]).exists())
            self.assertEqual(len(first_result["artifacts"]["step_checkpoints"]), 1)

            resumed = self._small_stage_b_config(
                root,
                [
                    "optimizer.max_steps=2",
                    "checkpoint.save_final=true",
                    f"checkpoint.resume_path={final_checkpoint}",
                ],
            )
            resumed_result = TrainingRunner(resumed).run()
            self.assertTrue(resumed_result["resume"]["loaded"])
            self.assertEqual(resumed_result["resume"]["next_step"], 1)
            self.assertEqual([item["step"] for item in resumed_result["history"]], [0, 1])
            self.assertTrue(Path(resumed_result["artifacts"]["manifest"]).exists())

    def test_validation_best_checkpoint_and_scheduler_on_cpu(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = self._small_stage_b_config(
                root,
                [
                    "optimizer.max_steps=1",
                    "optimizer.gradient_accumulation_steps=1",
                    "optimizer.max_grad_norm=0.25",
                    "scheduler.name=cosine",
                    "scheduler.eta_min=0.0",
                    "validation.enabled=true",
                    "validation.every_steps=1",
                    "validation.config=configs/eval/offline.yaml",
                    "validation.metric=psnr",
                    "validation.direction=max",
                    "validation.mode=offline",
                    "validation.overrides=['data.batch_size=1', 'data.frames=2', 'data.height=4', 'data.width=4', 'data.scale=2.0']",
                    "checkpoint.save_final=true",
                ],
            )

            result = TrainingRunner(config).run()

            self.assertEqual(len(result["validation"]), 1)
            self.assertEqual(result["validation"][0]["status"], "ok")
            self.assertIn("psnr", result["validation"][0]["metrics"])
            self.assertIsNotNone(result["best_metric"])
            self.assertTrue(Path(result["artifacts"]["best_checkpoint"]).exists())
            self.assertTrue(Path(result["artifacts"]["final_checkpoint"]).exists())

    def _small_stage_b_config(self, root: Path, extra_overrides: list[str]):
        return apply_overrides(
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
                f"project.output_dir={root / 'outputs'}",
                f"checkpoint.output_dir={root / 'checkpoints'}",
                *extra_overrides,
            ],
        )


if __name__ == "__main__":
    unittest.main()
