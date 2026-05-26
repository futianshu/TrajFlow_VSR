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

from trajflow_vsr.evaluation import EvaluationRunner  # noqa: E402
from trajflow_vsr.data import build_frame_manifest  # noqa: E402
from trajflow_vsr.data.degradation import build_degraded_frame_dataset, write_training_frame  # noqa: E402
from trajflow_vsr.training.runner import TrainingRunner  # noqa: E402
from trajflow_vsr.utils.config import apply_overrides, load_config  # noqa: E402
from trajflow_vsr.utils.torch_utils import require_torch  # noqa: E402


class EvaluationCpuTest(unittest.TestCase):
    def setUp(self):
        self.torch = require_torch()

    def test_offline_eval_on_cpu(self):
        config = self._small_eval_config("configs/eval/offline.yaml")
        result = EvaluationRunner(config).run()
        self.assertIn("offline", result["metrics"])
        self.assertIn("psnr", result["metrics"]["offline"])
        self.assertIn("ssim", result["metrics"]["offline"])
        self.assertIn("fps", result["metrics"]["offline"])
        self.assertGreater(result["metrics"]["offline"]["fps"], 0.0)
        self.assertGreater(result["profile"]["parameters"], 0)
        self.assertGreater(result["profile"]["macs"], 0)

    def test_streaming_eval_on_cpu(self):
        config = self._small_eval_config("configs/eval/streaming.yaml")
        result = EvaluationRunner(config).run()
        self.assertIn("streaming", result["metrics"])
        self.assertIn("causal_violation", result["metrics"]["streaming"])

    def test_streaming_eval_dry_run_summary(self):
        summary = EvaluationRunner(load_config("configs/eval/streaming.yaml")).summarize()
        self.assertEqual(summary.evaluation["mode"], "streaming")
        self.assertEqual(summary.data["name"], "synthetic")

    def test_official_metric_status_and_posterior_samples_on_cpu(self):
        config = self._small_eval_config("configs/eval/offline.yaml")
        config = apply_overrides(
            config,
            [
                "evaluation.metric_backend=official",
                "evaluation.posterior_samples=2",
                "evaluation.official_metric_candidates=['lpips', 'dists', 'niqe', 'vmaf', 'fvd']",
            ],
        )
        result = EvaluationRunner(config).run()

        offline = result["metrics"]["offline"]
        self.assertEqual(offline["posterior_samples"], 2.0)
        self.assertIn("posterior_variance", offline)
        self.assertIn("lpips", result["metric_status"])
        self.assertIn("vmaf", result["metric_status"])

    def test_frame_manifest_eval_on_cpu(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path = self._make_degraded_manifest(root)
            config = apply_overrides(
                load_config("configs/eval/frame_manifest.yaml"),
                [
                    "runtime.dry_run=false",
                    "runtime.device=cpu",
                    f"data.manifest_path={manifest_path}",
                    "data.batch_size=1",
                    "data.frames=2",
                    "data.scale=2.0",
                    "evaluation.clip_count=2",
                    "evaluation.save_results=true",
                    f"evaluation.output_path={root / 'eval_metrics.json'}",
                    "model.hidden_channels=8",
                    "model.transport.sinkhorn_iterations=3",
                    "model.transport.bridge_steps=2",
                ],
            )
            result = EvaluationRunner(config).run()

            self.assertIn("offline", result["metrics"])
            self.assertEqual(len(result["per_clip_metrics"]["offline"]), 2)
            self.assertIn("latency_seconds", result["per_clip_metrics"]["offline"][0]["metrics"])
            self.assertTrue((root / "eval_metrics.json").exists())

    def test_no_reference_eval_on_cpu(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            lr_root = root / "lr"
            for frame_idx in range(3):
                write_training_frame(lr_root / "seq" / f"{frame_idx:04d}.png", self.torch.rand(3, 6, 8))
            manifest_path = root / "unpaired_manifest.json"

            build_frame_manifest(lr_root, manifest_path, dataset="unit_noref", split="real_test", clip_length=2, stride=1)
            config = apply_overrides(
                load_config("configs/eval/no_reference.yaml"),
                [
                    "runtime.dry_run=false",
                    "runtime.device=cpu",
                    f"data.manifest_path={manifest_path}",
                    "data.batch_size=1",
                    "data.frames=2",
                    "data.height=6",
                    "data.width=8",
                    "evaluation.clip_count=1",
                    "evaluation.metric_backend=proxy",
                    "model.hidden_channels=8",
                    "model.transport.sinkhorn_iterations=3",
                    "model.transport.bridge_steps=2",
                ],
            )
            result = EvaluationRunner(config).run()

            self.assertTrue(result["reference_free"])
            offline = result["metrics"]["offline"]
            self.assertIn("temporal_activity", offline)
            self.assertIn("spatial_sharpness", offline)
            self.assertIn("blockiness", offline)
            self.assertNotIn("psnr", offline)

    def test_eval_loads_checkpoint_on_cpu(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            train_config = apply_overrides(
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
                    "optimizer.max_steps=1",
                    "checkpoint.save_final=true",
                    f"project.output_dir={root / 'train_outputs'}",
                    f"checkpoint.output_dir={root / 'checkpoints'}",
                ],
            )
            checkpoint = TrainingRunner(train_config).run()["artifacts"]["final_checkpoint"]
            eval_config = self._small_eval_config("configs/eval/offline.yaml")
            eval_config = apply_overrides(eval_config, [f"evaluation.checkpoint_path={checkpoint}"])

            result = EvaluationRunner(eval_config).run()

            self.assertTrue(result["checkpoint"]["loaded"])
            self.assertEqual(result["checkpoint"]["path"], checkpoint)

    def _small_eval_config(self, path: str):
        return apply_overrides(
            load_config(path),
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
            ],
        )

    def _make_degraded_manifest(self, root: Path) -> Path:
        hr_root = root / "hr"
        for frame_idx in range(3):
            write_training_frame(hr_root / "seq" / f"{frame_idx:04d}.png", self.torch.rand(3, 10, 12))
        manifest_path = root / "manifest.json"
        build_degraded_frame_dataset(
            hr_root=hr_root,
            lr_output_root=root / "lr",
            profile={"profile": "bicubic", "scale": 2.0},
            manifest_output=manifest_path,
            dataset="unit_eval",
            split="test",
            clip_length=2,
            stride=1,
            min_frames=2,
            overwrite=True,
        )
        return manifest_path


if __name__ == "__main__":
    unittest.main()
