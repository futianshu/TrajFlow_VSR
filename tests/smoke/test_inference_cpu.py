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

from trajflow_vsr.inference import InferenceRunner  # noqa: E402
from trajflow_vsr.utils.config import apply_overrides, load_config  # noqa: E402


class InferenceCpuTest(unittest.TestCase):
    def test_synthetic_inference_on_cpu(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = apply_overrides(
                load_config("configs/infer/synthetic.yaml"),
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
                    "inference.export_visualization=true",
                    "inference.max_visualization_frames=1",
                    "inference.top_edges=4",
                    f"inference.output_dir={tmpdir}",
                ],
            )
            result = InferenceRunner(config).run()
            self.assertEqual(result["mode"], "offline")
            self.assertEqual(result["scale"], 2.0)
            self.assertTrue(Path(result["files"]["manifest"][0]).exists())
            self.assertTrue(Path(result["files"]["hr_frames"][0]).exists())
            self.assertIn("diagnostic_trajectory_graph", result["files"])

    def test_inference_dry_run_summary(self):
        summary = InferenceRunner(load_config("configs/infer/synthetic.yaml")).summarize()
        self.assertEqual(summary.inference["mode"], "offline")
        self.assertEqual(summary.data["name"], "synthetic")

    def test_inference_exports_posterior_samples_on_cpu(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = apply_overrides(
                load_config("configs/infer/synthetic.yaml"),
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
                    "inference.export_visualization=false",
                    "inference.posterior_samples=2",
                    f"inference.output_dir={tmpdir}",
                ],
            )

            result = InferenceRunner(config).run()

            self.assertEqual(result["posterior_samples"], 2)
            self.assertIn("posterior_sample_frames", result["files"])
            self.assertTrue(Path(result["files"]["posterior_sample_frames"][0]).exists())


if __name__ == "__main__":
    unittest.main()
