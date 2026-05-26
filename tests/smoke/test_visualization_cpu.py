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

from trajflow_vsr.utils.config import apply_overrides, load_config  # noqa: E402
from trajflow_vsr.visualization import VisualizationRunner  # noqa: E402


class VisualizationCpuTest(unittest.TestCase):
    def test_visualization_bundle_on_cpu(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = apply_overrides(
                load_config("configs/eval/visualization.yaml"),
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
                    "visualization.max_frames=1",
                    "visualization.top_edges=4",
                    "visualization.posterior_samples=1",
                    f"visualization.output_dir={tmpdir}",
                ],
            )
            result = VisualizationRunner(config).run(kind="bundle")
            self.assertEqual(result["mode"], "offline")
            self.assertTrue(Path(result["files"]["manifest"][0]).exists())
            self.assertTrue(Path(result["files"]["trajectory_graph"][0]).exists())
            self.assertTrue(Path(result["files"]["uncertainty_summary"][0]).exists())
            self.assertIn("posterior_hr", result["files"])

    def test_visualization_dry_run_summary(self):
        summary = VisualizationRunner(load_config("configs/eval/visualization.yaml")).summarize()
        self.assertEqual(summary.visualization["mode"], "offline")
        self.assertEqual(summary.data["name"], "synthetic")


if __name__ == "__main__":
    unittest.main()
