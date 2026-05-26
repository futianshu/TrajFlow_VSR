import json
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

from trajflow_vsr.utils.torch_utils import require_torch  # noqa: E402
from trajflow_vsr.visualization import (  # noqa: E402
    export_trajectory_maps,
    export_uncertainty_maps,
    export_visualization_bundle,
    scalar_heatmap,
    write_image,
    write_ppm,
)


class VisualizationExportTest(unittest.TestCase):
    def setUp(self):
        self.torch = require_torch()

    def test_write_ppm_exports_binary_rgb_image(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image = self.torch.rand(3, 4, 5)
            path = write_ppm(Path(tmpdir) / "image.ppm", image)
            data = path.read_bytes()
            self.assertTrue(data.startswith(b"P6\n5 4\n255\n"))
            self.assertGreater(len(data), len(b"P6\n5 4\n255\n"))

    def test_write_image_exports_png(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_image(Path(tmpdir) / "image.png", self.torch.rand(3, 4, 5))
            self.assertTrue(path.exists())
            self.assertTrue(path.read_bytes().startswith(b"\x89PNG"))

    def test_scalar_heatmap_returns_rgb_chw(self):
        heatmap = scalar_heatmap(self.torch.rand(4, 5))
        self.assertEqual(tuple(heatmap.shape), (3, 4, 5))
        self.assertGreaterEqual(float(heatmap.min()), 0.0)
        self.assertLessEqual(float(heatmap.max()), 1.0)

    def test_exports_uncertainty_maps_and_summary(self):
        outputs = {"uncertainty": self._fake_uncertainty()}
        with tempfile.TemporaryDirectory() as tmpdir:
            files = export_uncertainty_maps(outputs, tmpdir, max_frames=1)
            self.assertTrue(Path(files["artifact"][0]).exists())
            self.assertTrue(Path(files["reliability"][0]).exists())
            summary = json.loads(Path(files["summary"][0]).read_text(encoding="utf-8"))
            self.assertIn("artifact", summary)
            self.assertIn("mean", summary["reliability"])

    def test_exports_trajectory_maps_and_graph(self):
        outputs = {"transport": self._fake_transport()}
        with tempfile.TemporaryDirectory() as tmpdir:
            files = export_trajectory_maps(outputs, tmpdir, max_frames=1, top_edges=3)
            self.assertTrue(Path(files["target_frame"][0]).exists())
            graph = json.loads(Path(files["graph"][0]).read_text(encoding="utf-8"))
            self.assertEqual(graph["token_count"], 6)
            self.assertEqual(len(graph["top_edges"]), 3)

    def test_exports_visualization_bundle(self):
        outputs = {
            "hr": self.torch.rand(1, 2, 3, 4, 6),
            "hr_raw": self.torch.rand(1, 2, 3, 4, 6),
            "uncertainty": self._fake_uncertainty(),
            "transport": self._fake_transport(),
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            files = export_visualization_bundle(outputs, tmpdir, max_frames=1, top_edges=2)
            self.assertIn("uncertainty_summary", files)
            self.assertIn("trajectory_graph", files)
            self.assertIn("sample_hr", files)

    def _fake_uncertainty(self):
        base = self.torch.linspace(0.0, 1.0, steps=12).reshape(1, 2, 1, 2, 3)
        return {
            "artifact": base,
            "reliability": 1.0 - base,
            "motion_uncertainty": base.flip(-1),
            "texture_uncertainty": base.flip(-2),
        }

    def _fake_transport(self):
        plan = self.torch.eye(6).unsqueeze(0)
        grid = self.torch.zeros(1, 2, 1, 3, 4)
        return {
            "transport_plan": plan,
            "bridge_grid": grid,
            "causal": True,
            "causal_violation": self.torch.tensor(0.0),
        }


if __name__ == "__main__":
    unittest.main()
