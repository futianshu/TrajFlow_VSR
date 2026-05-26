import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from trajflow_vsr.data import build_frame_manifest, estimate_degradation_from_manifest  # noqa: E402
from trajflow_vsr.experiments import build_data_inventory, export_data_inventory  # noqa: E402
from trajflow_vsr.utils.torch_utils import require_torch  # noqa: E402
from trajflow_vsr.visualization import write_image  # noqa: E402


class DataInventoryTest(unittest.TestCase):
    def setUp(self):
        self.torch = require_torch()

    def test_inventory_flags_unpaired_training_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_frames(root / "frames" / "seq")
            manifest_path = root / "train_manifest.json"
            build_frame_manifest(root / "frames", manifest_path, dataset="unit", split="train", clip_length=2)

            inventory = build_data_inventory(root, manifest_paths=[str(manifest_path)])

            self.assertEqual(inventory["manifest_count"], 1)
            self.assertGreaterEqual(inventory["issue_count"], 1)
            self.assertIn("unpaired", inventory["issues"][0]["message"])

    def test_export_data_inventory_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_frames(root / "frames" / "seq")
            manifest_path = root / "test_manifest.json"
            build_frame_manifest(root / "frames", manifest_path, dataset="unit", split="test", clip_length=2)
            inventory = build_data_inventory(root, manifest_paths=[str(manifest_path)])

            export = export_data_inventory(inventory, root / "out", name="unit_inventory")

            self.assertEqual(export.manifest_count, 1)
            for path in export.files.values():
                self.assertTrue(Path(path).exists())

    def test_degradation_estimate_from_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_frames(root / "frames" / "seq", frame_count=4)
            manifest_path = root / "manifest.json"
            build_frame_manifest(root / "frames", manifest_path, dataset="unit", split="train", clip_length=2)

            estimate = estimate_degradation_from_manifest(
                manifest_path,
                profiles=["mild_real", "codec_motion"],
                scale=4.0,
                seconds_per_frame=0.1,
                shards=2,
            )

            self.assertEqual(estimate.frames, 4)
            self.assertEqual(len(estimate.profiles), 2)
            self.assertAlmostEqual(estimate.estimated_seconds, 0.8)
            self.assertEqual(len(estimate.suggested_shards), 2)

    def _write_frames(self, sequence_dir: Path, frame_count: int = 3) -> None:
        sequence_dir.mkdir(parents=True)
        for frame_idx in range(frame_count):
            write_image(sequence_dir / f"{frame_idx:04d}.png", self.torch.rand(3, 5, 7))


if __name__ == "__main__":
    unittest.main()
