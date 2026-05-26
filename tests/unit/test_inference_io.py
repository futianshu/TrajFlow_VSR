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

from trajflow_vsr.inference import read_video_source, write_video_frames  # noqa: E402
from trajflow_vsr.utils.torch_utils import require_torch  # noqa: E402
from trajflow_vsr.visualization import write_image  # noqa: E402


class InferenceIoTest(unittest.TestCase):
    def setUp(self):
        self.torch = require_torch()

    def test_read_png_sequence_to_batched_video(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for idx in range(3):
                write_image(root / f"{idx:03d}.png", self.torch.rand(3, 5, 7))

            video = read_video_source(root, device="cpu", max_frames=2)
            self.assertEqual(tuple(video.shape), (1, 2, 3, 5, 7))
            self.assertGreaterEqual(float(video.min()), 0.0)
            self.assertLessEqual(float(video.max()), 1.0)

    def test_write_video_frames_as_png(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video = self.torch.rand(1, 2, 3, 4, 6)
            files = write_video_frames(video, tmpdir, prefix="hr", image_format="png")
            self.assertEqual(len(files), 2)
            self.assertTrue(Path(files[0]).exists())
            self.assertTrue(files[0].endswith(".png"))


if __name__ == "__main__":
    unittest.main()
