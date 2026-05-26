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

from trajflow_vsr.data.degradation import build_degraded_frame_dataset, write_training_frame  # noqa: E402
from trajflow_vsr.data.mixed import (  # noqa: E402
    describe_stage_a_mixed_config,
    make_stage_a_mixed_batch,
    resolve_stage_a_mixed_source,
)
from trajflow_vsr.utils.torch_utils import require_torch  # noqa: E402


class StageAMixedDataTest(unittest.TestCase):
    def setUp(self):
        self.torch = require_torch()

    def test_alternating_schedule_resolves_sources(self):
        config = {"name": "stage_a_mixed", "schedule": "alternating", "synthetic_steps": 2, "real_steps": 1}

        self.assertEqual(resolve_stage_a_mixed_source(config, step=0), "synthetic")
        self.assertEqual(resolve_stage_a_mixed_source(config, step=1), "synthetic")
        self.assertEqual(resolve_stage_a_mixed_source(config, step=2), "real")
        self.assertEqual(resolve_stage_a_mixed_source(config, step=3), "synthetic")

    def test_make_stage_a_mixed_batch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            hr_root = root / "hr"
            lr_root = root / "lr"
            manifest_path = root / "manifest.json"
            self._write_sequence(hr_root / "seq")
            build_degraded_frame_dataset(
                hr_root=hr_root,
                lr_output_root=lr_root,
                profile={"profile": "mild_real", "scale": 2.0, "noise_std": 0.0},
                manifest_output=manifest_path,
                dataset="unit",
                split="train",
                clip_length=2,
                stride=1,
                min_frames=2,
                overwrite=True,
            )
            config = {
                "name": "stage_a_mixed",
                "schedule": "alternating",
                "synthetic_steps": 1,
                "real_steps": 1,
                "synthetic": {
                    "name": "synthetic",
                    "batch_size": 1,
                    "frames": 2,
                    "channels": 3,
                    "height": 5,
                    "width": 6,
                    "scale": 2.0,
                },
                "real": {
                    "name": "frame_manifest",
                    "manifest_path": str(manifest_path),
                    "batch_size": 1,
                    "frames": 2,
                    "scale": 2.0,
                },
            }

            summary = describe_stage_a_mixed_config(config)
            synthetic_batch = make_stage_a_mixed_batch(config, step=0, device="cpu")
            real_batch = make_stage_a_mixed_batch(config, step=1, device="cpu")

            self.assertEqual(summary["name"], "stage_a_mixed")
            self.assertEqual(synthetic_batch["source"], "synthetic")
            self.assertEqual(real_batch["source"], "real")
            for key in ["clean_lr", "artifact", "reliability", "degradation"]:
                self.assertIn(key, synthetic_batch)
                self.assertIn(key, real_batch)

    def _write_sequence(self, sequence_dir: Path) -> None:
        sequence_dir.mkdir(parents=True)
        for frame_idx in range(3):
            write_training_frame(sequence_dir / f"{frame_idx:04d}.png", self.torch.rand(3, 10, 12))


if __name__ == "__main__":
    unittest.main()
