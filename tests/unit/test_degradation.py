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

from trajflow_vsr.data import load_frame_manifest, make_frame_manifest_batch, make_stage_a_frame_manifest_batch  # noqa: E402
from trajflow_vsr.data.degradation import (  # noqa: E402
    apply_realistic_degradation,
    build_degraded_frame_dataset,
    degradation_profile_from_config,
    write_training_frame,
)
from trajflow_vsr.utils.config import load_config  # noqa: E402
from trajflow_vsr.utils.torch_utils import require_torch  # noqa: E402


class DegradationTest(unittest.TestCase):
    def setUp(self):
        self.torch = require_torch()

    def test_apply_realistic_degradation_shapes(self):
        profile = degradation_profile_from_config({"profile": "codec_motion", "scale": 2.0})
        video = self.torch.rand(1, 3, 3, 12, 14)
        result = apply_realistic_degradation(video, profile=profile)

        self.assertEqual(tuple(result["lr"].shape), (1, 3, 3, 6, 7))
        self.assertEqual(tuple(result["clean_lr"].shape), (1, 3, 3, 6, 7))
        self.assertEqual(tuple(result["artifact"].shape), (1, 3, 1, 6, 7))
        self.assertEqual(tuple(result["reliability"].shape), (1, 3, 1, 6, 7))
        self.assertEqual(tuple(result["degradation"].shape), (1, 8))
        self.assertEqual(result["profile"]["name"], "codec_motion")

    def test_build_degraded_dataset_and_paired_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            hr_root = root / "hr"
            lr_root = root / "lr"
            manifest_path = root / "manifest.json"
            self._write_hr_sequence(hr_root / "seq_a", frame_count=3)

            summary = build_degraded_frame_dataset(
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
                seed=7,
            )
            manifest = load_frame_manifest(manifest_path)
            batch = make_frame_manifest_batch(
                {
                    "name": "frame_manifest",
                    "manifest_path": str(manifest_path),
                    "batch_size": 1,
                    "frames": 2,
                    "scale": 2.0,
                },
                device="cpu",
            )
            stage_a_batch = make_stage_a_frame_manifest_batch(
                {
                    "name": "frame_manifest",
                    "manifest_path": str(manifest_path),
                    "batch_size": 1,
                    "frames": 2,
                    "scale": 2.0,
                },
                device="cpu",
            )

            self.assertEqual(summary["sequences"], 1)
            self.assertEqual(summary["frames"], 3)
            self.assertEqual(summary["clips"], 2)
            self.assertTrue((lr_root / "seq_a" / "0000.png").exists())
            self.assertTrue(manifest["paired"])
            self.assertEqual(manifest["degradation"]["name"], "mild_real")
            self.assertEqual(tuple(batch["lr"].shape), (1, 2, 3, 5, 6))
            self.assertEqual(tuple(batch["hr"].shape), (1, 2, 3, 10, 12))
            self.assertTrue(batch["metadata"][0]["paired_hr"])
            self.assertEqual(tuple(stage_a_batch["clean_lr"].shape), (1, 2, 3, 5, 6))
            self.assertEqual(tuple(stage_a_batch["artifact"].shape), (1, 2, 1, 5, 6))
            self.assertEqual(tuple(stage_a_batch["reliability"].shape), (1, 2, 1, 5, 6))
            self.assertEqual(tuple(stage_a_batch["degradation"].shape), (1, 8))
            self.assertGreater(float(stage_a_batch["degradation"][0, 0]), 0.0)

    def test_degraded_manifest_reuses_sequence_glob_for_hr_pairing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            hr_root = root / "hr"
            lr_root = root / "lr"
            manifest_path = root / "manifest.json"
            self._write_hr_sequence(hr_root / "keep", frame_count=3)
            self._write_hr_sequence(hr_root / "skip", frame_count=3)

            summary = build_degraded_frame_dataset(
                hr_root=hr_root,
                lr_output_root=lr_root,
                profile={"profile": "bicubic", "scale": 2.0},
                manifest_output=manifest_path,
                dataset="unit",
                split="train",
                sequence_glob="keep",
                clip_length=2,
                stride=1,
                min_frames=2,
                overwrite=True,
            )
            manifest = load_frame_manifest(manifest_path)

            self.assertEqual(summary["sequences"], 1)
            self.assertEqual(summary["clips"], 2)
            self.assertEqual(manifest["sequence_glob"], "keep")
            self.assertEqual(manifest["hr_sequence_glob"], "keep")
            self.assertEqual([sequence["sequence_id"] for sequence in manifest["sequences"]], ["keep"])

    def test_degradation_config_template_parses(self):
        config = load_config("configs/data/degradation_mild_real.yaml")
        self.assertEqual(config["degradation"]["profile"], "mild_real")
        self.assertEqual(config["prepare"]["layout"], "generic")
        self.assertEqual(config["data"]["name"], "frame_manifest")

    def _write_hr_sequence(self, sequence_dir: Path, frame_count: int) -> None:
        sequence_dir.mkdir(parents=True)
        for frame_idx in range(frame_count):
            image = self.torch.rand(3, 10, 12)
            write_training_frame(sequence_dir / f"{frame_idx:04d}.png", image)


if __name__ == "__main__":
    unittest.main()
