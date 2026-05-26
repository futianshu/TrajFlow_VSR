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

from trajflow_vsr.data import build_frame_manifest, discover_frame_sequences, load_frame_manifest, make_frame_manifest_batch  # noqa: E402
from trajflow_vsr.data.factory import describe_data  # noqa: E402
from trajflow_vsr.utils.config import load_config  # noqa: E402
from trajflow_vsr.utils.torch_utils import require_torch  # noqa: E402
from trajflow_vsr.visualization import write_image  # noqa: E402


class DataManifestTest(unittest.TestCase):
    def setUp(self):
        self.torch = require_torch()

    def test_build_and_load_frame_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = self._make_sequence_root(Path(tmpdir))
            sequences = discover_frame_sequences(root, min_frames=3)
            self.assertEqual(len(sequences), 2)

            output = Path(tmpdir) / "manifest.json"
            manifest = build_frame_manifest(root, output, dataset="unit", split="train", clip_length=3, stride=2)
            loaded = load_frame_manifest(output)

            self.assertEqual(manifest["dataset"], "unit")
            self.assertEqual(loaded["split"], "train")
            self.assertEqual(len(loaded["sequences"]), 2)
            self.assertEqual(len(loaded["clips"]), 4)
            self.assertTrue(loaded["clips"][0]["frame_indices"])
            summary = describe_data({"name": "frame_manifest", "manifest_path": str(output), "batch_size": 1, "frames": 3})
            self.assertEqual(summary["sequences"], 2)
            self.assertEqual(summary["clips"], 4)

    def test_make_frame_manifest_batch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = self._make_sequence_root(Path(tmpdir))
            output = Path(tmpdir) / "manifest.json"
            build_frame_manifest(root, output, dataset="unit", split="train", clip_length=3, stride=1)

            batch = make_frame_manifest_batch(
                {
                    "name": "frame_manifest",
                    "manifest_path": str(output),
                    "batch_size": 1,
                    "frames": 2,
                    "height": 4,
                    "width": 6,
                    "scale": 2.0,
                },
                device="cpu",
            )
            self.assertEqual(tuple(batch["lr"].shape), (1, 2, 3, 4, 6))
            self.assertEqual(tuple(batch["hr"].shape), (1, 2, 3, 8, 12))
            self.assertEqual(batch["metadata"][0]["frame_indices"], [0, 1])

    def test_require_paired_rejects_unpaired_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = self._make_sequence_root(Path(tmpdir))
            output = Path(tmpdir) / "manifest.json"
            build_frame_manifest(root, output, dataset="unit", split="train", clip_length=3, stride=1)

            with self.assertRaisesRegex(ValueError, "requires paired HR targets"):
                make_frame_manifest_batch(
                    {
                        "name": "frame_manifest",
                        "manifest_path": str(output),
                        "batch_size": 1,
                        "frames": 2,
                        "scale": 2.0,
                        "require_paired": True,
                    },
                    device="cpu",
                )

    def test_paired_frame_manifest_uses_hr_target_frames(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            lr_root = root / "lr"
            hr_root = root / "hr"
            self._write_constant_frames(lr_root / "seq", frame_count=3, height=5, width=7, value=0.1)
            self._write_constant_frames(hr_root / "seq", frame_count=3, height=11, width=13, value=0.8)
            output = root / "paired_manifest.json"

            manifest = build_frame_manifest(
                lr_root,
                output,
                dataset="unit_paired",
                split="train",
                clip_length=2,
                stride=1,
                hr_root=hr_root,
            )
            batch = make_frame_manifest_batch(
                {
                    "name": "frame_manifest",
                    "manifest_path": str(output),
                    "batch_size": 1,
                    "frames": 2,
                    "scale": 2.0,
                },
                device="cpu",
            )

            self.assertTrue(manifest["paired"])
            self.assertEqual(len(manifest["sequences"][0]["hr_frames"]), 3)
            self.assertEqual(tuple(batch["lr"].shape), (1, 2, 3, 5, 7))
            self.assertEqual(tuple(batch["hr"].shape), (1, 2, 3, 11, 13))
            self.assertTrue(batch["metadata"][0]["paired_hr"])

    def test_vimeo90k_split_file_layout(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_frames(root / "sequences" / "00001" / "0001", frame_count=4)
            self._write_frames(root / "sequences" / "00002" / "0001", frame_count=4)
            split_file = root / "sep_trainlist.txt"
            split_file.write_text("00001/0001\n", encoding="utf-8")
            manifest_path = root / "vimeo_manifest.json"

            manifest = build_frame_manifest(
                root,
                manifest_path,
                dataset="vimeo90k",
                split="train",
                layout="vimeo90k",
                split_file=split_file,
                clip_length=2,
                stride=1,
            )

            self.assertEqual(manifest["layout"], "vimeo90k")
            self.assertEqual(manifest["split_file"], str(split_file))
            self.assertEqual(len(manifest["sequences"]), 1)
            self.assertEqual(len(manifest["clips"]), 3)
            self.assertEqual(manifest["sequences"][0]["sequence_id"], "sequences/00001/0001")

    def test_dataset_layout_templates_parse(self):
        for path, layout in [
            ("configs/data/vimeo90k.yaml", "vimeo90k"),
            ("configs/data/reds.yaml", "reds"),
            ("configs/data/vid4.yaml", "vid4"),
            ("configs/data/udm10.yaml", "udm10"),
            ("configs/data/spmcs.yaml", "generic"),
            ("configs/data/realvsr.yaml", "generic"),
            ("configs/data/videolq.yaml", "generic"),
        ]:
            config = load_config(path)
            self.assertEqual(config["prepare"]["layout"], layout)
            self.assertEqual(config["data"]["layout"], layout)

        videolq = load_config("configs/data/videolq.yaml")
        self.assertFalse(videolq["data"]["paired"])
        self.assertEqual(videolq["data"]["protocol"], "no_reference_qualitative")

    def _make_sequence_root(self, root: Path) -> Path:
        for sequence_name in ["seq_a", "seq_b"]:
            self._write_frames(root / sequence_name, frame_count=5)
        return root

    def _write_frames(self, sequence_dir: Path, frame_count: int) -> None:
        sequence_dir.mkdir(parents=True)
        for frame_idx in range(frame_count):
            image = self.torch.rand(3, 5, 7)
            write_image(sequence_dir / f"{frame_idx:04d}.png", image)

    def _write_constant_frames(
        self,
        sequence_dir: Path,
        frame_count: int,
        height: int,
        width: int,
        value: float,
    ) -> None:
        sequence_dir.mkdir(parents=True)
        for frame_idx in range(frame_count):
            image = self.torch.full((3, height, width), value)
            write_image(sequence_dir / f"{frame_idx:04d}.png", image)


if __name__ == "__main__":
    unittest.main()
