import os
import subprocess
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
from trajflow_vsr.utils.torch_utils import require_torch  # noqa: E402
from trajflow_vsr.visualization import write_image  # noqa: E402


class PrepareDataCpuTest(unittest.TestCase):
    def setUp(self):
        self.torch = require_torch()

    def test_prepare_data_cli_and_manifest_training_on_cpu(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "raw_lr"
            hr_root = Path(tmpdir) / "raw_hr"
            self._write_sequence(root / "seq", height=6, width=6)
            self._write_sequence(hr_root / "seq", height=12, width=12)
            manifest_path = Path(tmpdir) / "manifest.json"

            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/prepare_data.py",
                    "--root",
                    str(root),
                    "--output",
                    str(manifest_path),
                    "--dataset",
                    "vid4",
                    "--split",
                    "test",
                    "--layout",
                    "vid4",
                    "--hr-root",
                    str(hr_root),
                    "--clip-length",
                    "2",
                    "--stride",
                    "1",
                ],
                cwd=ROOT,
                check=True,
                text=True,
                capture_output=True,
            )
            self.assertIn('"clips"', completed.stdout)
            self.assertIn('"layout": "vid4"', completed.stdout)
            self.assertIn('"paired": true', completed.stdout)
            self.assertTrue(manifest_path.exists())

            config = apply_overrides(
                load_config("configs/train/stage_b_deterministic.yaml"),
                [
                    "runtime.dry_run=false",
                    "runtime.device=cpu",
                    "data.name=frame_manifest",
                    f"data.manifest_path={manifest_path}",
                    "data.batch_size=1",
                    "data.frames=2",
                    "data.height=4",
                    "data.width=4",
                    "model.hidden_channels=8",
                    "model.transport.sinkhorn_iterations=3",
                    "model.transport.bridge_steps=2",
                    "optimizer.max_steps=1",
                    f"project.output_dir={Path(tmpdir) / 'outputs'}",
                    f"checkpoint.output_dir={Path(tmpdir) / 'checkpoints'}",
                ],
            )
            result = TrainingRunner(config).run()
            self.assertEqual(len(result["history"]), 1)
            self.assertGreater(result["history"][0]["total"], 0.0)

    def _write_sequence(self, directory: Path, height: int, width: int) -> None:
        directory.mkdir(parents=True)
        for frame_idx in range(3):
            write_image(directory / f"{frame_idx:04d}.png", self.torch.rand(3, height, width))


if __name__ == "__main__":
    unittest.main()
