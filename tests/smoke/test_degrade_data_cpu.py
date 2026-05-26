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

from trajflow_vsr.data.degradation import write_training_frame  # noqa: E402
from trajflow_vsr.training.runner import TrainingRunner  # noqa: E402
from trajflow_vsr.utils.config import apply_overrides, load_config  # noqa: E402
from trajflow_vsr.utils.torch_utils import require_torch  # noqa: E402


class DegradeDataCpuTest(unittest.TestCase):
    def setUp(self):
        self.torch = require_torch()

    def test_degrade_data_cli_and_manifest_training_on_cpu(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            hr_root = root / "hr"
            lr_root = root / "lr"
            manifest_path = root / "manifest.json"
            self._write_sequence(hr_root / "seq")

            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/degrade_data.py",
                    "--hr-root",
                    str(hr_root),
                    "--lr-output-root",
                    str(lr_root),
                    "--manifest-output",
                    str(manifest_path),
                    "--dataset",
                    "unit",
                    "--split",
                    "train",
                    "--profile",
                    "bicubic",
                    "--scale",
                    "2",
                    "--clip-length",
                    "2",
                    "--stride",
                    "1",
                    "--overwrite",
                ],
                cwd=ROOT,
                check=True,
                text=True,
                capture_output=True,
            )
            self.assertIn('"profile"', completed.stdout)
            self.assertIn('"clips": 2', completed.stdout)
            self.assertTrue((lr_root / "seq" / "0000.png").exists())
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
                    "data.scale=2.0",
                    "model.hidden_channels=8",
                    "model.transport.sinkhorn_iterations=3",
                    "model.transport.bridge_steps=2",
                    "optimizer.max_steps=1",
                    f"project.output_dir={root / 'outputs'}",
                    f"checkpoint.output_dir={root / 'checkpoints'}",
                ],
            )
            result = TrainingRunner(config).run()
            self.assertEqual(len(result["history"]), 1)
            self.assertGreater(result["history"][0]["total"], 0.0)

            stage_a_config = apply_overrides(
                load_config("configs/train/stage_a_tokenizer.yaml"),
                [
                    "runtime.dry_run=false",
                    "runtime.device=cpu",
                    "data.name=frame_manifest",
                    f"data.manifest_path={manifest_path}",
                    "data.batch_size=1",
                    "data.frames=2",
                    "data.scale=2.0",
                    "model.hidden_channels=8",
                    "optimizer.max_steps=1",
                    f"project.output_dir={root / 'stage_a_outputs'}",
                    f"checkpoint.output_dir={root / 'stage_a_checkpoints'}",
                ],
            )
            stage_a_result = TrainingRunner(stage_a_config).run()
            self.assertEqual(len(stage_a_result["history"]), 1)
            self.assertGreater(stage_a_result["history"][0]["total"], 0.0)

            mixed_config = apply_overrides(
                load_config("configs/train/stage_a_mixed.yaml"),
                [
                    "runtime.dry_run=false",
                    "runtime.device=cpu",
                    "data.synthetic.batch_size=1",
                    "data.synthetic.frames=2",
                    "data.synthetic.height=5",
                    "data.synthetic.width=6",
                    "data.synthetic.scale=2.0",
                    f"data.real.manifest_path={manifest_path}",
                    "data.real.batch_size=1",
                    "data.real.frames=2",
                    "data.real.scale=2.0",
                    "model.hidden_channels=8",
                    "optimizer.max_steps=2",
                    f"project.output_dir={root / 'mixed_outputs'}",
                    f"checkpoint.output_dir={root / 'mixed_checkpoints'}",
                ],
            )
            mixed_result = TrainingRunner(mixed_config).run()
            self.assertEqual([item["data_source"] for item in mixed_result["history"]], ["synthetic", "real"])

    def _write_sequence(self, sequence_dir: Path) -> None:
        sequence_dir.mkdir(parents=True)
        for frame_idx in range(3):
            write_training_frame(sequence_dir / f"{frame_idx:04d}.png", self.torch.rand(3, 10, 12))


if __name__ == "__main__":
    unittest.main()
