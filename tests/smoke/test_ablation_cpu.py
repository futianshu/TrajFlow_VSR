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

from trajflow_vsr.experiments import AblationRunner  # noqa: E402
from trajflow_vsr.utils.config import load_config  # noqa: E402
from trajflow_vsr.utils.torch_utils import require_torch  # noqa: E402


class AblationCpuTest(unittest.TestCase):
    def test_stage_b_scan_ablation_two_variants_on_cpu(self):
        require_torch()
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = AblationRunner(load_config("configs/ablation/stage_b_scan_policy_grid.yaml"))
            result = runner.run(
                variants=["raster", "ot_sb_no_koopman"],
                output_dir=Path(tmpdir) / "ablation",
                common_overrides=[
                    "runtime.dry_run=false",
                    "runtime.device=cpu",
                    "data.batch_size=1",
                    "data.frames=2",
                    "data.height=5",
                    "data.width=5",
                    "model.hidden_channels=8",
                    "model.transport.sinkhorn_iterations=3",
                    "model.transport.bridge_steps=2",
                    "optimizer.max_steps=1",
                    "checkpoint.save_final=true",
                ],
            )

            self.assertEqual(result["variant_count"], 2)
            self.assertTrue(Path(result["summary_path"]).exists())
            self.assertEqual(result["records"][0]["name"], "raster")
            self.assertGreater(result["records"][0]["final_total"], 0.0)
            self.assertEqual(result["records"][0]["evaluation"]["status"], "ok")
            self.assertIn("offline", result["records"][0]["metrics"])
            self.assertGreater(result["records"][0]["metrics"]["offline"]["fps"], 0.0)
            self.assertEqual(result["records"][1]["name"], "ot_sb_no_koopman")
            self.assertNotIn("koopman", result["records"][1]["final_losses"])
            self.assertEqual(result["records"][1]["evaluation"]["status"], "ok")
            self.assertEqual(len(result["comparison"]["rows"]), 2)
            self.assertEqual(len(result["comparison"]["ranking"]), 2)
            self.assertIn("psnr", result["comparison"]["best_by_metric"])
            self.assertIn("fps", result["comparison"]["best_by_metric"])
            self.assertIn("profile.parameters", result["comparison"]["rows"][0])
            self.assertTrue(Path(result["comparison"]["json_path"]).exists())
            self.assertTrue(Path(result["comparison"]["csv_path"]).exists())
            self.assertTrue(Path(result["comparison"]["markdown_path"]).exists())


if __name__ == "__main__":
    unittest.main()
