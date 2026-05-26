import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from trajflow_vsr.experiments import AblationRunner  # noqa: E402
from trajflow_vsr.utils.config import load_config  # noqa: E402


class AblationRunnerTest(unittest.TestCase):
    def test_stage_b_scan_policy_grid_summary(self):
        runner = AblationRunner(load_config("configs/ablation/stage_b_scan_policy_grid.yaml"))
        summary = runner.summarize(variants=["raster", "ot_sb_no_koopman"])

        self.assertEqual(summary.name, "stage_b_scan_policy_grid")
        self.assertEqual(summary.runner_type, "training")
        self.assertEqual(summary.evaluation_config, "configs/eval/offline.yaml")
        self.assertEqual(len(summary.variants), 2)
        self.assertIn("model.memory.scan_policy=raster", summary.variants[0]["overrides"])
        self.assertIn("losses.koopman=0.0", summary.variants[1]["overrides"])

        config = load_config("configs/ablation/stage_b_scan_policy_grid.yaml")
        self.assertEqual(config["ablation"]["selection"]["directions"]["psnr"], "max")
        self.assertEqual(config["ablation"]["selection"]["directions"]["fps"], "max")
        self.assertGreater(config["ablation"]["selection"]["weights"]["psnr"], 0.0)

    def test_stage_b_context_length_grid_summary(self):
        runner = AblationRunner(load_config("configs/ablation/stage_b_context_length_grid.yaml"))
        summary = runner.summarize(variants=["frames_3", "frames_7"])

        self.assertEqual(summary.name, "stage_b_context_length_grid")
        self.assertEqual(summary.runner_type, "training")
        self.assertEqual(len(summary.variants), 2)
        self.assertIn("data.frames=3", summary.variants[0]["overrides"])
        self.assertIn("data.frames=7", summary.variants[1]["overrides"])

    def test_stage_b_context_scan_matrix_summary(self):
        runner = AblationRunner(load_config("configs/ablation/stage_b_context_scan_matrix.yaml"))
        summary = runner.summarize(variants=["frames_3_x_raster", "frames_7_x_ot_sb"])

        self.assertEqual(summary.name, "stage_b_context_scan_matrix")
        self.assertEqual(summary.runner_type, "training")
        self.assertEqual(len(summary.variants), 2)
        self.assertIn("data.frames=3", summary.variants[0]["overrides"])
        self.assertIn("model.memory.scan_policy=raster", summary.variants[0]["overrides"])
        self.assertIn("data.frames=7", summary.variants[1]["overrides"])
        self.assertIn("model.memory.scan_policy=ot_sb", summary.variants[1]["overrides"])

    def test_stage_b_context_scan_matrix_max_variants(self):
        runner = AblationRunner(load_config("configs/ablation/stage_b_context_scan_matrix.yaml"))
        summary = runner.summarize(max_variants=3)

        self.assertEqual([variant["name"] for variant in summary.variants], [
            "frames_3_x_raster",
            "frames_3_x_hilbert",
            "frames_3_x_content",
        ])

    def test_stage_b_transport_curriculum_summary(self):
        runner = AblationRunner(load_config("configs/ablation/stage_b_transport_curriculum_grid.yaml"))
        summary = runner.summarize(variants=["ot_sb_no_curriculum", "lowtemp_radius2_warmup"])

        self.assertEqual(summary.name, "stage_b_transport_curriculum_grid")
        self.assertEqual(summary.runner_type, "training")
        self.assertEqual(len(summary.variants), 2)
        self.assertIn("curriculum.enabled=false", summary.variants[0]["overrides"])
        self.assertIn("model.transport.temperature=0.1", summary.variants[1]["overrides"])

    def test_stage_b_transport_two_phase_summary(self):
        runner = AblationRunner(load_config("configs/ablation/stage_b_transport_two_phase_grid.yaml"))
        summary = runner.summarize(variants=["lowtemp_radius2_no_curriculum_100", "two_phase25_light_recovery"])

        self.assertEqual(summary.name, "stage_b_transport_two_phase_grid")
        self.assertEqual(summary.runner_type, "training")
        self.assertEqual(len(summary.variants), 2)
        self.assertIn("curriculum.enabled=false", summary.variants[0]["overrides"])
        self.assertIn("curriculum.phases.reconstruction_recovery.start_step=25", summary.variants[1]["overrides"])


if __name__ == "__main__":
    unittest.main()
