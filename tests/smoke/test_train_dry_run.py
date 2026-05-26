import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from trajflow_vsr.training.runner import TrainingRunner  # noqa: E402
from trajflow_vsr.utils.config import load_config  # noqa: E402


class TrainDryRunTest(unittest.TestCase):
    def test_stage_a_dry_run_summary(self):
        config = load_config("configs/train/stage_a_tokenizer.yaml")
        summary = TrainingRunner(config).summarize()
        self.assertEqual(summary.stage["name"], "stage_a_tokenizer")
        self.assertEqual(summary.data["name"], "synthetic")
        self.assertEqual(summary.model["components"][0]["name"], "MultiScaleEvidenceTokenizer")

    def test_stage_c_dry_run_summary(self):
        config = load_config("configs/train/stage_c_rectified_flow.yaml")
        summary = TrainingRunner(config).summarize()
        self.assertEqual(summary.stage["name"], "stage_c_rectified_flow")
        self.assertGreater(summary.losses["flow_matching"], 0.0)
        self.assertGreater(summary.losses["bridge_consistency"], 0.0)

    def test_stage_d_dry_run_summary(self):
        config = load_config("configs/train/stage_d_distill.yaml")
        summary = TrainingRunner(config).summarize()
        self.assertEqual(summary.stage["name"], "stage_d_distill")
        self.assertGreater(summary.losses["distillation"], 0.0)
        self.assertGreater(summary.losses["teacher_target"], 0.0)

    def test_stage_e_dry_run_summary(self):
        config = load_config("configs/train/stage_e_streaming.yaml")
        summary = TrainingRunner(config).summarize()
        self.assertEqual(summary.stage["name"], "stage_e_streaming")
        self.assertEqual(summary.stage["mode"], "mixed")
        self.assertGreater(summary.losses["streaming_causality"], 0.0)


if __name__ == "__main__":
    unittest.main()
