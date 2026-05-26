import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from trajflow_vsr.models.flow.rectified_flow import ConditionalRectifiedFlowResidualGenerator  # noqa: E402
from trajflow_vsr.utils.torch_utils import require_torch  # noqa: E402


class RectifiedFlowTest(unittest.TestCase):
    def test_flow_training_outputs_matching_targets(self):
        torch = require_torch()
        flow = ConditionalRectifiedFlowResidualGenerator(hidden_channels=4)
        memory = {
            "memory_grid": torch.rand(2, 3, 4, 4, 4),
            "bridge_drift": torch.rand(2, 3, 4, 4, 4),
        }

        output = flow(memory, uncertainty={}, sample_noise=True)

        self.assertEqual(tuple(output["residual_grid"].shape), (2, 3, 4, 4, 4))
        self.assertEqual(tuple(output["residual_gate"].shape), (2, 3, 4, 4, 1))
        self.assertEqual(tuple(output["flow_velocity"].shape), (2, 3, 4, 4, 4))
        self.assertEqual(tuple(output["flow_target_velocity"].shape), (2, 3, 4, 4, 4))
        self.assertEqual(tuple(output["tau"].shape), (2,))
        self.assertTrue(torch.all(output["tau"] >= 0.0))
        self.assertTrue(torch.all(output["tau"] <= 1.0))
        max_residual = output["residual_grid"].abs().max().detach()
        self.assertLessEqual(float(max_residual), 0.25 + 1e-6)

    def test_flow_gate_suppresses_high_reliability_regions(self):
        torch = require_torch()
        flow = ConditionalRectifiedFlowResidualGenerator(hidden_channels=4, gate_max=0.5)
        memory = {
            "memory_grid": torch.rand(1, 1, 4, 4, 4),
            "bridge_drift": torch.rand(1, 1, 4, 4, 4),
        }

        confident = flow(
            memory,
            uncertainty={
                "reliability": torch.ones(1, 1, 1, 4, 4),
                "texture_uncertainty": torch.zeros(1, 1, 1, 4, 4),
            },
            sample_noise=False,
        )
        uncertain = flow(
            memory,
            uncertainty={
                "reliability": torch.zeros(1, 1, 1, 4, 4),
                "texture_uncertainty": torch.ones(1, 1, 1, 4, 4),
            },
            sample_noise=False,
        )

        self.assertAlmostEqual(float(confident["residual_gate"].max()), 0.0, places=6)
        self.assertGreater(float(uncertain["residual_gate"].mean()), 0.49)

    def test_flow_distillation_outputs_teacher_and_student(self):
        torch = require_torch()
        flow = ConditionalRectifiedFlowResidualGenerator(hidden_channels=4)
        memory = {
            "memory_grid": torch.rand(1, 2, 3, 3, 4),
            "bridge_drift": torch.rand(1, 2, 3, 3, 4),
        }

        output = flow(memory, uncertainty={}, sample_noise=True, distill=True, teacher_steps=3)

        self.assertEqual(tuple(output["student_residual"].shape), (1, 2, 3, 3, 4))
        self.assertEqual(tuple(output["teacher_residual"].shape), (1, 2, 3, 3, 4))
        self.assertEqual(output["teacher_steps"], 3)


if __name__ == "__main__":
    unittest.main()
