import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from trajflow_vsr.losses.reconstruction import koopman_dynamics_loss  # noqa: E402
from trajflow_vsr.models.memory.trajectory_koopman_ssm import TrajectoryKoopmanSSMMemory  # noqa: E402
from trajflow_vsr.utils.torch_utils import require_torch  # noqa: E402


class TrajectoryMemoryScanTest(unittest.TestCase):
    def _transport(self):
        torch = require_torch()
        source = torch.linspace(0.0, 1.0, steps=1 * 3 * 2 * 3 * 4).reshape(1, 3, 2, 3, 4)
        transported = source.flip(dims=[1])
        token_count = source.flatten(1, 3).shape[1]
        return {
            "source_grid": source,
            "transported_grid": transported,
            "bridge_grid": 0.5 * (source + transported),
            "bridge_weight": torch.ones(1, 3, 2, 3, 1),
            "transport_plan": torch.eye(token_count).unsqueeze(0),
            "bridge_drift": transported - source,
        }

    def test_scan_policies_preserve_memory_grid_shape(self):
        for policy in [
            "ot_sb",
            "ot_sb_topk",
            "ot_sb_hard",
            "bridge_temporal",
            "temporal",
            "raster",
            "hilbert",
            "content",
            "soft_trajectory",
        ]:
            with self.subTest(policy=policy):
                memory = TrajectoryKoopmanSSMMemory(hidden_channels=4, scan_policy=policy)
                output = memory(self._transport())

                self.assertEqual(tuple(output["memory_grid"].shape), (1, 3, 2, 3, 4))
                self.assertEqual(tuple(output["memory_tokens"].shape), (1, 18, 4))
                self.assertGreater(output["scan_sequence_length"], 0)
                self.assertGreaterEqual(float(koopman_dynamics_loss(output).detach()), 0.0)

    def test_koopman_can_be_disabled_for_ablation(self):
        memory = TrajectoryKoopmanSSMMemory(
            hidden_channels=4,
            scan_policy="raster",
            use_koopman=False,
        )
        output = memory(self._transport())

        self.assertEqual(output["koopman_prediction"].numel(), 0)
        self.assertEqual(output["koopman_target"].numel(), 0)
        self.assertEqual(float(koopman_dynamics_loss(output).detach()), 0.0)

    def test_ot_sb_builds_plan_conditioned_soft_trajectory_sequence(self):
        torch = require_torch()
        source = torch.tensor([[[[[0.0], [10.0]]], [[[100.0], [110.0]]]]])
        plan = torch.zeros(1, 4, 4)
        plan[:, :, 0] = 1.0
        plan[0, 2] = torch.tensor([0.0, 0.5, 0.5, 0.0])
        transport = {
            "source_grid": source,
            "bridge_grid": source,
            "bridge_weight": torch.ones(1, 2, 1, 2, 1),
            "transport_plan": plan,
        }
        memory = TrajectoryKoopmanSSMMemory(hidden_channels=1, scan_policy="ot_sb")

        seq, _, diagnostics = memory._make_soft_trajectory_sequence(transport)

        self.assertTrue(diagnostics["soft_trajectory_used"])
        self.assertTrue(torch.allclose(seq[2, :, 0], torch.tensor([10.0, 100.0])))

    def test_topk_and_hard_trajectory_reduce_soft_averaging(self):
        torch = require_torch()
        source = torch.tensor([[[[[0.0], [10.0], [20.0]]], [[[100.0], [110.0], [120.0]]]]])
        plan = torch.zeros(1, 6, 6)
        plan[:, :, 0] = 1.0
        plan[0, 3] = torch.tensor([0.0, 0.2, 0.8, 0.1, 0.3, 0.6])
        transport = {
            "source_grid": source,
            "bridge_grid": source,
            "bridge_weight": torch.ones(1, 2, 1, 3, 1),
            "transport_plan": plan,
        }

        soft = TrajectoryKoopmanSSMMemory(hidden_channels=1, scan_policy="ot_sb")
        topk = TrajectoryKoopmanSSMMemory(hidden_channels=1, scan_policy="ot_sb_topk", trajectory_topk=1)
        hard = TrajectoryKoopmanSSMMemory(hidden_channels=1, scan_policy="ot_sb_hard")
        soft_seq, _, soft_diag = soft._make_soft_trajectory_sequence(transport)
        topk_seq, _, topk_diag = topk._make_soft_trajectory_sequence(transport)
        hard_seq, _, hard_diag = hard._make_soft_trajectory_sequence(transport)

        self.assertEqual(soft_diag["trajectory_scan_mode"], "soft_expectation")
        self.assertEqual(topk_diag["trajectory_scan_mode"], "top1_expectation")
        self.assertEqual(hard_diag["trajectory_scan_mode"], "hard_top1_straight_through")
        self.assertLess(float(soft_seq[3, 0, 0]), float(topk_seq[3, 0, 0]))
        self.assertTrue(torch.allclose(topk_seq[3, :, 0], hard_seq[3, :, 0]))

    def test_high_reliability_anchor_keeps_fixed_pixel_temporal_sequence(self):
        torch = require_torch()
        source = torch.tensor([[[[[0.0], [10.0]]], [[[100.0], [110.0]]]]])
        plan = torch.zeros(1, 4, 4)
        plan[:, :, 0] = 1.0
        plan[0, 2] = torch.tensor([0.0, 0.5, 0.5, 0.0])
        transport = {
            "source_grid": source,
            "bridge_grid": source,
            "bridge_weight": torch.zeros(1, 2, 1, 2, 1),
            "transport_plan": plan,
        }
        memory = TrajectoryKoopmanSSMMemory(hidden_channels=1, scan_policy="ot_sb")

        seq, _, diagnostics = memory._make_soft_trajectory_sequence(transport)

        self.assertTrue(diagnostics["soft_trajectory_used"])
        self.assertTrue(torch.allclose(seq[2, :, 0], torch.tensor([0.0, 100.0])))


if __name__ == "__main__":
    unittest.main()
