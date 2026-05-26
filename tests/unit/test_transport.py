import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from trajflow_vsr.models.transport.ot_sb_bridge import OTSBTrajectoryBridge  # noqa: E402
from trajflow_vsr.ops.sinkhorn import sinkhorn_plan  # noqa: E402
from trajflow_vsr.utils.torch_utils import require_torch  # noqa: E402


class SinkhornTransportTest(unittest.TestCase):
    def test_sinkhorn_plan_matches_requested_marginals(self):
        torch = require_torch()
        cost = torch.tensor(
            [[[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]]],
            dtype=torch.float32,
        )
        source = torch.tensor([[0.2, 0.3, 0.5]], dtype=torch.float32)
        target = torch.tensor([[0.4, 0.4, 0.2]], dtype=torch.float32)
        plan = sinkhorn_plan(cost, source, target, epsilon=0.25, iterations=50)

        self.assertEqual(tuple(plan.shape), (1, 3, 3))
        self.assertTrue(torch.allclose(plan.sum(dim=-1), source, atol=1e-4))
        self.assertTrue(torch.allclose(plan.sum(dim=-2), target, atol=1e-4))

    def test_bridge_returns_ot_and_conditional_transport_plans(self):
        torch = require_torch()
        bridge = OTSBTrajectoryBridge(hidden_channels=4, temperature=0.3, sinkhorn_iterations=20, bridge_steps=3)
        feature_grid = torch.rand(1, 2, 3, 3, 4)
        token_bundle = {
            "tokens": feature_grid.flatten(1, 3),
            "feature_grid": feature_grid,
        }
        uncertainty = {"reliability": torch.ones(1, 2, 1, 3, 3)}
        output = bridge(token_bundle, uncertainty)

        self.assertEqual(tuple(output["bridge_grid"].shape), tuple(feature_grid.shape))
        self.assertEqual(tuple(output["bridge_states"].shape), (1, 3, 2, 3, 3, 4))
        self.assertEqual(tuple(output["bridge_times"].shape), (3,))
        self.assertEqual(tuple(output["ot_plan"].shape), (1, 18, 18))
        self.assertEqual(tuple(output["candidate_mask"].shape), (18, 18))
        self.assertEqual(tuple(output["unmatched_mass"].shape), (1, 18))
        self.assertEqual(tuple(output["bridge_weight"].shape), (1, 2, 3, 3, 1))
        self.assertGreaterEqual(float(output["occlusion_mass"].detach()), 0.0)
        self.assertTrue(torch.allclose(output["bridge_grid"], feature_grid, atol=1e-6))
        self.assertTrue(torch.allclose(output["bridge_weight"], torch.zeros_like(output["bridge_weight"]), atol=1e-6))
        self.assertTrue(torch.allclose(output["transport_plan"].sum(dim=-1), torch.ones(1, 18), atol=1e-5))

    def test_bridge_uses_transported_grid_in_low_reliability_regions(self):
        torch = require_torch()
        bridge = OTSBTrajectoryBridge(hidden_channels=4, temperature=0.3, sinkhorn_iterations=20)
        feature_grid = torch.rand(1, 2, 2, 2, 4)
        token_bundle = {
            "tokens": feature_grid.flatten(1, 3),
            "feature_grid": feature_grid,
        }
        uncertainty = {"reliability": torch.zeros(1, 2, 1, 2, 2)}

        output = bridge(token_bundle, uncertainty)

        self.assertTrue(torch.allclose(output["bridge_grid"], output["transported_grid"], atol=1e-6))
        self.assertTrue(torch.allclose(output["bridge_weight"], torch.ones_like(output["bridge_weight"]), atol=1e-6))

    def test_causal_bridge_masks_future_tokens(self):
        torch = require_torch()
        bridge = OTSBTrajectoryBridge(hidden_channels=4, temperature=0.3, sinkhorn_iterations=20)
        feature_grid = torch.rand(1, 3, 1, 1, 4)
        token_bundle = {
            "tokens": feature_grid.flatten(1, 3),
            "feature_grid": feature_grid,
        }
        uncertainty = {"reliability": torch.ones(1, 3, 1, 1, 1)}
        output = bridge(token_bundle, uncertainty, causal=True)
        future_mask = ~output["causal_mask"]

        self.assertTrue(output["causal"])
        future_mass = output["transport_plan"].masked_select(future_mask.unsqueeze(0)).sum().detach()
        self.assertLess(float(future_mass), 1e-6)
        self.assertGreaterEqual(float(output["causal_violation"].detach()), 0.0)

    def test_bridge_can_disable_unbalanced_reliability_for_ablation(self):
        torch = require_torch()
        bridge = OTSBTrajectoryBridge(
            hidden_channels=4,
            temperature=0.3,
            sinkhorn_iterations=10,
            use_unbalanced=False,
            use_reliability=False,
        )
        feature_grid = torch.rand(1, 2, 2, 2, 4)
        output = bridge(
            {"tokens": feature_grid.flatten(1, 3), "feature_grid": feature_grid},
            {"reliability": torch.rand(1, 2, 1, 2, 2)},
        )

        self.assertTrue(torch.allclose(output["unmatched_mass"], torch.zeros_like(output["unmatched_mass"])))
        self.assertTrue(torch.allclose(output["source_mass"], output["target_mass"]))


if __name__ == "__main__":
    unittest.main()
