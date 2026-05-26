import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from trajflow_vsr.losses.factory import compute_training_loss  # noqa: E402
from trajflow_vsr.utils.torch_utils import require_torch  # noqa: E402


class TrainingLossTest(unittest.TestCase):
    def test_data_consistency_loss_is_weighted_into_total(self):
        torch = require_torch()
        lr = torch.rand(1, 1, 3, 4, 4)
        hr = torch.nn.functional.interpolate(lr.flatten(0, 1), scale_factor=2, mode="nearest")
        hr = hr.unflatten(0, (1, 1))
        outputs = {
            "hr": hr,
            "uncertainty": {"reliability": torch.ones(1, 1, 1, 4, 4)},
            "memory": {},
        }
        batch = {"lr": lr, "hr": hr}

        total, parts = compute_training_loss(
            outputs,
            batch,
            {"charbonnier": 0.0, "koopman": 0.0, "data_consistency": 1.0},
        )

        self.assertIn("data_consistency", parts)
        self.assertAlmostEqual(float(total), float(parts["data_consistency"]), places=6)
        self.assertGreater(float(total), 0.0)

    def test_optimal_transport_loss_is_weighted_into_total(self):
        torch = require_torch()
        lr = torch.rand(1, 1, 3, 2, 2)
        plan = torch.eye(4).unsqueeze(0) * 0.25
        outputs = {
            "hr": lr,
            "memory": {},
            "transport": {
                "ot_plan": plan,
                "cost": torch.ones(1, 4, 4),
                "row_marginal_error": torch.zeros(()),
                "column_marginal_error": torch.zeros(()),
            },
        }
        batch = {"lr": lr, "hr": lr}

        total, parts = compute_training_loss(
            outputs,
            batch,
            {"charbonnier": 0.0, "koopman": 0.0, "optimal_transport": 0.5},
        )

        self.assertIn("optimal_transport", parts)
        self.assertAlmostEqual(float(total), 0.5 * float(parts["optimal_transport"]), places=6)

    def test_motion_transport_and_entropy_losses_are_weighted_into_total(self):
        torch = require_torch()
        lr = torch.rand(1, 2, 3, 2, 3)
        token_count = 12
        oracle_plan = torch.zeros(1, token_count, token_count)
        for source in range(token_count):
            frame = source // 6
            rem = source % 6
            y = rem // 3
            x = rem % 3
            for target_frame in range(2):
                target_x = (x + (target_frame - frame)) % 3
                target = target_frame * 6 + y * 3 + target_x
                oracle_plan[0, source, target] = 0.5
        uniform_plan = torch.full_like(oracle_plan, 1.0 / token_count)
        source_grid = torch.rand(1, 2, 2, 3, 4)
        batch = {
            "lr": lr,
            "hr": lr,
            "controlled_motion": {"shift_x": 1, "shift_y": 0},
        }

        oracle_outputs = {
            "hr": lr,
            "memory": {},
            "transport": {
                "transport_plan": oracle_plan,
                "source_grid": source_grid,
            },
        }
        uniform_outputs = {
            "hr": lr,
            "memory": {},
            "transport": {
                "transport_plan": uniform_plan,
                "source_grid": source_grid,
            },
        }

        oracle_total, oracle_parts = compute_training_loss(
            oracle_outputs,
            batch,
            {"charbonnier": 0.0, "koopman": 0.0, "motion_transport": 1.0, "transport_entropy": 0.1},
        )
        uniform_total, uniform_parts = compute_training_loss(
            uniform_outputs,
            batch,
            {"charbonnier": 0.0, "koopman": 0.0, "motion_transport": 1.0, "transport_entropy": 0.1},
        )

        self.assertIn("motion_transport", oracle_parts)
        self.assertIn("transport_entropy", oracle_parts)
        self.assertLess(float(oracle_parts["motion_transport"]), float(uniform_parts["motion_transport"]))
        self.assertLess(float(oracle_total), float(uniform_total))

    def test_schrodinger_bridge_loss_is_weighted_into_total(self):
        torch = require_torch()
        lr = torch.rand(1, 1, 3, 2, 2)
        source = torch.zeros(1, 1, 2, 2, 4)
        target = torch.ones(1, 1, 2, 2, 4)
        times = torch.tensor([0.25, 0.5, 0.75])
        states = torch.stack([(1.0 - t) * source + t * target for t in times], dim=1)
        outputs = {
            "hr": lr,
            "memory": {},
            "transport": {
                "source_grid": source,
                "target_grid": target,
                "bridge_states": states,
                "bridge_times": times,
            },
        }
        batch = {"lr": lr, "hr": lr}

        total, parts = compute_training_loss(
            outputs,
            batch,
            {"charbonnier": 0.0, "koopman": 0.0, "schrodinger_bridge": 0.25},
        )

        self.assertIn("schrodinger_bridge", parts)
        self.assertAlmostEqual(float(total), 0.25 * float(parts["schrodinger_bridge"]), places=6)
        self.assertGreater(float(parts["schrodinger_bridge"]), 0.0)

    def test_flow_matching_losses_are_weighted_into_total(self):
        torch = require_torch()
        lr = torch.rand(1, 2, 3, 2, 2)
        velocity = torch.zeros(1, 2, 2, 2, 4)
        target_velocity = torch.ones(1, 2, 2, 2, 4)
        residual_grid = torch.zeros(1, 2, 2, 2, 4)
        target_residual = torch.ones(1, 2, 2, 2, 4)
        residual_low_band = torch.ones(1, 2, 2, 2, 4) * 0.25
        outputs = {
            "hr": lr,
            "memory": {},
            "uncertainty": {"reliability": torch.ones(1, 2, 1, 2, 2)},
            "residual": {
                "flow_velocity": velocity,
                "flow_target_velocity": target_velocity,
                "residual_grid": residual_grid,
                "residual_low_band": residual_low_band,
                "residual_gate": torch.ones(1, 2, 2, 2, 1) * 0.5,
                "flow_target_residual": target_residual,
                "tau": torch.tensor([0.5]),
            },
        }
        batch = {"lr": lr, "hr": lr}

        total, parts = compute_training_loss(
            outputs,
            batch,
            {
                "charbonnier": 0.0,
                "koopman": 0.0,
                "flow_matching": 0.25,
                "bridge_consistency": 0.5,
                "residual_amplitude": 0.1,
                "residual_low_frequency": 0.2,
            },
        )

        expected = (
            0.25 * parts["flow_matching"]
            + 0.5 * parts["bridge_consistency"]
            + 0.1 * parts["residual_amplitude"]
            + 0.2 * parts["residual_low_frequency"]
        )
        self.assertIn("flow_matching", parts)
        self.assertIn("bridge_consistency", parts)
        self.assertIn("residual_amplitude", parts)
        self.assertIn("residual_low_frequency", parts)
        self.assertAlmostEqual(float(total), float(expected), places=6)

    def test_distillation_losses_are_weighted_into_total(self):
        torch = require_torch()
        lr = torch.rand(1, 2, 3, 2, 2)
        student = torch.zeros(1, 2, 2, 2, 4)
        teacher = torch.ones(1, 2, 2, 2, 4)
        outputs = {
            "hr": lr,
            "memory": {},
            "residual": {
                "student_residual": student,
                "teacher_residual": teacher,
                "flow_target_residual": teacher * 0.5,
            },
        }
        batch = {"lr": lr, "hr": lr}

        total, parts = compute_training_loss(
            outputs,
            batch,
            {
                "charbonnier": 0.0,
                "koopman": 0.0,
                "distillation": 0.25,
                "teacher_target": 0.5,
            },
        )

        expected = 0.25 * parts["distillation"] + 0.5 * parts["teacher_target"]
        self.assertIn("distillation", parts)
        self.assertIn("teacher_target", parts)
        self.assertAlmostEqual(float(total), float(expected), places=6)

    def test_streaming_causality_loss_is_weighted_into_total(self):
        torch = require_torch()
        lr = torch.rand(1, 1, 3, 2, 2)
        outputs = {
            "hr": lr,
            "memory": {},
            "transport": {"causal_violation": torch.tensor(0.25)},
        }
        batch = {"lr": lr, "hr": lr}

        total, parts = compute_training_loss(
            outputs,
            batch,
            {
                "charbonnier": 0.0,
                "koopman": 0.0,
                "streaming_causality": 0.5,
            },
        )

        self.assertIn("streaming_causality", parts)
        self.assertAlmostEqual(float(total), 0.5 * float(parts["streaming_causality"]), places=6)

    def test_wavelet_and_aliasing_losses_are_weighted_into_total(self):
        torch = require_torch()
        lr = torch.rand(1, 2, 3, 4, 4)
        hr = torch.nn.functional.interpolate(lr.flatten(0, 1), scale_factor=2, mode="nearest")
        hr = hr.unflatten(0, (1, 2))
        outputs = {
            "hr": hr * 0.9,
            "memory": {},
            "decoded": {
                "wavelet_high": torch.ones_like(hr) * 0.25,
                "anti_alias_gate": torch.ones_like(hr) * 0.5,
            },
        }
        batch = {"lr": lr, "hr": hr}

        total, parts = compute_training_loss(
            outputs,
            batch,
            {
                "charbonnier": 0.0,
                "koopman": 0.0,
                "wavelet_frequency": 0.5,
                "anti_aliasing": 0.25,
            },
        )

        expected = 0.5 * parts["wavelet_frequency"] + 0.25 * parts["anti_aliasing"]
        self.assertIn("wavelet_frequency", parts)
        self.assertIn("anti_aliasing", parts)
        self.assertAlmostEqual(float(total), float(expected), places=6)

    def test_temporal_and_trajectory_losses_are_weighted_into_total(self):
        torch = require_torch()
        lr = torch.rand(1, 3, 3, 2, 2)
        hr = torch.nn.functional.interpolate(lr.flatten(0, 1), scale_factor=2, mode="nearest")
        hr = hr.unflatten(0, (1, 3))
        plan = torch.full((1, 12, 12), 1.0 / 12.0)
        outputs = {
            "hr": hr,
            "memory": {},
            "uncertainty": {"reliability": torch.ones(1, 3, 1, 2, 2)},
            "transport": {
                "transport_plan": plan,
                "bridge_grid": torch.rand(1, 3, 2, 2, 4),
            },
        }
        batch = {"lr": lr, "hr": hr}

        total, parts = compute_training_loss(
            outputs,
            batch,
            {
                "charbonnier": 0.0,
                "koopman": 0.0,
                "temporal": 0.25,
                "trajectory": 0.5,
            },
        )

        expected = 0.25 * parts["temporal"] + 0.5 * parts["trajectory"]
        self.assertIn("temporal", parts)
        self.assertIn("trajectory", parts)
        self.assertAlmostEqual(float(total), float(expected), places=6)

    def test_uncertainty_calibration_loss_is_weighted_into_total(self):
        torch = require_torch()
        lr = torch.rand(1, 2, 3, 3, 3)
        hr = torch.nn.functional.interpolate(lr.flatten(0, 1), scale_factor=2, mode="nearest").unflatten(0, (1, 2))
        outputs = {
            "hr": hr * 0.9,
            "memory": {},
            "uncertainty": {"reliability": torch.ones(1, 2, 1, 3, 3) * 0.8},
        }
        batch = {"lr": lr, "hr": hr}

        total, parts = compute_training_loss(
            outputs,
            batch,
            {"charbonnier": 0.0, "koopman": 0.0, "uncertainty": 0.5},
        )

        self.assertIn("uncertainty", parts)
        self.assertAlmostEqual(float(total), 0.5 * float(parts["uncertainty"]), places=6)


if __name__ == "__main__":
    unittest.main()
