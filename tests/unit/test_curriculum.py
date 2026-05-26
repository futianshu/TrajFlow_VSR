import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from trajflow_vsr.training.curriculum import TrainingCurriculum  # noqa: E402


class TrainingCurriculumTest(unittest.TestCase):
    def test_transport_warmup_overrides_and_schedules_losses(self):
        curriculum = TrainingCurriculum(
            {
                "enabled": True,
                "phases": {
                    "transport_warmup": {
                        "start_step": 0,
                        "end_step": 3,
                        "freeze_components": ["decoder", "consistency"],
                        "loss_multipliers": {
                            "charbonnier": 0.1,
                            "data_consistency": 0.25,
                        },
                        "loss_overrides": {
                            "motion_transport": 0.5,
                        },
                        "loss_schedules": {
                            "transport_entropy": {"start": 0.05, "end": 0.01, "mode": "linear"},
                        },
                    }
                },
            },
            {
                "charbonnier": 1.0,
                "data_consistency": 0.2,
                "motion_transport": 0.0,
                "transport_entropy": 0.0,
            },
        )

        start = curriculum.state_for_step(0)
        middle = curriculum.state_for_step(1)
        end = curriculum.state_for_step(2)
        base = curriculum.state_for_step(3)

        self.assertEqual(start.phase, "transport_warmup")
        self.assertEqual(start.freeze_components, ("decoder", "consistency"))
        self.assertAlmostEqual(start.loss_config["charbonnier"], 0.1)
        self.assertAlmostEqual(start.loss_config["data_consistency"], 0.05)
        self.assertAlmostEqual(start.loss_config["motion_transport"], 0.5)
        self.assertAlmostEqual(start.loss_config["transport_entropy"], 0.05)
        self.assertAlmostEqual(middle.loss_config["transport_entropy"], 0.03)
        self.assertAlmostEqual(end.loss_config["transport_entropy"], 0.01)

        self.assertEqual(base.phase, "base")
        self.assertAlmostEqual(base.loss_config["charbonnier"], 1.0)
        self.assertAlmostEqual(base.loss_config["transport_entropy"], 0.0)

    def test_disabled_curriculum_returns_base_losses(self):
        curriculum = TrainingCurriculum({"enabled": False}, {"charbonnier": 1.0})
        state = curriculum.state_for_step(12)

        self.assertFalse(state.enabled)
        self.assertIsNone(state.phase)
        self.assertEqual(state.loss_config, {"charbonnier": 1.0})


if __name__ == "__main__":
    unittest.main()
