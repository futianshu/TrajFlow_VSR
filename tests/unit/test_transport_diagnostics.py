import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from trajflow_vsr.diagnostics import transport_plan_diagnostics  # noqa: E402
from trajflow_vsr.utils.torch_utils import require_torch  # noqa: E402


class TransportDiagnosticsTest(unittest.TestCase):
    def test_identity_plan_reports_same_frame_mass(self):
        torch = require_torch()
        frames, height, width = 2, 2, 3
        token_count = frames * height * width
        plan = torch.eye(token_count).unsqueeze(0)

        diagnostics = transport_plan_diagnostics(
            {"transport_plan": plan},
            frames=frames,
            height=height,
            width=width,
            shift_x=1,
            shift_y=0,
        )

        self.assertAlmostEqual(diagnostics["plan_diagonal_mass"], 1.0)
        self.assertAlmostEqual(diagnostics["plan_same_frame_mass"], 1.0)
        self.assertAlmostEqual(diagnostics["plan_cross_frame_mass"], 0.0)
        self.assertAlmostEqual(diagnostics["plan_oracle_mass"], 1.0)

    def test_oracle_cross_frame_plan_is_detected(self):
        torch = require_torch()
        frames, height, width = 2, 2, 4
        token_count = frames * height * width
        plan = torch.zeros(1, token_count, token_count)
        for source in range(token_count):
            frame = source // (height * width)
            rem = source % (height * width)
            y = rem // width
            x = rem % width
            target_frame = min(frame + 1, frames - 1)
            target_y = y
            target_x = (x + (target_frame - frame)) % width
            target = target_frame * height * width + target_y * width + target_x
            plan[0, source, target] = 1.0

        diagnostics = transport_plan_diagnostics(
            {"transport_plan": plan},
            frames=frames,
            height=height,
            width=width,
            shift_x=1,
            shift_y=0,
        )

        self.assertGreater(diagnostics["plan_cross_frame_mass"], 0.0)
        self.assertGreater(diagnostics["plan_oracle_cross_frame_mass"], 0.0)
        self.assertAlmostEqual(diagnostics["plan_top1_oracle_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
