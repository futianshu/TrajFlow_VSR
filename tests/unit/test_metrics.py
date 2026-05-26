import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from trajflow_vsr.metrics import (  # noqa: E402
    blockiness_proxy,
    dists_proxy,
    lpips_proxy,
    profile_model_macs,
    psnr,
    reliability_ece,
    selective_reconstruction_auc,
    spatial_sharpness,
    ssim,
    temporal_activity,
    temporal_delta_error,
    tof_proxy,
    uncertainty_error_correlation,
)
from trajflow_vsr.utils.torch_utils import require_torch  # noqa: E402


class MetricsTest(unittest.TestCase):
    def test_quality_metrics_are_finite_for_video_tensors(self):
        torch = require_torch()
        target = torch.rand(1, 3, 3, 8, 8)
        prediction = (target * 0.9).clamp(0.0, 1.0)
        outputs = {
            "hr": prediction,
            "uncertainty": {"reliability": torch.ones(1, 3, 1, 4, 4) * 0.5},
        }

        for value in [
            psnr(prediction, target),
            ssim(prediction, target),
            temporal_delta_error(prediction, target),
            tof_proxy(prediction, target),
            lpips_proxy(prediction, target),
            dists_proxy(prediction, target),
            temporal_activity(prediction),
            spatial_sharpness(prediction),
            blockiness_proxy(prediction),
            uncertainty_error_correlation(outputs, target),
            reliability_ece(outputs, target),
            selective_reconstruction_auc(outputs, target),
        ]:
            self.assertTrue(torch.isfinite(value))

    def test_identical_video_has_zero_temporal_delta_error(self):
        torch = require_torch()
        video = torch.rand(1, 3, 3, 6, 6)
        self.assertAlmostEqual(float(temporal_delta_error(video, video)), 0.0, places=6)

    def test_profile_model_macs_counts_conv_and_linear_layers(self):
        torch = require_torch()
        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 4, kernel_size=3, padding=1),
            torch.nn.Flatten(),
            torch.nn.Linear(4 * 8 * 8, 2),
        )
        profile = profile_model_macs(model, torch.rand(1, 3, 8, 8))

        self.assertGreater(profile["macs"], 0)
        self.assertGreater(profile["profiled_modules"], 0)
        self.assertIn("macs_by_module", profile)

    def test_profile_model_macs_accumulates_reused_modules(self):
        torch = require_torch()

        class ReusedLinear(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Linear(3, 4)

            def forward(self, x):
                return self.proj(x) + self.proj(x)

        profile = profile_model_macs(ReusedLinear(), torch.rand(2, 3))

        self.assertEqual(profile["macs_by_module"]["proj"], 48)


if __name__ == "__main__":
    unittest.main()
