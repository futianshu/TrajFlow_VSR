import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from trajflow_vsr.models.decoder.wavelet_operator_decoder import SpacetimeWaveletOperatorDecoder  # noqa: E402
from trajflow_vsr.ops.wavelet import split_low_high_2d, split_video_low_high  # noqa: E402
from trajflow_vsr.utils.torch_utils import require_torch  # noqa: E402


class WaveletDecoderTest(unittest.TestCase):
    def test_low_high_split_reconstructs_image_and_video(self):
        torch = require_torch()
        image = torch.rand(2, 3, 7, 9)
        low, high = split_low_high_2d(image)
        self.assertEqual(tuple(low.shape), tuple(image.shape))
        self.assertTrue(torch.allclose(low + high, image, atol=1e-6))

        video = torch.rand(1, 2, 3, 7, 9)
        video_low, video_high = split_video_low_high(video)
        self.assertEqual(tuple(video_low.shape), tuple(video.shape))
        self.assertTrue(torch.allclose(video_low + video_high, video, atol=1e-6))

    def test_decoder_returns_wavelet_bands_for_fractional_scale(self):
        torch = require_torch()
        decoder = SpacetimeWaveletOperatorDecoder(hidden_channels=4, out_channels=3)
        memory = {"memory_grid": torch.rand(1, 2, 5, 6, 4)}
        residual = {"residual_grid": torch.rand(1, 2, 5, 6, 4)}
        decoded = decoder(torch.rand(1, 2, 3, 5, 6), memory, residual, scale=1.5)

        self.assertEqual(tuple(decoded["hr_raw"].shape), (1, 2, 3, 8, 9))
        self.assertEqual(tuple(decoded["wavelet_low"].shape), (1, 2, 3, 8, 9))
        self.assertEqual(tuple(decoded["wavelet_high"].shape), (1, 2, 3, 8, 9))
        self.assertEqual(tuple(decoded["anti_alias_gate"].shape), (1, 2, 3, 8, 9))
        self.assertEqual(tuple(decoded["query_coordinates"].shape), (1, 2, 8, 9, 3))
        self.assertEqual(tuple(decoded["query_footprint"].shape), (1, 2, 8, 9, 1))

    def test_decoder_ablation_flags_disable_wavelet_and_operator_paths(self):
        torch = require_torch()
        decoder = SpacetimeWaveletOperatorDecoder(
            hidden_channels=4,
            out_channels=3,
            use_operator=False,
            use_fourier=False,
            use_wavelet=False,
            use_anti_aliasing=False,
        )
        memory = {"memory_grid": torch.rand(1, 1, 3, 3, 4)}
        residual = {"residual_grid": torch.rand(1, 1, 3, 3, 4)}
        decoded = decoder(torch.rand(1, 1, 3, 3, 3), memory, residual, scale=2.0)

        self.assertEqual(tuple(decoded["hr_raw"].shape), (1, 1, 3, 6, 6))
        self.assertTrue(torch.allclose(decoded["wavelet_high"], torch.zeros_like(decoded["wavelet_high"])))
        self.assertTrue(torch.allclose(decoded["anti_alias_gate"], torch.ones_like(decoded["anti_alias_gate"])))


if __name__ == "__main__":
    unittest.main()
