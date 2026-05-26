import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from trajflow_vsr.models.tokenizer.evidence_tokenizer import MultiScaleEvidenceTokenizer  # noqa: E402
from trajflow_vsr.utils.torch_utils import require_torch  # noqa: E402


class EvidenceTokenizerTest(unittest.TestCase):
    def test_tokenizer_returns_multiscale_coordinates_and_footprints(self):
        torch = require_torch()
        tokenizer = MultiScaleEvidenceTokenizer(in_channels=3, hidden_channels=8)
        output = tokenizer(torch.rand(1, 3, 3, 5, 6), scale=4.0)

        self.assertEqual(tuple(output["feature_grid"].shape), (1, 3, 5, 6, 8))
        self.assertEqual(tuple(output["tokens"].shape), (1, 90, 8))
        self.assertEqual(tuple(output["low_band"].shape), (1, 3, 3, 5, 6))
        self.assertEqual(tuple(output["high_band"].shape), (1, 3, 3, 5, 6))
        self.assertEqual(tuple(output["coordinates"].shape), (1, 3, 5, 6, 3))
        self.assertEqual(tuple(output["footprints"].shape), (1, 3, 5, 6, 2))
        self.assertAlmostEqual(float(output["footprints"][0, 0, 0, 0, 0]), 0.25, places=6)


if __name__ == "__main__":
    unittest.main()
