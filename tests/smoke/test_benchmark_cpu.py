import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from trajflow_vsr.experiments import BenchmarkRunner  # noqa: E402
from trajflow_vsr.utils.torch_utils import require_torch  # noqa: E402


class BenchmarkCpuTest(unittest.TestCase):
    def test_internal_benchmark_subset_runs_on_cpu(self):
        require_torch()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = {
                "benchmark": {
                    "name": "unit_cpu_benchmark",
                    "metrics_root": str(root / "planned_metrics"),
                    "methods": {
                        "trajflow": {
                            "kind": "internal",
                            "eval_config": "configs/eval/offline.yaml",
                            "overrides": [
                                "model.hidden_channels=8",
                                "model.transport.sinkhorn_iterations=3",
                                "model.transport.bridge_steps=2",
                            ],
                        }
                    },
                    "datasets": {
                        "synthetic": {
                            "overrides": [
                                "data.name=synthetic",
                                "data.batch_size=1",
                                "data.frames=2",
                                "data.height=4",
                                "data.width=4",
                            ]
                        }
                    },
                    "degradations": {"clean": {"overrides": ["data.degradation.profile=synthetic"]}},
                    "scales": {"x2": {"overrides": ["data.scale=2"]}},
                    "protocols": {"offline": {"eval_config": "configs/eval/offline.yaml", "overrides": ["evaluation.mode=offline"]}},
                }
            }

            result = BenchmarkRunner(config).run(output_dir=root / "benchmark")

            self.assertEqual(result["run_count"], 1)
            self.assertEqual(result["ok_count"], 1)
            record = result["records"][0]
            self.assertEqual(record["status"], "ok")
            self.assertGreater(record["fps"], 0.0)
            self.assertIn("psnr", record)
            self.assertTrue(Path(record["metrics_path"]).exists())
            self.assertTrue(Path(result["export"]["files"]["summary"]).exists())
            self.assertTrue(Path(result["export"]["files"]["csv"]).exists())
            self.assertTrue(Path(result["export"]["files"]["markdown"]).exists())


if __name__ == "__main__":
    unittest.main()
