import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from trajflow_vsr.experiments import BenchmarkPlan  # noqa: E402
from trajflow_vsr.utils.config import load_config  # noqa: E402


class BenchmarkPlanTest(unittest.TestCase):
    def test_core_benchmark_plan_expands_expected_matrix(self):
        plan = BenchmarkPlan(load_config("configs/benchmark/ccfa_core_benchmark.yaml"))
        rows = plan.rows()

        self.assertEqual(len(rows), 720)
        internal = rows[0]
        self.assertEqual(internal["method"], "trajflow_stage_b")
        self.assertEqual(internal["status"], "planned_internal")
        self.assertIn('CUDA_VISIBLE_DEVICES=""', internal["command"])
        self.assertIn("--set runtime.device=cpu", internal["command"])
        self.assertIn("data.scale=2", internal["command"])
        spmcs = next(row for row in rows if row["dataset"] == "spmcs" and row["method"] == "trajflow_stage_b")
        self.assertIn("data/splits/spmcs_bix4_test_manifest.json", spmcs["command"])
        self.assertIn("data.frames=31", spmcs["command"])
        realvsr = next(row for row in rows if row["dataset"] == "realvsr" and row["method"] == "trajflow_stage_b")
        self.assertIn("data/splits/realvsr_test_manifest.json", realvsr["command"])
        self.assertIn("data.frames=50", realvsr["command"])

        external = next(row for row in rows if row["method"] == "basicvsrpp")
        self.assertEqual(external["status"], "planned_external")
        self.assertIn("registry_id=basicvsrpp", external["command"])
        self.assertIn("experiments/benchmark_metrics/ccfa_core/basicvsrpp", external["metrics_path"])

    def test_export_benchmark_plan_files(self):
        config = {
            "benchmark": {
                "name": "unit_benchmark",
                "metrics_root": "metrics",
                "methods": {
                    "trajflow": {
                        "kind": "internal",
                        "eval_config": "configs/eval/offline.yaml",
                        "checkpoint_path": "checkpoints/final.pt",
                    }
                },
                "datasets": {"toy": {"overrides": ["data.name=synthetic"]}},
                "degradations": {"synthetic": {"overrides": ["data.degradation.profile=synthetic"]}},
                "scales": {"x2": {"overrides": ["data.scale=2"]}},
                "protocols": {"offline": {"eval_config": "configs/eval/offline.yaml", "overrides": ["evaluation.mode=offline"]}},
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            export = BenchmarkPlan(config).export(tmpdir, name="unit_benchmark")

            self.assertEqual(export.run_count, 1)
            for path in export.files.values():
                self.assertTrue(Path(path).exists())
            payload = json.loads(Path(export.files["json"]).read_text(encoding="utf-8"))
            self.assertEqual(payload["run_count"], 1)
            table = Path(export.files["markdown"]).read_text(encoding="utf-8")
            self.assertIn("trajflow_toy_synthetic_x2_offline", table)


if __name__ == "__main__":
    unittest.main()
