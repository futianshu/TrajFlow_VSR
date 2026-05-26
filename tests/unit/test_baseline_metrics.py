import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from trajflow_vsr.baselines import collect_baseline_metrics, export_baseline_metrics  # noqa: E402


class BaselineMetricsTest(unittest.TestCase):
    def test_collect_baseline_metrics_from_nested_and_flat_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            nested = root / "nested.json"
            flat = root / "flat.json"
            nested.write_text(
                json.dumps(
                    {
                        "metrics": {
                            "offline": {
                                "psnr": 30.5,
                                "ssim": 0.91,
                                "temporal_delta_error": 0.08,
                                "fps": 12.0,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            flat.write_text(
                json.dumps({"psnr": 29.0, "ssim": 0.88, "temporal_delta_error": 0.06, "fps": 20.0}),
                encoding="utf-8",
            )
            config = _metrics_config(nested.name, flat.name, "missing.json")

            rows = collect_baseline_metrics(
                config,
                metrics=["psnr", "ssim", "temporal_delta_error", "fps"],
                root=root,
            )

            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0]["metrics_status"], "ok")
            self.assertEqual(rows[0]["psnr"], 30.5)
            self.assertEqual(rows[1]["fps"], 20.0)
            self.assertEqual(rows[2]["metrics_status"], "missing_metrics")
            self.assertEqual(rows[2]["psnr"], "")

    def test_export_baseline_metrics_files_and_best_by_metric(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            first = root / "first.json"
            second = root / "second.json"
            first.write_text(json.dumps({"metrics": {"offline": {"psnr": 30.5, "fps": 12.0}}}), encoding="utf-8")
            second.write_text(json.dumps({"psnr": 29.0, "fps": 20.0}), encoding="utf-8")
            config = _metrics_config(first.name, second.name, "missing.json")

            export = export_baseline_metrics(
                config,
                root / "exports",
                name="unit_baseline_metrics",
                metrics=["psnr", "fps"],
                root=root,
            )

            self.assertEqual(export.baseline_count, 3)
            self.assertEqual(export.available_count, 2)
            for path in export.files.values():
                self.assertTrue(Path(path).exists())

            payload = json.loads(Path(export.files["json"]).read_text(encoding="utf-8"))
            self.assertEqual(payload["best_by_metric"]["psnr"]["id"], "first")
            self.assertEqual(payload["best_by_metric"]["fps"]["id"], "second")
            table = Path(export.files["markdown"]).read_text(encoding="utf-8")
            self.assertIn("missing_metrics", table)


def _metrics_config(first_path: str, second_path: str, missing_path: str) -> dict:
    return {
        "baselines": {
            "first": {
                "display_name": "First",
                "category": "unit",
                "status": "ready",
                "repo_url": "https://example.test/first",
                "commit": "abc123",
                "weights": {"source": "local", "local_path": "checkpoints/first.pt"},
                "commands": {"setup": "true", "infer": "true", "evaluate": "true"},
                "metrics_path": first_path,
            },
            "second": {
                "display_name": "Second",
                "category": "unit",
                "status": "ready",
                "repo_url": "https://example.test/second",
                "commit": "def456",
                "weights": {"source": "local", "local_path": "checkpoints/second.pt"},
                "commands": {"setup": "true", "infer": "true", "evaluate": "true"},
                "metrics_path": second_path,
            },
            "missing": {
                "display_name": "Missing",
                "category": "unit",
                "status": "planned",
                "repo_url": "https://example.test/missing",
                "commit": "ghi789",
                "weights": {"source": "local", "local_path": "checkpoints/missing.pt"},
                "commands": {"setup": "true", "infer": "true", "evaluate": "true"},
                "metrics_path": missing_path,
            },
        }
    }


if __name__ == "__main__":
    unittest.main()
