import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from trajflow_vsr.baselines import baseline_records, export_baseline_registry, load_baseline_registry  # noqa: E402


class BaselineRegistryTest(unittest.TestCase):
    def test_core_baseline_registry_marks_placeholders_incomplete(self):
        config = load_baseline_registry("configs/baselines/core_vsr_baselines.yaml")
        records = baseline_records(config)

        self.assertGreaterEqual(len(records), 5)
        self.assertEqual(records[0]["id"], "basicvsrpp")
        self.assertFalse(records[0]["complete"])
        self.assertIn("repo_url", records[0]["missing_fields"])
        self.assertIn("weights.source", records[0]["missing_fields"])
        self.assertIn("weights.local_path", records[0]["missing_fields"])
        self.assertIn("commands.evaluate", records[0]["missing_fields"])

    def test_export_baseline_registry_files(self):
        config = {
            "registry": {"name": "unit_baselines", "purpose": "unit_test"},
            "baselines": {
                "toy": {
                    "display_name": "ToyBaseline",
                    "category": "unit",
                    "status": "ready",
                    "repo_url": "https://example.test/repo",
                    "commit": "abc123",
                    "weights": {"source": "local", "local_path": "checkpoints/toy.pt"},
                    "commands": {
                        "setup": "python -m pip install -e .",
                        "infer": "python infer.py",
                        "evaluate": "python eval.py",
                    },
                    "metrics_path": "experiments/baselines/toy/metrics.json",
                }
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            export = export_baseline_registry(config, tmpdir, name="toy_baselines")

            self.assertEqual(export.baseline_count, 1)
            self.assertEqual(export.complete_count, 1)
            for path in export.files.values():
                self.assertTrue(Path(path).exists())

            payload = json.loads(Path(export.files["json"]).read_text(encoding="utf-8"))
            self.assertEqual(payload["complete_count"], 1)
            table = Path(export.files["markdown"]).read_text(encoding="utf-8")
            checklist = Path(export.files["checklist"]).read_text(encoding="utf-8")
            self.assertIn("ToyBaseline", table)
            self.assertIn("missing: none", checklist)


if __name__ == "__main__":
    unittest.main()
