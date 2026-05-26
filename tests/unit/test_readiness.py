import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from trajflow_vsr.experiments import audit_project_readiness, export_readiness_audit  # noqa: E402


class ReadinessAuditTest(unittest.TestCase):
    def test_audit_reports_external_gaps_without_failing(self):
        report = audit_project_readiness(ROOT)

        self.assertIn("counts", report)
        self.assertGreater(report["counts"]["ok"], 0)
        self.assertGreater(report["counts"]["external_required"], 0)
        categories = {check["category"] for check in report["checks"]}
        self.assertIn("data_manifests", categories)
        self.assertIn("baselines", categories)
        self.assertIn("official_metrics", categories)

    def test_export_readiness_audit_files(self):
        report = audit_project_readiness(ROOT)
        with tempfile.TemporaryDirectory() as tmpdir:
            export = export_readiness_audit(report, tmpdir, name="unit_readiness")

            for path in export.files.values():
                self.assertTrue(Path(path).exists())
            payload = json.loads(Path(export.files["json"]).read_text(encoding="utf-8"))
            self.assertEqual(payload["name"], "ccfa_proposal_readiness")
            table = Path(export.files["markdown"]).read_text(encoding="utf-8")
            self.assertIn("CCF-A Readiness Audit", table)


if __name__ == "__main__":
    unittest.main()
