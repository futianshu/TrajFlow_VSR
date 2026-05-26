import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from trajflow_vsr.experiments import export_paper_tables, load_comparison  # noqa: E402


class PaperTablesTest(unittest.TestCase):
    def test_export_ranking_and_matrix_tables(self):
        comparison = {
            "metrics": ["psnr", "fps"],
            "rows": [
                {
                    "name": "frames_3_x_raster",
                    "status": "ok",
                    "psnr": 30.125,
                    "fps": 12.0,
                    "latency_seconds": 0.1,
                    "profile.parameters": 1000,
                },
                {
                    "name": "frames_7_x_ot_sb",
                    "status": "ok",
                    "psnr": 31.5,
                    "fps": 9.25,
                    "latency_seconds": 0.2,
                    "profile.parameters": 1000,
                },
            ],
            "ranking": [
                {"name": "frames_7_x_ot_sb", "rank": 1, "selection_score": 0.9},
                {"name": "frames_3_x_raster", "rank": 2, "selection_score": 0.4},
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "summary.json"
            source.write_text(json.dumps({"comparison": comparison}), encoding="utf-8")
            loaded = load_comparison(source)
            export = export_paper_tables(loaded, Path(tmpdir) / "tables", name="stage_b", metrics=["psnr", "fps"])

            ranking_md = Path(export.files["ranking_markdown"])
            psnr_matrix = Path(export.files["psnr_matrix_markdown"])
            fps_matrix = Path(export.files["fps_matrix_latex"])
            manifest = Path(export.files["manifest"])

            self.assertTrue(ranking_md.exists())
            self.assertTrue(psnr_matrix.exists())
            self.assertTrue(fps_matrix.exists())
            self.assertTrue(manifest.exists())
            self.assertIn("frames_7_x_ot_sb", ranking_md.read_text(encoding="utf-8"))
            self.assertIn("| frames_3 | 30.12 |  |", psnr_matrix.read_text(encoding="utf-8"))
            self.assertIn("\\caption{stage\\_b fps matrix}", fps_matrix.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
