import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from trajflow_vsr.utils.config import apply_overrides, load_config, parse_simple_yaml  # noqa: E402


class ConfigTest(unittest.TestCase):
    def test_parse_simple_yaml_nested_mapping(self):
        config = parse_simple_yaml(
            """
project:
  name: trajflow-vsr
runtime:
  seed: 20260524
  dry_run: true
data:
  scales: [2, 3, 4]
"""
        )
        self.assertEqual(config["project"]["name"], "trajflow-vsr")
        self.assertEqual(config["runtime"]["seed"], 20260524)
        self.assertIs(config["runtime"]["dry_run"], True)
        self.assertEqual(config["data"]["scales"], [2, 3, 4])

    def test_load_stage_a_config(self):
        config = load_config("configs/train/stage_a_tokenizer.yaml")
        self.assertEqual(config["stage"]["name"], "stage_a_tokenizer")
        self.assertEqual(config["model"]["name"], "trajflow_vsr")

    def test_load_stage_b_frame_manifest_full_config(self):
        config = load_config("configs/train/stage_b_frame_manifest_full.yaml")
        self.assertEqual(config["stage"]["name"], "stage_b_deterministic")
        self.assertEqual(config["data"]["name"], "frame_manifest")
        self.assertTrue(config["data"]["sample_sequential_clips"])
        self.assertTrue(config["data"]["require_paired"])
        self.assertFalse(config["model"]["flow"]["enabled"])
        self.assertGreater(config["optimizer"]["max_steps"], 1)

    def test_load_vimeo90k_formal_training_templates(self):
        for path in [
            "configs/train/stage_a_vimeo90k_mild_real_x4.yaml",
            "configs/train/stage_b_vimeo90k_mild_real_x4.yaml",
            "configs/train/stage_c_vimeo90k_mild_real_x4.yaml",
            "configs/train/stage_d_vimeo90k_mild_real_x4.yaml",
            "configs/train/stage_e_vimeo90k_mild_real_x4.yaml",
        ]:
            config = load_config(path)
            self.assertTrue(config["data"]["require_paired"])
            self.assertIn("vimeo90k_mild_real_x4", config["data"]["manifest_path"])
            if config["stage"]["name"] == "stage_b_deterministic":
                self.assertFalse(config["model"]["flow"]["enabled"])
            if config["stage"]["name"] in {"stage_c_rectified_flow", "stage_d_distill", "stage_e_streaming"}:
                self.assertTrue(config["model"]["flow"]["enabled"])

    def test_load_no_reference_eval_config(self):
        config = load_config("configs/eval/no_reference.yaml")
        self.assertTrue(config["evaluation"]["reference_free"])
        self.assertEqual(config["evaluation"]["protocol"], "no_reference")
        self.assertFalse(config["data"]["require_paired"])

    def test_load_stage_b_scan_ablation_config(self):
        config = load_config("configs/ablation/stage_b_scan_policy_grid.yaml")
        self.assertEqual(config["base_config"], "configs/train/stage_b_deterministic.yaml")
        self.assertIn("ot_sb_no_koopman", config["ablation"]["variants"])

    def test_load_stage_b_controlled_motion_ablation_config(self):
        config = load_config("configs/ablation/stage_b_controlled_motion_scan.yaml")
        self.assertEqual(config["base_config"], "configs/train/stage_b_deterministic.yaml")
        self.assertIn("data.name=controlled_motion", config["ablation"]["common_overrides"])
        self.assertIn("bridge_temporal", config["ablation"]["variants"])
        self.assertIn("ot_sb_topk_supervised", config["ablation"]["variants"])

    def test_load_stage_b_context_length_ablation_config(self):
        config = load_config("configs/ablation/stage_b_context_length_grid.yaml")
        self.assertEqual(config["base_config"], "configs/train/stage_b_deterministic.yaml")
        self.assertIn("frames_63", config["ablation"]["variants"])
        self.assertEqual(config["ablation"]["selection"]["directions"]["fps"], "max")

    def test_load_stage_b_context_scan_matrix_config(self):
        config = load_config("configs/ablation/stage_b_context_scan_matrix.yaml")
        self.assertEqual(config["base_config"], "configs/train/stage_b_deterministic.yaml")
        axes = config["ablation"]["matrix"]["axes"]
        self.assertEqual(axes["frames"]["values"][0], "frames_3")
        self.assertIn("ot_sb", axes["scan"]["values"])
        self.assertEqual(config["ablation"]["matrix"]["joiner"], "_x_")

    def test_load_core_module_and_flow_ablation_configs(self):
        core = load_config("configs/ablation/stage_b_core_module_grid.yaml")
        self.assertIn("w_o_unbalanced_ot", core["ablation"]["variants"])
        self.assertIn("w_o_data_consistency", core["ablation"]["variants"])

        flow = load_config("configs/ablation/stage_c_flow_step_grid.yaml")
        self.assertIn("deterministic_no_flow", flow["ablation"]["variants"])
        self.assertIn("flow_four_step", flow["ablation"]["variants"])

        distill = load_config("configs/ablation/stage_d_distillation_grid.yaml")
        self.assertIn("w_o_consistency_distillation", distill["ablation"]["variants"])

        protocol = load_config("configs/ablation/stage_e_protocol_grid.yaml")
        self.assertIn("mixed_offline_streaming", protocol["ablation"]["variants"])

    def test_load_ccfa_core_benchmark_config(self):
        config = load_config("configs/benchmark/ccfa_core_benchmark.yaml")
        self.assertEqual(config["benchmark"]["name"], "ccfa_core_benchmark")
        self.assertIn("trajflow_stage_b", config["benchmark"]["methods"])
        self.assertIn("codec_motion", config["benchmark"]["degradations"])
        self.assertEqual(config["benchmark"]["scales"]["x4"]["value"], 4)

    def test_apply_overrides(self):
        config = {"runtime": {"dry_run": True}, "optimizer": {"max_steps": 1}}
        merged = apply_overrides(config, ["runtime.dry_run=false", "optimizer.max_steps=3"])
        self.assertIs(merged["runtime"]["dry_run"], False)
        self.assertEqual(merged["optimizer"]["max_steps"], 3)


if __name__ == "__main__":
    unittest.main()
