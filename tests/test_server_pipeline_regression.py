import unittest

from SCOPE.app import launch_config_lines, deterministic_code_regions
from SCOPE.generate_ir.stage_controller import derive_fields, evaluate_text_condition


def tiled_ir():
    return {"tiling": {"block_m": 64, "block_n": 64, "block_k": 16,
                       "thread_m": 4, "thread_n": 4,
                       "warp_tile": {"warp_m": 32, "warp_n": 32,
                                     "warp_m_iter": 1, "warp_n_iter": 1}},
            "hardware": {"warp_size": 32, "max_shared_memory_per_block_bytes": 49152},
            "memory": {"use_shared_memory": True, "shared_A": {"enabled": True},
                       "shared_B": {"enabled": True}}}


class ServerPipelineRegressionTests(unittest.TestCase):
    def test_derivation_does_not_depend_on_field_metadata(self):
        ir = tiled_ir()
        derive_fields(ir)
        warp = ir["tiling"]["warp_tile"]
        self.assertEqual((warp["warp_m_iter"], warp["warp_n_iter"]), (16, 32))
        self.assertEqual(ir["mapping"]["outputs_per_thread"], 32)
        self.assertEqual(ir["resource"]["shared_memory"]["pipeline_multiplier"], 1)

    def test_launch_cannot_materialize_default_one_fragment(self):
        lines = "\n".join(launch_config_lines(tiled_ir()))
        self.assertIn("WMITER = 16;", lines)
        self.assertIn("WNITER = 32;", lines)

    def test_mapping_synchronizes_launch(self):
        self.assertIn("LAUNCH_CONFIG", deterministic_code_regions(
            "Mapping.WarpThreadTile.WarpLaneFragment2D", tiled_ir()))

    def test_pipeline_multiplier_counted_once(self):
        ir = tiled_ir()
        ir["pipeline"] = {"stage_count": 4, "double_buffering": True}
        derive_fields(ir)
        self.assertEqual(ir["resource"]["shared_memory"]["total_bytes"], 32768)
        result = evaluate_text_condition(ir,
            "shared_memory_bytes * pipeline_multiplier <= hardware.max_shared_memory_per_block_bytes")
        self.assertEqual(result.status, "pass")


if __name__ == "__main__":
    unittest.main()
