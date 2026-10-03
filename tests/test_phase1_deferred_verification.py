import unittest
from unittest.mock import patch

from SCOPE.generate_ir.stage_controller import StageController, tiling_hierarchy_reject_reason


class Phase1VerificationTests(unittest.TestCase):
    def test_ir_stage_check_does_not_check_partial_source(self):
        controller = StageController({}, {}, "Tiling", {})
        with patch("SCOPE.generate_ir.stage_controller.run_stage_static_code_checks",
                   side_effect=AssertionError("partial source must not be checked")):
            controller.verify_stage(include_code_checks=False)

    def test_block_prunes_warp(self):
        ir = {"tiling": {"block_m": 64, "block_n": 64},
              "hardware": {"warp_size": 32, "max_threads_per_block": 1024}}
        self.assertIsNone(tiling_hierarchy_reject_reason("Tiling.WarpTile.32x32", ir))
        self.assertIsNotNone(tiling_hierarchy_reject_reason("Tiling.WarpTile.128x32", ir))

    def test_warp_prunes_thread(self):
        ir = {"tiling": {"warp_tile": {"warp_m": 32, "warp_n": 32}},
              "hardware": {"warp_size": 32}}
        self.assertIsNone(tiling_hierarchy_reject_reason("Tiling.ThreadTile.4x4", ir))
        self.assertIsNotNone(tiling_hierarchy_reject_reason("Tiling.ThreadTile.3x4", ir))


if __name__ == "__main__":
    unittest.main()
