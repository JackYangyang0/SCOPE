import unittest

from SCOPE.app import unique_region_names
from SCOPE.llm.patch_generator import normalize_requested_regions
from SCOPE.llm.concrete_code_generator import CUDA_PATCH_REGIONS
from SCOPE.verification.gemm_semantic_checker import check_warp_lane_fragment_coverage


class CoupledTileRegionTests(unittest.TestCase):
    def test_next_tile_is_supported_through_protocol(self):
        self.assertIn("NEXT_TILE_LOAD", CUDA_PATCH_REGIONS)
        self.assertEqual(normalize_requested_regions(["NEXT_TILE_LOAD"]), ["NEXT_TILE_LOAD"])

    def test_load_context_includes_next_tile(self):
        self.assertIn("NEXT_TILE_LOAD", unique_region_names(["GLOBAL_TO_SHARED_LOAD"]))

    def test_layout_context_includes_all_consumers(self):
        regions = unique_region_names(["SHARED_DECL"])
        for name in ("GLOBAL_TO_SHARED_LOAD", "NEXT_TILE_LOAD", "COMPUTE_INNER", "MAIN_LOOP"):
            self.assertIn(name, regions)

    def test_equivalent_mapping_not_rejected(self):
        code = "const int lane_cols = WNITER / TN; const int Trow = lane / lane_cols; const int Tcol = lane - Trow * lane_cols;"
        config = dict(WM=32, WN=32, WMITER=16, WNITER=32, TM=4, TN=4)
        result = check_warp_lane_fragment_coverage(code, config, {"hardware": {"warp_size": 32}})
        self.assertEqual(result["status"], "pass")


if __name__ == "__main__":
    unittest.main()
