import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from SCOPE.tune_cuda import tune_sources
from SCOPE.tests.test_shape_search import TILE, state, verified


class TuneExistingCudaTests(unittest.TestCase):
    @patch("SCOPE.tune_cuda.enumerate_tiles")
    def test_candidates_only_change_source_constants_and_continue_after_failure(self,build):
        tiles = [TILE,{**TILE,"BK":16},{**TILE,"BK":32}]
        build.return_value = tiles
        source = "template<int BM> void gemm() { /* original optimized body */ }\nvoid cuda_gemm() {\n/* LAUNCH_CONFIG_BEGIN */\n"
        source += "\n".join(f"static const int {key} = {value};" for key,value in TILE.items()
                            if key not in ("threads_per_block","warps_per_block"))
        source += "\n// static const int BM = 999;\n/* LAUNCH_CONFIG_END */\n}\n"
        sources = {"cuda_kernel.cuh":source,"main.cpp":"// fixed harness","kernel.h":"// fixed interface"}
        calls = []
        def verify(ir,**kwargs):
            calls.append(kwargs)
            self.assertEqual((kwargs["source_dir"]/"main.cpp").read_text(),sources["main.cpp"])
            self.assertIn("original optimized body",(kwargs["source_dir"]/"cuda_kernel.cuh").read_text())
            if len(calls)==2:
                raise RuntimeError("invalid candidate")
            return verified(ir,10+len(calls))
        with tempfile.TemporaryDirectory() as tmp:
            original = copy.deepcopy(sources)
            result = tune_sources(sources,"cuda_kernel.cuh",state(),{},Path(tmp)/"out",
                                  max_candidates=3,rounds=3,verify=verify)
            self.assertEqual(sources,original)
            self.assertEqual(len(calls),3)  # Original tuple is deduplicated.
            self.assertFalse(result["results"][1]["accepted"])
            self.assertEqual(len(result["top_results"]),2)
            self.assertTrue((Path(tmp)/"out"/"top_3"/"rank_1"/"cuda_kernel.cuh").exists())
            self.assertTrue(all(c["benchmark_runs"]==5 for c in calls))

    @patch("SCOPE.tune_cuda.enumerate_tiles",return_value=[TILE])
    def test_unsupported_source_is_reported_without_template_replacement(self,*_):
        sources = {"cuda_kernel.cuh":"// arbitrary source","main.cpp":"// harness"}
        with tempfile.TemporaryDirectory() as tmp:
            result = tune_sources(sources,"cuda_kernel.cuh",state(),{},Path(tmp)/"out",
                                  verify=lambda ir,**kwargs:verified(ir))
            self.assertIn("unsupported_source_parameters",result["stop_reason"])
            self.assertEqual(len(result["results"]),1)


if __name__ == "__main__":
    unittest.main()
