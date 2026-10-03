from __future__ import annotations

import unittest

from SCOPE.app import (
    anchor_regions_for_compiler_lines,
    apply_deterministic_compile_repairs,
    check_patch_anchor_pairs,
    extract_cuda_symbol_contract,
    repair_regions_from_diagnosis,
)
from SCOPE.verification.build_run_verifier import mark_unrun


SOURCE = """/* SHARED_DECL_BEGIN */
__shared__ float As[2][16][64];
/* SHARED_DECL_END */
/* REGISTER_DECL_BEGIN */
float regM[4];
/* REGISTER_DECL_END */
/* COMPUTE_INNER_BEGIN */
regA[0] = As[0][0][0];
/* COMPUTE_INNER_END */
/* LAUNCH_CONFIG_BEGIN */
gemm<64, 64, 16>
    __forceinline <<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
/* LAUNCH_CONFIG_END */
"""


class TieredRepairTests(unittest.TestCase):
    def test_compiler_lines_select_innermost_anchor(self):
        self.assertEqual(anchor_regions_for_compiler_lines(SOURCE, [7]), ["COMPUTE_INNER"])

    def test_first_repair_level_limits_edit_scope(self):
        diagnosis = {
            "defects": [{
                "defect_type": "Compile.UndeclaredKernelSymbol",
                "evidence": {"error_message": "cuda_kernel.cuh(7): identifier regA is undefined"},
            }]
        }
        regions = repair_regions_from_diagnosis(SOURCE, diagnosis, repair_level=1)
        self.assertLessEqual(len(regions), 2)
        self.assertIn("COMPUTE_INNER", regions)

    def test_symbol_contract_exposes_existing_arrays(self):
        contract = extract_cuda_symbol_contract(SOURCE)
        self.assertIn("As", contract["array_identifiers"])
        self.assertIn("regM", contract["array_identifiers"])
        self.assertNotIn("regA", contract["array_identifiers"])

    def test_deterministic_launch_repair_removes_invalid_qualifier(self):
        repaired = apply_deterministic_compile_repairs(
            SOURCE,
            {"defects": [{"defect_type": "Compile.CUDACompilationError", "message": "expected a ;"}]},
        )
        self.assertNotIn("__forceinline <<<", repaired)
        self.assertIn("gemm<64, 64, 16>", repaired)
        self.assertIn("<<<grid, block>>>", repaired)

    def test_compile_failure_removes_stale_performance_metrics(self):
        ir = {
            "performance": {
                "gflops": 5000.0,
                "gflops_mean": 4990.0,
                "latency_ms_trimmed_mean": 0.2,
                "benchmark_successful_runs": 5,
                "relative_to_cublas": 0.5,
            }
        }
        mark_unrun(ir, "compile failed")
        self.assertIsNone(ir["performance"]["gflops"])
        self.assertIsNone(ir["performance"]["latency_ms"])
        self.assertNotIn("gflops_mean", ir["performance"])
        self.assertNotIn("benchmark_successful_runs", ir["performance"])
        self.assertNotIn("relative_to_cublas", ir["performance"])

    def test_plain_cuda_region_anchor_damage_is_rejected(self):
        results = check_patch_anchor_pairs(
            "cuda_kernel.cuh",
            "/* COMPUTE_INNER_BEGIN */\nacc += a * b;\n",
        )
        failed = [item for item in results if item["status"] == "fail"]
        self.assertEqual(len(failed), 1)
        self.assertIn("COMPUTE_INNER", failed[0]["id"])


if __name__ == "__main__":
    unittest.main()
