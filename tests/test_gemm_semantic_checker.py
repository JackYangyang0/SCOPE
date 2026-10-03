from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from SCOPE.diagnosis.defect_diagnosis import diagnose_defects
from SCOPE.verification.gemm_semantic_checker import (
    check_gemm_semantic_obligations,
    check_k_loop_updates_A_and_B_tiles,
    check_register_accumulator_coverage,
    check_shared_memory_is_cooperatively_loaded,
    check_vector_shared_load_mapping_bounds,
)
from SCOPE.verification.gemm_semantic_repair import (
    build_throughput_cuda_kernel_cuh,
    generate_semantic_repair_candidate,
    normalize_throughput_launch_config,
)


SCOPE_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_CHAIN = SCOPE_ROOT / "gemm_code" / "code" / "chain.1-1-1-1-1-1-1-1-1-1-1-1"
REPAIRED_CHAIN = SCOPE_ROOT / "results" / "code" / "semantic_repair_test" / "chain.1-1-1-1-1-1-1-1-1-1-1-1.repaired"
SAMPLE_IR = {
    "problem": {
        "M": 512,
        "N": 512,
        "K": 512,
    },
    "tiling": {
        "block_m": 128,
        "block_n": 128,
        "block_k": 8,
        "thread_m": 2,
        "thread_n": 2,
    },
    "memory": {"use_shared_memory": True},
    "safety": {"boundary_policy": "static_divisible_no_guard"},
    "vectorization": {
        "A": {"alignment_proven": True},
        "B": {"alignment_proven": True},
        "C": {"alignment_proven": True},
    },
}


class GemmSemanticCheckerTests(unittest.TestCase):
    def test_vector_shared_load_mapping_rejects_overprovisioned_legacy_mapping(self):
        code = """
        __shared__ float As[2][BK][BM];
        __shared__ float Bs[2][BK][BN];
        const int tid = threadIdx.x;
        const int load_a_smem_m = tid / (BK / 4);
        const int load_a_smem_k = (tid % (BK / 4)) * 4;
        const int load_b_smem_k = tid / (BN / 4);
        const int load_b_smem_n = (tid % (BN / 4)) * 4;
        float4 tmp = FLOAT4(A[0]);
        As[0][load_a_smem_k][load_a_smem_m] = tmp.x;
        """
        report = check_vector_shared_load_mapping_bounds(
            code,
            {"BM": 64, "BN": 128, "BK": 16, "WM": 16, "WN": 32},
        )
        self.assertEqual(report["status"], "fail")
        self.assertEqual(report["failure_type"], "GEMM.Semantic.SharedLoadThreadMappingOutOfBounds")
        self.assertEqual(report["detail"]["hazards"][0]["operand"], "A")

    def test_vector_shared_load_mapping_accepts_flattened_bounded_loop(self):
        code = """
        __shared__ float As[2][BK][BM];
        __shared__ float Bs[2][BK][BN];
        const int tid = threadIdx.x;
        for (int loadIdx = tid * 4; loadIdx < BM * BK; loadIdx += blockDim.x * 4) {
            float4 tmp = FLOAT4(A[loadIdx]);
            As[0][loadIdx % BK][loadIdx / BK] = tmp.x;
        }
        """
        report = check_vector_shared_load_mapping_bounds(
            code,
            {"BM": 64, "BN": 128, "BK": 16, "WM": 16, "WN": 32},
        )
        self.assertEqual(report["status"], "pass")

    def test_vector_shared_load_mapping_accepts_explicit_legacy_guard(self):
        code = """
        __shared__ float As[2][BK][BM];
        __shared__ float Bs[2][BK][BN];
        const int tid = threadIdx.x;
        const int load_a_smem_m = tid / (BK / 4);
        const int load_a_smem_k = (tid % (BK / 4)) * 4;
        if (load_a_smem_m < BM) {
            float4 tmp = FLOAT4(A[0]);
            As[0][load_a_smem_k][load_a_smem_m] = tmp.x;
        }
        """
        report = check_vector_shared_load_mapping_bounds(
            code,
            {"BM": 64, "BN": 128, "BK": 16, "WM": 16, "WN": 32},
        )
        self.assertEqual(report["status"], "pass")

    def test_semantic_checks_accept_equivalent_names_and_temp_loads(self):
        code = """
        __shared__ float As[2][BK][BM];
        __shared__ float Bs[2][BK][BN];
        const int tid = threadIdx.x;
        const int load_a_smem_m = tid / (BK / 4);
        const int load_a_smem_k = (tid % (BK / 4)) * 4;
        const int load_b_smem_k = tid / (BN / 4);
        const int load_b_smem_n = (tid % (BN / 4)) * 4;
        const int hightA = 16;
        const int hightB = 1;
        float results[TM][TN] = {0.0f};

        for (int k0 = 0; k0 < K; k0 += BK) {
            for (int loadOffset = 0; loadOffset < BM; loadOffset += hightA) {
                float4 tmp = FLOAT4(A[OFFSET(load_a_smem_m + loadOffset, k0 + load_a_smem_k, K)]);
                As[0][load_a_smem_k][load_a_smem_m + loadOffset] = tmp.x;
                As[0][load_a_smem_k + 1][load_a_smem_m + loadOffset] = tmp.y;
            }
            for (int loadOffset = 0; loadOffset < BK; loadOffset += hightB) {
                FLOAT4(Bs[0][load_b_smem_k + loadOffset][load_b_smem_n]) =
                    FLOAT4(B[OFFSET(k0 + load_b_smem_k + loadOffset, load_b_smem_n, N)]);
            }
            for (int kk = 0; kk < BK; ++kk) {
                results[0][0] += As[0][kk][0] * Bs[0][kk][0];
            }
        }
        """

        self.assertEqual(check_register_accumulator_coverage(code, 4, 4)["status"], "pass")
        self.assertEqual(check_k_loop_updates_A_and_B_tiles(code)["status"], "pass")
        self.assertEqual(check_shared_memory_is_cooperatively_loaded(code, 128, 128, 16)["status"], "pass")

    def test_current_chain_reports_repairable_semantic_defects(self):
        if not SAMPLE_CHAIN.exists():
            self.skipTest(f"sample chain does not exist: {SAMPLE_CHAIN}")

        report = check_gemm_semantic_obligations(
            SAMPLE_CHAIN,
            SAMPLE_IR,
        )

        failed_ids = {item["id"] for item in report["results"] if item["status"] == "fail"}
        if report["accepted"]:
            self.assertEqual(failed_ids, set())
            return
        self.assertFalse(report["accepted"])
        self.assertIn("GEMM_THREAD_TILE_MAPPING_COVERS_BLOCK", failed_ids)

    def test_semantic_failures_become_structured_repair_defects(self):
        semantic_report = {
            "accepted": False,
            "launch_config": {"BM": 128, "BN": 128, "BK": 8, "TM": 2, "TN": 2},
            "results": [
                {
                    "id": "GEMM_THREAD_TILE_STORE_COVERS_TM_TN",
                    "status": "fail",
                    "message": "TM=2, TN=2, but the store region does not cover all TM*TN C elements per thread.",
                    "failure_type": "GEMM.Semantic.ThreadTileStoreIncomplete",
                    "repair_action": "add nested TM/TN store loops",
                    "detail": {"tm": 2, "tn": 2},
                },
                {
                    "id": "GEMM_K_LOOP_UPDATES_A_AND_B_TILES",
                    "status": "fail",
                    "message": "K loop does not reload both A and B tiles for each BK slice.",
                    "failure_type": "GEMM.Semantic.KLoopDataflowIncomplete",
                    "repair_action": "move cooperative A/B tile loads into the K loop",
                    "detail": {"updates_a": False, "updates_b": True},
                },
            ],
        }
        diagnosis = diagnose_defects({"code_completeness": {"semantic_obligations": semantic_report}})
        defect_types = {defect["defect_type"] for defect in diagnosis["defects"]}

        self.assertEqual(diagnosis["defect_count"], 2)
        self.assertIn("GEMM.Semantic.ThreadTileStoreIncomplete", defect_types)
        self.assertIn("GEMM.Semantic.KLoopDataflowIncomplete", defect_types)
        for defect in diagnosis["defects"]:
            self.assertTrue(defect["repair_action"])
            self.assertEqual(defect["evidence"]["source"], "code_semantic_obligation")

    def test_unlock_kernel_rejects_warp_lane_fragment_holes(self):
        bad_config = {
            "BM": 128,
            "BN": 64,
            "BK": 16,
            "WM": 32,
            "WN": 32,
            "WMITER": 32,
            "WNITER": 32,
            "TM": 4,
            "TN": 4,
        }
        with tempfile.TemporaryDirectory() as tmp:
            chain_dir = Path(tmp)
            (chain_dir / "cuda_kernel.cuh").write_text(build_throughput_cuda_kernel_cuh(bad_config), encoding="utf-8")
            report = check_gemm_semantic_obligations(
                chain_dir,
                {
                    "problem": {"M": 512, "N": 512, "K": 512},
                    "hardware": {"warp_size": 32},
                    "safety": {"boundary_policy": "static_divisible_no_guard"},
                    "vectorization": {
                        "A": {"alignment_proven": True},
                        "B": {"alignment_proven": True},
                        "C": {"alignment_proven": True},
                    },
                },
            )

        failed = {item["failure_type"] for item in report["results"] if item["status"] == "fail"}
        self.assertIn("GEMM.Semantic.WarpLaneFragmentCoverageIncomplete", failed)

    def test_unlock_launch_normalization_covers_one_warp(self):
        config = normalize_throughput_launch_config(
            {
                "BM": 128,
                "BN": 64,
                "BK": 16,
                "WM": 32,
                "WN": 32,
                "WMITER": 32,
                "WNITER": 32,
                "TM": 4,
                "TN": 4,
            }
        )

        self.assertEqual(config["WMITER"], 16)
        self.assertEqual(config["WNITER"], 32)
        self.assertEqual((config["WMITER"] // config["TM"]) * (config["WNITER"] // config["TN"]), 32)

    def test_current_chain_generates_repaired_code_candidate(self):
        if not SAMPLE_CHAIN.exists():
            self.skipTest(f"sample chain does not exist: {SAMPLE_CHAIN}")

        before_report = check_gemm_semantic_obligations(SAMPLE_CHAIN, SAMPLE_IR)
        diagnosis = diagnose_defects({"code_completeness": {"semantic_obligations": before_report}})
        repair_result = generate_semantic_repair_candidate(
            source_chain_dir=SAMPLE_CHAIN,
            output_chain_dir=REPAIRED_CHAIN,
            diagnosis=diagnosis,
            ir=SAMPLE_IR,
        )

        self.assertEqual(repair_result["status"], "repaired")
        self.assertTrue((REPAIRED_CHAIN / "cuda_kernel.cuh").exists())
        self.assertTrue((REPAIRED_CHAIN / "main.cpp").exists())
        self.assertTrue((REPAIRED_CHAIN / "kernel.h").exists())
        self.assertTrue((REPAIRED_CHAIN / "semantic_repair_result.json").exists())
        self.assertTrue(repair_result["semantic_obligations"]["accepted"])

        repaired_code = (REPAIRED_CHAIN / "cuda_kernel.cuh").read_text(encoding="utf-8")
        self.assertIn("for (int load_idx = tid; load_idx < BM * BK; load_idx += thread_count)", repaired_code)
        self.assertIn("const int tile_cols = BN / TN;", repaired_code)
        self.assertIn("const int local_m_base = tile_m * TM;", repaired_code)
        self.assertIn("const int local_n_base = tile_n * TN;", repaired_code)
        self.assertIn("for (int tm = 0; tm < TM; ++tm)", repaired_code)
        self.assertIn("for (int tn = 0; tn < TN; ++tn)", repaired_code)
        self.assertIn("C[OFFSET(global_m, global_n, N)]", repaired_code)

    def test_repaired_chain_has_no_remaining_semantic_defects(self):
        if not REPAIRED_CHAIN.exists():
            self.skipTest(f"repaired chain does not exist: {REPAIRED_CHAIN}")

        report = check_gemm_semantic_obligations(REPAIRED_CHAIN, SAMPLE_IR)
        diagnosis = diagnose_defects({"code_completeness": {"semantic_obligations": report}})

        self.assertTrue(report["accepted"])
        self.assertEqual(diagnosis["defect_count"], 0)


if __name__ == "__main__":
    unittest.main()
