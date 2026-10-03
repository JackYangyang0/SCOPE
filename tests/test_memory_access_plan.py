import tempfile
import unittest
import shutil
import hashlib
from pathlib import Path

from SCOPE.verification.memory_access_plan import (
    duplicate_block_offset_defects, enforce_memory_access_plan, memory_access_consistency_defects,
)

SCOPE_ROOT = Path(__file__).resolve().parents[1]


class MemoryAccessPlanTests(unittest.TestCase):
    def test_offset_macro_does_not_mix_b_row_and_column_coordinates(self):
        valid = 'float4 value = FLOAT4(B[OFFSET(global_k, tile_n0 + bn, N)]);'
        self.assertNotIn(
            'B_K_INDEX_USES_N_TILE_OFFSET',
            {item['id'] for item in duplicate_block_offset_defects(valid)},
        )
        invalid = 'float value = B[OFFSET(tile_n0 + global_k, bn, N)];'
        self.assertIn(
            'B_K_INDEX_USES_N_TILE_OFFSET',
            {item['id'] for item in duplicate_block_offset_defects(invalid)},
        )

    def test_generated_address_and_register_contract_defects_are_detected(self):
        source = r"""
/* SHARED_DECL_BEGIN */
__shared__ float As[2][BK][BM];
__shared__ float Bs[2][BK][BN];
/* SHARED_DECL_END */
/* INDEX_MAPPING_BEGIN */
const int a_g_m = tile_m0 + a_m;
/* INDEX_MAPPING_END */
/* REGISTER_DECL_BEGIN */
float regM[TM];
float results[(WM / WMITER * TM) * (WN / WNITER * TN)];
/* REGISTER_DECL_END */
const int A_offset = tile_m0 * K;
/* GLOBAL_TO_SHARED_LOAD_BEGIN */
As[0][0][0] = A[a_g_m * K + k + A_offset];
/* GLOBAL_TO_SHARED_LOAD_END */
/* MAIN_LOOP_BEGIN */
for (int bkIdx = 1; bkIdx < k_tiles; ++bkIdx) {
    As[0][0][0] = A[bkIdx * BK];
    regM[wm * TM + i] = As[0][0][0];
    results[(wm * TM + i) * WN + j] += regM[0];
}
/* MAIN_LOOP_END */
/* STORE_BEGIN */
C[0] = results[(wm * TM + i) * (WN / WNITER * TN) + j];
/* STORE_END */
/* LAUNCH_CONFIG_BEGIN */
static const int BM=64, BN=64, BK=16, WM=32, WN=32, WMITER=8, WNITER=16, TM=2, TN=2;
/* LAUNCH_CONFIG_END */
"""
        ids = {item["id"] for item in memory_access_consistency_defects(source)}
        self.assertIn("A_BLOCK_OFFSET_APPLIED_TWICE", ids)
        self.assertIn("REGISTER_EXTENT_MISMATCH_REGM", ids)
        self.assertIn("ACCUMULATOR_LAYOUT_MISMATCH_RESULTS", ids)
        self.assertIn("FIRST_K_TILE_NOT_ACCUMULATED", ids)

    def test_mixed_layout_and_unproven_vector_load_are_detected(self):
        source = """
__shared__ float As[2][BM][BK]; __shared__ float Bs[2][BN][BK];
/* GLOBAL_TO_SHARED_LOAD_BEGIN */ scalar(); /* GLOBAL_TO_SHARED_LOAD_END */
/* NEXT_TILE_LOAD_BEGIN */ float4 v = FLOAT4(A[load_a_smem_k]); As[x][load_a_smem_k][m]=0; /* NEXT_TILE_LOAD_END */
"""
        ids = {item["id"] for item in memory_access_consistency_defects(source)}
        self.assertIn("A_SHARED_LAYOUT_MIXED", ids)
        self.assertIn("TILE_LOAD_VECTOR_POLICY_MISMATCH", ids)
        self.assertIn("UNPROVEN_FLOAT4_TILE_LOAD", ids)

    def test_inconsistent_kernel_is_materialized_to_safe_protocol(self):
        template = (SCOPE_ROOT / "gemm_code" / "skeleton_template" / "cuda_kernel.cuh").read_text(encoding="utf-8")
        broken = template.replace("As[2][BK][BM]", "As[2][BM][BK]")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "cuda_kernel.cuh"
            path.write_text(broken, encoding="utf-8")
            result = enforce_memory_access_plan(Path(folder), allow_scalar_fallback=True)
            fixed = path.read_text(encoding="utf-8")
        self.assertEqual(result["status"], "pass")
        self.assertTrue(result["materialized"])
        self.assertIn("As[2][BK][BM]", fixed)
        self.assertIn("const int k_tiles = CEIL_DIV(K, BK);", fixed)
        self.assertNotIn("float4 tmp =", fixed)
        self.assertIn("if (global_m < M && global_n < N)", fixed)
        self.assertEqual(memory_access_consistency_defects(fixed), [])

    def test_real_generated_terminal_chains_are_repairable_without_mutating_sources(self):
        code_root = Path(__file__).resolve().parents[1] / "gemm_code" / "code"
        terminal_dirs = sorted(path for path in code_root.glob("chain.*") if (path / "terminal_chain.json").exists())
        if not terminal_dirs:
            self.skipTest("no generated terminal chains are available")

        defective_count = 0
        for source_dir in terminal_dirs:
            kernel = source_dir / "cuda_kernel.cuh"
            before_bytes = kernel.read_bytes()
            defects = memory_access_consistency_defects(before_bytes.decode("utf-8", errors="replace"))
            if not defects:
                continue
            defective_count += 1
            with self.subTest(chain=source_dir.name), tempfile.TemporaryDirectory() as folder:
                target = Path(folder)
                for name in ("cuda_kernel.cuh", "main.cpp", "kernel.h"):
                    if (source_dir / name).exists():
                        shutil.copy2(source_dir / name, target / name)
                result = enforce_memory_access_plan(target, allow_scalar_fallback=True)
                repaired = (target / "cuda_kernel.cuh").read_text(encoding="utf-8")
                self.assertEqual(result["status"], "pass")
                self.assertTrue(result["materialized"])
                self.assertEqual(memory_access_consistency_defects(repaired), [])
            self.assertEqual(hashlib.sha256(kernel.read_bytes()).digest(), hashlib.sha256(before_bytes).digest())

        if not defective_count:
            self.skipTest("current generated terminal chains contain no matching memory-access defects")


if __name__ == "__main__":
    unittest.main()
