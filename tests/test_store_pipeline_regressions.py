import unittest
from SCOPE.verification.memory_access_plan import k_tile_coverage_defects
from SCOPE.verification.gemm_semantic_checker import check_store_bounds_use_global_coordinates


class StorePipelineRegressions(unittest.TestCase):
    def test_global_alias_guard(self):
        source = '''/* STORE_BEGIN */
const int gm = tile_m0 + m;
const int gn_base = tile_n0 + n;
const int gn = gn_base + 1;
if (gm < M && gn < N) { C[OFFSET(gm,gn,N)] = x; }
/* STORE_END */'''
        self.assertEqual(check_store_bounds_use_global_coordinates(source, source)['status'], 'pass')
        bad = source.replace('if (gm < M && gn < N)', 'if (m < M && n < N)')
        self.assertEqual(check_store_bounds_use_global_coordinates(bad, bad)['status'], 'fail')

    def test_other_buffer_load_does_not_drop_first_tile(self):
        source = '''/* MAIN_LOOP_BEGIN */
for (int bkIdx = 1; bkIdx < k_tiles; ++bkIdx) {
const int comp_flag = (bkIdx - 1) & 1;
const int mem_flag = bkIdx & 1;
As[mem_flag][k][m] = a;
Bs[mem_flag][k][n] = b;
__syncthreads();
regM[i] = As[comp_flag][k][m];
regN[j] = Bs[comp_flag][k][n];
results[i][j] += regM[i] * regN[j];
__syncthreads();
}
const int comp_flag = (k_tiles - 1) & 1;
/* MAIN_LOOP_END */'''
        self.assertEqual(k_tile_coverage_defects(source), [])
        bad = source.replace('const int mem_flag = bkIdx & 1;', 'const int mem_flag = (bkIdx - 1) & 1;')
        self.assertTrue(k_tile_coverage_defects(bad))

    def test_guarded_current_tile_load_starves_final_drain(self):
        source = '''/* MAIN_LOOP_BEGIN */
for (int bkIdx = 1; bkIdx < k_tiles; ++bkIdx) {
const int comp_flag = (bkIdx - 1) & 1;
const int mem_flag = bkIdx & 1;
results[0][0] += As[comp_flag][0][0] * Bs[comp_flag][0][0];
/* NEXT_TILE_LOAD_BEGIN */
if (bkIdx < k_tiles - 1) {
  const int global_k = bkIdx * BK;
  As[mem_flag][0][0] = A[global_k];
  Bs[mem_flag][0][0] = B[OFFSET(global_k, tile_n0, N)];
}
/* NEXT_TILE_LOAD_END */
}
const int comp_flag = (k_tiles - 1) & 1;
results[0][0] += As[comp_flag][0][0] * Bs[comp_flag][0][0];
/* MAIN_LOOP_END */'''
        defects = k_tile_coverage_defects(source)
        self.assertEqual([item['id'] for item in defects], ['FINAL_K_TILE_NOT_LOADED'])
        valid = source.replace('if (bkIdx < k_tiles - 1) {', '{')
        self.assertEqual(k_tile_coverage_defects(valid), [])

    def test_synchronous_load_then_compute_rejects_previous_stage(self):
        source = '''/* MAIN_LOOP_BEGIN */
results[0][0] += As[0][0][0] * Bs[0][0][0];
for (int bkIdx = 1; bkIdx < k_tiles; ++bkIdx) {
const int comp_flag = (bkIdx - 1) & 1;
const int mem_flag = bkIdx & 1;
const int global_k = bkIdx * BK;
As[mem_flag][0][0] = A[global_k];
Bs[mem_flag][0][0] = B[global_k * N];
__syncthreads();
results[0][0] += As[comp_flag][0][0] * Bs[comp_flag][0][0];
}
/* MAIN_LOOP_END */'''
        defects = k_tile_coverage_defects(source)
        self.assertEqual(
            [item['id'] for item in defects],
            ['SYNCHRONOUS_DOUBLE_BUFFER_READS_PREVIOUS_STAGE'],
        )

    def test_synchronous_fmaf_form_is_also_rejected(self):
        source = '''/* MAIN_LOOP_BEGIN */
results[0][0] = fmaf(As[0][0][0], Bs[0][0][0], results[0][0]);
for (int bkIdx = 1; bkIdx < k_tiles; ++bkIdx) {
const int comp_flag = (bkIdx - 1) & 1;
const int mem_flag = bkIdx & 1;
const int global_k = bkIdx * BK;
As[mem_flag][0][0] = A[global_k];
Bs[mem_flag][0][0] = B[global_k * N];
__syncthreads();
results[0][0] = fmaf(As[comp_flag][0][0], Bs[comp_flag][0][0], results[0][0]);
}
/* MAIN_LOOP_END */'''
        self.assertEqual(
            [item['id'] for item in k_tile_coverage_defects(source)],
            ['SYNCHRONOUS_DOUBLE_BUFFER_READS_PREVIOUS_STAGE'],
        )
