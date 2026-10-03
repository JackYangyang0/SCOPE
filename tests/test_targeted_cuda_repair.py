import unittest
from SCOPE.verification.targeted_cuda_repair import (
    repair_guarded_current_tile_load,
    repair_known_cuda_defects,
    repair_synchronous_double_buffer_schedule,
    remove_known_redeclarations,
    remove_duplicate_first_tile,
)
from SCOPE.verification.build_run_verifier import mark_unrun

CONFIG = '''static const int BM=128;
static const int BN=128;
static const int BK=16;
static const int WM=16;
static const int WN=64;'''
LOAD = '''int local_k = vec_idx / (BM / 4);
int local_m = (vec_idx % (BM / 4)) * 4;
int gm = tile_m0 + local_m;
int gk = local_k;
float4 tmp = FLOAT4(A[gm * K + gk]);
As[0][local_k][local_m] = tmp.x;
As[0][local_k][local_m + 1] = tmp.y;
As[0][local_k][local_m + 2] = tmp.z;
As[0][local_k][local_m + 3] = tmp.w;'''
IR={'problem':{'M':1024,'N':1024,'K':1024,'layout_A':'row_major','trans_A':False}}


class TargetedRepairTests(unittest.TestCase):
    def test_both_loads_preserve_padding_and_vector_width(self):
        source=CONFIG+'\n__shared__ float As[2][BK][BM + 1];\n'+LOAD+'\n'+LOAD.replace('gk = local_k','gk = bkIdx * BK + local_k').replace('As[0]', 'As[mem_flag]')
        repaired, edits=repair_known_cuda_defects(source, IR)
        self.assertEqual(len(edits), 2)
        self.assertIn('As[mem_flag][local_k + 3][local_m]', repaired)
        self.assertIn('As[2][BK][BM + 1]',repaired)
        self.assertEqual(repaired.count('FLOAT4(A['),2)
        self.assertEqual(repair_known_cuda_defects(repaired, IR), (repaired, []))

    def test_unsupported_shape_or_transpose_is_not_rewritten(self):
        source=CONFIG+'\n__shared__ float As[2][BK][BM];\n'+LOAD
        for problem in ({**IR['problem'],'K':19},{**IR['problem'],'trans_A':True}):
            self.assertEqual(repair_known_cuda_defects(source, {'problem':problem}), (source, []))

    def test_duplicate_declarations_only_in_same_scope(self):
        declaration='dim3 threadsPerBlock((BM * BN) / (WM * WN) * 32);'
        source='{'+declaration+'{'+declaration+'}'+declaration+'}'
        repaired, changes=remove_known_redeclarations(source,{})
        self.assertEqual(len(changes),1)
        self.assertEqual(repaired.count(declaration),2)
        unequal='{dim3 threadsPerBlock(32); dim3 threadsPerBlock(64);}'
        self.assertEqual(remove_known_redeclarations(unequal,{}),(unequal,[]))

    def test_beta_fast_path_keeps_general_fallback(self):
        source='C[c_index] = alpha * results[m][n] + C[c_index];'
        ir={'strategy':{'applied_strategy_ids':['Epilogue.BetaOne.FastPath']}}
        repaired, changes=repair_known_cuda_defects(source,ir)
        self.assertIn('if (beta == 1.0f)',repaired)
        self.assertIn('beta * C[c_index]',repaired)
        self.assertEqual(repair_known_cuda_defects(repaired,ir),(repaired,[]))

    def test_compile_failure_clears_stale_runtime_error(self):
        ir={'verification':{'runtime_safety':{'returncode':1,'misaligned_address':True,'cuda_error':'misaligned address'}}}
        mark_unrun(ir,'compile failed')
        safety=ir['verification']['runtime_safety']
        self.assertFalse(safety['misaligned_address'])
        self.assertNotIn('returncode',safety)

    def test_duplicate_tile_zero_reduction_removed_only_if_identical(self):
        compute='#pragma unroll\nfor (int k = 0; k < BK; ++k) { results[0] += As[0][k][0] * Bs[0][k][0]; }'
        loop='for (int bkIdx = 1; bkIdx < k_tiles; ++bkIdx) { __syncthreads(); const int comp_flag = (bkIdx - 1) & 1; const int mem_flag = bkIdx & 1; '
        source='/* MAIN_LOOP_BEGIN */\n'+compute+'\n'+loop+compute.replace('[0][k]','[comp_flag][k]')+' }'
        fixed,changes=remove_duplicate_first_tile(source)
        self.assertEqual(len(changes),1)
        self.assertEqual(fixed.count('results[0] +='),1)
        different=source.replace('As[comp_flag][k][0]','As[comp_flag][k][1]')
        self.assertEqual(remove_duplicate_first_tile(different),(different,[]))

    def test_guarded_current_tile_load_is_unwrapped(self):
        source = '''for (int bkIdx = 1; bkIdx < k_tiles; ++bkIdx) {
const int mem_flag = bkIdx & 1;
/* NEXT_TILE_LOAD_BEGIN */
if (bkIdx + 1 < k_tiles) {
  const int global_k = bkIdx * BK;
  As[mem_flag][0][0] = A[global_k];
  Bs[mem_flag][0][0] = B[global_k * N];
}
/* NEXT_TILE_LOAD_END */
}
const int comp_flag = (k_tiles - 1) & 1;'''
        repaired, changes = repair_guarded_current_tile_load(source)
        self.assertEqual([item['id'] for item in changes], ['FINAL_K_TILE_LOAD_GUARD_REMOVED'])
        self.assertNotIn('bkIdx + 1 < k_tiles', repaired)
        self.assertIn('const int global_k = bkIdx * BK;', repaired)
        self.assertEqual(repair_guarded_current_tile_load(repaired), (repaired, []))

    def test_true_next_tile_prefetch_guard_is_preserved(self):
        source = '''for (int bkIdx = 1; bkIdx < k_tiles; ++bkIdx) {
/* NEXT_TILE_LOAD_BEGIN */
if (bkIdx + 1 < k_tiles) {
  const int global_k = (bkIdx + 1) * BK;
  As[0][0][0] = A[global_k]; Bs[0][0][0] = B[global_k * N];
}
/* NEXT_TILE_LOAD_END */
}
const int comp_flag = (k_tiles - 1) & 1;'''
        self.assertEqual(repair_guarded_current_tile_load(source), (source, []))

    def test_load_then_compute_double_buffer_reads_current_stage(self):
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
  __syncthreads();
}
/* MAIN_LOOP_END */'''
        repaired, changes = repair_synchronous_double_buffer_schedule(source)
        self.assertEqual([item['id'] for item in changes], ['SYNCHRONOUS_DOUBLE_BUFFER_READ_STAGE'])
        self.assertIn('const int comp_flag = bkIdx & 1;', repaired)
        self.assertEqual(repair_synchronous_double_buffer_schedule(repaired), (repaired, []))

    def test_legacy_guarded_float4_mapping_and_next_b_offset_are_repaired(self):
        source = CONFIG + r'''
const int tid = threadIdx.x;
const int thread_num = BM * BN / WM / WN * 32;
const int load_a_smem_m = tid % BM;
const int load_a_smem_k = (tid / BM) % BK;
const int load_b_smem_n = tid % BN;
const int load_b_smem_k = (tid / BN) % BK;
const int hightA = CEIL_DIV(BM * BK, thread_num);
const int hightB = CEIL_DIV(BN * BK, thread_num);
As[0][smem_k_base + 3][smem_m] = tmp.w;
Bs[0][smem_k][smem_n_base + 3] = tmp.w;
/* NEXT_TILE_LOAD_BEGIN */
const int global_k_col = smem_k_base + bkIdx * BK;
const int global_k_row = smem_k;
As[mem_flag][smem_k_base + 3][smem_m] = tmp.w;
Bs[mem_flag][smem_k][smem_n_base + 3] = tmp.w;
/* NEXT_TILE_LOAD_END */
dim3 threadsPerBlock((BM * BN) / (WM * WN) * 32);
'''
        repaired, changes = repair_known_cuda_defects(source, IR)
        ids = [item['id'] for item in changes]
        self.assertIn('LEGACY_FLOAT4_LOAD_OWNERSHIP', ids)
        self.assertIn('NEXT_B_TILE_K_OFFSET', ids)
        self.assertIn('const int load_a_smem_k = (tid % (BK / 4)) * 4;', repaired)
        self.assertIn('const int load_b_smem_n = (tid % (BN / 4)) * 4;', repaired)
        self.assertIn('const int global_k_row = bkIdx * BK + smem_k;', repaired)


if __name__=='__main__':
    unittest.main()
