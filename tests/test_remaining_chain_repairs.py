import unittest
import tempfile
from pathlib import Path
from SCOPE.verification.targeted_cuda_repair import repair_known_cuda_defects
from SCOPE.verification.optimization_preservation import preservation_defects


class RemainingChainRepairs(unittest.TestCase):
    def case(self, number):
        source = (Path(__file__).parent / 'fixtures' / f'remaining_chain{number}.cuh').read_text()
        bm, bk = (128, 32) if number == 1 else (64, 16)
        ir = {'problem': {'M':1024, 'N':1024, 'K':1024,
                         'layout_A':'row_major', 'trans_A':False},
              'tiling': {'block_m':bm, 'block_n':128, 'block_k':bk},
              'hardware': {'max_shared_memory_per_block_bytes':49152,
                           'max_shared_memory_per_block_optin_bytes':101376}}
        return source, ir

    def test_missing_half_b(self):
        source, ir = self.case(3)
        fixed, changes = repair_known_cuda_defects(source, ir)
        self.assertEqual(fixed.count('for (int v = tid; v < b_vec_count; v += num_threads)'), 2)
        self.assertTrue(changes)
        self.assertFalse(preservation_defects(source, fixed, {}))
        self.assertEqual(repair_known_cuda_defects(fixed, ir)[0], fixed)
        from SCOPE.verification.memory_access_plan import vector_mapping_is_proven
        self.assertTrue(vector_mapping_is_proven(fixed))
        bad = fixed.replace('bkIdx * BK + ak;', 'bkIdx * BK + ak + 1;')
        self.assertFalse(vector_mapping_is_proven(bad))

    def test_optin_preserves_storage_and_load_width(self):
        source, ir = self.case(1)
        fixed, changes = repair_known_cuda_defects(source, ir)
        self.assertIn('cudaFuncAttributeMaxDynamicSharedMemorySize, 65536', fixed)
        self.assertIn('load_a_smem_k = (tid % (BK / 4)) * 4', fixed)
        self.assertFalse(preservation_defects(source, fixed, {}))
        self.assertEqual(ir['memory']['shared_memory_bytes'], 65536)
        self.assertEqual(repair_known_cuda_defects(fixed, ir)[0], fixed)
        overlapping = fixed.replace('scope_shared + 2 * BK * BM', 'scope_shared + BK * BM')
        self.assertTrue(preservation_defects(source, overlapping, {}))
        from SCOPE.app import check_terminal_code_completeness
        ir['memory']['use_shared_memory'] = True
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / 'cuda_kernel.cuh').write_text(fixed)
            report = check_terminal_code_completeness(Path(directory), ir)
        check = next(r for r in report['results'] if r['id'] == 'HAS_SHARED_A_B')
        self.assertEqual(check['status'], 'pass')

    def test_no_unprobed_optin(self):
        source, ir = self.case(1)
        del ir['hardware']['max_shared_memory_per_block_optin_bytes']
        fixed, _ = repair_known_cuda_defects(source, ir)
        self.assertNotIn('extern __shared__', fixed)
