import unittest

from SCOPE.verification.initial_shared_sync import inspect_initial_shared_sync, repair_initial_shared_sync
from SCOPE.verification.load_layout_trials import transform, TRIALS
from SCOPE.verification.load_transform import Expressions, fixed_load_iterations, vector_shared_b


def kernel_source():
    tile = dict(BM=64, BN=64, BK=32, WM=32, WN=32, TM=4, TN=4, WMITER=16, WNITER=32)
    groups = []
    for stage in ('stage', 'write_stage'):
        for array, width in (('As', 'BK'), ('Bs', 'BN')):
            scatter = '\n'.join(f'{array}[{stage}][k][n + {i}] = tmp.{c};' for i, c in enumerate('xyzw'))
            groups.append(f'''#pragma unroll
for (int v = linear_tid; v < {'a' if array == 'As' else 'b'}_vectors; v += threads) {{
    const int k = v / ({width} / vector_width);
    const int n_vec = v % ({width} / vector_width);
    const int n = n_vec * vector_width;
    float4 tmp = FLOAT4(B[0]);
    {scatter}
}}''')
    return ('''__global__ void gemm(float* A, float* B, float* C) {
__shared__ float As[2][BK][BM + 1];
__shared__ float Bs[2][BK][BN + 1];
const int tid = threadIdx.x;
const int linear_tid = tid;
const int threads = blockDim.x;
const int vector_width = 4;
const int a_vectors = (BM * BK) / vector_width;
const int b_vectors = (BK * BN) / vector_width;
int stage = 0, write_stage = 1;
/* GLOBAL_TO_SHARED_LOAD_BEGIN */
''' + '\n'.join(groups[:2]) + '\n/* GLOBAL_TO_SHARED_LOAD_END */\n/* MAIN_LOOP_BEGIN */\n'
            + '\n'.join(groups[2:]) + '''
float x = As[stage][0][0];
float y = Bs[stage][0][0];
__syncthreads();
}
void cuda_gemm(float* A, float* B, float* C) {
/* LAUNCH_CONFIG_BEGIN */
''' + '\n'.join(f'static const int {key} = {value};' for key, value in tile.items()) + '''
/* LAUNCH_CONFIG_END */
if (reinterpret_cast<uintptr_t>(A) % 16 != 0) return;
dim3 block((BM * BN) / (WM * WN) * 32);
dim3 grid(1);
gemm<<<grid, block>>>(A, B, C);
}''')


class ScopedLoadTransformTests(unittest.TestCase):
    def test_actual_style_supports_all_trials_and_preserves_anchors(self):
        for name in TRIALS:
            with self.subTest(name=name):
                code, _ = transform(kernel_source(), name)
                self.assertIn('/* MAIN_LOOP_BEGIN */', code)
                self.assertIn('reinterpret_cast<uintptr_t>(A)', code)
                if 'vector_b' in name:
                    self.assertIn('__shared__ __align__(16) float Bs', code)
                    self.assertEqual(code.count(']) = tmp;'), 2)
                if name.endswith('_fixed') or name == 'fixed_loads':
                    self.assertEqual(code.count('const int v = linear_tid + scope_fixed_load_offset;'), 4)

    def test_scopes_do_not_leak_local_aliases(self):
        source = 'const int n = 4; { const int n = 8; use(n); } use(n);'
        expressions = Expressions(source)
        self.assertEqual(expressions.value('n', source.index('use')), 8)
        self.assertEqual(expressions.value('n', source.rindex('use')), 4)

    def test_alias_and_loop_variable_rename_are_supported(self):
        code = kernel_source().replace('n_vec * vector_width', '(v % (BN / 4)) * 4')
        # Only test B vector ownership here; A has a different width.
        code, _ = transform(code, 'none')
        import re
        code = re.sub(r'\bv\b', 'vector_id', code)
        self.assertEqual(vector_shared_b(code).count(']) = tmp;'), 2)

    def test_reject_misaligned_store(self):
        code, _ = transform(kernel_source(), 'none')
        with self.assertRaisesRegex(ValueError, 'alignment'):
            vector_shared_b(code.replace('n_vec * vector_width;', 'n_vec * vector_width + 1;'))

    def test_reject_bad_stride_incomplete_tasks_and_mutation(self):
        for old, new in [('threads = blockDim.x', 'threads = 64'),
                         ('(BM * BK) / vector_width', '(BM * BK) / vector_width - 1'),
                         ('float4 tmp =', 'v += 1; float4 tmp =')]:
            with self.subTest(new=new), self.assertRaises(ValueError):
                fixed_load_iterations(kernel_source().replace(old, new))

    def test_shared_pointer_cast_still_rejected(self):
        code = kernel_source().replace('float x =', 'auto p = reinterpret_cast<float4*>(As); float x =')
        with self.assertRaises(ValueError):
            transform(code, 'none')

    def test_missing_initial_barrier_and_idempotent_repair(self):
        code = kernel_source()
        self.assertEqual(inspect_initial_shared_sync(code)['status'], 'fail')
        repaired, changes = repair_initial_shared_sync(code)
        self.assertEqual(len(changes), 1)
        self.assertLess(repaired.index('__syncthreads'), repaired.index('float x ='))
        self.assertEqual(inspect_initial_shared_sync(repaired)['status'], 'pass')
        self.assertEqual(repair_initial_shared_sync(repaired), (repaired, []))

    def test_existing_initial_barrier_preserved(self):
        code = kernel_source().replace('/* MAIN_LOOP_BEGIN */', '__syncthreads();\n/* MAIN_LOOP_BEGIN */')
        self.assertEqual(repair_initial_shared_sync(code), (code, []))

    def test_divergent_return_and_unknown_protocol_not_repaired(self):
        code = kernel_source().replace('/* MAIN_LOOP_BEGIN */', 'if (tid == 0) return;\n/* MAIN_LOOP_BEGIN */')
        duplicate = kernel_source() + '\n/* MAIN_LOOP_BEGIN */'
        conditional = kernel_source().replace('/* MAIN_LOOP_BEGIN */', 'if (tid == 0)\n/* MAIN_LOOP_BEGIN */')
        for source in (code, '__global__ void other() {}', duplicate, conditional):
            self.assertFalse(inspect_initial_shared_sync(source)['detail']['recognized'])
            self.assertEqual(repair_initial_shared_sync(source), (source, []))

    def test_production_repair_includes_initial_barrier(self):
        from SCOPE.verification.targeted_cuda_repair import repair_known_cuda_defects
        code, changes = repair_known_cuda_defects(kernel_source(), {})
        self.assertTrue(any(c['id'] == 'GEMM_INITIAL_SHARED_PRODUCER_BARRIER' for c in changes))
        self.assertEqual(inspect_initial_shared_sync(code)['status'], 'pass')

    def test_semantic_checker_reports_initial_barrier_defect(self):
        import tempfile
        from pathlib import Path
        from SCOPE.verification.gemm_semantic_checker import check_gemm_semantic_obligations
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, 'cuda_kernel.cuh').write_text(kernel_source(), encoding='utf-8')
            report = check_gemm_semantic_obligations(Path(folder), {})
        found = [r for r in report['results'] if r['id'] == 'GEMM_INITIAL_SHARED_PRODUCER_BARRIER']
        self.assertEqual(found[0]['status'], 'fail')

    def test_terminal_preparation_preserves_checkpoint_and_is_idempotent(self):
        import tempfile
        import json
        from pathlib import Path
        from SCOPE.verification.initial_shared_sync import prepare_initial_shared_sync
        source = kernel_source()
        with tempfile.TemporaryDirectory() as folder:
            kernel = Path(folder, 'cuda_kernel.cuh')
            kernel.write_text(source, encoding='utf-8')
            report = prepare_initial_shared_sync(folder)
            checkpoint = json.loads(Path(report['checkpoint']).read_text(encoding='utf-8'))
            self.assertEqual(checkpoint['source_before'], source)
            self.assertIn('Publish the initial cooperative tile.', kernel.read_text(encoding='utf-8'))
            self.assertIsNone(prepare_initial_shared_sync(folder))
            self.assertEqual(len(list(Path(folder).glob('initial_shared_sync_repair.*.json'))), 1)

    def test_terminal_compiles_repaired_source_without_reselecting(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from SCOPE.app import verify_terminal_chain_once
        class ReachedCompiler(Exception):
            pass
        ir = {'strategy': {'applied_strategy_ids': ['Tiling.BlockTile.64x64x32']}}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'cuda_kernel.cuh'
            source.write_text(kernel_source(), encoding='utf-8')
            def compiler(**kwargs):
                self.assertEqual(inspect_initial_shared_sync(source.read_text(encoding='utf-8'))['status'], 'pass')
                self.assertEqual(kwargs['ir']['strategy'], ir['strategy'])
                self.assertIn('initial_shared_sync_repair', kwargs['ir'])
                raise ReachedCompiler()
            with patch('SCOPE.app.check_after_codegen', return_value={}), \
                 patch('SCOPE.app.extract_code_ast', return_value={}), \
                 patch('SCOPE.app.check_terminal_code_completeness', return_value={}), \
                 patch('SCOPE.verification.memory_access_plan.enforce_memory_access_plan', return_value={}), \
                 patch('SCOPE.verification.cooperative_load_repair.diagnose_cooperative_loads', return_value={}), \
                 patch('SCOPE.app.verify_build_and_run', side_effect=compiler):
                with self.assertRaises(ReachedCompiler):
                    verify_terminal_chain_once(ir, root, 'test', {}, 1)
        self.assertNotIn('initial_shared_sync_repair', ir)

    def test_retile_rebuilds_launch_bounds_and_new_task_counts(self):
        from SCOPE.compare_tile_loads import retile
        code = kernel_source().replace('__global__ void', 'template<int BM> __global__ __launch_bounds__(128, 2) void')
        code, tile = retile(code, [512] * 3, [64, 128, 16])
        self.assertEqual(tile['threads_per_block'], 256)
        self.assertIn('__launch_bounds__(256, 2)', code)
        code, _ = transform(code, 'none_vector_b_fixed')
        self.assertIn('scope_fixed_load_offset < 256;', code)
        self.assertIn('scope_fixed_load_offset < 512;', code)
        with self.assertRaisesRegex(ValueError, 'Retile the unfixed'):
            retile(code, [512] * 3, [64, 64, 32])

    def test_missing_seed_does_not_discard_measurements(self):
        from SCOPE.compare_load_layout import finalize_report
        from unittest.mock import patch
        report = {'source': 'missing', 'results': []}
        with patch('SCOPE.compare_load_layout.read_sources', side_effect=ValueError('Source missing')):
            finalize_report(report, {})
        self.assertFalse(report['inputs_unchanged'])
        self.assertFalse(report['comparison_valid'])
        self.assertEqual(report['source_recheck_error'], 'Source missing')


if __name__ == '__main__':
    unittest.main()
