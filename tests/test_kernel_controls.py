import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
from SCOPE.specialization.kernel_controls import hoist_a_fragment, store_variant, protect_pingpong_reuse


class KernelControlTests(unittest.TestCase):
    def test_prepare_only_preserves_sources_and_never_benchmarks(self):
        from SCOPE import compare_kernel_controls as command
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source'
            source.mkdir()
            original = '// immutable source'
            (source / 'cuda_kernel.cuh').write_text(original)
            (source / 'main.cpp').write_text('// fixed harness')
            (source / 'verified_ir.json').write_text(json.dumps({'compiler': {'max_register_count': 96}}))
            output = root / 'comparison'
            argv = ['controls', '--source', str(source), '--output', str(output),
                    '--matrix-size', '512', '512', '512', '--prepare-only',
                    '--ir', str(source / 'verified_ir.json')]
            with patch('sys.argv', argv), patch.object(command, 'tune_sources') as benchmark, \
                    patch.object(command, 'build_extracted_ir', return_value={'problem': {}}):
                command.main()
            benchmark.assert_not_called()
            self.assertEqual((source / 'cuda_kernel.cuh').read_text(), original)
            report = json.loads((output / 'comparison.json').read_text())
            self.assertFalse(report['comparison_valid'])
            self.assertEqual(report['ranking'], [])
            self.assertTrue(all(r['status'] == 'prepared_not_measured' for r in report['results']))
            frozen_ir = json.loads((output / 'frozen_benchmark_ir.json').read_text())
            self.assertEqual(frozen_ir['compiler']['max_register_count'], 96)
            self.assertEqual((output / 'baseline_repeat/cuda_kernel.cuh').read_text(), original)

    def test_reuse_barrier_is_idempotent_and_rejects_divergent_exit(self):
        source = '''for (int bkIdx = 1; bkIdx < k_tiles; ++bkIdx) {
const int comp_flag = (bkIdx - 1) & 1;
const int mem_flag = bkIdx & 1;
As[mem_flag][k][m] = a; Bs[mem_flag][k][n] = b;
__syncthreads();
a = As[comp_flag][k][m]; b = Bs[comp_flag][k][n];
}'''
        result = protect_pingpong_reuse(source)
        self.assertEqual(result.count('__syncthreads();'), 2)
        self.assertEqual(protect_pingpong_reuse(result), result)
        with self.assertRaises(ValueError):
            protect_pingpong_reuse(source.replace('a = As', 'if (tid) return; a = As'))

    def test_hoist_requires_two_recognized_reductions(self):
        load = '''#pragma unroll
for (int wn = 0; wn < WN / WNITER; ++wn) {
#pragma unroll
for (int i = 0; i < TM; ++i) {
regM[i] = As[comp_flag][k][Wrow * WM + wm * WMITER + Trow * TM + i];
}'''
        with self.assertRaises(ValueError):
            hoist_a_fragment(load)
        code = hoist_a_fragment(load + '\n}\n' + load + '\n}')
        self.assertEqual(code.count('regM[i] = As'), 2)
        self.assertLess(code.index('regM[i] = As'), code.index('for (int wn'))
        with self.assertRaises(ValueError):
            hoist_a_fragment(load.replace('wm * WMITER', 'wn * WMITER') * 2)
        grouped = (load + '\n}\n' + load + '\n}').replace('[k]', '[k + u]')
        self.assertLess(hoist_a_fragment(grouped).index('regM[i] = As'),
                        hoist_a_fragment(grouped).index('for (int wn'))

    def test_store_preserves_other_regions_and_has_guarded_tail(self):
        source = '''prefix
/* LAUNCH_CONFIG_BEGIN */
static const int TN = 4;
/* LAUNCH_CONFIG_END */
/*
 * STORE_BEGIN
 */
results[m + wm * TM][n + wn * TN];
/*
 * STORE_END
 */
suffix'''
        result = store_variant(source, 4)
        self.assertEqual(result.split('STORE_BEGIN')[0], source.split('STORE_BEGIN')[0])
        self.assertEqual(result.split('STORE_END')[1], source.split('STORE_END')[1])
        self.assertIn('& 15ULL', result)
        self.assertIn('gn + u < N', result)
        self.assertIn('beta * C[', result)
        self.assertIn('if (beta != 0.0f)', result)
        self.assertIn('beta == 0.0f ? value', result)
        equivalent = source.replace('results[m + wm * TM][n + wn * TN]',
                                    'results[wm * TM + m][wn * TN + n]')
        self.assertEqual(store_variant(equivalent, 4), result)
        with self.assertRaises(ValueError):
            store_variant(source.replace('TN = 4', 'TN = 2'), 4)
