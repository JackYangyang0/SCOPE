import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from SCOPE.app import record_unlock_feedback
from SCOPE.verification.cuda_build_options import cuda_optimization_flags
from SCOPE.verification.implementation_identity import normalized_source, unchanged_verified_parent
from SCOPE.verification.strategy_realization_oracle import verify_strategy_realization
from SCOPE.verification.build_run_verifier import compile_gemm, update_compile_result


class EffectiveImplementationTests(unittest.TestCase):
    def ir(self):
        return {'problem': {'M': 512, 'N': 512, 'K': 512},
                'verification': {'accepted': True, 'compile': {'status': 'pass',
                    'build_platform': 'linux', 'effective_optimization_flags': []}}}

    def test_comment_only_change_is_duplicate(self):
        ir = self.ir()
        self.assertTrue(unchanged_verified_parent(ir, {'cuda_kernel.cuh': '  float x; // old\n'},
                       copy.deepcopy(ir), {'cuda_kernel.cuh': 'float x; /* new */\n'}, 'linux'))

    def test_flag_change_is_not_duplicate(self):
        ir = self.ir()
        changed = copy.deepcopy(ir)
        changed['compiler'] = {'use_fast_math': True}
        self.assertIsNone(unchanged_verified_parent(ir, {'a.cuh': 'x;'}, changed, {'a.cuh': 'x;'}, 'linux'))

    def test_changed_code_shape_or_platform_not_duplicate(self):
        ir = self.ir()
        changed = copy.deepcopy(ir)
        changed['problem']['M'] = 1024
        self.assertIsNone(unchanged_verified_parent(ir, {'a.cuh': 'x;'}, changed, {'a.cuh': 'x;'}, 'linux'))
        self.assertIsNone(unchanged_verified_parent(ir, {'a.cuh': 'x;'}, ir, {'a.cuh': 'y;'}, 'linux'))
        self.assertIsNone(unchanged_verified_parent(ir, {'a.cuh': 'x;'}, ir, {'a.cuh': 'x;'}, 'windows'))

    def test_unknown_flags_and_failed_parent_are_not_reused(self):
        for field in ('accepted', 'flags'):
            ir = self.ir()
            if field == 'accepted':
                ir['verification']['accepted'] = False
            else:
                ir['verification']['compile'].pop('effective_optimization_flags')
            self.assertIsNone(unchanged_verified_parent(ir, {'a.cuh': 'x;'}, ir, {'a.cuh': 'x;'}, 'linux'))

    def test_strings_preserved_and_special_preprocessing_skipped(self):
        self.assertNotEqual(normalized_source('puts("//one");'), normalized_source('puts("//two");'))
        self.assertIsNone(normalized_source('int x = __LINE__;'))
        self.assertIsNone(normalized_source('R"(a\nb)"'))

    def realization(self, source, sid, flags):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, 'cuda_kernel.cuh').write_text(source, encoding='utf-8')
            ir = {'verification': {'compile': {'effective_optimization_flags': flags,
                    'stderr': 'label --use_fast_math is NOT authoritative'}}}
            return verify_strategy_realization(Path(directory), [sid], ir)['strategy_reports'][0]

    def test_fastmath_uses_effective_flags_not_label_or_stderr(self):
        sid = 'Compiler.FastMath.Enabled'
        for flags, expected in [(None, 'unknown'), ([], 'not_realized'), (['--use_fast_math'], 'realized')]:
            self.assertEqual(self.realization('// FastMath', sid, flags)['status'], expected)

    def test_noguard_rejects_guarded_output(self):
        source = '''/* STORE_BEGIN */
        if (global_m < M && global_n + 1 < N) {
            *reinterpret_cast<float2*>(&C[index]) = out;
        }
        /* STORE_END */'''
        self.assertEqual(self.realization(source, 'Vectorization.StoreC.AlignedNoGuard', [])['status'], 'not_realized')

    def test_noguard_without_region_stays_unknown(self):
        self.assertEqual(self.realization('FLOAT4(C[i]) = x;', 'Vectorization.StoreC.AlignedNoGuard', [])['status'], 'unknown')

    def test_flags_are_validated(self):
        self.assertEqual(cuda_optimization_flags({'compiler': {'use_fast_math': True, 'max_register_count': 64}}),
                         ['--use_fast_math', '--maxrregcount=64'])
        with self.assertRaises(ValueError):
            cuda_optimization_flags({'compiler': {'use_fast_math': 'false'}})

    def test_compile_records_real_flags(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'main.cpp').write_text('int main() {}')
            with patch('SCOPE.verification.build_run_verifier.shutil.which', return_value='nvcc'), \
                 patch('SCOPE.verification.build_run_verifier.resolve_supported_cuda_arch', return_value='sm_89'), \
                 patch('SCOPE.verification.build_run_verifier.run_command', return_value={
                     'status': 'pass', 'command': ['nvcc'], 'stdout': '', 'stderr': '', 'returncode': 0}) as run:
                result = compile_gemm({'compiler': {'use_fast_math': True}}, root, root/'gemm', 10, build_platform='linux')
            self.assertIn('--use_fast_math', run.call_args.args[0])
            self.assertIn('--use_fast_math', result['nvcc_command'])
            ir = {}
            update_compile_result(ir, result)
            self.assertEqual(ir['verification']['compile']['effective_optimization_flags'], ['--use_fast_math'])

    def test_noop_feedback_not_counted_as_runtime_failure(self):
        state = {'history': {}, 'current_ir': {}}
        record_unlock_feedback(state, {'strategy_ids': ['hint']},
                               {'summary': {'status': 'unchanged_implementation'}}, 1)
        self.assertEqual(state['history']['ineffective_unlock_bundles'], ['hint'])
        self.assertFalse(state['history'].get('failed_strategy_counts'))


if __name__ == '__main__':
    unittest.main()
