import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from SCOPE.llm.concrete_code_generator import build_patch_to_code_messages, generate_code_files_from_patch_with_llm
from SCOPE.llm.implementation_context import (
    compact_verifier_feedback, cuda_generation_context, is_harness_file, kernel_implementation_summary,
)
from SCOPE.llm.patch_generator import build_patch_messages, generate_patch_with_llm
from SCOPE.llm.strategy_examples import strategy_with_examples
from SCOPE.llm.unlock_category_selector import select_category_plan


ROOT = Path(__file__).resolve().parents[1]
SOURCE = """
// Current parent helper outside anchors must remain visible.
__device__ float helper(float x) { return x; }
__shared__ __align__(16) float As[2][BK][BM];
__shared__ float Bs[2][BK][BN];
float4 av = FLOAT4(A[a]);
float4 bv = *reinterpret_cast<const float4*>(&B[b]);
As[stage][k][m] = av.x;
FLOAT4(Bs[stage][k][n]) = bv;
__launch_bounds__(256, 2)
"""


class ImplementationContextTests(unittest.TestCase):
    def test_widths_and_staging_are_independent(self):
        result = kernel_implementation_summary(SOURCE)
        self.assertEqual(result['global_load_widths_observed'], {'A': [4], 'B': [4]})
        self.assertEqual(result['shared_float4_store_targets_observed'], ['Bs'])
        self.assertEqual(result['staged_shared_arrays_observed'], {'As': 2, 'Bs': 2})
        self.assertTrue(result['double_buffer_storage_observed'])
        self.assertIsNone(result['async_copy_syntax_observed'])
        self.assertTrue(result['overlap_status'].startswith('unproven'))

    def test_declarations_comments_and_names_do_not_prove_realization(self):
        source = '// float4 av = FLOAT4(A[0]);\nfloat4 *Ap; float double_buffer; float x=As[2];'
        result = kernel_implementation_summary(source)
        self.assertEqual(result['global_load_widths_observed'], {'A': None, 'B': None})
        self.assertIsNone(result['double_buffer_storage_observed'])
        self.assertEqual(result['shared_float4_store_targets_observed'], [])

    def test_stage_constants_and_unknown_aliases(self):
        result = kernel_implementation_summary('constexpr int STAGES=3; __shared__ float s[STAGES][BK][BN];')
        self.assertEqual(result['staged_shared_arrays_observed'], {'s': 3})
        unknown = kernel_implementation_summary('__shared__ float s[DEPTH][BK][BN];')
        self.assertIsNone(unknown['double_buffer_storage_observed'])

    def test_source_fingerprint_changes(self):
        self.assertNotEqual(kernel_implementation_summary(SOURCE)['source_sha256'],
                            kernel_implementation_summary(SOURCE + '\nint new_step;')['source_sha256'])

    def test_current_source_preserved_and_harness_excluded_without_mutation(self):
        context = {'files': {'cuda_kernel.cuh': SOURCE, 'main.cpp': 'DO_NOT_SEND_HARNESS',
                             'kernel.h': 'void cuda_gemm();'},
                   'patch_anchors': {'main.cpp': ['MAIN'], 'cuda_kernel.cuh': ['STORE']}}
        before = copy.deepcopy(context)
        result = cuda_generation_context({}, {'strategy_id': 'Pipeline.DoubleBuffer.SharedAB'}, context)
        self.assertEqual(context, before)
        self.assertEqual(result['files']['cuda_kernel.cuh'], SOURCE)
        self.assertNotIn('main.cpp', result['files'])
        self.assertNotIn('main.cpp', result['patch_anchors'])
        self.assertNotIn('DO_NOT_SEND_HARNESS', json.dumps(result))

    def test_feedback_is_bounded_measured_and_does_not_invent_data(self):
        ir = {'performance': {'gflops': 9000, 'latency_ms_std': 0.001, 'benchmark_runs': 5,
                              'benchmark_successful_runs': 5, 'gflops_std': float('nan')},
              'verification': {'summary': {'gflops': None, 'compile_status': 'pass'},
                               'compile': {'stdout': 'RAW_BUILD_LOG',
                                           'ptxas_resources': {'register_counts': [81], 'spill_bytes_sum': 0}}},
              'strategy': {'failed_strategy_counts': {'Pipeline.X': 1}, 'changed_regions': ['MAIN_LOOP']},
              'defect_diagnosis': {'defects': [{'defect_type': 'gap', 'evidence': {
                  'message': 'missing proof', 'detail': 'LARGE_UNUSED_DETAIL'}}] * 20}}
        result = compact_verifier_feedback(ir)
        self.assertEqual(result['metrics']['gflops'], 9000)
        self.assertEqual(result['metrics']['benchmark_runs'], 5)
        self.assertNotIn('gflops_std', result['metrics'])
        self.assertEqual(result['compiler_resources']['register_counts'], [81])
        self.assertEqual(result['semantic_defect_count'], 20)
        self.assertEqual(len(result['semantic_defect_summary']), 4)
        self.assertEqual(result['semantic_defect_summary'][0]['kind'], 'static_proof_gap')
        self.assertEqual(result['failed_strategy_counts'], {'Pipeline.X': 1})
        self.assertIn('inherited', result['scope'])
        self.assertNotIn('RAW_BUILD_LOG', str(result))
        self.assertNotIn('LARGE_UNUSED_DETAIL', str(result))
        json.dumps(result, allow_nan=False)

    def test_empty_feedback_is_not_fabricated_pass_or_zero(self):
        result = compact_verifier_feedback({})
        self.assertIsNone(result['compile_status'])
        self.assertEqual(result['metrics'], {})
        self.assertEqual(result['compiler_resources'], {})

    def test_patch_and_materialization_both_receive_same_context_and_policy(self):
        strategy = {'strategy_id': 'Pipeline.DoubleBuffer.SharedAB'}
        ir = {'pipeline': {'stages': 2}, 'compiler': {'hint': 'test'}, 'epilogue': {'beta': 0},
              'performance': {'gflops': 1234.5}}
        context = {'files': {'cuda_kernel.cuh': SOURCE, 'main.cpp': 'PRIVATE_HARNESS'}}
        for messages in (build_patch_messages(ir, strategy, {}, context),
                         build_patch_to_code_messages(ir, {}, strategy, context)):
            text = messages[-1]['content']
            self.assertIn('helper(float x)', text)
            self.assertNotIn('PRIVATE_HARNESS', text)
            self.assertIn('global_load_widths_observed', text)
            self.assertIn('1234.5', text)
            self.assertIn('Strategy labels and source tokens', text)
            self.assertIn('"epilogue"', text)
            self.assertIn('actual implementation change', text)
            self.assertIn('selected fast-path policy', text)
            self.assertNotIn('must keep boundary guards.', text)

    def test_locked_repairs_receive_policy_without_relaxing_output_scope(self):
        strategy = {'strategy_id': 'Repair.PreserveAppliedStrategies', 'applied_strategies': [
            {'strategy_id': 'Reordering.LoadCompute.SeparatePhases'},
            {'strategy_id': 'Epilogue.StoreC.Vectorized.float4'}]}
        for name in ('repair_locked_cuda_prompt.txt', 'repair_locked_cuda_full_prompt.txt'):
            messages = build_patch_to_code_messages({}, {}, strategy, {'files': {'cuda_kernel.cuh': SOURCE}},
                                                    prompt_path=ROOT/'llm/prompts'/name)
            text = messages[-1]['content']
            self.assertIn('not automatically faster', text)
            self.assertIn('retaining locked strategies', text)
            self.assertIn('unless full-kernel locked repair is explicitly requested', text)
            self.assertIn('Do not change', text)
            self.assertIn('verifier_feedback', text)

    def test_only_relevant_family_contracts_no_strategy_mutation(self):
        strategy = {'strategy_id': 'Reordering.LoadCompute.SeparatePhases'}
        before = copy.deepcopy(strategy)
        result = strategy_with_examples(strategy)
        self.assertEqual(strategy, before)
        self.assertIn('without serializing', str(result['coupling_contracts']))
        self.assertNotIn('minimum-block target', str(result['coupling_contracts']))

    def test_cpu_unchanged(self):
        context = {'files': {'cpu_kernel.c': 'CPU_SOURCE', 'main.c': 'CPU_MAIN'}}
        self.assertIs(cuda_generation_context({}, {'strategy_id': 'CPU.LoopOrder.IJK'}, context), context)
        self.assertIs(cuda_generation_context({'target': {'backend': 'cpu'}}, {}, context), context)
        self.assertNotIn('implementation_policy', strategy_with_examples({'strategy_id': 'CPU.LoopOrder.IJK'}))
        self.assertNotIn('implementation_policy', strategy_with_examples({'applied_strategies': [
            {'strategy_id': 'CPU.LoopOrder.IJK'}]}))
        from SCOPE.llm.c_code_generator import build_cpu_c_code_messages
        text = build_cpu_c_code_messages({'strategy': {'applied_strategy_ids': ['CPU.LoopOrder.IJK']}}, {})[-1]['content']
        self.assertNotIn('implementation_policy', text)

    def test_harness_paths_are_portable(self):
        for path in ('main.cpp', r'C:\code\main.cu', '/code/main.c', 'nested/main.cuh'):
            self.assertTrue(is_harness_file(path))
        self.assertFalse(is_harness_file('cuda_kernel.cuh'))

    def test_deterministic_compiler_hint_does_not_call_model(self):
        from SCOPE.verification.compiler_hint_transform import HINTS
        strategy = {'strategy_id': next(iter(HINTS))}
        client = Mock()
        generate_patch_with_llm(client, {}, strategy, {}, {})
        generate_code_files_from_patch_with_llm(client, {}, {}, strategy, {})
        client.complete_json.assert_not_called()
        client.complete_codegen_json.assert_not_called()

    def test_app_summary_passes_observed_storage_and_feedback(self):
        from SCOPE.app import build_unlock_code_summary, build_unlock_verifier_summary
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'cuda_kernel.cuh').write_text(SOURCE, encoding='utf-8')
            (root/'main.cpp').write_text('PRIVATE_HARNESS', encoding='utf-8')
            result = build_unlock_code_summary(root, ['cuda_kernel.cuh', 'main.cpp'])
            self.assertEqual(len(result['files']), 1)
            self.assertTrue(result['files'][0]['contains_double_buffer'])
            self.assertNotIn('helper(float x)', json.dumps(result))
        feedback = build_unlock_verifier_summary({'verified_ir': {'performance': {'gflops_mean': 3000}}})
        self.assertEqual(feedback['measured_feedback']['metrics']['gflops_mean'], 3000)

    def test_category_selection_keeps_single_objective_and_receives_evidence(self):
        client = Mock()
        client.complete_selection_json.return_value = {'selected_strategy_id': None}
        result = select_category_plan(client, {'strategies': [{'strategy_id': 'Pipeline.DoubleBuffer.SharedAB'}]},
                                      {}, {}, code_summary={'implementation_evidence': kernel_implementation_summary(SOURCE)})
        self.assertEqual(result['batches'], [])
        messages = client.complete_selection_json.call_args.args[0]
        self.assertIn('at most ONE', messages[0]['content'])
        self.assertIn('Static instruction counts', messages[0]['content'])
        self.assertIn('staged_shared_arrays_observed', messages[1]['content'])


if __name__ == '__main__':
    unittest.main()
