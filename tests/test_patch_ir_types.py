import copy
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from SCOPE.generate_ir.ir_types import normalize_ir_types
from SCOPE.generate_ir.ir_checker import (
    check_vector_alignment, check_vector_tail_handling, evaluate_semantic_constraint,
)
from SCOPE.llm.patch_generator import build_patch_ir, validate_patch_response


class PatchIRTypeTests(unittest.TestCase):
    def test_cpu_patch_cannot_override_previous_strategy_state(self):
        from SCOPE.app import merge_patch_ir_updates_for_backend

        updates, discarded = merge_patch_ir_updates_for_backend(
            base_ir={"target": {"backend": "cpu"}, "cpu_memory": {"pack_b": False}},
            strategy={
                "strategy_id": "CPU.MicroKernel.AVX2.FMA.4x8",
                "ir_updates": {
                    "cpu_microkernel.family": "avx2_fma",
                    "cpu_microkernel.mr": 4,
                    "cpu_microkernel.nr": 8,
                },
            },
            proposed_updates={
                "cpu_microkernel.family": "invented_value",
                "cpu_memory.pack_b": True,
                "cpu_memory.pack_buffer_scope": "l2",
            },
        )

        self.assertEqual(updates["cpu_microkernel.family"], "avx2_fma")
        self.assertNotIn("cpu_memory.pack_b", updates)
        self.assertEqual(discarded["cpu_memory.pack_b"], True)
        self.assertEqual(discarded["cpu_memory.pack_buffer_scope"], "l2")

    def test_schema_type_placeholders_do_not_override_catalog_values(self):
        from SCOPE.app import merge_expected_ir_updates

        updates = merge_expected_ir_updates(
            "Pipeline.NoAsyncCopy.V1",
            {"resource.shared_memory.pipeline_multiplier": 1},
            {
                "resource.shared_memory.pipeline_multiplier": "integer",
                "pipeline.async_copy": "boolean",
            },
        )

        self.assertEqual(updates["resource.shared_memory.pipeline_multiplier"], 1)
        self.assertIs(updates["pipeline.async_copy"], False)

    def test_numeric_strings_normalized_at_patch_boundary_without_changing_parent(self):
        parent = {'vectorization': {'A': {'vector_width': 1}}}
        original = copy.deepcopy(parent)
        response = validate_patch_response({
            'strategy_id': 'Vectorization.GlobalLoadAB.float4',
            'ir_updates': {'vectorization.A.vector_width': '4',
                           'vectorization.B': {'vector_width': '4', 'alignment_proven': 'true'}},
        }, 'Vectorization.GlobalLoadAB.float4')
        ir = build_patch_ir(parent, {'strategy_id': response['strategy_id']}, {}, response)
        self.assertEqual(parent, original)
        self.assertIs(type(ir['vectorization']['A']['vector_width']), int)
        self.assertEqual(ir['vectorization']['B']['vector_width'], 4)
        self.assertIs(ir['vectorization']['B']['alignment_proven'], True)
        self.assertEqual(check_vector_alignment(ir).status, 'fail')  # A still needs a proof.

    def test_declared_booleans_and_numbers_only(self):
        source = {'vectorization': {'enabled': 'false'},
                  'tiling': {'block_m': '64', 'thread_m': 4.0},
                  'formula': 'BM / WM', 'code': 'float4 a;'}
        result = normalize_ir_types(source)
        self.assertIs(result['vectorization']['enabled'], False)
        self.assertEqual(result['tiling'], {'block_m': 64, 'thread_m': 4})
        self.assertEqual(result['formula'], source['formula'])
        self.assertEqual(result['code'], source['code'])
        self.assertEqual(source['tiling']['block_m'], '64')

    def test_invalid_ir_rejected_not_silently_scalarized(self):
        for value in ('float4', 'BK / 4', 3.5, True, float('inf'), []):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'vectorization.A.vector_width'):
                validate_patch_response({'ir_updates': {'vectorization.A.vector_width': value}}, 'S')
        for value in ([], 'invalid', None):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'must be an object'):
                validate_patch_response({'ir_updates': value}, 'S')

    def test_merged_catalog_updates_also_validated(self):
        with self.assertRaisesRegex(ValueError, 'mapping.threads_per_block'):
            build_patch_ir({}, {}, {}, {'ir_updates': {'mapping.threads_per_block': 'many'}})

    def test_standalone_checks_handle_old_string_ir(self):
        ir = {'vectorization': {'A': {'vector_width': '4', 'alignment_guard': 'false',
                                      'alignment_proven': 'false', 'tail_handling': 'true'}}}
        self.assertEqual(check_vector_alignment(ir).status, 'fail')
        self.assertEqual(check_vector_tail_handling(ir).status, 'pass')
        predicate = {'constraint_id': 'C_VECTOR_TAIL_HANDLING', 'verifier': 'ir_eval',
                     'args': {'tensors': ['A']}}
        self.assertEqual(evaluate_semantic_constraint(ir, predicate)['status'], 'pass')
        ir['vectorization']['A']['vector_width'] = 'invalid'
        self.assertEqual(check_vector_alignment(ir).id, 'IR_FIELD_TYPE')
        self.assertEqual(evaluate_semantic_constraint(ir, predicate)['failure_type'], 'IR.InvalidFieldType')

    def test_invalid_merged_ir_stays_in_bounded_candidate_repair(self):
        from SCOPE import app
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(app, 'chain_code_path', return_value=Path(directory)), \
                patch.object(app, 'load_strategy_scoped_code_context', return_value={}), \
                patch.object(app, 'save_candidate_json'), \
                patch.object(app, 'generate_patch_with_llm', return_value={'ir_updates': {}}) as generate, \
                patch.object(app, 'generate_code_files_from_patch_with_llm') as codegen:
            result = app.run_strategy_candidate(stage='PerformanceUnlock', base_ir={},
                base_source_snapshot={'cuda_kernel.cuh': '// preserved parent'},
                strategy={'strategy_id': 'Test.InvalidIR', 'ir_updates': {'vectorization.A.vector_width': 'float4'}},
                precheck_item={}, client=object())
        self.assertFalse(result['accepted'])
        self.assertEqual(len(result['attempts']), app.DEFAULT_MAX_REPAIR_ATTEMPTS)
        self.assertEqual(generate.call_count, app.DEFAULT_MAX_REPAIR_ATTEMPTS)
        codegen.assert_not_called()
        self.assertEqual(result['source_snapshot']['cuda_kernel.cuh'], '// preserved parent')


if __name__ == '__main__':
    unittest.main()
