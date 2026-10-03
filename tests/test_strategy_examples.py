import copy
import json
import unittest
from pathlib import Path

from SCOPE.llm.strategy_examples import load_example_catalog, strategy_with_examples
from SCOPE.llm.patch_generator import load_strategy, build_patch_messages
from SCOPE.llm.concrete_code_generator import build_patch_to_code_messages


ROOT = Path(__file__).resolve().parents[1]


class StrategyExampleTests(unittest.TestCase):
    def test_library_bindings_and_example_contracts(self):
        from SCOPE.data.lib.build_strategy_examples import nodes
        entries = [entry for name in ('strategy_library.json', 'cpu_strategy_library.json')
                   for entry in nodes(json.loads((ROOT/'data/lib'/name).read_text(encoding='utf-8')))]
        library = {'strategies': entries}
        catalog = load_example_catalog()
        self.assertEqual(set(catalog['strategy_bindings']), {e['strategy_id'] for e in entries})
        self.assertEqual(len(entries), 189)
        for sid, ids in catalog['strategy_bindings'].items():
            entry = load_strategy(library, sid)
            self.assertEqual(entry['implementation_example_ids'], ids)
            for eid in ids:
                example = catalog['examples'][eid]
                self.assertEqual(example['example_id'], eid)
                for field in ('applicability', 'adaptation_rules', 'code_lines',
                              'verification_obligations', 'validation_status'):
                    self.assertTrue(example[field])

    def test_only_selected_examples_and_no_mutation(self):
        strategy = {'strategy_id': 'Vectorization.GlobalLoadAB.float4'}
        before = copy.deepcopy(strategy)
        result = strategy_with_examples(strategy)
        self.assertEqual(strategy, before)
        self.assertEqual([e['example_id'] for e in result['implementation_examples']],
                         ['fp32_vector_tile_load_v1'])
        self.assertNotIn('cp.async', str(result))

    def test_repair_and_batch_deduplicate(self):
        strategy = {'strategy_id': 'Repair.PreserveAppliedStrategies', 'applied_strategies': [
            {'strategy_id': 'Vectorization.GlobalLoadAB.float4'},
            {'strategy_id': 'Reordering.CooperativeVectorLoadAB.float4'},
            {'strategy_id': 'Pipeline.DoubleBuffer.SharedAB'}]}
        self.assertEqual(len(strategy_with_examples(strategy)['implementation_examples']), 2)

    def test_cpu_and_deterministic_tiling_have_references(self):
        for sid in ('CPU.LoopOrder.IJK', 'Tiling.BlockTile.128x128x16'):
            strategy = {'strategy_id': sid}
            result = strategy_with_examples(strategy)
            self.assertEqual(len(result['implementation_examples']), 1)
        tile = strategy_with_examples({'strategy_id': 'Tiling.BlockTile.128x128x16'})
        self.assertIn('executor applies', str(tile))

    def test_build_is_reproducible(self):
        from SCOPE.data.lib.build_strategy_examples import build
        libraries, generated = build()
        self.assertEqual(generated, json.loads((ROOT/'data/lib/strategy_examples_full.json').read_text(encoding='utf-8')))
        for name, value in libraries.items():
            self.assertEqual(value, json.loads((ROOT/'data/lib'/name).read_text(encoding='utf-8')))

    def test_cpu_prompt_retains_example_after_compaction(self):
        from SCOPE.llm.c_code_generator import build_cpu_c_patch_to_code_messages, build_cpu_c_code_messages
        sid = 'CPU.MicroKernel.AVX2.FMA.4x8'
        messages = build_cpu_c_patch_to_code_messages({}, {}, {'strategy_id': sid}, {})
        self.assertIn('_mm256_fmadd_ps', messages[-1]['content'])
        self.assertNotIn('cp.async', messages[-1]['content'])
        messages = build_cpu_c_code_messages({'strategy':{'applied_strategy_ids':[sid]}}, {})
        self.assertIn('_mm256_fmadd_ps', messages[-1]['content'])

    def test_all_strategy_prompts_format_without_losing_examples(self):
        from SCOPE.llm.c_code_generator import build_cpu_c_patch_to_code_messages
        for sid, refs in load_example_catalog()['strategy_bindings'].items():
            strategy = {'strategy_id': sid}
            build_patch_messages({}, strategy, {}, {})
            builder = build_cpu_c_patch_to_code_messages if sid.startswith('CPU.') else build_patch_to_code_messages
            messages = builder({}, {}, strategy, {})
            for ref in refs:
                self.assertIn(ref, messages[-1]['content'])

    def test_bad_reference_rejected(self):
        with self.assertRaises(ValueError):
            strategy_with_examples({'strategy_id': 'GPU.X', 'implementation_example_ids': ['missing']})

    def test_patch_and_code_prompts_actually_contain_example(self):
        strategy = {'strategy_id': 'Vectorization.GlobalLoadAB.float4'}
        context = {'files': {'cuda_kernel.cuh': '// actual parent kernel'}}
        patch_messages = build_patch_messages({}, strategy, {}, context)
        code_messages = build_patch_to_code_messages({}, {}, strategy, context)
        for messages in (patch_messages, code_messages):
            text = messages[-1]['content']
            self.assertIn('fp32_vector_tile_load_v1', text)
            self.assertIn('v += threads', text)
            self.assertIn('actual parent kernel', text)
            self.assertIn('Locked strategies', text)

    def test_both_locked_repair_prompts_receive_examples(self):
        strategy = {'strategy_id': 'Repair.PreserveAppliedStrategies', 'applied_strategies': [
            {'strategy_id': 'Pipeline.DoubleBuffer.SharedAB'}]}
        for name in ('repair_locked_cuda_prompt.txt', 'repair_locked_cuda_full_prompt.txt'):
            messages = build_patch_to_code_messages({}, {}, strategy, {},
                prompt_path=ROOT/'llm/prompts'/name)
            self.assertIn('sync_double_buffer_v1', messages[-1]['content'])
            self.assertIn('not proof of memory/compute overlap', messages[-1]['content'])
