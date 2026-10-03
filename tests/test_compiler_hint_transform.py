import tempfile
import unittest
from pathlib import Path
from SCOPE.verification.compiler_hint_transform import apply_compiler_hint
from SCOPE.llm.concrete_code_generator import apply_generated_code_files, generate_code_files_from_patch_with_llm
from SCOPE.llm.patch_generator import generate_patch_with_llm
from SCOPE.verification.pipeline_evidence import pipeline_evidence


class CompilerHintTests(unittest.TestCase):
    source = '__global__ void gemm(int M, const float *A, float *B, float *C) { C[0] = A[0] * B[0]; }\nvoid cuda_gemm() { gemm<<<g,b>>>(M,A,B,C); }'

    def test_restrict_changes_only_signature_and_is_idempotent(self):
        code = apply_compiler_hint(self.source, 'Compiler.RestrictPointer')
        self.assertEqual(code.replace('__restrict__ ', ''), self.source)
        self.assertEqual(apply_compiler_hint(code, 'Compiler.RestrictPointer'), code)

    def test_forceinline_never_changes_kernel(self):
        code = self.source + '\n__device__ float helper(float x) { return x; }'
        result = apply_compiler_hint(code, 'Compiler.ForceInline.DeviceFunctions')
        self.assertEqual(result.replace('__forceinline__ ', ''), code)
        self.assertIn('__device__ __forceinline__ float helper', result)
        self.assertIn(self.source, result)

    def test_no_llm_call_and_ignore_unrelated_generated_edits(self):
        strategy = {'strategy_id': 'Compiler.RestrictPointer'}
        patch = generate_patch_with_llm(None, {}, strategy, {}, {})
        result = generate_code_files_from_patch_with_llm(None, {}, patch, strategy, {})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'cuda_kernel.cuh'
            path.write_text(self.source, encoding='utf-8')
            result['files'] = [{'path': 'cuda_kernel.cuh', 'content': 'broken kernel'}]
            status = apply_generated_code_files(result, Path(directory), {}, strategy)
            self.assertEqual(status['status'], 'pass')
            self.assertEqual(path.read_text(encoding='utf-8').replace('__restrict__ ', ''), self.source)

    def test_serialized_double_buffer_is_not_overlap_proof(self):
        code = 'float As[2][BK][BM]; for (int bkIdx = 1; bkIdx < count; ++bkIdx) { As[mem_flag][k][m] = a; __syncthreads(); for (int k = 0; k < BK; ++k) {} }'
        evidence = pipeline_evidence(code)
        self.assertTrue(evidence['load_barrier_compute_pattern'])
        self.assertFalse(evidence['overlap_proven'])
