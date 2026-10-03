import unittest
from tempfile import TemporaryDirectory

from SCOPE.generate_ir.ir_extraction import extract_problem_description
from SCOPE.llm.c_code_generator import apply_cpu_c_code_files
from SCOPE.llm.c_code_generator import extract_cpu_codegen_response
from SCOPE.llm.c_code_generator import validate_cpu_c_code_response
from SCOPE.llm.c_code_generator import validate_cpu_kernel_content
from SCOPE.llm.concrete_code_generator import validate_generated_code_files_response
from SCOPE.llm.concrete_code_generator import complete_codegen_json
from SCOPE.llm.concrete_code_generator import apply_generated_code_files
from SCOPE.llm.patch_generator import load_strategy
from SCOPE.llm.patch_generator import load_code_region_context


class ConcreteCodeGeneratorTests(unittest.TestCase):
    def test_schema_placeholder_strategy_id_is_corrected(self):
        response = {
            "strategy_id": "string",
            "files": [
                {
                    "path": "cuda_kernel.cuh",
                    "content": "#pragma once\n",
                    "change_summary": "minimal test content",
                }
            ],
        }

        validated = validate_generated_code_files_response(
            response,
            "Reordering.KLoop.Unroll8",
        )

        self.assertEqual(validated["strategy_id"], "Reordering.KLoop.Unroll8")
        self.assertIn(
            "Corrected schema placeholder strategy_id",
            " ".join(validated["code_generation_notes"]),
        )

    def test_real_strategy_id_mismatch_still_fails(self):
        response = {
            "strategy_id": "Vectorization.GlobalLoadAB.float4",
            "files": [
                {
                    "path": "cuda_kernel.cuh",
                    "content": "#pragma once\n",
                    "change_summary": "minimal test content",
                }
            ],
        }

        with self.assertRaises(ValueError):
            validate_generated_code_files_response(
                response,
                "Reordering.KLoop.Unroll8",
            )

    def test_load_strategy_synthesizes_index_only_micro_strategy(self):
        strategy = load_strategy({"stages": []}, "Reordering.KLoop.Unroll8")

        self.assertEqual(strategy["strategy_id"], "Reordering.KLoop.Unroll8")
        self.assertEqual(strategy["stage"], "MappingReordering")
        self.assertEqual(strategy["subphase"], "Reordering.ComputeSchedule")
        self.assertEqual(strategy["ir_updates"]["schedule.unrolling"], "Unroll8")

    def test_load_strategy_finds_large_matrix_strategy_in_library(self):
        strategy_library = {
            "stages": [
                {
                    "stage_id": "Pipeline",
                    "ordered_subphases": [
                        {
                            "subphase": "Pipeline.LargeMatrixScheduling",
                            "strategies": [
                                {
                                    "strategy_id": "Memory.L2Reuse.CTASwizzleGroupedN",
                                    "stage": "Pipeline",
                                    "subphase": "Pipeline.LargeMatrixScheduling",
                                }
                            ],
                        }
                    ],
                }
            ]
        }

        strategy = load_strategy(strategy_library, "Memory.L2Reuse.CTASwizzleGroupedN")

        self.assertEqual(strategy["subphase"], "Pipeline.LargeMatrixScheduling")

    def test_load_strategy_falls_back_to_cpu_strategy_library(self):
        strategy = load_strategy({"stages": []}, "CPU.LoopOrder.IJK")

        self.assertEqual(strategy["strategy_id"], "CPU.LoopOrder.IJK")
        self.assertEqual(strategy["stage"], "CPULoopSchedule")
        self.assertEqual(strategy["ir_updates"]["cpu_schedule.loop_order"], "ijk")

    def test_codegen_uses_dedicated_timeout_entrypoint_when_available(self):
        class FakeClient:
            def __init__(self):
                self.used_codegen = False

            def complete_codegen_json(self, messages):
                self.used_codegen = True
                return {"ok": True, "message_count": len(messages)}

            def complete_json(self, messages):
                raise AssertionError("complete_json should not be used for codegen when complete_codegen_json exists")

        client = FakeClient()
        result = complete_codegen_json(client, [{"role": "user", "content": "x"}])

        self.assertTrue(client.used_codegen)
        self.assertEqual(result["message_count"], 1)

    def test_cuda_codegen_accepts_short_region_edit(self):
        with TemporaryDirectory() as tmp:
            from pathlib import Path

            root = Path(tmp)
            (root / "cuda_kernel.cuh").write_text(
                "\n".join(
                    [
                        "__global__ void gemm(float *C) {",
                        "    /*",
                        "     * COMPUTE_INNER_BEGIN",
                        "     */",
                        "    float acc = 0.0f;",
                        "    /*",
                        "     * COMPUTE_INNER_END",
                        "     */",
                        "    /*",
                        "     * STORE_BEGIN",
                        "     */",
                        "    C[0] = acc;",
                        "    /*",
                        "     * STORE_END",
                        "     */",
                        "}",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            response = {
                "strategy_id": "Reordering.KLoop.Unroll8",
                "edits": [
                    {
                        "path": "cuda_kernel.cuh",
                        "region": "COMPUTE_INNER",
                        "replacement_lines": [
                            "float acc = 0.0f;",
                            "acc += C[0] * C[0];",
                        ],
                        "change_summary": "replace compute region",
                    }
                ],
            }
            patch_ir = {
                "patch_generation": {
                    "modified_code_regions": [
                        {"file": "cuda_kernel.cuh", "anchor": "COMPUTE_INNER_BEGIN/END"}
                    ]
                }
            }

            validated = validate_generated_code_files_response(response, "Reordering.KLoop.Unroll8")
            applied = apply_generated_code_files(validated, root, patch_ir, {"strategy_id": "Reordering.KLoop.Unroll8"})

            self.assertEqual(applied["status"], "pass")
            self.assertIn("acc += C[0] * C[0];", (root / "cuda_kernel.cuh").read_text(encoding="utf-8"))

    def test_cuda_region_context_does_not_include_full_source(self):
        with TemporaryDirectory() as tmp:
            from pathlib import Path

            root = Path(tmp)
            long_lines = ["// prefix filler"] * 40
            long_lines.extend(
                [
                    "/*",
                    " * COMPUTE_INNER_BEGIN",
                    " */",
                    "acc += a * b;",
                    "/*",
                    " * COMPUTE_INNER_END",
                    " */",
                ]
            )
            long_lines.extend(["// suffix filler"] * 40)
            (root / "cuda_kernel.cuh").write_text("\n".join(long_lines) + "\n", encoding="utf-8")

            context = load_code_region_context(root, ["cuda_kernel.cuh"], regions=["COMPUTE_INNER"], context_radius=2)

        kernel_context = context["files"]["cuda_kernel.cuh"]
        self.assertEqual(context["context_kind"], "cuda_anchor_regions_only")
        self.assertIn("COMPUTE_INNER", kernel_context["regions"])
        self.assertNotIn("// prefix filler", str(kernel_context["regions"]["COMPUTE_INNER"]["body"]))
        self.assertLess(len(str(context)), 1200)

    def test_cpu_target_is_extracted_from_description(self):
        extracted = extract_problem_description(
            "Generate ordinary C code for CPU row-major fp32 NN GEMM. M:512 N:512 K:512."
        )

        self.assertEqual(extracted["target"]["backend"], "cpu")
        self.assertEqual(extracted["target"]["language"], "c")
        self.assertEqual(extracted["target"]["device"], "cpu")

    def test_cpu_c_generator_rejects_cuda_tokens(self):
        response = {
            "files": [
                {
                    "path": "cpu_kernel.c",
                    "content": "#include <cuda_runtime.h>\n__global__ void gemm() {}\n",
                    "change_summary": "invalid cuda code",
                }
            ]
        }

        with self.assertRaises(ValueError):
            validate_cpu_c_code_response(response)

    def test_cpu_c_generator_accepts_plain_c_kernel(self):
        response = {
            "files": [
                {
                    "path": "cpu_kernel.c",
                    "content": (
                        "void cpu_gemm(int M,int N,int K,float alpha,const float *A,const float *B,float beta,float *C){"
                        "for(int m=0;m<M;++m){for(int n=0;n<N;++n){float acc=0.0f;"
                        "for(int k=0;k<K;++k){acc += A[m*K+k] * B[k*N+n];}"
                        "C[m*N+n]=alpha*acc+beta*C[m*N+n];}}}"
                    ),
                    "change_summary": "plain c gemm",
                }
            ]
        }

        validated = validate_cpu_c_code_response(response)

        self.assertEqual(validated["backend"], "cpu")
        self.assertEqual(validated["language"], "c")

    def test_cpu_c_generator_cannot_replace_fixed_main_harness(self):
        response = {
            "files": [
                {
                    "path": "main.c",
                    "content": "int main(void) { return 0; }",
                    "change_summary": "must remain fixed",
                }
            ]
        }

        with self.assertRaises(ValueError):
            validate_cpu_c_code_response(response)

    def test_cpu_c_generator_accepts_short_anchor_edit(self):
        with TemporaryDirectory() as tmp:
            from pathlib import Path

            root = Path(tmp)
            (root / "cpu_kernel.c").write_text(
                '#include "kernel.h"\n'
                "void cpu_gemm(int M,int N,int K,float alpha,const float *A,const float *B,float beta,float *C){\n"
                "    /* SCOPE_CPU_PATCH_KERNEL_BEGIN */\n"
                "    /* old body */\n"
                "    /* SCOPE_CPU_PATCH_KERNEL_END */\n"
                "}\n",
                encoding="utf-8",
            )
            (root / "kernel.h").write_text("", encoding="utf-8")
            (root / "main.c").write_text("", encoding="utf-8")
            response = {
                "files": None,
                "edits": [
                    {
                        "path": "cpu_kernel.c",
                        "anchor": "SCOPE_CPU_PATCH_KERNEL_BEGIN/END",
                        "replacement_lines": [
                            "for(int m=0;m<M;++m){for(int n=0;n<N;++n){float acc=0.0f;",
                            "for(int k=0;k<K;++k){acc += A[m*K+k] * B[k*N+n];}",
                            "C[m*N+n]=alpha*acc+beta*C[m*N+n];}}",
                        ],
                        "change_summary": "replace kernel body",
                    }
                ],
            }

            validated = validate_cpu_c_code_response(response)
            applied = apply_cpu_c_code_files(validated, root)

            self.assertEqual(applied["status"], "pass")
            self.assertIn("acc += A", (root / "cpu_kernel.c").read_text(encoding="utf-8"))

    def test_cpu_anchor_normalizes_accidental_full_function_and_hoists_include(self):
        with TemporaryDirectory() as tmp:
            from pathlib import Path

            root = Path(tmp)
            (root / "cpu_kernel.c").write_text(
                '#include "kernel.h"\n'
                "void cpu_gemm(int M,int N,int K,float alpha,const float *A,const float *B,float beta,float *C){\n"
                "    /* SCOPE_CPU_PATCH_KERNEL_BEGIN */\n"
                "    /* old body */\n"
                "    /* SCOPE_CPU_PATCH_KERNEL_END */\n"
                "}\n",
                encoding="utf-8",
            )
            (root / "kernel.h").write_text("", encoding="utf-8")
            (root / "main.c").write_text("", encoding="utf-8")
            response = validate_cpu_c_code_response({
                "files": None,
                "edits": [{
                    "path": "cpu_kernel.c",
                    "anchor": "SCOPE_CPU_PATCH_KERNEL_BEGIN/END",
                    "replacement_lines": [
                        "#include <immintrin.h>",
                        "void cpu_gemm(int M,int N,int K,float alpha,const float *A,const float *B,float beta,float *C) {",
                        "for(int m=0;m<M;++m){for(int n=0;n<N;++n){float acc=0.0f;",
                        "for(int k=0;k<K;++k){acc += A[m*K+k] * B[k*N+n];}",
                        "C[m*N+n]=alpha*acc+beta*C[m*N+n];}}",
                        "}",
                    ],
                    "change_summary": "accidental full function",
                }],
            })

            applied = apply_cpu_c_code_files(response, root)
            content = (root / "cpu_kernel.c").read_text(encoding="utf-8")

            self.assertEqual(applied["status"], "pass")
            self.assertEqual(content.count("void cpu_gemm("), 1)
            self.assertEqual(content.count("#include <immintrin.h>"), 1)
            self.assertLess(content.index("#include <immintrin.h>"), content.index("void cpu_gemm("))

    def test_cpu_codegen_recovers_unescaped_multiline_replacement(self):
        malformed = '''{
  "backend": "cpu",
  "language": "c",
  "edits": [
    {
      "path": "cpu_kernel.c",
      "anchor": "SCOPE_CPU_PATCH_KERNEL_BEGIN/END",
      "replacement": "for(int m=0;m<M;++m){
for(int n=0;n<N;++n){float acc=0.0f;
for(int k=0;k<K;++k){acc += A[m*K+k] * B[k*N+n];}
C[m*N+n]=alpha*acc+beta*C[m*N+n];}}
",
      "change_summary": "bad json"
    }
  ]
}'''

        recovered = extract_cpu_codegen_response(malformed)
        validated = validate_cpu_c_code_response(recovered)

        self.assertTrue(validated["recovered_from_malformed_json"])
        self.assertIn("acc += A", validated["edits"][0]["replacement"])

    def test_cpu_static_semantics_rejects_a_k_lane_simd_mapping(self):
        kernel = (
            "#include <immintrin.h>\n"
            "void cpu_gemm(int M,int N,int K,float alpha,const float *A,const float *B,float beta,float *C){"
            "for(int m=0;m<M;++m){for(int n=0;n<N;n+=16){__m512 acc=_mm512_setzero_ps();"
            "for(int kk=0;kk<K;kk+=16){__m512 a=_mm512_loadu_ps(&A[m*K+kk]);"
            "__m512 b=_mm512_loadu_ps(&B[kk*N+n]); acc=_mm512_fmadd_ps(a,b,acc);} "
            "_mm512_storeu_ps(&C[m*N+n],acc);}}}"
        )

        result = validate_cpu_kernel_content(kernel)

        self.assertEqual(result["status"], "fail")
        self.assertIn("SIMD micro-kernel must not load a vector along A's K dimension", result["error_message"])

    def test_cpu_static_semantics_rejects_shared_pack_buffer_before_openmp(self):
        kernel = (
            "void cpu_gemm(int M,int N,int K,float alpha,const float *A,const float *B,float beta,float *C){"
            "float *b_panel = 0; #pragma omp parallel for "
            "for(int m=0;m<M;++m){for(int n=0;n<N;++n){float acc=0.0f;"
            "for(int k=0;k<K;++k){acc += A[m*K+k] * B[k*N+n];}"
            "C[m*N+n]=alpha*acc+beta*C[m*N+n];}}}"
        )

        result = validate_cpu_kernel_content(kernel)

        self.assertEqual(result["status"], "fail")
        self.assertIn("OpenMP GEMM must not share packed panel buffers", result["error_message"])

    def test_cpu_static_semantics_accepts_restrict_pointer_aliases(self):
        kernel = (
            "#include <immintrin.h>\n"
            "void cpu_gemm(int M,int N,int K,float alpha,const float *A,const float *B,float beta,float *C){"
            "const float * restrict A_data=A; const float * restrict B_data=B; float * restrict C_data=C;"
            "for(int m=0;m<M;++m){for(int n=0;n<N;n+=8){__m256 acc=_mm256_setzero_ps();"
            "for(int k=0;k<K;++k){__m256 a=_mm256_broadcast_ss(&A_data[m*K+k]);"
            "__m256 b=_mm256_loadu_ps(&B_data[k*N+n]); acc=_mm256_fmadd_ps(a,b,acc);}"
            "_mm256_storeu_ps(&C_data[m*N+n],acc);}}}"
        )

        result = validate_cpu_kernel_content(kernel)

        self.assertEqual(result["status"], "pass")


if __name__ == "__main__":
    unittest.main()
