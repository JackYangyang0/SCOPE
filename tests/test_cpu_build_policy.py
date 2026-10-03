from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from SCOPE.verification.c_build_run_verifier import (
    cpu_benchmark_environment,
    gcc_like_compile_flags,
    infer_cpu_source_compile_policy,
    mark_cpu_unrun,
    verify_cpu_build_and_run,
)
from SCOPE.verification.gemm_semantic_repair import (
    generate_cpu_semantic_repair_candidate,
    remove_redundant_cpu_compiler_pragmas,
)


class CpuBuildPolicyTests(unittest.TestCase):
    def test_cpu_benchmark_workers_are_bounded_by_output_tile_tasks(self):
        env = cpu_benchmark_environment({
            "problem": {"M": 512, "N": 512, "K": 512},
            "cpu_tiling": {"l2_block_m": 256, "l2_block_n": 128, "l1_block_m": 32},
            "cpu_parallel": {"policy": "openmp_row_block"},
            "cpu_microkernel": {"family": "avx2_fma"},
        })
        self.assertEqual(env["OMP_DYNAMIC"], "FALSE")
        self.assertEqual(int(env["OMP_NUM_THREADS"]), min(8, __import__("os").cpu_count() or 1))

    def test_removes_function_body_gcc_option_pragmas_without_rewriting_kernel(self):
        source = (
            '#include "kernel.h"\n'
            'void cpu_gemm(void) {\n'
            '  #pragma GCC optimize("O3,unroll-loops")\n'
            '  #pragma GCC target("avx2,fma")\n'
            '  #pragma omp parallel for\n'
            '  for (int i=0;i<8;++i) {}\n'
            '}\n'
        )
        repaired = remove_redundant_cpu_compiler_pragmas(source)
        self.assertNotIn("#pragma GCC", repaired)
        self.assertIn("#pragma omp parallel for", repaired)
        self.assertIn("for (int i=0", repaired)

    def test_cpu_repair_uses_terminal_repair_contract(self):
        with tempfile.TemporaryDirectory() as source_temp, tempfile.TemporaryDirectory() as output_temp:
            source_dir = Path(source_temp)
            output_dir = Path(output_temp)
            (source_dir / "main.c").write_text("int main(void){return 0;}\n", encoding="utf-8")
            (source_dir / "kernel.h").write_text("void cpu_gemm(void);\n", encoding="utf-8")
            (source_dir / "cpu_kernel.c").write_text(
                'void cpu_gemm(void){\n#pragma GCC optimize("O3")\nint kept=1;\n}\n',
                encoding="utf-8",
            )
            result = generate_cpu_semantic_repair_candidate(
                source_dir,
                output_dir,
                {"defects": [{"defect_type": "Compile.CPUCompilationError"}]},
                {"target": {"backend": "cpu"}, "strategy": {"applied_strategy_ids": ["CPU.Compiler.NativeO3"]}},
            )

        self.assertEqual(result["status"], "repair_generated")
        self.assertTrue(result["source_hash_after"])
        self.assertEqual(result["locked_strategy_ids"], ["CPU.Compiler.NativeO3"])
        self.assertEqual(result["method"], "deterministic_remove_redundant_compiler_pragmas")

    def test_compile_failure_clears_stale_performance(self):
        ir = {
            "verification": {
                "accepted": True,
                "run_stdout": "old successful run",
                "benchmark": {"runs": 5},
            },
            "performance": {
                "gflops": 350.0,
                "gflops_mean": 352.0,
                "latency_ms": 0.7,
                "benchmark_successful_runs": 5,
            },
        }
        mark_cpu_unrun(ir, "compile failed")
        self.assertFalse(ir["verification"]["accepted"])
        self.assertNotIn("run_stdout", ir["verification"])
        self.assertIsNone(ir["performance"]["gflops_mean"])
        self.assertEqual(ir["performance"]["benchmark_successful_runs"], 0)

    def test_infers_avx2_fma_and_openmp_from_generated_source(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            source_dir = Path(temporary_dir)
            (source_dir / "cpu_kernel.c").write_text(
                "#include <immintrin.h>\n"
                "#pragma omp parallel for\n"
                "void kernel(float *x){__m256 v=_mm256_setzero_ps();"
                "v=_mm256_fmadd_ps(v,v,v);_mm256_storeu_ps(x,v);}\n",
                encoding="utf-8",
            )

            policy = infer_cpu_source_compile_policy(source_dir, {"optimization_level": "O3"})
            flags = gcc_like_compile_flags(policy)

        self.assertEqual(policy["vector_isa"], "avx2_fma")
        self.assertTrue(policy["openmp"])
        self.assertIn("-mavx2", flags)
        self.assertIn("-mfma", flags)
        self.assertIn("-fopenmp", flags)

    def test_verifier_resolves_relative_build_directory(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary_dir:
            source_dir = Path(temporary_dir)
            (source_dir / "kernel.h").write_text(
                "void cpu_gemm(int,int,int,float,const float*,const float*,float,float*);\n",
                encoding="utf-8",
            )
            (source_dir / "cpu_kernel.c").write_text(
                '#include "kernel.h"\nvoid cpu_gemm(int M,int N,int K,float alpha,const float*A,const float*B,float beta,float*C){C[0]=0;}\n',
                encoding="utf-8",
            )
            (source_dir / "main.c").write_text("int main(void){return 0;}\n", encoding="utf-8")
            relative_source = source_dir.relative_to(Path.cwd())

            result = verify_cpu_build_and_run(
                {"problem": {"M": 1, "N": 1, "K": 1}},
                source_dir=relative_source,
                build_dir=relative_source / "build",
                benchmark_runs=1,
                benchmark_warmup_runs=0,
            )

        self.assertEqual(result["verification"]["compile"]["status"], "pass")


if __name__ == "__main__":
    unittest.main()
