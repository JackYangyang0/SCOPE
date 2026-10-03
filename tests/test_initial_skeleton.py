from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from SCOPE.app import (
    ROOT,
    apply_deterministic_code_patch,
    backend_ast_files,
    backend_llm_context_files,
    restore_initial_cpu_skeleton,
    restore_initial_cuda_skeleton,
)
from SCOPE.verification.cpu_skeleton_audit import audit_initial_cpu_skeleton


class InitialSkeletonTests(unittest.TestCase):
    def test_new_run_restores_pristine_kernel(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "cuda_kernel.cuh").write_text("broken previous candidate", encoding="utf-8")
            (root / "main.cpp").write_text("fixed harness", encoding="utf-8")
            restore_initial_cuda_skeleton(root, {})
            expected = (ROOT / "gemm_code/baseline_template/cuda_kernel.cuh").read_text(encoding="utf-8")
            self.assertEqual((root / "cuda_kernel.cuh").read_text(encoding="utf-8"), expected)
            self.assertEqual((root / "main.cpp").read_text(encoding="utf-8"), "fixed harness")

    def test_initial_cuda_skeleton_is_intentionally_naive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            restore_initial_cuda_skeleton(root, {})
            content = (root / "cuda_kernel.cuh").read_text(encoding="utf-8")
            self.assertIn("SCOPE_NAIVE_BASELINE", content)
            self.assertNotIn("__shared__ float", content)
            self.assertNotIn("float4", content)
            self.assertIn("for (int k = 0; k < K; ++k)", content)

    def test_first_deterministic_strategy_activates_construction_scaffold(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            restore_initial_cuda_skeleton(root, {})
            result, _ = apply_deterministic_code_patch(root, "Tiling.BlockTile.64x64x16", {})
            self.assertEqual(result["status"], "pass")
            content = (root / "cuda_kernel.cuh").read_text(encoding="utf-8")
            self.assertNotIn("SCOPE_NAIVE_BASELINE", content)
            self.assertIn("__shared__ float", content)

    def test_register_strategy_preserves_tile_reduction_and_shared_dataflow(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            restore_initial_cuda_skeleton(root, {})
            result, _ = apply_deterministic_code_patch(root, "Register.AccumulatorLayout.2DArray", {})
            self.assertEqual(result["status"], "pass")
            content = (root / "cuda_kernel.cuh").read_text(encoding="utf-8")
            inner = content.split("COMPUTE_INNER_BEGIN", 1)[1].split("COMPUTE_INNER_END", 1)[0]
            self.assertIn("k < BK", inner)
            self.assertNotIn("k < K;", inner)
            self.assertIn("As[comp_flag][k]", inner)
            self.assertIn("Bs[comp_flag][k]", inner)
            self.assertNotIn("A[OFFSET", inner)
            self.assertNotIn("B[OFFSET", inner)
            self.assertEqual(content.count("GLOBAL_TO_SHARED_LOAD_BEGIN"), 1)
            self.assertEqual(content.count("MAIN_LOOP_BEGIN"), 1)

    def test_cpu_run_restores_pristine_scalar_skeleton(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("main.c", "kernel.h", "cpu_kernel.c"):
                (root / name).write_text("stale optimized candidate", encoding="utf-8")
            ir = {"target": {"backend": "cpu"}}
            restore_initial_cpu_skeleton(root, ir)
            report = audit_initial_cpu_skeleton(root)
            kernel = (root / "cpu_kernel.c").read_text(encoding="utf-8")

        self.assertTrue(report["accepted"])
        self.assertIn("for (int k = 0; k < K; ++k)", kernel)
        self.assertNotIn("__m256", kernel)
        self.assertNotIn("#pragma omp", kernel)

    def test_cpu_main_is_not_sent_to_llm_or_ast(self):
        ir = {"target": {"backend": "cpu"}}
        self.assertEqual(backend_llm_context_files(ir), ["kernel.h", "cpu_kernel.c"])
        self.assertEqual(backend_ast_files(ir), ["kernel.h", "cpu_kernel.c"])

    def test_cpu_harness_warms_up_and_averages_inside_each_process(self):
        content = (ROOT / "gemm_code/cpu_baseline_template/main.c").read_text(encoding="utf-8")
        self.assertIn("int warmup_iterations = 2;", content)
        self.assertIn("int iterations = 10;", content)
        self.assertLess(content.index("for (int run = 0; run < warmup_iterations"), content.index("const double start"))
        self.assertGreater(content.rindex("reference_gemm("), content.index("const double latency_ms"))


if __name__ == "__main__":
    unittest.main()
