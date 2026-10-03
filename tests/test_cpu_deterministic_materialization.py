from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from SCOPE.app import (
    apply_deterministic_cpu_code_patch,
    build_fallback_micro_selection,
    check_cpu_strategy_code_obligations,
    deterministic_cpu_strategy_codegen,
    is_deterministic_strategy,
    restore_selected_cpu_microkernel,
)
from SCOPE.verification.gemm_semantic_repair import build_repaired_cpu_kernel_c


class CpuDeterministicMaterializationTests(unittest.TestCase):
    def test_cpu_tiling_is_deterministic_and_replaces_constants(self):
        self.assertTrue(is_deterministic_strategy("CPU.Tiling.L2Block.256x128x128"))
        source = (
            '#include "kernel.h"\nvoid cpu_gemm(void) {\n'
            'enum { MC = 64 }; enum { NC = 64 }; enum { KC = 32 };\n'
            'enum { M1 = 16 }; enum { N1 = 16 }; enum { K1 = 16 };\n'
            'enum { MR = 2 }; enum { NR = 4 };\n}\n'
        )
        with tempfile.TemporaryDirectory() as temporary_dir:
            directory = Path(temporary_dir)
            kernel = directory / "cpu_kernel.c"
            kernel.write_text(source, encoding="utf-8")
            result, generated = apply_deterministic_cpu_code_patch(
                directory,
                "CPU.Tiling.L2Block.256x128x128",
                {"cpu_tiling": {"l2_block_m": 256, "l2_block_n": 128, "l2_block_k": 128}},
            )
            updated = kernel.read_text(encoding="utf-8")

        self.assertEqual(result["status"], "pass")
        self.assertEqual(generated["generation_method"], "deterministic_cpu_tiling_materialization")
        self.assertIn("L2_BLOCK_M = 256", updated)
        self.assertIn("L2_BLOCK_N = 128", updated)
        self.assertIn("L2_BLOCK_K = 128", updated)

    def test_avx_microkernel_bypasses_llm_codegen(self):
        ir = {
            "cpu_tiling": {"l2_block_m": 256, "l2_block_n": 128, "l2_block_k": 128},
            "cpu_microkernel": {"family": "avx2_fma", "mr": 4, "nr": 16},
            "strategy": {
                "applied_strategy_ids": ["CPU.MicroKernel.AVX2.FMA.4x16"],
                "applied_micro_strategies": [
                    {"strategy_id": "CPU.MicroKernel.AVX2.FMA.4x16"}
                ],
            },
        }
        generated = deterministic_cpu_strategy_codegen("CPU.MicroKernel.AVX2.FMA.4x16", ir)
        self.assertIsNotNone(generated)
        source = generated["files"][0]["content"]
        self.assertIn("_mm256_fmadd_ps", source)
        self.assertEqual(generated["generation_method"], "deterministic_cpu_microkernel")

    def test_no_pack_and_cpu_microkernels_are_deterministic_fallbacks(self):
        self.assertTrue(is_deterministic_strategy("CPU.Memory.NoPack"))
        self.assertTrue(is_deterministic_strategy("CPU.MicroKernel.AVX2.FMA.4x8"))
        selection = build_fallback_micro_selection(
            {
                "strategies": [
                    {"strategy_id": "CPU.Memory.PackB.ContiguousPanel", "priority": 99},
                    {"strategy_id": "CPU.Memory.NoPack", "priority": 50},
                ]
            },
            "CPUPacking.PanelPackingPolicy",
            "offline",
        )
        self.assertEqual(selection["selected_strategy_id"], "CPU.Memory.PackB.ContiguousPanel")

    def test_deterministic_microkernel_materializes_intrinsics(self):
        ir = {
            "hardware": {"cpu_isa": {"avx2": True, "fma": True}},
            "cpu_tiling": {
                "l2_block_m": 128, "l2_block_n": 128, "l2_block_k": 128,
                "l1_block_m": 32, "l1_block_n": 32, "l1_block_k": 64,
                "register_m": 4, "register_n": 8,
            },
            "cpu_microkernel": {"family": "avx2_fma", "mr": 4, "nr": 8},
        }
        with tempfile.TemporaryDirectory() as temporary_dir:
            directory = Path(temporary_dir)
            kernel = directory / "cpu_kernel.c"
            kernel.write_text('#include "kernel.h"\n', encoding="utf-8")
            result, generated = apply_deterministic_cpu_code_patch(
                directory, "CPU.MicroKernel.AVX2.FMA.4x8", ir,
            )
            source = kernel.read_text(encoding="utf-8")
        self.assertEqual(result["status"], "pass")
        self.assertEqual(
            generated["generation_method"],
            "deterministic_cpu_microkernel_materialization",
        )
        self.assertIn("_mm256_fmadd_ps", source)

    def test_terminal_restores_simd_erased_by_later_patch(self):
        ir = {
            "cpu_tiling": {"l2_block_m": 256, "l2_block_n": 128, "l2_block_k": 128},
            "cpu_microkernel": {"family": "avx2_fma", "mr": 4, "nr": 16},
            "strategy": {
                "applied_strategy_ids": ["CPU.MicroKernel.AVX2.FMA.4x16"],
                "applied_micro_strategies": [
                    {"strategy_id": "CPU.MicroKernel.AVX2.FMA.4x16"}
                ],
            },
        }
        with tempfile.TemporaryDirectory() as temporary_dir:
            directory = Path(temporary_dir)
            kernel = directory / "cpu_kernel.c"
            kernel.write_text(
                '#include "kernel.h"\nvoid cpu_gemm(int M,int N,int K,float a,'
                'const float*A,const float*B,float b,float*C){C[0]=0;}\n',
                encoding="utf-8",
            )
            report = restore_selected_cpu_microkernel(directory, ir)
            restored = kernel.read_text(encoding="utf-8")

        self.assertEqual(report["status"], "restored")
        self.assertIn("_mm256_fmadd_ps", restored)

    def test_final_ir_shape_overrides_historical_4x8_strategy(self):
        ir = {
            "hardware": {"cpu_isa": {"avx2": True, "fma": True}},
            "cpu_tiling": {"l2_block_m": 192, "l2_block_n": 128, "l2_block_k": 96},
            "cpu_microkernel": {"family": "avx2_fma", "mr": 4, "nr": 16},
            "strategy": {
                "applied_strategy_ids": [
                    "CPU.MicroKernel.AVX2.FMA.4x8",
                    "CPU.MicroKernel.AVX2.FMA.4x16",
                ]
            },
        }
        source = build_repaired_cpu_kernel_c(ir)
        self.assertIn("const int MR = 4", source)
        self.assertIn("const int NR = 16", source)
        self.assertNotIn("const int NR = 8", source)

    def test_final_avx2_family_overrides_historical_avx512_strategy(self):
        ir = {
            "hardware": {"cpu_isa": {"avx2": True, "fma": True, "avx512f": True}},
            "cpu_tiling": {"l2_block_m": 128, "l2_block_n": 128, "l2_block_k": 64},
            "cpu_microkernel": {"family": "avx2_fma", "mr": 4, "nr": 16},
            "strategy": {
                "applied_strategy_ids": [
                    "CPU.MicroKernel.AVX512.FMA.8x16",
                    "CPU.MicroKernel.AVX2.FMA.4x16",
                ]
            },
        }
        source = build_repaired_cpu_kernel_c(ir)
        self.assertIn("_mm256_fmadd_ps", source)
        self.assertNotIn("_mm512_fmadd_ps", source)

    def test_packab_panel_driver_is_materialized_and_verified(self):
        strategy_ids = [
            "CPU.Packing.PackAB.MRxKC_KCxNR",
            "CPU.MicroKernel.AVX2.FMA.4x16",
            "CPU.MacroKernel.OpenBLASStyle.PanelDriver",
            "CPU.KLoop.Unroll4",
            "CPU.Parallel.OpenMP.RowBlock",
        ]
        ir = {
            "hardware": {"cpu_isa": {"avx2": True, "fma": True}},
            "cpu_tiling": {"l2_block_m": 192, "l2_block_n": 128, "l2_block_k": 96},
            "cpu_microkernel": {"family": "avx2_fma", "mr": 4, "nr": 16},
            "cpu_memory": {
                "pack_a": True, "pack_b": True,
                "pack_buffer_scope": "thread_private", "prefetch": "ab_panel",
            },
            "cpu_schedule": {"k_unroll": 4, "loop_order": "packed_panel_major"},
            "cpu_macro_kernel": {"driver": "openblas_style_panel_driver"},
            "strategy": {"applied_strategy_ids": strategy_ids},
        }
        source = build_repaired_cpu_kernel_c(ir)
        failures = [
            item["id"] for item in check_cpu_strategy_code_obligations(source, ir)
            if item["status"] == "fail"
        ]
        self.assertEqual([], failures)
        self.assertIn("float *packed_b", source)
        self.assertIn("float *a_panel", source)
        self.assertIn("k += 4", source)
        self.assertIn("_mm_prefetch", source)
        self.assertIn("#pragma omp parallel", source)

    def test_terminal_rebuilds_wrong_microkernel_shape(self):
        ir = {
            "hardware": {"cpu_isa": {"avx2": True, "fma": True}},
            "cpu_tiling": {"l2_block_m": 128, "l2_block_n": 128, "l2_block_k": 64},
            "cpu_microkernel": {"family": "avx2_fma", "mr": 4, "nr": 16},
            "strategy": {"applied_strategy_ids": ["CPU.MicroKernel.AVX2.FMA.4x16"]},
        }
        wrong_ir = {**ir, "cpu_microkernel": {"family": "avx2_fma", "mr": 4, "nr": 8}}
        with tempfile.TemporaryDirectory() as temporary_dir:
            directory = Path(temporary_dir)
            kernel = directory / "cpu_kernel.c"
            kernel.write_text(build_repaired_cpu_kernel_c(wrong_ir), encoding="utf-8")
            report = restore_selected_cpu_microkernel(directory, ir)
            restored = kernel.read_text(encoding="utf-8")
        self.assertEqual("restored", report["status"])
        self.assertIn("CPU_STRATEGY_MICROKERNEL_SHAPE_MATCH", report["repaired_check_ids"])
        self.assertIn("const int NR = 16", restored)


if __name__ == "__main__":
    unittest.main()
