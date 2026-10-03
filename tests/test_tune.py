import copy
import tempfile
import unittest
from pathlib import Path

from SCOPE.tune import (
    build_cpu_tile_space,
    extract_cpu_tile,
    publish_tuned_shape_bundle,
    specialize_cpu_source,
    tune_cpu_sources,
)


CPU_SOURCE = """#include \"kernel.h\"
void cpu_gemm(void) {
    const int L2_BLOCK_M = 128;
    const int L2_BLOCK_N = 128;
    const int L2_BLOCK_K = 128;
    const int L1_BLOCK_M = 32;
    const int L1_BLOCK_N = 32;
    const int L1_BLOCK_K = 64;
    const int MR = 4;
    const int NR = 8;
    /* optimized body must remain */
}
"""


def verified(ir, gflops):
    result = copy.deepcopy(ir)
    result["verification"] = {
        "accepted": True,
        "compile": {"status": "pass"},
        "correctness": {"status": "pass"},
        "runtime_safety": {"status": "pass"},
    }
    result["performance"] = {
        "gflops": gflops,
        "gflops_mean": gflops,
        "latency_ms": 1.0,
        "latency_ms_mean": 1.0,
        "cpu_blas_gflops": 100.0,
    }
    return result


class UnifiedTuneTests(unittest.TestCase):
    def test_single_shape_cpu_publication_writes_dynamic_makefile(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            winner = root / "winner"
            winner.mkdir()
            (winner / "cpu_kernel.c").write_text("void cpu_gemm(void) {}\n", encoding="utf-8")
            result = publish_tuned_shape_bundle(
                root,
                {"main.c": "int main(void) { return 0; }\n", "kernel.h": "#pragma once\n"},
                [{
                    "shape": {"M": 768, "N": 768, "K": 768},
                    "status": "completed",
                    "final_code_dir": str(winner),
                }],
                "cpu",
            )
            makefile = (result / "Makefile").read_text(encoding="utf-8")
            dispatch = (result / "kernel_dispatch.c").read_text(encoding="utf-8")
            self.assertIn("cpu_kernel_768.c", makefile)
            self.assertIn("run-768: all", makefile)
            self.assertNotIn("cpu_kernel_1024.c", makefile)
            self.assertIn("cpu_gemm_768(M, N, K", dispatch)
            self.assertNotIn("    else {", dispatch)

    def test_gpu_publication_removes_single_kernel_include_and_isolates_helpers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            items = []
            for size in (512, 1024):
                winner = root / f"winner_{size}"
                winner.mkdir()
                (winner / "cuda_kernel.cuh").write_text(
                    "__global__ void scope_reduce(float *value) {}\n"
                    "void cuda_gemm(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {}\n",
                    encoding="utf-8",
                )
                items.append({
                    "shape": {"M": size, "N": size, "K": size},
                    "status": "completed",
                    "final_code_dir": str(winner),
                })
            result = publish_tuned_shape_bundle(
                root,
                {
                    "main.cpp": '#include "kernel.h"\nint main() { return 0; }\n',
                    "kernel.h": '#include <cuda_runtime.h>\n#include "cuda_kernel.cuh"\nvoid cuda_gemm();\n',
                },
                items,
                "cuda",
            )
            header = (result / "kernel.h").read_text(encoding="utf-8")
            dispatch = (result / "kernel_dispatch.cu").read_text(encoding="utf-8")
            self.assertNotIn('cuda_kernel.cuh', header)
            self.assertIn("#define scope_reduce scope_reduce_512", dispatch)
            self.assertIn("#define scope_reduce scope_reduce_1024", dispatch)

    def test_cpu_space_expands_l2_axes_independently(self):
        source = """void cpu_gemm(void) {
 const int L2_BLOCK_M = 128; const int L2_BLOCK_N = 128; const int L2_BLOCK_K = 128;
 const int MR = 4; const int NR = 8;
}"""
        tiles, audit = build_cpu_tile_space(
            {}, source,
            {"problem": {"M": 512, "N": 512, "K": 512}, "hardware": {"cpu_physical_cores": 8}},
        )
        self.assertGreater(len(tiles), 100)
        self.assertGreater(len({tile["L2_N"] for tile in tiles}), 2)
        self.assertGreater(len({tile["L2_K"] for tile in tiles}), 2)
        self.assertEqual(len(tiles), audit["accepted_candidate_count"])

    def test_cpu_specialization_only_replaces_constants(self):
        tile = {
            "L2_M": 256, "L2_N": 128, "L2_K": 128,
            "L1_M": 16, "L1_N": 32, "L1_K": 64,
            "MR": 4, "NR": 8,
        }
        updated = specialize_cpu_source(CPU_SOURCE, tile)
        self.assertIsNotNone(updated)
        self.assertIn("L2_BLOCK_M = 256", updated)
        self.assertIn("L1_BLOCK_M = 16", updated)
        self.assertIn("optimized body must remain", updated)
        self.assertEqual(tile, extract_cpu_tile(updated))

    def test_cpu_tuning_ranks_passing_candidates_and_preserves_harness(self):
        sources = {
            "cpu_kernel.c": CPU_SOURCE,
            "main.c": "/* fixed CPU harness */",
            "kernel.h": "void cpu_gemm(void);",
        }
        state = {
            "problem": {"M": 512, "N": 512, "K": 512},
            "target": {"backend": "cpu"},
        }
        tiles = [
            {"L2_M": 256, "L2_N": 128, "L2_K": 128, "L1_M": 32, "L1_N": 32, "L1_K": 64, "MR": 4, "NR": 8},
            {"L2_M": 192, "L2_N": 128, "L2_K": 128, "L1_M": 32, "L1_N": 32, "L1_K": 64, "MR": 4, "NR": 8},
        ]
        calls = []

        def verify(ir, **kwargs):
            calls.append(kwargs)
            self.assertEqual("/* fixed CPU harness */", (kwargs["source_dir"] / "main.c").read_text())
            return verified(ir, 200.0 + len(calls) * 10.0)

        with tempfile.TemporaryDirectory() as tmp:
            report = tune_cpu_sources(
                sources, "cpu_kernel.c", state, {}, Path(tmp) / "out",
                candidate_tiles=tiles, rounds=2, verify=verify,
            )
            self.assertEqual(3, len(calls))
            self.assertEqual(230.0, report["top_results"][0]["gflops"])
            self.assertEqual(2.3, report["top_results"][0]["relative_to_cpu_blas"])
            self.assertTrue((Path(tmp) / "out" / "top_3" / "rank_1" / "cpu_kernel.c").exists())


if __name__ == "__main__":
    unittest.main()
