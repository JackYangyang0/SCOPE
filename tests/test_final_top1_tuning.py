import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from SCOPE.tune import publish_tuned_shape_bundle, tune_top1_shapes


class FinalTop1TuningTests(unittest.TestCase):
    def test_gpu_bundle_matches_flat_multi_shape_results_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            items = []
            for size in (512, 1024, 2048, 4096):
                top1 = root / str(size) / "top_1"
                top1.mkdir(parents=True)
                (top1 / "cuda_kernel.cuh").write_text(f"// kernel {size}\n", encoding="utf-8")
                items.append({
                    "status": "completed",
                    "shape": {"M": size, "N": size, "K": size},
                    "final_code_dir": str(top1),
                })
            result_dir = publish_tuned_shape_bundle(
                root,
                {"main.cpp": "int main() { return 0; }", "kernel.h": "void cuda_gemm();"},
                items,
                "cuda",
            )
            names = {path.name for path in result_dir.iterdir()}
            self.assertTrue({
                "cuda_kernel_512.cuh", "cuda_kernel_1024.cuh",
                "cuda_kernel_2048.cuh", "cuda_kernel_4096.cuh",
                "main.cpp", "kernel.h", "kernel_dispatch.cu", "kernel_variants.h",
                "Makefile", "build_windows.bat", "shape_manifest.json",
            }.issubset(names))
            dispatch = (result_dir / "kernel_dispatch.cu").read_text(encoding="utf-8")
            self.assertIn("cuda_gemm_512", dispatch)
            self.assertIn("cuda_gemm_4096", dispatch)

    def test_cpu_shapes_reuse_one_frozen_top1_and_publish_shape_winners(self):
        seed_sources = {
            "main.c": "int main(void) { return 0; }",
            "cpu_kernel.c": "void cpu_gemm(void) {}",
            "kernel.h": "void cpu_gemm(void);",
        }
        seed = {
            "accepted": True,
            "candidate_code_dir": "seed",
            "source_snapshot": copy.deepcopy(seed_sources),
            "verified_ir": {
                "problem": {"M": 1024, "N": 1024, "K": 1024},
                "verification": {"accepted": True},
                "performance": {"gflops": 100.0},
            },
        }
        calls = []

        def fake_tune(sources, kernel, ir, library, output, **kwargs):
            calls.append((copy.deepcopy(sources), dict(ir["problem"]), kwargs["backend"]))
            candidate = Path(output) / "candidate_fast"
            candidate.mkdir(parents=True)
            for name, content in sources.items():
                (candidate / name).write_text(content, encoding="utf-8")
            (candidate / "verified_ir.json").write_text(json.dumps(ir), encoding="utf-8")
            (candidate / "candidate.json").write_text("{}", encoding="utf-8")
            return {
                "stop_reason": "candidate_pool_exhausted",
                "top_results": [
                    {"accepted": True, "gflops": 200.0, "code_dir": str(candidate), "tile": {"L2_M": 64}},
                    {"accepted": True, "gflops": 150.0, "code_dir": str(candidate), "tile": {"L2_M": 32}},
                ],
            }

        with tempfile.TemporaryDirectory() as tmp, \
             patch("SCOPE.tune.passed", return_value=True), \
             patch("SCOPE.tune.tune_sources", side_effect=fake_tune):
            report = tune_top1_shapes(
                seed,
                {},
                Path(tmp),
                {"cpu_shapes": [256, 512, 768, 1024], "max_candidates_per_shape": 4},
                backend="cpu",
                platform="windows",
                benchmark_runs=5,
                warmup_runs=2,
            )

        self.assertEqual(report["status"], "completed")
        self.assertEqual([call[1]["M"] for call in calls], [256, 512, 768, 1024])
        self.assertEqual([call[2] for call in calls], ["cpu", "cpu", "cpu", "cpu"])
        self.assertTrue(all(call[0] == seed_sources for call in calls))
        self.assertTrue(all(item["gflops"] == 200.0 for item in report["shapes"]))
        self.assertTrue(all(Path(item["final_code_dir"]).name == "top_1" for item in report["shapes"]))
        self.assertEqual(report["publication_status"], "completed")
        self.assertEqual(Path(report["published_results_dir"]).name, "results")


if __name__ == "__main__":
    unittest.main()
