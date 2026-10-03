import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from SCOPE.benchmark import (
    DEFAULT_CPU_PROGRAMS,
    benchmark_case,
    generated_openblas_baselines,
    parse_args,
    parse_metrics,
    reported_cublas_baselines,
    validate_error_metrics,
)


class BenchmarkDefaultsTest(unittest.TestCase):
    def test_default_cpu_programs_are_named(self):
        self.assertTrue(DEFAULT_CPU_PROGRAMS)
        self.assertTrue(all("=" in program for program in DEFAULT_CPU_PROGRAMS))

    def test_cpu_openblas_baseline_is_rebenchmarked_by_default(self):
        with patch.object(sys, "argv", ["benchmark.py", "--backend", "cpu"]):
            args = parse_args()
        self.assertEqual(args.openblas_baseline_source, "rebenchmark")

    def test_absolute_error_gate_is_enabled_by_default(self):
        with patch.object(sys, "argv", ["benchmark.py", "--backend", "gpu"]):
            args = parse_args()
        self.assertEqual(args.max_abs_error_tolerance, 1.0e-6)
        self.assertTrue(args.require_error_metrics)

    def test_generation_openblas_baseline_uses_verified_ir_value(self):
        with tempfile.TemporaryDirectory() as temporary:
            shape_dir = Path(temporary) / "1024"
            shape_dir.mkdir()
            (shape_dir / "verified_ir.json").write_text(
                json.dumps({
                    "performance": {
                        "cpu_blas_gflops": 633.25,
                        "cpu_blas_latency_ms": 3.4,
                    },
                    "verification": {
                        "optimized_cpu_baseline": {
                            "command": ["openblas_sgemm_baseline.exe", "1024", "1024", "1024"]
                        }
                    },
                }),
                encoding="utf-8",
            )
            rows = generated_openblas_baselines([(1024, 1024, 1024)], Path(temporary))
        self.assertEqual(rows[0]["baseline_source"], "generation_verified_ir")
        self.assertEqual(rows[0]["gflops"]["mean"], 633.25)

    def test_scope_and_cublas_metrics_are_parsed(self):
        metrics = parse_metrics(
            "SCOPE GEMM Performance= 1234.5 GFlop/s, Time= 0.250 msec\n"
            "CuBlas Performance= 4321.0 GFlop/s, Time= 0.100 msec\n"
            "Result= PASS\n"
        )
        self.assertEqual(metrics["gflops"], 1234.5)
        self.assertEqual(metrics["latency_ms"], 0.25)
        self.assertEqual(metrics["cublas_gflops"], 4321.0)
        self.assertEqual(metrics["correctness"], "pass")

    def test_reported_cublas_is_an_independent_default_baseline(self):
        rows = reported_cublas_baselines([
            {
                "program": "scope",
                "status": "pass",
                "shape": {"M": 512, "N": 512, "K": 512},
                "command": ["gemm.exe", "512", "512", "512"],
                "warmup_runs": 2,
                "reported_cublas_gflops": {
                    "count": 8,
                    "raw_count": 10,
                    "mean": 4000.0,
                },
            }
        ])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["program"], "cuBLAS")
        self.assertEqual(rows[0]["source_program"], "scope")
        self.assertEqual(rows[0]["gflops"]["mean"], 4000.0)

    def test_structured_error_metrics_are_parsed(self):
        metrics = parse_metrics(
            "SCOPE_METRIC backend=gpu correctness=pass gflops=1234.5 latency_ms=0.25 "
            "max_abs_error=2.0e-5 max_rel_error=4.0e-4 relative_l2_error=8.0e-5 "
            "atol=1.0e-5 rtol=1.0e-3\n"
        )
        self.assertEqual(metrics["correctness"], "pass")
        self.assertAlmostEqual(metrics["max_abs_error"], 2.0e-5)
        self.assertAlmostEqual(metrics["relative_l2_error"], 8.0e-5)

    @staticmethod
    def error_args(require_error_metrics=False):
        return argparse.Namespace(
            fp32_atol=1.0e-5,
            fp32_rtol=1.0e-3,
            max_abs_error_tolerance=1.0e-6,
            relative_l2_tolerance=1.0e-3,
            require_error_metrics=require_error_metrics,
        )

    def test_error_validation_accepts_standard_fp32_error(self):
        validation = validate_error_metrics(
            {"correctness": "pass", "max_abs_error": 9.9e-7, "relative_l2_error": 8.0e-5},
            self.error_args(),
        )
        self.assertEqual(validation["status"], "pass")

    def test_absolute_error_at_or_above_one_e_minus_six_fails(self):
        for maximum in (1.0e-6, 2.0e-5):
            with self.subTest(max_abs_error=maximum):
                validation = validate_error_metrics(
                    {"correctness": "pass", "max_abs_error": maximum, "relative_l2_error": 8.0e-5},
                    self.error_args(),
                )
                self.assertEqual(validation["status"], "fail")
                self.assertIn("max_abs_error", validation["failures"][0])

    def test_nonfinite_absolute_error_is_missing_and_fails_closed(self):
        validation = validate_error_metrics(
            {"correctness": "pass", "max_abs_error": float("nan"), "relative_l2_error": 0.0},
            self.error_args(require_error_metrics=True),
        )
        self.assertEqual(validation["status"], "fail")
        self.assertIn("max_abs_error", validation["missing_metrics"])

    def test_case_with_missing_error_metrics_cannot_pass(self):
        args = argparse.Namespace(
            arg_order="MKN", warmup_runs=0, runs=1, timeout=1.0, quiet=True,
            trim=0, **vars(self.error_args(require_error_metrics=False)),
        )
        result = {"status": "pass", "returncode": 0, "metrics": {
            "gflops": 100.0, "latency_ms": 1.0, "cublas_gflops": None,
            "correctness": "pass",
        }, "output": "Result= PASS"}
        program = {"name": "candidate", "path": Path("candidate.exe"), "template": None}
        with patch("SCOPE.benchmark.run_once", return_value=result):
            row = benchmark_case(program, (512, 512, 512), args, None)
        self.assertEqual(row["error_validation"]["status"], "unverified")
        self.assertEqual(row["status"], "fail")

    def test_error_validation_rejects_excessive_relative_l2(self):
        validation = validate_error_metrics(
            {"correctness": "pass", "max_abs_error": 0.1, "relative_l2_error": 2.0e-3},
            self.error_args(),
        )
        self.assertEqual(validation["status"], "fail")

    def test_legacy_output_is_unverified_unless_metrics_are_required(self):
        metrics = {"correctness": "pass"}
        self.assertEqual(validate_error_metrics(metrics, self.error_args())["status"], "unverified")
        self.assertEqual(
            validate_error_metrics(metrics, self.error_args(require_error_metrics=True))["status"],
            "fail",
        )


if __name__ == "__main__":
    unittest.main()
