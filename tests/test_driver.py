"""The independent driver: case file contract, guard bands, parameter order.

These tests need a C compiler; they skip themselves when none is available, so
the suite still runs on a bare machine.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from SCOPE.verification import driver_build as build
from SCOPE.data import case_io as cases
from SCOPE.verification import controls as controls
from SCOPE.data import matrix_cases as data
from SCOPE.verification import reference_fp64 as reference
from SCOPE.utils.harness_util import run_command

import support


@unittest.skipIf(support.cpu_toolchain() is None, "no CPU C compiler available")
class DriverTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = Path(tempfile.mkdtemp(prefix="scope-exp-driver-"))
        cls.toolchain = support.cpu_toolchain()
        cls.kernel = cls.temp / "kernel"
        cls.kernel.mkdir(parents=True)
        (cls.kernel / "cpu_kernel.c").write_bytes(
            (controls.CONTROL_SOURCES / "cpu_reference_fp64.c").read_bytes())
        (cls.kernel / "kernel.h").write_bytes(
            (support.CPU_SKELETON / "kernel.h").read_bytes())
        cls.build = build.build_eval_driver(cls.kernel, cls.temp / "build",
                                            backend="cpu",
                                            toolchain=cls.toolchain)
        assert cls.build.ok, cls.build.record.get("error")

    def test_case_file_round_trip(self):
        case = data.make_case("uniform_small.s01", 7, 9, 5, c0_variant="scalar")
        path = self.temp / cases.case_file_name(data.case_key(case))
        record = cases.write_case_file(path, case)
        self.assertEqual(record["shape"], {"M": 7, "N": 9, "K": 5})
        restored = cases.read_case_file(path)
        self.assertTrue(np.array_equal(restored["A"], case["A"]))
        self.assertTrue(np.array_equal(restored["C0"], case["C0"]))
        self.assertEqual(restored["sha256"], record["sha256"])

    def test_compute_returns_exact_binary_output(self):
        case = data.make_case("uniform_small.s01", 16, 16, 16)
        case_path = self.temp / "compute.case"
        cases.write_case_file(case_path, case)
        output = self.temp / "compute.out"
        result = run_command([str(self.build.binary), "--case", str(case_path),
                              "--mode", "compute", "--out", str(output), "--guard"],
                             timeout=120)
        self.assertEqual(result["returncode"], 0, result["stderr"])
        parsed = cases.parse_driver_stdout(result["stdout"])
        self.assertTrue(parsed["found"], parsed)
        record = parsed["record"]
        self.assertEqual(record["shape"], {"M": 16, "N": 16, "K": 16})
        self.assertEqual(record["call_shape"], {"M": 16, "N": 16, "K": 16})
        self.assertFalse(record["guard_corrupted"])
        self.assertFalse(record["legacy_mkn_order"])
        computed = cases.read_output_file(output, 16, 16)
        reference_output, _ = reference.cpu_blas_fp64(case["A"], case["B"])
        self.assertLess(float(np.max(np.abs(computed.astype(np.float64)
                                           - reference_output))), 1e-5)

    def test_timing_reports_every_launch(self):
        case = data.make_case("uniform_small.s01", 32, 32, 32)
        case_path = self.temp / "timing.case"
        cases.write_case_file(case_path, case)
        result = run_command([str(self.build.binary), "--case", str(case_path),
                              "--mode", "time", "--warmup-launches", "2",
                              "--iters", "5"], timeout=120)
        self.assertEqual(result["returncode"], 0, result["stderr"])
        record = cases.parse_driver_stdout(result["stdout"])["record"]
        self.assertEqual(len(record["latencies_ms"]), 5)
        self.assertGreater(record["gflops"], 0)
        self.assertEqual(record["warmup_launches"], 2)

    def test_legacy_mkn_order_fails_on_non_square(self):
        """The historical M,K,N positional trap (§5.4)."""
        M, N, K = 127, 129, 65
        case = data.make_case("uniform_small.s02", M, N, K)
        case_path = self.temp / "nonsquare.case"
        cases.write_case_file(case_path, case)

        good = run_command([str(self.build.binary), "--case", str(case_path),
                            "--mode", "compute", "--out",
                            str(self.temp / "good.out")], timeout=120)
        self.assertEqual(good["returncode"], 0)
        good_record = cases.parse_driver_stdout(good["stdout"])["record"]
        self.assertEqual(good_record["call_shape"], {"M": M, "N": N, "K": K})
        good_output = cases.read_output_file(self.temp / "good.out", M, N)
        expected, _ = reference.cpu_blas_fp64(case["A"], case["B"])
        self.assertLess(float(np.max(np.abs(good_output.astype(np.float64)
                                           - expected))), 1e-5)

        bad = run_command([str(self.build.binary), "--case", str(case_path),
                           "--mode", "compute", "--out",
                           str(self.temp / "bad.out"), "--legacy-mkn-order"],
                          timeout=120)
        bad_record = cases.parse_driver_stdout(bad["stdout"])["record"]
        self.assertEqual(bad_record["call_shape"], {"M": M, "N": K, "K": N})
        bad_output = cases.read_output_file(self.temp / "bad.out", M, N)
        self.assertGreater(float(np.max(np.abs(bad_output.astype(np.float64)
                                              - expected))), 1e-3)

    def test_guard_band_catches_an_out_of_bounds_write(self):
        kernel = self.temp / "oob_kernel"
        kernel.mkdir(parents=True, exist_ok=True)
        (kernel / "cpu_kernel.c").write_bytes(
            (controls.CONTROL_SOURCES / "cpu_oob_write.c").read_bytes())
        (kernel / "kernel.h").write_bytes(
            (support.CPU_SKELETON / "kernel.h").read_bytes())
        built = build.build_eval_driver(kernel, self.temp / "oob_build",
                                        backend="cpu", toolchain=self.toolchain)
        self.assertTrue(built.ok, built.record.get("error"))
        case = data.make_case("uniform_small.s00", 8, 8, 8)
        case_path = self.temp / "oob.case"
        cases.write_case_file(case_path, case)
        result = run_command([str(built.binary), "--case", str(case_path),
                              "--mode", "compute", "--out",
                              str(self.temp / "oob.out"), "--guard"], timeout=120)
        self.assertEqual(result["returncode"], 5)
        record = cases.parse_driver_stdout(result["stdout"])["record"]
        self.assertTrue(record["guard_corrupted"])

    def test_clean_kernel_does_not_trip_the_guard(self):
        case = data.make_case("uniform_small.s00", 8, 8, 8)
        case_path = self.temp / "clean.case"
        cases.write_case_file(case_path, case)
        result = run_command([str(self.build.binary), "--case", str(case_path),
                              "--mode", "compute", "--out",
                              str(self.temp / "clean.out"), "--guard"], timeout=120)
        self.assertEqual(result["returncode"], 0)
        record = cases.parse_driver_stdout(result["stdout"])["record"]
        self.assertFalse(record["guard_corrupted"])


class CaseHeaderTest(unittest.TestCase):
    def test_magic_is_exactly_eight_bytes(self):
        self.assertEqual(len(cases.CASE_MAGIC), 8)
        self.assertTrue(cases.CASE_MAGIC_TEXT.startswith(cases.CASE_MAGIC))

    def test_wrong_size_output_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "short.out"
            path.write_bytes(b"\x00" * 12)
            with self.assertRaises(ValueError):
                cases.read_output_file(path, 4, 4)

    def test_missing_record_is_reported(self):
        self.assertFalse(cases.parse_driver_stdout("nothing here")["found"])
        self.assertFalse(cases.parse_driver_stdout('SCOPE_EVAL {broken')["found"])


if __name__ == "__main__":
    unittest.main()
