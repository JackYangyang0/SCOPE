"""Evaluator behaviour: full-output revalidation, taxonomy, byte-drift guard."""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from SCOPE.verification import driver_build as build
from SCOPE.verification import controls as controls
from SCOPE.data import matrix_cases as data
from SCOPE.verification import reference_fp64 as reference
from SCOPE.verification import full_output_verify as verify

import support

SHAPE = 24


@unittest.skipIf(support.cpu_toolchain() is None, "no CPU C compiler available")
class EvaluatorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = Path(tempfile.mkdtemp(prefix="scope-exp-verify-"))
        cls.toolchain = support.cpu_toolchain()
        cls.kernel = cls.temp / "kernel"
        cls.kernel.mkdir(parents=True)
        (cls.kernel / "cpu_kernel.c").write_bytes(
            (controls.CONTROL_SOURCES / "cpu_reference_fp64.c").read_bytes())
        (cls.kernel / "kernel.h").write_bytes(
            (support.CPU_SKELETON / "kernel.h").read_bytes())
        cls.build = build.build_eval_driver(cls.kernel, cls.temp / "build",
                                            backend="cpu", toolchain=cls.toolchain)
        assert cls.build.ok, cls.build.record.get("error")
        cls.manifest = data.build_manifest([SHAPE], split="development")
        cls.cache = reference.ReferenceCache(cls.temp / "references")

    def _spec(self, **overrides):
        defaults = dict(kernel_dir=self.kernel, driver=self.build.binary,
                        build_record=self.build.record, backend="cpu",
                        shape=SHAPE, split="development", timeout_seconds=120.0,
                        repeat_checks=2, hypotheses_max_shape=SHAPE,
                        task_id="test", manifest=self.manifest)
        defaults.update(overrides)
        return verify.EvaluationSpec(**defaults)

    def test_reference_kernel_passes_all_checks(self):
        spec = self._spec()
        record = verify.evaluate_kernel(spec, self.cache,
                                        scratch_dir=self.temp / "eval-pass")
        summary = record["summary"]
        self.assertEqual(summary["verdict"], "pass")
        self.assertEqual(summary["main"]["planned"], 64)
        self.assertEqual(summary["main"]["passed"], 64)
        self.assertEqual(summary["coefficients"]["planned"], 128)
        self.assertEqual(summary["coefficients"]["passed"], 128)
        self.assertTrue(summary["delivery"])
        self.assertEqual(len(record["rows"]), 192)
        # every policy is reported separately
        row = record["rows"][0]
        self.assertEqual(row["main_policy"], "pass")
        self.assertEqual(row["online_gate"], "pass")
        self.assertIn(row["strict_abs"], ("pass", "fail"))
        self.assertEqual(summary["determinism"]["repeats"], 2)

    def test_output_files_are_written(self):
        spec = self._spec()
        record = verify.evaluate_kernel(spec, self.cache,
                                        scratch_dir=self.temp / "eval-write")
        paths = verify.write_validation(self.temp / "out", record,
                                        validation_csv=self.temp / "out" / "validation.csv",
                                        coefficient_csv=self.temp / "out" / "coeff.csv",
                                        validation_json=self.temp / "out" / "validation.json")
        self.assertTrue(paths["validation_csv"].exists())
        self.assertTrue(paths["coefficient_csv"].exists())
        lines = paths["validation_csv"].read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 64 + 1)     # main rows only, plus header

    def test_wrong_kernel_fails_with_a_named_defect(self):
        kernel = self.temp / "short_k"
        kernel.mkdir(parents=True, exist_ok=True)
        (kernel / "cpu_kernel.c").write_bytes(
            (controls.CONTROL_SOURCES / "cpu_short_k.c").read_bytes())
        (kernel / "kernel.h").write_bytes(
            (support.CPU_SKELETON / "kernel.h").read_bytes())
        built = build.build_eval_driver(kernel, self.temp / "short_build",
                                        backend="cpu", toolchain=self.toolchain)
        self.assertTrue(built.ok, built.record.get("error"))
        record = verify.evaluate_kernel(
            self._spec(kernel_dir=kernel, driver=built.binary,
                       build_record=built.record),
            self.cache, scratch_dir=self.temp / "eval-fail")
        summary = record["summary"]
        self.assertEqual(summary["verdict"], "fail")
        self.assertGreater(summary["main"]["failed"], 0)
        diagnoses = {row["diagnosis"]["primary"] for row in record["rows"]
                     if row["diagnosis"]}
        self.assertIn("missing_last_k_term", diagnoses)

    def test_non_finite_output_is_classified(self):
        kernel = self.temp / "nan_k"
        kernel.mkdir(parents=True, exist_ok=True)
        (kernel / "cpu_kernel.c").write_bytes(
            (controls.CONTROL_SOURCES / "cpu_nan_output.c").read_bytes())
        (kernel / "kernel.h").write_bytes(
            (support.CPU_SKELETON / "kernel.h").read_bytes())
        built = build.build_eval_driver(kernel, self.temp / "nan_build",
                                        backend="cpu", toolchain=self.toolchain)
        record = verify.evaluate_kernel(
            self._spec(kernel_dir=kernel, driver=built.binary,
                       build_record=built.record),
            self.cache, scratch_dir=self.temp / "eval-nan")
        row = next(item for item in record["rows"] if item["group"] == "main")
        self.assertGreater(row["non_finite_count"], 0)
        self.assertEqual(row["diagnosis"]["primary"], "non_finite_output")

    def test_byte_drift_is_refused_not_evaluated(self):
        manifest = copy.deepcopy(self.manifest)
        manifest["entries"][0]["hashes"]["A"] = "0" * 64
        spec = self._spec(manifest=manifest, include_coefficients=False,
                          repeat_checks=0)
        record = verify.evaluate_kernel(spec, self.cache,
                                        scratch_dir=self.temp / "eval-drift")
        self.assertEqual(record["rows"][0]["status"], "input_byte_drift")
        self.assertTrue(record["rows"][0]["main_policy"] != "pass")
        self.assertEqual(record["summary"]["verdict"], "incomplete")

    def test_case_files_are_not_left_behind_by_default(self):
        spec = self._spec(include_coefficients=False, repeat_checks=0,
                          manifest=data.build_manifest([SHAPE], split="development",
                                                       include_coefficients=False))
        verify.evaluate_kernel(spec, self.cache, scratch_dir=self.temp / "eval-clean")
        work = self.temp / "eval-clean" / "eval"
        leftovers = list(work.glob("*.case")) if work.exists() else []
        self.assertEqual(leftovers, [])


if __name__ == "__main__":
    unittest.main()
