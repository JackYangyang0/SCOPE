"""FP64 reference: agreement across constructions, caching and validity."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from SCOPE.data import matrix_cases as data
from SCOPE.verification import reference_fp64 as reference


class ConstructionTest(unittest.TestCase):
    def test_blas_and_compensated_agree_on_small_matrices(self):
        case = data.make_case("uniform_small.s03", 16, 16, 16)
        blas, _ = reference.cpu_blas_fp64(case["A"], case["B"])
        compensated = reference.compensated_fp64(case["A"], case["B"])
        self.assertLess(float(np.max(np.abs(blas - compensated))), 1e-12)

    def test_compensated_dot_matches_numpy_in_float64(self):
        rng = np.random.default_rng(7)
        a = rng.normal(size=64)
        b = rng.normal(size=64)
        self.assertAlmostEqual(reference.compensated_dot(a, b), float(a @ b),
                               places=12)

    def test_compensated_refuses_large_shape(self):
        case = data.make_case("uniform_small.s00", 128, 128, 128)
        with self.assertRaises(ValueError):
            reference.compensated_fp64(case["A"], case["B"])


class AnalyticTest(unittest.TestCase):
    def test_structural_families_match_closed_form(self):
        for family in ("zero", "left_identity", "right_identity", "impulse",
                       "cancellation"):
            case = data.make_case(family, 12, 16, 10)
            analytic = reference.analytic_output(case)
            self.assertIsNotNone(analytic, family)
            computed, _ = reference.cpu_blas_fp64(case["A"], case["B"],
                                                  case["alpha"], case["beta"],
                                                  case["C0"])
            self.assertLess(float(np.max(np.abs(computed - analytic))), 1e-9,
                            family)

    def test_non_square_identity_reference(self):
        """A rectangular identity catches a transposed M/N adapter (§5.4)."""
        case = data.make_case("left_identity", 127, 129, 65)
        computed, _ = reference.cpu_blas_fp64(case["A"], case["B"])
        expected = reference.analytic_output(case)
        self.assertIsNotNone(expected)
        self.assertTrue(np.allclose(computed, expected))
        # rows 65..126 must be exactly zero: A has fewer rows than M
        self.assertFalse(np.any(computed[65:]))

    def test_random_family_has_no_closed_form(self):
        case = data.make_case("uniform_small.s00", 8, 8, 8)
        self.assertIsNone(reference.analytic_output(case))


class ValidityTest(unittest.TestCase):
    def test_build_reference_records_evidence(self):
        case = data.make_case("uniform_wide.s01", 12, 12, 12)
        built = reference.build_reference(case)
        self.assertTrue(built.meta["valid"])
        self.assertEqual(built.meta["kind"], "cpu_blas_fp64")
        self.assertIn("compensated_point_check", built.meta["validity"])
        self.assertEqual(len(built.sha256), 64)

    def test_cross_check_agrees(self):
        case = data.make_case("normal.s02", 10, 14, 9)
        built = reference.build_reference(case)
        result = reference.cross_check_reference(case, built)
        self.assertEqual(result["status"], "agree")

    def test_coefficients_are_applied_in_float64(self):
        case = data.make_case("uniform_small.s02", 8, 8, 8,
                              coefficients={"name": "t", "alpha": -0.75, "beta": 0.5})
        computed, product = reference.cpu_blas_fp64(case["A"], case["B"],
                                                    case["alpha"], case["beta"],
                                                    case["C0"])
        expected = -0.75 * product + 0.5 * case["C0"].astype(np.float64)
        self.assertTrue(np.allclose(computed, expected))


class CacheTest(unittest.TestCase):
    def test_second_lookup_is_the_same_reference(self):
        case = data.make_case("uniform_small.s04", 10, 10, 10)
        with tempfile.TemporaryDirectory() as directory:
            cache = reference.ReferenceCache(Path(directory))
            first = cache.get(case)
            second = cache.get(case)
            self.assertEqual(first.sha256, second.sha256)
            self.assertTrue((Path(directory) / cache.key(case) / "reference.meta.json")
                            .exists())

    def test_different_coefficients_use_a_different_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = reference.ReferenceCache(Path(directory))
            unit = data.make_case("uniform_small.s04", 8, 8, 8,
                                  coefficients=data.MAIN_COEFFICIENTS)
            beta = data.make_case("uniform_small.s04", 8, 8, 8,
                                  coefficients=data.COEFFICIENT_SETS[0])
            self.assertNotEqual(cache.key(unit), cache.key(beta))
            self.assertNotEqual(cache.get(unit).sha256, cache.get(beta).sha256)

    def test_point_check_finds_layout_error(self):
        case = data.make_case("uniform_small.s05", 12, 12, 12)
        positions = reference.point_check_indices(12, 12, 8)
        correct = reference.point_check(case["A"], case["B"], positions)
        # Reading A as column-major is exactly the negative control's defect.
        wrong = reference.point_check(case["A"].T, case["B"], positions)
        self.assertNotEqual(correct["values"], wrong["values"])


if __name__ == "__main__":
    unittest.main()
