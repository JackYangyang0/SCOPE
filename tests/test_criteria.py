"""Named numerical policies: thresholds, separation, and failure taxonomy."""

from __future__ import annotations

import unittest

import numpy as np

from SCOPE.verification import criteria as criteria


class MetricsTest(unittest.TestCase):
    def test_identical_arrays_pass_every_policy(self):
        array = np.array([[1.0, -2.0], [3.5, 0.0]])
        metrics = criteria.output_metrics(array, array)
        self.assertEqual(metrics["max_abs_error"], 0.0)
        self.assertEqual(metrics["elementwise_bad_count"], 0)
        self.assertEqual(metrics["non_finite_count"], 0)
        self.assertEqual(criteria.apply_main_policy(metrics)["status"], "pass")
        self.assertEqual(criteria.apply_online_gate(metrics)["status"], "pass")
        self.assertEqual(criteria.apply_strict_abs(metrics)["status"], "pass")

    def test_mixed_tolerance_boundary(self):
        reference = np.array([[1000.0]])
        allowed = criteria.ATOL + criteria.RTOL * 1000.0
        inside = np.array([[1000.0 + allowed * 0.5]])
        outside = np.array([[1000.0 + allowed * 1.5]])
        self.assertEqual(criteria.apply_main_policy(
            criteria.output_metrics(inside, reference))["status"], "pass")
        self.assertEqual(criteria.apply_main_policy(
            criteria.output_metrics(outside, reference))["status"], "fail")

    def test_absolute_floor_decides_near_zero(self):
        """A near-zero reference is judged by atol, not by rtol."""
        reference = np.array([[1e-9]])
        caught = np.array([[2e-5]])
        metrics = criteria.output_metrics(caught, reference)
        self.assertEqual(criteria.apply_main_policy(metrics)["status"], "fail")
        self.assertEqual(metrics["elementwise_bad_count"], 1)

    def test_strict_abs_is_diagnostic_only(self):
        reference = np.array([[1.0]])
        actual = np.array([[1.0 + 5e-6]])   # inside mixed, outside strict
        metrics = criteria.output_metrics(actual, reference)
        self.assertEqual(criteria.apply_main_policy(metrics)["status"], "pass")
        self.assertEqual(criteria.apply_strict_abs(metrics)["status"], "fail")
        self.assertEqual(criteria.apply_strict_abs(metrics)["role"], "diagnostic")

    def test_nan_is_counted_and_never_reported_as_infinite_metric(self):
        reference = np.array([[1.0, 2.0]])
        actual = np.array([[np.nan, 2.0]])
        metrics = criteria.output_metrics(actual, reference)
        self.assertEqual(metrics["non_finite_count"], 1)
        self.assertTrue(np.isfinite(metrics["max_abs_error"]))
        verdict = criteria.apply_main_policy(metrics)
        self.assertEqual(verdict["status"], "fail")
        self.assertTrue(any("non-finite" in reason for reason in verdict["reasons"]))

    def test_inf_is_counted(self):
        metrics = criteria.output_metrics(np.array([[np.inf, 1.0]]),
                                          np.array([[1.0, 1.0]]))
        self.assertEqual(metrics["non_finite_count"], 1)
        self.assertEqual(criteria.apply_main_policy(metrics)["status"], "fail")

    def test_shape_mismatch_is_an_error_not_a_pass(self):
        with self.assertRaises(ValueError):
            criteria.output_metrics(np.zeros((2, 2)), np.zeros((2, 3)))


class PolicyIdentityTest(unittest.TestCase):
    def test_three_policies_stay_distinct(self):
        identifiers = {policy.identifier for policy in criteria.ALL_POLICIES}
        self.assertEqual(identifiers, {criteria.POLICY_ONLINE, criteria.POLICY_MAIN,
                                       criteria.POLICY_STRICT, criteria.POLICY_REFERENCE})
        self.assertEqual(criteria.MAIN.role, criteria.DELIVERY)
        self.assertEqual(criteria.ONLINE_GENERATION_GATE.role, criteria.DIAGNOSTIC)
        self.assertEqual(criteria.STRICT.role, criteria.DIAGNOSTIC)

    def test_all_policies_returns_one_verdict_per_policy(self):
        metrics = criteria.output_metrics(np.zeros((2, 2)), np.zeros((2, 2)))
        verdicts = criteria.all_policies(metrics)
        self.assertEqual(set(verdicts), {criteria.POLICY_MAIN, criteria.POLICY_ONLINE,
                                         criteria.POLICY_STRICT})


class ReferenceVerdictTest(unittest.TestCase):
    def test_verdict_flip_marks_reference_invalid(self):
        agreement = {"max_abs_error": 1e-3, "relative_l2_error": 1e-6,
                     "elementwise_bad_count": 0, "non_finite_count": 0}
        result = criteria.reference_verdict(agreement, {"passed": False},
                                            {"passed": True})
        self.assertEqual(result["status"], "reference_invalid")
        self.assertFalse(result["passed"])

    def test_same_verdict_is_agreement(self):
        agreement = {"max_abs_error": 1e-9, "relative_l2_error": 1e-12,
                     "elementwise_bad_count": 0, "non_finite_count": 0}
        result = criteria.reference_verdict(agreement, {"passed": True},
                                            {"passed": True})
        self.assertEqual(result["status"], "agree")


class DiagnosisTest(unittest.TestCase):
    def setUp(self):
        self.reference = np.arange(16, dtype=np.float64).reshape(4, 4) + 1.0

    def test_transposed_hypothesis_is_named(self):
        diagnosis = criteria.diagnose_mismatch(
            self.reference.T, {"reference": self.reference,
                               "transposed_output": self.reference.T})
        self.assertEqual(diagnosis.primary, "transposed_output")

    def test_output_matching_the_reference_is_a_precision_difference(self):
        actual = self.reference + 1e-2
        diagnosis = criteria.diagnose_mismatch(
            actual, {"reference": self.reference})
        self.assertEqual(diagnosis.primary,
                         "numerical_difference_no_structural_defect")

    def test_all_zero_output_is_named(self):
        diagnosis = criteria.diagnose_mismatch(
            np.zeros_like(self.reference), {"reference": self.reference})
        self.assertEqual(diagnosis.primary, "output_all_zero")

    def test_non_finite_output_is_named_first(self):
        actual = self.reference.copy()
        actual[0, 0] = np.nan
        diagnosis = criteria.diagnose_mismatch(actual, {"reference": self.reference})
        self.assertEqual(diagnosis.primary, "non_finite_output")

    def test_unexplained_when_no_hypothesis_matches(self):
        diagnosis = criteria.diagnose_mismatch(
            np.full_like(self.reference, 1e6), {"reference": self.reference})
        self.assertEqual(diagnosis.primary, "unexplained")


if __name__ == "__main__":
    unittest.main()
