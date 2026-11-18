"""Statistics: intervals, small-n rules, paired resampling, multiplicity."""

from __future__ import annotations

import math
import unittest

import numpy as np

from SCOPE.experiments import stats as stats


class WilsonTest(unittest.TestCase):
    def test_known_value_for_ten_of_ten(self):
        low, high = stats.wilson_interval(10, 10)
        self.assertAlmostEqual(low, 0.7225, places=3)
        self.assertEqual(high, 1.0)

    def test_known_value_for_half(self):
        low, high = stats.wilson_interval(5, 10)
        self.assertLess(low, 0.5)
        self.assertGreater(high, 0.5)
        self.assertAlmostEqual(low, 0.2366, places=3)

    def test_zero_of_total_is_bounded_above(self):
        low, high = stats.wilson_interval(0, 10)
        self.assertEqual(low, 0.0)
        self.assertLess(high, 0.35)

    def test_empty_total_has_no_interval(self):
        self.assertIsNone(stats.wilson_interval(0, 0))

    def test_rate_summary_reports_complete_queue_range(self):
        record = stats.rate_summary(6, 10, unresolved=2)
        self.assertEqual(record["complete_queue_range"], [0.6, 0.8])
        self.assertIn("conditional", record["interval_scope"])
        self.assertIsNotNone(record["wilson_lower"])

    def test_ten_of_ten_carries_the_reliability_caveat(self):
        record = stats.rate_summary(10, 10)
        self.assertIn("not evidence of 99%", record["note"])


class DescribeTest(unittest.TestCase):
    def test_n_zero_reports_no_mean(self):
        record = stats.describe([])
        self.assertEqual(record["n"], 0)
        self.assertIsNone(record["mean"])
        self.assertIsNone(record["ci_low"])
        self.assertIn("n=0", record["note"])

    def test_n_one_reports_no_spread(self):
        record = stats.describe([4.0])
        self.assertEqual(record["n"], 1)
        self.assertEqual(record["mean"], 4.0)
        self.assertIsNone(record["sd"])
        self.assertIsNone(record["ci_low"])
        self.assertIn("n=1", record["note"])

    def test_mean_sd_median_cv(self):
        record = stats.describe([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(record["mean"], 2.5)
        self.assertEqual(record["median"], 2.5)
        self.assertAlmostEqual(record["sd"], float(np.std([1, 2, 3, 4], ddof=1)))
        self.assertAlmostEqual(record["cv"], record["sd"] / 2.5)
        self.assertLess(record["ci_low"], record["mean"])
        self.assertGreater(record["ci_high"], record["mean"])

    def test_non_finite_values_are_dropped_not_averaged(self):
        record = stats.describe([1.0, float("nan"), float("inf"), 3.0])
        self.assertEqual(record["n"], 2)
        self.assertEqual(record["mean"], 2.0)

    def test_bootstrap_is_deterministic(self):
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        first = stats.bootstrap_mean_interval(values, resamples=500, seed=11)
        second = stats.bootstrap_mean_interval(values, resamples=500, seed=11)
        third = stats.bootstrap_mean_interval(values, resamples=500, seed=12)
        self.assertEqual(first, second)
        self.assertNotEqual(first, third)


class PairedTest(unittest.TestCase):
    def test_paired_differences_removed_shared_variance(self):
        differences = [0.5, 0.6, 0.4, 0.55, 0.45, 0.5]
        record = stats.paired_bootstrap(differences, resamples=1000, seed=3)
        self.assertEqual(record["n"], 6)
        self.assertAlmostEqual(record["mean_difference"], 0.5, places=6)
        self.assertLess(record["ci_low"], 0.5)
        self.assertGreater(record["ci_high"], 0.5)

    def test_single_pair_has_no_interval(self):
        record = stats.paired_bootstrap([0.3])
        self.assertEqual(record["n"], 1)
        self.assertIsNone(record["ci_low"])


class EffectiveThroughputTest(unittest.TestCase):
    def test_undelivered_tasks_contribute_zero(self):
        record = stats.effective_throughput([True, False, True],
                                            [10.0, 99.0, 30.0])
        self.assertEqual(record["n"], 3)
        self.assertAlmostEqual(record["mean"], (10.0 + 0.0 + 30.0) / 3)
        self.assertEqual(record["undelivered"], 1)
        self.assertIn("per generation attempt", record["interpretation"])

    def test_unknown_delivery_is_not_a_silent_zero_mean(self):
        record = stats.effective_throughput([True, None], [10.0, None])
        self.assertEqual(record["unknown_throughput"], 1)


class MultiplicityTest(unittest.TestCase):
    def test_holm_orders_and_adjusts(self):
        adjusted = stats.holm_correction([0.001, 0.02, 0.5])
        self.assertTrue(adjusted[0]["reject_null"])
        self.assertEqual(adjusted[0]["rank"], 1)
        self.assertAlmostEqual(adjusted[0]["holm_threshold"], 0.05 / 3)
        self.assertFalse(adjusted[2]["reject_null"])

    def test_rate_difference_reports_both_intervals(self):
        record = stats.rate_difference(8, 10, 5, 10)
        self.assertAlmostEqual(record["difference"], 0.3)
        self.assertIsNotNone(record["a"]["wilson_lower"])
        self.assertIn("not an equivalence test", record["note"])


class NumericSanityTest(unittest.TestCase):
    def test_z_quantile_matches_the_normal_table(self):
        # P(|Z| < 1.96) ~ 0.95; the helper is used for every Wilson interval.
        z = stats._z_for_alpha(0.05)
        self.assertTrue(math.isclose(z, 1.95996, rel_tol=1e-3), z)


if __name__ == "__main__":
    unittest.main()
