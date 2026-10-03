import unittest

from SCOPE.tune_cuda import aggregate_value


class TuneCudaAggregationTests(unittest.TestCase):
    def test_mean_uses_all_measured_runs(self):
        ir = {"performance": {
            "gflops": 110.0,
            "gflops_mean": 100.0,
            "gflops_trimmed_mean": 110.0,
        }}
        self.assertEqual(aggregate_value(ir, "gflops", "mean"), 100.0)
        self.assertEqual(aggregate_value(ir, "gflops", "trimmed_mean"), 110.0)

    def test_missing_requested_statistic_falls_back_to_primary_value(self):
        self.assertEqual(
            aggregate_value({"performance": {"latency_ms": 2.5}}, "latency_ms", "mean"),
            2.5,
        )


if __name__ == "__main__":
    unittest.main()
