from __future__ import annotations

from pathlib import Path
import sys
import time
import unittest


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from SCOPE.verification.build_run_verifier import aggregate_metrics, run_command, trim_extremes


class BuildRunVerifierTests(unittest.TestCase):
    def test_trimmed_mean_removes_one_minimum_and_maximum(self):
        self.assertEqual(trim_extremes([1.0, 2.0, 3.0, 4.0, 100.0]), [2.0, 3.0, 4.0])
        metrics = aggregate_metrics([{"gflops": value} for value in [1, 2, 3, 4, 100]])
        self.assertEqual(metrics["gflops_trimmed_mean"], 3.0)
        self.assertEqual(metrics["gflops"], 3.0)

    def test_error_metrics_keep_worst_observed_run(self):
        metrics = aggregate_metrics([
            {"max_abs_error": 1.0e-5, "max_rel_error": 2.0e-4, "relative_l2_error": 3.0e-5},
            {"max_abs_error": 4.0e-5, "max_rel_error": 1.0e-4, "relative_l2_error": 8.0e-5},
        ])
        self.assertEqual(metrics["max_abs_error"], 4.0e-5)
        self.assertEqual(metrics["max_rel_error"], 2.0e-4)
        self.assertEqual(metrics["relative_l2_error"], 8.0e-5)

    def test_timeout_returns_without_waiting_for_command_completion(self):
        started = time.monotonic()

        result = run_command(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            cwd=ROOT,
            timeout_seconds=0.1,
        )

        self.assertEqual(result["status"], "fail")
        self.assertTrue(result["timeout"])
        self.assertIn("process tree terminated", result["stderr"])
        self.assertLess(time.monotonic() - started, 5)


if __name__ == "__main__":
    unittest.main()
