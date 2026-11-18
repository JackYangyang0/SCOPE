"""Reporting from raw records: denominators, intervals, pairing, unknown cost.

The records here are synthetic on purpose: the summariser must be testable
without a GPU, and the arithmetic is the thing under test.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from SCOPE.experiments import summarize as summarize
from SCOPE.utils.harness_util import write_json

import support


def _write_status(campaign, task_id, *, block, kind, shape, status, delivered,
                  gflops=None, validation_verdict=None, requests=None,
                  cost_usd=None, cost_complete=None, variant=None,
                  model_role=None, safety="pass"):
    directory = campaign.task_dir(task_id)
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "task_schema": "test",
        "task_id": task_id,
        "block": block,
        "kind": kind,
        "shape": shape,
        "status": status,
        "provenance": "formal",
        "variant": variant,
        "model_role": model_role,
        "outcome": {"kind": "resolved" if delivered else "science_failure",
                    "delivered": delivered},
        "validation": ({"verdict": validation_verdict,
                        "main": {"planned": 64, "passed": 64 if validation_verdict == "pass" else 60,
                                 "failed": 0 if validation_verdict == "pass" else 4},
                        "coefficients": {"planned": 128,
                                         "passed": 128 if validation_verdict == "pass" else 120},
                        "strict_abs_total_pass": 40,
                        "binary_sha256": "a" * 64}
                       if validation_verdict else {}),
        "timing": ({"gflops_mean": gflops, "gflops_median": gflops,
                    "gflops_sd": 1.0, "n_sessions": 10,
                    "binary_sha256": "a" * 64} if gflops is not None else {}),
        "sanitizer": {"status": safety},
        "cost": ({"requests": requests, "cost_usd_known": cost_usd,
                  "cost_is_complete": cost_complete,
                  "tokens_known": {"prompt_tokens": 100, "completion_tokens": 50}}
                 if requests is not None else {}),
        "attempts": [{"attempt_id": "a1", "wall_seconds": 100.0}],
        "resolved_config_hash": "c" * 64,
        "delivered_phase": "p3" if delivered else None,
    }
    write_json(directory / "status.json", payload)
    return payload


class SummarizeTest(unittest.TestCase):
    def setUp(self):
        self.campaign = support.temp_campaign("summary")
        # variant A: three complete deliveries, one numerical failure
        for index in range(1, 4):
            _write_status(self.campaign, f"E1-full-512-r{index:02d}", block="E1",
                          kind="independent", shape=512, status="resolved",
                          delivered=True, gflops=10.0 + index,
                          validation_verdict="pass", requests=5, cost_usd=0.5,
                          cost_complete=True, variant="full")
        _write_status(self.campaign, "E1-full-512-r04", block="E1",
                      kind="independent", shape=512, status="resolved",
                      delivered=False, validation_verdict="fail", requests=5,
                      cost_usd=0.5, cost_complete=True, variant="full")
        # variant B: one delivery, one infrastructure unknown, one missing
        _write_status(self.campaign, "E1-minus_state_in_prompt-512-r01", block="E1",
                      kind="independent", shape=512, status="resolved",
                      delivered=True, gflops=8.0, validation_verdict="pass",
                      requests=5, cost_usd=0.5, cost_complete=True,
                      variant="minus_state_in_prompt")
        _write_status(self.campaign, "E1-minus_state_in_prompt-512-r02", block="E1",
                      kind="independent", shape=512, status="infra_incomplete",
                      delivered=False, validation_verdict=None, requests=None,
                      variant="minus_state_in_prompt")
        # a branch pair for the full parent r01
        _write_status(self.campaign, "E1-woFB-512-r01", block="E1", kind="branch",
                      shape=512, status="resolved", delivered=True,
                      gflops=9.0, validation_verdict="pass", requests=2,
                      cost_usd=0.2, cost_complete=True,
                      variant="minus_feedback_optimization")
        # a rehearsal row that must be excluded by default
        _write_rehearsal(self.campaign)

    def test_rows_exclude_rehearsal_by_default(self):
        rows = summarize.task_rows(self.campaign)
        self.assertTrue(all(row["provenance"] != "rehearsal" for row in rows))
        included = summarize.task_rows(self.campaign, include_rehearsal=True)
        self.assertEqual(len(included), len(rows) + 1)

    def test_component_table_denominators_and_intervals(self):
        summary = summarize.summarize_campaign(self.campaign)
        table = {(row["variant"], row["shape"]): row
                 for row in summary["tables"]["rq2_components"]}
        full = table[("full", 512)]
        self.assertEqual(full["n_planned"], 4)
        self.assertEqual(full["n_complete"], 3)
        self.assertAlmostEqual(full["rate_complete"], 0.75)
        self.assertLess(full["rate_complete_lower"], 0.75)
        self.assertGreater(full["rate_complete_upper"], 0.75)
        self.assertEqual(full["n_science_failure"], 1)
        self.assertAlmostEqual(full["throughput_mean"], 12.0)   # 11, 12, 13
        self.assertEqual(full["throughput_n"], 3)

        partial = table[("minus_state_in_prompt", 512)]
        self.assertEqual(partial["n_planned"], 2)
        self.assertEqual(partial["n_infra_unknown"], 1)
        self.assertEqual(partial["complete_queue_range"], [0.5, 1.0])
        self.assertIn("conditional", partial["rate_interval_scope"])

    def test_unresolved_task_is_not_counted_as_success(self):
        summary = summarize.summarize_campaign(self.campaign)
        table = {(row["variant"], row["shape"]): row
                 for row in summary["tables"]["rq2_components"]}
        partial = table[("minus_state_in_prompt", 512)]
        self.assertEqual(partial["n_complete"], 1)
        self.assertEqual(partial["n_resolved"], 1)

    def test_stage_pair_uses_the_parent(self):
        summary = summarize.summarize_campaign(self.campaign)
        pairs = [row for row in summary["tables"]["stage_pairs"]
                 if row["shape"] == "512"
                 and row["stage"] == "without_feedback"]
        self.assertEqual(len(pairs), 1)
        pair = pairs[0]
        self.assertEqual(pair["n_pairs_resolved"], 1)
        self.assertAlmostEqual(pair["mean_throughput_difference"], 9.0 - 11.0)
        self.assertEqual(pair["independent_n"], 0)

    def test_effective_throughput_counts_undelivered_as_zero(self):
        summary = summarize.summarize_campaign(self.campaign)
        table = {(row["variant"], row["shape"]): row
                 for row in summary["tables"]["rq2_components"]}
        full = table[("full", 512)]
        self.assertAlmostEqual(full["effective_throughput_mean"], (11 + 12 + 13) / 4)
        self.assertIn("per generation attempt", full["effective_throughput"]["interpretation"])

    def test_cost_table_keeps_unknown_distinct(self):
        summary = summarize.summarize_campaign(self.campaign)
        rows = {row["task_id"]: row for row in summary["tables"]["cost"]}
        self.assertIsNone(rows["E1-minus_state_in_prompt-512-r02"]["requests"])
        self.assertTrue(rows["E1-full-512-r01"]["cost_complete"])

    def test_markdown_and_csv_are_written(self):
        summary = summarize.summarize_campaign(self.campaign)
        outputs = summarize.write_reports(self.campaign, summary)
        self.assertTrue(outputs["RESULTS_zh.md"].exists())
        text = outputs["RESULTS_zh.md"].read_text(encoding="utf-8")
        self.assertIn("RQ2", text)
        self.assertIn("Wilson", text)
        self.assertIn("3/4", text)
        self.assertTrue((self.campaign.analysis / "rq2_components.csv").exists())

    def test_zero_rows_does_not_crash(self):
        campaign = support.temp_campaign("summary-empty")
        summary = summarize.summarize_campaign(campaign)
        self.assertEqual(summary["counts"]["rows"], 0)
        outputs = summarize.write_reports(campaign, summary)
        self.assertTrue(outputs["RESULTS_zh.md"].exists())


def _write_rehearsal(campaign) -> None:
    directory = campaign.task_dir("REH-64-r01")
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "status.json", {
        "task_id": "REH-64-r01", "block": "REHEARSAL", "kind": "independent",
        "shape": 64, "status": "resolved", "provenance": "rehearsal",
        "outcome": {"kind": "resolved", "delivered": True},
        "validation": {"verdict": "pass", "main": {"planned": 64, "passed": 64}},
        "timing": {"gflops_mean": 4.0},
        "sanitizer": {"status": "not_applicable"},
        "attempts": [], "resolved_config_hash": "d" * 64,
    })


if __name__ == "__main__":
    unittest.main()
