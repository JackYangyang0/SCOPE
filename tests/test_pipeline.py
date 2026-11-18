"""End-to-end pipeline on one task: workspace, checkpoint, validation, timing.

This is the test that would have caught every wiring bug: it runs the whole
lifecycle with the labelled rehearsal generator (no GPU, no API key), then asks
the summariser and the audit whether the archive is self-consistent.
"""

from __future__ import annotations

import types
import unittest
from pathlib import Path

from SCOPE.experiments import audit as audit
from SCOPE.experiments import checkpoints as checkpoints
from SCOPE.data import splits as datasets
from SCOPE.experiments import runner as runner
from SCOPE.experiments import summarize as summarize
from SCOPE.utils.harness_util import read_json, write_json

import support

SHAPE = 32


@unittest.skipIf(support.cpu_toolchain() is None, "no CPU C compiler available")
class PipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.campaign = support.temp_campaign("pipeline")
        plan = datasets.DatasetPlan(shapes=[SHAPE],
                                    splits=("development", "generation",
                                            "postselection_test"),
                                    include_coefficients=False)
        datasets.write_datasets(cls.campaign, plan, jobs=1)
        cls.protocol = support.tiny_protocol(cls.campaign, shape=SHAPE)
        cls.protocol["data"]["manifest_hashes"] = {
            split: read_json(cls.campaign.dataset_manifest(split))["manifest_hash"]
            for split in ("development", "generation", "postselection_test")
        }
        # A stub gate: the real gate is exercised by `SCOPE.exp_app controls`.
        write_json(cls.campaign.audit / "controls_gate.json",
                   {"ok": True, "backend": "cpu", "created_at": "test",
                    "checks": [], "problems": []})
        cls.options = runner.RunOptions(
            campaign_id=cls.campaign.campaign_id, backend="cpu",
            generator="rehearsal", mode="full", provenance="rehearsal",
            skip_sanitizer=False, time_kernel=True)

    def test_1_single_task_lifecycle(self):
        status = runner.run_task(self.campaign, self.protocol, "T-32-r01", self.options)
        self.assertEqual(status["status"], "resolved")
        self.assertTrue(status["outcome"]["delivered"])
        self.assertEqual(status["validation"]["verdict"], "pass")
        self.assertEqual(status["validation"]["main"]["passed"], 64)
        self.assertEqual(status["timing"]["n_sessions"], 10)
        self.assertEqual(status["timing"]["n_launches"], 500)
        self.assertEqual(status["timing"]["binary_sha256"],
                         status["validation"]["binary_sha256"])
        self.assertEqual(len(status["attempts"]), 1)

    def test_2_records_exist(self):
        task_dir = self.campaign.task_dir("T-32-r01")
        for name in ("status.json", "selected.json", "validation.csv",
                     "validation.json", "timing.csv", "timing.json", "build.json",
                     "resolved_config_redacted.json"):
            self.assertTrue((task_dir / name).exists(), name)
        self.assertEqual(len(list(task_dir.glob("checkpoints/p1/*/manifest.json"))), 1)
        # scratch case files must not be archived
        self.assertEqual(list((task_dir / "scratch").rglob("*.case"))
                         if (task_dir / "scratch").exists() else [], [])

    def test_3_selected_top1_is_frozen_before_validation(self):
        selected = read_json(self.campaign.task_selected("T-32-r01"))
        self.assertEqual(selected["delivered"]["delivered_phase"], "p1")
        self.assertTrue(selected["delivered"]["no_reselection_after_validation"])
        validation = read_json(self.campaign.task_validation_json("T-32-r01"))
        self.assertLessEqual(selected["frozen_at"], validation["started_at"])

    def test_4_checkpoint_kernel_is_immutable_copy(self):
        manifests = list(self.campaign.task_dir("T-32-r01")
                         .glob("checkpoints/p1/*/manifest.json"))
        manifest = read_json(manifests[0])
        kernel_dir = manifests[0].parent / "kernel"
        self.assertTrue((kernel_dir / "cpu_kernel.c").exists())
        self.assertEqual(manifest["kernel"]["files"]["cpu_kernel.c"],
                         __import__("SCOPE.utils.harness_util", fromlist=["sha256_file"])
                         .sha256_file(kernel_dir / "cpu_kernel.c"))
        self.assertEqual(manifest["provenance"]["config_hash"],
                         read_json(self.campaign.task_status("T-32-r01"))
                         ["resolved_config_hash"])

    def test_5_summary_reports_the_task(self):
        summary = summarize.summarize_campaign(self.campaign, include_rehearsal=True,
                                               bootstrap_resamples=200)
        self.assertEqual(summary["counts"]["rows"], 1)
        self.assertEqual(summary["counts"]["by_status"], {"resolved": 1})
        table = summary["tables"]["completeness"]
        campaign_row = next(row for row in table if row["scope"] == "campaign")
        self.assertEqual(campaign_row["n_planned"], 1)
        self.assertEqual(campaign_row["n_resolved"], 1)

    def test_6_audit_accepts_the_archive(self):
        summary = summarize.summarize_campaign(self.campaign, include_rehearsal=True,
                                               bootstrap_resamples=200)
        summarize.write_reports(self.campaign, summary)
        report = audit.run(self.campaign, protocol=self.protocol)
        self.assertTrue(report["ok"], report["problems"])
        self.assertTrue(report["sections"]["identity"]["ok"])
        self.assertTrue(report["sections"]["no_reselection"]["ok"])
        self.assertTrue(report["sections"]["redaction"]["ok"])
        self.assertTrue((self.campaign.audit / "index.csv").exists())
        self.assertTrue((self.campaign.audit / "claims.json").exists())

    def test_7_second_run_is_idempotent(self):
        before = read_json(self.campaign.task_status("T-32-r01"))
        after = runner.run_task(self.campaign, self.protocol, "T-32-r01", self.options)
        self.assertEqual(before["updated_at"], after["updated_at"])
        self.assertEqual(len(after["attempts"]), 1)

    def test_8_rehearsal_rows_are_excluded_from_formal_analysis(self):
        summary = summarize.summarize_campaign(self.campaign, include_rehearsal=False)
        self.assertEqual(summary["counts"]["rows"], 0)


class GenerationFailureClassificationTest(unittest.TestCase):
    """An API outage must not be reported as the method finding nothing."""

    def test_all_failed_requests_are_infrastructure(self):
        campaign = support.temp_campaign("classify")
        attempt = "a1"
        campaign.attempt_ledger("T-1", attempt).parent.mkdir(parents=True, exist_ok=True)
        from SCOPE.utils.harness_util import append_jsonl
        for index in range(3):
            append_jsonl(campaign.attempt_ledger("T-1", attempt),
                         {"request_id": f"r{index}", "failed": True,
                          "call_type": "codegen"})
        self.assertIsNotNone(runner._generation_failure_category(
            campaign, "T-1", attempt, {"exit_code": 0,
                                       "stdout_tail": "", "stderr_tail": ""}))

    def test_one_success_means_the_method_had_a_chance(self):
        campaign = support.temp_campaign("classify-ok")
        attempt = "a1"
        from SCOPE.utils.harness_util import append_jsonl
        campaign.attempt_ledger("T-1", attempt).parent.mkdir(parents=True, exist_ok=True)
        append_jsonl(campaign.attempt_ledger("T-1", attempt),
                     {"request_id": "r1", "failed": True})
        append_jsonl(campaign.attempt_ledger("T-1", attempt),
                     {"request_id": "r2", "failed": False})
        self.assertIsNone(runner._generation_failure_category(
            campaign, "T-1", attempt, {"exit_code": 0,
                                       "stdout_tail": "", "stderr_tail": ""}))

    def test_connection_error_in_the_output_is_infrastructure(self):
        campaign = support.temp_campaign("classify-net")
        self.assertIsNotNone(runner._generation_failure_category(
            campaign, "T-1", "missing", {"exit_code": 0, "stdout_tail": "",
                                         "stderr_tail": "APIConnectionError: boom"}))

    def test_missing_key_is_infrastructure_without_any_ledger(self):
        campaign = support.temp_campaign("classify-key")
        self.assertIsNotNone(runner._generation_failure_category(
            campaign, "T-1", "missing",
            {"exit_code": 1, "stdout_tail": "",
             "stderr_tail": "Missing API key for cmecloud"}))

    def test_a_clean_run_with_no_candidate_stays_a_science_failure(self):
        campaign = support.temp_campaign("classify-science")
        self.assertIsNone(runner._generation_failure_category(
            campaign, "T-1", "missing", {"exit_code": 0, "stdout_tail": "done",
                                         "stderr_tail": ""}))


class CheckpointSelectionTest(unittest.TestCase):
    def test_p3_wins_over_p2_and_p1(self):
        artifacts = {
            "top_results": {"results": [{"rank": 1, "code_dir": "/p1",
                                        "chain_id": "c1", "gflops": 1.0}]},
            "performance_unlock": {"best_code_dir": "/p2", "best_gflops": 2.0},
            "final_tuning": {"shapes": [{"shape": {"M": 512, "N": 512, "K": 512},
                                        "status": "completed",
                                        "final_code_dir": "/p3", "gflops": 3.0}]},
        }
        delivered = checkpoints.select_delivered(artifacts, 512)
        self.assertEqual(delivered["delivered_phase"], "p3")
        self.assertEqual(delivered["selection"]["code_dir"], "/p3")

    def test_p2_wins_when_p3_did_not_complete(self):
        artifacts = {
            "top_results": {"results": [{"rank": 1, "code_dir": "/p1", "gflops": 1.0}]},
            "performance_unlock": {"best_code_dir": "/p2", "best_gflops": 2.0},
            "final_tuning": {"shapes": [{"shape": {"M": 512},
                                        "status": "no_verified_result"}]},
        }
        delivered = checkpoints.select_delivered(artifacts, 512)
        self.assertEqual(delivered["delivered_phase"], "p2")

    def test_no_phase_means_no_delivery(self):
        delivered = checkpoints.select_delivered({}, 512)
        self.assertFalse(delivered["delivered"])
        self.assertIsNone(delivered["delivered_phase"])

    def test_search_accounting_names_a_prefix_search_as_not_exhaustive(self):
        artifacts = {"final_tuning": {"shapes": [{
            "shape": {"M": 512}, "status": "completed",
            "stop_reason": "budget_exhausted",
            "search_cost": {"tested_candidates": 32, "compile_attempts": 32,
                            "accepted_candidates": 3, "benchmark_process_runs": 9,
                            "rounds_executed": 1, "legal_candidates": 1710,
                            "budget_omitted_candidates": 1678}}]}}
        accounting = checkpoints.search_accounting(
            artifacts, config={"final_top1_tuning": {"max_candidates_per_shape": 1710,
                                                     "rounds": 3}}, shape=512)
        self.assertFalse(accounting["exhaustive"])
        self.assertIn("not exhaustive", accounting["claim"])
        self.assertEqual(accounting["space_total"], 1710)
        self.assertEqual(accounting["budget_omitted"], 1678)

    def test_exhaustive_only_when_nothing_was_omitted(self):
        artifacts = {"final_tuning": {"shapes": [{
            "shape": {"M": 512}, "status": "completed", "stop_reason": "target_reached",
            "search_cost": {"tested_candidates": 96, "compile_attempts": 96,
                            "accepted_candidates": 4, "benchmark_process_runs": 12,
                            "rounds_executed": 3, "legal_candidates": 96,
                            "budget_omitted_candidates": 0}}]}}
        accounting = checkpoints.search_accounting(artifacts, shape=512)
        self.assertTrue(accounting["exhaustive"])
        self.assertEqual(accounting["claim"], "exhaustive")


if __name__ == "__main__":
    unittest.main()


class GenerationDiagnosisTest(unittest.TestCase):
    """A no-candidate task must be able to explain itself after the workspace is gone."""

    def test_terminal_chain_verdicts_are_summarised(self):
        import json
        from SCOPE.experiments import runner as runner
        paths = support.temp_campaign("diagnosis")
        task_dir = paths.task_dir("T-32-r01")
        chain_dir = task_dir / "app_evidence" / "gemm_code" / "cpu_code" / "chain.1-1-1"
        chain_dir.mkdir(parents=True, exist_ok=True)
        (chain_dir / "terminal_chain.json").write_text(json.dumps({
            "accepted": False,
            "verification": {"compile_status": "pass",
                             "correctness_status": "fail",
                             "runtime_safety_status": "pass",
                             "gflops_mean": None},
        }), encoding="utf-8")
        check = task_dir / "app_evidence" / "results" / "check"
        check.mkdir(parents=True, exist_ok=True)
        (check / "evolution_summary.json").write_text(json.dumps({
            "terminal_chain_count": 1, "verified_terminal_chain_count": 1,
            "failed_strategy_counts": {"CPU.TailKernel.Scalar": 1},
        }), encoding="utf-8")
        diagnosis = runner._diagnose_generation(
            paths, "T-32-r01", {"file_count": 2, "bytes": 100, "files": []})
        self.assertEqual(diagnosis["chain_count"], 1)
        self.assertEqual(diagnosis["accepted_count"], 0)
        self.assertEqual(diagnosis["correctness_pass_count"], 0)
        self.assertIn("correctness", diagnosis["reason"])
        self.assertEqual(diagnosis["evolution"]["results/check/evolution_summary.json"]
                         ["terminal_chain_count"], 1)

    def test_harvest_is_bounded_and_hashed(self):
        from SCOPE.experiments import runner as runner
        workspace = types.SimpleNamespace(root=Path(self.directory) / "ws")
        scope = workspace.root / "SCOPE"
        (scope / "results" / "check").mkdir(parents=True)
        (scope / "results" / "check" / "evolution_summary.json").write_text("{}", encoding="utf-8")
        (scope / "gemm_code" / "cpu_code" / "chain.1").mkdir(parents=True)
        (scope / "gemm_code" / "cpu_code" / "chain.1" / "terminal_chain.json").write_text(
            "{}", encoding="utf-8")
        destination = Path(self.directory) / "evidence"
        record = runner._harvest_app_evidence(workspace, destination)
        self.assertEqual(record["file_count"], 2)
        self.assertEqual(len(record["files"]), 2)
        self.assertTrue(all(len(entry["sha256"]) == 64 for entry in record["files"]))
        self.assertTrue((destination / "results" / "check" / "evolution_summary.json").exists())

    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self.directory = self._tmp.name
        self.addCleanup(self._tmp.cleanup)
