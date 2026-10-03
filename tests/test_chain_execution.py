from __future__ import annotations

import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Event, Lock
from unittest.mock import patch

from SCOPE.app import run_exhaustive_stage, run_unlocked_performance_phase, save_candidate_json, stage_path
from SCOPE.utils.common_utils import save_json
from SCOPE.utils.execution import (chain_client, configure_execution, defer_latest,
                                  execution_slot, parallel_chain_map)


class ChainExecutionTests(unittest.TestCase):
    def tearDown(self):
        configure_execution()

    def test_chains_overlap_and_return_in_input_order(self):
        configure_execution({"chain_workers": 3})
        together = Barrier(3)
        def worker(index):
            together.wait(timeout=3)
            return index
        outcomes = parallel_chain_map(worker, [1, 2, 3], str)
        self.assertTrue(all(outcome.error is None for outcome in outcomes))
        self.assertEqual([outcome.value for outcome in outcomes], [1, 2, 3])

    def test_chain_failure_does_not_cancel_other_chains(self):
        def worker(index):
            if index == 2:
                raise ValueError("broken patch")
            return index
        outcomes = parallel_chain_map(worker, [1, 2, 3], str)
        self.assertEqual(outcomes[0].value, 1)
        self.assertIsInstance(outcomes[1].error, ValueError)
        self.assertEqual(outcomes[2].value, 3)

    def assert_gate_limit(self, kind, limit):
        configure_execution({"llm_max_concurrency": 3, "compile_workers": 2})
        ready = Barrier(6)
        filled, release = Event(), Event()
        lock = Lock()
        active, maximum = 0, 0
        def worker():
            nonlocal active, maximum
            ready.wait(timeout=3)
            with execution_slot(kind):
                with lock:
                    active += 1
                    maximum = max(maximum, active)
                    if active == limit:
                        filled.set()
                try:
                    self.assertTrue(release.wait(timeout=3))
                    with execution_slot(kind):
                        pass
                finally:
                    with lock:
                        active -= 1
        with ThreadPoolExecutor(max_workers=5) as pool:
            tasks = [pool.submit(worker) for _ in range(5)]
            try:
                ready.wait(timeout=3)
                self.assertTrue(filled.wait(timeout=3))
            finally:
                release.set()
            for task in tasks:
                task.result(timeout=3)
        self.assertEqual(maximum, limit)

    def test_llm_requests_are_bounded(self):
        self.assert_gate_limit("llm", 3)

    def test_compiles_are_bounded(self):
        self.assert_gate_limit("compile", 2)

    def test_benchmark_slot_is_exclusive_and_reentrant(self):
        self.assert_gate_limit("gpu_verification", 1)

    def test_chain_artifacts_are_isolated_and_latest_is_deferred(self):
        with tempfile.TemporaryDirectory() as directory:
            latest = Path(directory) / "generated_patch.json"
            check = Path(directory) / "post_check.json"
            together = Barrier(3)
            def worker(index):
                together.wait(timeout=3)
                save_candidate_json(latest, "Layout", "Strategy", 1, {"chain": index}, str(index))
                own_check = stage_path(check, "Layout.StageVerification")
                save_json(own_check, {"chain": index})
                return own_check
            outcomes = parallel_chain_map(worker, [1, 2, 3], str)
            self.assertFalse(latest.exists())
            self.assertTrue(all(outcome.error is None for outcome in outcomes))
            self.assertEqual(len({outcome.value for outcome in outcomes}), 3)
            for index, outcome in enumerate(outcomes, start=1):
                self.assertEqual(json.loads(outcome.value.read_text(encoding="utf-8")), {"chain": index})
                self.assertEqual(outcome.latest[latest], {"chain": index})
            for outcome in outcomes:
                for output, data in outcome.latest.items():
                    save_json(output, data)
            self.assertEqual(json.loads(latest.read_text(encoding="utf-8")), {"chain": 3})
            self.assertFalse(defer_latest(latest, {}))

    def test_clients_are_separate_and_closed(self):
        created, closed = [], []
        class Client:
            def __init__(self, config):
                self._config = config
                self._client = self
                created.append(self)
            def close(self):
                closed.append(self)
        base = Client({"model": "mock"})
        def worker(index):
            with chain_client(base) as client:
                return client
        outcomes = parallel_chain_map(worker, [1, 2, 3], str)
        self.assertTrue(all(outcome.error is None for outcome in outcomes))
        self.assertEqual(len({id(outcome.value) for outcome in outcomes}), 3)
        self.assertNotIn(base, closed)
        self.assertEqual(len(closed), 3)

    def test_stage_dispatch_is_parallel_and_does_not_mutate_parent_ir(self):
        states = [{"state_id": str(index), "current_ir": {"marker": index}, "history": {"events": []},
                   "path": [f"Tiling.{index}"], "path_code": [index],
                   "source_snapshot": {"cuda_kernel.cuh": str(index)}} for index in (1, 2, 3)]
        together = Barrier(3)
        def stage(**kwargs):
            together.wait(timeout=3)
            kwargs["current_ir"]["changed"] = True
            return {"summary": {}, "accepted_candidates": [], "events": [], "failed_strategy_counts": {}}
        with patch("SCOPE.app.run_stage", side_effect=stage):
            result = run_exhaustive_stage("Layout", states, {}, {}, {}, object(), "test")
        self.assertEqual(len(result["terminal_states"]), 3)
        self.assertFalse(any("changed" in state["current_ir"] for state in states))
        self.assertFalse(any(item.get("status") == "chain_error" for item in result["summary"]["state_summaries"]))

    def test_configuration_rejects_unsafe_or_invalid_limits(self):
        for config in ({"chain_workers": 0}, {"compile_workers": True}, {"gpu_verification_workers": 2}):
            with self.assertRaises(ValueError):
                configure_execution(config)

    def test_unlock_chains_run_concurrently(self):
        candidates = [{"strategy_id": f"chain_{index}"} for index in (1, 2, 3)]
        together = Barrier(3)
        def unlock(**kwargs):
            together.wait(timeout=3)
            return {"base_chain_id": kwargs["base_candidate"]["strategy_id"],
                    "verified_candidates": [], "summary": {"stage": "PerformanceUnlock"}}
        with patch("SCOPE.app.select_top_correct_candidates", return_value=candidates), \
             patch("SCOPE.app.run_batch_unlock_for_terminal_candidate", side_effect=unlock):
            result = run_unlocked_performance_phase({}, {}, {}, {}, object(), "test", "profile",
                                                    "linux", False, 60, 5, 2)
        reports = result["terminal_verification"]["summary"]["chain_reports"]
        self.assertEqual([report["base_chain_id"] for report in reports], ["chain_1", "chain_2", "chain_3"])
        self.assertTrue(all("status" not in report["summary"] for report in reports))


if __name__ == "__main__":
    unittest.main()
