from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from SCOPE.app import (build_top_terminal_results, deduplicate_code_candidates, repair_chain_locally,
                      run_exhaustive_stage, terminal_chain_is_accepted)


class SearchResultIntegrityTests(unittest.TestCase):
    def test_same_source_different_path_does_not_occupy_multiple_ranks(self):
        def candidate(name, speed, source):
            return {"strategy_id": name, "source_snapshot": {"cuda_kernel.cuh": source},
                    "verified_ir": {"problem": {"M": 512, "N": 512, "K": 512},
                                    "performance": {"gflops": speed}}}
        results = deduplicate_code_candidates([
            candidate("batch1", 100, "kernel-a"), candidate("batch2", 105, "kernel-a"),
            candidate("other", 95, "kernel-b")])
        self.assertEqual({item["strategy_id"] for item in results}, {"batch2", "other"})

    def test_static_findings_do_not_discard_runtime_verified_code(self):
        ir = {"verification": {"accepted": True,
                               "summary": {"optimization_realization_status": "fail"}}}
        ir['locked_repair_verification'] = {'status': 'fail'}
        ir['phase1_hard_constraints'] = {'hard_constraints_ok': False}
        self.assertTrue(terminal_chain_is_accepted(ir))
        ir["verification"]["summary"]["optimization_realization_status"] = "not_disproven"
        self.assertTrue(terminal_chain_is_accepted(ir))
        ir['verification']['accepted'] = False
        self.assertFalse(terminal_chain_is_accepted(ir))
        ir['verification']['accepted'] = None
        self.assertFalse(terminal_chain_is_accepted(ir))

    def test_advisory_failure_candidate_can_win_ranking(self):
        candidates = []
        for sid, speed, advisory in [('baseline', 9607, False), ('vector', 10077, True)]:
            ir = {'verification': {'accepted': True, 'summary': {'code_checker_advisory_failure': advisory}},
                  'problem': {'M': 1024, 'N': 1024, 'K': 1024}, 'performance': {'gflops': speed},
                  'locked_repair_verification': {'status': 'fail' if advisory else 'pass'}}
            candidates.append({'strategy_id': sid, 'verified_ir': ir,
                               'accepted': terminal_chain_is_accepted(ir),
                               'source_snapshot': {'cuda_kernel.cuh': sid}})
        top = build_top_terminal_results(candidates, 3)
        self.assertEqual(top[0]['chain_id'], 'vector')
        self.assertTrue(top[0]['code_checker_advisory_failure'])

    def test_repair_rejects_full_file_and_keeps_original(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = "// existing cp.async optimized kernel\n"
            (root / "cuda_kernel.cuh").write_text(original, encoding="utf-8")
            (root / "kernel.h").write_text("// interface\n", encoding="utf-8")
            with patch("SCOPE.app.generate_code_files_from_patch_with_llm", return_value={
                "files": [{"path": "cuda_kernel.cuh", "content": "generic kernel"}]}):
                result = repair_chain_locally(root, root, ir={}, client=object(), strategy_library={})
            self.assertEqual(result["status"], "repair_failed")
            self.assertEqual((root / "cuda_kernel.cuh").read_text(encoding="utf-8"), original)

    def test_no_successor_preserves_parent_for_terminal_verification(self):
        state = {"state_id": "parent", "current_ir": {}, "history": {"events": []},
                 "path": ["Tiling.BlockTile.64x64x16"], "path_code": [1],
                 "source_snapshot": {"cuda_kernel.cuh": "last valid code"}}
        with patch("SCOPE.app.run_stage", return_value={"summary": {}, "accepted_candidates": [],
                                                       "events": [], "failed_strategy_counts": {}}):
            result = run_exhaustive_stage("Layout", [state], {}, {}, {}, object(), "test")
        self.assertEqual(result["frontier"], [])
        self.assertEqual(result["terminal_states"][0]["source_snapshot"], state["source_snapshot"])

    def test_failed_results_do_not_become_top_results(self):
        self.assertEqual(build_top_terminal_results([{"accepted": False, "verified_ir": {}}], 3), [])


if __name__ == "__main__":
    unittest.main()
