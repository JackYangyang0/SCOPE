"""Task matrix arithmetic and protocol gating.

``实验设计.md`` fixes N.  These tests fail if a shape list, a repeat count or a
variant name is edited without the counts being revisited on purpose.
"""

from __future__ import annotations

import copy
import unittest

from SCOPE.experiments import protocol as protocol_module
from SCOPE.experiments import task_matrix as tasks
from SCOPE.utils.harness_errors import ProtocolError

import support


class MatrixTest(unittest.TestCase):
    def test_core_matrix_counts(self):
        matrix = tasks.build_task_matrix(blocks=("E1", "E2"))
        record = tasks.validate_matrix(matrix, blocks=("E1", "E2"))
        self.assertEqual(record["observed"]["E1_independent"], 200)
        self.assertEqual(record["observed"]["E2_independent_new"], 80)
        self.assertEqual(record["observed"]["E1_branches_no_feedback"], 40)
        self.assertEqual(record["observed"]["E1_aliases_no_parameter_search"], 40)
        self.assertEqual(record["observed"]["E2_aliases_reused"], 40)
        self.assertEqual(record["observed"]["core_independent_total"], 280)

    def test_cpu_matrix_skips_the_feedback_branch(self):
        matrix = tasks.build_task_matrix(blocks=("E1",), backend="cpu",
                                         shapes=tasks.CPU_SHAPES, repeats=2)
        record = tasks.validate_matrix(matrix, blocks=("E1",), backend="cpu",
                                       shapes=tasks.CPU_SHAPES, repeats=2)
        self.assertEqual(record["observed"]["E1_branches_no_feedback"], 0)
        self.assertEqual(record["observed"]["E1_independent"], 40)
        self.assertEqual(record["observed"]["E1_aliases_no_parameter_search"], 8)
        self.assertFalse([task for task in matrix if task.kind == "branch"])

    def test_cpu_matrix_refuses_a_wrong_repeat_count(self):
        matrix = tasks.build_task_matrix(blocks=("E1",), backend="cpu",
                                         shapes=(256,), repeats=2)
        with self.assertRaises(AssertionError):
            tasks.validate_matrix(matrix, blocks=("E1",), backend="cpu",
                                  shapes=(256,), repeats=3)

    def test_extension_counts(self):
        matrix = tasks.build_task_matrix(blocks=("E3", "D1"))
        record = tasks.validate_matrix(matrix, blocks=("E3", "D1"))
        self.assertEqual(record["observed"]["E3_independent"], 40)
        self.assertEqual(record["observed"]["D1_independent"], 80)

    def test_task_ids_are_unique_except_the_declared_reuse(self):
        matrix = tasks.build_task_matrix(blocks=("E1", "E2"))
        identifiers = [task.task_id for task in matrix]
        reused = [task for task in matrix
                  if task.kind == "alias" and task.block == "E2"]
        self.assertEqual(len(identifiers) - len(reused), len(set(identifiers)))
        for task in matrix:
            self.assertEqual(len(set(task.shape)), 1)
            self.assertIn(task.shape[0], tasks.SHAPES)

    def test_reused_qwen_slots_point_at_e1_full(self):
        matrix = tasks.build_task_matrix(blocks=("E2",))
        reused = [task for task in matrix if task.reuse_of]
        self.assertEqual(len(reused), 40)
        for task in reused:
            self.assertEqual(task.kind, "alias")
            self.assertTrue(task.task_id.startswith("E1-full-"))

    def test_branches_have_parents(self):
        matrix = tasks.build_task_matrix(blocks=("E1",))
        for task in matrix:
            if task.kind == "branch":
                self.assertEqual(task.parent_task_id,
                                 f"E1-full-{task.shape[0]}-r{task.repeat:02d}")
                self.assertEqual(task.variant, "minus_feedback_optimization")

    def test_order_is_interleaved_across_groups(self):
        matrix = tasks.build_task_matrix(blocks=("E1",))
        ordered = sorted(matrix, key=lambda task: task.order_index)
        expected_groups = set(tasks.E1_VARIANTS) | set(tasks.STAGE_VARIANTS)
        first_round = [task.variant for task in ordered[: len(expected_groups)]]
        self.assertEqual(set(first_round), expected_groups,
                         "the first round must touch every configuration once")

    def test_validation_detects_a_missing_task(self):
        matrix = tasks.build_task_matrix(blocks=("E1",))
        with self.assertRaises(AssertionError):
            tasks.validate_matrix(matrix[:-1], blocks=("E1",))

    def test_duplicate_ids_are_rejected(self):
        matrix = tasks.build_task_matrix(blocks=("REHEARSAL",))
        duplicated = list(matrix) + [copy.copy(matrix[0])]
        with self.assertRaises(AssertionError):
            tasks.validate_matrix(duplicated, blocks=("REHEARSAL",))

    def test_pilot_is_ten_tasks_across_four_shapes(self):
        pilot = tasks.pilot_tasks()
        self.assertEqual(len(pilot), 10)
        self.assertEqual({task.shape[0] for task in pilot}, set(tasks.SHAPES))


class ProtocolGateTest(unittest.TestCase):
    def test_complete_protocol_passes(self):
        campaign = support.temp_campaign("protocol-ok")
        protocol = support.tiny_protocol(campaign)
        checks = protocol_module.validate(protocol, require_frozen=True,
                                          paths=None,
                                          allow_missing_credentials=True)
        self.assertTrue(checks)
        self.assertTrue(all(check["ok"] for check in checks))

    def test_missing_budget_is_refused(self):
        campaign = support.temp_campaign("protocol-budget")
        protocol = support.tiny_protocol(campaign)
        protocol["budget"].pop("per_task_wall_seconds")
        with self.assertRaises(ProtocolError) as raised:
            protocol_module.validate(protocol, require_frozen=True, paths=None,
                                     allow_missing_credentials=True)
        self.assertIn("per_task_wall_seconds", str(raised.exception))

    def test_compact_budget_is_refused(self):
        campaign = support.temp_campaign("protocol-compact")
        protocol = support.tiny_protocol(campaign)
        protocol["budget"]["budget_profile"] = "compact"
        with self.assertRaises(ProtocolError):
            protocol_module.validate(protocol, require_frozen=True, paths=None,
                                     allow_missing_credentials=True)

    def test_draft_status_is_refused_by_run(self):
        campaign = support.temp_campaign("protocol-draft")
        protocol = support.tiny_protocol(campaign)
        protocol["status"] = "draft"
        with self.assertRaises(ProtocolError):
            protocol_module.validate(protocol, require_frozen=True, paths=None,
                                     allow_missing_credentials=True)

    def test_unresolved_model_is_refused(self):
        campaign = support.temp_campaign("protocol-model")
        protocol = support.tiny_protocol(campaign)
        protocol["models"] = {"rehearsal": {"model": "", "api_key_env": None,
                                            "api_key_env_present": False}}
        with self.assertRaises(ProtocolError):
            protocol_module.validate(protocol, require_frozen=True, paths=None,
                                     allow_missing_credentials=True)

    def test_red_controls_gate_is_refused(self):
        campaign = support.temp_campaign("protocol-gate")
        protocol = support.tiny_protocol(campaign)
        protocol["controls_gate"]["ok"] = False
        with self.assertRaises(ProtocolError):
            protocol_module.validate(protocol, require_frozen=True, paths=None,
                                     allow_missing_credentials=True)

    def test_wrong_timing_protocol_is_refused(self):
        campaign = support.temp_campaign("protocol-timing")
        protocol = support.tiny_protocol(campaign)
        protocol["timing"]["measure_processes"] = 3
        with self.assertRaises(ProtocolError):
            protocol_module.validate(protocol, require_frozen=True, paths=None,
                                     allow_missing_credentials=True)

    def test_missing_credentials_need_an_explicit_opt_in(self):
        campaign = support.temp_campaign("protocol-credentials")
        protocol = support.tiny_protocol(campaign)
        protocol["models"]["rehearsal"]["api_key_env_present"] = False
        with self.assertRaises(ProtocolError):
            protocol_module.validate(protocol, require_frozen=True, paths=None)
        protocol_module.validate(protocol, require_frozen=True, paths=None,
                                 allow_missing_credentials=True)

    def test_protocol_hash_detects_editing(self):
        campaign = support.temp_campaign("protocol-hash")
        protocol = support.tiny_protocol(campaign)
        # `load` re-checks the dataset manifests on disk, so give it the file the
        # protocol claims to have frozen.
        from SCOPE.utils.harness_util import write_json as _write_json
        _write_json(campaign.dataset_manifest("postselection_test"),
                    {"manifest_hash": protocol["data"]["manifest_hashes"]
                     ["postselection_test"]})
        protocol_module.write_protocol(campaign, protocol,
                                       filename="FROZEN_PROTOCOL.json")
        loaded = protocol_module.load(campaign, frozen=True,
                                      allow_missing_credentials=True)
        self.assertEqual(loaded["status"], "frozen")
        stored = dict(loaded)
        stored["shapes"] = [999]
        from SCOPE.utils.harness_util import write_json
        write_json(campaign.frozen_protocol, stored)
        with self.assertRaises(ProtocolError):
            protocol_module.load(campaign, frozen=True,
                                 allow_missing_credentials=True)


if __name__ == "__main__":
    unittest.main()
