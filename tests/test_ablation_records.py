from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from SCOPE.app import build_rq2_ablation_record, build_rq3_optimization_record
from SCOPE.generate_ir.stage_controller import LocalCheckResult, StageController, record_micro_strategy_application
from SCOPE.llm.patch_generator import build_patch_ir, compact_ir_for_prompt
from SCOPE.utils.ablation import (
    ablation_snapshot,
    configure_ablation,
    materialize_intent_mapping_fields,
    model_visible_ir,
    strategy_for_control,
)
from SCOPE.scripts.summarize_rq_experiments import rq3_rows, standalone_tune_row


class AblationRecordTests(unittest.TestCase):
    def tearDown(self) -> None:
        configure_ablation({})

    def test_structured_state_ablation_hides_generation_state_from_prompt(self) -> None:
        configure_ablation({"enabled": True, "structured_state": False})
        compact = compact_ir_for_prompt({
            "problem": {"M": 512, "N": 512, "K": 512},
            "hardware": {"sm_count": 36},
            "tiling": {"block_m": 128},
            "mapping": {"threads_per_block": 256},
            "strategy": {"applied_strategy_ids": ["Tiling.BlockTile.128x128x16"]},
        })
        self.assertIn("problem", compact)
        self.assertIn("hardware", compact)
        self.assertNotIn("tiling", compact)
        self.assertNotIn("mapping", compact)
        self.assertNotIn("strategy", compact)
        self.assertGreater(ablation_snapshot()["counters"]["structured_state.prompt_suppressed"], 0)

    def test_disabled_stage_gate_retains_shadow_failure(self) -> None:
        configure_ablation({"enabled": True, "stage_constraints": False})
        controller = StageController.__new__(StageController)
        controller.current_stage = "Tiling"
        controller.stage_spec = {}
        with patch(
            "SCOPE.generate_ir.stage_controller.get_stage_verification",
            return_value={"required_fields": ["tiling.block_m"], "predicates": []},
        ):
            report = controller.verify_stage({"tiling": {}}, include_code_checks=False)
        self.assertTrue(report["accepted"])
        self.assertFalse(report["shadow_accepted"])
        self.assertFalse(report["gate_enabled"])
        self.assertEqual(ablation_snapshot()["counters"]["stage_constraint.rejected"], 1)

    def test_intent_only_hides_observations_but_retains_strategy_intent(self) -> None:
        configure_ablation({"enabled": True, "observed_state_control": False})
        visible = model_visible_ir({
            "problem": {"M": 512, "N": 512, "K": 512},
            "hardware": {"sm_count": 34},
            "strategy": {
                "current_strategy_id": "Tiling.BlockTile.64x64x16",
                "applied_strategy_ids": ["Tiling.BlockTile.64x64x16"],
            },
            "performance": {"gflops": 10000.0},
            "verification": {"compile": {"status": "pass"}},
        })
        self.assertEqual(
            visible["strategy_intent"]["current_strategy_id"],
            "Tiling.BlockTile.64x64x16",
        )
        self.assertNotIn("performance", visible)
        self.assertNotIn("verification", visible)

        controlled = strategy_for_control({
            "strategy_id": "test",
            "preconditions": ["problem.K >= 16", "resource.shared_memory.total_bytes <= 49152"],
        })
        self.assertEqual(controlled["preconditions"], ["problem.K >= 16"])

    def test_intent_only_materializes_only_deterministic_mapping_fields(self) -> None:
        configure_ablation({"enabled": True, "observed_state_control": False})
        ir = {
            "tiling": {
                "block_m": 128,
                "block_n": 64,
                "warp_tile": {"warp_m": 32, "warp_n": 32},
            },
            "hardware": {"warp_size": 32},
            "mapping": {
                "warps_m": "derived_from_tile_sizes",
                "warps_n": "derived_from_tile_sizes",
                "warps_per_block": "derived_from_block_tile_and_warp_count",
                "threads_per_block": "derived_from_block_tile_and_warp_count",
            },
        }
        changed = materialize_intent_mapping_fields(ir)
        self.assertEqual(ir["mapping"]["warps_m"], 4)
        self.assertEqual(ir["mapping"]["warps_n"], 2)
        self.assertEqual(ir["mapping"]["warps_per_block"], 8)
        self.assertEqual(ir["mapping"]["threads_per_block"], 256)
        self.assertEqual(len(changed), 4)
        self.assertEqual(
            ablation_snapshot()["counters"]["intent_mapping_materialized.count"],
            1,
        )

    def test_intent_only_patch_ir_accepts_derivable_mapping_placeholders(self) -> None:
        configure_ablation({"enabled": True, "observed_state_control": False})
        patch_ir = build_patch_ir(
            ir={
                "target": {"backend": "cuda"},
                "tiling": {
                    "block_m": 64,
                    "block_n": 64,
                    "warp_tile": {"warp_m": 32, "warp_n": 32},
                },
                "hardware": {"warp_size": 32},
                "mapping": {},
            },
            strategy={"stage": "Tiling", "strategy_id": "Tiling.ThreadTile.4x4"},
            precheck_item={},
            patch_result={
                "ir_updates": {
                    "mapping.threads_per_block": "derived_from_tile_sizes",
                }
            },
        )
        self.assertEqual(patch_ir["mapping"]["threads_per_block"], 128)

    def test_full_variant_does_not_use_intent_interface_adapter(self) -> None:
        configure_ablation({"enabled": False, "observed_state_control": True})
        ir = {
            "tiling": {
                "block_m": 64,
                "block_n": 64,
                "warp_tile": {"warp_m": 32, "warp_n": 32},
            },
            "hardware": {"warp_size": 32},
            "mapping": {"threads_per_block": "derived_from_tile_sizes"},
        }
        self.assertEqual(materialize_intent_mapping_fields(ir), [])
        self.assertEqual(ir["mapping"]["threads_per_block"], "derived_from_tile_sizes")

    def test_disabled_local_gate_advances_failed_shadow_candidate(self) -> None:
        configure_ablation({"enabled": True, "local_checks": False})
        ir = {}
        record_micro_strategy_application(
            ir,
            "Layout",
            "Layout.SharedMemory",
            "Layout.SharedMemory.AB.Basic",
            {},
            [LocalCheckResult("layout", "fail", "shadow failure")],
        )
        record = ir["strategy"]["applied_micro_strategies"][0]
        self.assertEqual(record["status"], "local_fail_shadow_bypassed")
        self.assertIn("Layout.SharedMemory", ir["strategy"]["completed_subphases"])

    def test_rq_records_compute_acceptance_and_incremental_gain(self) -> None:
        configure_ablation({"run_id": "rq2-test"})
        candidate1 = {"verified_ir": {"performance": {"gflops": 100.0}}}
        candidate2 = {"verified_ir": {"performance": {"gflops": 125.0}}}
        terminal = {
            "best_candidate": candidate1,
            "summary": {
                "terminal_chain_count": 4,
                "verified_terminal_chain_count": 3,
                "accepted_terminal_chain_count": 2,
            },
        }
        rq2 = build_rq2_ablation_record(
            ablation_snapshot(),
            {"target": {"backend": "cuda"}, "problem": {"M": 512, "N": 512, "K": 512}},
            {"llm": {"provider": "mock", "mock": {"model": "mock-model"}}},
            [], terminal, 12.0, {"request_count": 2},
        )
        self.assertEqual(rq2["variant"], "full")
        self.assertEqual(rq2["outcome"]["acceptance_rate"], 0.5)
        rq3 = build_rq3_optimization_record(
            terminal,
            {
                "best_candidate": candidate2,
                "terminal_verification": {"summary": {
                    "input_correct_phase1_chain_count": 1,
                    "terminal_chain_count": 2,
                    "accepted_terminal_chain_count": 1,
                    "chain_reports": [{"summary": {"compilation_count": 3, "feedback_round_count": 2}}],
                }},
            },
            {"backend": "cuda", "completed_shape_count": 1, "shapes": [{
                "shape": {"M": 1024, "N": 1024, "K": 1024},
                "status": "completed", "seed_gflops": 80.0, "gflops": 120.0,
                "search_cost": {"tested_candidates": 5, "compile_attempts": 5},
            }]},
            3.0, 4.0, {},
        )
        self.assertEqual(rq3["phase2_feedback_optimization"]["gain_percent"], 25.0)
        self.assertEqual(rq3["phase3_constrained_reinstantiation"]["shapes"][0]["gain_percent"], 50.0)
        self.assertEqual(rq3["phase2_feedback_optimization"]["search_cost"]["compilation_count"], 3)

    def test_rq3_summary_ignores_inactive_phase2_and_accepts_standalone_tuning(self) -> None:
        self.assertEqual(rq3_rows([(Path("inactive.json"), {
            "phase2_feedback_optimization": {"triggered": False},
            "phase3_constrained_reinstantiation": {"shapes": []},
        })]), [])
        row = standalone_tune_row(Path("search.json"), {
            "backend": "cpu",
            "shape": {"M": 1024, "N": 1024, "K": 1024},
            "elapsed_seconds": 12.5,
            "tested_candidate_count": 2,
            "accepted_candidate_count": 2,
            "compile_attempt_count": 3,
            "stop_reason": "budget_exhausted",
            "results": [
                {"candidate_id": 0, "accepted": True, "gflops": 100.0,
                 "performance": {"benchmark_successful_runs": 3}},
                {"candidate_id": 1, "accepted": True, "gflops": 125.0,
                 "performance": {"benchmark_successful_runs": 3}},
            ],
        })
        self.assertIsNotNone(row)
        self.assertEqual(row["gain_percent"], 25.0)
        self.assertEqual(row["benchmark_process_runs"], 6)


if __name__ == "__main__":
    unittest.main()
