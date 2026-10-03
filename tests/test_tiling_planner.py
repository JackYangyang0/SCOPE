from __future__ import annotations

import unittest
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

from SCOPE.app import (
    apply_deterministic_code_patch,
    compact_joint_tiling_ir_summary,
    build_joint_tiling_micro_selection,
    build_phase1_compile_shortlist,
    frontier_limit_for_stage,
    planned_tiling_micro_selection,
    prune_frontier_after_core_construction,
    prune_frontier_per_tiling_lineage,
    take_frontier_batch_per_tiling_lineage,
    tiling_plan_ir_updates,
    tiling_lineage_keys,
)
from SCOPE.generate_ir.tiling_planner import (
    build_joint_tiling_candidates,
    make_diverse_tiling_pool,
    rebalance_low_reuse_tiling_choices,
    select_diverse_joint_tilings,
    tiling_plan_for_ir,
)
from SCOPE.generate_ir.gpu_architecture import (
    build_gpu_architecture_profile,
    estimate_tiling_execution,
    score_tiling,
)
from SCOPE.llm.strategy_selector import (
    build_joint_tiling_messages,
    compact_joint_tiling_option_space,
    validate_joint_tiling_response,
)
from SCOPE.utils.common_utils import load_json


ROOT = Path(__file__).resolve().parents[1]


class JointTilingPlannerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.library = load_json(ROOT / "data" / "lib" / "strategy_library.json")
        cls.ir = {
            "problem": {"dtype": "fp32", "M": 4096, "N": 4096, "K": 4096},
            "hardware": {
                "warp_size": 32,
                "max_threads_per_block": 1024,
                "max_shared_memory_per_block_bytes": 49152,
            },
        }

    def test_enumerates_complete_legal_asymmetric_block_tuple(self):
        candidates = build_joint_tiling_candidates(self.library, self.ir)
        match = next(
            item
            for item in candidates
            if (item["BM"], item["BN"], item["BK"], item["WM"], item["WN"], item["TM"], item["TN"])
            == (64, 128, 16, 32, 64, 2, 2)
        )
        self.assertEqual(match["threads_per_block"], 128)
        self.assertNotEqual(match["threads_per_block"], match["BM"] * match["BN"])
        self.assertTrue(
            any(
                (item["BM"], item["BN"], item["BK"], item["WM"], item["WN"], item["TM"], item["TN"])
                == (64, 128, 16, 32, 64, 2, 4)
                for item in candidates
            )
        )

    def test_all_enumerated_tuples_satisfy_hard_constraints(self):
        candidates = build_joint_tiling_candidates(self.library, self.ir)
        self.assertTrue(candidates)
        for item in candidates:
            self.assertEqual(item["BM"] % item["WM"], 0)
            self.assertEqual(item["BN"] % item["WN"], 0)
            self.assertLessEqual(item["threads_per_block"], 1024)
            self.assertLessEqual(item["shared_memory_bytes"], 49152)

    def test_candidates_include_target_architecture_execution_estimates(self):
        state = {
            "problem": {"dtype": "fp32", "M": 512, "N": 512, "K": 512},
            "hardware": {
                "compute_capability": "8.0",
                "sm_count": 108,
                "warp_size": 32,
                "max_threads_per_block": 1024,
                "max_threads_per_multiprocessor": 2048,
                "max_blocks_per_multiprocessor": 32,
                "max_shared_memory_per_block_bytes": 49152,
                "max_shared_memory_per_multiprocessor_bytes": 167936,
                "registers_per_multiprocessor": 65536,
            },
        }
        candidate = build_joint_tiling_candidates(self.library, state)[0]
        self.assertEqual(candidate["architecture_family"], "ampere_cc80_datacenter")
        self.assertIn("resident_ctas_per_sm", candidate)
        self.assertIn("estimated_occupancy", candidate)
        self.assertIn("cta_waves", candidate)
        self.assertIn("architecture_score", candidate)

    def test_many_sm_small_matrix_penalizes_low_cta_coverage(self):
        profile = build_gpu_architecture_profile({
            "compute_capability": "8.0",
            "sm_count": 108,
            "max_threads_per_multiprocessor": 2048,
            "max_blocks_per_multiprocessor": 32,
            "max_shared_memory_per_multiprocessor_bytes": 167936,
            "registers_per_multiprocessor": 65536,
        })
        common = dict(
            problem={"M": 512, "N": 512}, profile=profile, bk=16,
            threads_per_block=128, warps_per_block=4,
            estimated_registers_per_thread=32, element_bytes=4,
        )
        small_tile = estimate_tiling_execution(
            bm=64, bn=64, shared_memory_bytes=8192, **common
        )
        large_tile = estimate_tiling_execution(
            bm=128, bn=128, shared_memory_bytes=16384, **common
        )
        self.assertGreater(small_tile["sm_coverage"], large_tile["sm_coverage"])
        self.assertGreater(small_tile["architecture_score"], large_tile["architecture_score"])

    def test_cc80_minor_zero_is_not_treated_as_missing(self):
        profile = build_gpu_architecture_profile({
            "compute_capability_major": 8,
            "compute_capability_minor": 0,
        })
        self.assertEqual(profile["architecture_family"], "ampere_cc80_datacenter")
        self.assertEqual(profile["max_warps_per_sm"], 64)

    def test_blackwell_scoring_prefers_reuse_and_thread_ilp_over_tiny_tile(self):
        profile = build_gpu_architecture_profile({
            "compute_capability": "12.0",
            "sm_count": 170,
            "max_threads_per_multiprocessor": 1536,
            "max_blocks_per_multiprocessor": 24,
            "max_shared_memory_per_multiprocessor_bytes": 131072,
            "registers_per_multiprocessor": 65536,
        })
        common = dict(
            problem={"M": 1024, "N": 1024}, profile=profile, bk=16,
            threads_per_block=128, warps_per_block=4,
            estimated_registers_per_thread=48, element_bytes=4,
        )
        small = estimate_tiling_execution(
            bm=32, bn=64, shared_memory_bytes=6144,
            estimated_accumulators_per_thread=8, **common,
        )
        balanced = estimate_tiling_execution(
            bm=64, bn=64, shared_memory_bytes=8192,
            estimated_accumulators_per_thread=16, **common,
        )
        self.assertEqual(profile["architecture_family"], "blackwell_cc10plus")
        self.assertGreater(balanced["architecture_score"], small["architecture_score"])

    def test_reduced_resident_slot_capacity_is_not_a_score_bonus(self):
        common = dict(occupancy=0.5, sm_coverage=0.9, arithmetic_intensity=20,
                      accumulators_per_thread=16, architecture_family='ada_cc89')
        self.assertEqual(score_tiling(wave_utilization=0.1, **common),
                         score_tiling(wave_utilization=1.0, **common))

    def test_large_thread_blocks_remain_legal_without_automatic_occupancy_preference(self):
        state = {**self.ir, 'problem': {'M': 512, 'N': 512, 'K': 512},
                 'hardware': {**self.ir['hardware'], 'compute_capability': '8.9', 'sm_count': 34}}
        candidates = build_joint_tiling_candidates(self.library, state)
        same_block = [c for c in candidates if (c['BM'], c['BN'], c['BK']) == (128, 64, 16)]
        many_threads = [c for c in same_block if c['threads_per_block'] == 1024]
        self.assertTrue(many_threads)  # Soft score correction, not a new hard filter.
        balanced = next(c for c in same_block if
                        (c['WM'], c['WN'], c['TM'], c['TN']) == (32, 32, 4, 4))
        self.assertGreater(balanced['architecture_score'], max(c['architecture_score'] for c in many_threads))
        self.assertIn('not measured throughput', balanced['score_basis'])

    def test_last_wave_utilization_counts_tail_only(self):
        profile = build_gpu_architecture_profile({'sm_count': 2, 'compute_capability': '8.9'})
        result = estimate_tiling_execution(problem={'M': 320, 'N': 64}, profile=profile,
            bm=64, bn=64, bk=16, threads_per_block=1024, warps_per_block=32,
            estimated_registers_per_thread=32, shared_memory_bytes=8192, element_bytes=4)
        self.assertEqual(result['cta_count'], 5)
        self.assertEqual(result['cta_waves'], 3)
        self.assertEqual(result['last_wave_utilization'], 0.5)

    def test_joint_llm_receives_bounded_resource_pool(self):
        captured = {}

        def select(**kwargs):
            captured["count"] = len(kwargs["tiling_candidates"])
            return {
                "candidate_ids": [item["candidate_id"] for item in kwargs["tiling_candidates"][:9]],
                "reason": "test",
            }

        available = {
            "strategies": [
                {"strategy_id": item}
                for item in sorted({
                    candidate["block_strategy_id"]
                    for candidate in build_joint_tiling_candidates(self.library, self.ir)
                })
            ]
        }
        with patch("SCOPE.app.get_joint_tiling_selection_from_llm", side_effect=select):
            result = build_joint_tiling_micro_selection(
                object(), self.ir, self.library, available, top_k=9, resource_pool_size=24
            )
        self.assertLessEqual(captured["count"], 24)
        self.assertLessEqual(len(result["candidates"]), 9)
        self.assertEqual(result["resource_pool_count"], captured["count"])

    def test_joint_tiling_fills_short_llm_selection_to_top_k(self):
        available = {
            "strategies": [
                {"strategy_id": item}
                for item in sorted({
                    candidate["block_strategy_id"]
                    for candidate in build_joint_tiling_candidates(self.library, self.ir)
                })
            ]
        }

        def select(**kwargs):
            return {
                "candidate_ids": [item["candidate_id"] for item in kwargs["tiling_candidates"][:2]],
                "reason": "intentionally short response",
            }

        with patch("SCOPE.app.get_joint_tiling_selection_from_llm", side_effect=select):
            result = build_joint_tiling_micro_selection(
                object(), self.ir, self.library, available, top_k=3, resource_pool_size=32
            )

        self.assertEqual(len(result["candidates"]), 3)
        blocks = {
            (item["tiling_plan"]["BM"], item["tiling_plan"]["BN"], item["tiling_plan"]["BK"])
            for item in result["candidates"]
        }
        self.assertEqual(len(blocks), 3)
        self.assertTrue(all(
            item["tiling_plan"]["fragment_fmas_per_shared_element"] >= 1.5
            for item in result["candidates"]
        ))

    def test_compile_shortlist_prefers_completed_states_and_is_global(self):
        completed = [
            {"path": [f"Tiling.{index}", "Layout.done"], "path_code": [index], "current_ir": {}}
            for index in range(8)
        ]
        fallback = [
            {"path": ["Tiling.fallback"], "path_code": [1], "current_ir": {}}
        ]
        shortlist = build_phase1_compile_shortlist(completed, fallback, 6)
        self.assertEqual(len(shortlist), 6)
        self.assertTrue(all(len(item["path"]) == 2 for item in shortlist))

    def test_compile_shortlist_excludes_incomplete_failed_stage_states(self):
        completed = [{"path": ["all", "stages", "done"], "path_code": [1], "current_ir": {}}]
        fallback = [{
            "path": ["tiling", "only"],
            "path_code": [2],
            "current_ir": {},
            "terminal_reason": "all_children_failed_or_stage_gate_failed",
        }]
        shortlist = build_phase1_compile_shortlist(completed, fallback, 3)
        self.assertEqual([item["path"] for item in shortlist], [["all", "stages", "done"]])

    def test_compile_shortlist_keeps_parallel_balanced_and_reuse_tiles(self):
        completed = []
        for rank, (bm, bn, score) in enumerate(
            ((32, 64, 99.0), (64, 64, 90.0), (128, 64, 80.0), (32, 32, 70.0)),
            start=1,
        ):
            completed.append({
                "path": [f"Tiling.BlockTile.{bm}x{bn}x16", "Layout.done"],
                "path_code": [rank],
                "current_ir": {
                    "tiling": {"block_m": bm, "block_n": bn},
                    "resource": {"tiling_candidate": {"architecture_score": score}},
                },
            })

        shortlist = build_phase1_compile_shortlist(completed, [], 3)
        areas = {item["current_ir"]["tiling"]["block_m"] * item["current_ir"]["tiling"]["block_n"] for item in shortlist}
        self.assertEqual(areas, {2048, 4096, 8192})

    def test_pool_and_selection_keep_block_tiles_unique(self):
        candidates = build_joint_tiling_candidates(self.library, self.ir)
        pool = make_diverse_tiling_pool(candidates, 18)
        self.assertTrue(
            any(
                (item["BM"], item["BN"], item["BK"], item["WM"], item["WN"], item["TM"], item["TN"])
                == (64, 128, 16, 32, 32, 4, 4)
                for item in pool
            )
        )
        selected = select_diverse_joint_tilings(pool, [pool[0]["candidate_id"]] * 2 + ["invalid"], 6)
        block_tiles = [(item["BM"], item["BN"], item["BK"]) for item in selected]
        self.assertEqual(len(selected), 6)
        self.assertEqual(len(block_tiles), len(set(block_tiles)))

    def test_pool_preserves_high_score_and_balanced_tuple_per_block(self):
        state = {
            "problem": {"dtype": "fp32", "M": 1024, "N": 1024, "K": 1024},
            "hardware": {
                "compute_capability": "8.9",
                "sm_count": 34,
                "warp_size": 32,
                "max_threads_per_block": 1024,
                "max_threads_per_multiprocessor": 1536,
                "max_blocks_per_multiprocessor": 24,
                "max_shared_memory_per_block_bytes": 49152,
                "max_shared_memory_per_multiprocessor_bytes": 102400,
                "registers_per_multiprocessor": 65536,
            },
        }
        candidates = build_joint_tiling_candidates(self.library, state)
        pool = make_diverse_tiling_pool(candidates, 32)
        block_candidates = [
            item for item in pool
            if (item["BM"], item["BN"], item["BK"]) == (128, 128, 16)
        ]
        self.assertGreaterEqual(len(block_candidates), 2)
        self.assertTrue(any(
            (item["WM"], item["WN"], item["TM"], item["TN"]) == (32, 32, 4, 4)
            for item in block_candidates
        ))

    def test_low_reuse_llm_mapping_is_rebalanced_within_same_block(self):
        candidates = build_joint_tiling_candidates(self.library, self.ir)
        low_reuse = next(item for item in candidates if
            item["candidate_id"] == "tile_64x128x16__32x64__2x2")

        adjusted, changes = rebalance_low_reuse_tiling_choices(
            candidates,
            [low_reuse["candidate_id"]],
        )

        self.assertEqual(adjusted, ["tile_64x128x16__32x32__4x4"])
        self.assertEqual(changes[0]["block"], [64, 128, 16])
        self.assertGreater(changes[0]["fragment_reuse_after"], changes[0]["fragment_reuse_before"])

    def test_planned_warp_component_bypasses_new_choice(self):
        candidate = build_joint_tiling_candidates(self.library, self.ir)[0]
        plan = tiling_plan_for_ir(candidate)
        available = {"strategies": [{"strategy_id": plan["warp_strategy_id"]}]}
        selection = planned_tiling_micro_selection(
            {"strategy": {"tiling_joint_plan": plan}},
            "Tiling.WarpTileSelection",
            available,
        )
        self.assertEqual(selection["candidates"][0]["strategy_id"], plan["warp_strategy_id"])
        self.assertEqual(selection["selection_mode"], "planned_joint_tiling")

    def test_joint_prompt_does_not_include_history(self):
        prompt = ROOT / "llm" / "prompts" / "select_joint_tiling_prompt.txt"
        summary = compact_joint_tiling_ir_summary(
            {"problem": {"M": 512}, "hardware": {"warp_size": 32}, "strategy": {"history": ["secret"]}}
        )
        self.assertNotIn("strategy", summary)
        messages = build_joint_tiling_messages(summary, [{"candidate_id": "tile_a"}], 3, prompt)
        self.assertNotIn("benchmark history", messages[1]["content"].lower())
        validated = validate_joint_tiling_response(
            {"candidate_ids": ["bad", "tile_a", "tile_a"]},
            [{"candidate_id": "tile_a"}],
            3,
        )
        self.assertEqual(validated["candidate_ids"], ["tile_a"])

    def test_llm_selects_explicit_block_warp_thread_components(self):
        candidates = build_joint_tiling_candidates(self.library, self.ir)
        target = next(
            item
            for item in candidates
            if (item["BM"], item["BN"], item["BK"], item["WM"], item["WN"], item["TM"], item["TN"])
            == (64, 128, 16, 32, 64, 2, 4)
        )
        response = {
            "selections": [
                {
                    "block_strategy_id": target["block_strategy_id"],
                    "warp_strategy_id": target["warp_strategy_id"],
                    "thread_strategy_id": target["thread_strategy_id"],
                }
            ]
        }
        validated = validate_joint_tiling_response(response, candidates, 6)
        self.assertEqual(validated["candidate_ids"], [target["candidate_id"]])

        options = compact_joint_tiling_option_space(candidates)
        self.assertIn(target["block_strategy_id"], {item["block_strategy_id"] for item in options["block_options"]})
        self.assertIn(target["warp_strategy_id"], {item["warp_strategy_id"] for item in options["warp_options"]})
        self.assertIn(target["thread_strategy_id"], {item["thread_strategy_id"] for item in options["thread_options"]})

    def test_joint_plan_iteration_values_are_applied_with_thread_tile(self):
        candidate = build_joint_tiling_candidates(self.library, self.ir)[0]
        plan = tiling_plan_for_ir(candidate)
        updates = tiling_plan_ir_updates(plan["thread_strategy_id"], plan)
        self.assertEqual(updates["tiling.thread_m"], plan["TM"])
        self.assertEqual(updates["tiling.thread_n"], plan["TN"])
        self.assertEqual(updates["tiling.warp_tile.warp_m_iter"], plan["WMITER"])
        self.assertEqual(updates["tiling.warp_tile.warp_n_iter"], plan["WNITER"])

    def test_selected_joint_tiling_is_materialized_without_llm_codegen(self):
        candidate = next(
            item
            for item in build_joint_tiling_candidates(self.library, self.ir)
            if (item["BM"], item["BN"], item["BK"], item["WM"], item["WN"], item["TM"], item["TN"])
            == (64, 128, 16, 32, 64, 2, 4)
        )
        ir = {
            "tiling": {
                "enabled": True,
                "block_m": candidate["BM"],
                "block_n": candidate["BN"],
                "block_k": candidate["BK"],
                "thread_m": candidate["TM"],
                "thread_n": candidate["TN"],
                "warp_tile": {
                    "warp_m": candidate["WM"],
                    "warp_n": candidate["WN"],
                    "warp_m_iter": candidate["WMITER"],
                    "warp_n_iter": candidate["WNITER"],
                },
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            code_root = Path(directory)
            shutil.copy2(ROOT / "gemm_code" / "skeleton" / "cuda_kernel.cuh", code_root / "cuda_kernel.cuh")
            result, generated = apply_deterministic_code_patch(
                code_root,
                candidate["thread_strategy_id"],
                ir,
            )
            content = (code_root / "cuda_kernel.cuh").read_text(encoding="utf-8")

        self.assertEqual(result["method"], "deterministic_region_materialization")
        self.assertEqual(generated["generation_method"], "deterministic_region_materialization")
        for name in ("BM", "BN", "BK", "WM", "WN", "WMITER", "WNITER", "TM", "TN"):
            declaration = f"static const int {name} = {candidate[name]};"
            self.assertEqual(content.count(declaration), 1)

    def test_frontier_is_wide_until_layout_then_pruned(self):
        self.assertEqual(frontier_limit_for_stage("Tiling", "cuda", 3, 6, "Layout"), 6)
        self.assertEqual(frontier_limit_for_stage("Layout", "cuda", 3, 6, "Layout"), 6)
        self.assertEqual(frontier_limit_for_stage("Vectorization", "cuda", 3, 6, "Layout"), 3)
        frontier = [
            {"current_ir": {"performance": {"gflops": value}}}
            for value in (10.0, 30.0, 20.0, 40.0)
        ]
        pruned = prune_frontier_after_core_construction(frontier, 3)
        self.assertEqual([item["current_ir"]["performance"]["gflops"] for item in pruned], [40.0, 30.0, 20.0])

    def test_post_tiling_beam_keeps_top_three_per_block_tile_root(self):
        states = []
        for root, base in (("64x64x16", 10.0), ("128x64x16", 20.0)):
            for child in range(5):
                states.append(
                    {
                        "path": [
                            f"Tiling.BlockTile.{root}",
                            f"Tiling.WarpTile.{16 if child % 2 else 32}x32",
                            f"Tiling.ThreadTile.{2 if child % 2 else 4}x2",
                            f"Layout.Child.{child}",
                        ],
                        "current_ir": {"performance": {"gflops": base + child}},
                    }
                )

        kept = prune_frontier_per_tiling_lineage(states, 3)
        active, pending = take_frontier_batch_per_tiling_lineage(states, 3)

        self.assertEqual(len(tiling_lineage_keys(kept)), 2)
        self.assertEqual(len(kept), 6)
        self.assertEqual(len(active), 6)
        self.assertEqual(len(pending), 4)
        for key in tiling_lineage_keys(kept):
            scores = sorted(
                state["current_ir"]["performance"]["gflops"]
                for state in kept
                if tuple(state["path"][:1]) == key
            )
            self.assertEqual(len(scores), 3)

    def test_lazy_root_beam_keeps_one_active_path_per_block_tile(self):
        states = [
            {
                "path": [
                    f"Tiling.BlockTile.{root}",
                    f"Tiling.WarpTile.{warp}",
                    "Tiling.ThreadTile.2x2",
                ],
                "current_ir": {"performance": {"gflops": score}},
            }
            for root, warp, score in (
                ("64x64x16", "16x32", 10.0),
                ("64x64x16", "32x32", 20.0),
                ("128x64x16", "32x32", 30.0),
                ("128x64x16", "64x32", 40.0),
            )
        ]

        active, pending = take_frontier_batch_per_tiling_lineage(states, 1)

        self.assertEqual(len(active), 2)
        self.assertEqual(len(pending), 2)
        self.assertEqual(len(tiling_lineage_keys(active)), 2)
        self.assertEqual(
            sorted(state["current_ir"]["performance"]["gflops"] for state in active),
            [20.0, 40.0],
        )


if __name__ == "__main__":
    unittest.main()
