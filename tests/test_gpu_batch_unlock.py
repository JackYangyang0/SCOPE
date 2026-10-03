from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

from SCOPE.app import (
    PHASE2_LARGE_MATRIX,
    apply_deterministic_code_patch,
    all_available_strategies_are_deterministic,
    build_matrix_profile,
    build_phase2_unlock_strategy_index,
    build_unlock_bundle_strategy,
    expand_unlock_plan_to_strategy_steps,
    is_deterministic_strategy,
    resource_profile_reject_reason,
    sanitize_unlock_batch_plan,
    unlock_experiment_compilation_budget,
)
from SCOPE.utils.common_utils import load_json
from SCOPE.llm.patch_generator import load_code_context
from SCOPE.verification.code_ast_extractor import extract_code_ast


class GPUBatchUnlockTests(unittest.TestCase):
    def test_unlock_batches_are_executed_as_single_strategy_steps(self):
        plan = {
            "batches": [
                {
                    "batch_id": "load_pipeline",
                    "purpose": "ordered coupled candidates",
                    "strategy_ids": ["Strategy.A", "Strategy.B"],
                }
            ]
        }

        steps = expand_unlock_plan_to_strategy_steps(plan)

        self.assertEqual([item["strategy_ids"] for item in steps], [["Strategy.A"], ["Strategy.B"]])
        self.assertEqual([item["batch_id"] for item in steps], ["load_pipeline.s1", "load_pipeline.s2"])
        self.assertTrue(all(item["execution_granularity"] == "single_strategy" for item in steps))

    def test_single_unlock_step_uses_original_strategy_contract(self):
        strategy = make_strategy("Compiler.UnrollHint", phase="phase2_unlock_general")
        result = build_unlock_bundle_strategy(
            {"batch_id": "compiler.s1", "strategy_ids": ["Compiler.UnrollHint"]},
            {"strategies": [strategy]},
            {"stages": []},
        )

        self.assertEqual(result["strategy_id"], "Compiler.UnrollHint")
        self.assertEqual(result["unlock_execution"]["execution_granularity"], "single_strategy")

    def test_feedback_budget_reserves_trials_for_all_rounds(self):
        self.assertEqual(unlock_experiment_compilation_budget(1, 1, 0), 2)
        self.assertEqual(unlock_experiment_compilation_budget(1, 2, 2), 1)
        self.assertEqual(unlock_experiment_compilation_budget(2, 1, 4), 2)

    def test_sanitizer_does_not_repeat_failed_bundle(self):
        index = {
            "strategies": [
                make_strategy("Compiler.UnrollHint", phase="phase2_unlock_general"),
                make_strategy("Tuning.LaunchBounds", phase="phase2_unlock_general"),
            ]
        }
        plan = {"batches": [{
            "batch_id": "repeat",
            "strategy_ids": ["Compiler.UnrollHint", "Tuning.LaunchBounds"],
        }]}
        result = sanitize_unlock_batch_plan(
            plan,
            index,
            {"large": False},
            {"failed_unlock_bundles": ["Compiler.UnrollHint|Tuning.LaunchBounds"]},
        )
        self.assertFalse(any(batch.get("batch_id") == "repeat" for batch in result["batches"]))

    def test_gpu_strategy_catalog_and_subphase_counts_are_aligned(self):
        root = Path(__file__).resolve().parents[1]
        index = load_json(root / "data" / "lib" / "strategy_index.json")
        library = load_json(root / "data" / "lib" / "strategy_library.json")
        index_ids = {item["strategy_id"] for item in index["strategies"]}
        library_ids = {
            item["strategy_id"]
            for stage in library["stages"]
            for subphase in stage["ordered_subphases"]
            for item in subphase["strategies"]
        }

        self.assertEqual(index_ids, library_ids)
        self.assertIn("Pipeline.CpAsync.Multistage3.SharedAB", index_ids)
        self.assertIn("Scheduling.PersistentCTA.GridStrideTiles", index_ids)
        for stage in index["stages"].values():
            for subphase in stage["subphases"]:
                self.assertEqual(subphase["candidate_count"], len(subphase["allowed_strategy_ids"]))

    def test_cp_async_stage_count_is_filtered_by_shared_memory_capacity(self):
        ir = make_ir(m=4096, n=4096, k=4096, sm_count=108)
        ir["hardware"]["gpu"]["max_shared_memory_per_block_bytes"] = 49152
        profile = build_matrix_profile(ir)

        self.assertIsNone(resource_profile_reject_reason("Pipeline.CpAsync.Multistage3.SharedAB", ir, profile))
        self.assertIn(
            "above hardware limit",
            resource_profile_reject_reason("Pipeline.CpAsync.Multistage4.SharedAB", ir, profile),
        )

    def test_executor_required_strategies_are_not_unlock_candidates(self):
        ir = make_ir(m=4096, n=4096, k=4096, sm_count=108)
        index = {
            "strategies": [
                make_strategy("Reduction.SplitK.Workspace", phase="excluded_default"),
                make_strategy("Scheduling.PersistentCTA.GridStrideTiles", phase=PHASE2_LARGE_MATRIX),
            ]
        }

        filtered = build_phase2_unlock_strategy_index(
            index,
            ir,
            {},
            {"applied_strategy_ids": []},
            build_matrix_profile(ir),
        )

        self.assertEqual(
            [item["strategy_id"] for item in filtered["strategies"]],
            ["Scheduling.PersistentCTA.GridStrideTiles"],
        )

    def test_cp_async_supersedes_phase1_sync_copy_after_requirements_pass(self):
        root = Path(__file__).resolve().parents[1]
        index = load_json(root / "data" / "lib" / "strategy_index.json")
        graph = load_json(root / "data" / "graph" / "dependency_graph.json")
        ir = make_ir(m=4096, n=4096, k=4096, sm_count=108)
        ir["hardware"].update({"supports_cp_async": True, "max_shared_memory_per_block_bytes": 49152})
        ir["hardware"]["gpu"]["max_shared_memory_per_block_bytes"] = 49152
        ir["memory"].update({"use_shared_memory": True, "use_register_tile": True})
        ir["resource"]["shared_memory"] = {"total_bytes": 16384}
        history = {
            "applied_strategy_ids": [
                "Layout.SharedMemory.AB.Basic",
                "Reordering.LoadCompute.SeparatePhases",
                "Pipeline.NoAsyncCopy.V1",
            ]
        }

        filtered = build_phase2_unlock_strategy_index(index, ir, graph, history, build_matrix_profile(ir))
        candidate_ids = {item["strategy_id"] for item in filtered["strategies"]}

        self.assertIn("Pipeline.CpAsync.Multistage3.SharedAB", candidate_ids)
        self.assertNotIn("Pipeline.CpAsync.Multistage4.SharedAB", candidate_ids)

    def test_small_matrix_filters_large_alias_and_applied(self):
        ir = make_ir(m=512, n=512, k=512, sm_count=108)
        index = {
            "strategies": [
                make_strategy("Vectorization.GlobalLoadAB.float4", phase="phase2_unlock_general"),
                make_strategy("Vectorization.StoreC.float4", phase="phase2_unlock_general", alias_of="Epilogue.StoreC.Vectorized.float4"),
                make_strategy("Memory.L2Reuse.CTASwizzleGroupedN", phase=PHASE2_LARGE_MATRIX),
                make_strategy("Compiler.LaunchBounds.256", phase="excluded_default"),
                make_strategy("Pipeline.DoubleBuffer.SharedAB", phase="phase2_unlock_general"),
            ]
        }
        history = {"applied_strategy_ids": ["Pipeline.DoubleBuffer.SharedAB"], "failed_strategy_counts": {}}

        filtered = build_phase2_unlock_strategy_index(index, ir, {}, history, build_matrix_profile(ir))

        self.assertEqual([item["strategy_id"] for item in filtered["strategies"]], ["Vectorization.GlobalLoadAB.float4"])

    def test_large_matrix_allows_large_matrix_strategies(self):
        ir = make_ir(m=4096, n=4096, k=4096, sm_count=108, bm=128, bn=128, bk=16)
        index = {"strategies": [make_strategy("Memory.L2Reuse.CTASwizzleGroupedN", phase=PHASE2_LARGE_MATRIX)]}

        filtered = build_phase2_unlock_strategy_index(index, ir, {}, {"applied_strategy_ids": []}, build_matrix_profile(ir))

        self.assertEqual(filtered["strategies"][0]["strategy_id"], "Memory.L2Reuse.CTASwizzleGroupedN")

    def test_square_1024_is_throughput_sized_for_unlock(self):
        ir = make_ir(m=1024, n=1024, k=1024, sm_count=170, bm=64, bn=64, bk=16)
        profile = build_matrix_profile(ir)

        self.assertTrue(profile["large_mn"])
        self.assertTrue(profile["large"])

    def test_long_k_sync_pipeline_gets_guarded_async_exploration_batch(self):
        index = {
            "strategies": [
                {
                    **make_strategy("Pipeline.CpAsync.Multistage3.SharedAB", phase="phase2_unlock_general"),
                    "requires_bundle_strategy_ids": ["Memory.SharedMemory.DynamicOptIn"],
                },
                make_strategy("Memory.SharedMemory.DynamicOptIn", phase="phase2_unlock_general"),
            ]
        }
        plan = sanitize_unlock_batch_plan(
            {"batches": [{"batch_id": "other", "strategy_ids": ["Memory.SharedMemory.DynamicOptIn"]}]},
            index,
            {"large": True, "k_tiles": 64},
            {"applied_strategy_ids": ["Pipeline.NoAsyncCopy.V1"]},
        )

        async_batch = next(batch for batch in plan["batches"] if batch["batch_id"] == "async_pipeline_upgrade")
        self.assertEqual(
            async_batch["strategy_ids"],
            ["Memory.SharedMemory.DynamicOptIn", "Pipeline.CpAsync.Multistage3.SharedAB"],
        )

    def test_batch_plan_sanitizer_removes_invalid_duplicates_and_conflicts(self):
        index = {
            "strategies": [
                make_strategy("Vectorization.GlobalLoadAB.float4", phase="phase2_unlock_general", conflict_group="load"),
                make_strategy("Reordering.CooperativeVectorLoadAB.float4", phase="phase2_unlock_general", conflict_group="load"),
                make_strategy("Epilogue.StoreC.Vectorized.float4", phase="phase2_unlock_general", conflict_group="store"),
            ]
        }
        plan = {
            "batches": [
                {
                    "batch_id": "b/1",
                    "strategy_ids": [
                        "Vectorization.GlobalLoadAB.float4",
                        "Vectorization.GlobalLoadAB.float4",
                        "Reordering.CooperativeVectorLoadAB.float4",
                        "Missing.Strategy",
                        "Epilogue.StoreC.Vectorized.float4",
                    ],
                }
            ]
        }

        cleaned = sanitize_unlock_batch_plan(plan, index, {"large": False}, {"applied_strategy_ids": []})

        self.assertEqual(
            cleaned["batches"][0]["strategy_ids"],
            ["Vectorization.GlobalLoadAB.float4", "Epilogue.StoreC.Vectorized.float4"],
        )
        self.assertEqual(cleaned["batch_order"], ["b_1"])

    def test_cuda_llm_context_and_ast_exclude_main_harness(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "main.cpp").write_text("int main(){return 0;}\n", encoding="utf-8")
            (root / "kernel.h").write_text("void cuda_gemm();\n", encoding="utf-8")
            (root / "cuda_kernel.cuh").write_text("__global__ void gemm_kernel(){}\n", encoding="utf-8")

            context = load_code_context(root)
            ast = extract_code_ast(root)

        self.assertNotIn("main.cpp", context["files"])
        self.assertIn("kernel.h", context["files"])
        self.assertIn("cuda_kernel.cuh", context["files"])
        self.assertEqual(set(ast["files"]), {"cuda_kernel.cuh"})

    def test_deterministic_strategy_classifier_keeps_complex_codegen_on_llm_path(self):
        self.assertTrue(is_deterministic_strategy("Tiling.BlockTile.128x128x16"))
        self.assertTrue(is_deterministic_strategy("Layout.SharedMemory.AB.Basic"))
        self.assertTrue(is_deterministic_strategy("Epilogue.StoreC.CoalescedScalar"))
        self.assertFalse(is_deterministic_strategy("Reordering.LoadCompute.SeparatePhases"))
        self.assertFalse(is_deterministic_strategy("Vectorization.GlobalLoadAB.float4"))
        self.assertFalse(is_deterministic_strategy("Pipeline.DoubleBuffer.SharedAB"))
        self.assertFalse(is_deterministic_strategy("CPU.Tiling.L2Block.128x128x256"))

    def test_all_deterministic_candidates_bypass_llm_selection(self):
        self.assertTrue(
            all_available_strategies_are_deterministic(
                {
                    "strategies": [
                        {"strategy_id": "Layout.SharedMemory.AB.Basic"},
                        {"strategy_id": "Register.AccumulatorLayout.2DArray"},
                    ]
                }
            )
        )
        self.assertFalse(
            all_available_strategies_are_deterministic(
                {"strategies": [{"strategy_id": "Vectorization.GlobalLoadAB.float4"}]}
            )
        )

    def test_deterministic_materializer_replaces_only_requested_anchor_region(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "cuda_kernel.cuh").write_text(
                "\n".join(
                    [
                        "void cuda_gemm() {",
                        "    /*",
                        "     * LAUNCH_CONFIG_BEGIN",
                        "     */",
                        "    static const int BM = 1;",
                        "    static const int BN = 1;",
                        "    static const int BK = 1;",
                        "    /*",
                        "     * LAUNCH_CONFIG_END",
                        "     */",
                        "}",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            result, generated = apply_deterministic_code_patch(
                root,
                "Tiling.BlockTile.128x64x16",
                {
                    "tiling": {
                        "block_m": 128,
                        "block_n": 64,
                        "block_k": 16,
                        "warp_tile": {"warp_m": 32, "warp_n": 32, "warp_m_iter": 8, "warp_n_iter": 16},
                        "thread_m": 4,
                        "thread_n": 4,
                    }
                },
            )

        self.assertEqual(result["status"], "pass")
        self.assertIn("static const int BM = 128;", generated["files"][0]["content"])
        self.assertIn("static const int BN = 64;", generated["files"][0]["content"])
        self.assertIn("LAUNCH_CONFIG_BEGIN", generated["files"][0]["content"])

    def test_load_compute_cannot_rebuild_kernel_deterministically(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "cuda_kernel.cuh").write_text("#pragma once\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "full-kernel materialization is forbidden"):
                apply_deterministic_code_patch(root, "Reordering.LoadCompute.SeparatePhases", {})
            self.assertEqual((root / "cuda_kernel.cuh").read_text(encoding="utf-8"), "#pragma once\n")

    def test_deterministic_materializer_cleans_malformed_stale_launch_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "cuda_kernel.cuh").write_text(
                "\n".join(
                    [
                        "void cuda_gemm() {",
                        "    /*",
                        "     * LAUNCH_CONFIG_BEGIN",
                        "     */",
                        "    static const int BM = 128;",
                        "    static const int BN = 128;",
                        "    static const int BK = 16;",
                        "    static const int BM = 64;",
                        "    static const int BN = 64;",
                        "    static const int BK = 8;",
                        "     * LAUNCH_CONFIG_END",
                        "     */",
                        "}",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            result, generated = apply_deterministic_code_patch(
                root,
                "Tiling.BlockTile.32x32x8",
                {
                    "tiling": {
                        "block_m": 32,
                        "block_n": 32,
                        "block_k": 8,
                        "warp_tile": {"warp_m": 32, "warp_n": 32, "warp_m_iter": 1, "warp_n_iter": 1},
                        "thread_m": 1,
                        "thread_n": 1,
                    }
                },
            )

        content = generated["files"][0]["content"]
        self.assertEqual(result["status"], "pass")
        self.assertEqual(content.count("static const int BM ="), 1)
        self.assertEqual(content.count("static const int BN ="), 1)
        self.assertIn("    /*\n     * LAUNCH_CONFIG_END", content)


def make_ir(m: int, n: int, k: int, sm_count: int, bm: int = 128, bn: int = 128, bk: int = 16) -> dict:
    return {
        "problem": {"M": m, "N": n, "K": k},
        "tiling": {"block_m": bm, "block_n": bn, "block_k": bk, "thread_m": 4, "thread_n": 4},
        "hardware": {"gpu": {"sm_count": sm_count, "max_shared_memory_per_block_bytes": 98304}},
        "memory": {"shared_memory_bytes": 4 * (bm * bk + bk * bn)},
        "resource": {"estimated_registers_per_thread": 48},
    }


def make_strategy(
    strategy_id: str,
    phase: str,
    alias_of: str | None = None,
    conflict_group: str = "",
) -> dict:
    return {
        "strategy_id": strategy_id,
        "stage": "Vectorization",
        "phase": phase,
        "alias_of": alias_of,
        "canonical_strategy_id": alias_of or strategy_id,
        "conflict_group": conflict_group,
        "modifies_regions": ["global_load"],
        "provides_fields": [],
        "requires_fields": [],
    }


if __name__ == "__main__":
    unittest.main()
