from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from SCOPE.generate_ir.stage_controller import StageController
from SCOPE.generate_ir.ir_checker import check_before_codegen
from SCOPE.generate_ir.ir_extraction import extract_problem_description
from SCOPE.generate_ir.stage_controller import evaluate_predicate_like, evaluate_text_condition, synthesize_ir_updates
from SCOPE.utils.common_utils import load_json
from SCOPE.app import (
    DEFAULT_POSTCHECK_OUTPUT,
    DEFAULT_UNLOCKED_PERFORMANCE_PROFILE,
    apply_strategy_profile_to_candidates,
    apply_strategy_profile_to_stage_order,
    build_top_terminal_results,
    candidate_path,
    chain_code_path,
    chain_id_for_state,
    chain_key_for_path,
    compact_hierarchical_tiling_ir_summary,
    infer_app_stage_order,
    inject_structurally_diverse_candidates,
    load_dependency_graph,
    load_strategy_documents,
    merge_expected_ir_updates,
    merge_terminal_verifications,
    select_single_path_candidates,
    select_top_k_candidates,
    stage_failure_requires_strategy_fallback,
    take_frontier_batch,
)
from SCOPE.llm.concrete_code_generator import ensure_generated_code_has_minimal_kernel, validate_generated_code_materialization
from SCOPE.verification.build_run_verifier import parse_run_metrics
from SCOPE.verification.gemm_semantic_repair import (
    build_throughput_cuda_kernel_cuh,
    normalize_throughput_launch_config,
    should_emit_throughput_kernel,
)


SCOPE_ROOT = Path(__file__).resolve().parents[1]
STRATEGY_INDEX = load_json(SCOPE_ROOT / "data" / "lib" / "strategy_index.json")
STRATEGY_LIBRARY = load_json(SCOPE_ROOT / "data" / "lib" / "strategy_library.json")
CPU_STRATEGY_INDEX = load_json(SCOPE_ROOT / "data" / "lib" / "cpu_strategy_index.json")
CPU_STRATEGY_LIBRARY = load_json(SCOPE_ROOT / "data" / "lib" / "cpu_strategy_library.json")


class StageControllerCompletionTests(unittest.TestCase):
    def controller(self, ir, history=None):
        return StageController(
            strategy_index=STRATEGY_INDEX,
            workflow_library=STRATEGY_LIBRARY,
            current_stage=(history or {}).get("current_stage", "Tiling"),
            optir=ir,
            history=history or {},
        )

    def test_stage_controller_accepts_structured_predicate_schema(self):
        ir = {"tiling": {"block_k": 16}, "problem": {"K": 512}, "memory": {"use_register_tile": True}}

        self.assertEqual(
            evaluate_predicate_like(
                ir,
                {
                    "id": "BK_AT_LEAST_8",
                    "kind": "ir_predicate",
                    "op": "ge",
                    "lhs": {"field": "tiling.block_k"},
                    "rhs": {"const": 8},
                },
            ).status,
            "pass",
        )
        self.assertEqual(
            evaluate_predicate_like(
                ir,
                {
                    "id": "THREAD_OUTPUT_MAPPING_READY",
                    "kind": "ir_predicate",
                    "op": "is_not_null",
                    "lhs": {"field": "mapping.thread_to_output"},
                },
            ).status,
            "fail",
        )

    def test_stage_controller_accepts_not_null_schema_alias(self):
        ir = {"problem": {"M": 512, "N": 512, "K": 512}}
        controller = self.controller(ir)
        candidates = controller.get_subphase_candidates()
        available_ids = {item["strategy_id"] for item in candidates["strategies"]}

        self.assertIn("Tiling.BlockTile.32x128x16", available_ids)
        self.assertIn("Tiling.BlockTile.64x64x32", available_ids)
        self.assertFalse(
            any(
                item["strategy_id"] == "Tiling.BlockTile.32x128x16"
                and item["reason"].startswith("preconditions failed")
                for item in candidates["filter_context"]["rejected"]
            )
        )

    def test_retired_warp_iteration_strategy_is_not_a_thread_tile_candidate(self):
        ir = {
            "problem": {"M": 512, "N": 512, "K": 512},
            "hardware": {"warp_size": 32},
            "tiling": {
                "block_m": 64,
                "block_n": 128,
                "block_k": 16,
                "warp_tile": {"warp_m": 32, "warp_n": 64},
            },
        }
        history = {
            "completed_subphases": ["Tiling.BlockTileSelection", "Tiling.WarpTileSelection"],
            "applied_strategy_ids": ["Tiling.BlockTile.64x128x16", "Tiling.WarpTile.32x64"],
        }
        candidates = self.controller(ir, history).get_subphase_candidates()
        available_ids = {item["strategy_id"] for item in candidates["strategies"]}

        self.assertNotIn("Tiling.WarpTile.Iteration.WMITERxWNITER", available_ids)

    def test_warp_candidates_are_filtered_by_selected_block(self):
        ir = {
            "problem": {"M": 512, "N": 512, "K": 512},
            "hardware": {"warp_size": 32, "max_threads_per_block": 1024},
            "tiling": {"enabled": True, "block_m": 64, "block_n": 32, "block_k": 16},
        }
        history = {
            "completed_subphases": ["Tiling.BlockTileSelection"],
            "applied_strategy_ids": ["Tiling.BlockTile.64x32x16"],
        }
        candidates = self.controller(ir, history).get_subphase_candidates()
        available_ids = {item["strategy_id"] for item in candidates["strategies"]}
        rejected = {item["strategy_id"]: item["reason"] for item in candidates["filter_context"]["rejected"]}

        self.assertIn("Tiling.WarpTile.32x32", available_ids)
        self.assertIn("Tiling.WarpTile.64x32", available_ids)
        self.assertNotIn("Tiling.WarpTile.32x64", available_ids)
        self.assertIn("does not divide BlockTile", rejected["Tiling.WarpTile.32x64"])
        self.assertNotIn("Tiling.WarpTile.WMxWN", available_ids)
        self.assertNotIn("Tiling.WarpTile.ParametricWMxWN", available_ids)
        self.assertIn("not selectable", rejected["Tiling.WarpTile.WMxWN"])

    def test_thread_candidates_are_filtered_by_selected_warp(self):
        ir = {
            "problem": {"M": 512, "N": 512, "K": 512},
            "hardware": {"warp_size": 32, "max_threads_per_block": 1024},
            "tiling": {
                "enabled": True,
                "block_m": 32,
                "block_n": 32,
                "block_k": 8,
                "warp_tile": {"warp_m": 16, "warp_n": 16},
            },
        }
        history = {
            "completed_subphases": ["Tiling.BlockTileSelection", "Tiling.WarpTileSelection"],
            "applied_strategy_ids": ["Tiling.BlockTile.32x32x8", "Tiling.WarpTile.16x16"],
        }
        candidates = self.controller(ir, history).get_subphase_candidates()
        available_ids = {item["strategy_id"] for item in candidates["strategies"]}
        rejected = {item["strategy_id"]: item["reason"] for item in candidates["filter_context"]["rejected"]}

        self.assertIn("Tiling.ThreadTile.2x2", available_ids)
        self.assertNotIn("Tiling.ThreadTile.4x8", available_ids)
        self.assertIn("no legal 32-lane mapping", rejected["Tiling.ThreadTile.4x8"])

    def test_hierarchical_tiling_summary_excludes_search_history_and_performance(self):
        summary = compact_hierarchical_tiling_ir_summary(
            {
                "problem": {"M": 512, "N": 512, "K": 512},
                "hardware": {"warp_size": 32},
                "tiling": {"block_m": 64, "block_n": 32, "block_k": 16},
                "mapping": {"threads_per_block": 64},
                "strategy": {"applied_strategy_ids": ["Tiling.BlockTile.64x32x16"]},
                "performance": {"gflops": 9999.0},
                "verification": {"correctness": {"status": "pass"}},
            }
        )

        self.assertEqual(summary["tiling"]["block_m"], 64)
        self.assertEqual(summary["mapping"]["threads_per_block"], 64)
        self.assertNotIn("strategy", summary)
        self.assertNotIn("performance", summary)
        self.assertNotIn("verification", summary)

    def test_launch_resource_failure_requests_strategy_fallback(self):
        ir = {
            "verification": {
                "runtime_safety": {
                    "status": "fail",
                    "cuda_error": "too many resources requested for launch",
                }
            }
        }

        self.assertTrue(stage_failure_requires_strategy_fallback(ir))
        self.assertFalse(
            stage_failure_requires_strategy_fallback(
                {"verification": {"runtime_safety": {"cuda_error": "illegal memory access"}}}
            )
        )

    def test_precondition_rejection_reports_unknown_predicate_reason(self):
        ir = {"problem": {"M": 512, "N": 512, "K": 512}}
        controller = self.controller(ir)
        strategy = {
            "strategy_id": "Test.UnsupportedPredicate",
            "preconditions": {
                "predicates": [
                    {
                        "id": "PRE_UNSUPPORTED",
                        "kind": "ir_predicate",
                        "op": "not_a_real_operator",
                        "lhs": {"field": "problem.M"},
                        "rhs": {"const": 1},
                    }
                ]
            },
        }

        report = check_before_codegen(ir, strategy)
        failed = [item for item in report["results"] if item["status"] == "unknown"]

        self.assertEqual(failed[0]["id"], "PRECONDITION:PRE_UNSUPPORTED")
        self.assertEqual(failed[0]["detail"]["evaluated"][0]["reason"], "unsupported_op")

    def test_throughput_repair_uses_bounded_flattened_shared_loads(self):
        source = build_throughput_cuda_kernel_cuh(
            {
                "BM": 64,
                "BN": 128,
                "BK": 16,
                "WM": 16,
                "WN": 32,
                "WMITER": 8,
                "WNITER": 16,
                "TM": 2,
                "TN": 2,
            }
        )

        self.assertIn("loadIdx < BM * BK", source)
        self.assertIn("loadIdx < BK * BN", source)
        self.assertNotIn("load_a_smem_m = tid / (BK / 4)", source)

    def test_vectorization_stage_constraints_are_executable(self):
        ir = {
            "vectorization": {
                "A": {"vector_width": 4, "alignment_guard": False, "alignment_proven": False, "tail_handling": True},
                "B": {"vector_width": 1, "alignment_guard": False, "alignment_proven": False, "tail_handling": True},
                "C": {"vector_width": 1, "alignment_guard": False, "alignment_proven": False, "tail_handling": True},
            },
            "memory": {"global_store_C": {"boundary_guard": True}},
            "safety": {},
        }

        self.assertEqual(evaluate_text_condition(ir, "C_VECTOR_ALIGNMENT").status, "fail")
        ir["vectorization"]["A"]["alignment_guard"] = True
        self.assertEqual(evaluate_text_condition(ir, "C_VECTOR_ALIGNMENT").status, "pass")

    def test_default_thread_tile_does_not_skip_thread_tile_selection(self):
        ir = {
            "tiling": {
                "block_m": 64,
                "block_n": 64,
                "block_k": 16,
                "thread_m": 1,
                "thread_n": 1,
                "warp_tile": {"warp_m": 32, "warp_n": 64},
            },
            "hardware": {"warp_size": 32},
            "field_meta": {
                "tiling.thread_m": {"origin": "default_template", "resolved": False},
                "tiling.thread_n": {"origin": "default_template", "resolved": False},
            },
        }
        history = {
            "completed_subphases": ["Tiling.BlockTileSelection", "Tiling.WarpTileSelection"],
            "applied_strategy_ids": ["Tiling.BlockTile.64x64x16", "Tiling.WarpTile.32x64"],
        }
        self.assertEqual(self.controller(ir, history).current_subphase_id(), "Tiling.ThreadTileSelection")

    def test_pipeline_shared_memory_alias_predicate_passes(self):
        ir = {
            "resource": {
                "shared_memory": {
                    "total_bytes": 16384,
                    "pipeline_multiplier": 1,
                }
            },
            "hardware": {"max_shared_memory_per_block_bytes": 49152},
        }

        old_result = evaluate_text_condition(
            ir,
            "shared_memory_bytes * pipeline_multiplier <= hardware.max_shared_memory_per_block_bytes",
        )
        new_result = evaluate_text_condition(
            ir,
            "resource.shared_memory.total_bytes <= hardware.max_shared_memory_per_block_bytes",
        )

        self.assertEqual(old_result.status, "pass")
        self.assertEqual(new_result.status, "pass")

    def test_cpu_backend_loads_cpu_strategy_graph(self):
        ir = {
            "target": {"backend": "cpu", "language": "c"},
            "problem": {"M": 512, "N": 512, "K": 512},
            "strategy": {"profile": "cpu_stable", "stage_order": ["Tiling", "Layout", "Vectorization"]},
        }
        strategy_index, strategy_library = load_strategy_documents(ir)
        graph = load_dependency_graph(ir)
        stage_order = infer_app_stage_order(ir, graph, strategy_index, strategy_library)

        self.assertEqual(stage_order[0], "CPUTiling")
        self.assertNotIn("Tiling", stage_order)
        self.assertIn("CPUPacking", stage_order)
        self.assertIn("CPUMicroKernel", stage_order)
        self.assertIn("CPULoopSchedule", stage_order)
        self.assertIn("CPUCompiler", stage_order)
        self.assertIn("CPUTiling", strategy_index["stages"])

    def test_cpu_stage_controller_uses_cpu_subphase_candidates(self):
        ir = {
            "target": {"backend": "cpu", "language": "c"},
            "problem": {"M": 512, "N": 512, "K": 512},
            "hardware": {"cpu_cache": {"l1_data_cache_bytes": 32768, "l2_cache_bytes": 1048576}},
        }
        controller = StageController(
            strategy_index=CPU_STRATEGY_INDEX,
            workflow_library=CPU_STRATEGY_LIBRARY,
            current_stage="CPUTiling",
            optir=ir,
            history={},
        )
        candidates = controller.get_subphase_candidates()

        self.assertEqual(controller.current_subphase_id(), "CPUTiling.L2BlockSelection")
        self.assertEqual(candidates["strategies"][0]["strategy_id"], "CPU.Tiling.L2Block.128x128x128")

    def test_cpu_medium_problem_keeps_packing_candidates(self):
        ir = {
            "target": {"backend": "cpu", "language": "c"},
            "problem": {"M": 512, "N": 512, "K": 512, "size_class": "medium"},
            "cpu_tiling": {"l1_block_k": 64, "l1_block_n": 32, "register_m": 4, "register_n": 8},
        }
        controller = StageController(
            strategy_index=CPU_STRATEGY_INDEX,
            workflow_library=CPU_STRATEGY_LIBRARY,
            current_stage="CPUPacking",
            optir=ir,
            history={},
        )

        candidate_ids = {
            item["strategy_id"] for item in controller.get_subphase_candidates()["strategies"]
        }
        self.assertIn("CPU.Memory.NoPack", candidate_ids)
        self.assertIn("CPU.Packing.PackB.KCxNR", candidate_ids)
        self.assertIn("CPU.Packing.PackAB.MRxKC_KCxNR", candidate_ids)

    def test_cpu_strategy_updates_cpu_ir_fields(self):
        updates = synthesize_ir_updates("CPU.Tiling.L2Block.128x128x128")

        self.assertEqual(updates["target.backend"], "cpu")
        self.assertEqual(updates["cpu_tiling.l2_block_m"], 128)
        self.assertEqual(updates["cpu_tiling.l2_block_n"], 128)
        self.assertEqual(updates["cpu_tiling.l2_block_k"], 128)

    def test_cpu_l1_tile_is_checked_against_l1_budget(self):
        ir = {
            "target": {"backend": "cpu", "language": "c"},
            "problem": {"M": 512, "N": 512, "K": 512},
            "hardware": {"cpu_cache": {"l1_data_cache_bytes": 32768, "l2_cache_bytes": 1048576}},
            "cpu_tiling": {"l2_block_m": 128, "l2_block_n": 128, "l2_block_k": 128},
            "cpu_resource": {"l2_working_set_bytes": 196608, "l2_budget_bytes": 838860},
            "field_meta": {
                "cpu_tiling.l2_block_m": {"origin": "micro_strategy", "resolved": True, "subphase": "CPUTiling.L2BlockSelection"},
                "cpu_tiling.l2_block_n": {"origin": "micro_strategy", "resolved": True, "subphase": "CPUTiling.L2BlockSelection"},
                "cpu_tiling.l2_block_k": {"origin": "micro_strategy", "resolved": True, "subphase": "CPUTiling.L2BlockSelection"},
            },
            "strategy": {"completed_subphases": ["CPUTiling.L2BlockSelection"]},
        }
        controller = StageController(
            strategy_index=CPU_STRATEGY_INDEX,
            workflow_library=CPU_STRATEGY_LIBRARY,
            current_stage="CPUTiling",
            optir=ir,
            history={"completed_subphases": ["CPUTiling.L2BlockSelection"]},
        )

        checked_ir = controller.apply_micro_strategy(
            "CPU.Tiling.L1Block.32x64x64",
            synthesize_ir_updates("CPU.Tiling.L1Block.32x64x64"),
        )
        local_check = checked_ir["stage_controller"]["last_local_check"]

        self.assertFalse(local_check["accepted"])
        self.assertIn("CPU_L1_WORKING_SET_FITS", [item["id"] for item in local_check["results"]])

    def test_explicit_thread_tile_allows_mapping_derivation(self):
        ir = {
            "tiling": {"block_m": 64, "block_n": 64, "block_k": 16},
            "hardware": {"warp_size": 32, "max_threads_per_block": 1024},
            "mapping": {},
        }
        history = {"applied_strategy_ids": [], "completed_subphases": [], "applied_micro_strategies": []}
        ir = self.controller(ir, history).apply_micro_strategy("Tiling.BlockTile.64x64x16")
        history = self.history_from_ir(ir)
        ir = self.controller(ir, history).apply_micro_strategy("Tiling.WarpTile.32x64")
        history = self.history_from_ir(ir)
        ir = self.controller(ir, history).apply_micro_strategy("Tiling.ThreadTile.4x4")
        history = self.history_from_ir(ir)

        self.assertIn("Tiling.ThreadTileSelection", history["completed_subphases"])
        self.assertEqual(ir["field_meta"]["tiling.thread_m"]["resolved_by"], "Tiling.ThreadTile.4x4")
        self.assertEqual(ir["field_meta"]["tiling.thread_n"]["resolved_by"], "Tiling.ThreadTile.4x4")
        self.assertEqual(self.controller(ir, history).current_subphase_id(), "Tiling.MappingDerivation")

    def test_mapping_derivation_requires_warp_fields(self):
        ir = {
            "tiling": {
                "block_m": 64,
                "block_n": 64,
                "block_k": 16,
                "thread_m": 4,
                "thread_n": 4,
            },
            "hardware": {"warp_size": 32},
            "field_meta": {
                "tiling.block_m": {"origin": "micro_strategy", "resolved": True, "resolved_by": "Tiling.BlockTile.64x64x16", "subphase": "Tiling.BlockTileSelection"},
                "tiling.block_n": {"origin": "micro_strategy", "resolved": True, "resolved_by": "Tiling.BlockTile.64x64x16", "subphase": "Tiling.BlockTileSelection"},
                "tiling.block_k": {"origin": "micro_strategy", "resolved": True, "resolved_by": "Tiling.BlockTile.64x64x16", "subphase": "Tiling.BlockTileSelection"},
                "tiling.thread_m": {"origin": "micro_strategy", "resolved": True, "resolved_by": "Tiling.ThreadTile.4x4", "subphase": "Tiling.ThreadTileSelection"},
                "tiling.thread_n": {"origin": "micro_strategy", "resolved": True, "resolved_by": "Tiling.ThreadTile.4x4", "subphase": "Tiling.ThreadTileSelection"},
            },
        }
        history = {
            "completed_subphases": ["Tiling.BlockTileSelection", "Tiling.ThreadTileSelection"],
            "applied_strategy_ids": ["Tiling.BlockTile.64x64x16", "Tiling.ThreadTile.4x4"],
        }
        controller = self.controller(ir, history)
        self.assertNotEqual(controller.current_subphase_id(), "Tiling.MappingDerivation")
        self.assertEqual(controller.current_subphase_id(), "Tiling.WarpTileSelection")

    def test_llm_expected_updates_do_not_override_strategy_id_updates(self):
        merged = merge_expected_ir_updates(
            "Tiling.BlockTile.128x64x16",
            {
                "tiling.enabled": True,
                "tiling.block_m": 128,
                "tiling.block_n": 64,
                "tiling.block_k": 16,
            },
            {
                "tiling.block_m": 128,
                "tiling.block_n": 128,
                "tiling.block_k": 8,
                "tiling": {"block_m": 128, "block_n": 128, "block_k": 8},
            },
        )

        self.assertEqual(merged["tiling.block_m"], 128)
        self.assertEqual(merged["tiling.block_n"], 64)
        self.assertEqual(merged["tiling.block_k"], 16)
        self.assertNotIn("tiling", merged)

    def test_register_accumulator_strategy_enables_register_tile(self):
        ir = {
            "tiling": {
                "thread_m": 2,
                "thread_n": 2,
            },
            "memory": {"use_register_tile": False},
            "field_meta": {
                "tiling.thread_m": {"origin": "micro_strategy", "resolved": True, "subphase": "Tiling.ThreadTileSelection"},
                "tiling.thread_n": {"origin": "micro_strategy", "resolved": True, "subphase": "Tiling.ThreadTileSelection"},
            },
            "strategy": {
                "completed_subphases": [
                    "Layout.SharedMemoryDeclaration",
                    "Layout.SharedMemoryTransform",
                ],
            },
        }
        history = {
            "current_stage": "Layout",
            "completed_subphases": [
                "Layout.SharedMemoryDeclaration",
                "Layout.SharedMemoryTransform",
            ],
            "applied_strategy_ids": [],
            "applied_micro_strategies": [],
        }
        controller = self.controller(ir, history)
        self.assertEqual(controller.current_subphase_id(), "Layout.RegisterAccumulatorLayout")

        next_ir = controller.apply_micro_strategy("Register.AccumulatorLayout.2DArray")
        local_check = next_ir["stage_controller"]["last_local_check"]

        self.assertTrue(next_ir["memory"]["use_register_tile"])
        self.assertEqual(next_ir["register"]["accumulator"]["layout"], "2d_array")
        self.assertTrue(local_check["accepted"])

    def test_warp_iteration_is_derived_after_thread_tile_resolution(self):
        ir = {
            "tiling": {"block_m": 128, "block_n": 128, "block_k": 8},
            "hardware": {"warp_size": 32, "max_threads_per_block": 1024},
            "mapping": {},
        }
        history = {"applied_strategy_ids": [], "completed_subphases": [], "applied_micro_strategies": []}
        ir = self.controller(ir, history).apply_micro_strategy("Tiling.BlockTile.128x128x8")
        history = self.history_from_ir(ir)
        ir = self.controller(ir, history).apply_micro_strategy("Tiling.WarpTile.16x32")
        history = self.history_from_ir(ir)

        self.assertEqual(ir["tiling"]["warp_tile"]["warp_n_iter"], 1)

        ir = self.controller(ir, history).apply_micro_strategy("Tiling.ThreadTile.2x2")
        warp_tile = ir["tiling"]["warp_tile"]

        self.assertEqual(warp_tile["warp_m_iter"], 8)
        self.assertEqual(warp_tile["warp_n_iter"], 16)
        self.assertEqual(
            ir["hardware"]["warp_size"],
            (warp_tile["warp_m_iter"] // ir["tiling"]["thread_m"])
            * (warp_tile["warp_n_iter"] // ir["tiling"]["thread_n"]),
        )
        self.assertEqual(warp_tile["warp_m"] % warp_tile["warp_m_iter"], 0)
        self.assertEqual(warp_tile["warp_n"] % warp_tile["warp_n_iter"], 0)

    def test_derived_warp_iteration_passes_tiling_stage_verification(self):
        ir = {
            "tiling": {"block_m": 128, "block_n": 128, "block_k": 16},
            "hardware": {"warp_size": 32, "max_threads_per_block": 1024},
            "mapping": {},
        }
        history = {"applied_strategy_ids": [], "completed_subphases": [], "applied_micro_strategies": []}
        for strategy_id in (
            "Tiling.BlockTile.128x128x16",
            "Tiling.WarpTile.32x32",
            "Tiling.ThreadTile.4x4",
            "Mapping.WarpThreadTile.WarpLaneFragment2D",
        ):
            ir = self.controller(ir, history).apply_micro_strategy(strategy_id)
            history = self.history_from_ir(ir)

        report = self.controller(ir, history).verify_stage(ir)

        self.assertTrue(report["accepted"], report["results"])

    def test_skipped_subphase_advances_to_next_graph_node(self):
        ir = {
            "tiling": {
                "thread_m": 2,
                "thread_n": 2,
            },
            "memory": {
                "use_shared_memory": True,
                "shared_A": {"shape": ["block_m", "block_k"]},
                "shared_B": {"shape": ["block_k", "block_n"]},
            },
            "strategy": {
                "completed_subphases": ["Layout.SharedMemoryDeclaration"],
            },
        }
        history = {
            "current_stage": "Layout",
            "completed_subphases": ["Layout.SharedMemoryDeclaration"],
            "applied_strategy_ids": ["Layout.SharedMemory.AB.Basic"],
            "applied_micro_strategies": [],
        }
        controller = self.controller(ir, history)
        self.assertEqual(controller.current_subphase_id(), "Layout.SharedMemoryTransform")

        skipped_ir = controller.skip_current_subphase("no selected strategy")
        skipped_history = {
            **history,
            "optional_skipped_subphases": skipped_ir["strategy"]["optional_skipped_subphases"],
        }

        self.assertEqual(
            self.controller(skipped_ir, skipped_history).current_subphase_id(),
            "Layout.RegisterAccumulatorLayout",
        )

    def test_candidate_path_uses_numeric_chain_code_namespace(self):
        path = candidate_path(
            DEFAULT_POSTCHECK_OUTPUT,
            "Layout",
            "Register.AccumulatorLayout.2DArray",
            1,
            "1-1-2",
        )

        self.assertEqual(path.name, "post_check_result.Layout.1_1_2.attempt_1.json")
        self.assertNotIn("Register_AccumulatorLayout", path.name)

    def test_chain_artifact_names_use_numeric_path_code(self):
        strategy_path = [
            "Tiling.BlockTile.32x32x8",
            "Tiling.WarpTile.32x32",
            "Tiling.ThreadTile.2x2",
        ]
        path_code = [1, 1, 2]

        self.assertEqual(chain_key_for_path(strategy_path, path_code), "chain.1-1-2")
        self.assertTrue(str(chain_code_path(strategy_path, path_code)).endswith("chain.1-1-2"))
        self.assertEqual(chain_id_for_state(3, {"path": strategy_path, "path_code": path_code}), "chain_0003.1-1-2")

    def test_single_path_candidate_selection_keeps_first_candidate_only(self):
        candidates = [
            {"strategy_id": "A"},
            {"strategy_id": "B"},
            {"strategy_id": "C"},
        ]

        selected = select_single_path_candidates(candidates)

        self.assertEqual([item["strategy_id"] for item in selected], ["A"])
        self.assertIsNot(selected[0], candidates[0])

    def test_top_k_candidate_selection_keeps_best_three_by_confidence(self):
        candidates = [
            {"strategy_id": "A", "confidence": 0.2},
            {"strategy_id": "B", "confidence": 0.9},
            {"strategy_id": "C", "confidence": 0.5},
            {"strategy_id": "D", "confidence": 0.7},
        ]

        selected = select_top_k_candidates(candidates, 3)

        self.assertEqual([item["strategy_id"] for item in selected], ["B", "D", "C"])
        self.assertIsNot(selected[0], candidates[1])

    def test_layout_top_k_keeps_distinct_implementation_families(self):
        candidates = [
            {"strategy_id": "Layout.SharedMemory.PaddingAB.Plus1", "confidence": 0.95},
            {"strategy_id": "Layout.SharedMemory.PaddingA.Plus1", "confidence": 0.9},
            {"strategy_id": "Layout.SharedMemory.PaddingB.Plus1", "confidence": 0.8},
            {"strategy_id": "Layout.SharedMemory.TransposeB", "confidence": 0.5},
            {"strategy_id": "Layout.SharedMemory.SkewA.Xor", "confidence": 0.4},
        ]
        selected = select_top_k_candidates(candidates, 3, "Layout.SharedMemoryTransform")
        self.assertEqual(
            [item["strategy_id"] for item in selected],
            [
                "Layout.SharedMemory.PaddingAB.Plus1",
                "Layout.SharedMemory.TransposeB",
                "Layout.SharedMemory.SkewA.Xor",
            ],
        )

    def test_missing_layout_families_are_injected_from_filtered_index(self):
        selection = {"candidates": [
            {"strategy_id": "Layout.SharedMemory.PaddingAB.Plus1", "confidence": 0.95},
            {"strategy_id": "Layout.SharedMemory.PaddingA.Plus1", "confidence": 0.8},
        ]}
        available = {"strategies": [
            {"strategy_id": "Layout.SharedMemory.PaddingB.Plus1"},
            {"strategy_id": "Layout.SharedMemory.TransposeB"},
            {"strategy_id": "Layout.SharedMemory.SkewA.Xor"},
        ]}
        result = inject_structurally_diverse_candidates(
            selection, available, "Layout.SharedMemoryTransform")
        ids = [item["strategy_id"] for item in result["candidates"]]
        self.assertIn("Layout.SharedMemory.TransposeB", ids)
        self.assertIn("Layout.SharedMemory.SkewA.Xor", ids)

    def test_frontier_batch_keeps_pruned_paths_for_backtracking(self):
        states = [
            {"state_id": "slow", "last_candidate": {"verified_ir": {"performance": {"gflops": 10.0}}}},
            {"state_id": "fast", "last_candidate": {"verified_ir": {"performance": {"gflops": 30.0}}}},
            {"state_id": "mid", "last_candidate": {"verified_ir": {"performance": {"gflops": 20.0}}}},
        ]

        active, pending = take_frontier_batch(states, 2)

        self.assertEqual([item["state_id"] for item in active], ["fast", "mid"])
        self.assertEqual([item["state_id"] for item in pending], ["slow"])

    def test_frontier_batch_diversifies_across_parent_branches(self):
        states = [
            {"state_id": "1-1", "path_code": [1, 1]},
            {"state_id": "1-2", "path_code": [1, 2]},
            {"state_id": "1-3", "path_code": [1, 3]},
            {"state_id": "2-1", "path_code": [2, 1]},
            {"state_id": "2-2", "path_code": [2, 2]},
            {"state_id": "3-1", "path_code": [3, 1]},
        ]

        active, pending = take_frontier_batch(states, 3)

        self.assertEqual([item["state_id"] for item in active], ["1-1", "2-1", "3-1"])
        self.assertEqual([item["state_id"] for item in pending[:3]], ["1-2", "2-2", "1-3"])

    def test_top_terminal_results_rank_accepted_candidates_by_gflops(self):
        candidates = [
            {
                "accepted": True,
                "strategy_id": "chain_a",
                "candidate_code_dir": "code/a",
                "path": ["A"],
                "path_code": [1],
                "verified_ir": {
                    "performance": {"gflops": 100.0, "latency_ms": 1.0},
                    "verification": {"summary": {"correctness_status": "pass"}},
                },
            },
            {
                "accepted": True,
                "strategy_id": "chain_b",
                "candidate_code_dir": "code/b",
                "path": ["B"],
                "path_code": [2],
                "verified_ir": {
                    "performance": {"gflops": 300.0, "latency_ms": 0.3},
                    "verification": {"summary": {"correctness_status": "pass"}},
                },
            },
            {
                "accepted": True,
                "strategy_id": "chain_c",
                "candidate_code_dir": "code/c",
                "path": ["C"],
                "path_code": [3],
                "verified_ir": {
                    "performance": {"gflops": 200.0, "latency_ms": 0.5},
                    "verification": {"summary": {"correctness_status": "pass"}},
                },
            },
        ]

        top = build_top_terminal_results(candidates, 2)

        self.assertEqual([item["chain_id"] for item in top], ["chain_b", "chain_c"])
        self.assertEqual(top[0]["rank"], 1)
        self.assertEqual(top[0]["code_dir"], "code/b")

    def test_top_terminal_results_exclude_unaccepted_unlock_candidate(self):
        candidates = [
            {
                "accepted": True,
                "strategy_id": "stable",
                "candidate_code_dir": "code/stable",
                "path": ["stable"],
                "path_code": [1],
                "verified_ir": {
                    "performance": {"gflops": 5000.0},
                    "verification": {"summary": {"compile_status": "pass", "correctness_status": "pass"}},
                },
            },
            {
                "accepted": False,
                "strategy_id": "unlock",
                "source_phase": "performance_unlock",
                "candidate_code_dir": "code/unlock",
                "path": ["unlock"],
                "path_code": [9],
                "verified_ir": {
                    "performance": {"gflops": 8800.0},
                    "verification": {
                        "summary": {
                            "compile_status": "pass",
                            "correctness_status": "pass",
                            "code_verification_status": "fail",
                        }
                    },
                },
            },
        ]

        top = build_top_terminal_results(candidates, 2)

        self.assertEqual([item["chain_id"] for item in top], ["stable"])
        self.assertTrue(top[0]["accepted"])

    def test_merge_terminal_verifications_ranks_stable_and_unlock_together(self):
        stable = {
            "verified_candidates": [
                {
                    "accepted": True,
                    "strategy_id": "stable",
                    "candidate_code_dir": "code/stable",
                    "path": ["stable"],
                    "path_code": [1],
                    "verified_ir": {
                        "performance": {"gflops": 5100.0},
                        "verification": {"summary": {"compile_status": "pass", "correctness_status": "pass"}},
                    },
                }
            ],
            "best_candidate": None,
            "summary": {"phase": "stable_baseline", "terminal_chain_count": 2},
        }
        unlock = {
            "verified_candidates": [
                {
                    "accepted": False,
                    "strategy_id": "unlock",
                    "candidate_code_dir": "code/unlock",
                    "path": ["unlock"],
                    "path_code": [9],
                    "verified_ir": {
                        "performance": {"gflops": 8700.0},
                        "verification": {"summary": {"compile_status": "pass", "correctness_status": "pass"}},
                    },
                }
            ],
            "best_candidate": None,
            "summary": {"phase": "performance_unlock", "terminal_chain_count": 3},
        }

        merged = merge_terminal_verifications(stable, unlock, phases=["stable_baseline", "performance_unlock"])

        self.assertEqual(merged["top_terminal_results"][0]["chain_id"], "stable")
        self.assertEqual(merged["top_terminal_results"][0]["source_phase"], "stable_baseline")
        self.assertEqual(merged["best_candidate"]["strategy_id"], "stable")
        self.assertEqual(merged["summary"]["terminal_chain_count"], 5)
        self.assertEqual(merged["summary"]["verified_terminal_chain_count"], 2)

    def test_parse_run_metrics_accepts_skeleton_main_output(self):
        text = """
My gemm Performance= 5748.14 GFlop/s, Time= 0.047 msec, Size= 268435456 Ops,
CuBlas Performance= 448.97 GFlop/s, Time= 0.598 msec, Size= 268435456 Ops,
Result= PASS
ratio= 12.802955
"""

        metrics = parse_run_metrics(text)

        self.assertEqual(metrics["correctness"], "pass")
        self.assertAlmostEqual(metrics["gflops"], 5748.14)
        self.assertAlmostEqual(metrics["latency_ms"], 0.047)
        self.assertAlmostEqual(metrics["cublas_gflops"], 448.97)
        self.assertAlmostEqual(metrics["relative_to_cublas"], 5748.14 / 448.97)

    def test_large_square_problem_size_is_extracted(self):
        facts = extract_problem_description(
            "row-major fp32 NN GEMM. Matrix Size: (M:2048 N:2048 K:2048)."
        )

        self.assertEqual(facts["problem"]["size_class"], "large_square")
        self.assertEqual(facts["problem"]["k_intensity"], "high")
        self.assertEqual(facts["problem"]["alignment_class"], "vector4_friendly")

    def test_generic_warp_lane_fragment_strategy_updates_mapping(self):
        updates = synthesize_ir_updates("Mapping.WarpThreadTile.WarpLaneFragment2D")

        self.assertEqual(updates["mapping.thread_to_output"], "warp_lane_fragment_tile")
        self.assertEqual(updates["mapping.lane_layout"], "warp_lane_fragment_2d")

    def test_lane_layout_strategies_define_warp_output_mapping(self):
        expected_layouts = {
            "Mapping.LaneLayout.1DContiguousM": "lane_1d_contiguous_m",
            "Mapping.LaneLayout.1DContiguousN": "lane_1d_contiguous_n",
            "Mapping.LaneLayout.2D_8x4": "lane_2d_8x4",
        }

        for strategy_id, expected_layout in expected_layouts.items():
            with self.subTest(strategy_id=strategy_id):
                updates = synthesize_ir_updates(strategy_id)
                self.assertTrue(updates["mapping.warp.enabled"])
                self.assertEqual(updates["mapping.warp_to_output"], "2d_warp_tile")
                self.assertEqual(updates["mapping.lane_layout"], expected_layout)

    def test_large_matrix_scheduling_strategy_updates_ir(self):
        updates = synthesize_ir_updates("Memory.L2Reuse.CTASwizzleGroupedN")

        self.assertEqual(updates["memory.l2_reuse_policy"], "CTASwizzleGroupedN")

    def test_stable_profile_filters_high_risk_stages(self):
        stage_order = ["Tiling", "Layout", "MappingReordering", "Vectorization", "Pipeline", "Epilogue", "Compiler"]

        self.assertEqual(
            apply_strategy_profile_to_stage_order(stage_order, "stable_correctness_first"),
            ["Tiling", "Layout", "MappingReordering", "Vectorization", "Epilogue"],
        )
        self.assertEqual(apply_strategy_profile_to_stage_order(stage_order, "all"), stage_order)

    def test_dependency_graph_stage_order_overrides_legacy_ir_order(self):
        ir = {
            "strategy": {
                "profile": "core_construction_coupled",
                "stage_order": ["Tiling", "Layout", "Reordering", "Vectorization", "Pipeline"],
            }
        }
        graph = {
            "stage_order": [
                "Tiling",
                "Layout",
                "MappingReordering",
                "Vectorization",
                "Pipeline",
                "Epilogue",
                "Compiler",
            ]
        }

        self.assertEqual(
            infer_app_stage_order(ir, graph, STRATEGY_INDEX, STRATEGY_LIBRARY),
            ["Tiling", "Layout", "MappingReordering", "Vectorization", "Pipeline", "Epilogue"],
        )

    def test_throughput_profile_unlocks_pipeline_stage(self):
        stage_order = ["Tiling", "Layout", "MappingReordering", "Vectorization", "Pipeline", "Epilogue", "Compiler"]

        self.assertEqual(
            apply_strategy_profile_to_stage_order(stage_order, DEFAULT_UNLOCKED_PERFORMANCE_PROFILE),
            ["Tiling", "Layout", "MappingReordering", "Vectorization", "Pipeline", "Epilogue"],
        )

    def test_throughput_profile_prioritizes_aggressive_candidates(self):
        index = {
            "strategies": [
                {"strategy_id": "Pipeline.NoAsyncCopy.V1"},
                {"strategy_id": "Pipeline.DoubleBuffer.SharedAB"},
                {"strategy_id": "Pipeline.WarpAwareDoubleBuffer.SharedAB"},
            ],
            "filter_context": {"current_subphase": "Pipeline.BufferingSelection"},
        }

        filtered = apply_strategy_profile_to_candidates(index, DEFAULT_UNLOCKED_PERFORMANCE_PROFILE)

        self.assertEqual(filtered["strategies"][0]["strategy_id"], "Pipeline.DoubleBuffer.SharedAB")
        self.assertEqual(filtered["filter_context"]["strategy_profile"], DEFAULT_UNLOCKED_PERFORMANCE_PROFILE)

    def test_stable_profile_blocks_pipeline_prefetch_and_vectorized_candidates(self):
        index = {
            "strategies": [
                {"strategy_id": "Epilogue.StoreC.CoalescedScalar"},
                {"strategy_id": "Safety.BoundaryPolicy.GeneralGuarded"},
                {"strategy_id": "Vectorization.StoreC.SafeScalar"},
                {"strategy_id": "Pipeline.DoubleBuffer.SharedAB"},
                {"strategy_id": "Memory.Prefetch.GlobalToRegisterA"},
                {"strategy_id": "Vectorization.GlobalLoadAB.float4"},
                {"strategy_id": "Reordering.CooperativeVectorLoadAB.float4"},
            ],
            "filter_context": {"rejected": []},
        }

        filtered = apply_strategy_profile_to_candidates(index, "stable_correctness_first")

        self.assertEqual(
            [item["strategy_id"] for item in filtered["strategies"]],
            [
                "Epilogue.StoreC.CoalescedScalar",
                "Safety.BoundaryPolicy.GeneralGuarded",
                "Vectorization.StoreC.SafeScalar",
            ],
        )
        rejected_ids = {item["strategy_id"] for item in filtered["filter_context"]["rejected"]}
        self.assertIn("Pipeline.DoubleBuffer.SharedAB", rejected_ids)
        self.assertIn("Memory.Prefetch.GlobalToRegisterA", rejected_ids)
        self.assertIn("Vectorization.GlobalLoadAB.float4", rejected_ids)
        self.assertIn("Reordering.CooperativeVectorLoadAB.float4", rejected_ids)

    def test_throughput_repair_kernel_contains_double_buffer_and_float4(self):
        ir = {
            "pipeline": {"double_buffering": True},
            "strategy": {"applied_strategy_ids": ["Pipeline.DoubleBuffer.SharedAB"]},
        }
        config = normalize_throughput_launch_config(
            {"BM": 64, "BN": 64, "BK": 16, "WM": 16, "WN": 32, "WMITER": 2, "WNITER": 4, "TM": 4, "TN": 4}
        )

        content = build_throughput_cuda_kernel_cuh(config)

        self.assertTrue(should_emit_throughput_kernel(ir))
        self.assertEqual(config["WMITER"], 16)
        self.assertEqual(config["WNITER"], 32)
        self.assertIn("As[2][BK][BM]", content)
        self.assertIn("Bs[2][BK][BN]", content)
        self.assertIn("FLOAT4", content)
        self.assertIn("comp_flag", content)

    def test_comment_only_mapping_patch_is_rejected(self):
        generated_code = {
            "strategy_id": "Mapping.LaneLayout.2D_8x4",
            "files": [
                {
                    "path": "cuda_kernel.cuh",
                    "content": """
__global__ void gemm() {
    /*
     * INDEX_MAPPING_BEGIN
     * Insert block, warp, and thread mapping here.
     * Mapping.LaneLayout.2D_8x4: comments only.
     * INDEX_MAPPING_END
     */
}
""",
                }
            ],
        }
        patch_ir = {
            "patch_generation": {
                "modified_code_regions": [
                    {"file": "cuda_kernel.cuh", "anchor": "INDEX_MAPPING_BEGIN/END"}
                ]
            }
        }

        result = validate_generated_code_materialization(
            generated_code,
            patch_ir,
            {"strategy_id": "Mapping.LaneLayout.2D_8x4"},
        )

        self.assertEqual(result["status"], "fail")
        self.assertIn("INDEX_MAPPING", result["error_message"])

    def test_empty_kernel_body_gets_minimal_gemm_fallback(self):
        generated_code = {
            "strategy_id": "Tiling.BlockTile.64x64x16",
            "files": [
                {
                    "path": "cuda_kernel.cuh",
                    "content": """
template <int BM, int BN, int BK, int WM, int WN, int WMITER, int WNITER, int TM, int TN>
__global__ void gemm(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    (void)M;
    (void)N;
    (void)K;
    (void)alpha;
    (void)A;
    (void)B;
    (void)beta;
    (void)C;
}
""",
                }
            ],
        }

        ensure_generated_code_has_minimal_kernel(generated_code)
        content = generated_code["files"][0]["content"]

        self.assertIn("scope_acc +=", content)
        self.assertIn("C[OFFSET", content)

    @staticmethod
    def history_from_ir(ir):
        strategy = ir.get("strategy", {})
        return {
            "applied_strategy_ids": [
                item["strategy_id"] for item in strategy.get("applied_micro_strategies", [])
            ],
            "applied_micro_strategies": strategy.get("applied_micro_strategies", []),
            "completed_subphases": strategy.get("completed_subphases", []),
        }


if __name__ == "__main__":
    unittest.main()
