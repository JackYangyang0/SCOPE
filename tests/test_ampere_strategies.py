from __future__ import annotations

import copy
import ctypes
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from SCOPE.app import (build_matrix_profile, resource_profile_reject_reason, find_strategy_object,
                      build_unlock_bundle_strategy, build_phase2_unlock_strategy_index,
                      sanitize_unlock_batch_plan, build_unlock_fallback_batches,
                      candidate_is_accepted)
from SCOPE.generate_ir.gpu_resources import shared_memory_usage, proposed_pipeline_bytes, shared_memory_limit
from SCOPE.generate_ir.ir_checker import check_code_verification, check_shared_memory_limit
from SCOPE.generate_ir.stage_controller import derive_resource_fields
from SCOPE.generate_ir.cuda_device_probe import query_cuda_attributes
from SCOPE.llm.patch_generator import build_patch_ir
from SCOPE.llm.concrete_code_generator import validate_generated_code_materialization
from SCOPE.verification.code_ast_extractor import extract_file_ast
from SCOPE.verification.cuda_launch_config import configure_shared_launch
from SCOPE.verification.gpu_resource_sweep import run_resource_sweep
from SCOPE.verification.build_run_verifier import update_compile_result, run_async_safety_checks
from SCOPE.verification.patch_apply_verify import mark_patch_apply_failed
from SCOPE.utils.upgrade_ampere_strategies import migrate


ROOT = Path(__file__).resolve().parents[1]


def documents():
    return [json.loads((ROOT / path).read_text(encoding="utf-8")) for path in
            ("data/lib/strategy_index.json", "data/lib/strategy_library.json", "data/graph/dependency_graph.json")]


def ir():
    return {"problem": {"M": 512, "N": 512, "K": 1024, "dtype_A": "fp32", "dtype_B": "fp32"},
            "tiling": {"block_m": 128, "block_n": 128, "block_k": 16},
            "pipeline": {"stage_count": 2, "double_buffering": True},
            "memory": {"use_shared_memory": True, "use_register_tile": True,
                       "shared_A": {"enabled": True, "shape": [16, 128]},
                       "shared_B": {"enabled": True, "shape": [16, 128]}},
            "resource": {"shared_memory": {"total_bytes": 32768, "pipeline_multiplier": 2}},
            "hardware": {"sm_count": 108, "supports_cp_async": True,
                         "max_shared_memory_per_block_bytes": 49152,
                         "max_shared_memory_per_block_optin_bytes": 166912,
                         "supports_shared_memory_optin": True}}


class AmpereResourceTests(unittest.TestCase):
    def test_two_to_three_stages_not_six(self):
        state = ir()
        self.assertEqual(proposed_pipeline_bytes(state, 3), 49152)
        self.assertEqual(build_matrix_profile(state)["shared_memory_bytes"], 32768)
        self.assertIsNone(resource_profile_reject_reason("Pipeline.CpAsync.Multistage3.SharedAB", state, build_matrix_profile(state)))

    def test_padding_physical_shape_is_not_counted_twice(self):
        state = ir()
        state["memory"]["shared_A"].update(shape=[16, 129], padding=1)
        self.assertEqual(proposed_pipeline_bytes(state, 3), (16 * 129 + 16 * 128) * 4 * 3)

    def test_symbolic_shape_and_auxiliary_allocation(self):
        state = ir()
        state["memory"]["shared_A"]["shape"] = ["BK", "BM + 1"]
        state["resource"]["shared_memory"]["auxiliary_bytes"] = 128
        self.assertEqual(proposed_pipeline_bytes(state, 3), (16 * 129 + 16 * 128) * 12 + 128)

    def test_dtype_is_per_tensor(self):
        state = ir()
        state["problem"]["dtype_A"] = "fp16"
        self.assertEqual(proposed_pipeline_bytes(state, 2), 16 * 128 * 6 * 2)

    def test_optin_must_be_declared_not_assumed_from_device(self):
        state = ir()
        self.assertEqual(shared_memory_limit(state), 49152)
        state["memory"].update(shared_memory_allocation="dynamic", shared_memory_optin=True)
        self.assertEqual(shared_memory_limit(state), 166912)

    def test_resource_derivation_recomputes_target_stage_count(self):
        state = ir()
        state["pipeline"]["stage_count"] = 3
        derive_resource_fields(state)
        self.assertEqual(state["resource"]["shared_memory"]["total_bytes"], 49152)

    def test_patch_ir_recomputes_even_when_field_meta_exists(self):
        state = ir()
        state["field_meta"] = {"existing": {"origin": "derived"}}
        result = build_patch_ir(state, {"strategy_id": "example"}, {},
                                {"ir_updates": {"pipeline.stage_count": 3}})
        self.assertEqual(result["resource"]["shared_memory"]["total_bytes"], 49152)
        self.assertEqual(state["resource"]["shared_memory"]["total_bytes"], 32768)

    def test_no_shape_total_is_normalized_by_existing_stages(self):
        state = {"resource": {"shared_memory": {"total_bytes": 4000, "pipeline_multiplier": 2}}}
        self.assertEqual(proposed_pipeline_bytes(state, 3), 6000)

    def test_hard_constraint_uses_dynamic_optin_limit(self):
        state = ir()
        state["resource"]["shared_memory"]["total_bytes"] = 65536
        self.assertEqual(check_shared_memory_limit(state).status, "fail")
        state["memory"].update(shared_memory_allocation="dynamic", shared_memory_optin=True)
        self.assertEqual(check_shared_memory_limit(state).status, "pass")

    def test_runtime_query_uses_cuda_logical_device_and_unknown_stays_unknown(self):
        seen = []
        def query(pointer, attribute, device):
            seen.append(device)
            values = {75: 8, 76: 0, 97: 166912}
            if attribute not in values:
                return 1
            ctypes.cast(pointer, ctypes.POINTER(ctypes.c_int))[0] = values[attribute]
            return 0
        runtime = Mock()
        runtime.cudaDeviceGetAttribute = Mock(side_effect=query)
        result = query_cuda_attributes(runtime)
        self.assertEqual(set(seen), {0})
        self.assertEqual(result["compute_capability"], "8.0")
        self.assertNotIn("max_persisting_l2_cache_bytes", result)


class AmpereCatalogTests(unittest.TestCase):
    def test_migration_idempotent_and_ids_aligned(self):
        docs = documents()
        migrate(*docs)
        before = copy.deepcopy(docs)
        migrate(*docs)
        self.assertEqual(docs, before)
        index, library, graph = docs
        entries = [s for stage in library["stages"] for sub in stage["ordered_subphases"] for s in sub["strategies"]]
        self.assertEqual({s["strategy_id"] for s in entries}, {s["strategy_id"] for s in index["strategies"]})
        self.assertEqual(len(entries), len({s["strategy_id"] for s in entries}))
        self.assertIn("Pipeline.CpAsync.Multistage2.SharedAB", graph["fallback"]["Pipeline.CpAsync.Multistage3.SharedAB"][0]["strategies"])

    def test_library_beats_lightweight_index(self):
        index, library, _ = documents()
        sid = "Pipeline.CpAsync.Multistage2.SharedAB"
        found = find_strategy_object(sid, index, library)
        self.assertTrue(found["postconditions"]["code_verification"]["constraints"])
        self.assertIn("code_requirements", found)

    def test_batch_keeps_predicates_and_code_checks(self):
        index, library, _ = documents()
        result = build_unlock_bundle_strategy({"batch_id": "a", "strategy_ids": ["Pipeline.CpAsync.Multistage2.SharedAB"]}, index, library)
        self.assertTrue(result["preconditions"]["predicates"])
        self.assertTrue(result["postconditions"]["patch_ir_verification"]["predicates"])
        self.assertTrue(result["postconditions"]["code_verification"]["constraints"])

    def test_four_stage_is_not_implicitly_converted_to_dynamic_shared_memory(self):
        index, _, graph = documents()
        state = ir()
        history = {"applied_strategy_ids": ["Layout.SharedMemory.AB.Basic", "Reordering.LoadCompute.SeparatePhases"]}
        candidates = build_phase2_unlock_strategy_index(index, state, graph, history, build_matrix_profile(state))
        by_id = {s["strategy_id"]: s for s in candidates["strategies"]}
        sid = "Pipeline.CpAsync.Multistage4.SharedAB"
        self.assertNotIn(sid, by_id)
        self.assertIn("Memory.SharedMemory.DynamicOptIn", by_id)

    def test_unimplemented_executors_stay_excluded(self):
        index, _, _ = documents()
        entries = {s["strategy_id"]: s for s in index["strategies"]}
        for sid in ("Reduction.SplitK.Workspace", "Reduction.StreamK.WorkDecomposition", "Memory.L2Persistence.AccessWindow"):
            self.assertEqual(entries[sid]["phase"], "excluded_default")
            self.assertEqual(entries[sid]["activation_status"], "executor_required")
        self.assertEqual(entries["Compiler.ResourceFeedback.PtxasOccupancySweep"]["phase"], "phase2_unlock_general")

    def test_graph_fallback_includes_smaller_pipeline(self):
        _, _, graph = documents()
        results = build_unlock_fallback_batches({"batch_id": "a", "strategy_ids": ["Pipeline.CpAsync.Multistage3.SharedAB"]}, graph)
        self.assertIn(["Pipeline.CpAsync.Multistage2.SharedAB"], [r["strategy_ids"] for r in results])


class AsyncEvidenceTests(unittest.TestCase):
    def verify(self, source, correctness="pass", safety="pass"):
        index, library, _ = documents()
        strategy = find_strategy_object("Pipeline.CpAsync.Multistage2.SharedAB", index, library)
        state = {"verification": {"correctness": {"status": correctness}, "runtime_safety": {"status": safety}},
                 "code_ast": {"files": {"cuda_kernel.cuh": extract_file_ast("cuda_kernel.cuh", source)}}}
        return check_code_verification(state, strategy)

    def test_all_required_checks_must_pass(self):
        source = 'asm("cp.async.cg.shared.global; cp.async.commit_group; cp.async.wait_group 0;");'
        self.assertTrue(self.verify(source)["accepted_by_code_verifier"])
        self.assertFalse(self.verify(source, safety=None)["accepted_by_code_verifier"])
        self.assertFalse(self.verify(source, correctness="fail")["accepted_by_code_verifier"])

    def test_comments_not_accepted_as_async_copy(self):
        self.assertFalse(self.verify('// cp.async.cg.shared.global; cp.async.commit_group; cp.async.wait_group 0;')["accepted_by_code_verifier"])

    def test_cuda_pipeline_api_not_rejected_by_naming(self):
        self.assertTrue(self.verify('cuda::memcpy_async(dst, src, size, p); p.producer_commit(); p.consumer_wait();')["accepted_by_code_verifier"])

    def test_copy_without_wait_not_accepted(self):
        self.assertFalse(self.verify('asm("cp.async.cg.shared.global;");')["accepted_by_code_verifier"])

    def test_cp_async_rejects_implicit_dynamic_shared_materialization(self):
        source = """
extern __shared__ float storage[];
float *As = storage;
float *Bs = storage + 1024;
float acc = fmaf(As[0], Bs[0], 0.0f);
asm("cp.async.cg.shared.global; cp.async.commit_group; cp.async.wait_group 0;");
"""
        result = validate_generated_code_materialization(
            {"files": [{"path": "cuda_kernel.cuh", "content": source}]},
            {"memory": {"use_shared_memory": True}},
            {"strategy_id": "Pipeline.CpAsync.Multistage2.SharedAB"},
        )
        self.assertEqual(result["status"], "fail")
        self.assertIn("static staged __shared__", result["error_message"])

    def test_cp_async_accepts_static_double_buffer_declarations(self):
        source = """
__shared__ float As[2][16][64];
__shared__ float Bs[2][16][64];
float acc = fmaf(As[0][0][0], Bs[0][0][0], 0.0f);
asm("cp.async.cg.shared.global; cp.async.commit_group; cp.async.wait_group 0;");
"""
        result = validate_generated_code_materialization(
            {"files": [{"path": "cuda_kernel.cuh", "content": source}]},
            {"memory": {"use_shared_memory": True}},
            {"strategy_id": "Pipeline.CpAsync.Multistage2.SharedAB"},
        )
        self.assertEqual(result["status"], "pass")

    def test_cp_async_rejects_static_single_buffer(self):
        source = """
__shared__ float As[16][64];
__shared__ float Bs[16][64];
float acc = fmaf(As[0][0], Bs[0][0], 0.0f);
asm("cp.async.cg.shared.global; cp.async.commit_group; cp.async.wait_group 0;");
"""
        result = validate_generated_code_materialization(
            {"files": [{"path": "cuda_kernel.cuh", "content": source}]},
            {"memory": {"use_shared_memory": True}},
            {"strategy_id": "Pipeline.CpAsync.Multistage2.SharedAB"},
        )
        self.assertEqual(result["status"], "fail")
        self.assertIn("first extent [2]", result["error_message"])

    def test_explicit_dynamic_strategy_accepts_dynamic_shared_slices(self):
        source = """
extern __shared__ float storage[];
float *As = storage;
float *Bs = storage + 1024;
float acc = fmaf(As[0], Bs[0], 0.0f);
asm("cp.async.cg.shared.global; cp.async.commit_group; cp.async.wait_group 0;");
"""
        result = validate_generated_code_materialization(
            {"files": [{"path": "cuda_kernel.cuh", "content": source}]},
            {"memory": {"use_shared_memory": True}},
            {
                "strategy_id": "UnlockBatch.dynamic_cp_async",
                "bundle_strategy_ids": [
                    "Memory.SharedMemory.DynamicOptIn",
                    "Pipeline.CpAsync.Multistage2.SharedAB",
                ],
            },
        )
        self.assertEqual(result["status"], "pass")

    def test_patch_apply_failure_is_retryable_but_never_accepted(self):
        state = {"chain_step": {"status": "generated"}, "verification": {"accepted": None}}
        failed = mark_patch_apply_failed(
            state,
            {"status": "fail", "error_message": "invalid shared declaration", "candidate_code_dir": "candidate"},
        )
        self.assertFalse(candidate_is_accepted(failed))
        self.assertEqual(failed["chain_step"]["status"], "failed")
        self.assertTrue(failed["chain_step"]["retryable"])


class HostLaunchTests(unittest.TestCase):
    source = '''#include <cuda_runtime.h>
__global__ void gemm() { extern __shared__ float storage[]; }
void cuda_gemm(int M, int N, int K) {
  dim3 grid(1), block(32);
  gemm<<<grid, block, 0, stream>>>();
}
'''

    def test_dynamic_configuration_is_idempotent_and_preserves_stream(self):
        state = ir()
        state["memory"].update(shared_memory_allocation="dynamic", shared_memory_optin=True)
        result = configure_shared_launch(self.source, state)
        self.assertEqual(configure_shared_launch(result, state), result)
        self.assertIn("<<<grid, block,32768, stream>>>", result)
        self.assertIn("cudaFuncAttributeMaxDynamicSharedMemorySize", result)

    def test_dynamic_policy_requires_real_dynamic_declaration(self):
        state = ir()
        state["memory"].update(shared_memory_allocation="dynamic", shared_memory_optin=True)
        with self.assertRaises(ValueError):
            configure_shared_launch(self.source.replace("extern __shared__", "__shared__"), state)

    def test_unrelated_kernel_unchanged(self):
        self.assertEqual(configure_shared_launch(self.source, ir()), self.source)

    def test_carveout_zero_is_not_treated_as_missing(self):
        state = ir()
        state["memory"]["shared_memory_carveout_percent"] = 0
        self.assertIn("cudaFuncAttributePreferredSharedMemoryCarveout, 0", configure_shared_launch(self.source, state))


class ResourceSweepTests(unittest.TestCase):
    def test_missing_sanitizer_is_not_reported_as_pass(self):
        with patch("SCOPE.verification.build_run_verifier.shutil.which", return_value=None):
            self.assertEqual(run_async_safety_checks(Path("gemm"), ir(), 60)["status"], "not_run")

    def test_sanitizer_timeout_is_not_a_proven_semantic_defect(self):
        with patch("SCOPE.verification.build_run_verifier.shutil.which", return_value="compute-sanitizer"), patch(
            "SCOPE.verification.build_run_verifier.run_command", return_value={"status": "fail", "timeout": True}
        ):
            self.assertEqual(run_async_safety_checks(Path("gemm"), ir(), 60)["status"], "inconclusive")

    def test_sanitizer_failure_is_reported(self):
        result = {"status": "fail", "returncode": 86, "stderr": "Race reported", "stdout": ""}
        with patch("SCOPE.verification.build_run_verifier.shutil.which", return_value="compute-sanitizer"), patch(
            "SCOPE.verification.build_run_verifier.run_command", return_value=result
        ):
            self.assertEqual(run_async_safety_checks(Path("gemm"), ir(), 60)["status"], "fail")

    def test_all_failed_variants_remove_stale_winner_and_do_not_copy(self):
        def verify(state, *args):
            state["verification"] = {"compile": {"status": "fail"}}
            return state
        with tempfile.TemporaryDirectory() as temp, patch("SCOPE.verification.build_run_verifier.verify_build_and_run", side_effect=verify), patch("SCOPE.verification.gpu_resource_sweep.shutil.copy2") as copy_file:
            executable = Path(temp) / "gemm"
            executable.touch()
            result = run_resource_sweep({}, Path(temp), Path(temp), "gemm", 60, None, "linux", 5, 2)
            self.assertFalse(executable.exists())
            self.assertEqual(result["verification"]["resource_sweep"]["status"], "fail")
            copy_file.assert_not_called()

    def test_sweep_uses_distinct_variants_and_never_selects_incorrect_fastest(self):
        limits = []
        def verify(state, *args):
            limit = state["compiler"]["max_register_count"]
            limits.append(limit)
            self.assertFalse(state["compiler"]["resource_feedback"]["enabled"])
            state["verification"] = {name: {"status": "pass"} for name in ("compile", "correctness", "runtime_safety")}
            state["performance"] = {"latency_ms": {None: 4, 64: 1, 96: 2, 128: 3}[limit], "gflops": 123}
            if limit == 64:
                state["verification"]["correctness"]["status"] = "fail"
            return state
        with tempfile.TemporaryDirectory() as temp, patch("SCOPE.verification.build_run_verifier.verify_build_and_run", side_effect=verify), patch("SCOPE.verification.gpu_resource_sweep.shutil.copy2"):
            result = run_resource_sweep({}, Path(temp), Path(temp), "gemm", 60, None, "linux", 5, 2)
            self.assertEqual(limits, [None, 64, 96, 128])
            self.assertEqual(result["compiler"]["max_register_count"], 96)
            self.assertEqual(len(result["verification"]["resource_sweep"]["variants"]), 4)

    def test_ptxas_evidence_is_parsed_before_log_truncation(self):
        state = {}
        update_compile_result(state, {"status": "pass", "command": [], "returncode": 0, "stdout": b"",
                                     "stderr": "ptxas info : Used 72 registers, 16384 bytes smem\n8 bytes spill stores, 4 bytes spill loads"})
        self.assertEqual(state["resource"]["register"]["actual_per_thread"], 72)
        self.assertEqual(state["verification"]["compile"]["ptxas_resources"]["spill_bytes_sum"], 12)


if __name__ == "__main__":
    unittest.main()
