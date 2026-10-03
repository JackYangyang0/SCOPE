from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from SCOPE.app import deterministic_materialization_mode
from SCOPE.generate_ir.ir_checker import check_before_codegen, check_after_codegen, check_code_verification
from SCOPE.generate_ir.stage_controller import find_strategy_in_tree
from SCOPE.llm.patch_generator import validate_patch_scope, extract_anchor_region_snippet
from SCOPE.llm.concrete_code_generator import apply_generated_code_files, replace_cuda_anchor_region
from SCOPE.verification.gemm_semantic_repair import extract_launch_config_from_ir


ROOT = Path(__file__).resolve().parents[1]
SID = "Reordering.LoadCompute.SeparatePhases"


def region(name, body):
    return f"/*\n * {name}_BEGIN\n */\n{body}\n/*\n * {name}_END\n */\n"


class LoadComputeContractTests(unittest.TestCase):
    def setUp(self):
        library = json.loads((ROOT / "data/lib/strategy_library.json").read_text(encoding="utf-8"))
        self.strategy = find_strategy_in_tree(library, SID)
        self.ir = {
            "memory": {"use_shared_memory": True, "use_register_tile": True,
                       "shared_A": {"enabled": True, "shape": [16, 65]},
                       "shared_B": {"enabled": True, "shape": [16, 128]}},
            "tiling": {"block_k": 16}, "mapping": {"threads_per_block": 128},
        }
        self.source = "#pragma once\n" + "".join([
            region("LAUNCH_CONFIG", "static const int TM = 2;"),
            region("SHARED_DECL", "__shared__ float As[16][65];\n__shared__ float Bs[16][128];"),
            region("INDEX_MAPPING", "const int custom_lane = threadIdx.x;"),
            region("REGISTER_DECL", "float acc[8] = {0};"),
            region("MAIN_LOOP", region("COMPUTE_INNER", "acc[0] += As[0][0] * Bs[0][0];")),
            region("STORE", "if (m < M && n < N) C[m*N+n] = acc[0];"),
        ])

    def apply(self, payload):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cuda_kernel.cuh"
            path.write_text(self.source, encoding="utf-8")
            result = apply_generated_code_files(payload, Path(tmp), {}, self.strategy)
            return result, path.read_text(encoding="utf-8")

    def test_predicates_reject_false_flags_and_wrong_loop_structure(self):
        self.assertEqual(deterministic_materialization_mode(self.strategy), "llm_patch")
        self.assertTrue(check_before_codegen(self.ir, self.strategy)["strategy_applicable"])
        bad = copy.deepcopy(self.ir)
        bad["memory"]["use_shared_memory"] = False
        self.assertFalse(check_before_codegen(bad, self.strategy)["strategy_applicable"])
        after = {"schedule": {"loop_structure": "unrelated"},
                 "synchronization": {"sync_policy": {"after_global_to_shared_load": False}}}
        self.assertFalse(check_after_codegen(after, self.strategy, False)["accepted_by_ir_checker"])
        after["schedule"]["loop_structure"] = "separate_load_compute"
        after["synchronization"]["sync_policy"] = {"after_global_to_shared_load": True, "before_shared_buffer_reuse": True}
        self.assertTrue(check_after_codegen(after, self.strategy, False)["accepted_by_ir_checker"])
        self.assertFalse(check_code_verification(after, self.strategy)["accepted_by_code_verifier"])

    def test_protected_ir_and_region_declarations_are_rejected(self):
        patch = {"ir_updates": {"schedule.loop_structure": "separate_load_compute"},
                 "modified_code_regions": [{"file": "cuda_kernel.cuh", "anchor": "MAIN_LOOP_BEGIN/END"}]}
        validate_patch_scope(patch, self.strategy)
        patch["ir_updates"]["tiling.thread_m"] = 4
        with self.assertRaisesRegex(ValueError, "protected IR"):
            validate_patch_scope(patch, self.strategy)
        patch["ir_updates"].pop("tiling.thread_m")
        patch["modified_code_regions"][0]["anchor"] = "SHARED_DECL_BEGIN/END"
        with self.assertRaisesRegex(ValueError, "protected code region"):
            validate_patch_scope(patch, self.strategy)

    def test_local_fma_patch_preserves_prior_strategies(self):
        body = "__syncthreads();\nacc[0] = fmaf(As[0][0], Bs[0][0], acc[0]);\n__syncthreads();"
        result, content = self.apply({"strategy_id": SID, "edits": [
            {"path": "cuda_kernel.cuh", "region": "COMPUTE_INNER", "replacement": body}]})
        self.assertEqual(result["status"], "pass", result)
        self.assertIn("float acc[8]", content)
        self.assertIn("As[16][65]", content)
        self.assertIn("static const int TM = 2;", content)
        self.assertIn("custom_lane", content)
        self.assertIn("if (m < M && n < N)", content)

    def test_full_files_and_disallowed_edits_leave_disk_unchanged(self):
        payloads = [
            {"files": [{"path": "cuda_kernel.cuh", "content": "replacement"}]},
            {"edits": [{"path": "cuda_kernel.cuh", "region": "LAUNCH_CONFIG", "replacement": "static const int TM = 4;"}]},
            {"edits": [{"path": "cuda_kernel.cuh", "region": "MAIN_LOOP", "replacement": "float4 x; acc[0] += 1 * 2;"}]},
            {"edits": [{"path": "cuda_kernel.cuh", "region": "MAIN_LOOP", "replacement": "acc[0] += 1 * 2;"},
                       {"path": "cuda_kernel.cuh", "region": "COMPUTE_INNER", "replacement": "acc[0] += 1 * 2;"}]},
        ]
        for payload in payloads:
            with self.subTest(payload=payload):
                result, content = self.apply(payload)
                self.assertEqual(result["status"], "fail", result)
                self.assertEqual(content, self.source)

    def test_duplicate_load_anchors_are_visible_but_not_silently_patched(self):
        source = region("GLOBAL_TO_SHARED_LOAD", "load_first();") + region("GLOBAL_TO_SHARED_LOAD", "load_next();")
        context = extract_anchor_region_snippet(source, "GLOBAL_TO_SHARED_LOAD")
        self.assertTrue(context["ambiguous"])
        self.assertEqual(len(context["occurrences"]), 2)
        with self.assertRaisesRegex(ValueError, "ambiguous duplicate anchor|Missing or ambiguous anchor"):
            replace_cuda_anchor_region(source, "GLOBAL_TO_SHARED_LOAD", "load_new();")

    def test_nested_iteration_values_are_authoritative(self):
        config = extract_launch_config_from_ir({"tiling": {"thread_m": 2, "thread_n": 2,
            "warp_m_iter": 4, "warp_n_iter": 4, "warp_tile": {"warp_m_iter": 16, "warp_n_iter": 8}}})
        self.assertEqual((config["WMITER"], config["WNITER"], config["TM"], config["TN"]), (16, 8, 2, 2))


if __name__ == "__main__":
    unittest.main()
