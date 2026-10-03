import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from SCOPE.verification.repair_transaction import finish_repair
from SCOPE.verification.targeted_cuda_repair import remove_known_redeclarations
from SCOPE.verification.locked_repair import repair_evidence, source_repair_hints


def state(compile_status, runtime="not_run", correctness="not_run"):
    return {"verification": {"summary": {"compile_status": compile_status,
            "runtime_safety_status": runtime, "correctness_status": correctness}}}


class RepairEscalationTests(unittest.TestCase):
    def test_compile_regression_rolls_back_with_failed_patch_feedback(self):
        before = state("pass", "fail")
        original = copy.deepcopy(before)
        result = {"repair_attempt": 2}
        retained, rollback = finish_repair(before, state("fail"),
            {"cuda_kernel.cuh": "old"}, {"cuda_kernel.cuh": "broken"}, result, False)
        self.assertTrue(rollback)
        self.assertEqual(before, original)
        self.assertEqual(retained["verification"], before["verification"])
        self.assertIn("+broken", retained["repair_feedback"]["patch_diff"])
        self.assertEqual(result["status"], "repair_validation_failed")

    def test_success_is_verified_not_merely_generated(self):
        result = {}
        _, rollback = finish_repair(state("fail"), state("pass", "pass", "pass"),
                                   {}, {}, result, True)
        self.assertFalse(rollback)
        self.assertEqual(result["status"], "repair_verified")

    def test_equal_progress_retains_new_candidate_but_not_verified(self):
        result = {}
        _, rollback = finish_repair(state("pass", "fail"), state("pass", "fail"),
                                   {}, {}, result, False)
        self.assertFalse(rollback)
        self.assertEqual(result["status"], "repair_validation_failed")

    def test_cross_anchor_duplicates_same_scope_only(self):
        decl = "const int tid = threadIdx.x;"
        source = "{" + decl + "/* INDEX_MAPPING_BEGIN */" + decl + "{" + decl + "}}"
        repaired, changes = remove_known_redeclarations(source, {})
        self.assertEqual(len(changes), 1)
        self.assertEqual(repaired.count(decl), 2)
        different = "{const int tid=threadIdx.x;const int tid=threadIdx.y;}"
        self.assertEqual(remove_known_redeclarations(different, {}), (different, []))

    def test_evidence_includes_real_verification_and_coupling(self):
        ir = state("fail")
        evidence = repair_evidence(ir, {}, "", "")
        self.assertEqual(evidence["verification"], ir["verification"])
        self.assertTrue(evidence["coupled_repair_obligations"])

    def test_scatter_hints_do_not_depend_on_local_variable_names(self):
        for k, m, value in (("k_idx", "m_idx", "tmp"), ("kk", "rr", "v")):
            source = ("__shared__ float cache[2][BK][BM];"
                      f"float4 {value}=FLOAT4(A[row*K+col]);"
                      f"cache[0][{k}][{m}]={value}.x;cache[0][{k}][{m}+1]={value}.y;")
            self.assertEqual(source_repair_hints(source)[0]["defect_type"],
                             "Vectorization.AScatterAxisSuspect")
            corrected = source.replace(f"[{k}][{m}+1]", f"[{k}+1][{m}]")
            self.assertEqual(source_repair_hints(corrected), [])

    def test_array_swap_hint_and_pointer_swap_distinction(self):
        self.assertEqual(source_repair_hints("__shared__ float As[2][BK][BM];As[0]=As[1];")[0]
                         ["defect_type"], "SharedMemory.ArrayAssignment")
        self.assertEqual(source_repair_hints("float *As[2];As[0]=As[1];"), [])

    def test_third_attempt_requests_full_kernel_and_forbids_other_files(self):
        from SCOPE.app import repair_chain_locally
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = "// unchanged kernel\n"
            (root / "cuda_kernel.cuh").write_text(source)
            (root / "kernel.h").write_text("// fixed interface\n")
            with patch("SCOPE.app.generate_code_files_from_patch_with_llm", return_value={
                "files": [{"path": "kernel.h", "content": "bad"}]
            }) as generate:
                result = repair_chain_locally(root, root, ir={}, client=object(), repair_attempt=3)
            self.assertEqual(result["status"], "repair_failed")
            self.assertIn("only cuda_kernel.cuh", result["error_message"])
            self.assertEqual(generate.call_args.kwargs["prompt_path"].name,
                             "repair_locked_cuda_full_prompt.txt")
            self.assertFalse(generate.call_args.kwargs["strategy"]["patch_contract"]["region_edits_only"])
            self.assertEqual((root / "cuda_kernel.cuh").read_text(), source)


if __name__ == "__main__":
    unittest.main()
