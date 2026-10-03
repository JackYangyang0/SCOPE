"""Keep source and verification together when a repair regresses."""
import copy
import difflib


def verification_progress(ir):
    summary = ir.get("verification", {}).get("summary", {})
    return tuple(summary.get(key) == "pass" for key in (
        "compile_status", "runtime_safety_status", "correctness_status",
        "oracle_acceptance_status"))


def finish_repair(before_ir, after_ir, before_source, after_source, result, accepted):
    feedback = {
        "attempt": result.get("repair_attempt"),
        "method": result.get("method"),
        "verification": copy.deepcopy(after_ir.get("verification", {})),
        "defect_diagnosis": copy.deepcopy(after_ir.get("defect_diagnosis", {})),
        "patch_diff": list(difflib.unified_diff(
            before_source.get("cuda_kernel.cuh", "").splitlines(),
            after_source.get("cuda_kernel.cuh", "").splitlines(), lineterm="")),
    }
    rollback = not accepted and verification_progress(after_ir) < verification_progress(before_ir)
    result["status"] = "repair_verified" if accepted else "repair_validation_failed"
    result["rolled_back"] = rollback
    result["validation_summary"] = copy.deepcopy(after_ir.get("verification", {}).get("summary", {}))
    feedback["rolled_back"] = rollback
    retained = copy.deepcopy(before_ir if rollback else after_ir)
    retained["repair_feedback"] = feedback
    return retained, rollback
