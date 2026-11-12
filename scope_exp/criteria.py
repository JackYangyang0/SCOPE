from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np

ATOL = 1.0e-5
RTOL = 1.0e-3
RELATIVE_L2_TOLERANCE = 1.0e-3
STRICT_ABS_TOLERANCE = 1.0e-6
REFERENCE_FLOOR = 1.0e-12

POLICY_ONLINE = "online_generation_gate"
POLICY_MAIN = "independent_mixed_tolerance_fp64"
POLICY_STRICT = "strict_abs_1e-6_diagnostic"
POLICY_REFERENCE = "reference_agreement_check"

DELIVERY = "delivery"
DIAGNOSTIC = "diagnostic"
REFERENCE_CHECK = "reference_check"


def output_metrics(actual: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    actual = np.asarray(actual, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if actual.shape != reference.shape:
        raise ValueError(f"shape mismatch {actual.shape} vs {reference.shape}")

    finite = np.isfinite(actual)
    finite_actual = np.where(finite, actual, 0.0)
    non_finite = int(np.size(actual) - np.count_nonzero(finite))

    difference = np.where(finite, actual - reference, 0.0)
    absolute = np.abs(difference)
    with np.errstate(invalid="ignore"):
        scaled = absolute / np.maximum(np.abs(reference), REFERENCE_FLOOR)

    max_abs_error = float(np.max(absolute)) if absolute.size else 0.0
    max_abs_index = int(np.argmax(absolute)) if absolute.size else 0

    with np.errstate(invalid="ignore"):
        allowed = ATOL + RTOL * np.abs(reference)
        bad = (~finite) | (np.abs(actual - reference) > allowed)
    bad_count = int(np.count_nonzero(bad))
    bad_positions = np.flatnonzero(bad.reshape(-1))[:16].tolist()

    squared_error_sum = float(np.sum(difference ** 2))
    squared_reference_sum = float(np.sum(reference ** 2))
    relative_l2 = float(np.sqrt(squared_error_sum / max(squared_reference_sum, 1e-30)))

    return {
        "elements": int(np.size(actual)),
        "non_finite_count": non_finite,
        "max_abs_error": max_abs_error,
        "max_abs_index": max_abs_index,
        "max_scaled_error": float(np.max(scaled)) if scaled.size else 0.0,
        "max_rel_error": float(np.max(scaled)) if scaled.size else 0.0,
        "elementwise_bad_count": bad_count,
        "elementwise_bad_positions_flat": bad_positions,
        "squared_error_sum": squared_error_sum,
        "squared_reference_sum": squared_reference_sum,
        "relative_l2_error": relative_l2,
        "reference_norm": float(np.sqrt(squared_reference_sum)),
        "actual_norm": float(np.sqrt(np.sum(finite_actual ** 2))),
    }


@dataclass(frozen=True)
class Policy:
    identifier: str
    role: str
    description: str

    def as_record(self) -> dict[str, Any]:
        return {"policy": self.identifier, "role": self.role,
                "description": self.description}


ONLINE_GENERATION_GATE = Policy(
    identifier=POLICY_ONLINE,
    role=DIAGNOSTIC,
    description="generation-time gate: elementwise atol/rtol vs the build's own "
                "reference, relL2<=1e-3; reported only, not a delivery verdict",
)

MAIN = Policy(
    identifier=POLICY_MAIN,
    role=DELIVERY,
    description="independent full-output revalidation vs FP64 reference: "
                "all finite, |C-R|<=1e-5+1e-3|R| for every element, "
                "||C-R||F/sqrt(max(||R||F^2,1e-30))<=1e-3",
)

STRICT = Policy(
    identifier=POLICY_STRICT,
    role=DIAGNOSTIC,
    description="max|C-R|<1e-6 reported as a separate column; never a pass/fail",
)

REFERENCE_CHECK = Policy(
    identifier=POLICY_REFERENCE,
    role=REFERENCE_CHECK,
    description="compare FP32 and FP64 references; a verdict flip marks the case "
                "reference_invalid instead of choosing the convenient reference",
)

ALL_POLICIES = (MAIN, ONLINE_GENERATION_GATE, STRICT, REFERENCE_CHECK)


def _verdict(policy: Policy, passed: bool | None, reasons: list[str],
             details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "policy": policy.identifier,
        "role": policy.role,
        "applies_to": "actual vs reference",
        "status": "not_run" if passed is None else ("pass" if passed else "fail"),
        "passed": passed,
        "reasons": reasons,
        "details": details or {},
        "thresholds": {
            "atol": ATOL, "rtol": RTOL,
            "relative_l2": RELATIVE_L2_TOLERANCE,
            "strict_abs": STRICT_ABS_TOLERANCE,
        },
    }


def apply_main_policy(metrics: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    if metrics["non_finite_count"]:
        reasons.append(f"{metrics['non_finite_count']} non-finite output element(s)")
    if metrics["elementwise_bad_count"]:
        reasons.append(f"{metrics['elementwise_bad_count']} element(s) outside "
                       f"atol={ATOL:g}+rtol={RTOL:g}|R|")
    if not metrics["relative_l2_error"] <= RELATIVE_L2_TOLERANCE:
        reasons.append(f"relative L2 {metrics['relative_l2_error']:.6e} > "
                       f"{RELATIVE_L2_TOLERANCE:g}")
    passed = not reasons
    return _verdict(MAIN, passed, reasons, {
        "relative_l2_error": metrics["relative_l2_error"],
        "elementwise_bad_count": metrics["elementwise_bad_count"],
        "non_finite_count": metrics["non_finite_count"],
    })


def apply_online_gate(metrics: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    if metrics["non_finite_count"]:
        reasons.append(f"{metrics['non_finite_count']} non-finite output element(s)")
    if metrics["elementwise_bad_count"]:
        reasons.append(f"{metrics['elementwise_bad_count']} element(s) outside "
                       f"atol={ATOL:g}+rtol={RTOL:g}|R|")
    if not metrics["relative_l2_error"] <= RELATIVE_L2_TOLERANCE:
        reasons.append(f"relative L2 {metrics['relative_l2_error']:.6e} > "
                       f"{RELATIVE_L2_TOLERANCE:g}")
    return _verdict(ONLINE_GENERATION_GATE, not reasons, reasons)


def apply_strict_abs(metrics: dict[str, Any]) -> dict[str, Any]:
    passed = metrics["max_abs_error"] < STRICT_ABS_TOLERANCE and not metrics["non_finite_count"]
    reasons = [] if passed else [
        f"max|C-R| {metrics['max_abs_error']:.6e} not below {STRICT_ABS_TOLERANCE:g}"
        " (diagnostic only: FP32 rounding can make this unreachable)"]
    return _verdict(STRICT, passed, reasons,
                    {"max_abs_error": metrics["max_abs_error"]})


def compare_references(fp32_reference: np.ndarray,
                       fp64_reference: np.ndarray) -> dict[str, Any]:
    metrics = output_metrics(np.asarray(fp32_reference, dtype=np.float64),
                             np.asarray(fp64_reference, dtype=np.float64))
    return {
        "max_abs_error": metrics["max_abs_error"],
        "relative_l2_error": metrics["relative_l2_error"],
        "elementwise_bad_count": metrics["elementwise_bad_count"],
        "non_finite_count": metrics["non_finite_count"],
    }


def reference_verdict(agreement: dict[str, Any],
                      fp32_gate: dict[str, Any] | None,
                      main_verdict: dict[str, Any] | None) -> dict[str, Any]:
    reasons: list[str] = []
    status = "agree"
    if agreement["non_finite_count"]:
        status = "reference_invalid"
        reasons.append("FP32 reference contains non-finite values")
    if fp32_gate is not None and main_verdict is not None:
        if fp32_gate["passed"] != main_verdict["passed"]:
            status = "reference_invalid"
            reasons.append("FP32 and FP64 references produce opposite verdicts; "
                           "case needs an independent high-precision recomputation")
    return {
        "policy": POLICY_REFERENCE,
        "role": REFERENCE_CHECK,
        "status": status,
        "passed": status == "agree",
        "reasons": reasons,
        "details": agreement,
    }


def all_policies(metrics: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        POLICY_MAIN: apply_main_policy(metrics),
        POLICY_ONLINE: apply_online_gate(metrics),
        POLICY_STRICT: apply_strict_abs(metrics),
    }

@dataclass
class MismatchDiagnosis:
    primary: str
    evidence: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def as_record(self) -> dict[str, Any]:
        return {"primary": self.primary, "evidence": self.evidence,
                "notes": self.notes}


def diagnose_mismatch(actual: np.ndarray, alternatives: dict[str, np.ndarray],
                      *, max_positional_checks: int = 4096) -> MismatchDiagnosis:
    actual = np.asarray(actual, dtype=np.float64)
    if not np.all(np.isfinite(actual)):
        non_finite = int(np.size(actual) - np.count_nonzero(np.isfinite(actual)))
        return MismatchDiagnosis("non_finite_output",
                                 {"non_finite_count": non_finite})
    if actual.size and not np.any(actual):
        zero_expected_zero = any(np.allclose(candidate, actual, atol=0.0, rtol=0.0)
                                 for candidate in alternatives.values())
        if not zero_expected_zero:
            return MismatchDiagnosis("output_all_zero",
                                     {"elements": int(actual.size)})

    best_name = "unexplained"
    best_score = 0.0
    scores: dict[str, float] = {}
    for name, candidate in alternatives.items():
        candidate = np.asarray(candidate, dtype=np.float64)
        if candidate.shape != actual.shape:
            continue
        with np.errstate(invalid="ignore"):
            allowed = ATOL + RTOL * np.abs(candidate)
            matches = np.abs(actual - candidate) <= allowed
        score = float(np.count_nonzero(matches)) / float(max(actual.size, 1))
        scores[name] = score
        if score > best_score:
            best_name, best_score = name, score

    if best_name == "unexplained":
        return MismatchDiagnosis("unexplained", {"scores": scores})

    if best_name == "reference":
        return MismatchDiagnosis(
            "numerical_difference_no_structural_defect",
            {"scores": scores, "match_fraction": best_score},
            ["output matches the expected product better than any defect "
             "hypothesis; treat as an accumulation/precision difference"])

    evidence: dict[str, Any] = {"scores": scores,
                                "match_fraction": best_score,
                                "checked_positions": min(int(actual.size),
                                                         max_positional_checks)}
    return MismatchDiagnosis(best_name, evidence,
                             ["hypothesis matches more elements than the reference; "
                              "confirm with the raw arrays before publishing"])
