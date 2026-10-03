from __future__ import annotations

import copy
import threading
from collections import Counter
from typing import Any


_DEFAULT_POLICY = {
    "enabled": False,
    "run_id": None,
    "variant": "full",
    "structured_state": True,
    "observed_state_control": True,
    "local_checks": True,
    "stage_constraints": True,
    "fixed_stage_ordering": True,
    "feedback_optimization": True,
    "parameter_search": True,
    "shadow_evaluation": True,
}
_lock = threading.Lock()
_policy: dict[str, Any] = copy.deepcopy(_DEFAULT_POLICY)
_counters: Counter[str] = Counter()
_events: list[dict[str, Any]] = []


def configure_ablation(config: dict[str, Any] | None) -> dict[str, Any]:
    global _policy, _counters, _events
    supplied = config or {}
    with _lock:
        _policy = {**_DEFAULT_POLICY, **supplied}
        _counters = Counter()
        _events = []
        return copy.deepcopy(_policy)


def ablation_policy() -> dict[str, Any]:
    with _lock:
        return copy.deepcopy(_policy)


def gate_enabled(name: str) -> bool:
    with _lock:
        return bool(_policy.get(name, True))


def record_shadow(kind: str, accepted: bool, **context: Any) -> None:
    with _lock:
        _counters[f"{kind}.evaluated"] += 1
        _counters[f"{kind}.accepted" if accepted else f"{kind}.rejected"] += 1
        if len(_events) < 2000:
            _events.append({"kind": kind, "shadow_accepted": bool(accepted), **context})


def record_count(name: str, value: int = 1) -> None:
    with _lock:
        _counters[name] += int(value)


def ablation_snapshot() -> dict[str, Any]:
    with _lock:
        return {
            "policy": copy.deepcopy(_policy),
            "counters": dict(sorted(_counters.items())),
            "shadow_events": copy.deepcopy(_events),
            "shadow_event_count": len(_events),
        }


def model_visible_ir(ir: dict[str, Any]) -> dict[str, Any]:
    """Return the IR visible to the LLM while retaining full IR internally."""
    if gate_enabled("structured_state") and gate_enabled("observed_state_control"):
        return ir
    if gate_enabled("structured_state"):
        record_count("observed_state.prompt_suppressed")
        visible = {
            key: copy.deepcopy(ir[key])
            for key in ("optir_name", "optir_version", "problem", "target", "hardware", "precision")
            if key in ir
        }
        strategy = ir.get("strategy", {}) or {}
        visible["strategy_intent"] = {
            key: copy.deepcopy(strategy[key])
            for key in (
                "profile", "current_stage", "current_subphase", "current_strategy_id",
                "current_strategy_category", "current_strategy_intent", "applied_strategy_ids",
                "completed_subphases",
            )
            if key in strategy
        }
        patch = ir.get("patch", {}) or {}
        visible["patch_intent"] = {
            key: copy.deepcopy(patch[key])
            for key in ("patch_id", "patch_type", "changed_fields", "expected_effect")
            if key in patch
        }
        visible["ablation"] = {
            "observed_generation_state_hidden": True,
            "note": "Only persistent intent is visible; verification, performance, resources and observed effects are hidden.",
        }
        return visible
    record_count("structured_state.prompt_suppressed")
    visible = {
        key: copy.deepcopy(ir[key])
        for key in ("optir_name", "optir_version", "problem", "target", "hardware", "precision")
        if key in ir
    }
    visible["ablation"] = {
        "structured_generation_state_hidden": True,
        "note": "Tiling, mapping, memory, pipeline, resource and strategy history are hidden from the model.",
    }
    return visible


def record_event(kind: str, **context: Any) -> None:
    """Record a non-check controller event without inventing pass/fail counts."""
    with _lock:
        _counters[f"{kind}.count"] += 1
        if len(_events) < 2000:
            _events.append({"kind": kind, **context})


def strategy_for_control(strategy: dict[str, Any]) -> dict[str, Any]:
    """Remove observed-state planning predicates for the intent-only variant."""
    if gate_enabled("observed_state_control"):
        return strategy
    result = copy.deepcopy(strategy)
    preconditions = result.get("preconditions", [])
    predicates = preconditions.get("predicates", []) if isinstance(preconditions, dict) else preconditions
    predicates = predicates or []
    observed_roots = (
        '"tiling.', '"mapping.', '"memory.', '"vectorization.', '"pipeline.',
        '"resource.', '"performance.', '"verification.', '"synchronization.',
        "tiling.", "mapping.", "memory.", "vectorization.", "pipeline.",
        "resource.", "performance.", "verification.", "synchronization.",
    )
    retained = [item for item in predicates if not any(root in repr(item) for root in observed_roots)]
    record_count("observed_state.preconditions_suppressed", len(predicates) - len(retained))
    if isinstance(preconditions, dict):
        result["preconditions"]["predicates"] = retained
    else:
        result["preconditions"] = retained
    return result


def control_requirement_enabled(requirement: Any) -> bool:
    if gate_enabled("observed_state_control"):
        return True
    text = str(requirement)
    return not text.startswith((
        "tiling.", "mapping.", "memory.", "vectorization.", "pipeline.",
        "resource.", "performance.", "verification.", "synchronization.",
    ))


def materialize_intent_mapping_fields(ir: dict[str, Any]) -> list[str]:
    """Type deterministic mapping fields without restoring observed source state.

    Intent-only prompts may legitimately emit symbolic mapping placeholders.  If
    the selected BlockTile/WarpTile and hardware warp size uniquely determine
    the mapping, convert only those fields to their typed representation before
    schema validation.  Missing or non-divisible intent remains unresolved and
    is handled by the normal planner/retry path.
    """
    if gate_enabled("observed_state_control"):
        return []

    tiling = ir.get("tiling", {}) or {}
    warp_tile = tiling.get("warp_tile", {}) or {}
    hardware = ir.get("hardware", {}) or {}

    def positive_int(value: Any) -> int | None:
        if isinstance(value, bool):
            return None
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        return parsed if parsed > 0 else None

    bm = positive_int(tiling.get("block_m"))
    bn = positive_int(tiling.get("block_n"))
    wm = positive_int(warp_tile.get("warp_m"))
    wn = positive_int(warp_tile.get("warp_n"))
    warp_size = positive_int(hardware.get("warp_size")) or 32
    if not all((bm, bn, wm, wn)) or bm % wm != 0 or bn % wn != 0:
        return []

    warps_m = bm // wm
    warps_n = bn // wn
    warps_per_block = warps_m * warps_n
    derived = {
        "warps_m": warps_m,
        "warps_n": warps_n,
        "warps_per_block": warps_per_block,
        "threads_per_block": warps_per_block * warp_size,
    }
    mapping = ir.setdefault("mapping", {})
    changed = []
    previous = {}
    for field, value in derived.items():
        if mapping.get(field) == value:
            continue
        previous[field] = mapping.get(field)
        mapping[field] = value
        changed.append(f"mapping.{field}")

    if changed:
        record_event(
            "intent_mapping_materialized",
            changed_fields=changed,
            previous_values=previous,
            derivation_basis={
                "block_m": bm,
                "block_n": bn,
                "warp_m": wm,
                "warp_n": wn,
                "warp_size": warp_size,
            },
        )
    return changed
