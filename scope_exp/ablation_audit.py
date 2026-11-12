from __future__ import annotations

import ast
import copy
import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Any, Sequence

from . import HARNESS_VERSION
from .config import ABLATION_SWITCHES, ablation_block
from .paths import CampaignPaths, repo_root
from .util import ensure_dir, read_json, sha256_value, utc_now, write_json

ABLATION_AUDIT_VERSION = "scope-exp-ablation-audit-v1"

ALIASES = {"minus_structured_state": "minus_state_in_prompt",
           "minus_stage_constraints": "minus_stage_ordering"}

FIXTURE_IR: dict[str, Any] = {
    "optir_name": "optir",
    "optir_version": "1.0",
    "problem": {"M": 512, "N": 512, "K": 512, "batch": 1, "op": "gemm"},
    "target": {"backend": "cuda", "language": "cuda"},
    "hardware": {"warp_size": 32, "shared_memory_bytes": 101376},
    "precision": {"input": "fp32", "accumulate": "fp32", "output": "fp32"},
    "tiling": {"block_m": 64, "block_n": 64, "block_k": 16,
               "warp_tile": {"warp_m": 32, "warp_n": 32}},
    "mapping": {"warps_m": 2, "warps_n": 2},
    "memory": {"layout": "row_major"},
    "performance": {"gflops": 1.0},
    "verification": {"correctness_status": "pass"},
    "resource": {"registers": 64},
    "strategy": {
        "profile": "core_construction_coupled",
        "current_stage": "Tiling",
        "current_subphase": "BlockTileSelection",
        "current_strategy_id": "Tiling.BlockTile.1",
        "current_strategy_category": "Tiling",
        "current_strategy_intent": "increase block tile to 64x64",
        "applied_strategy_ids": ["Tiling.BlockTile.1"],
        "completed_subphases": ["Setup"],
        "observed": {"verified": True, "gflops": 1.0},
    },
    "patch": {"patch_id": "p1", "patch_type": "replace",
              "changed_fields": ["tiling.block_m"], "expected_effect": "more reuse"},
}

FIXTURE_STRATEGY: dict[str, Any] = {
    "strategy_id": "Tiling.BlockTile.2",
    "preconditions": {"predicates": [
        {"field": "tiling.block_m", "op": "<=", "value": 64},
        {"field": "performance.gflops", "op": ">=", "value": 0.5},
        'memory.layout == "row_major"',
        "strategy_intent == increase_tile",
    ]},
}

CONTROL_REQUIREMENTS = (
    "tiling.block_m", "memory.layout", "performance.gflops",
    "strategy_intent", "target.backend",
)

REQUIRED_PROMPT_FIELDS = ("optir_name", "optir_version", "problem", "target",
                          "hardware", "precision")


def extract_resolve_ablation_config(app_path: Path | None = None) -> Any:
    app_path = Path(app_path or (repo_root() / "app.py"))
    source = app_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    target = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "resolve_ablation_config":
            target = node
            break
    if target is None:
        raise RuntimeError("app.py has no resolve_ablation_config; the audit cannot run")
    module = ast.Module(body=[target], type_ignores=[])
    code = compile(ast.fix_missing_locations(module), str(app_path), "exec")
    namespace: dict[str, Any] = {"copy": copy, "argparse": __import__("argparse"),
                                 "Any": Any}
    exec(code, namespace)
    return namespace["resolve_ablation_config"]


def check_mapping(app_path: Path | None = None) -> dict[str, Any]:
    resolver = extract_resolve_ablation_config(app_path)
    problems: list[str] = []
    rows: list[dict[str, Any]] = []
    variants = list(ABLATION_SWITCHES) + list(ALIASES)
    for variant in variants:
        args = types.SimpleNamespace(ablation=variant, ablation_run_id=f"audit-{variant}")
        resolved = resolver({"ablation": {}}, args)
        expected_variant = ALIASES.get(variant, variant)
        expected = ABLATION_SWITCHES[expected_variant]
        observed = {key: bool(resolved.get(key)) for key in expected}
        ok = observed == expected and resolved.get("variant") == expected_variant
        if not ok:
            problems.append(f"{variant}: method {observed} / variant "
                            f"{resolved.get('variant')} != harness {expected}")
        rows.append({"variant": variant, "method_variant": resolved.get("variant"),
                     "method_switches": observed, "harness_switches": expected,
                     "ok": ok,
                     "method_enabled": resolved.get("enabled"),
                     "harness_enabled": expected_variant != "full"})
    return {"ok": not problems, "problems": problems, "variants": rows,
            "source": str(app_path or (repo_root() / "app.py")),
            "checked_method": "app.resolve_ablation_config"}


def load_ablation_module() -> Any:
    path = repo_root() / "utils" / "ablation.py"
    spec = importlib.util.spec_from_file_location("scope_exp_abi_audit_utils_ablation", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _visible_ir(module: Any, variant: str) -> dict[str, Any]:
    module.configure_ablation(ablation_block(variant))
    return module.model_visible_ir(copy.deepcopy(FIXTURE_IR))


def _flatten_keys(value: Any, prefix: str = "") -> list[str]:
    keys: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            keys.append(path)
            keys.extend(_flatten_keys(item, path))
    return keys


def audit_fixture() -> dict[str, Any]:
    module = load_ablation_module()
    full_view = _visible_ir(module, "full")
    full_keys = set(_flatten_keys(full_view))
    rows: list[dict[str, Any]] = []
    problems: list[str] = []

    for variant in ABLATION_SWITCHES:
        view = _visible_ir(module, variant)
        keys = set(_flatten_keys(view))
        added = sorted(keys - full_keys)
        removed = sorted(full_keys - keys)
        module.configure_ablation(ablation_block(variant))
        controlled = module.strategy_for_control(copy.deepcopy(FIXTURE_STRATEGY))
        predicates = (controlled.get("preconditions") or {})
        if isinstance(predicates, dict):
            predicates = predicates.get("predicates") or []
        retained = len(predicates)
        original = len((FIXTURE_STRATEGY.get("preconditions") or {}).get("predicates") or [])
        gate_rows = {requirement: module.control_requirement_enabled(requirement)
                     for requirement in CONTROL_REQUIREMENTS}
        mapping_fields = module.materialize_intent_mapping_fields(
            copy.deepcopy({**FIXTURE_IR, "mapping": {"warps_m": None, "warps_n": None}}))
        snapshot = module.ablation_snapshot()

        record = {
            "variant": variant,
            "visible_digest": sha256_value(view),
            "visible_keys": sorted(keys),
            "added_relative_to_full": added,
            "removed_relative_to_full": removed,
            "mentions_observed_state": any("observed" in key for key in keys),
            "predicates_retained": retained,
            "predicates_original": original,
            "predicates_suppressed": original - retained,
            "control_requirements": gate_rows,
            "mapping_fields_materialized": mapping_fields,
            "shadow_events_recorded": snapshot.get("shadow_event_count", 0),
            "counters": snapshot.get("counters", {}),
        }
        rows.append(record)

        if variant == "full":
            if removed or added:
                problems.append("full is not a fixed point of the audit")
            if not record["mentions_observed_state"]:
                problems.append("the fixture has no observed state for Full to show")
            if retained != original:
                problems.append("Full suppressed a strategy predicate")
        elif variant == "minus_state_in_prompt":
            if not removed:
                problems.append("minus_state_in_prompt hid no prompt field")
            if retained != original:
                problems.append("minus_state_in_prompt changed controller predicates as "
                                "well; that would confound the two factors")
            if any(key in full_keys and "tiling" in key for key in keys):
                problems.append("tiling state is still visible under minus_state_in_prompt")
        elif variant == "intent_only_control":
            if not removed:
                problems.append("intent_only_control hid no prompt field")
            if retained >= original:
                problems.append("intent_only_control did not suppress any controller "
                                "predicate, so the second factor is not active")
        elif variant == "minus_local_checks":
            if removed or added:
                problems.append("minus_local_checks changed the prompt, which it must not")
            if retained != original:
                problems.append("minus_local_checks changed controller predicates, which "
                                "it must not")
        elif variant == "minus_stage_ordering":
            if removed or added:
                problems.append("minus_stage_ordering changed the prompt")
        elif variant == "minus_feedback_optimization":
            if removed or added:
                problems.append("minus_feedback_optimization changed the prompt")


    module.configure_ablation(ablation_block("minus_local_checks"))
    module.record_shadow("unit_audit", True, note="shadow must still record")
    snapshot = module.ablation_snapshot()
    shadow_ok = snapshot.get("shadow_event_count", 0) > 0 and \
        snapshot.get("counters", {}).get("unit_audit.evaluated", 0) == 1
    if not shadow_ok:
        problems.append("shadow logging stopped when local checks were ablated")

    module.configure_ablation(ablation_block("full"))
    return {"ok": not problems, "problems": problems, "variants": rows,
            "shadow_under_local_check_ablation": {
                "recorded": shadow_ok,
                "counters": snapshot.get("counters", {}),
            }}


def activation_summary(fixture: dict[str, Any], *,
                       backend: str = "cuda") -> dict[str, Any]:
    declared = {
        "minus_state_in_prompt": ("prompt fields removed while controller unchanged",
                                  "prompt"),
        "intent_only_control": ("controller predicates suppressed in addition to "
                                "prompt reduction", "prompt+controller"),
        "minus_local_checks": ("controller gate off; prompt and shadow unchanged",
                               "controller"),
        "minus_stage_ordering": ("stage order chosen by the planner instead of the fixed "
                                 "order", "runtime"),
        "minus_feedback_optimization": ("P2 skipped; P3 runs from the P1 checkpoint",
                                        "runtime"),
        "minus_parameter_search": ("P3 skipped; the P2 seed is delivered", "runtime"),
    }
    rows: list[dict[str, Any]] = []
    for record in fixture["variants"]:
        variant = record["variant"]
        description, where = declared.get(variant, ("(reference)", "none"))
        if variant == "full":
            activated = None
            evidence = "reference configuration"
        elif where == "prompt":
            activated = bool(record["removed_relative_to_full"])
            evidence = (f"{len(record['removed_relative_to_full'])} prompt field(s) "
                        f"removed, {record['predicates_suppressed']} predicate(s) "
                        f"suppressed")
        elif where == "prompt+controller":
            activated = bool(record["removed_relative_to_full"]) and \
                record["predicates_suppressed"] > 0
            evidence = (f"{len(record['removed_relative_to_full'])} prompt field(s) "
                        f"removed, {record['predicates_suppressed']} predicate(s) "
                        f"suppressed")
        elif variant == "minus_feedback_optimization" and str(backend) == "cpu":
            activated = None
            evidence = ("not applicable on the CPU backend: the frozen method "
                        "requires target_backend != cpu to run P2 "
                        "(app.py:auto_unlock_performance), so this factor cannot "
                        "be tested on CPU and no without_feedback branch is planned")
        elif where == "controller":
            activated = True
            evidence = ("the local-check gate is a controller condition; the audit "
                        "verifies the flag, the kernel-level gates are unchanged by "
                        "design (compile/final/safety still apply)")
        else:
            activated = None
            evidence = ("decided at run time; a fixture cannot prove it. The runner "
                        "records the actual stage order and the P2/P3 checkpoint "
                        "presence for each task, and the summary reports 'not "
                        "activated' when the order or the checkpoint did not change")
        rows.append({"variant": variant, "declared_factor": description,
                     "where": where, "activated_in_fixture": activated,
                     "evidence": evidence})
    return {"rows": rows,
            "warning": "an ablation whose factor did not activate must be reported as "
                       "not activated, never as a zero effect"}


def run(paths: CampaignPaths, *, backend: str = "cuda") -> dict[str, Any]:
    mapping = check_mapping()
    fixture = audit_fixture()
    activation = activation_summary(fixture, backend=backend)
    record = {
        "ablation_audit_version": ABLATION_AUDIT_VERSION,
        "harness_version": HARNESS_VERSION,
        "created_at": utc_now(),
        "mapping": mapping,
        "fixture": fixture,
        "activation": activation,
        "backend": backend,
        "ok": mapping["ok"] and fixture["ok"],
        "problems": mapping["problems"] + fixture["problems"],
        "coverage": {
            "variants_checked": [row["variant"] for row in fixture["variants"]],
            "historical_aliases": ALIASES,
            "note": "all seven ablation variants are audited, including the two that "
                    "are stage comparisons rather than independent builds",
        },
    }
    ensure_dir(paths.audit)
    write_json(paths.audit / "ablation_diff.json", record)
    return record


def audit_status(paths: CampaignPaths) -> dict[str, Any]:
    path = paths.audit / "ablation_diff.json"
    if not path.exists():
        return {"present": False, "path": str(path)}
    record = read_json(path)
    return {"present": True, "ok": record.get("ok"),
            "problems": record.get("problems"),
            "created_at": record.get("created_at"), "path": str(path)}
