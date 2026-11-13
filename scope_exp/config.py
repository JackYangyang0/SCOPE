
from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml

from . import HARNESS_VERSION
from .errors import ConfigError
from .redact import live_credentials, redact_config
from .util import read_json, sha256_value, utc_now, write_json

CONFIG_VERSION = "scope-exp-config-v1"

ABLATION_SWITCHES = {
    "full": {"structured_state": True, "observed_state_control": True,
             "local_checks": True, "fixed_stage_ordering": True,
             "feedback_optimization": True, "parameter_search": True},
    "minus_state_in_prompt": {"structured_state": False, "observed_state_control": True,
                              "local_checks": True, "fixed_stage_ordering": True,
                              "feedback_optimization": True, "parameter_search": True},
    "intent_only_control": {"structured_state": True, "observed_state_control": False,
                            "local_checks": True, "fixed_stage_ordering": True,
                            "feedback_optimization": True, "parameter_search": True},
    "minus_local_checks": {"structured_state": True, "observed_state_control": True,
                           "local_checks": False, "fixed_stage_ordering": True,
                           "feedback_optimization": True, "parameter_search": True},
    "minus_stage_ordering": {"structured_state": True, "observed_state_control": True,
                             "local_checks": True, "fixed_stage_ordering": False,
                             "feedback_optimization": True, "parameter_search": True},
    "minus_feedback_optimization": {
        "structured_state": True, "observed_state_control": True, "local_checks": True,
        "fixed_stage_ordering": True, "feedback_optimization": False,
        "parameter_search": True},
    "minus_parameter_search": {
        "structured_state": True, "observed_state_control": True, "local_checks": True,
        "fixed_stage_ordering": True, "feedback_optimization": True,
        "parameter_search": False},
}

PLACEHOLDER_MARKERS = ("", "REPLACE_ME", "TODO", "<", "changeme", "your-")

PROVIDER_KEYS = ("qwen", "deepseek", "chatgpt", "cmecloud")
BUDGET_PROFILES = ("default",)


def ablation_block(variant: str, run_id: str | None = None) -> dict[str, Any]:
    if variant not in ABLATION_SWITCHES:
        raise ConfigError(f"unknown ablation variant: {variant}")
    switches = dict(ABLATION_SWITCHES[variant])
    return {
        "enabled": variant != "full",
        "run_id": run_id,
        "variant": variant,
        "stage_constraints": True,
        "shadow_evaluation": True,
        **switches,
    }


def load_yaml(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise ConfigError(
            f"configuration not found: {path}. conf.yaml is not committed because it "
            "holds an API key; copy conf.experiment.yaml to conf.yaml and fill in the "
            "provider fields, or pass --config explicitly.")
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    return config


def resolve_model(config: dict[str, Any], role: str) -> dict[str, Any]:
    llm = config.get("llm") or {}
    provider = llm.get("provider")
    candidates: list[tuple[str, dict[str, Any]]] = []
    for key, value in llm.items():
        if key in ("provider",) or not isinstance(value, dict):
            continue
        candidates.append((key, value))
    matched: tuple[str, dict[str, Any]] | None = None
    for key, value in candidates:
        declared = str(value.get("model") or "")
        if declared == role or key == role:
            matched = (key, value)
            break
    if matched is None:
        raise ConfigError(f"llm block has no entry for model role {role!r}; "
                          f"available: {[key for key, _ in candidates]}")

    key, entry = matched
    if key not in PROVIDER_KEYS:
        raise ConfigError(
            f"the llm block for role {role!r} is keyed {key!r}, which the frozen "
            f"client does not accept as a provider (allowed: {list(PROVIDER_KEYS)}). "
            "Rename the block to a provider key; it is how one task switches model.")
    model_id = str(entry.get("model") or "").strip()
    if any(marker and marker in model_id for marker in PLACEHOLDER_MARKERS):
        raise ConfigError(f"model id for role {role!r} is unresolved: {model_id!r}")
    base_url = str(entry.get("base_url") or "").strip()
    if any(marker and marker in base_url for marker in PLACEHOLDER_MARKERS):
        raise ConfigError(f"base_url for role {role!r} is unresolved: {base_url!r}")
    api_key_env = str(entry.get("api_key_env") or "").strip()
    inline_key = str(entry.get("api_key") or "").strip()
    return {
        "role": role,
        "config_key": key,
        "provider": key,
        "declared_provider": provider,
        "model": model_id,
        "base_url": base_url,
        "api_key_env": api_key_env,
        "api_key_env_present": bool(api_key_env and os.environ.get(api_key_env)),
        "api_key_env_set_but_empty": bool(api_key_env and api_key_env in os.environ
                                          and not os.environ.get(api_key_env)),
        "api_key_present_inline": bool(inline_key),
        "api_key_source": ("environment" if (api_key_env and os.environ.get(api_key_env))
                           else "config_file" if inline_key else "none"),
        "credential_available": bool((api_key_env and os.environ.get(api_key_env))
                                     or inline_key),
        "sampling": {name: entry.get(name) for name in
                     ("temperature", "max_tokens", "selection_max_tokens",
                      "codegen_max_tokens", "timeout_seconds",
                      "selection_timeout_seconds", "codegen_timeout_seconds",
                      "max_retries", "selection_max_retries", "thinking",
                      "reasoning_effort", "use_json_response_format")
                     if name in entry},
        "seed_supported": None,
    }


def load_literal_credential(config: dict[str, Any], role: str) -> str | None:
    llm = config.get("llm") or {}
    for key, value in llm.items():
        if not isinstance(value, dict):
            continue
        if str(value.get("model") or "") == role or key == role:
            inline = str(value.get("api_key") or "").strip()
            if inline:
                return inline
            name = str(value.get("api_key_env") or "").strip()
            if name and os.environ.get(name):
                return os.environ[name]
    return None


def credential_env_for_role(config: dict[str, Any], role: str) -> tuple[str, str] | None:
    value = load_literal_credential(config, role)
    if not value:
        return None
    llm = config.get("llm") or {}
    for key, entry in llm.items():
        if not isinstance(entry, dict):
            continue
        if str(entry.get("model") or "") == role or key == role:
            name = str(entry.get("api_key_env") or "").strip() or "SCOPE_EXP_API_KEY"
            return name, value
    return None


def credential_values_for_scan(config: dict[str, Any],
                               roles: Iterable[str]) -> list[str]:
    values: list[str] = []
    for role in roles:
        value = load_literal_credential(config, role)
        if value and len(value) >= 8:
            values.append(value)
    return values


def default_model_roles(config: dict[str, Any]) -> list[str]:
    llm = config.get("llm") or {}
    provider = llm.get("provider")
    entry = llm.get(provider) if isinstance(llm.get(provider), dict) else None
    if entry and str(entry.get("model") or "").strip():
        return [str(entry["model"]).strip()]
    return []


def resolve_task_config(base: dict[str, Any], task: dict[str, Any], *,
                        model: dict[str, Any] | None,
                        budget_profile: str = "default",
                        backend: str = "cuda",
                        base_config_path: Path | None = None) -> dict[str, Any]:
    if budget_profile not in BUDGET_PROFILES:
        raise ConfigError(f"formal runs require budget_profile='default'; got "
                          f"{budget_profile!r} (compact rewrites search width and "
                          "target shapes and must never reach a formal task)")
    M, N, K = task["shape_M"], task["shape_N"], task["shape_K"]
    if not (M == N == K):
        raise ConfigError(f"task {task['task_id']} is not a cube: {M}x{N}x{K}; the "
                          "core campaign only covers square shapes")
    config = copy.deepcopy(base)
    llm_block = config.get("llm") or {}
    provider = llm_block.get("provider")
    if provider and provider not in llm_block:
        raise ConfigError(
            f"llm.provider={provider!r} has no matching block under llm; "
            f"available: {[key for key, value in llm_block.items() if isinstance(value, dict)]}")

    variant = task.get("variant") or "full"
    run_id = task["task_id"]
    config["ablation"] = ablation_block(variant, run_id)

    shape_specialization = config.setdefault("shape_specialization", {})
    shape_specialization["enabled"] = False
    shape_specialization["shapes"] = [int(M)]

    tuning = config.setdefault("final_top1_tuning", {})
    tuning["enabled"] = bool(tuning.get("enabled", True))
    if backend == "cpu":
        tuning["cpu_shapes"] = [int(M)]
        tuning["gpu_shapes"] = []
    else:
        tuning["gpu_shapes"] = [int(M)]
        tuning["cpu_shapes"] = []

    config.setdefault("build", {})
    config["build"]["platform"] = config["build"].get("platform") or "windows"

    execution = config.setdefault("execution", {})
    execution.setdefault("chain_workers", 3)
    execution.setdefault("llm_max_concurrency", 3)
    execution.setdefault("compile_workers", 2)
    execution["gpu_verification_workers"] = 1

    if model is not None:
        entry = config.setdefault("llm", {}).setdefault(model["config_key"], {})
        entry["model"] = model["model"]
        if model.get("base_url"):
            entry["base_url"] = model["base_url"]
        config["llm"]["provider"] = model.get("provider") or config["llm"].get("provider")

    redacted, redacted_paths = redact_config(config)
    effective = {
        "config_version": CONFIG_VERSION,
        "harness_version": HARNESS_VERSION,
        "task_id": task["task_id"],
        "created_at": utc_now(),
        "base_config_path": str(base_config_path) if base_config_path else None,
        "base_config_sha256": (sha256_value(base) if base else None),
        "redacted_paths": redacted_paths,
        "redacted": redacted,
        "derived": {
            "backend": backend,
            "ablation_variant": variant,
            "ablation_switches": ABLATION_SWITCHES[variant],
            "matrix_size_MNK": [int(M), int(N), int(K)],
            "final_top1_tuning_targets": [int(M)],
            "final_top1_tuning_gpu_shapes": ([int(M)] if backend != "cpu" else []),
            "final_top1_tuning_cpu_shapes": ([int(M)] if backend == "cpu" else []),
            "shape_reuse": "off",
            "budget_profile": budget_profile,
            "model_role": task.get("model_role"),
            "model": model,
        },
    }
    effective["config_hash"] = sha256_value({
        "redacted": redacted,
        "derived": effective["derived"],
        "harness_version": HARNESS_VERSION,
    })
    effective["checks"] = assert_effective_config(effective, task, base)
    effective["_runtime_config"] = config
    return effective


def runtime_config(effective: dict[str, Any]) -> dict[str, Any]:
    return effective.get("_runtime_config") or effective["redacted"]


def assert_effective_config(effective: dict[str, Any], task: dict[str, Any],
                            base: dict[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    redacted = effective["redacted"]
    derived = effective["derived"]
    M = int(task["shape_M"])

    def check(name: str, expected: Any, observed: Any, why: str) -> None:
        ok = observed == expected
        checks.append({"check": name, "expected": expected, "observed": observed,
                       "ok": ok, "why": why})
        if not ok:
            raise ConfigError(f"effective configuration check failed: {name} "
                              f"(expected {expected!r}, observed {observed!r}); {why}")

    check("matrix_size", [M, M, M], derived["matrix_size_MNK"],
          "the task shape must be the shape the generator works on (§2)")
    backend = derived.get("backend", "cuda")
    target_key = "cpu_shapes" if backend == "cpu" else "gpu_shapes"
    check(f"final_top1_tuning.{target_key}", [M],
          redacted.get("final_top1_tuning", {}).get(target_key),
          "P3 must target exactly the task shape, otherwise one task produces "
          "four tuning outputs (§7.1)")
    other_key = "gpu_shapes" if backend == "cpu" else "cpu_shapes"
    check(f"final_top1_tuning.{other_key}", [],
          redacted.get("final_top1_tuning", {}).get(other_key),
          "the other platform's tuning list must be empty, or the task would "
          "produce outputs for a platform it is not running on")
    check("shape_specialization.enabled", False,
          redacted.get("shape_specialization", {}).get("enabled"),
          "shape reuse would make several tasks share one generation")
    check("ablation.variant", task.get("variant") or "full",
          redacted.get("ablation", {}).get("variant"),
          "the recorded variant must be the executed variant")
    check("ablation.enabled", (task.get("variant") or "full") != "full",
          redacted.get("ablation", {}).get("enabled"),
          "the Full configuration is the only one with ablation disabled")
    for switch, value in ABLATION_SWITCHES[task.get("variant") or "full"].items():
        check(f"ablation.{switch}", value,
              redacted.get("ablation", {}).get(switch),
              "only the declared factor may differ between configurations")
    check("ablation.stage_constraints", True,
          redacted.get("ablation", {}).get("stage_constraints"),
          "every ablation keeps the stage constraints (§6.1)")
    check("ablation.shadow_evaluation", True,
          redacted.get("ablation", {}).get("shadow_evaluation"),
          "shadow logging stays on so the local-check ablation can be audited")
    check("execution.gpu_verification_workers", 1,
          redacted.get("execution", {}).get("gpu_verification_workers"),
          "the GPU is never shared by two verification jobs (§2)")

    model = derived.get("model")
    if model is not None:
        check("model.resolved", True, bool(model.get("model")),
              "an unresolved model id must stop the run (§13 REQ-01)")
        check("model.config_key_present",
              bool(redacted.get("llm", {}).get(model["config_key"])), True,
              "the redacted config must still carry the model entry")
    return checks


def assert_no_compact_leak(effective: dict[str, Any], base: dict[str, Any]) -> None:
    compact_markers = {
        "final_top1_tuning": {"max_candidates_per_shape": 6, "rounds": 2},
        "feedback_search": {"max_rounds": 1, "max_total_compilations": 3},
        "execution": {"chain_workers": 1, "llm_max_concurrency": 1},
    }
    redacted = effective["redacted"]
    hits: list[str] = []
    for section, markers in compact_markers.items():
        values = redacted.get(section) or {}
        for key, compact_value in markers.items():
            if values.get(key) == compact_value:
                hits.append(f"{section}.{key}={compact_value}")
    if hits and effective["derived"].get("budget_profile") != "compact":
        raise ConfigError(
            "configuration looks like a compact/debug budget but the task claims a "
            f"formal profile: {hits}. Set the formal search limits explicitly.")


def cli_arguments(task: dict[str, Any], *, config_path: Path, build_platform: str,
                  vcvars64: str | None = None, backend: str = "cuda") -> list[str]:
    M, N, K = task["shape_M"], task["shape_N"], task["shape_K"]
    argv = [
        "-m", "scope_exp.entry",
        "--config", str(config_path),
        "--matrix-size", str(M), str(N), str(K),
        "--ablation", task.get("variant") or "full",
        "--ablation-run-id", task["task_id"],
        "--experiment-budget", "default",
        "--shape-reuse", "off",
        "--build-platform", build_platform,
    ]
    if vcvars64:
        argv += ["--vcvars64", str(vcvars64)]
    if task.get("keep_cache", False):
        argv.append("--keep-generated-cache")
    if task.get("skip_skeleton_check"):
        argv.append("--skip-initial-skeleton-check")
    argv = [part for part in argv if part is not None]
    return argv


def write_effective_config(path: Path, effective: dict[str, Any]) -> dict[str, Any]:
    payload = {k: v for k, v in effective.items()
               if k not in ("redacted", "_runtime_config")}
    payload["redacted_config"] = effective["redacted"]
    write_json(Path(path), payload)
    return payload


def environment_credential_status(roles: Iterable[dict[str, Any]]) -> dict[str, Any]:
    status: dict[str, Any] = {}
    live = live_credentials()
    for role in roles:
        name = role.get("api_key_env") or ""
        status[role.get("role") or role.get("model")] = {
            "api_key_env": name,
            "present": bool(name and os.environ.get(name)),
            "value_recorded": False,
            "live_credential_values_in_environment": len(live),
        }
    return status
