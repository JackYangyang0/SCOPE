
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from . import HARNESS_VERSION
from . import build as build_module
from . import controls as controls_module
from . import (criteria, data as data_module, datasets, tasks as tasks_module,
               timing as timing_module)
from . import config as config_module
from .config import (ABLATION_SWITCHES, CONFIG_VERSION, load_yaml,
                     resolve_model, environment_credential_status)
from .errors import ConfigError, GateError, ProtocolError, ScopeExpError
from .paths import LAYOUT_VERSION, CampaignPaths, repo_root
from .protocol import (PROTOCOL_VERSION, blank_protocol, validate, write_protocol)
from .util import (environment_record, ensure_dir, read_json, run_command,
                   sha256_file, sha256_value, tree_hash, utc_now, which, write_csv,
                   write_json)

METHOD_FILE_PATTERNS = (
    "*.py",
    "llm/**/*.py",
    "llm/prompts/*",
    "utils/**/*.py",
    "verification/**/*.py",
    "specialization/**/*.py",
    "generate_ir/**/*.py",
    "generate_ir/**/*.json",
    "data/**/*.json",
    "gemm_code/skeleton/*",
    "gemm_code/cpu_skeleton/*.c",
    "gemm_code/cpu_skeleton/*.h",
    "gemm_code/code/**/*",
    "scope_exp/**/*.py",
    "scope_exp/drivers/*",
    "scope_exp/drivers/controls/*",
    "requirements.txt",
    "conf.example.yaml",
    "实验设计.md",
    "README.md",
)

TASK_MANIFEST_COLUMNS = (
    "order_index", "task_id", "block", "kind", "shape_M", "shape_N", "shape_K",
    "repeat", "variant", "model_role", "arm", "parent_task_id", "reuse_of",
    "coefficients_only", "planned_status", "notes",
)


def source_snapshot() -> dict[str, Any]:
    root = repo_root()
    git_head = None
    git_dirty = None
    if (root / ".git").exists() and which("git"):
        head = run_command(["git", "-C", str(root), "rev-parse", "HEAD"], timeout=60)
        if head["returncode"] == 0:
            git_head = head["stdout"].strip()
        status = run_command(["git", "-C", str(root), "status", "--porcelain"], timeout=120)
        if status["returncode"] == 0:
            lines = [line for line in status["stdout"].splitlines()
                     if line.strip() and not line.strip().startswith("?? results/")]
            git_dirty = bool(lines)
    files: dict[str, str] = {}
    for pattern in METHOD_FILE_PATTERNS:
        for path in sorted(root.glob(pattern)):
            if not path.is_file():
                continue
            try:
                relative = path.relative_to(root).as_posix()
            except ValueError:
                continue
            if relative.startswith("results/"):
                continue
            files[relative] = sha256_file(path)
    return {
        "repo_root": str(root),
        "git_head": git_head,
        "git_dirty": git_dirty,
        "file_count": len(files),
        "method_files": files,
        "source_digest": sha256_value(files),
        "patterns": list(METHOD_FILE_PATTERNS),
        "captured_at": utc_now(),
        "note": "results/ is excluded: it holds evidence, not method source",
    }


def _search_limits(base: dict[str, Any]) -> dict[str, Any]:
    search = base.get("search") or {}
    feedback = base.get("feedback_search") or {}
    tuning = base.get("final_top1_tuning") or {}
    return {
        "max_candidates_per_shape": int(tuning.get("max_candidates_per_shape", 1710)),
        "rounds": int(tuning.get("rounds", 3)),
        "screening_benchmark_runs": int(tuning.get("screening_benchmark_runs", 1)),
        "final_remeasure_top_k": int(tuning.get("final_remeasure_top_k", 3)),
        "max_frontier_states_per_stage": int(search.get("max_frontier_states_per_stage", 3)),
        "tiling_resource_pool_size": int(search.get("tiling_resource_pool_size", 32)),
        "tiling_llm_top_n": int(search.get("tiling_llm_top_n", 3)),
        "compile_shortlist_size": int(search.get("compile_shortlist_size", 3)),
        "compile_fallback_limit": int(search.get("compile_fallback_limit", 6)),
        "core_construction_frontier_states": int(
            search.get("core_construction_frontier_states", 3)),
        "max_feedback_rounds": int(feedback.get("max_rounds", 3)),
        "experiments_per_round": int(feedback.get("experiments_per_round", 3)),
        "max_total_compilations": int(feedback.get("max_total_compilations", 12)),
        "candidate_prefix_truncation": True,
        "prefix_truncation_note": (
            "the current code truncates a candidate list prefix; the campaign "
            "records space_total/proposed/attempted/compiled/... so a prefix run "
            "is never reported as exhaustive (实验设计.md §6.3)"),
    }


def default_stop_rules(per_task_wall_seconds: float) -> list[str]:
    return [
        "stop a task at its pre-registered wall-clock limit; the task counts as "
        "budget_exhausted and stays in the denominator",
        "stop a task when the search finds no terminal candidate "
        "(science failure: no_candidate)",
        "stop retrying an API call after max_retries; unknown responses are "
        "recorded as unknown, never re-requested silently",
        "never extend a limit for one group after the formal queue starts",
        f"per-task wall limit: {per_task_wall_seconds:.0f}s (including API waits)",
    ]


def build_protocol(paths: CampaignPaths, *, base_config_path: Path,
                   blocks: Sequence[str], backend: str, campaign_id: str,
                   budget: dict[str, Any], statistics: dict[str, Any] | None = None,
                   include_branches: bool = True,
                   allow_missing_credentials: bool = False,
                   rehearsal: bool = False,
                   shapes: Sequence[int] | None = None,
                   repeats: int | None = None,
                   model_roles: Sequence[str] | None = None,
                   dataset_jobs: int = 1) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    base = load_yaml(base_config_path)
    requested_shapes = list(shapes) if shapes else None
    if requested_shapes:
        shape_list = sorted({int(shape) for shape in requested_shapes})
    else:
        shape_set: set[int] = set()
        for block in blocks:
            name = block.upper()
            if name in ("E1", "E2", "D1", "PILOT"):
                shape_set.update(tasks_module.SHAPES)
            elif name == "E3":
                shape_set.update(tasks_module.E3_SHAPES)
            elif name == "REHEARSAL":
                shape_set.add(tasks_module.REHEARSAL_SHAPE)
        shape_list = sorted(shape_set)
    shapes = shape_list
    if repeats:
        repeats = int(repeats)
    elif {block.upper() for block in blocks} == {"REHEARSAL"}:
        repeats = tasks_module.REHEARSAL_REPEATS
    else:
        repeats = tasks_module.REPEATS
    if model_roles is None:
        has_e2 = "E2" in {str(block).upper() for block in blocks}
        model_roles = (list(tasks_module.MODEL_ROLES) if has_e2
                       else config_module.default_model_roles(base) or
                       list(tasks_module.MODEL_ROLES))
    model_roles = [str(role) for role in model_roles]

    protocol = blank_protocol(campaign_id, backend=backend, shapes=shapes,
                              repeats=repeats, blocks=[block.upper() for block in blocks])

    snapshot = source_snapshot()
    protocol["source"] = snapshot
    protocol["source"]["base_config_sha256"] = sha256_value(base)
    protocol["source"]["base_config_path"] = str(base_config_path)

    environment = environment_record()
    environment["build"] = build_module.build_environment_record()
    environment["toolchain_driver"] = build_module.toolchain_summary(backend)
    protocol["environment"] = environment

    task_list = tasks_module.build_task_matrix(
        blocks=blocks, shapes=requested_shapes, repeats=repeats,
        model_roles=model_roles, backend=backend,
        include_branches=include_branches)
    matrix_check = tasks_module.validate_matrix(
        task_list, blocks=blocks, shapes=requested_shapes, repeats=repeats,
        model_roles=model_roles, backend=backend)
    protocol["tasks"] = {
        "counts": matrix_check["observed"],
        "expected": matrix_check["expected"],
        "task_ids": [task.task_id for task in task_list],
        "order": [task.task_id for task in sorted(task_list,
                                                  key=lambda t: t.order_index)],
        "manifest_sha256": sha256_value([task.task_id for task in task_list]),
        "blocks": [block.upper() for block in blocks],
        "task_records": [task.as_record() for task in task_list],
        "validation": matrix_check,
        "denominators": {
            "independent": sum(
                matrix_check["observed"].get(key, 0) for key in
                ("E1_independent", "E2_independent_new", "E3_independent",
                 "D1_independent")),
            "pilot": matrix_check["observed"].get("PILOT_independent", 0),
            "branches": matrix_check["observed"].get("E1_branches_no_feedback", 0),
            "aliases": matrix_check["observed"].get("E1_aliases_no_parameter_search", 0),
        },
        "profile": {
            "shapes": shapes,
            "repeats": repeats,
            "model_roles": model_roles,
            "backend": backend,
            "note": "the shape list and repeat count are frozen here; a CPU campaign "
                    "is the cross-platform extension, not the core GPU matrix",
        },
    }

    roles = sorted({task.model_role for task in task_list if task.model_role})
    models: dict[str, Any] = {}
    if rehearsal:
        models["rehearsal"] = {
            "role": "rehearsal", "config_key": "rehearsal", "provider": "none",
            "model": "rehearsal-no-llm", "base_url": "",
            "api_key_env": None, "api_key_env_present": True,
            "api_key_env_set_but_empty": False, "sampling": {},
            "seed_supported": None,
            "note": "rehearsal protocol: no model is called",
        }
    else:
        for role in (roles or model_roles or list(tasks_module.MODEL_ROLES)):
            models[role] = resolve_model(base, role)
    protocol["models"] = models
    protocol["credential_status"] = environment_credential_status(models.values())

    dataset_summary = datasets.dataset_summary(paths)
    if not dataset_summary.get("present"):
        raise ProtocolError("datasets are not prepared; run `scope-exp prepare` first")
    if dataset_summary.get("manifest_hash") != dataset_summary.get("manifest_hash"):
        raise ProtocolError("dataset summary is inconsistent")
    splits = {split: record["manifest_hash"]
              for split, record in dataset_summary["splits"].items()}
    protocol["data"] = {
        "generator_version": data_module.GENERATOR_VERSION,
        "splits": sorted(splits),
        "manifest_hashes": splits,
        "manifest_files": {split: record["manifest"]
                           for split, record in dataset_summary["splits"].items()},
        "ledger_hashes": {split: record["ledger_hash"]
                          for split, record in dataset_summary["splits"].items()},
        "checks_per_kernel_per_shape": {"main": 64, "coefficients": 128, "total": 192},
        "coefficient_sets": list(data_coefficient_sets()),
        "c0_variants": ["pattern", "scalar"],
        "generation_split": "generation",
        "evaluation_split": "postselection_test",
        "pilot_split": "development",
        "selection_rule": "the generation split is the only input the frozen "
                          "method uses; the postselection split is fixed before "
                          "formal runs and never shown to it",
        "datasets_summary_sha256": dataset_summary.get("manifest_hash"),
    }

    protocol["validation"] = {
        "policies": [policy.as_record() for policy in criteria.ALL_POLICIES],
        "atol": criteria.ATOL,
        "rtol": criteria.RTOL,
        "relative_l2": criteria.RELATIVE_L2_TOLERANCE,
        "strict_abs": criteria.STRICT_ABS_TOLERANCE,
        "reference": {
            "kind": "cpu_blas_fp64",
            "arithmetic": "float64",
            "cross_check": "compensated_fp64_point_check + analytic families",
            "optional_cublas": "drivers/ref_dgemm.cu when a GPU is present",
            "rule": "a reference disagreement marks the case reference_invalid; "
                    "the harness never keeps the convenient reference",
        },
        "timeout_seconds": float(budget.get("validation_timeout_seconds", 1800.0)),
        "repeat_checks": int(budget.get("repeat_checks", 8)),
        "hypotheses_max_shape": int(budget.get("hypotheses_max_shape", 512)),
        "summary_checks": sorted({512, 1024, 2048, 4096}),
    }

    protocol["timing"] = timing_module.timing_protocol_record()
    protocol["timing"]["timing_case"] = {
        "case_id": timing_module.TIMING_CASE["case_id"],
        "c0_variant": timing_module.TIMING_CASE["c0_variant"],
        "split": timing_module.TIMING_CASE["split"],
        "rule": "fixed before measurement, identical for every method",
    }

    protocol["budget"] = {
        "per_task_wall_seconds": float(budget["per_task_wall_seconds"]),
        "per_phase_wall_seconds": dict(budget.get("per_phase_wall_seconds") or {
            "p1": float(budget["per_task_wall_seconds"]) * 0.5,
            "p2": float(budget["per_task_wall_seconds"]) * 0.3,
            "p3": float(budget["per_task_wall_seconds"]) * 0.2,
        }),
        "pilot_wall_seconds": float(budget.get("pilot_wall_seconds",
                                               budget["per_task_wall_seconds"])),
        "api_timeout_seconds": float(budget.get("api_timeout_seconds", 300.0)),
        "max_retries": int(budget.get("max_retries", 5)),
        "max_requests": int(budget.get("max_requests", 0)) or None,
        "search": _search_limits(base),
        "stop_rules": default_stop_rules(float(budget["per_task_wall_seconds"])),
        "budget_profile": "default",
        "budget_exhaustion_policy": "counts as a science failure inside the frozen "
                                    "limit; the incumbent may be used only if the "
                                    "pre-registered rule below says so",
        "incumbent_fallback": {
            "allowed": False,
            "note": "set explicitly at freeze time if a fallback to an accepted "
                    "incumbent is part of the method; leaving it false means a "
                    "budget-exhausted task delivers nothing",
        },
        "cost_model": {
            "formula": "total wall = sum(task) + sum(branch P3) + evaluation/audit",
            "reported_separately": ["gpu_active_seconds", "api_wait_seconds",
                                    "cpu_compile_seconds", "wall_seconds"],
            "scenario_only": {"280_tasks_at_1h": 280, "at_3h": 840, "at_6h": 1680,
                              "note": "scenario arithmetic, not a forecast"},
        },
    }

    gate = controls_module.gate_status(paths)
    protocol["controls_gate"] = {
        "path": gate.get("path"),
        "sha256": (sha256_file(Path(gate["path"])) if gate.get("present") else None),
        "backend": gate.get("backend"),
        "ok": bool(gate.get("ok")),
        "failed_checks": gate.get("failed_checks"),
        "created_at": gate.get("created_at"),
    }

    protocol["sanitizer"] = {
        "timeout_seconds": float(budget.get("sanitizer_timeout_seconds", 900.0)),
        "tools": ["memcheck", "racecheck", "synccheck"],
        "applicability_rule": "memcheck always; racecheck when the kernel uses "
                              "shared memory; synccheck when it uses shared memory "
                              "or explicit synchronisation",
        "states": ["pass", "fail", "not_run", "inconclusive", "not_applicable",
                   "unsupported"],
        "not_run_is_not_safe": True,
    }

    protocol["selection_rules"] = {
        "top1": "the frozen method's own final selection; the evaluator never "
                "re-ranks or picks a runner-up",
        "no_reselection_after_validation": True,
        "without_feedback_start": "derive from the parent's P1 selected checkpoint "
                                  "and apply the same P3 protocol",
        "alias_without_parameter_search": "reuse the parent's P2 checkpoint as the "
                                          "seed and time it independently",
        "budget_exhaustion": "the task ends without a delivered kernel; the "
                             "incumbent is used only if budget.incumbent_fallback "
                             "was registered true at freeze time",
    }

    protocol["statistics"] = {
        "alpha": float((statistics or {}).get("alpha", 0.05)),
        "bootstrap_resamples": int((statistics or {}).get("bootstrap_resamples", 10000)),
        "bootstrap_seed": int((statistics or {}).get("bootstrap_seed", 20261008)),
        "wilson_intervals": True,
        "paired_by_parent": True,
        "unit_of_analysis": "one independent generation task",
        "input_level_counts_are_not_independent": True,
        "undelivered_throughput_metric": "Y_i = delivered_i * throughput_i, 0 when "
                                         "undelivered; a rate per generation attempt, "
                                         "not a kernel speed",
        "complete_queue_range": "[s/N, (s+u)/N] reported whenever u > 0",
        "notes": "10/10 success has a Wilson 95% lower bound of about 0.722; the "
                 "campaign never claims 99% reliability from 10 samples",
    }

    protocol["extensions"] = {
        "dependency_revision": {
            "planned": "D1" in {block.upper() for block in blocks},
            "block": "D1",
            "required_version": "scope-exp-dependency-v2",
            "note": "the current soft ordering overlay is not a build-time edge "
                    "revision and cannot test that claim (实验设计.md §11.1)",
        },
        "budget_efficiency": {
            "planned": "E3" in {block.upper() for block in blocks},
            "block": "E3",
            "note": "strict equal wall-clock budget from the first request",
        },
        "cross_platform": {
            "planned": backend == "cpu",
            "note": ("this campaign runs the CPU extension: same structure, CPU "
                     "shapes, no GPU; it is not the core 280 and must not be "
                     "merged with GPU numbers" if backend == "cpu" else
                     "H800/14700KF published kernels can be re-validated as fixed "
                     "artifacts; that is not a generation result"),
        },
        "feedback_stage": {
            "applicable": backend != "cpu",
            "evidence": "app.py auto_unlock_performance requires target_backend != cpu",
            "note": (tasks_module.FEEDBACK_STAGE_CPU_NOTE
                     if backend == "cpu" else
                     "P2 runs; without_feedback branches are planned"),
        },
    }

    rows = [task.as_record() for task in sorted(task_list,
                                                key=lambda t: t.order_index)]
    write_csv(paths.task_manifest_csv, rows, TASK_MANIFEST_COLUMNS)
    write_json(paths.task_manifest_json, {
        "campaign_id": campaign_id,
        "counts": matrix_check["observed"],
        "expected": matrix_check["expected"],
        "tasks": rows,
        "manifest_sha256": sha256_value([task.task_id for task in task_list]),
    })
    protocol["tasks"]["manifest_file"] = str(paths.task_manifest_csv)
    protocol["tasks"]["manifest_file_sha256"] = sha256_file(paths.task_manifest_csv)

    checks = validate(protocol, require_frozen=False, paths=paths,
                      allow_missing_credentials=allow_missing_credentials)
    return protocol, checks


def data_coefficient_sets() -> Sequence[dict[str, Any]]:
    return data_module.COEFFICIENT_SETS


def freeze_campaign(paths: CampaignPaths, *, base_config_path: Path,
                    blocks: Sequence[str], backend: str, campaign_id: str,
                    budget: dict[str, Any], statistics: dict[str, Any] | None = None,
                    include_branches: bool = True,
                    allow_missing_credentials: bool = False,
                    rehearsal: bool = False,
                    shapes: Sequence[int] | None = None,
                    repeats: int | None = None,
                    model_roles: Sequence[str] | None = None) -> dict[str, Any]:
    protocol, checks = build_protocol(
        paths, base_config_path=base_config_path, blocks=blocks, backend=backend,
        campaign_id=campaign_id, budget=budget, statistics=statistics,
        include_branches=include_branches,
        allow_missing_credentials=allow_missing_credentials, rehearsal=rehearsal,
        shapes=shapes, repeats=repeats, model_roles=model_roles)
    protocol["status"] = "frozen"
    protocol["frozen_at"] = utc_now()
    protocol["frozen_by_harness"] = HARNESS_VERSION
    protocol["validation_checks"] = checks
    protocol["draft_checks"] = "see validation_checks; every entry must be ok"

    validate(protocol, require_frozen=True, paths=paths,
             allow_missing_credentials=allow_missing_credentials)

    existing = paths.frozen_protocol
    if existing.exists():
        old = read_json(existing)
        if old.get("protocol_hash") and old.get("protocol_hash") != sha256_value(
                {k: v for k, v in protocol.items() if k != "protocol_hash"}):
            archive = paths.root / f"FROZEN_PROTOCOL.{utc_now().replace(':', '')}.json"
            archive.write_bytes(existing.read_bytes())
            protocol["supersedes"] = archive.name
    written = write_protocol(paths, protocol, filename="FROZEN_PROTOCOL.json",
                             validate_first=False)
    write_protocol(paths, dict(written, status="draft"), filename="protocol.json",
                   validate_first=False)
    return written
