from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from . import (HARNESS_VERSION, ablation_audit, audit as audit_module,
               branches as branches_module, build, config as config_module,
               controls as controls_module, data, datasets, freeze, isolation,
               ledger as ledger_module, paths as paths_module, pilot as pilot_module,
               protocol as protocol_module, runner, selfcheck, summarize, tasks,
               timing)
from .errors import ConfigError, GateError, InfraError, ProtocolError, ScopeExpError
from .util import ensure_dir, read_json, sha256_value, utc_now, write_json


def _campaign(args: argparse.Namespace) -> paths_module.CampaignPaths:
    return paths_module.campaign_paths(args.campaign).initialize()


def _blocks(args: argparse.Namespace) -> list[str]:
    return [block.upper() for block in args.blocks]


def _base_config(args: argparse.Namespace) -> Path:
    if args.base_config:
        return Path(args.base_config)
    repo = paths_module.repo_root()
    for name in ("conf.yaml", "conf.experiment.yaml", "conf.example.yaml"):
        candidate = repo / name
        if candidate.exists():
            return candidate
    raise ConfigError("no base configuration found; copy conf.experiment.yaml to "
                      "conf.yaml and fill in the provider section")


def _pricing(args: argparse.Namespace) -> dict[str, Any] | None:
    if not getattr(args, "pricing", None):
        return None
    return read_json(Path(args.pricing))


def _print(payload: Any) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=False))


def _load_frozen(args: argparse.Namespace, *,
                 allow_missing_credentials: bool = False) -> dict[str, Any]:
    return protocol_module.load(_campaign(args), frozen=True,
                                allow_missing_credentials=allow_missing_credentials)


def cmd_prepare(args: argparse.Namespace) -> int:
    paths = _campaign(args)
    repo = paths_module.repo_root()
    write_json(paths.layout_json, paths_module.layout_manifest(paths))
    environment = build.build_environment_record()
    write_json(paths.environment, environment)

    base_config = _base_config(args)
    snapshot = freeze.source_snapshot()
    snapshot["base_config_path"] = str(base_config)
    snapshot["base_config_sha256"] = sha256_value(config_module.load_yaml(base_config))
    write_json(paths.source_snapshot, snapshot)

    if args.shapes:
        shapes = [int(shape) for shape in args.shapes]
    elif args.profile == "cpu":
        shapes = list(tasks.CPU_SHAPES)
    else:
        shapes = tasks.shapes_for_blocks(_blocks(args))
    splits = list(args.splits) if args.splits else list(data.SPLITS)
    summary: dict[str, Any] = {
        "campaign_id": paths.campaign_id,
        "campaign_root": str(paths.root),
        "blocks": _blocks(args),
        "shapes": shapes,
        "splits": list(splits),
        "backend": args.backend,
        "created_at": utc_now(),
        "harness_version": HARNESS_VERSION,
        "base_config": str(base_config),
        "base_config_sha256": snapshot["base_config_sha256"],
        "source_digest": snapshot["source_digest"],
    }
    if args.skip_datasets:
        summary["datasets"] = {"status": "skipped"}
    else:
        plan = datasets.DatasetPlan(shapes=shapes, splits=tuple(splits),
                                    include_coefficients=not args.no_coefficients)
        summary["datasets"] = datasets.write_datasets(paths, plan, jobs=args.dataset_jobs)
    protocol = protocol_module.blank_protocol(paths.campaign_id, backend=args.backend,
                                              shapes=shapes, blocks=_blocks(args))
    protocol["source"] = snapshot
    protocol["environment"] = environment
    protocol["created_at"] = utc_now()
    protocol["prepare_summary"] = summary
    protocol_module.write_protocol(paths, protocol, filename="protocol.json",
                                   validate_first=False)
    _print({"prepared": str(paths.root), "summary": summary})
    if not args.quiet:
        print(f"\nnext: python -m scope_exp selfcheck --campaign {paths.campaign_id}\n"
              f"      python -m scope_exp controls --campaign {paths.campaign_id} "
              f"--backend {args.backend}")
    return 0


def cmd_selfcheck(args: argparse.Namespace) -> int:
    paths = _campaign(args)
    record = selfcheck.run(paths, config_path=args.config and Path(args.config),
                           skip_controls=args.skip_controls)
    print(selfcheck.render(record))
    return 0 if record["ok"] else 4


def cmd_controls(args: argparse.Namespace) -> int:
    paths = _campaign(args)
    record = controls_module.run_controls(paths, backend=args.backend,
                                          shape=args.shape,
                                          include_coefficients=not args.no_coefficients,
                                          timeout_seconds=args.timeout)
    print(f"controls {'PASS' if record['ok'] else 'FAIL'} "
          f"(backend={record.get('backend')}, gate={paths.audit / 'controls_gate.json'})")
    for check in record.get("checks", []):
        mark = "ok  " if check["ok"] else "FAIL"
        print(f"  [{mark}] {check['control']['name']}: main={check.get('observed_main')} "
              f"coefficients={check.get('observed_coefficients')}")
    print(f"  [{'ok  ' if record.get('reference', {}).get('ok') else 'FAIL'}] reference validity")
    order = record.get("parameter_order", {})
    print(f"  [{'ok  ' if order.get('ok') else 'FAIL'}] parameter order "
          f"(declared={order.get('results', {}).get('declared_mnk_order', {}).get('verdict')}, "
          f"legacy={order.get('results', {}).get('legacy_mkn_order', {}).get('verdict')})")
    if record.get("problems"):
        print("problems:")
        for problem in record["problems"]:
            print(f"  - {problem}")
    return 0 if record["ok"] else 4


def cmd_ablation_audit(args: argparse.Namespace) -> int:
    paths = _campaign(args)
    backend = args.backend
    if protocol_module.protocol_status(paths).get("present"):
        try:
            backend = str(protocol_module.load(
                paths, frozen=True, allow_missing_credentials=True).get("backend")
                or backend)
        except ProtocolError:
            pass
    record = ablation_audit.run(paths, backend=backend)
    print(f"ablation audit {'PASS' if record['ok'] else 'FAIL'} -> "
          f"{paths.audit / 'ablation_diff.json'}")
    for problem in record.get("problems", []):
        print(f"  - {problem}")
    for row in record["activation"]["rows"]:
        state = ("n/a" if row["activated_in_fixture"] is None
                 else "yes" if row["activated_in_fixture"] else "NO")
        print(f"  {row['variant']:<30} activated={state:<4} {row['evidence'][:90]}")
    return 0 if record["ok"] else 4


def cmd_freeze(args: argparse.Namespace) -> int:
    paths = _campaign(args)
    if args.allow_missing_credentials:
        print("warning: freezing with unresolved credentials; only rehearsal or "
              "pre-registration is valid in this state")
    budget = {
        "per_task_wall_seconds": args.per_task_wall_seconds,
        "pilot_wall_seconds": args.pilot_wall_seconds or args.per_task_wall_seconds,
        "api_timeout_seconds": args.api_timeout_seconds,
        "max_retries": args.max_retries,
        "validation_timeout_seconds": args.validation_timeout_seconds,
        "sanitizer_timeout_seconds": args.sanitizer_timeout_seconds,
        "repeat_checks": args.repeat_checks,
        "hypotheses_max_shape": args.hypotheses_max_shape,
        "per_phase_wall_seconds": {
            "p1": args.per_task_wall_seconds * args.phase_split[0],
            "p2": args.per_task_wall_seconds * args.phase_split[1],
            "p3": args.per_task_wall_seconds * args.phase_split[2],
        },
    }
    statistics = {"alpha": args.alpha,
                  "bootstrap_resamples": args.bootstrap_resamples,
                  "bootstrap_seed": args.bootstrap_seed}
    protocol = freeze.freeze_campaign(
        paths, base_config_path=_base_config(args), blocks=_blocks(args),
        backend=args.backend, campaign_id=paths.campaign_id, budget=budget,
        statistics=statistics, include_branches=not args.no_branches,
        allow_missing_credentials=args.allow_missing_credentials or args.rehearsal,
        rehearsal=args.rehearsal, shapes=args.shapes, repeats=args.repeats,
        model_roles=args.models)
    counts = protocol["tasks"]["counts"]
    _print({"frozen": str(paths.frozen_protocol),
            "protocol_hash": protocol.get("protocol_hash"),
            "backend": protocol["backend"],
            "shapes": protocol["shapes"],
            "counts": counts,
            "denominators": protocol["tasks"]["denominators"],
            "controls_gate": protocol["controls_gate"].get("ok"),
            "models": sorted(protocol["models"])})
    return 0


def _run_options(args: argparse.Namespace, *, provenance: str = "formal",
                 backend: str | None = None) -> runner.RunOptions:
    return runner.RunOptions(
        campaign_id=args.campaign, backend=backend or args.backend,
        generator=args.generator,
        mode=args.mode, device=args.device, keep_workspace=args.keep_workspace,
        skip_sanitizer=args.skip_sanitizer, allow_contended=args.allow_contended,
        dry_run=args.dry_run, evaluate=not args.no_evaluate,
        time_kernel=not args.no_timing, provenance=provenance,
        pricing=_pricing(args), lock_timeout_seconds=args.lock_timeout_seconds)


def cmd_run(args: argparse.Namespace) -> int:
    paths = _campaign(args)
    protocol = _load_frozen(args,
                            allow_missing_credentials=args.allow_missing_credentials)
    if args.allow_missing_credentials:
        print("warning: credentials are not resolved; only a rehearsal or a "
              "dry run makes sense in this state")
    controls_module.require_gate(paths, backend=protocol["backend"])
    if args.backend != protocol["backend"]:
        print(f"note: the frozen protocol runs backend={protocol['backend']!r}; "
              f"--backend {args.backend!r} is ignored")
    options = _run_options(args, backend=protocol["backend"])
    if options.generator == "rehearsal" and options.provenance != "rehearsal":
        options.provenance = "rehearsal"
        print("warning: rehearsal generator selected; rows are labelled rehearsal")
    summary = runner.run_campaign(
        paths, protocol, options, task_ids=args.task_ids, limit=args.limit,
        blocks=args.blocks, stop_on_infra_error=not args.continue_on_infra,
        progress=lambda stage, task_id, payload=None:
        print(f"[{stage}] {task_id}", flush=True))
    if options.dry_run:
        _print({"dry_run": True,
                "backend": protocol["backend"],
                "queue": len(summary["queue"]),
                "previews": [{"task_id": record.get("task_id"),
                              "argv": record.get("argv"),
                              "config_hash": record.get("config_hash")}
                             for record in summary["completed"]],
                "note": "no status was written; SCOPE_EXP_BACKEND is injected at run time"})
        return 0
    _print({"counts": summary["counts"], "queue": len(summary["queue"]),
            "infra_failures": summary["infra_failures"],
            "run_summary": str(paths.analysis / "run_summary.json")})
    return 0 if summary["infra_failures"] == 0 else 4


def cmd_evaluate(args: argparse.Namespace) -> int:
    paths = _campaign(args)
    protocol = _load_frozen(args, allow_missing_credentials=True)
    options = _run_options(args, provenance=args.provenance,
                           backend=protocol["backend"])
    options.mode = "full"
    if args.task_ids:
        task_ids = list(args.task_ids)
    else:
        task_ids = [status_path.parent.name
                    for status_path in sorted(Path(paths.tasks).glob("*/status.json"))
                    if read_json(status_path).get("status") == "frozen_top1"]
    if not task_ids:
        print("nothing to evaluate: no task is in status 'frozen_top1'")
        return 0
    if args.limit:
        task_ids = task_ids[: args.limit]
    results = []
    for task_id in task_ids:
        try:
            status = runner.evaluate_task(paths, protocol, task_id, options,
                                         progress=lambda stage, tid, payload=None:
                                         print(f"[{stage}] {tid}", flush=True))
            results.append({"task_id": task_id, "status": status.get("status"),
                            "verdict": (status.get("validation") or {}).get("verdict"),
                            "gflops": (status.get("timing") or {}).get("gflops_mean")})
        except ScopeExpError as exc:
            results.append({"task_id": task_id, "error": str(exc)})
            break
    _print({"evaluated": results})
    return 0


def cmd_pilot(args: argparse.Namespace) -> int:
    paths = _campaign(args)
    protocol = _load_frozen(args, allow_missing_credentials=args.allow_missing_credentials)
    plan = pilot_module.plan(paths, protocol)
    print(f"pilot plan: {len(plan['tasks'])} tasks, "
          f"budget {plan['budget']['per_task_wall_seconds']}s each")
    if args.plan_only:
        _print(plan["tasks"])
        return 0
    controls_module.require_gate(paths, backend=protocol["backend"])
    options = _run_options(args, provenance="pilot", backend=protocol["backend"])
    outcome = pilot_module.run(paths, protocol, options, task_ids=args.task_ids)
    _print({"run": outcome["run"]["counts"],
            "report": str(paths.analysis / "pilot_report.json"),
            "verdict": outcome["report"]["verdict"],
            "sample_size": outcome["report"]["sample_size"]["recommendation"],
            "budget": outcome["report"]["budget_recommendation"]})
    return 0 if outcome["report"]["verdict"] != "infrastructure_blocked" else 4


def cmd_branches(args: argparse.Namespace) -> int:
    paths = _campaign(args)
    protocol = _load_frozen(args, allow_missing_credentials=True)
    plan = branches_module.enumerate_branch_plan(protocol)
    if args.dry_run:
        _print({"planned": len(plan), "branches": plan[:20]})
        return 0
    results: list[dict[str, Any]] = []
    selected = [item for item in plan
                if args.kind == "both" or
                (args.kind == "without_feedback" and item["kind"] == "without_feedback") or
                (args.kind == "without_parameter_search"
                 and item["kind"] == "without_parameter_search")]
    if args.parent:
        selected = [item for item in selected if item["parent_task_id"] == args.parent]
    if args.limit:
        selected = selected[: args.limit]
    for item in selected:
        parent_status = runner.read_status(paths, item["parent_task_id"])
        if parent_status.get("status") not in ("resolved", "no_candidate",
                                               "generation_failed"):
            results.append({"parent": item["parent_task_id"],
                            "status": "skipped_parent_not_finished"})
            continue
        if item["kind"] == "without_parameter_search":
            results.append(branches_module.run_alias_without_parameter_search(
                paths, protocol, item["parent_task_id"],
                backend=protocol["backend"], device=args.device))
        else:
            results.append(branches_module.run_without_feedback(
                paths, protocol, item["parent_task_id"],
                backend=protocol["backend"], device=args.device))
    _print({"branches": [{"parent": record.get("parent_task_id"),
                          "kind": record.get("kind"),
                          "status": record.get("status")}
                         for record in results]})
    return 0


def cmd_abandon(args: argparse.Namespace) -> int:
    paths = _campaign(args)
    results = []
    for task_id in args.task_ids:
        try:
            results.append(runner.abandon_task(paths, task_id, reason=args.reason))
        except ScopeExpError as exc:
            results.append({"task_id": task_id, "error": str(exc)})
            break
    _print({"abandoned": results,
            "log": str(paths.audit / "abandoned.jsonl")})
    return 0


def cmd_summarize(args: argparse.Namespace) -> int:
    paths = _campaign(args)
    protocol = None
    try:
        protocol = protocol_module.load(paths, frozen=True,
                                        allow_missing_credentials=True)
    except ProtocolError as exc:
        print(f"warning: no usable frozen protocol ({exc}); reporting from raw records only")
    summary = summarize.summarize_campaign(
        paths, include_rehearsal=args.include_rehearsal, alpha=args.alpha,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed, protocol=protocol)
    outputs = summarize.write_reports(paths, summary)
    figure_result = summarize.figures(paths, summary) if args.figures else \
        {"status": "skipped"}
    _print({"tables": {name: str(path) for name, path in outputs.items()},
            "rows": summary["counts"],
            "figures": figure_result})
    print(f"\nreport: {outputs['RESULTS_zh.md']}")
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    paths = _campaign(args)
    protocol = None
    try:
        protocol = protocol_module.load(paths, frozen=True,
                                        allow_missing_credentials=True)
    except ProtocolError as exc:
        print(f"warning: no usable frozen protocol ({exc})")
    report = audit_module.run(paths, protocol=protocol, deep_inputs=args.deep)
    print(f"audit {'PASS' if report['ok'] else 'FAIL'} -> {paths.audit / 'report.json'}")
    for name, section in report["sections"].items():
        mark = "ok  " if section.get("ok") else "FAIL"
        print(f"  [{mark}] {name}")
    for problem in report["problems"]:
        print(f"  - {problem}")
    return 0 if report["ok"] else 4


def cmd_status(args: argparse.Namespace) -> int:
    paths = _campaign(args)
    protocol_state = protocol_module.protocol_status(paths)
    record: dict[str, Any] = {
        "campaign": paths.campaign_id,
        "root": str(paths.root),
        "harness_version": HARNESS_VERSION,
        "protocol": protocol_state,
        "controls_gate": controls_module.gate_status(paths),
        "ablation_audit": ablation_audit.audit_status(paths),
        "selfcheck": (read_json(paths.audit / "selfcheck.json").get("ok")
                      if (paths.audit / "selfcheck.json").exists() else None),
        "datasets": datasets.dataset_summary(paths),
        "analysis": summarize.analysis_status(paths),
        "audit": audit_module.audit_status(paths),
    }
    if protocol_state.get("present"):
        try:
            protocol = protocol_module.load(paths, frozen=protocol_state["frozen"],
                                            allow_missing_credentials=True)
            record["counts"] = runner._count_statuses(
                paths, [task["task_id"] for task in
                        (protocol.get("tasks") or {}).get("task_records") or []])
            record["denominators"] = (protocol.get("tasks") or {}).get("denominators")
        except ProtocolError as exc:
            record["counts_error"] = str(exc)
    _print(record)
    return 0


def cmd_rehearse(args: argparse.Namespace) -> int:
    paths = _campaign(args)
    repo = paths_module.repo_root()
    base_config = Path(args.base_config) if args.base_config else repo / "conf.experiment.yaml"
    print(f"[1/6] prepare {paths.root}")
    plan = datasets.DatasetPlan(shapes=[tasks.REHEARSAL_SHAPE],
                                splits=("development", "generation",
                                        "postselection_test"),
                                include_coefficients=True)
    datasets_summary = datasets.write_datasets(paths, plan, jobs=args.dataset_jobs)
    environment = build.build_environment_record()
    write_json(paths.environment, environment)
    write_json(paths.layout_json, paths_module.layout_manifest(paths))
    snapshot = freeze.source_snapshot()
    snapshot["base_config_path"] = str(base_config)
    write_json(paths.source_snapshot, snapshot)

    print("[2/6] controls (cpu)")
    gate = controls_module.run_controls(paths, backend="cpu", shape=32,
                                        include_coefficients=False)
    print(f"      controls {'ok' if gate.get('ok') else 'FAILED'}")
    if not gate.get("ok"):
        print("      cannot rehearse with a red control gate")
        return 3

    print("[3/6] ablation audit")
    ablation = ablation_audit.run(paths)
    print(f"      ablation audit {'ok' if ablation.get('ok') else 'FAILED'}")

    print("[4/6] freeze rehearsal protocol")
    protocol = freeze.freeze_campaign(
        paths, base_config_path=base_config, blocks=["REHEARSAL"], backend="cpu",
        shapes=[int(tasks.REHEARSAL_SHAPE)],
        campaign_id=paths.campaign_id,
        budget={"per_task_wall_seconds": args.per_task_wall_seconds,
                "pilot_wall_seconds": args.per_task_wall_seconds,
                "api_timeout_seconds": 60.0, "max_retries": 1,
                "validation_timeout_seconds": 900.0,
                "sanitizer_timeout_seconds": 300.0,
                "repeat_checks": 2, "hypotheses_max_shape": 128},
        statistics={"alpha": 0.05, "bootstrap_resamples": 10000},
        include_branches=False, allow_missing_credentials=True, rehearsal=True)
    print(f"      frozen {protocol.get('protocol_hash', '')[:16]} "
          f"tasks={protocol['tasks']['counts']}")

    print("[5/6] run rehearsal tasks")
    options = runner.RunOptions(campaign_id=paths.campaign_id, backend="cpu",
                                generator="rehearsal", mode="full", device=0,
                                keep_workspace=args.keep_workspace,
                                skip_sanitizer=True, provenance="rehearsal")
    run_summary = runner.run_campaign(paths, protocol, options)
    print(f"      {run_summary['counts']}")

    print("[6/6] summarize + audit")
    summary = summarize.summarize_campaign(paths, include_rehearsal=True,
                                           protocol=protocol)
    outputs = summarize.write_reports(paths, summary)
    report = audit_module.run(paths, protocol=protocol)
    _print({"datasets": datasets_summary.get("manifest_hash"),
            "run": run_summary["counts"],
            "analysis": str(outputs["RESULTS_zh.md"]),
            "audit_ok": report["ok"],
            "audit_problems": report["problems"],
            "note": "all rows are provenance=rehearsal; they are excluded from "
                    "formal analysis unless --include-rehearsal is passed"})
    return 0 if report["ok"] else 4


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scope_exp",
        description="SCOPE experiment harness: reproducible, auditable, "
                    "gate-controlled runs of 实验设计.md.")
    parser.add_argument("--campaign", default="scope-core",
                        help="campaign id; results/<campaign_id>/")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(sp: argparse.ArgumentParser, *, default_backend: str = "cuda") -> None:
        sp.add_argument("--campaign", default=argparse.SUPPRESS,
                        help="campaign id (overrides the global one)")
        sp.add_argument("--base-config", default=None,
                        help="path to the base YAML (default: conf.yaml, then "
                             "conf.experiment.yaml)")
        sp.add_argument("--backend", choices=("cuda", "cpu"), default=default_backend)

    sp = sub.add_parser("prepare", help="M0: layout, environment, datasets, draft")
    add_common(sp)
    sp.add_argument("--blocks", nargs="+", default=["E1", "E2"],
                    choices=["E1", "E2", "E3", "D1", "PILOT", "REHEARSAL"])
    sp.add_argument("--shapes", nargs="+", type=int, default=None)
    sp.add_argument("--profile", choices=("gpu", "cpu"), default="gpu",
                    help="gpu = 512/1024/2048/4096; cpu = 256/512/768/1024")
    sp.add_argument("--splits", nargs="+", default=None,
                    choices=["development", "generation", "postselection_test"])
    sp.add_argument("--dataset-jobs", type=int, default=4)
    sp.add_argument("--no-coefficients", action="store_true",
                    help="skip the 128 coefficient checks (debug only)")
    sp.add_argument("--skip-datasets", action="store_true")
    sp.add_argument("--quiet", action="store_true")
    sp.set_defaults(func=cmd_prepare)

    sp = sub.add_parser("selfcheck", help="environment + reference + control + timing")
    add_common(sp, default_backend="cpu")
    sp.add_argument("--config", default=None)
    sp.add_argument("--skip-controls", action="store_true")
    sp.set_defaults(func=cmd_selfcheck)

    sp = sub.add_parser("controls", help="E0 positive/negative controls")
    add_common(sp)
    sp.add_argument("--shape", type=int, default=controls_module.CONTROL_SHAPE)
    sp.add_argument("--no-coefficients", action="store_true")
    sp.add_argument("--timeout", type=float, default=600.0)
    sp.set_defaults(func=cmd_controls)

    sp = sub.add_parser("ablation-audit", help="REQ-05 diff and leakage audit")
    add_common(sp, default_backend="cpu")
    sp.set_defaults(func=cmd_ablation_audit)

    sp = sub.add_parser("freeze", help="M3: write FROZEN_PROTOCOL.json")
    add_common(sp)
    sp.add_argument("--blocks", nargs="+", default=["E1", "E2"],
                    choices=["E1", "E2", "E3", "D1", "PILOT", "REHEARSAL"])
    sp.add_argument("--per-task-wall-seconds", type=float, required=True,
                    help="hard per-task wall limit; required by design")
    sp.add_argument("--pilot-wall-seconds", type=float, default=None)
    sp.add_argument("--api-timeout-seconds", type=float, default=300.0)
    sp.add_argument("--max-retries", type=int, default=5)
    sp.add_argument("--validation-timeout-seconds", type=float, default=1800.0)
    sp.add_argument("--sanitizer-timeout-seconds", type=float, default=900.0)
    sp.add_argument("--repeat-checks", type=int, default=8)
    sp.add_argument("--hypotheses-max-shape", type=int, default=512)
    sp.add_argument("--phase-split", nargs=3, type=float, default=[0.5, 0.3, 0.2],
                    metavar=("P1", "P2", "P3"))
    sp.add_argument("--bootstrap-resamples", type=int, default=10000)
    sp.add_argument("--bootstrap-seed", type=int, default=20261008)
    sp.add_argument("--alpha", type=float, default=0.05)
    sp.add_argument("--shapes", nargs="+", type=int, default=None,
                    help="override the shape list (CPU extension: 256 512 768 1024)")
    sp.add_argument("--repeats", type=int, default=None,
                    help="independent generations per configuration per shape")
    sp.add_argument("--models", nargs="+", default=None,
                    help="model roles to resolve (default: the provider's own model)")
    sp.add_argument("--no-branches", action="store_true")
    sp.add_argument("--allow-missing-credentials", action="store_true")
    sp.add_argument("--rehearsal", action="store_true")
    sp.set_defaults(func=cmd_freeze)

    sp = sub.add_parser("run", help="M4: execute the frozen queue")
    add_common(sp)
    sp.add_argument("--task-ids", nargs="+", default=None)
    sp.add_argument("--limit", type=int, default=None)
    sp.add_argument("--blocks", nargs="+", default=None)
    sp.add_argument("--device", type=int, default=0)
    sp.add_argument("--generator", choices=("app", "rehearsal"), default="app")
    sp.add_argument("--mode", choices=("full", "generate"), default="full")
    sp.add_argument("--provenance", default="formal",
                    choices=["formal", "rehearsal", "pilot"])
    sp.add_argument("--dry-run", action="store_true")
    sp.add_argument("--keep-workspace", action="store_true")
    sp.add_argument("--skip-sanitizer", action="store_true")
    sp.add_argument("--allow-contended", action="store_true")
    sp.add_argument("--allow-missing-credentials", action="store_true")
    sp.add_argument("--no-evaluate", action="store_true")
    sp.add_argument("--no-timing", action="store_true")
    sp.add_argument("--continue-on-infra", action="store_true")
    sp.add_argument("--lock-timeout-seconds", type=float, default=7200.0)
    sp.add_argument("--pricing", default=None, help="JSON price table per model")
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("evaluate", help="M5: validate and time frozen Top-1s")
    add_common(sp)
    sp.add_argument("--task-ids", nargs="+", default=None)
    sp.add_argument("--limit", type=int, default=None)
    sp.add_argument("--device", type=int, default=0)
    sp.add_argument("--provenance", default="formal")
    sp.add_argument("--generator", choices=("app", "rehearsal"), default="app")
    sp.add_argument("--mode", choices=("full", "generate"), default="full")
    sp.add_argument("--dry-run", action="store_true")
    sp.add_argument("--keep-workspace", action="store_true")
    sp.add_argument("--skip-sanitizer", action="store_true")
    sp.add_argument("--allow-contended", action="store_true")
    sp.add_argument("--no-evaluate", action="store_true")
    sp.add_argument("--no-timing", action="store_true")
    sp.add_argument("--continue-on-infra", action="store_true")
    sp.add_argument("--lock-timeout-seconds", type=float, default=7200.0)
    sp.add_argument("--pricing", default=None)
    sp.set_defaults(func=cmd_evaluate)

    sp = sub.add_parser("pilot", help="M2: ten-task pilot")
    add_common(sp)
    sp.add_argument("--plan-only", action="store_true")
    sp.add_argument("--task-ids", nargs="+", default=None)
    sp.add_argument("--device", type=int, default=0)
    sp.add_argument("--generator", choices=("app", "rehearsal"), default="app")
    sp.add_argument("--mode", choices=("full", "generate"), default="full")
    sp.add_argument("--dry-run", action="store_true")
    sp.add_argument("--keep-workspace", action="store_true")
    sp.add_argument("--skip-sanitizer", action="store_true")
    sp.add_argument("--allow-contended", action="store_true")
    sp.add_argument("--allow-missing-credentials", action="store_true")
    sp.add_argument("--no-evaluate", action="store_true")
    sp.add_argument("--no-timing", action="store_true")
    sp.add_argument("--lock-timeout-seconds", type=float, default=7200.0)
    sp.add_argument("--pricing", default=None)
    sp.set_defaults(func=cmd_pilot)

    sp = sub.add_parser("branches", help="shared-parent stage branches")
    add_common(sp)
    sp.add_argument("--kind", choices=("both", "without_feedback",
                                       "without_parameter_search"), default="both")
    sp.add_argument("--parent", default=None)
    sp.add_argument("--limit", type=int, default=None)
    sp.add_argument("--device", type=int, default=0)
    sp.add_argument("--dry-run", action="store_true")
    sp.set_defaults(func=cmd_branches)

    sp = sub.add_parser("abandon", help="archive failed attempts after a harness fix")
    add_common(sp, default_backend="cpu")
    sp.add_argument("--task-ids", nargs="+", required=True)
    sp.add_argument("--reason", required=True,
                    help="what was wrong with the harness; recorded in audit/abandoned.jsonl")
    sp.set_defaults(func=cmd_abandon)

    sp = sub.add_parser("summarize", help="rebuild every table from raw records")
    add_common(sp, default_backend="cpu")
    sp.add_argument("--include-rehearsal", action="store_true")
    sp.add_argument("--alpha", type=float, default=0.05)
    sp.add_argument("--bootstrap-resamples", type=int, default=10000)
    sp.add_argument("--bootstrap-seed", type=int, default=20261008)
    sp.add_argument("--figures", action="store_true")
    sp.set_defaults(func=cmd_summarize)

    sp = sub.add_parser("audit", help="E4 completeness and identity audit")
    add_common(sp, default_backend="cpu")
    sp.add_argument("--deep", action="store_true",
                    help="re-derive every input hash (slow)")
    sp.set_defaults(func=cmd_audit)

    sp = sub.add_parser("status", help="one screen of campaign state")
    add_common(sp, default_backend="cpu")
    sp.set_defaults(func=cmd_status)

    sp = sub.add_parser("rehearse", help="labelled end-to-end pipeline exercise")
    add_common(sp, default_backend="cpu")
    sp.add_argument("--per-task-wall-seconds", type=float, default=600.0)
    sp.add_argument("--dataset-jobs", type=int, default=2)
    sp.add_argument("--keep-workspace", action="store_true")
    sp.set_defaults(func=cmd_rehearse)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except GateError as exc:
        print(f"gate refused: {exc}", file=sys.stderr)
        return 3
    except ProtocolError as exc:
        print(f"protocol error: {exc}", file=sys.stderr)
        return 3
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 3
    except InfraError as exc:
        print(f"infrastructure error: {exc}", file=sys.stderr)
        return 4
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
