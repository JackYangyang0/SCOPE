#!/usr/bin/env python
"""Environment and gate summary for a campaign (M0 helper).

    python scripts/check_env.py
    python scripts/check_env.py --campaign scope-core

Prints the toolchain, the campaign gates and the denominator counts, and exits
non-zero when a formal run would be refused.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.harness import ablation_audit, build, controls, datasets, protocol as protocol_module, runner
from experiments.harness import (selfcheck)
from experiments.harness.paths import campaign_paths  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", default="scope-core")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    toolchains = {name: build.discover_toolchain(name).as_record()
                  for name in ("cpu", "cuda")}
    print("toolchains")
    for name, record in toolchains.items():
        print(f"  {name:<5} available={record['available']} "
              f"kind={record['compiler_kind']} {record['version_text'][:60]}")
    print(f"  sanitizer={__import__('SCOPE.verification.sanitizer', fromlist=['x']).sanitizer_available()}")

    paths = campaign_paths(args.campaign)
    if not paths.root.exists():
        print(f"\ncampaign {args.campaign} not prepared ({paths.root})")
        return 0 if args.json else 1

    report = {
        "campaign": args.campaign,
        "root": str(paths.root),
        "toolchains": toolchains,
        "selfcheck": (Path(paths.audit / "selfcheck.json").exists() and
                      __import__("SCOPE.utils.harness_util", fromlist=["read_json"])
                      .read_json(paths.audit / "selfcheck.json").get("ok")),
        "controls_gate": controls.gate_status(paths),
        "ablation_audit": ablation_audit.audit_status(paths),
        "datasets": datasets.dataset_summary(paths),
        "protocol": protocol_module.protocol_status(paths),
    }
    if report["protocol"].get("present"):
        try:
            protocol = protocol_module.load(paths, frozen=report["protocol"]["frozen"],
                                            allow_missing_credentials=True)
            report["counts"] = runner._count_statuses(
                paths, [task["task_id"] for task in
                        (protocol.get("tasks") or {}).get("task_records") or []])
            report["denominators"] = (protocol.get("tasks") or {}).get("denominators")
        except Exception as exc:
            report["counts_error"] = str(exc)

    if args.json:
        import json
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        gate = report["controls_gate"]
        print(f"\ncampaign {args.campaign}")
        print(f"  self-check      : {report['selfcheck']}")
        print(f"  controls gate   : present={gate.get('present')} ok={gate.get('ok')} "
              f"backend={gate.get('backend')}")
        print(f"  ablation audit  : {report['ablation_audit'].get('ok')}")
        print(f"  protocol        : {report['protocol'].get('status')}")
        print(f"  counts          : {report.get('counts')}")
        print(f"  denominators    : {report.get('denominators')}")
        blockers = []
        if not gate.get("ok"):
            blockers.append("controls gate missing or red")
        if not report["protocol"].get("frozen"):
            blockers.append("protocol not frozen")
        if report.get("counts_error"):
            blockers.append(report["counts_error"])
        if blockers:
            print(f"  formal run would be refused: {'; '.join(blockers)}")
            return 1
        print("  formal run preconditions look satisfied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
