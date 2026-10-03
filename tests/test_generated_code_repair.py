from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from SCOPE.verification.gemm_semantic_checker import check_gemm_semantic_obligations
from SCOPE.verification.memory_access_plan import (
    enforce_memory_access_plan,
    memory_access_consistency_defects,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CODE_ROOT = ROOT / "gemm_code" / "code"
DEFAULT_REPORT = ROOT / "results" / "check" / "generated_code_repair_audit.json"


def audit_generated_terminal_chains(
    code_root: Path,
    compile_repaired: bool = False,
    run_repaired: bool = False,
    build_platform: str = "windows",
) -> dict:
    records = []
    terminal_dirs = sorted(
        path for path in code_root.glob("chain.*")
        if path.is_dir() and (path / "terminal_chain.json").exists() and (path / "cuda_kernel.cuh").exists()
    )
    for source_dir in terminal_dirs:
        source = (source_dir / "cuda_kernel.cuh").read_text(encoding="utf-8", errors="replace")
        before = memory_access_consistency_defects(source)
        record = {
            "chain": source_dir.name,
            "defects_before": before,
            "repair_required": bool(before),
        }
        if not before:
            record.update(status="clean", semantic_status="not_run", compile_status="not_run")
            records.append(record)
            continue

        with tempfile.TemporaryDirectory(prefix="scope-chain-repair-") as folder:
            repaired_dir = Path(folder)
            for name in ("cuda_kernel.cuh", "main.cpp", "kernel.h"):
                path = source_dir / name
                if path.exists():
                    shutil.copy2(path, repaired_dir / name)
            # Standalone scalar baseline audit, not strategy-preserving repair.
            repair = enforce_memory_access_plan(repaired_dir, allow_scalar_fallback=True)
            record["preserves_selected_strategies"] = not repair.get("materialized", False)
            semantic = check_gemm_semantic_obligations(repaired_dir, {})
            record.update(
                status="pass" if repair.get("status") == "pass" and semantic.get("accepted") else "fail",
                repair=repair,
                semantic_status="pass" if semantic.get("accepted") else "fail",
                semantic_failures=[
                    item for item in semantic.get("results", []) if item.get("status") == "fail"
                ],
                defects_after=memory_access_consistency_defects(
                    (repaired_dir / "cuda_kernel.cuh").read_text(encoding="utf-8", errors="replace")
                ),
                compile_status="not_run",
            )
            if (compile_repaired or run_repaired) and record["status"] == "pass":
                from SCOPE.utils.common_utils import load_json
                from SCOPE.verification.build_run_verifier import (
                    DEFAULT_VCVARS64,
                    compile_gemm,
                    verify_build_and_run,
                )

                ir_path = ROOT / "data" / "IRs" / "ir_patch" / "optir.extracted.json"
                ir = load_json(ir_path) if ir_path.exists() else {"hardware": {"compute_capability": "8.9"}}
                executable = repaired_dir / "build" / ("gemm" if build_platform == "linux" else "gemm.exe")
                executable.parent.mkdir(parents=True, exist_ok=True)
                if run_repaired:
                    verified = verify_build_and_run(
                        ir, repaired_dir, executable.parent, executable.name, 120,
                        DEFAULT_VCVARS64, build_platform, benchmark_runs=1, benchmark_warmup_runs=0,
                    )
                    verification = verified.get("verification", {}) or {}
                    record["compile_status"] = (verification.get("compile") or {}).get("status")
                    record["correctness_status"] = (verification.get("correctness") or {}).get("status")
                    record["runtime_safety_status"] = (verification.get("runtime_safety") or {}).get("status")
                    record["run_error"] = (
                        (verification.get("runtime_safety") or {}).get("cuda_error")
                        or (verification.get("correctness") or {}).get("error_message")
                        or (verification.get("compile") or {}).get("error_message")
                    )
                    if not (
                        record["compile_status"] == "pass"
                        and record["correctness_status"] == "pass"
                        and record["runtime_safety_status"] != "fail"
                    ):
                        record["status"] = "fail"
                else:
                    compile_result = compile_gemm(
                        ir, repaired_dir, executable, 120, DEFAULT_VCVARS64, build_platform,
                    )
                    record["compile_status"] = compile_result.get("status")
                    record["compile_error"] = compile_result.get("error_message")
                    if record["compile_status"] != "pass":
                        record["status"] = "fail"
        records.append(record)

    failed = [item for item in records if item.get("status") == "fail"]
    return {
        "code_root": str(code_root),
        "terminal_chain_count": len(terminal_dirs),
        "repair_required_count": sum(bool(item.get("repair_required")) for item in records),
        "repair_pass_count": sum(item.get("status") == "pass" for item in records),
        "repair_fail_count": len(failed),
        "accepted": not failed,
        "chains": records,
    }


class GeneratedCodeRepairTests(unittest.TestCase):
    def test_current_generated_terminal_chains_can_be_repaired_in_isolation(self):
        if not DEFAULT_CODE_ROOT.exists():
            self.skipTest("generated code directory is unavailable")
        report = audit_generated_terminal_chains(DEFAULT_CODE_ROOT)
        if not report["terminal_chain_count"]:
            self.skipTest("no terminal chains are available")
        self.assertGreater(report["repair_required_count"], 0)
        self.assertTrue(report["accepted"], json.dumps(report, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit deterministic repair of generated SCOPE terminal chains.")
    parser.add_argument("--code-root", type=Path, default=DEFAULT_CODE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--compile", action="store_true", help="also compile each repaired temporary chain")
    parser.add_argument("--run", action="store_true", help="compile and run correctness once for each repaired chain")
    parser.add_argument("--build-platform", choices=("windows", "linux"), default="windows")
    args = parser.parse_args()
    report = audit_generated_terminal_chains(args.code_root, args.compile, args.run, args.build_platform)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "chains"}, indent=2, ensure_ascii=False))
    raise SystemExit(0 if report["accepted"] else 1)


if __name__ == "__main__":
    main()
