from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

PACKAGE_PARENT = Path(__file__).resolve().parents[1]
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from SCOPE.generate_ir.ir_extraction import build_extracted_ir
from SCOPE.llm.c_code_generator import (
    CPU_CODE_FILES,
    apply_cpu_c_code_files,
    generate_cpu_c_code_with_llm,
)
from SCOPE.llm.openai_client import OpenAICompatibleClient
from SCOPE.llm.patch_generator import load_code_context
from SCOPE.utils.common_utils import load_config, load_json, save_json
from SCOPE.verification.c_build_run_verifier import verify_cpu_build_and_run


ROOT = Path(__file__).resolve().parent
DEFAULT_DESCRIPTION = (
    "I need to generate a high-performance CPU C GEMM implementation for "
    "row-major fp32 NN GEMM. Matrix Size: (M:768 N:768 K:768)."
)
DEFAULT_TEMPLATE = ROOT / "data" / "IRs" / "optir.json"
DEFAULT_IR_OUTPUT = ROOT / "data" / "IRs" / "ir_patch" / "optir.cpu.extracted.json"
DEFAULT_VERIFIED_IR_OUTPUT = ROOT / "data" / "IRs" / "ir_patch" / "optir.cpu.verified.json"
DEFAULT_CPU_SKELETON = ROOT / "gemm_code" / "cpu_skeleton"
DEFAULT_CPU_CODE_ROOT = ROOT / "gemm_code" / "cpu_code" / "chain.cpu"
DEFAULT_CPU_CODE_OUTPUT = ROOT / "results" / "code" / "cpu_c_code_files.json"
DEFAULT_CPU_SUMMARY_OUTPUT = ROOT / "results" / "check" / "cpu_generation_summary.json"
DEFAULT_PROMPT = ROOT / "llm" / "prompts" / "generate_cpu_c_code_prompt.txt"
DEFAULT_CONFIG = ROOT / "conf.yaml"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="SCOPE staged CPU/C GEMM generation entrypoint.")
    parser.add_argument("--description", default=DEFAULT_DESCRIPTION)
    parser.add_argument("--template", default=str(DEFAULT_TEMPLATE))
    parser.add_argument("--ir-output", default=str(DEFAULT_IR_OUTPUT))
    parser.add_argument("--verified-ir-output", default=str(DEFAULT_VERIFIED_IR_OUTPUT))
    parser.add_argument("--skeleton-dir", default=str(DEFAULT_CPU_SKELETON))
    parser.add_argument("--code-root", default=str(DEFAULT_CPU_CODE_ROOT))
    parser.add_argument("--code-output", default=str(DEFAULT_CPU_CODE_OUTPUT))
    parser.add_argument("--summary-output", default=str(DEFAULT_CPU_SUMMARY_OUTPUT))
    parser.add_argument("--prompt", default=str(DEFAULT_PROMPT))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--build-platform", choices=("windows", "linux"))
    parser.add_argument("--resume-terminal", action="store_true")
    parser.add_argument("--matrix-size", nargs=3, type=int, metavar=("M", "N", "K"))
    parser.add_argument("--skip-final-tuning", action="store_true")
    parser.add_argument("--keep-generated-cache", action="store_true")
    parser.add_argument("--skip-step-compile", action="store_true")
    parser.add_argument("--skip-stage-compile", action="store_true")
    parser.add_argument("--skip-initial-skeleton-check", action="store_true")
    parser.add_argument("--benchmark-runs", type=int)
    parser.add_argument("--benchmark-warmup-runs", type=int)
    parser.add_argument(
        "--direct-codegen",
        action="store_true",
        help="Use the legacy one-shot whole-kernel generator instead of staged strategy selection.",
    )
    parser.add_argument("--skip-llm", action="store_true", help="Direct-codegen mode only: compile the CPU skeleton without LLM generation.")
    args = parser.parse_args(argv)

    config = load_config(Path(args.config))
    if not args.direct_codegen:
        from SCOPE import app as strategy_app

        forwarded = [
            "--description", args.description,
            "--template", args.template,
            "--config", args.config,
        ]
        build_platform = args.build_platform or str((config.get("build", {}) or {}).get("platform") or "windows")
        forwarded.extend(("--build-platform", build_platform))
        if args.resume_terminal:
            forwarded.append("--resume-terminal")
        if args.matrix_size:
            forwarded.extend(("--matrix-size", *(str(value) for value in args.matrix_size)))
        if args.skip_final_tuning:
            forwarded.append("--skip-final-tuning")
        if args.keep_generated_cache:
            forwarded.append("--keep-generated-cache")
        if args.skip_step_compile:
            forwarded.append("--skip-step-compile")
        if args.skip_stage_compile:
            forwarded.append("--skip-stage-compile")
        if args.skip_initial_skeleton_check:
            forwarded.append("--skip-initial-skeleton-check")
        if args.benchmark_runs is not None:
            forwarded.extend(("--benchmark-runs", str(args.benchmark_runs)))
        if args.benchmark_warmup_runs is not None:
            forwarded.extend(("--benchmark-warmup-runs", str(args.benchmark_warmup_runs)))
        pipeline_args = strategy_app.parse_args(forwarded)
        strategy_app.run_pipeline(pipeline_args, backend="cpu")
        return

    template = load_json(Path(args.template))
    ir = build_extracted_ir(template, args.description)
    ir.setdefault("target", {}).update({"backend": "cpu", "language": "c", "device": "cpu"})
    save_json(Path(args.ir_output), ir)

    code_root = prepare_cpu_code_root(Path(args.skeleton_dir), Path(args.code_root))
    generated_code: dict[str, Any] | None = None
    apply_result = {"status": "skipped", "reason": "skip_llm enabled"}

    if not args.skip_llm:
        code_context = load_code_context(code_root, CPU_CODE_FILES)
        client = OpenAICompatibleClient(config["llm"])
        generated_code = generate_cpu_c_code_with_llm(
            client=client,
            ir=ir,
            code_context=code_context,
            prompt_path=Path(args.prompt),
        )
        apply_result = apply_cpu_c_code_files(generated_code, code_root)
        generated_code["apply_result"] = apply_result
        save_json(Path(args.code_output), generated_code)

    timeout_seconds = int(config.get("verification", {}).get("timeout_seconds", 60))
    build_config = config.get("build", {}) or {}
    baseline_config = config.get("cpu_baseline", {}) or {}
    openblas_root = baseline_config.get("openblas_root")
    verified_ir = verify_cpu_build_and_run(
        ir=ir,
        source_dir=code_root,
        build_dir=code_root / "build",
        timeout_seconds=timeout_seconds,
        openblas_root=Path(openblas_root) if openblas_root else None,
        build_platform=str(build_config.get("platform") or "windows"),
        benchmark_runs=int(build_config.get("benchmark_runs", 5)),
        benchmark_warmup_runs=int(build_config.get("benchmark_warmup_runs", 2)),
    )
    save_json(Path(args.verified_ir_output), verified_ir)

    summary = {
        "target": verified_ir.get("target", {}),
        "ir_output": str(Path(args.ir_output)),
        "verified_ir_output": str(Path(args.verified_ir_output)),
        "code_root": str(code_root),
        "code_output": str(Path(args.code_output)) if generated_code is not None else None,
        "apply_result": apply_result,
        "compile_status": verified_ir.get("verification", {}).get("compile", {}).get("status"),
        "correctness_status": verified_ir.get("verification", {}).get("correctness", {}).get("status"),
        "host_error": verified_ir.get("verification", {}).get("runtime_safety", {}).get("host_error"),
        "latency_ms": verified_ir.get("performance", {}).get("latency_ms"),
        "gflops": verified_ir.get("performance", {}).get("gflops"),
    }
    save_json(Path(args.summary_output), summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def prepare_cpu_code_root(skeleton_dir: Path, code_root: Path) -> Path:
    code_root.mkdir(parents=True, exist_ok=True)
    for relative_path in CPU_CODE_FILES:
        source = skeleton_dir / relative_path
        target = code_root / relative_path
        if not source.exists():
            raise FileNotFoundError(f"CPU skeleton file missing: {source}")
        shutil.copyfile(source, target)
    return code_root


if __name__ == "__main__":
    main()
