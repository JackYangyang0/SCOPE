"""Create a sanitized, reproducible SCOPE source release directory."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Iterable

import yaml


TEXT_SUFFIXES = {
    ".py", ".json", ".yaml", ".yml", ".txt", ".md", ".c", ".cpp",
    ".cu", ".cuh", ".h", ".hpp", ".sh", ".bat", ".csv", ".tex",
}
PACKAGE_DIRS = (
    "diagnosis",
    "generate_ir",
    "llm",
    "specialization",
    "utils",
    "verification",
    "tests",
    "docs",
)
TOP_LEVEL_CODE = (
    "__init__.py",
    "app.py",
    "cpu_app.py",
    "benchmark.py",
    "tune.py",
    "tune_cuda.py",
    "compare_kernel_controls.py",
    "compare_load_layout.py",
    "compare_reference_kernel.py",
    "compare_tile_loads.py",
    "diagnose_tiling_response.py",
)
DATA_FILES = (
    "data/IRs/optir.json",
    "data/IRs/win-ir.json",
    "data/graph/dependency_graph.json",
    "data/graph/dependency_graph_metadata.json",
    "data/graph/cpu_dependency_graph.json",
    "data/lib/build_strategy_examples.py",
    "data/lib/cpu_strategy_index.json",
    "data/lib/cpu_strategy_library.json",
    "data/lib/strategy_coupling_contracts.json",
    "data/lib/strategy_examples.json",
    "data/lib/strategy_examples_full.json",
    "data/lib/strategy_examples.md",
    "data/lib/strategy_index.json",
    "data/lib/strategy_library.json",
    "data/lib/verification_schema.json",
)
METHOD_TEMPLATES = (
    "gemm_code/skeleton",
    "gemm_code/skeleton_template",
    "gemm_code/cpu_skeleton",
    "gemm_code/baseline",
    "gemm_code/baseline_template",
    "gemm_code/cpu_baseline_template",
)
EXPERIMENT_SCRIPTS = (
    "scripts/prepare_github_release.py",
    "scripts/build_rq2_complete_report.py",
    "scripts/summarize_intent_only_repaired.py",
    "scripts/summarize_rq_experiments.py",
)
SUMMARY_FILES = (
    "RQ2_RQ3_FINDINGS.md",
    "EXPERIMENT_PROTOCOL_AUDIT.md",
    "rq2_complete_runs.csv",
    "rq2_complete_report.json",
    "rq3_optimization_runs.csv",
    "intent_only_repaired_runs.csv",
    "intent_only_repaired_summary.json",
    "figures/plot_rq2_figures.py",
    "figures/rq2_compact_ablation_analysis.pdf",
    "figures/rq2_compact_ablation_analysis.png",
)


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def copy_tree(source: Path, destination: Path) -> None:
    for path in source.rglob("*"):
        if not path.is_file() or (
            path.suffix.lower() not in TEXT_SUFFIXES and path.name != "Makefile"
        ):
            continue
        if "__pycache__" in path.parts or ".pytest_cache" in path.parts:
            continue
        if path.name.endswith("_compile_with_vcvars.bat"):
            continue
        copy_file(path, destination / path.relative_to(source))


def sanitized_config(source: Path) -> dict:
    config = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    llm = config.get("llm", {}) or {}
    env_names = {
        "qwen": "QWEN_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "chatgpt": "OPENAI_API_KEY",
        "cmecloud": "CMECLOUD_API_KEY",
    }
    for provider, env_name in env_names.items():
        provider_config = llm.get(provider)
        if isinstance(provider_config, dict):
            provider_config["api_key"] = ""
            provider_config["api_key_env"] = env_name
    if isinstance(config.get("cpu_baseline"), dict):
        config["cpu_baseline"]["openblas_root"] = None
    return config


def release_readme() -> str:
    return """# SCOPE

SCOPE is a structured, strategy-guided system for generating and optimizing
shape-specialized FP32 GEMM kernels for CUDA GPUs and multicore CPUs.

The repository and the importable Python package are both named `SCOPE`.

## Included artifact

- staged GPU and CPU generation entry points;
- typed IR extraction and validation;
- strategy libraries and dependency graphs;
- deterministic transformations and LLM patch generation;
- semantic checking, locked repair, compilation, correctness, and runtime gates;
- feedback optimization and constrained tile re-instantiation;
- CUDA/cuBLAS and CPU/OpenBLAS benchmark harnesses;
- RQ2/RQ3 ablation scripts, compact results, protocol audit, and tests;
- representative final CUDA and CPU kernel bundles.

Generated chains, API responses, build products, caches, and full raw experiment
logs are intentionally excluded from this source release.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate              # Linux
# .venv\\Scripts\\activate             # Windows
pip install -r requirements.txt
```

Set the API key for the provider selected in `SCOPE/conf.yaml`, for example:

```bash
export CMECLOUD_API_KEY=...
```

CUDA execution additionally requires an NVIDIA driver and CUDA Toolkit with
`nvcc`. CPU baseline comparison requires OpenBLAS headers and libraries.

## Run

```bash
python -m SCOPE.app --build-platform linux --matrix-size 512 512 512
python -m SCOPE.cpu_app --build-platform linux --matrix-size 512 512 512
python -m SCOPE.benchmark --backend gpu --program scope=/path/to/gemm
python -m SCOPE.tune --backend cuda --source /path/to/generated/kernel
```

Use `--build-platform windows` on Windows. See `SCOPE/docs/` for the chain,
shape-reuse, architecture, and RQ2/RQ3 experiment descriptions.

## Reproduce the compact RQ2 figure

```bash
python SCOPE/experiments/rq2_4060ti_512/figures/plot_rq2_figures.py
```

The exact input distributions, correctness thresholds, reference precision,
cuBLAS math mode, CPU threading, and timing boundaries are documented in
`SCOPE/experiments/rq2_4060ti_512/EXPERIMENT_PROTOCOL_AUDIT.md`.

## Credentials and data

No API credentials are included. Full raw generations can contain model output,
absolute paths, and substantial intermediate data, so publish them separately as
an archival dataset if required by the artifact evaluation policy.

## License

No license is selected in the research workspace. Add the license chosen by the
authors before making the GitHub repository public.
"""


def gitignore_text() -> str:
    return """__pycache__/
*.py[cod]
.pytest_cache/
.venv/
.env
*.exe
*.obj
*.o
*.lib
*.exp
*.pdb
build/
SCOPE/results/
SCOPE/gemm_code/code/
SCOPE/gemm_code/cpu_code/
SCOPE/gemm_code/shape_tuning/
SCOPE/gemm_code/final_shape_tuning/
SCOPE/data/IRs/ir_patch/
"""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scan_release(root: Path) -> list[str]:
    findings: list[str] = []
    secret_patterns = (
        re.compile(r"\bsk-[A-Za-z0-9_-]{12,}"),
        re.compile(r"(?i)api_key\s*:\s*['\"]?(?!['\"]?\s*$)([^\s#'\"]{12,})"),
    )
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for pattern in secret_patterns:
            if pattern.search(text):
                findings.append(f"possible credential: {path.relative_to(root)}")
                break
        if re.search(r"api_key_env\s*:\s*['\"]?[A-Za-z0-9]{24,}", text):
            findings.append(f"api_key_env is not an environment-variable name: {path.relative_to(root)}")
    return findings


def sanitize_exported_paths(root: Path, repo: Path) -> None:
    """Remove source-machine paths from copied human-readable artifacts."""
    replacements = {
        str(repo): "${SCOPE_REPO}",
        repo.as_posix(): "${SCOPE_REPO}",
    }
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        sanitized = text
        for source, replacement in replacements.items():
            sanitized = sanitized.replace(source, replacement)
        if sanitized != text:
            path.write_text(sanitized, encoding="utf-8")


def configure_release_benchmark(package: Path) -> None:
    benchmark = package / "benchmark.py"
    text = benchmark.read_text(encoding="utf-8")
    gpu = """DEFAULT_GPU_PROGRAMS = [
    f"scope={ROOT / '4060ti' / 'results' / 'build' / f'gemm{EXEEXT}'}",
]"""
    cpu = """DEFAULT_CPU_PROGRAMS = [
    f"scope={ROOT / 'CPU' / 'results' / 'build' / f'gemm_cpu{EXEEXT}'}",
]"""
    text, gpu_count = re.subn(
        r"DEFAULT_GPU_PROGRAMS\s*=\s*\[.*?\]\nDEFAULT_CPU_PROGRAMS",
        gpu + "\nDEFAULT_CPU_PROGRAMS",
        text,
        count=1,
        flags=re.DOTALL,
    )
    text, cpu_count = re.subn(
        r"DEFAULT_CPU_PROGRAMS\s*=\s*\[.*?\]\nDEFAULT_CPU_GENERATION_RESULTS",
        cpu + "\nDEFAULT_CPU_GENERATION_RESULTS",
        text,
        count=1,
        flags=re.DOTALL,
    )
    if gpu_count != 1 or cpu_count != 1:
        raise RuntimeError("Could not rewrite benchmark defaults for release")
    benchmark.write_text(text, encoding="utf-8")


def write_manifest(root: Path) -> None:
    generated_suffixes = {
        ".exe", ".obj", ".o", ".dll", ".so", ".lib", ".exp", ".pdb", ".pyc", ".ptx", ".cubin",
    }

    def is_native_binary(path: Path) -> bool:
        with path.open("rb") as stream:
            magic = stream.read(4)
        return magic.startswith(b"MZ") or magic == b"\x7fELF"

    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.name != "MANIFEST.json"
        and path.suffix.lower() not in generated_suffixes
        and not is_native_binary(path)
        and not any(part == "build" or part.startswith("build_") for part in path.relative_to(root).parts[:-1])
        and not {".git", ".idea", ".vscode"}.intersection(path.relative_to(root).parts)
        and "__pycache__" not in path.parts
    )
    manifest = {
        "artifact": "SCOPE source and compact experiment release",
        "python_package": "SCOPE",
        "file_count": len(files),
        "files": [
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in files
        ],
    }
    (root / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def require_new_destination(destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(
            f"Destination already exists: {destination}. Choose a new empty path; "
            "the release builder never deletes existing files."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    output = args.output.resolve()
    require_new_destination(output)
    package = output / "SCOPE"
    package.mkdir(parents=True)

    for relative in TOP_LEVEL_CODE:
        copy_file(repo / relative, package / relative)
    for relative in PACKAGE_DIRS:
        copy_tree(repo / relative, package / relative)
    for relative in DATA_FILES:
        copy_file(repo / relative, package / relative)
    for relative in METHOD_TEMPLATES:
        copy_tree(repo / relative, package / relative)
    for relative in EXPERIMENT_SCRIPTS:
        copy_file(repo / relative, package / relative)

    summary = repo / "results" / "experiments" / "rq2-4060ti-512-r1-summary"
    for relative in SUMMARY_FILES:
        source = summary / relative
        if source.exists():
            copy_file(source, package / "experiments" / "rq2_4060ti_512" / relative)

    artifact_sources = (
        (
            (
                repo / "4060ti" / "results",
                repo / "4060ti" / "qwen3.5-397b-a17b",
                repo / "4060ti" / "deepseek-v4.1-flash",
                repo / "4060ti" / "GLM-5.2",
            ),
            "4060ti/results",
        ),
        (
            (
                repo / "CPU" / "results",
                repo / "14700KF" / "SCOPE",
            ),
            "CPU/results",
        ),
    )
    for candidates, relative in artifact_sources:
        source = next((candidate for candidate in candidates if candidate.is_dir()), None)
        if source is not None:
            copy_tree(source, package / relative)

    for relative in ("EXPERIMENT_AUDIT.md", "EXPERIMENT_AUDIT.json"):
        source = repo / relative
        if source.exists():
            copy_file(source, package / "docs" / relative)

    config = sanitized_config(repo / "conf.yaml")
    config_text = yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
    (package / "conf.yaml").write_text(config_text, encoding="utf-8")
    (package / "conf.example.yaml").write_text(config_text, encoding="utf-8")
    copy_file(repo / "requirement.txt", output / "requirements.txt")
    (output / "README.md").write_text(release_readme(), encoding="utf-8")
    (output / ".gitignore").write_text(gitignore_text(), encoding="utf-8")

    configure_release_benchmark(package)
    sanitize_exported_paths(package, repo)
    findings = scan_release(output)
    if findings:
        raise RuntimeError("Release scan failed:\n" + "\n".join(findings))
    write_manifest(output)
    print(json.dumps({
        "output": str(output),
        "file_count": len([path for path in output.rglob("*") if path.is_file()]),
        "secret_scan": "pass",
    }, indent=2))


if __name__ == "__main__":
    main()
