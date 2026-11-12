
from __future__ import annotations

import re

import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from . import HARNESS_VERSION
from .errors import InfraError
from .util import (ensure_dir, environment_record, read_json, run_command,
                   sha256_file, sha256_value, utc_now, which, write_json)

DRIVERS_DIR = Path(__file__).resolve().parent / "drivers"
BUILD_VERSION = "scope-exp-build-v1"

CUDA_SOURCE_EXTENSIONS = (".cu", ".cuh", ".cpp", ".cc", ".cxx", ".h", ".hpp")
CPU_SOURCE_EXTENSIONS = (".c", ".h")


@dataclass
class Toolchain:
    backend: str
    compiler: str | None
    compiler_kind: str
    version_text: str = ""
    vcvars64: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def available(self) -> bool:
        return bool(self.compiler)

    def as_record(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "compiler": self.compiler,
            "compiler_kind": self.compiler_kind,
            "version_text": self.version_text,
            "vcvars64": self.vcvars64,
            "extra": self.extra,
            "available": self.available,
        }


def discover_vcvars64(hint: str | Path | None = None) -> str | None:
    if hint:
        candidate = Path(hint)
        if candidate.exists():
            return str(candidate)
    vswhere = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / \
        "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    if vswhere.exists():
        result = run_command([str(vswhere), "-latest", "-products", "*",
                              "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                              "-property", "installationPath"], timeout=60)
        install = (result["stdout"] or "").strip().splitlines()
        if install:
            candidate = Path(install[0].strip()) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
            if candidate.exists():
                return str(candidate)
    roots = [Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Microsoft Visual Studio",
             Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Microsoft Visual Studio"]
    for root in roots:
        if not root.exists():
            continue
        for candidate in sorted(root.glob("**/VC/Auxiliary/Build/vcvars64.bat"), reverse=True):
            return str(candidate)
    return None


def discover_toolchain(backend: str, *, vcvars64: str | Path | None = None) -> Toolchain:
    if backend == "cuda":
        nvcc = which("nvcc")
        if not nvcc:
            cuda_home = os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME")
            if cuda_home:
                candidate = Path(cuda_home) / "bin" / ("nvcc.exe" if os.name == "nt" else "nvcc")
                if candidate.exists():
                    nvcc = str(candidate)
        if not nvcc:
            return Toolchain("cuda", None, "none", "", None,
                             {"reason": "nvcc not found in PATH or CUDA_PATH"})
        version = run_command([nvcc, "--version"], timeout=120)
        toolchain = Toolchain("cuda", nvcc, "nvcc", _first_version_line(version["stdout"]))
        if os.name == "nt":
            toolchain.vcvars64 = discover_vcvars64(vcvars64)
            if not toolchain.vcvars64:
                toolchain.extra["warning"] = "vcvars64.bat not found; nvcc host compile may fail"
        return toolchain

    if backend == "cpu":
        if os.name == "nt":
            vcvars = discover_vcvars64(vcvars64)
            cl = which("cl")
            if cl or vcvars:
                version = ""
                if vcvars:
                    version = _cl_version_via_vcvars(vcvars)
                return Toolchain("cpu", cl or "cl", "cl", version, vcvars)
        for name in ("gcc", "cc", "clang"):
            compiler = which(name)
            if compiler:
                version = run_command([compiler, "--version"], timeout=60)
                return Toolchain("cpu", compiler, "gcc",
                                 _first_version_line(version["stdout"]))
        return Toolchain("cpu", None, "none", "", None,
                         {"reason": "no C compiler (cl/gcc/cc/clang) found"})

    raise ValueError(f"unknown backend: {backend}")


def _first_version_line(text: str) -> str:
    for line in (text or "").splitlines():
        if line.strip():
            return line.strip()
    return ""


def _cl_version_via_vcvars(vcvars: str) -> str:
    batch = (
        "@echo off\r\n"
        "chcp 65001 >nul\r\n"
        f'call "{vcvars}" >nul 2>nul\r\n'
        "if errorlevel 1 exit /b 1\r\n"
        "cl 2>&1 | findstr /C:\"Version\"\r\n"
        "exit /b 0\r\n"
    )
    with tempfile.TemporaryDirectory(prefix="scope-clver-") as directory:
        script = Path(directory) / "clver.bat"
        script.write_text(batch, encoding="utf-8")
        result = run_command(["cmd", "/c", str(script)], timeout=180)
    return _first_version_line(result["stdout"])


def cuda_device_compute_capability(device: int = 0) -> tuple[int, int] | None:
    smi = which("nvidia-smi")
    if not smi:
        return None
    result = run_command([smi, "--query-gpu=compute_cap", "--format=csv,noheader",
                          "-i", str(device)], timeout=60)
    text = (result["stdout"] or "").strip().splitlines()
    if not text:
        return None
    try:
        major, minor = text[0].strip().split(".")
        return int(major), int(minor)
    except ValueError:
        return None


def resolve_cuda_arch(nvcc: str, device: int = 0, requested: str | None = None) -> dict[str, Any]:
    supported: list[str] = []
    listing = run_command([nvcc, "--list-gpu-arch"], timeout=120)
    for line in (listing["stdout"] or "").splitlines():
        line = line.strip()
        if line.startswith("compute_"):
            supported.append(line.replace("compute_", "sm_"))
    device_capability = cuda_device_compute_capability(device)
    if requested:
        chosen = requested
        source = "explicit"
    elif device_capability:
        chosen = f"sm_{device_capability[0]}{device_capability[1]}"
        source = "device"
    elif supported:
        chosen = sorted(supported, key=lambda item: tuple(int(x) for x in item[3:]))[-1]
        source = "highest_supported_no_device"
    else:
        chosen = "sm_75"
        source = "fallback_default"
    return {
        "arch": chosen,
        "arch_source": source,
        "device_compute_capability": (
            f"{device_capability[0]}.{device_capability[1]}" if device_capability else None),
        "supported_arches": supported,
        "listing_stderr": (listing["stderr"] or "").strip()[:500],
    }


def kernel_sources(kernel_dir: Path, backend: str) -> dict[str, Any]:
    kernel_dir = Path(kernel_dir)
    if not kernel_dir.is_dir():
        raise InfraError(f"kernel directory not found: {kernel_dir}", stage="build")
    if backend == "cuda":
        required = ("cuda_kernel.cuh",)
        sources = [p for p in sorted(kernel_dir.iterdir())
                   if p.is_file() and p.suffix in (".cu", ".cuh")]
    else:
        required = ("cpu_kernel.c", "kernel.h")
        sources = [p for p in sorted(kernel_dir.iterdir())
                   if p.is_file() and p.suffix in (".c", ".h")]
    missing = [name for name in required if not (kernel_dir / name).exists()]
    if missing:
        raise InfraError(f"kernel directory {kernel_dir} is missing {missing} "
                         "required by the independent evaluator", stage="build")
    if not sources:
        raise InfraError(f"kernel directory {kernel_dir} has no compilable source", stage="build")
    entry = "cuda_kernel.cuh" if backend == "cuda" else "cpu_kernel.c"
    return {
        "kernel_dir": str(kernel_dir),
        "entry": entry,
        "required": list(required),
        "files": {p.name: sha256_file(p) for p in sources},
        "digest": sha256_value({p.name: sha256_file(p) for p in sources}),
        "sizes": {p.name: p.stat().st_size for p in sources},
    }


CPU_FLAGS = ["/nologo", "/O2", "/W3", "/MD"]
CPU_DEFINES = ["/DSCOPE_EXP_EVAL_DRIVER=1"]


def _build_batch(vcvars: str, arguments: Sequence[str]) -> str:
    lines = ["@echo off", "chcp 65001 >nul", f'call "{vcvars}" >nul 2>nul',
             "if errorlevel 1 exit /b 7"]
    lines.append(" ".join(_quote_for_batch(a) for a in arguments))
    lines.append("exit /b %errorlevel%")
    return "\r\n".join(lines) + "\r\n"


def _quote_for_batch(argument: str) -> str:
    argument = str(argument)
    if any(ch in argument for ch in ' ()[]&|<>^'):
        return f'"{argument}"'
    return argument


@dataclass
class BuildResult:
    ok: bool
    backend: str
    binary: Path | None
    record: dict[str, Any]

    def as_record(self) -> dict[str, Any]:
        return self.record


def _cpu_isa_flags(kernel_c: Path) -> tuple[list[str], dict[str, Any]]:
    try:
        source = kernel_c.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return [], {"isa": None, "reason": "kernel source not readable"}

    flags: list[str] = []
    if re.search(r"\b__m512\b|\b_mm512_", source):
        flags.append("-mavx512f")
        isa = "avx512"
        if "_fmadd_" in source:
            flags.append("-mfma")
            isa = "avx512_fma"
    elif re.search(r"\b__m256\b|\b_mm256_", source):
        flags.append("-mavx2")
        isa = "avx2"
        if "_fmadd_" in source:
            flags.append("-mfma")
            isa = "avx2_fma"
    else:
        isa = None
    if "#pragma omp" in source:
        flags.append("-fopenmp")
    return flags, {"isa": isa, "openmp": "#pragma omp" in source,
                   "inferred_from": kernel_c.name}


def build_eval_driver(kernel_dir: Path, out_dir: Path, *, backend: str = "cpu",
                      toolchain: Toolchain | None = None, device: int = 0,
                      arch: str | None = None, timeout: float = 900.0,
                      extra_cflags: Sequence[str] = (),
                      extra_ldflags: Sequence[str] = (),
                      driver_source: str | None = None) -> BuildResult:
    out_dir = ensure_dir(Path(out_dir).resolve())
    kernel_dir = Path(kernel_dir).resolve()
    toolchain = toolchain or discover_toolchain(backend)
    sources = kernel_sources(kernel_dir, backend)
    started_at = utc_now()
    monotonic = time.monotonic()
    binary_name = "scope_eval_cpu.exe" if backend == "cpu" else "scope_eval_cuda"
    if os.name != "nt":
        binary_name = binary_name.replace(".exe", "") if backend == "cpu" else binary_name
    binary = out_dir / binary_name

    record: dict[str, Any] = {
        "build_version": BUILD_VERSION,
        "harness_version": HARNESS_VERSION,
        "backend": backend,
        "started_at": started_at,
        "kernel": sources,
        "toolchain": toolchain.as_record(),
        "out_dir": str(out_dir),
        "binary": str(binary),
        "sources_hashed": {},
        "argv": [],
        "compile_flags": {},
        "object_hashes": {},
        "binary_sha256": None,
        "ok": False,
        "error": None,
        "env": {"OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES")},
    }

    if not toolchain.available:
        record["error"] = f"no {backend} toolchain available: {toolchain.extra}"
        record["duration_seconds"] = time.monotonic() - monotonic
        write_json(out_dir / "build.json", record)
        return BuildResult(False, backend, None, record)

    if backend == "cpu":
        driver = Path(driver_source or DRIVERS_DIR / "eval_driver.c")
        kernel_c = Path(kernel_dir) / "cpu_kernel.c"
        record["sources_hashed"] = {
            "driver": sha256_file(driver),
            "cpu_kernel.c": sha256_file(kernel_c),
            "kernel.h": sha256_file(Path(kernel_dir) / "kernel.h"),
        }
        if toolchain.compiler_kind == "cl":
            object_dir = ensure_dir(out_dir / "obj")
            arguments = [str(toolchain.compiler), *CPU_FLAGS, *CPU_DEFINES,
                         *extra_cflags,
                         f"/I{Path(kernel_dir)}",
                         str(driver), str(kernel_c),
                         f"/Fe{binary}", *extra_ldflags]
            if not toolchain.vcvars64:
                record["error"] = "cl found but vcvars64.bat not located"
                record["duration_seconds"] = time.monotonic() - monotonic
                write_json(out_dir / "build.json", record)
                return BuildResult(False, backend, None, record)
            batch = ensure_dir(out_dir / "build_scripts") / "compile_cpu.bat"
            batch.write_text(_build_batch(toolchain.vcvars64, arguments), encoding="utf-8")
            command = ["cmd", "/c", str(batch)]
            record["argv"] = arguments
            record["cwd"] = str(object_dir)
            record["compile_flags"] = {"cl": CPU_FLAGS, "defines": CPU_DEFINES,
                                       "include": str(Path(kernel_dir)),
                                       "vcvars64": toolchain.vcvars64}
            work_dir = object_dir
        else:
            isa_flags, isa_record = _cpu_isa_flags(kernel_c)
            arguments = [str(toolchain.compiler), "-O2", "-std=c11", "-ffp-contract=off",
                         *isa_flags, *extra_cflags, f"-I{Path(kernel_dir)}",
                         str(driver), str(kernel_c), "-lm",
                         "-o", str(binary), *extra_ldflags]
            command = arguments
            record["argv"] = arguments
            record["cwd"] = str(out_dir)
            record["compile_flags"] = {"cflags": ["-O2", "-std=c11", "-ffp-contract=off",
                                                  *isa_flags],
                                       "include": str(Path(kernel_dir)),
                                       "ffp_contract": "off",
                                       "isa_inferred": isa_record,
                                       "note": "no fast-math; accumulation order is the kernel's; "
                                               "ISA/OpenMP flags inferred from cpu_kernel.c"}
            work_dir = out_dir
    elif backend == "cuda":
        driver = Path(driver_source or DRIVERS_DIR / "eval_driver.cu")
        arch_info = resolve_cuda_arch(toolchain.compiler, device=device, requested=arch)
        record["arch"] = arch_info
        record["sources_hashed"] = {
            "driver": sha256_file(driver),
            "cuda_kernel.cuh": sha256_file(Path(kernel_dir) / "cuda_kernel.cuh"),
        }
        flags = ["-O3", "-std=c++14", f"-arch={arch_info['arch']}",
                 "--expt-relaxed-constexpr", "-lineinfo"]
        arguments = [str(toolchain.compiler), *flags, *extra_cflags,
                     f"-I{Path(kernel_dir)}", str(driver),
                     "-o", str(binary), *extra_ldflags]
        if os.name == "nt" and toolchain.vcvars64:
            batch = ensure_dir(out_dir / "build_scripts") / "compile_cuda.bat"
            batch.write_text(_build_batch(toolchain.vcvars64, arguments), encoding="utf-8")
            command = ["cmd", "/c", str(batch)]
        else:
            command = arguments
        record["argv"] = arguments
        record["cwd"] = str(out_dir)
        record["compile_flags"] = {"nvcc": flags, "include": str(Path(kernel_dir))}
        work_dir = out_dir
    else:
        raise ValueError(f"unknown backend: {backend}")

    result = run_command(command, cwd=work_dir, timeout=timeout,
                         env=_build_env(toolchain))
    record["compile_stdout"] = (result["stdout"] or "")[-8000:]
    record["compile_stderr"] = (result["stderr"] or "")[-8000:]
    record["compile_returncode"] = result["returncode"]
    record["compile_timed_out"] = result["timed_out"]
    record["duration_seconds"] = time.monotonic() - monotonic

    ok = binary.exists() and result["returncode"] == 0
    record["ok"] = ok
    if binary.exists():
        record["binary_sha256"] = sha256_file(binary)
        record["binary_bytes"] = binary.stat().st_size
    for candidate in sorted(out_dir.rglob("*.obj")) + sorted(out_dir.rglob("*.o")):
        record["object_hashes"][candidate.name] = sha256_file(candidate)
    if not record["ok"]:
        record["error"] = ("compile timed out" if result["timed_out"] else
                           f"compiler exited {result['returncode']}")
    write_json(out_dir / "build.json", record)
    return BuildResult(record["ok"], backend, binary if record["ok"] else None, record)


def _build_env(toolchain: Toolchain) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    if toolchain.compiler_kind == "nvcc":
        env.setdefault("NVCC_PREPEND_FLAGS", "")
    return env


def compile_reference_dgemm(out_dir: Path, *, device: int = 0, arch: str | None = None,
                            timeout: float = 900.0) -> BuildResult:
    out_dir = ensure_dir(Path(out_dir))
    toolchain = discover_toolchain("cuda")
    binary = out_dir / ("ref_dgemm.exe" if os.name == "nt" else "ref_dgemm")
    record: dict[str, Any] = {
        "build_version": BUILD_VERSION,
        "backend": "cuda",
        "driver": "ref_dgemm.cu",
        "toolchain": toolchain.as_record(),
        "binary": str(binary),
        "ok": False,
        "error": None,
    }
    if not toolchain.available:
        record["error"] = "nvcc not available; cuBLAS cross-check recorded as not_run"
        write_json(out_dir / "build_ref.json", record)
        return BuildResult(False, "cuda", None, record)
    arch_info = resolve_cuda_arch(toolchain.compiler, device=device, requested=arch)
    record["arch"] = arch_info
    flags = ["-O3", "-std=c++14", f"-arch={arch_info['arch']}"]
    arguments = [str(toolchain.compiler), *flags, str(DRIVERS_DIR / "ref_dgemm.cu"),
                 "-o", str(binary), "-lcublas"]
    if os.name == "nt" and toolchain.vcvars64:
        batch = ensure_dir(out_dir / "build_scripts") / "compile_ref.bat"
        batch.write_text(_build_batch(toolchain.vcvars64, arguments), encoding="utf-8")
        command = ["cmd", "/c", str(batch)]
    else:
        command = arguments
    record["argv"] = arguments
    result = run_command(command, cwd=out_dir, timeout=timeout, env=_build_env(toolchain))
    record["compile_returncode"] = result["returncode"]
    record["compile_stdout"] = (result["stdout"] or "")[-4000:]
    record["compile_stderr"] = (result["stderr"] or "")[-4000:]
    record["ok"] = binary.exists() and result["returncode"] == 0
    if binary.exists():
        record["binary_sha256"] = sha256_file(binary)
    if not record["ok"]:
        record["error"] = "cuBLAS reference driver did not build"
    write_json(out_dir / "build_ref.json", record)
    return BuildResult(record["ok"], "cuda", binary if record["ok"] else None, record)


def toolchain_summary(backend: str) -> dict[str, Any]:
    toolchain = discover_toolchain(backend)
    summary = toolchain.as_record()
    if backend == "cuda" and toolchain.available:
        summary["arch"] = resolve_cuda_arch(toolchain.compiler)
    return summary


def build_environment_record() -> dict[str, Any]:
    record = environment_record()
    record["build_version"] = BUILD_VERSION
    record["drivers"] = {p.name: sha256_file(p) for p in sorted(DRIVERS_DIR.glob("*"))
                         if p.is_file()}
    return record


def load_build_record(path: Path) -> dict[str, Any]:
    return read_json(path)
