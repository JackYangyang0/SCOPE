from __future__ import annotations

import re
from pathlib import Path
from typing import Any


REQUIRED_FILES = ("main.c", "kernel.h", "cpu_kernel.c")
PATCH_BEGIN = "SCOPE_CPU_PATCH_KERNEL_BEGIN"
PATCH_END = "SCOPE_CPU_PATCH_KERNEL_END"
OPTIMIZED_BASELINE_MARKERS = {
    "immintrin.h": "initial CPU skeleton embeds ISA intrinsics",
    "__m256": "initial CPU skeleton embeds an AVX micro-kernel",
    "__m512": "initial CPU skeleton embeds an AVX-512 micro-kernel",
    "#pragma omp": "initial CPU skeleton embeds a parallelization strategy",
    "L2_BLOCK_": "initial CPU skeleton embeds a cache-blocking strategy",
    "PackA": "initial CPU skeleton embeds a packing strategy",
    "PackB": "initial CPU skeleton embeds a packing strategy",
}


def audit_initial_cpu_skeleton(source_dir: Path) -> dict[str, Any]:
    source_dir = Path(source_dir)
    defects: list[dict[str, str]] = []
    contents: dict[str, str] = {}
    for relative_path in REQUIRED_FILES:
        path = source_dir / relative_path
        if not path.is_file():
            defects.append({"check": "required_file", "file": relative_path, "message": "required file is missing"})
            continue
        contents[relative_path] = path.read_text(encoding="utf-8")

    kernel = contents.get("cpu_kernel.c", "")
    header = contents.get("kernel.h", "")
    harness = contents.get("main.c", "")
    if kernel.count(PATCH_BEGIN) != 1 or kernel.count(PATCH_END) != 1:
        defects.append({
            "check": "patch_anchor",
            "file": "cpu_kernel.c",
            "message": "CPU kernel must contain exactly one begin/end patch anchor pair",
        })
    for marker, message in OPTIMIZED_BASELINE_MARKERS.items():
        if marker in kernel:
            defects.append({"check": "neutral_baseline", "file": "cpu_kernel.c", "message": message})
    if not re.search(r"\bvoid\s+cpu_gemm\s*\(", kernel):
        defects.append({"check": "kernel_definition", "file": "cpu_kernel.c", "message": "cpu_gemm definition is missing"})
    if not re.search(r"\bvoid\s+cpu_gemm\s*\(", header):
        defects.append({"check": "public_declaration", "file": "kernel.h", "message": "cpu_gemm declaration is missing"})
    if re.search(r"\bvoid\s+cpu_gemm\s*\([^;{]*\)\s*\{", harness, flags=re.DOTALL):
        defects.append({"check": "fixed_harness", "file": "main.c", "message": "main.c must not define cpu_gemm"})
    if "cpu_gemm(" not in harness:
        defects.append({"check": "harness_call", "file": "main.c", "message": "main.c does not call cpu_gemm"})
    if any(token in "\n".join(contents.values()) for token in ("cuda", "__global__", "threadIdx", "blockIdx")):
        defects.append({"check": "backend_purity", "file": "*", "message": "CPU skeleton contains CUDA-only tokens"})

    return {
        "status": "pass" if not defects else "fail",
        "accepted": not defects,
        "source_dir": str(source_dir),
        "checked_files": list(REQUIRED_FILES),
        "defect_count": len(defects),
        "defects": defects,
    }
