from __future__ import annotations

import json
import re
import struct
from pathlib import Path
from typing import Any

import numpy as np

from .util import ensure_dir, sha256_bytes

CASE_MAGIC_TEXT = b"SCOPECASE"
CASE_MAGIC = CASE_MAGIC_TEXT[:8]
CASE_VERSION = 1
CASE_HEADER = struct.Struct("<8sIiiiffI")
FLAG_HAS_C0 = 1

EVAL_LINE = re.compile(r"SCOPE_EVAL\s+(\{.*\})\s*$")
REF_LINE = re.compile(r"SCOPE_REF\s+(\{.*\})\s*$")


def case_file_name(case_key: str) -> str:
    from .util import safe_id, sha256_text
    return f"{safe_id(case_key, 'case')[:60]}.{sha256_text(case_key)[:12]}.case"


def write_case_file(path: Path, case: dict[str, Any]) -> dict[str, Any]:
    path = Path(path)
    ensure_dir(path.parent)
    shape = case["shape"]
    M, N, K = int(shape["M"]), int(shape["N"]), int(shape["K"])
    A = np.ascontiguousarray(case["A"], dtype="<f4")
    B = np.ascontiguousarray(case["B"], dtype="<f4")
    C0 = np.ascontiguousarray(case["C0"], dtype="<f4")
    if A.size != M * K or B.size != K * N or C0.size != M * N:
        raise ValueError("case payload does not match its declared shape")
    header = CASE_HEADER.pack(CASE_MAGIC, CASE_VERSION, M, N, K,
                              float(case["alpha"]), float(case["beta"]),
                              FLAG_HAS_C0)
    payload = b"".join((header, A.tobytes(), B.tobytes(), C0.tobytes()))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
    temporary.replace(path)
    return {
        "path": str(path),
        "bytes": len(payload),
        "sha256": sha256_bytes(payload),
        "header_version": CASE_VERSION,
        "shape": {"M": M, "N": N, "K": K},
        "alpha": float(case["alpha"]),
        "beta": float(case["beta"]),
        "labels": {"case_id": case["case_id"], "c0_variant": case["c0_variant"]},
    }


def read_output_file(path: Path, M: int, N: int, dtype=np.float32) -> np.ndarray:
    path = Path(path)
    data = path.read_bytes()
    itemsize = np.dtype(dtype).itemsize
    expected = int(M) * int(N) * itemsize
    if len(data) != expected:
        raise ValueError(f"{path}: expected {expected} bytes for {M}x{N}, got {len(data)}")
    return np.frombuffer(data, dtype=dtype).reshape(int(M), int(N))


def parse_driver_stdout(text: str) -> dict[str, Any]:
    for line in reversed(text.splitlines()):
        match = EVAL_LINE.search(line.strip())
        if match:
            try:
                return {"found": True, "record": json.loads(match.group(1))}
            except json.JSONDecodeError as exc:
                return {"found": False, "error": f"unparsable SCOPE_EVAL line: {exc}"}
    return {"found": False, "error": "no SCOPE_EVAL line in driver output"}


def parse_reference_stdout(text: str) -> dict[str, Any]:
    for line in reversed(text.splitlines()):
        match = REF_LINE.search(line.strip())
        if match:
            try:
                return {"found": True, "record": json.loads(match.group(1))}
            except json.JSONDecodeError as exc:
                return {"found": False, "error": f"unparsable SCOPE_REF line: {exc}"}
    return {"found": False, "error": "no SCOPE_REF line in driver output"}


def read_case_file(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    header = CASE_HEADER.size
    magic, version, M, N, K, alpha, beta, flags = CASE_HEADER.unpack_from(raw, 0)
    if magic != CASE_MAGIC:
        raise ValueError(f"{path}: bad magic {magic!r}")
    if version != CASE_VERSION:
        raise ValueError(f"{path}: unsupported case version {version}")
    offset = header
    A = np.frombuffer(raw, dtype="<f4", count=M * K, offset=offset).reshape(M, K).copy()
    offset += M * K * 4
    B = np.frombuffer(raw, dtype="<f4", count=K * N, offset=offset).reshape(K, N).copy()
    offset += K * N * 4
    C0 = np.frombuffer(raw, dtype="<f4", count=M * N, offset=offset).reshape(M, N).copy()
    return {
        "shape": {"M": M, "N": N, "K": K},
        "alpha": float(alpha), "beta": float(beta), "flags": int(flags),
        "A": A, "B": B, "C0": C0,
        "sha256": sha256_bytes(raw),
    }
