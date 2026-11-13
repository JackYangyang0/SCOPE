
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from . import data
from .util import ensure_dir, read_json, sha256_bytes, sha256_value, write_json

REFERENCE_VERSION = "scope-exp-reference-v1"

COMPENSATED_FULL_LIMIT = 64 ** 3
POINT_CHECK_COUNT = 64


def _as64(array: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(array, dtype=np.float64)


def cpu_blas_fp64(A: np.ndarray, B: np.ndarray, alpha: float = 1.0,
                  beta: float = 0.0, C0: np.ndarray | None = None,
                  ) -> tuple[np.ndarray, np.ndarray]:
    A64 = _as64(A)
    B64 = _as64(B)
    AB = A64 @ B64
    C = alpha * AB
    if beta != 0.0:
        if C0 is None:
            raise ValueError("beta != 0 requires C0")
        C = C + beta * _as64(C0)
    return C, AB


def compensated_dot(a: np.ndarray, b: np.ndarray) -> float:
    products = (float(x) * float(y) for x, y in zip(a.tolist(), b.tolist()))
    return math.fsum(products)


def compensated_fp64(A: np.ndarray, B: np.ndarray, alpha: float = 1.0,
                     beta: float = 0.0, C0: np.ndarray | None = None) -> np.ndarray:
    A64, B64 = _as64(A), _as64(B)
    M, K = A64.shape
    N = B64.shape[1]
    if M * N * K > COMPENSATED_FULL_LIMIT:
        raise ValueError("compensated_fp64 refused a large shape; use point checks")
    out = np.empty((M, N), dtype=np.float64)
    for m in range(M):
        row = A64[m]
        for n in range(N):
            column = B64[:, n]
            out[m, n] = alpha * compensated_dot(row, column)
    if beta != 0.0:
        if C0 is None:
            raise ValueError("beta != 0 requires C0")
        out = out + beta * _as64(C0)
    return out


def analytic_output(case: dict[str, Any]) -> np.ndarray | None:
    family = case["family"]
    shape = case["shape"]
    M, N, K = shape["M"], shape["N"], shape["K"]
    alpha, beta = float(case["alpha"]), float(case["beta"])
    A, B, C0 = case["A"], case["B"], case["C0"]

    if family == "zero":
        return np.zeros((M, N), dtype=np.float64) + (beta * _as64(C0) if beta else 0.0)

    if family == "left_identity":
        base = np.zeros((M, N), dtype=np.float64)
        rows = min(M, K)
        base[:rows] = _as64(B)[:rows]
        return alpha * base + (beta * _as64(C0) if beta else 0.0)

    if family == "right_identity":
        base = np.zeros((M, N), dtype=np.float64)
        cols = min(K, N)
        base[:, :cols] = _as64(A)[:, :cols]
        return alpha * base + (beta * _as64(C0) if beta else 0.0)

    if family == "impulse":
        base = np.zeros((M, N), dtype=np.float64)
        if M > 0 and N > 0:
            base[0, 0] = 2.0 * float(A[0, K // 2]) if K else 0.0
        return alpha * base + (beta * _as64(C0) if beta else 0.0)

    if family == "cancellation":
        base = np.zeros((M, N), dtype=np.float64)
        pairs = K // 2
        base[0, :] = float(pairs) * (2.0 ** -20)
        return alpha * base + (beta * _as64(C0) if beta else 0.0)

    return None


def point_check_indices(M: int, N: int, count: int = POINT_CHECK_COUNT,
                        seed: int = 12345) -> list[tuple[int, int]]:
    positions = {(0, 0), (0, N - 1), (M - 1, 0), (M - 1, N - 1),
                 (M // 2, N // 2)}
    rng = np.random.Generator(np.random.MT19937(seed))
    while len(positions) < min(count, M * N):
        positions.add((int(rng.integers(0, M)), int(rng.integers(0, N))))
    return sorted((m, n) for m, n in positions)


def point_check(A: np.ndarray, B: np.ndarray, positions: Iterable[tuple[int, int]],
                alpha: float = 1.0, beta: float = 0.0,
                C0: np.ndarray | None = None) -> dict[str, Any]:
    A64, B64 = _as64(A), _as64(B)
    values: list[float] = []
    for m, n in positions:
        value = alpha * compensated_dot(A64[m], B64[:, n])
        if beta != 0.0:
            if C0 is None:
                raise ValueError("beta != 0 requires C0")
            value += beta * float(C0[m, n])
        values.append(value)
    return {"positions": [list(p) for p in positions], "values": values}


@dataclass
class Reference:
    C: np.ndarray
    AB: np.ndarray | None
    kind: str
    sha256: str
    meta: dict[str, Any]

    def as_record(self) -> dict[str, Any]:
        return {"kind": self.kind, "sha256": self.sha256, **self.meta}


def build_reference(case: dict[str, Any], *, kind: str = "cpu_blas_fp64",
                    with_ab: bool = False) -> Reference:
    A, B, C0 = case["A"], case["B"], case["C0"]
    alpha, beta = float(case["alpha"]), float(case["beta"])
    shape = case["shape"]
    M, N, K = shape["M"], shape["N"], shape["K"]

    analytic = analytic_output(case)
    if kind == "analytic":
        if analytic is None:
            raise ValueError(f"family {case['family']} has no analytic reference")
        C = analytic
        AB = None
    elif kind == "compensated_fp64":
        C = compensated_fp64(A, B, alpha, beta, C0)
        AB = _as64(A) @ _as64(B) if with_ab else None
    elif kind == "cpu_blas_fp64":
        C, AB_full = cpu_blas_fp64(A, B, alpha, beta, C0)
        AB = AB_full if with_ab else None
    else:
        raise ValueError(f"unknown reference kind: {kind}")

    C = np.ascontiguousarray(C, dtype=np.float64)

    checks: dict[str, Any] = {}
    if analytic is not None and kind != "analytic":
        checks["analytic_agreement"] = _agreement(C, analytic)
    positions = point_check_indices(M, N)
    points = point_check(A, B, positions, alpha, beta, C0)
    sampled = np.array([C[m, n] for m, n in positions], dtype=np.float64)
    point_values = np.array(points["values"], dtype=np.float64)
    checks["compensated_point_check"] = {
        "count": len(positions),
        "max_abs_difference": float(np.max(np.abs(sampled - point_values))) if len(sampled) else 0.0,
        "mismatch_positions": [positions[i] for i in
                               np.flatnonzero(~np.isclose(sampled, point_values,
                                                          rtol=1e-12, atol=1e-12))][:8],
    }
    checks["all_finite"] = bool(np.all(np.isfinite(C)))

    meta = {
        "reference_version": REFERENCE_VERSION,
        "kind": kind,
        "shape": shape,
        "alpha": alpha,
        "beta": beta,
        "input_hashes": case["hashes"],
        "generator_version": case["generator_version"],
        "arithmetic": "float64",
        "summation": ("numpy.matmul (BLAS dgemm)" if kind == "cpu_blas_fp64"
                      else "math.fsum compensated" if kind == "compensated_fp64"
                      else "closed form"),
        "validity": checks,
        "valid": bool(checks["all_finite"] and _checks_pass(checks)),
    }
    digest = sha256_bytes(C.tobytes(order="C"))
    return Reference(C=C, AB=AB, kind=kind, sha256=digest, meta=meta)


def _agreement(C: np.ndarray, other: np.ndarray) -> dict[str, Any]:
    difference = np.abs(C - other)
    scale = max(float(np.max(np.abs(other))), 1e-30)
    return {"max_abs_difference": float(np.max(difference)) if difference.size else 0.0,
            "max_scaled_difference": float(np.max(difference) / scale) if difference.size else 0.0}


def _checks_pass(checks: dict[str, Any]) -> bool:
    analytic = checks.get("analytic_agreement")
    if analytic and analytic["max_scaled_difference"] > 1e-9:
        return False
    points = checks["compensated_point_check"]
    if points["mismatch_positions"]:
        return False
    return True


class ReferenceCache:

    STORE_BYTES_LIMIT = 32 * 1024 * 1024

    def __init__(self, root: Path, *, kind: str = "cpu_blas_fp64",
                 store_arrays: bool = True) -> None:
        self.root = ensure_dir(Path(root))
        self.kind = kind
        self.store_arrays = store_arrays
        self._memory: dict[str, Reference] = {}

    def key(self, case: dict[str, Any]) -> str:
        return sha256_value({
            "reference_version": REFERENCE_VERSION,
            "kind": self.kind,
            "input_hashes": case["hashes"],
            "shape": case["shape"],
            "alpha": float(case["alpha"]),
            "beta": float(case["beta"]),
        })

    def get(self, case: dict[str, Any], *, with_ab: bool = False) -> Reference:
        key = self.key(case)
        if key in self._memory and (not with_ab or self._memory[key].AB is not None):
            return self._memory[key]
        directory = self.root / key
        meta_path = directory / "reference.meta.json"
        array_path = directory / "reference.f64.npy"
        reference: Reference | None = None
        if array_path.exists() and meta_path.exists():
            C = np.load(array_path)
            meta = read_json(meta_path)
            if meta.get("input_hashes") == case["hashes"] and meta.get("sha256") == sha256_bytes(C.tobytes(order="C")):
                reference = Reference(C=C, AB=None, kind=self.kind,
                                      sha256=meta["sha256"], meta=meta)
        if reference is None:
            reference = build_reference(case, kind=self.kind, with_ab=with_ab)
            ensure_dir(directory)
            meta = dict(reference.meta, sha256=reference.sha256)
            write_json(meta_path, meta)
            if self.store_arrays and reference.C.nbytes <= self.STORE_BYTES_LIMIT:
                np.save(array_path, reference.C)
            reference.meta = meta
        if with_ab and reference.AB is None:
            reference.AB = _as64(case["A"]) @ _as64(case["B"])
        self._memory[key] = reference
        return reference


def cross_check_reference(case: dict[str, Any], primary: Reference,
                          *, sample: int = POINT_CHECK_COUNT) -> dict[str, Any]:
    positions = point_check_indices(case["shape"]["M"], case["shape"]["N"], sample)
    points = point_check(case["A"], case["B"], positions, float(case["alpha"]),
                         float(case["beta"]), case["C0"])
    sampled = np.array([primary.C[m, n] for m, n in positions], dtype=np.float64)
    expected = np.array(points["values"], dtype=np.float64)
    difference = np.abs(sampled - expected)
    tolerance = 1e-12 + 1e-12 * np.abs(expected)
    bad = np.flatnonzero(difference > tolerance)
    record: dict[str, Any] = {
        "method": "compensated_fp64_point_check",
        "positions": len(positions),
        "max_abs_difference": float(np.max(difference)) if difference.size else 0.0,
        "mismatch_positions": [list(positions[i]) for i in bad][:8],
        "status": "agree" if not len(bad) else "reference_invalid",
    }
    analytic = analytic_output(case)
    if analytic is not None:
        agreement = _agreement(primary.C, analytic)
        record["analytic"] = agreement
        if agreement["max_scaled_difference"] > 1e-9:
            record["status"] = "reference_invalid"
            record.setdefault("notes", []).append("closed-form and matmul disagree")
    return record


def dataset_manifest_hash(path: Path) -> str:
    return sha256_value(read_json(path))
