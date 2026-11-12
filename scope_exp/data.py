
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

from .util import sha256_value, write_json


GENERATOR_VERSION = "scope-exp-data-v3"

SPLITS = ("development", "generation", "postselection_test")

SEED_BASE = {
    "development": 100_000,
    "generation": 200_000,
    "postselection_test": 300_000,
}
RANDOM_INDEX_SPAN = 1_000
MAX_RANDOM_INDEX = 25

STRUCTURAL_SEEDS = {
    "cancellation": 901,
    "left_identity": 902,
    "right_identity": 903,
    "impulse": 904,
}

C0_VARIANTS = ("pattern", "scalar")
C0_SCALAR = np.float32(-123.25)
C0_PATTERN_MODULUS = 7
C0_PATTERN_OFFSET = 3

MAIN_COEFFICIENTS = {"name": "unit", "alpha": 1.0, "beta": 0.0}
COEFFICIENT_SETS = (
    {"name": "alpha1_beta1", "alpha": 1.0, "beta": 1.0},
    {"name": "alpha_m075_beta05", "alpha": -0.75, "beta": 0.5},
)

_SHIFT = 8
_SCALE = 2.0 ** -24


def _random_specs() -> list[tuple[str, str, tuple]]:
    specs: list[tuple[str, str, tuple]] = []
    index = 0
    for i in range(10):
        specs.append((f"uniform_small.s{i:02d}", "uniform", (index, -1.0, 1.0)))
        index += 1
    for i in range(10):
        specs.append((f"uniform_wide.s{i:02d}", "uniform", (index, -8.0, 8.0)))
        index += 1
    for i in range(3):
        specs.append((f"normal.s{i:02d}", "normal", (index, 0.0, 1.0)))
        index += 1
    for i in range(3):
        specs.append((f"sparse.s{i:02d}", "sparse", (index, 0.01)))
        index += 1
    return specs


def _structural_specs() -> list[tuple[str, str, None]]:
    return [
        ("cancellation", "cancellation", None),
        ("zero", "zero", None),
        ("left_identity", "left_identity", None),
        ("right_identity", "right_identity", None),
        ("impulse", "impulse", None),
        ("original_deterministic", "original_deterministic", None),
    ]


CONFIGS: dict[str, dict[str, Any]] = {}
for _cid, _fam, _p in _random_specs():
    CONFIGS[_cid] = {"family": _fam, "params": _p, "random": True, "compat_only": False}
for _cid, _fam, _p in _structural_specs():
    CONFIGS[_cid] = {"family": _fam, "params": _p, "random": False, "compat_only": False}
CONFIGS["original_deterministic"]["compat_only"] = True

assert len(CONFIGS) == 32, f"expected 32 configs, got {len(CONFIGS)}"

CONFIG_IDS = tuple(sorted(CONFIGS))

FAMILY_COUNTS = {
    "uniform_small": 10, "uniform_wide": 10, "normal": 3, "sparse": 3,
    "cancellation": 1, "zero": 1, "left_identity": 1, "right_identity": 1,
    "impulse": 1, "original_deterministic": 1,
}


def assert_split_disjoint() -> None:
    blocks: list[tuple[int, int, str]] = []
    for split, base in SEED_BASE.items():
        blocks.append((base, base + RANDOM_INDEX_SPAN * (MAX_RANDOM_INDEX + 1), split))
    for (a_start, a_end, a_name), (b_start, b_end, b_name) in itertools.combinations(blocks, 2):
        if a_start <= b_end and b_start <= a_end:
            raise AssertionError(f"seed blocks overlap: {a_name} vs {b_name}")
    for name, seed in STRUCTURAL_SEEDS.items():
        if seed >= min(SEED_BASE.values()):
            raise AssertionError(f"structural seed {name}={seed} is inside a random block")


assert_split_disjoint()


def _rng(seed: int | None) -> np.random.Generator:
    if seed is None:
        raise ValueError("_rng requires an explicit integer seed")
    return np.random.Generator(np.random.MT19937(int(seed)))


def _raw_bits(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.bit_generator.random_raw(n).astype(np.uint32, copy=False)


def _f32(x) -> np.ndarray:
    a = np.ascontiguousarray(x, dtype=np.float32)
    if not a.flags["C_CONTIGUOUS"]:
        raise AssertionError("layout contract violated")
    return a


def _uniform_signed(rng: np.random.Generator, n: int, low: float = -1.0,
                    high: float = 1.0) -> np.ndarray:
    raw = _raw_bits(rng, n)
    u = (raw >> _SHIFT).astype(np.float64) * _SCALE
    return _f32(low + (high - low) * u)


def make_c0(M: int, N: int, variant: str) -> np.ndarray:
    if variant not in C0_VARIANTS:
        raise KeyError(f"unknown c0 variant: {variant}")
    if variant == "scalar":
        return _f32(np.full((M, N), C0_SCALAR, dtype=np.float32))
    idx = np.arange(M * N, dtype=np.int64)
    return _f32((((idx % C0_PATTERN_MODULUS) - C0_PATTERN_OFFSET)
                 .astype(np.float32)).reshape(M, N))


def case_hashes(A: np.ndarray, B: np.ndarray, C0: np.ndarray) -> dict[str, str]:
    return {
        "A": hashlib.sha256(A.tobytes(order="C")).hexdigest(),
        "B": hashlib.sha256(B.tobytes(order="C")).hexdigest(),
        "C0": hashlib.sha256(C0.tobytes(order="C")).hexdigest(),
    }


def case_seed(config_id: str, split: str) -> int | None:
    spec = CONFIGS[config_id]
    if not spec["random"]:
        return STRUCTURAL_SEEDS.get(config_id)
    index = int(spec["params"][0])
    return SEED_BASE[split] + index * RANDOM_INDEX_SPAN


def _build_A_B(family: str, params: Any, M: int, N: int, K: int,
               seed: int | None) -> tuple[np.ndarray, np.ndarray]:

    if family == "uniform":
        _, low, high = params
        rng = _rng(seed)
        A = _uniform_signed(rng, M * K, low, high).reshape(M, K)
        B = _uniform_signed(rng, K * N, low, high).reshape(K, N)

    elif family == "normal":
        g = _rng(seed)
        A = _f32(g.standard_normal(M * K).reshape(M, K))
        B = _f32(g.standard_normal(K * N).reshape(K, N))

    elif family == "sparse":
        _, density = params
        rng = _rng(seed)

        def _sparse(shape: tuple[int, int]) -> np.ndarray:
            n = shape[0] * shape[1]
            mask = rng.random(n) < density
            vals = _uniform_signed(rng, n, -1.0, 1.0)
            return _f32(np.where(mask, vals, np.float32(0.0)).reshape(shape))

        A = _sparse((M, K))
        B = _sparse((K, N))

    elif family == "cancellation":
        A = np.zeros((M, K), dtype=np.float32)
        cols = K // 2
        A[0, 0:2 * cols:2] = np.float32(1.0)
        perturbation = np.float32(2.0 ** -20)
        A[0, 1:2 * cols:2] = np.float32(-1.0) + perturbation
        B = np.ones((K, N), dtype=np.float32)

    elif family == "zero":
        A = np.zeros((M, K), dtype=np.float32)
        B = np.zeros((K, N), dtype=np.float32)

    elif family in ("left_identity", "right_identity"):
        eye_A = _f32(np.eye(M, K, dtype=np.float32))
        eye_B = _f32(np.eye(K, N, dtype=np.float32))
        rng = _rng(seed)
        if family == "left_identity":
            A = eye_A
            B = _uniform_signed(rng, K * N).reshape(K, N)
        else:
            A = _uniform_signed(rng, M * K).reshape(M, K)
            B = eye_B

    elif family == "impulse":
        A = np.zeros((M, K), dtype=np.float32)
        B = np.zeros((K, N), dtype=np.float32)
        A[0, K // 2] = np.float32(1.0)
        B[K // 2, 0] = np.float32(2.0)

    elif family == "original_deterministic":
        ia = np.arange(M * K, dtype=np.int64)
        ib = np.arange(K * N, dtype=np.int64)
        A = _f32((ia // 13).reshape(M, K))
        B = _f32((ib % 13).reshape(K, N))

    else:
        raise KeyError(f"unknown family: {family}")

    return _f32(A), _f32(B)


def make_case(config_id: str, M: int, N: int, K: int, *, c0_variant: str = "pattern",
              split: str = "development", coefficients: dict[str, Any] | None = None,
              _ab_cache: dict | None = None) -> dict[str, Any]:
    if config_id not in CONFIGS:
        raise KeyError(f"unknown config_id: {config_id}")
    if c0_variant not in C0_VARIANTS:
        raise KeyError(f"unknown c0_variant: {c0_variant}")
    if split not in SPLITS:
        raise KeyError(f"unknown split: {split}")
    coeff = dict(coefficients or MAIN_COEFFICIENTS)
    for field in ("alpha", "beta", "name"):
        if field not in coeff:
            raise KeyError(f"coefficients missing {field!r}")

    seed = case_seed(config_id, split)
    cache_key = (config_id, M, N, K, split)
    if _ab_cache is not None and cache_key in _ab_cache:
        A, B = _ab_cache[cache_key]
    else:
        A, B = _build_A_B(CONFIGS[config_id]["family"], CONFIGS[config_id]["params"],
                          M, N, K, seed)
        if _ab_cache is not None:
            _ab_cache[cache_key] = (A, B)
    C0 = make_c0(M, N, c0_variant)

    return {
        "case_id": config_id,
        "family": CONFIGS[config_id]["family"],
        "compat_only": CONFIGS[config_id]["compat_only"],
        "shape": {"M": int(M), "N": int(N), "K": int(K)},
        "dtype": "float32",
        "layout": "row_major",
        "alpha": float(coeff["alpha"]),
        "beta": float(coeff["beta"]),
        "coefficient_set": coeff["name"],
        "split": split,
        "seed": seed,
        "c0_variant": c0_variant,
        "generator_version": GENERATOR_VERSION,
        "numpy_version": np.__version__,
        "A": A, "B": B, "C0": C0,
        "hashes": case_hashes(A, B, C0),
    }


def case_key(case: dict[str, Any]) -> str:
    s = case["shape"]
    return (f'{case["case_id"]}|c0={case["c0_variant"]}'
            f'|a={case["alpha"]!r}|b={case["beta"]!r}'
            f'|M{s["M"]}N{s["N"]}K{s["K"]}|{case["split"]}')


def manifest_entry(case: dict[str, Any], group: str) -> dict[str, Any]:
    entry = {k: case[k] for k in
             ("case_id", "family", "compat_only", "shape", "dtype", "layout",
              "alpha", "beta", "coefficient_set", "split", "seed", "c0_variant",
              "generator_version")}
    entry["group"] = group
    entry["numpy_version"] = case["numpy_version"]
    entry["key"] = case_key(case)
    entry["hashes"] = case["hashes"]
    return entry


def iter_cases(M: int, N: int, K: int, *, split: str = "development",
               include_coefficients: bool = True,
               ) -> Iterator[tuple[str, dict[str, Any]]]:
    coefficient_sets = [MAIN_COEFFICIENTS]
    if include_coefficients:
        coefficient_sets += list(COEFFICIENT_SETS)
    for config_id in CONFIG_IDS:
        for coeff in coefficient_sets:
            group = "main" if coeff["name"] == MAIN_COEFFICIENTS["name"] else "coefficients"
            for variant in C0_VARIANTS:
                yield group, make_case(config_id, M, N, K, c0_variant=variant,
                                       split=split, coefficients=coeff)


CHECK_GROUPS = {
    "main": {"per_shape": 64, "definition": "32 configs x 2 C0 variants, alpha=1 beta=0"},
    "coefficients": {"per_shape": 128, "definition": "2 extra (alpha,beta) x 32 configs x 2 C0"},
    "total": {"per_shape": 192, "definition": "main + coefficients"},
}


def build_manifest(shapes: Sequence[int], *, split: str = "development",
                   include_coefficients: bool = True) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    per_shape: dict[str, Any] = {}
    for shape in shapes:
        M = N = K = int(shape)
        counters = {"main": 0, "coefficients": 0}
        for group, case in iter_cases(M, N, K, split=split,
                                      include_coefficients=include_coefficients):
            entries.append(manifest_entry(case, group))
            counters[group] += 1
        counters["total"] = counters["main"] + counters["coefficients"]
        per_shape[str(shape)] = counters
    manifest = {
        "generator_version": GENERATOR_VERSION,
        "numpy_version": np.__version__,
        "split": split,
        "c0_variants": list(C0_VARIANTS),
        "main_coefficients": MAIN_COEFFICIENTS,
        "coefficient_sets": list(COEFFICIENT_SETS) if include_coefficients else [],
        "shapes": [int(s) for s in shapes],
        "families": FAMILY_COUNTS,
        "config_ids": list(CONFIG_IDS),
        "check_groups": CHECK_GROUPS,
        "per_shape_counts": per_shape,
        "entry_count": len(entries),
        "entries": entries,
    }
    manifest["manifest_hash"] = sha256_value({k: v for k, v in manifest.items()
                                              if k != "manifest_hash"})
    return manifest


def write_manifest(path: Path, shapes: Sequence[int], *, split: str = "development",
                   include_coefficients: bool = True) -> dict[str, Any]:
    manifest = build_manifest(shapes, split=split,
                              include_coefficients=include_coefficients)
    write_json(Path(path), manifest)
    return manifest


def golden_hashes(shapes: Sequence[int], *, split: str = "development",
                  include_coefficients: bool = True) -> dict[str, Any]:
    manifest = build_manifest(shapes, split=split,
                              include_coefficients=include_coefficients)
    return {
        "generator_version": GENERATOR_VERSION,
        "numpy_version": np.__version__,
        "split": split,
        "shapes": [int(s) for s in shapes],
        "entry_count": manifest["entry_count"],
        "ledger_hash": sha256_value([e["hashes"] for e in manifest["entries"]]),
        "per_shape": {
            str(shape): sha256_value([e["hashes"] for e in manifest["entries"]
                                      if e["shape"]["M"] == int(shape)])
            for shape in shapes
        },
    }


def verify_frozen_hashes(path: Path, shapes: Sequence[int], *, split: str,
                         include_coefficients: bool = True) -> dict[str, Any]:
    stored = json.loads(Path(path).read_text(encoding="utf-8"))
    current = golden_hashes(shapes, split=split,
                            include_coefficients=include_coefficients)
    problems: list[str] = []
    if stored.get("generator_version") != current["generator_version"]:
        problems.append(f"generator_version {stored.get('generator_version')} != "
                        f"{current['generator_version']}")
    if stored.get("numpy_version") != current["numpy_version"]:
        problems.append(f"numpy {stored.get('numpy_version')} != {current['numpy_version']}; "
                        "regenerate data under a new generator_version, do not accept silently")
    for shape in shapes:
        key = str(shape)
        if stored.get("per_shape", {}).get(key) != current["per_shape"][key]:
            problems.append(f"shape {shape} input bytes changed")
    if stored.get("ledger_hash") != current["ledger_hash"]:
        problems.append("whole-split ledger hash changed")
    return {"ok": not problems, "problems": problems, "stored": stored,
            "current": current, "path": str(path)}


def require_case_hashes(case: dict[str, Any], *, expect: dict[str, str]) -> None:
    if case["hashes"] != expect:
        raise AssertionError(f"case byte drift for {case_key(case)}: "
                             f"{case['hashes']} != {expect}")


def _main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="SCOPE experiment input generator")
    ap.add_argument("--shape", type=int, nargs="+", default=[512],
                    help="cube edge(s); M=N=K")
    ap.add_argument("--split", choices=SPLITS, default="development")
    ap.add_argument("--write-manifest", type=Path, default=None)
    ap.add_argument("--write-golden", type=Path, default=None)
    ap.add_argument("--verify-golden", type=Path, default=None)
    ap.add_argument("--no-coefficients", action="store_true")
    ap.add_argument("--show-seeds", action="store_true")
    ap.add_argument("--hash", action="store_true", help="print per-config hashes")
    args = ap.parse_args(argv)

    include_coeff = not args.no_coefficients
    print(f"generator_version = {GENERATOR_VERSION}   numpy = {np.__version__}")
    print(f"shapes {args.shape}   split = {args.split}")

    if args.show_seeds:
        print("seed blocks:", json.dumps(SEED_BASE, indent=2))
        print("structural seeds:", json.dumps(STRUCTURAL_SEEDS, indent=2))

    if args.hash:
        for shape in args.shape:
            M = N = K = shape
            cache: dict = {}
            print(f"\n-- shape {shape} ({args.split}) --")
            print(f'{"case_id":<26}{"c0":<9}{"A":<12}{"B":<12}{"C0":<12}')
            for config_id in CONFIG_IDS:
                for variant in C0_VARIANTS:
                    case = make_case(config_id, M, N, K, c0_variant=variant,
                                     split=args.split, _ab_cache=cache)
                    h = case["hashes"]
                    print(f'{config_id:<26}{variant:<9}{h["A"][:10]:<12}'
                          f'{h["B"][:10]:<12}{h["C0"][:10]:<12}')

    if args.write_manifest:
        manifest = write_manifest(args.write_manifest, args.shape, split=args.split,
                                 include_coefficients=include_coeff)
        print(f"\nmanifest -> {args.write_manifest}  "
              f"({manifest['entry_count']} entries, "
              f"per shape {manifest['per_shape_counts']})")

    if args.write_golden:
        ledger = golden_hashes(args.shape, split=args.split,
                               include_coefficients=include_coeff)
        write_json(args.write_golden, ledger)
        print(f"\ngolden -> {args.write_golden}  ledger_hash={ledger['ledger_hash'][:16]}")

    if args.verify_golden:
        result = verify_frozen_hashes(args.verify_golden, args.shape, split=args.split,
                                      include_coefficients=include_coeff)
        print(f"\nverify {args.verify_golden}: "
              f"{'OK' if result['ok'] else 'DRIFT'} {result['problems']}")
        if not result["ok"]:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
