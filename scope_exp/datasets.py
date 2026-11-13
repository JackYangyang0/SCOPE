
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from . import data
from .paths import CampaignPaths
from .util import ensure_dir, read_json, sha256_file, sha256_value, utc_now, write_json

DATASET_VERSION = "scope-exp-dataset-v1"


@dataclass
class DatasetPlan:
    shapes: Sequence[int]
    splits: Sequence[str]
    include_coefficients: bool = True

    def jobs(self) -> list[tuple[int, str]]:
        return [(int(shape), split) for split in self.splits for shape in self.shapes]

    def estimated_manifest_entries(self) -> int:
        per_shape = 64 + (128 if self.include_coefficients else 0)
        return len(self.jobs()) * per_shape


def _job(shape: int, split: str, include_coefficients: bool) -> dict[str, Any]:
    manifest = data.build_manifest([shape], split=split,
                                   include_coefficients=include_coefficients)
    return {"shape": int(shape), "split": split, "manifest": manifest}


def build_split_manifest(shapes: Sequence[int], split: str, *,
                         include_coefficients: bool = True,
                         jobs: int = 1) -> dict[str, Any]:
    shapes = [int(shape) for shape in shapes]
    results: list[dict[str, Any]] = []
    if jobs <= 1 or len(shapes) == 1:
        for shape in shapes:
            results.append(_job(shape, split, include_coefficients))
    else:
        with ProcessPoolExecutor(max_workers=max(1, min(jobs, len(shapes)))) as pool:
            futures = [pool.submit(_job, shape, split, include_coefficients)
                       for shape in shapes]
            for future in as_completed(futures):
                results.append(future.result())
    results.sort(key=lambda item: item["shape"])

    entries: list[dict[str, Any]] = []
    per_shape: dict[str, Any] = {}
    for item in results:
        manifest = item["manifest"]
        entries.extend(manifest["entries"])
        per_shape[str(item["shape"])] = manifest["per_shape_counts"][str(item["shape"])]
    combined = {
        "dataset_version": DATASET_VERSION,
        "generator_version": data.GENERATOR_VERSION,
        "numpy_version": np.__version__,
        "split": split,
        "shapes": shapes,
        "include_coefficients": include_coefficients,
        "config_ids": list(data.CONFIG_IDS),
        "families": data.FAMILY_COUNTS,
        "c0_variants": list(data.C0_VARIANTS),
        "main_coefficients": data.MAIN_COEFFICIENTS,
        "coefficient_sets": list(data.COEFFICIENT_SETS) if include_coefficients else [],
        "check_groups": data.CHECK_GROUPS,
        "per_shape_counts": per_shape,
        "entry_count": len(entries),
        "entries": entries,
        "created_at": utc_now(),
    }
    combined["manifest_hash"] = sha256_value(combined)
    return combined


def ledger_for(manifest: dict[str, Any]) -> dict[str, Any]:
    per_shape: dict[str, str] = {}
    for shape in manifest["shapes"]:
        per_shape[str(shape)] = sha256_value(
            [entry["hashes"] for entry in manifest["entries"]
             if int(entry["shape"]["M"]) == int(shape)])
    ledger = {
        "dataset_version": DATASET_VERSION,
        "generator_version": manifest["generator_version"],
        "split": manifest["split"],
        "shapes": manifest["shapes"],
        "entry_count": manifest["entry_count"],
        "per_shape": per_shape,
        "ledger_hash": sha256_value([entry["hashes"] for entry in manifest["entries"]]),
    }
    return ledger


def write_datasets(paths: CampaignPaths, plan: DatasetPlan, *, jobs: int = 1,
                   splits: Sequence[str] | None = None) -> dict[str, Any]:
    summary: dict[str, Any] = {"dataset_version": DATASET_VERSION,
                               "created_at": utc_now(),
                               "shapes": [int(s) for s in plan.shapes],
                               "splits": {}}
    for split in (splits or plan.splits):
        manifest = build_split_manifest(plan.shapes, split,
                                        include_coefficients=plan.include_coefficients,
                                        jobs=jobs)
        directory = ensure_dir(paths.dataset_dir(split))
        manifest_path = directory / "manifest.json"
        ledger_path = directory / "hashes.sha256.json"
        write_json(manifest_path, manifest)
        ledger = ledger_for(manifest)
        ledger["manifest_file_sha256"] = sha256_file(manifest_path)
        write_json(ledger_path, ledger)
        summary["splits"][split] = {
            "manifest": str(manifest_path),
            "manifest_hash": manifest["manifest_hash"],
            "manifest_file_sha256": ledger["manifest_file_sha256"],
            "ledger": str(ledger_path),
            "ledger_hash": ledger["ledger_hash"],
            "entry_count": manifest["entry_count"],
            "per_shape_counts": manifest["per_shape_counts"],
        }
    summary["manifest_hash"] = sha256_value(summary["splits"])
    write_json(paths.root / "datasets" / "datasets.json", summary)
    return summary


def load_manifest(paths: CampaignPaths, split: str) -> dict[str, Any]:
    path = paths.dataset_manifest(split)
    if not path.exists():
        raise FileNotFoundError(
            f"dataset manifest missing for split {split!r}: {path}. "
            "Run `scope-exp prepare` first.")
    return read_json(path)


def verify_dataset(paths: CampaignPaths, split: str, *, jobs: int = 1) -> dict[str, Any]:
    manifest = load_manifest(paths, split)
    ledger_path = paths.dataset_dir(split) / "hashes.sha256.json"
    stored = read_json(ledger_path)
    regenerated = build_split_manifest(manifest["shapes"], split,
                                       include_coefficients=bool(manifest.get("coefficient_sets")),
                                       jobs=jobs)
    current = ledger_for(regenerated)
    problems: list[str] = []
    if current["generator_version"] != stored.get("generator_version"):
        problems.append("generator_version changed")
    if current["entry_count"] != stored.get("entry_count"):
        problems.append(f"entry count {stored.get('entry_count')} -> {current['entry_count']}")
    for shape in manifest["shapes"]:
        key = str(shape)
        if current["per_shape"].get(key) != stored["per_shape"].get(key):
            problems.append(f"shape {shape} input bytes changed")
    if current["ledger_hash"] != stored.get("ledger_hash"):
        problems.append("ledger hash changed")
    if regenerated["manifest_hash"] != manifest.get("manifest_hash"):
        problems.append("manifest hash changed")
    return {"ok": not problems, "problems": problems, "split": split,
            "stored_ledger": str(ledger_path),
            "stored_ledger_hash": stored.get("ledger_hash"),
            "current_ledger_hash": current["ledger_hash"]}


def entries_for_shape(manifest: dict[str, Any], shape: int,
                      *, include_coefficients: bool = True) -> list[dict[str, Any]]:
    return [entry for entry in manifest["entries"]
            if int(entry["shape"]["M"]) == int(shape)
            and (include_coefficients or entry["group"] == "main")]


def dataset_summary(paths: CampaignPaths) -> dict[str, Any]:
    path = paths.root / "datasets" / "datasets.json"
    if not path.exists():
        return {"present": False, "path": str(path)}
    record = read_json(path)
    record["present"] = True
    record["path"] = str(path)
    return record
