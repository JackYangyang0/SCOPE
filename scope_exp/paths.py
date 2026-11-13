
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from . import HARNESS_VERSION
from .util import ensure_dir, safe_id

LAYOUT_VERSION = "scope-exp-layout-v1"


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class CampaignPaths:

    root: Path
    campaign_id: str

    @property
    def protocol(self) -> Path: return self.root / "protocol.json"

    @property
    def frozen_protocol(self) -> Path: return self.root / "FROZEN_PROTOCOL.json"

    @property
    def task_manifest_csv(self) -> Path: return self.root / "task_manifest.csv"

    @property
    def task_manifest_json(self) -> Path: return self.root / "task_manifest.json"

    @property
    def environment(self) -> Path: return self.root / "environment.json"

    @property
    def source_snapshot(self) -> Path: return self.root / "source_snapshot.json"

    @property
    def layout_json(self) -> Path: return self.root / "layout.json"

    def dataset_dir(self, split: str) -> Path: return self.root / "datasets" / split

    def dataset_manifest(self, split: str) -> Path:
        return self.dataset_dir(split) / "manifest.json"

    @property
    def references(self) -> Path: return self.root / "references"

    def reference_dir(self, key: str) -> Path:
        return self.references / safe_id(key)

    @property
    def tasks(self) -> Path: return self.root / "tasks"

    def task_dir(self, logical_run_id: str) -> Path:
        return self.tasks / safe_id(logical_run_id)

    def task_status(self, logical_run_id: str) -> Path:
        return self.task_dir(logical_run_id) / "status.json"

    def task_resolved_config(self, logical_run_id: str) -> Path:
        return self.task_dir(logical_run_id) / "resolved_config_redacted.json"

    def task_selected(self, logical_run_id: str) -> Path:
        return self.task_dir(logical_run_id) / "selected.json"

    def task_candidates_csv(self, logical_run_id: str) -> Path:
        return self.task_dir(logical_run_id) / "candidates.csv"

    def task_validation_csv(self, logical_run_id: str) -> Path:
        return self.task_dir(logical_run_id) / "validation.csv"

    def task_validation_json(self, logical_run_id: str) -> Path:
        return self.task_dir(logical_run_id) / "validation.json"

    def task_coefficient_csv(self, logical_run_id: str) -> Path:
        return self.task_dir(logical_run_id) / "coefficient_checks.csv"

    def task_timing_csv(self, logical_run_id: str) -> Path:
        return self.task_dir(logical_run_id) / "timing.csv"

    def task_timing_json(self, logical_run_id: str) -> Path:
        return self.task_dir(logical_run_id) / "timing.json"

    def task_sanitizer_dir(self, logical_run_id: str) -> Path:
        return self.task_dir(logical_run_id) / "sanitizer"

    def task_build_json(self, logical_run_id: str) -> Path:
        return self.task_dir(logical_run_id) / "build.json"

    def attempt_dir(self, logical_run_id: str, attempt_id: str) -> Path:
        return self.task_dir(logical_run_id) / "attempts" / safe_id(attempt_id)

    def attempt_events(self, logical_run_id: str, attempt_id: str) -> Path:
        return self.attempt_dir(logical_run_id, attempt_id) / "events.jsonl"

    def attempt_ledger(self, logical_run_id: str, attempt_id: str) -> Path:
        return self.attempt_dir(logical_run_id, attempt_id) / "api_ledger.jsonl"

    def attempt_logs(self, logical_run_id: str, attempt_id: str) -> tuple[Path, Path]:
        base = self.attempt_dir(logical_run_id, attempt_id)
        return base / "stdout.log", base / "stderr.log"

    def checkpoint_dir(self, logical_run_id: str, phase: str, artifact_hash: str) -> Path:
        if phase not in ("p1", "p2", "p3"):
            raise ValueError(f"unknown checkpoint phase: {phase}")
        return (self.task_dir(logical_run_id) / "checkpoints" / phase
                / safe_id(artifact_hash, "nohash"))

    @property
    def branches(self) -> Path: return self.root / "branches"

    def branch_dir(self, parent_id: str, kind: str) -> Path:
        return self.branches / safe_id(parent_id) / safe_id(kind)

    @property
    def baselines(self) -> Path: return self.root / "baselines"

    def baseline_dir(self, name: str) -> Path:
        return self.baselines / safe_id(name)

    @property
    def analysis(self) -> Path: return self.root / "analysis"

    @property
    def figures(self) -> Path: return self.root / "figures"

    @property
    def audit(self) -> Path: return self.root / "audit"

    @property
    def workspace_root(self) -> Path:
        return self.root.parent / "_workspaces" / safe_id(self.campaign_id)

    def workspace(self, logical_run_id: str) -> Path:
        return self.workspace_root / safe_id(logical_run_id)

    def initialize(self) -> "CampaignPaths":
        for directory in (self.root, self.references, self.tasks, self.branches,
                          self.baselines, self.analysis, self.figures, self.audit,
                          self.root / "datasets"):
            ensure_dir(directory)
        return self


def campaign_paths(campaign_id: str, *, results_root: Path | None = None) -> CampaignPaths:
    base = Path(results_root) if results_root else (repo_root() / "results")
    return CampaignPaths(root=base / safe_id(campaign_id, "campaign"),
                         campaign_id=campaign_id)


def layout_manifest(paths: CampaignPaths) -> dict:
    return {
        "layout_version": LAYOUT_VERSION,
        "harness_version": HARNESS_VERSION,
        "campaign_id": paths.campaign_id,
        "campaign_root": str(paths.root),
        "workspace_root": str(paths.workspace_root),
        "roles": {
            "protocol.json": "draft protocol, superseded by FROZEN_PROTOCOL.json",
            "FROZEN_PROTOCOL.json": "immutable once `freeze` completes",
            "task_manifest.csv": "one row per planned logical task, frozen order",
            "environment.json": "toolchain/GPU/python snapshot at prepare time",
            "source_snapshot.json": "hashes of the frozen method checkout",
            "datasets/<split>/manifest.json": "32 configs x 2 C0 = 64 checks per shape",
            "references/<hash>/": "cached FP64 reference metadata per case",
            "tasks/<id>/attempts/<aid>/": "append-only events + API ledger + logs",
            "tasks/<id>/checkpoints/<phase>/<hash>/": "immutable P1/P2/P3 artifacts",
            "tasks/<id>/validation.csv": "independent full-output revalidation rows",
            "tasks/<id>/timing.csv": "raw per-launch latency samples",
            "branches/<parent>/without_feedback/": "shared-parent stage branch",
            "analysis/": "regenerated from raw records only",
            "audit/": "completeness, identity chain, claim traceability",
        },
        "transient": {
            "_workspaces/<campaign_id>/<task>/": "isolated SCOPE checkout per attempt; "
                                                 "safe to delete once checkpoints are captured",
        },
    }
