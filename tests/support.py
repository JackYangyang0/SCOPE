"""Shared helpers for the harness test suite.

Everything runs on a machine with no GPU and no API key: the CPU toolchain (MSVC
or gcc) is the only external dependency, and tests that need a compiler skip
themselves when none is present.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from SCOPE.verification import driver_build as build
from SCOPE.utils import harness_config as config_module
from SCOPE.data import matrix_cases as data
from SCOPE.experiments import task_matrix as tasks
from SCOPE.utils import campaign_paths as paths_module
from SCOPE.utils.campaign_paths import CampaignPaths

REPO = paths_module.repo_root()
CPU_SKELETON = REPO / "gemm_code" / "cpu_skeleton"


def temp_campaign(campaign_id: str = "test") -> CampaignPaths:
    root = Path(tempfile.mkdtemp(prefix="scope-exp-test-"))
    return paths_module.campaign_paths(campaign_id, results_root=root).initialize()


def cpu_toolchain():
    toolchain = build.discover_toolchain("cpu")
    return toolchain if toolchain.available else None


def base_config() -> Path:
    for name in ("conf.experiment.yaml", "conf.example.yaml"):
        candidate = REPO / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError("no configuration template in the checkout")


def sample_task(**overrides: Any) -> dict[str, Any]:
    task = {"task_id": "T-512-r01", "shape_M": 512, "shape_N": 512, "shape_K": 512,
            "variant": "full", "model_role": None}
    task.update(overrides)
    return task


def resolved_sample_config() -> dict[str, Any]:
    base = config_module.load_yaml(base_config())
    return config_module.resolve_task_config(base, sample_task(), model=None,
                                             base_config_path=base_config())


def tiny_protocol(paths: CampaignPaths, *, shape: int = 32) -> dict[str, Any]:
    """A minimal but structurally complete protocol for pipeline tests.

    It is *not* a formal protocol: the controls gate is declared passed because
    the tests that need a real gate run the controls themselves.  Formal
    protocols are only ever produced by ``experiments/freeze.py``.
    """
    from SCOPE.verification import timing_protocol as timing
    from SCOPE.utils.harness_util import sha256_value, utc_now

    task = tasks.Task(task_id="T-32-r01", block="REHEARSAL", kind="independent",
                      shape=(shape, shape, shape), repeat=1, variant="full")
    protocol = {
        "protocol_version": "scope-exp-protocol-v1",
        "campaign_id": paths.campaign_id,
        "status": "frozen",
        "harness_version": "test",
        "layout_version": "test",
        "created_at": utc_now(),
        "backend": "cpu",
        "shapes": [shape],
        "repeats": 1,
        "blocks": ["REHEARSAL"],
        "source": {
            "repo_root": str(REPO), "git_head": None, "git_dirty": None,
            "source_digest": "0" * 64, "file_count": 1,
            "method_files": {"app.py": "1" * 64},
            "base_config_path": str(base_config()),
        },
        "environment": {"python": "test", "platform": "test",
                        "numpy_version": np.__version__, "toolchain": {}},
        "tasks": {
            "counts": {"REHEARSAL_independent": 1},
            "task_ids": [task.task_id],
            "order": [task.task_id],
            "manifest_sha256": sha256_value([task.task_id]),
            "blocks": ["REHEARSAL"],
            "task_records": [task.as_record()],
            "denominators": {"independent": 1, "branches": 0, "aliases": 0},
        },
        "models": {"rehearsal": {"role": "rehearsal", "model": "rehearsal-no-llm",
                                 "api_key_env": None, "api_key_env_present": True}},
        "data": {"generator_version": data.GENERATOR_VERSION,
                 "splits": ["development", "generation", "postselection_test"],
                 "manifest_hashes": {"postselection_test": "2" * 64},
                 "checks_per_kernel_per_shape": {"main": 64, "coefficients": 128,
                                                 "total": 192},
                 "coefficient_sets": list(data.COEFFICIENT_SETS),
                 "evaluation_split": "postselection_test",
                 "generation_split": "generation",
                 "pilot_split": "development"},
        "validation": {"policies": [], "atol": 1e-5, "rtol": 1e-3,
                       "relative_l2": 1e-3, "strict_abs": 1e-6,
                       "reference": {}, "timeout_seconds": 900.0,
                       "repeat_checks": 1, "hypotheses_max_shape": shape},
        "timing": timing.timing_protocol_record(),
        "budget": {"per_task_wall_seconds": 600.0,
                   "per_phase_wall_seconds": {"p1": 300.0, "p2": 180.0, "p3": 120.0},
                   "pilot_wall_seconds": 600.0, "api_timeout_seconds": 60.0,
                   "max_retries": 1, "search": {"max_candidates_per_shape": 32,
                                               "rounds": 1,
                                               "max_frontier_states_per_stage": 1,
                                               "max_total_compilations": 3,
                                               "tiling_resource_pool_size": 8},
                   "stop_rules": ["test"], "budget_profile": "default"},
        "selection_rules": {"top1": "test", "no_reselection_after_validation": True,
                            "without_feedback_start": "test",
                            "budget_exhaustion": "test"},
        "statistics": {"alpha": 0.05, "bootstrap_resamples": 10000,
                       "bootstrap_seed": 1, "wilson_intervals": True,
                       "paired_by_parent": True, "unit_of_analysis": "task"},
        "controls_gate": {"path": None, "sha256": None, "backend": "cpu", "ok": True},
        "sanitizer": {"timeout_seconds": 60.0, "tools": ["memcheck"]},
        "protocol_hash": "test-protocol",
    }
    return protocol
