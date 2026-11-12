
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from .errors import InfraError
from .paths import CampaignPaths, repo_root
from .util import (ensure_dir, read_json, sha256_file, sha256_value, tree_hash,
                   utc_now, which, write_json)

ISOLATION_VERSION = "scope-exp-isolation-v1"

COPY_DIRS = ("llm", "utils", "verification", "specialization", "generate_ir",
             "gemm_code", "data", "diagnosis", "scope_exp")
COPY_FILES = ("app.py", "tune.py", "tune_cuda.py", "benchmark.py", "cpu_app.py",
              "compare_kernel_controls.py", "compare_load_layout.py",
              "compare_reference_kernel.py", "compare_tile_loads.py",
              "diagnose_tiling_response.py", "requirements.txt",
              "conf.example.yaml", "README.md", "__init__.py", "MANIFEST.json")
SKIP_DIR_NAMES = {"__pycache__", ".git", ".venv", ".piptmp", "build", "obj"}

WATCH_PATTERNS = ("app.py", "tune.py", "llm/**/*.py", "utils/**/*.py",
                  "verification/**/*.py", "gemm_code/skeleton/*",
                  "gemm_code/cpu_skeleton/*.c", "gemm_code/cpu_skeleton/*.h",
                  "data/lib/*.json", "data/graph/*.json")


@dataclass
class Workspace:
    task_id: str
    root: Path
    package_dir: Path
    created_at: str = field(default_factory=utc_now)
    source_digest: str | None = None
    copy_digest: str | None = None
    verification: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)

    @property
    def pythonpath(self) -> str:
        return os.pathsep.join([str(self.root), str(self.package_dir)])

    def env(self, base: dict[str, str] | None = None) -> dict[str, str]:
        env = dict(base or os.environ)
        env["PYTHONPATH"] = self.pythonpath + (
            os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        env["SCOPE_EXP_TASK_ID"] = self.task_id
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("OPENBLAS_NUM_THREADS", "1")
        env.setdefault("MKL_NUM_THREADS", "1")
        return env

    def as_record(self) -> dict[str, Any]:
        return {"isolation_version": ISOLATION_VERSION, "task_id": self.task_id,
                "root": str(self.root), "package_dir": str(self.package_dir),
                "created_at": self.created_at, "source_digest": self.source_digest,
                "copy_digest": self.copy_digest,
                "verification": self.verification, "events": self.events}


def _copy_tree(source: Path, destination: Path) -> int:
    count = 0
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if any(part in SKIP_DIR_NAMES for part in relative.parts):
            continue
        target = destination / relative
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        if path.is_symlink() or path.is_junction():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        count += 1
    return count


def watch_snapshot(root: Path | None = None) -> dict[str, str]:
    root = Path(root or repo_root())
    files: dict[str, str] = {}
    for pattern in WATCH_PATTERNS:
        for path in sorted(root.glob(pattern)):
            if path.is_file():
                files[path.relative_to(root).as_posix()] = sha256_file(path)
    return files


def create_workspace(paths: CampaignPaths, task_id: str, *,
                     source: Path | None = None, clean: bool = True,
                     method_files: Sequence[str] | None = None) -> Workspace:
    source = Path(source or repo_root()).resolve()
    root = paths.workspace(task_id)
    package_dir = root / "SCOPE"
    if clean and root.exists():
        shutil.rmtree(root, ignore_errors=True)
    ensure_dir(root)
    if package_dir.exists():
        raise InfraError(f"workspace already exists: {package_dir}", stage="isolation")
    ensure_dir(package_dir)

    copied = 0
    for directory in COPY_DIRS:
        origin = source / directory
        if origin.is_dir():
            copied += _copy_tree(origin, package_dir / directory)
    for name in COPY_FILES:
        origin = source / name
        if origin.is_file():
            shutil.copyfile(origin, package_dir / name)
            copied += 1
    if method_files is None:
        method_files = sorted(watch_snapshot(source))
    for name in method_files:
        origin = source / name
        target = package_dir / name
        if origin.is_file() and not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(origin, target)
            copied += 1
    if not (package_dir / "app.py").exists():
        raise InfraError(f"workspace copy has no app.py: {package_dir}", stage="isolation")

    workspace = Workspace(task_id=task_id, root=root, package_dir=package_dir)
    missing = [name for name in method_files if not (package_dir / name).exists()]
    mismatched: list[str] = []
    for name in method_files:
        target = package_dir / name
        if not target.exists():
            continue
        if sha256_file(source / name) != sha256_file(target):
            mismatched.append(name)
    workspace.source_digest = sha256_value({name: sha256_file(source / name)
                                            for name in method_files
                                            if (source / name).exists()})
    workspace.copy_digest = sha256_value({name: sha256_file(package_dir / name)
                                          for name in method_files
                                          if (package_dir / name).exists()})
    workspace.verification = {
        "files_copied": copied,
        "method_files_expected": len(method_files),
        "missing_in_copy": missing,
        "hash_mismatch": mismatched,
        "source_digest": workspace.source_digest,
        "copy_digest": workspace.copy_digest,
        "identical": not missing and not mismatched,
    }
    if missing or mismatched:
        raise InfraError(f"isolated copy is not identical to the frozen checkout: "
                         f"missing={missing[:5]} mismatched={mismatched[:5]}",
                         stage="isolation")
    (root / "sitecustomize.py").write_text(
        "# generated by scope_exp: installs the per-request API ledger hook\n"
        "import os\n"
        "if os.environ.get('SCOPE_EXP_LEDGER'):\n"
        "    try:\n"
        "        from scope_exp.instrument import install as _install_ledger_hook\n"
        "        _install_ledger_hook()\n"
        "    except Exception:\n"
        "        pass\n",
        encoding="utf-8")
    write_json(root / ".scope_exp_workspace.json", workspace.as_record())
    return workspace


def assert_source_untouched(workspace: Workspace, *, source: Path | None = None) -> dict[str, Any]:
    source = Path(source or repo_root())
    current = watch_snapshot(source)
    record = {
        "checked_at": utc_now(),
        "source_root": str(source),
        "digest": sha256_value(current),
        "expected_digest": workspace.source_digest,
        "ok": True,
        "changed": [],
    }
    expected = None
    record_path = Path(workspace.root) / ".scope_exp_source_before.json"
    if record_path.exists():
        expected = read_json(record_path).get("digest")
    if expected is not None:
        record["ok"] = record["digest"] == expected
        if not record["ok"]:
            before = read_json(record_path).get("files") or {}
            record["changed"] = sorted(name for name in current
                                       if before.get(name) != current.get(name))
    write_json(Path(workspace.root) / ".scope_exp_source_after.json", record)
    return record


def snapshot_source_before(workspace: Workspace, *, source: Path | None = None) -> dict[str, Any]:
    source = Path(source or repo_root())
    files = watch_snapshot(source)
    record = {"captured_at": utc_now(), "source_root": str(source),
              "files": files, "digest": sha256_value(files)}
    write_json(Path(workspace.root) / ".scope_exp_source_before.json", record)
    return record


def artifact_paths(workspace: Workspace) -> dict[str, Path]:
    root = workspace.package_dir
    return {
        "root": root,
        "results": root / "results",
        "check": root / "results" / "check",
        "code": root / "results" / "code",
        "chain": root / "results" / "chain",
        "ir_patch": root / "data" / "IRs" / "ir_patch",
        "generated_code": root / "gemm_code" / "code",
        "final_skeleton": root / "gemm_code" / "skeleton",
        "final_shape_tuning": root / "gemm_code" / "final_shape_tuning",
        "shape_specialization": root / "gemm_code" / "shape_specialization",
    }


def app_command(workspace: Workspace, argv: Sequence[str]) -> list[str]:
    import sys
    return [sys.executable, *argv]


def cleanup_workspace(workspace: Workspace, *, keep: bool = False) -> None:
    if keep:
        return
    shutil.rmtree(workspace.root, ignore_errors=True)
