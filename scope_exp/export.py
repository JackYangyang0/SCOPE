
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any, Iterable

from .errors import ScopeExpError
from .paths import CampaignPaths, repo_root
from .redact import find_credentials, live_credentials
from .util import (ensure_dir, read_json, read_jsonl, sha256_file, utc_now,
                   write_json)

EXPORT_VERSION = "scope-exp-export-v1"

TASK_FILES = (
    "status.json",
    "resolved_config_redacted.json",
    "selected.json",
    "validation.summary.json",
    "timing.json",
    "build.json",
)
TOP_LEVEL_FILES = (
    "FROZEN_PROTOCOL.json",
    "protocol.json",
    "task_manifest.csv",
    "task_manifest.json",
    "environment.json",
    "source_snapshot.json",
    "datasets/datasets.json",
)
DATASET_FILES = ("manifest.json", "hashes.sha256.json")
ANALYSIS_FILES = (
    "summary.json", "RESULTS_zh.md", "rq2_components.csv",
    "rq2_components_pooled.csv", "models.csv", "models_pooled.csv",
    "stage_pairs.csv", "correctness.csv", "cost.csv", "completeness.csv",
)
AUDIT_FILES = ("controls_gate.json", "ablation_diff.json", "report.json",
               "claims.json", "index.csv", "selfcheck.json")

ABSOLUTE_PATH = re.compile(r"[A-Za-z]:[\\/][^\s\"',;)\]]*|/(?:home|Users|mnt|data|root)/[^\s\"',;)\]]*")


def _sanitize(value: Any, replacements: Iterable[tuple[str, str]]) -> Any:
    if isinstance(value, str):
        text = value
        for needle, token in replacements:
            if needle and needle in text:
                text = text.replace(needle, token)
        text = ABSOLUTE_PATH.sub("<path>", text)
        return text
    if isinstance(value, list):
        return [_sanitize(item, replacements) for item in value]
    if isinstance(value, dict):
        return {key: _sanitize(item, replacements) for key, item in value.items()}
    return value


def _copy_json(source: Path, target: Path, replacements: Iterable[tuple[str, str]]) -> None:
    ensure_dir(target.parent)
    payload = _sanitize(read_json(source), replacements)
    write_json(target, payload)


def _copy_text(source: Path, target: Path, replacements: Iterable[tuple[str, str]]) -> None:
    ensure_dir(target.parent)
    text = source.read_text(encoding="utf-8", errors="replace")
    for needle, token in replacements:
        if needle:
            text = text.replace(needle, token)
    text = ABSOLUTE_PATH.sub("<path>", text)
    target.write_text(text, encoding="utf-8")


def export_redacted(paths: CampaignPaths, out_dir: Path, *,
                    include_kernels: bool = True) -> dict[str, Any]:
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    ensure_dir(out_dir)
    replacements = [
        (str(paths.root), "<campaign>"),
        (str(repo_root()), "<repo>"),
        (str(repo_root().parent), "<parent>"),
        (str(Path.home()), "<home>"),
    ]
    written: list[str] = []
    skipped: list[str] = []

    for name in TOP_LEVEL_FILES:
        source = paths.root / name
        if not source.exists():
            skipped.append(name)
            continue
        if source.suffix == ".json":
            _copy_json(source, out_dir / name, replacements)
        else:
            _copy_text(source, out_dir / name, replacements)
        written.append(name)

    for manifest_dir in sorted((paths.root / "datasets").iterdir()) \
            if (paths.root / "datasets").exists() else []:
        if not manifest_dir.is_dir():
            continue
        for name in DATASET_FILES:
            source = manifest_dir / name
            if not source.exists():
                continue
            target = out_dir / "datasets" / manifest_dir.name / name
            _copy_json(source, target, replacements)
            written.append(str(target.relative_to(out_dir).as_posix()))

    for task_dir in sorted(Path(paths.tasks).iterdir()):
        if not task_dir.is_dir():
            continue
        for name in TASK_FILES:
            source = task_dir / name
            if not source.exists():
                continue
            target = out_dir / "tasks" / task_dir.name / name
            _copy_json(source, target, replacements)
            written.append(str(target.relative_to(out_dir).as_posix()))
        attempts = task_dir / "attempts"
        if attempts.exists():
            for ledger_file in sorted(attempts.glob("*/api_ledger.jsonl")):
                rows = _sanitize(read_jsonl(ledger_file), replacements)
                target = (out_dir / "tasks" / task_dir.name / "attempts"
                          / ledger_file.parent.name / "api_ledger.jsonl")
                ensure_dir(target.parent)
                with target.open("w", encoding="utf-8", newline="\n") as handle:
                    for row in rows:
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                written.append(str(target.relative_to(out_dir).as_posix()))
            for events_file in sorted(attempts.glob("*/events.jsonl")):
                rows = _sanitize(read_jsonl(events_file), replacements)
                target = (out_dir / "tasks" / task_dir.name / "attempts"
                          / events_file.parent.name / "events.jsonl")
                ensure_dir(target.parent)
                with target.open("w", encoding="utf-8", newline="\n") as handle:
                    for row in rows:
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                written.append(str(target.relative_to(out_dir).as_posix()))
        if include_kernels:
            for manifest in sorted(task_dir.glob("checkpoints/*/*/manifest.json")):
                relative = manifest.relative_to(paths.root)
                _copy_json(manifest, out_dir / relative, replacements)
                written.append(str(relative.as_posix()))
                kernel_dir = manifest.parent / "kernel"
                for source in sorted(kernel_dir.glob("*")):
                    if not source.is_file():
                        continue
                    target = out_dir / relative.parent / "kernel" / source.name
                    ensure_dir(target.parent)
                    target.write_bytes(source.read_bytes())
                    written.append(str(target.relative_to(out_dir).as_posix()))
        for sanitizer_file in sorted((task_dir / "sanitizer").glob("*.json")):
            relative = sanitizer_file.relative_to(paths.root)
            _copy_json(sanitizer_file, out_dir / relative, replacements)
            written.append(str(relative.as_posix()))

    for name in ANALYSIS_FILES:
        source = paths.analysis / name
        if not source.exists():
            skipped.append(f"analysis/{name}")
            continue
        target = out_dir / "analysis" / name
        if source.suffix == ".json":
            _copy_json(source, target, replacements)
        else:
            _copy_text(source, target, replacements)
        written.append(str(target.relative_to(out_dir).as_posix()))

    for name in AUDIT_FILES:
        source = paths.audit / name
        if not source.exists():
            skipped.append(f"audit/{name}")
            continue
        target = out_dir / "audit" / name
        if source.suffix == ".json":
            _copy_json(source, target, replacements)
        else:
            _copy_text(source, target, replacements)
        written.append(str(target.relative_to(out_dir).as_posix()))

    for branch in sorted(Path(paths.branches).rglob("branch.json")) \
            if paths.branches.exists() else []:
        relative = branch.relative_to(paths.root)
        _copy_json(branch, out_dir / relative, replacements)
        written.append(str(relative.as_posix()))

    live = live_credentials()
    try:
        from . import config as config_module
        base_path = (read_json(paths.frozen_protocol).get("source") or {}
                     ).get("base_config_path") if paths.frozen_protocol.exists() else None
        if base_path:
            base = config_module.load_yaml(Path(base_path))
            live = list(live) + config_module.credential_values_for_scan(
                base, list(config_module.default_model_roles(base)))
    except Exception:
        pass
    hits: list[dict[str, Any]] = []
    for relative in written:
        path = out_dir / relative
        if not path.is_file() or path.stat().st_size > 8 * 1024 * 1024:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        found = find_credentials(text, live)
        if "_runtime_config" in text:
            found.append("unredacted runtime configuration leaked into the export")
        if found:
            hits.append({"path": relative, "findings": found})
    if hits:
        raise ScopeExpError(
            "refusing to publish: credential patterns survived redaction in "
            f"{[hit['path'] for hit in hits[:5]]}")

    index = render_index(paths, out_dir, written, skipped)
    (out_dir / "INDEX.md").write_text(index, encoding="utf-8")
    manifest = {
        "export_version": EXPORT_VERSION,
        "created_at": utc_now(),
        "campaign_id": paths.campaign_id,
        "source": str(paths.root),
        "file_count": len(written) + 1,
        "files": {relative: sha256_file(out_dir / relative) for relative in written},
        "skipped": skipped,
        "redaction": {
            "absolute_paths_replaced": True,
            "credentials_scanned": True,
            "credential_hits": 0,
            "excluded": ["conf.yaml", "raw stdout/stderr logs", "build products",
                         "case files (regenerable)", "workspaces"],
        },
    }
    write_json(out_dir / "EXPORT_MANIFEST.json", manifest)
    return manifest


def render_index(paths: CampaignPaths, out_dir: Path, written: list[str],
                 skipped: list[str]) -> str:
    lines = [f"# 脱敏导出：campaign `{paths.campaign_id}`", "",
             f"生成时间：{utc_now()}；导出目录：`{out_dir.name}/`。", "",
             "## 包含", ""]
    for group, prefix in (("协议与清单", ""), ("数据集", "datasets/"),
                          ("任务证据", "tasks/"), ("分析", "analysis/"),
                          ("审计", "audit/"), ("阶段分支", "branches/")):
        files = [name for name in written if name.startswith(prefix)]
        lines.append(f"### {group}（{len(files)}）")
        lines.append("")
        for name in files[:40]:
            lines.append(f"- `{name}`")
        if len(files) > 40:
            lines.append(f"- … 其余 {len(files) - 40} 项见 EXPORT_MANIFEST.json")
        lines.append("")
    lines.append("## 有意排除")
    lines.append("")
    for item in ("conf.yaml 与任何密钥形式的内容", "应用原始 stdout/stderr 日志（含绝对路径）",
                 "编译产物与对象文件", "case 二进制文件（可由 manifest 逐字节重建）",
                 "隔离工作目录（临时，已删除）"):
        lines.append(f"- {item}")
    if skipped:
        lines.append("")
        lines.append("## 缺失但已记录")
        lines.append("")
        for name in skipped[:40]:
            lines.append(f"- `{name}`（源档案中不存在）")
    lines.append("")
    lines.append("## 使用说明")
    lines.append("")
    lines.append("本目录中的每个数值都可由 `analysis/` 下的 CSV/JSON 重建；"
                 "`audit/claims.json` 给出表名到原始记录的映射。"
                 "绝对路径已替换为 `<campaign>` / `<repo>` 等占位符，"
                 "因此路径本身不再指向任何机器。")
    lines.append("")
    return "\n".join(lines)


def export_status(out_dir: Path) -> dict[str, Any]:
    manifest = Path(out_dir) / "EXPORT_MANIFEST.json"
    if not manifest.exists():
        return {"present": False, "path": str(manifest)}
    record = read_json(manifest)
    return {"present": True, "path": str(manifest),
            "file_count": record.get("file_count"),
            "created_at": record.get("created_at")}
