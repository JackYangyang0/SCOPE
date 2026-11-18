from __future__ import annotations

import errno
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def parse_utc(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)


def elapsed_seconds(start: str, end: str | None = None) -> float:
    stop = parse_utc(end) if end else datetime.now(timezone.utc)
    return (stop - parse_utc(start)).total_seconds()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_value(value: Any) -> str:
    return sha256_text(canonical_json(value))


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def sha256_paths(paths: Iterable[Path], root: Path) -> dict[str, str]:
    root = Path(root)
    result: dict[str, str] = {}
    for path in sorted(Path(p) for p in paths):
        rel = path.relative_to(root).as_posix()
        result[rel] = sha256_file(path)
    return result


def tree_hash(root: Path, *, include: Sequence[str] | None = None,
              skip_dirs: Sequence[str] = (".git", "__pycache__", ".venv")) -> dict[str, Any]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(root)
    files: dict[str, str] = {}
    patterns = list(include) if include else None
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            continue
        rel_parts = path.relative_to(root).parts
        if any(part in skip_dirs for part in rel_parts):
            continue
        rel = path.relative_to(root).as_posix()
        if patterns is not None and not any(path.match(p) for p in patterns):
            continue
        files[rel] = sha256_file(path)
    return {"root": str(root), "file_count": len(files), "files": files,
            "digest": sha256_value(files)}


def ensure_dir(path: Path) -> Path:
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)


def write_json(path: Path, value: Any, *, indent: int = 2) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    payload = json.dumps(value, indent=indent, ensure_ascii=False, sort_keys=False,
                         allow_nan=False) + "\n"
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name[:40]}.",
                                    suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return path


def read_json(path: Path, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        if default is None:
            raise FileNotFoundError(path)
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_text(path: Path, text: str) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name[:40]}.",
                                    suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return path


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    line = json.dumps(record, ensure_ascii=False, sort_keys=False,
                      allow_nan=False) + "\n"
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: corrupt JSONL line: {exc}") from exc
    return records


def write_csv(path: Path, rows: Sequence[dict[str, Any]],
              columns: Sequence[str] | None = None) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    rows = list(rows)
    if columns is None:
        columns = list(rows[0].keys()) if rows else []
    lines = [",".join(_csv_cell(c) for c in columns)]
    for row in rows:
        lines.append(",".join(_csv_cell(row.get(c)) for c in columns))
    return write_text(path, "\n".join(lines) + "\n")


def _csv_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    text = str(value)
    if any(ch in text for ch in ',"\n\r'):
        return '"' + text.replace('"', '""') + '"'
    return text


_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def safe_id(value: str, fallback: str = "unnamed") -> str:
    cleaned = _SAFE.sub("_", str(value)).strip("._-")
    return cleaned or fallback


def new_attempt_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid.uuid4().hex[:8]}"


def short_hash(value: Any, length: int = 16) -> str:
    digest = value if isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) else sha256_value(value)
    return digest[:length]


def which(name: str) -> str | None:
    return shutil.which(name)


def run_command(command: Sequence[str], *, cwd: Path | None = None,
                timeout: float | None = None, env: dict[str, str] | None = None,
                stdin_text: str | None = None) -> dict[str, Any]:
    started = time.monotonic()
    record: dict[str, Any] = {
        "argv": [str(part) for part in command],
        "cwd": str(cwd) if cwd else None,
        "returncode": None,
        "stdout": "",
        "stderr": "",
        "duration_seconds": None,
        "timed_out": False,
        "error": None,
    }
    try:
        proc = subprocess.run(
            [str(part) for part in command],
            cwd=str(cwd) if cwd else None,
            env=env,
            input=stdin_text,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        record.update(returncode=proc.returncode, stdout=proc.stdout or "",
                      stderr=proc.stderr or "")
    except subprocess.TimeoutExpired as exc:
        record.update(timed_out=True, error="timeout",
                      stdout=_as_text(exc.stdout), stderr=_as_text(exc.stderr))
    except (OSError, ValueError) as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        record["duration_seconds"] = time.monotonic() - started
    return record


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(value)


def run_text(command: Sequence[str], *, cwd: Path | None = None,
             timeout: float = 60.0,
             stdin_text: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(part) for part in command],
        cwd=str(cwd) if cwd else None,
        input=stdin_text,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )


def terminate_process_tree(pid: int) -> None:
    try:
        if os.name == "nt":
            run_command(["taskkill", "/PID", str(pid), "/T", "/F"], timeout=30)
        else:
            run_command(["pkill", "-TERM", "-P", str(pid)], timeout=30)
            os.kill(pid, 15)
    except (OSError, ProcessLookupError):
        pass


def process_start_token(pid: int) -> str | None:
    try:
        if os.name == "nt":
            proc = run_text(
                ["powershell", "-NoProfile", "-Command",
                 f"(Get-Process -Id {int(pid)} -ErrorAction Stop).StartTime.ToUniversalTime().ToString('o')"],
                timeout=20)
            text = (proc.stdout or "").strip()
            return text or None
        stat = Path(f"/proc/{int(pid)}/stat").read_text(encoding="utf-8", errors="replace")
        fields = stat.rsplit(")", 1)[-1].split()
        return fields[19] if len(fields) > 19 else None
    except Exception:
        return None


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        proc = run_text(["tasklist", "/FO", "CSV", "/NH", "/FI", f"PID eq {int(pid)}"],
                        timeout=20)
        return str(pid) in (proc.stdout or "")
    try:
        os.kill(pid, 0)
        return True
    except OSError as exc:
        return exc.errno == errno.EPERM


PINNED_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


def pinned_thread_env(base: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(base or os.environ)
    for key, value in PINNED_THREAD_ENV.items():
        env.setdefault(key, value)
    return env


def environment_record() -> dict[str, Any]:
    record: dict[str, Any] = {
        "captured_at": utc_now(),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "platform_release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "hostname": platform.node(),
        "cwd": os.getcwd(),
        "cpu_count": os.cpu_count(),
        "env_thread_policy": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "toolchain": {},
    }
    for tool in ("nvcc", "cl", "gcc", "cc", "compute-sanitizer", "nvidia-smi",
                 "git", "cl", "robocopy"):
        path = which(tool)
        record["toolchain"][tool] = {"path": path, "available": bool(path)}
    for tool, args in (("nvcc", ["--version"]), ("git", ["--version"])):
        path = record["toolchain"][tool]["path"]
        if path:
            record["toolchain"][tool]["version_text"] = run_command([path, *args],
                                                                    timeout=60)["stdout"].strip()
    if record["toolchain"].get("nvidia-smi", {}).get("available"):
        record["gpu"] = nvidia_smi_snapshot()
    try:
        import numpy
        import numpy as np
        record["numpy_version"] = np.__version__
        record["numpy_config"] = {
            "blas": _numpy_blas_name(),
            "float64_matmul_backend": "numpy.matmul(dtype=float64): BLAS dgemm",
        }
    except Exception as exc:
        record["numpy_version"] = None
        record["numpy_error"] = f"{type(exc).__name__}: {exc}"
    return record


def _numpy_blas_name() -> str | None:
    try:
        cfg = __import__("numpy").show_config(mode="dicts")
        blas = cfg.get("Build Dependencies", {}).get("blas", {})
        return f"{blas.get('name')} {blas.get('version', '')}".strip() or None
    except Exception:
        return None


def nvidia_smi_snapshot() -> dict[str, Any] | None:
    smi = which("nvidia-smi")
    if not smi:
        return None
    query = run_command(
        [smi, "--query-gpu=name,driver_version,memory.total,clocks.sm,"
              "temperature.gpu,power.draw,persistence_mode,compute_mode",
         "--format=csv,noheader,nounits"], timeout=30)
    record: dict[str, Any] = {"raw": query["stdout"].strip(), "returncode": query["returncode"]}
    text = query["stdout"].strip()
    if text:
        parts = [p.strip() for p in text.splitlines()[0].split(",")]
        keys = ["name", "driver_version", "memory_total_mib", "clock_sm_mhz",
                "temperature_c", "power_draw_w", "persistence_mode", "compute_mode"]
        record.update(dict(zip(keys, parts)))
    apps = run_command(
        [smi, "--query-compute-apps=pid,process_name,used_memory",
         "--format=csv,noheader"], timeout=30)
    record["compute_apps_raw"] = apps["stdout"].strip()
    record["other_compute_processes"] = bool(apps["stdout"].strip())
    return record


def copy_tree(src: Path, dst: Path, *, ignore: Sequence[str] = ()) -> Path:
    src, dst = Path(src), Path(dst)
    ensure_dir(dst.parent)

    def _ignore(directory: str, names: list[str]) -> set[str]:
        return {name for name in names if name in set(ignore)}

    shutil.copytree(src, dst, ignore=_ignore, dirs_exist_ok=False)
    return dst


def iter_files(root: Path) -> Iterator[Path]:
    root = Path(root)
    for path in sorted(root.rglob("*")):
        if path.is_file():
            yield path
