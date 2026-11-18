
from __future__ import annotations

import importlib.abc
import importlib.machinery
import importlib.util
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

_INSTALLED = False
_LOCK = threading.Lock()
_COUNTER = {"ordinal": 0}
_STATE: dict[str, Any] = {"enabled": False, "log": None, "error": None,
                          "patched": False, "reason": None}

TARGET_MODULE = "llm.openai_client"
TARGET_MODULE_ALT = "SCOPE.llm.openai_client"


def _write(row: dict[str, Any]) -> None:
    path = _STATE.get("log")
    if not path:
        return
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
    except Exception as exc:
        _STATE["error"] = f"{type(exc).__name__}: {exc}"


def _tokens(response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    details = getattr(usage, "completion_tokens_details", None)
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
        "cached_tokens": getattr(details, "cached_tokens", None) if details else None,
        "reasoning_tokens": getattr(details, "reasoning_tokens", None) if details else None,
    }


def _patch(module: Any) -> None:
    cls = getattr(module, "OpenAICompatibleClient", None)
    if cls is None or getattr(cls, "_scope_exp_instrumented", False):
        _STATE["reason"] = "OpenAICompatibleClient not found or already patched"
        return
    original_tracked = getattr(cls, "_tracked_request", None)
    original_record = getattr(cls, "_record_usage", None)
    if original_tracked is None or original_record is None:
        _STATE["reason"] = "expected methods missing; instrumentation skipped"
        return

    def tracked_request(self, call_type, create, *args, **kwargs):
        started = time.perf_counter()
        with _LOCK:
            _COUNTER["ordinal"] += 1
            ordinal = _COUNTER["ordinal"]
        error = None
        try:
            return original_tracked(self, call_type, create, *args, **kwargs)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            _write({"kind": "request", "ordinal": ordinal, "call_type": call_type,
                    "started_at": time.time(), "duration_seconds": time.perf_counter() - started,
                    "failed": True, "error": error, "model": None, "tokens": {},
                    "phase": os.environ.get("SCOPE_EXP_LEDGER_PHASE")})
            raise
        finally:
            with _LOCK:
                _STATE.setdefault("requests", 0)
                _STATE["requests"] += 1

    def record_usage(self, call_type, response, elapsed, *, failed: bool, **kwargs):
        result = original_record(self, call_type, response, elapsed, failed=failed, **kwargs)
        with _LOCK:
            _COUNTER["ordinal"] += 1
            ordinal = _COUNTER["ordinal"]
        _write({"kind": "usage", "ordinal": ordinal, "call_type": call_type,
                "started_at": time.time(), "duration_seconds": float(elapsed),
                "failed": bool(failed), "model": getattr(response, "model", None),
                "request_id": getattr(response, "id", None),
                "tokens": _tokens(response),
                "phase": os.environ.get("SCOPE_EXP_LEDGER_PHASE")})
        return result

    cls._tracked_request = tracked_request
    cls._record_usage = record_usage
    cls._scope_exp_instrumented = True
    _STATE["patched"] = True


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname not in (TARGET_MODULE, TARGET_MODULE_ALT):
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return spec
        loader = spec.loader
        original_exec = getattr(loader, "exec_module", None)

        def exec_module(module, _original_exec=original_exec):
            _original_exec(module)
            try:
                _patch(module)
            except Exception as exc:
                _STATE["error"] = f"patch failed: {type(exc).__name__}: {exc}"

        loader.exec_module = exec_module
        return spec


def install(log_path: str | Path | None = None) -> dict[str, Any]:
    global _INSTALLED
    path = log_path or os.environ.get("SCOPE_EXP_LEDGER")
    if not path:
        _STATE["reason"] = "SCOPE_EXP_LEDGER is not set; instrumentation disabled"
        return dict(_STATE)
    _STATE["log"] = str(path)
    with _LOCK:
        if not _INSTALLED:
            sys.meta_path.insert(0, _Finder())
            _INSTALLED = True
        _STATE["enabled"] = True
    for name in (TARGET_MODULE, TARGET_MODULE_ALT):
        module = sys.modules.get(name)
        if module is not None and not _STATE["patched"]:
            _patch(module)
    return dict(_STATE)


def status() -> dict[str, Any]:
    return dict(_STATE)
