
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import InfraError
from .util import (ensure_dir, nvidia_smi_snapshot, pid_alive, process_start_token,
                   read_json, utc_now, write_json)


@dataclass
class LockRecord:
    path: Path
    device: int
    holder: dict[str, Any]
    acquired: bool = False
    waited_seconds: float = 0.0
    events: list[dict[str, Any]] = field(default_factory=list)

    def as_record(self) -> dict[str, Any]:
        return {"path": str(self.path), "device": self.device,
                "acquired": self.acquired,
                "waited_seconds": round(self.waited_seconds, 3),
                "holder": self.holder, "events": self.events}


class DeviceLock:

    def __init__(self, lock_dir: Path, *, device: int = 0,
                 timeout_seconds: float = 7200.0, poll_seconds: float = 5.0,
                 purpose: str = "unspecified") -> None:
        self.lock_dir = ensure_dir(Path(lock_dir))
        self.device = int(device)
        self.timeout_seconds = float(timeout_seconds)
        self.poll_seconds = float(poll_seconds)
        self.purpose = purpose
        self.path = self.lock_dir / f"gpu{self.device}.lock.json"
        self.record = LockRecord(path=self.path, device=self.device, holder={})
        self._heartbeat: threading.Thread | None = None
        self._stop = threading.Event()
        self._token = uuid.uuid4().hex

    def _payload(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "pid": os.getpid(),
            "start_token": process_start_token(os.getpid()),
            "host": os.uname().nodename if hasattr(os, "uname") else os.environ.get("COMPUTERNAME"),
            "task_id": os.environ.get("SCOPE_EXP_TASK_ID"),
            "attempt_id": os.environ.get("SCOPE_EXP_ATTEMPT_ID"),
            "purpose": self.purpose,
            "token": self._token,
            "acquired_at": utc_now(),
            "heartbeat_at": utc_now(),
        }

    def _try_create(self) -> bool:
        try:
            handle = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            return False
        except OSError as exc:
            raise InfraError(f"cannot create device lock {self.path}: {exc}",
                             stage="gpu_lock") from exc
        payload = json.dumps(self._payload(), ensure_ascii=False)
        try:
            os.write(handle, payload.encode("utf-8"))
            os.fsync(handle)
        finally:
            os.close(handle)
        return True

    def _read_holder(self) -> dict[str, Any] | None:
        try:
            return read_json(self.path)
        except FileNotFoundError:
            return None
        except (json.JSONDecodeError, OSError):
            return {"_unreadable": True}

    def _holder_state(self, holder: dict[str, Any]) -> str:
        if holder.get("_unreadable"):
            return "unverifiable"
        pid = int(holder.get("pid") or 0)
        if pid <= 0:
            return "unverifiable"
        if pid == os.getpid() and holder.get("token") == self._token:
            return "self"
        if not pid_alive(pid):
            return "dead"
        recorded = holder.get("start_token")
        current = process_start_token(pid)
        if recorded and current and recorded != current:
            return "recycled"
        if recorded and not current:
            return "unverifiable"
        return "alive"

    def _break_stale(self, holder: dict[str, Any], state: str) -> None:
        stale = self.path.with_suffix(f".stale.{int(time.time() * 1000)}")
        try:
            os.replace(self.path, stale)
        except OSError:
            return
        write_json(stale.with_suffix(".json"), {
            "reason": f"stale device lock ({state})",
            "holder": holder,
            "broken_by_pid": os.getpid(),
            "broken_at": utc_now(),
        })
        self.record.events.append({"event": "broke_stale_lock", "state": state,
                                   "holder": holder, "at": utc_now()})

    def acquire(self) -> "DeviceLock":
        started = time.monotonic()
        while True:
            if self._try_create():
                self.record.acquired = True
                self.record.holder = self._payload()
                self.record.waited_seconds = time.monotonic() - started
                self.record.events.append({"event": "acquired", "at": utc_now()})
                self._start_heartbeat()
                return self
            holder = self._read_holder() or {}
            state = self._holder_state(holder)
            if state in ("dead", "recycled"):
                self._break_stale(holder, state)
                continue
            if state == "self":
                self.record.acquired = True
                self.record.holder = holder
                return self
            if state == "unverifiable":
                raise InfraError(
                    f"device lock {self.path} is held by a process that cannot be "
                    f"verified: {holder}. Refusing to steal a live lock; remove the "
                    "file manually only after confirming no kernel is running.",
                    stage="gpu_lock", retryable=False)
            if time.monotonic() - started > self.timeout_seconds:
                raise InfraError(
                    f"device {self.device} stayed locked for "
                    f"{self.timeout_seconds:.0f}s by {holder}", stage="gpu_lock",
                    retryable=True, details={"holder": holder})
            self.record.events.append({"event": "waiting", "holder": holder,
                                       "at": utc_now()})
            time.sleep(self.poll_seconds)

    def _start_heartbeat(self) -> None:
        def _beat() -> None:
            while not self._stop.wait(max(30.0, self.poll_seconds * 6)):
                try:
                    holder = self._read_holder() or {}
                    if holder.get("token") != self._token:
                        return
                    holder["heartbeat_at"] = utc_now()
                    write_json(self.path, holder)
                except Exception:
                    return

        self._heartbeat = threading.Thread(target=_beat, name="scope-gpu-heartbeat",
                                          daemon=True)
        self._heartbeat.start()

    def release(self) -> None:
        self._stop.set()
        if self._heartbeat is not None:
            self._heartbeat.join(timeout=2.0)
        holder = self._read_holder() or {}
        if holder.get("token") == self._token:
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass
            self.record.events.append({"event": "released", "at": utc_now()})
        self.record.acquired = False

    def __enter__(self) -> "DeviceLock":
        return self.acquire()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


def device_state(device: int = 0) -> dict[str, Any]:
    snapshot = nvidia_smi_snapshot()
    return {
        "device": device,
        "snapshot": snapshot,
        "other_compute_processes": bool(snapshot and snapshot.get("other_compute_processes")),
        "captured_at": utc_now(),
    }


def assert_device_quiet(device: int = 0, *, allow_contended: bool = False) -> dict[str, Any]:
    state = device_state(device)
    if state["other_compute_processes"] and not allow_contended:
        raise InfraError(
            f"device {device} has another compute process; strict timing requires an "
            "exclusive GPU. Re-run when the device is quiet or pass "
            "--allow-contended to record the contention explicitly.",
            stage="timing", retryable=True, details=state)
    return state
