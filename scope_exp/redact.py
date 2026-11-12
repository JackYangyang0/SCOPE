
from __future__ import annotations

import os
import re
from typing import Any, Iterable

from .util import sha256_value

SECRET_EXACT = {
    "api_key", "apikey", "access_key", "secret", "token", "password", "passwd",
    "authorization", "auth_header", "bearer", "credential", "credentials",
    "private_key", "cookie", "session_id", "secret_key", "client_secret",
}
SECRET_SUFFIXES = ("_api_key", "api_key", "_access_key", "_secret", "_password",
                   "_passwd", "_credential", "_credentials", "_token",
                   "_private_key", "_auth_token")
NOT_SECRET = {
    "api_key_env", "api_key_env_var", "token_env", "auth_env",
    "max_tokens", "selection_max_tokens", "codegen_max_tokens", "total_tokens",
    "prompt_tokens", "completion_tokens", "cached_tokens", "reasoning_tokens",
    "token_budget", "tokens", "auth_method", "authorization_mode",
}

SECRET_KEY_PATTERN = re.compile(
    r"(api[_-]?key|apikey|access[_-]?key|secret|password|passwd|"
    r"authorization|auth[_-]?header|bearer|credential|private[_-]?key|"
    r"cookie|session[_-]?id)",
    re.IGNORECASE,
)

SECRET_NAME_KEYS = {"api_key_env", "api_key_env_var", "token_env", "auth_env"}

REDACTED = "<redacted>"

_CREDENTIAL_VALUE_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"\bghp_[A-Za-z0-9]{20,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\b[Aa]uthorization\s*[:=]\s*\S+"),
    re.compile(r"\bBearer\s+[A-Za-z0-9._\-]{20,}"),
    re.compile(r"(?i)api[_-]?key\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{16,}"),
)


def _is_secret_key(key: str) -> bool:
    lowered = str(key).strip().lower()
    if lowered in NOT_SECRET or lowered in SECRET_NAME_KEYS:
        return False
    if lowered in SECRET_EXACT:
        return True
    if lowered.endswith("_tokens"):
        return False
    if any(lowered.endswith(suffix) for suffix in SECRET_SUFFIXES):
        return True
    return bool(SECRET_KEY_PATTERN.search(lowered))


def is_secret_field(key: str, value: Any) -> bool:
    if isinstance(value, bool) or isinstance(value, (int, float)):
        return False
    if not isinstance(value, str):
        return _is_secret_key(key)
    return _is_secret_key(key)


def redact(value: Any, *, path: str = "", found: list[str] | None = None) -> Any:
    found = found if found is not None else []
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            if is_secret_field(str(key), item):
                if item not in ("", None, [], {}):
                    found.append(child_path)
                result[key] = REDACTED if item not in ("", None, [], {}) else ""
                continue
            result[key] = redact(item, path=child_path, found=found)
        return result
    if isinstance(value, list):
        return [redact(item, path=f"{path}[{index}]", found=found)
                for index, item in enumerate(value)]
    return value


def redact_config(config: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    found: list[str] = []
    redacted = redact(config, found=found)
    return redacted, sorted(found)


def live_credentials(env: dict[str, str] | None = None) -> list[str]:
    env = env if env is not None else os.environ
    values: list[str] = []
    for name, value in env.items():
        if not value or len(value) < 8:
            continue
        if SECRET_KEY_PATTERN.search(name):
            values.append(value)
    return values


def find_credentials(text: str, live: Iterable[str] = ()) -> list[str]:
    hits: list[str] = []
    for value in live:
        if len(value) >= 8 and value in text:
            hits.append(f"live credential value ({len(value)} chars) present")
    for pattern in _CREDENTIAL_VALUE_PATTERNS:
        match = pattern.search(text)
        if match:
            hits.append(f"pattern {pattern.pattern!r} matched {match.group(0)[:6]}...")
    return hits


def redacted_config_hash(redacted: dict[str, Any]) -> str:
    return sha256_value(redacted)


def assert_clean(payload: str, *, where: str, live: Iterable[str] = ()) -> None:
    hits = find_credentials(payload, live)
    if hits:
        raise ValueError(f"refusing to archive credentials from {where}: {hits}")
