from __future__ import annotations

import logging
import json
import os
import threading
import time
from dataclasses import dataclass
from typing import Any

from SCOPE.llm.base import LLMClient
from SCOPE.llm.provider_config import resolve_provider_config
from SCOPE.utils.execution import limited
from openai import OpenAI
import httpx

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 120
DEFAULT_SELECTION_TIMEOUT_SECONDS = 30
DEFAULT_CODEGEN_TIMEOUT_SECONDS = 300
DEFAULT_MAX_RETRIES = 3
DEFAULT_SELECTION_MAX_RETRIES = 1


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    model: str
    api_key: str
    base_url: str
    temperature: float = 0.1
    timeout_seconds: int = 120
    max_retries: int = 3


class OpenAICompatibleClient(LLMClient):
    def __init__(self, config):
        config = resolve_provider_config(config)
        api_key = resolve_api_key(config)
        if not api_key:
            env_name = config.get("api_key_env") or "configured environment variable"
            raise RuntimeError(
                f"Missing API key for {config.get('provider', 'configured provider')}. "
                f"Set api_key/api_key_env in the selected llm provider configuration or environment variable {env_name}."
            )

        self._config = config
        self._timeout_seconds = numeric_config(
            config,
            "timeout_seconds",
            fallback_key="request_timeout_seconds",
            default=DEFAULT_TIMEOUT_SECONDS,
        )
        self._codegen_timeout_seconds = numeric_config(
            config,
            "codegen_timeout_seconds",
            default=max(DEFAULT_CODEGEN_TIMEOUT_SECONDS, self._timeout_seconds),
        )
        self._selection_timeout_seconds = numeric_config(
            config,
            "selection_timeout_seconds",
            default=min(DEFAULT_SELECTION_TIMEOUT_SECONDS, self._timeout_seconds),
        )
        self._max_retries = int(config.get("max_retries", DEFAULT_MAX_RETRIES))
        self._selection_max_retries = int(
            config.get("selection_max_retries", min(DEFAULT_SELECTION_MAX_RETRIES, self._max_retries))
        )
        self._max_tokens = int(config.get("max_tokens", 4096))
        self._selection_max_tokens = int(config.get("selection_max_tokens", min(2048, self._max_tokens)))
        self._codegen_max_tokens = int(config.get("codegen_max_tokens", self._max_tokens))
        self._use_json_response_format = bool(config.get("use_json_response_format", True))
        self._thinking_mode = normalize_thinking_mode(config.get("thinking", "disabled"))
        self._selection_thinking_mode = normalize_thinking_mode(
            config.get("selection_thinking", self._thinking_mode)
        )
        self._codegen_thinking_mode = normalize_thinking_mode(
            config.get("codegen_thinking", self._thinking_mode)
        )
        self._reasoning_effort = config.get("reasoning_effort")
        self._selection_reasoning_effort = config.get("selection_reasoning_effort", self._reasoning_effort)
        self._codegen_reasoning_effort = config.get("codegen_reasoning_effort", self._reasoning_effort)
        client_options = {}
        if "trust_env" in config:
            client_options["http_client"] = httpx.Client(trust_env=bool(config.get("trust_env")))
        self._client = OpenAI(
            api_key=api_key,
            base_url=config.get("base_url"),
            timeout=self._timeout_seconds,
            max_retries=self._max_retries,
            **client_options,
        )
        self._usage_lock = threading.Lock()
        self._usage = {
            "request_count": 0,
            "successful_request_count": 0,
            "failed_request_count": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "reasoning_tokens": 0,
            "total_tokens": 0,
            "request_seconds": 0.0,
            "by_call_type": {},
        }

    @property
    def timeout_seconds(self) -> float:
        return self._timeout_seconds

    @property
    def codegen_timeout_seconds(self) -> float:
        return self._codegen_timeout_seconds

    @property
    def selection_timeout_seconds(self) -> float:
        return self._selection_timeout_seconds

    @limited("llm")
    def complete_text(self, messages: list[dict[str, str]], timeout_seconds: float | None = None) -> str:
        response = self._tracked_request("text", lambda: self._client.chat.completions.create(**self._completion_kwargs(
            messages, timeout_seconds or self._timeout_seconds, self._max_tokens,
            json_mode=False, thinking_mode=self._thinking_mode, reasoning_effort=self._reasoning_effort,
        )))
        return require_final_content(response)

    @limited("llm")
    def complete_json(self, messages: list[dict[str, str]], timeout_seconds: float | None = None) -> dict[str, Any]:
        response = self._tracked_request("json", lambda: self._client.chat.completions.create(**self._completion_kwargs(
            messages, timeout_seconds or self._timeout_seconds, self._max_tokens,
            json_mode=True, thinking_mode=self._thinking_mode, reasoning_effort=self._reasoning_effort,
        )))
        content = require_final_content(response)
        logger.info("LLM response received: %d chars", len(content))
        return extract_json_object(content)

    @limited("llm")
    def complete_tile_plan_text(self, messages):
        """Expose raw JSON-mode responses for local planning diagnostics."""
        response = self._tracked_request("tile_plan", lambda: self._client.chat.completions.create(**self._completion_kwargs(
            messages, self._selection_timeout_seconds, self._selection_max_tokens,
            json_mode=True, thinking_mode=self._selection_thinking_mode,
            reasoning_effort=self._selection_reasoning_effort,
        )))
        return require_final_content(response)

    @limited("llm")
    def complete_selection_json(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        response = self._tracked_request("selection", lambda: self._client.with_options(
            max_retries=self._selection_max_retries
        ).chat.completions.create(**self._completion_kwargs(
            messages, self._selection_timeout_seconds, self._selection_max_tokens,
            json_mode=True, thinking_mode=self._selection_thinking_mode,
            reasoning_effort=self._selection_reasoning_effort,
        )))
        content = require_final_content(response)
        logger.info("LLM selection response received: %d chars", len(content))
        return extract_json_object(content)

    @limited("llm")
    def complete_codegen_json(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        response = self._tracked_request("codegen_json", lambda: self._client.chat.completions.create(**self._completion_kwargs(
            messages, self._codegen_timeout_seconds, self._codegen_max_tokens,
            json_mode=True, thinking_mode=self._codegen_thinking_mode,
            reasoning_effort=self._codegen_reasoning_effort,
        )))
        content = require_final_content(response)
        logger.info("LLM codegen response received: %d chars", len(content))
        return extract_json_object(content)

    @limited("llm")
    def complete_codegen_text(self, messages: list[dict[str, str]]) -> str:
        response = self._tracked_request("codegen_text", lambda: self._client.chat.completions.create(**self._completion_kwargs(
            messages, self._codegen_timeout_seconds, self._codegen_max_tokens,
            json_mode=False, thinking_mode=self._codegen_thinking_mode,
            reasoning_effort=self._codegen_reasoning_effort,
        )))
        return require_final_content(response)

    def _tracked_request(self, call_type: str, create):
        started = time.perf_counter()
        try:
            response = create()
        except Exception:
            self._record_usage(call_type, None, time.perf_counter() - started, failed=True)
            raise
        self._record_usage(call_type, response, time.perf_counter() - started, failed=False)
        return response

    def _record_usage(self, call_type: str, response, elapsed: float, *, failed: bool) -> None:
        usage = getattr(response, "usage", None) if response is not None else None
        details = getattr(usage, "completion_tokens_details", None)
        values = {
            "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
            "reasoning_tokens": int(getattr(details, "reasoning_tokens", 0) or 0),
            "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
        }
        with self._usage_lock:
            self._usage["request_count"] += 1
            self._usage["failed_request_count" if failed else "successful_request_count"] += 1
            self._usage["request_seconds"] += float(elapsed)
            for key, value in values.items():
                self._usage[key] += value
            per_type = self._usage["by_call_type"].setdefault(call_type, {
                "request_count": 0, "failed_request_count": 0, "request_seconds": 0.0,
                "prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0, "total_tokens": 0,
            })
            per_type["request_count"] += 1
            per_type["failed_request_count"] += int(failed)
            per_type["request_seconds"] += float(elapsed)
            for key, value in values.items():
                per_type[key] += value

    def usage_snapshot(self) -> dict[str, Any]:
        with self._usage_lock:
            result = json.loads(json.dumps(self._usage))
        result["request_seconds"] = round(result["request_seconds"], 6)
        for item in result["by_call_type"].values():
            item["request_seconds"] = round(item["request_seconds"], 6)
        result["usage_available"] = bool(result["total_tokens"] or result["prompt_tokens"] or result["completion_tokens"])
        return result

    def merge_usage_snapshot(self, other: dict[str, Any]) -> None:
        with self._usage_lock:
            for key in (
                "request_count", "successful_request_count", "failed_request_count",
                "prompt_tokens", "completion_tokens", "reasoning_tokens", "total_tokens",
                "request_seconds",
            ):
                value = other.get(key, 0)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    self._usage[key] += value
            for call_type, incoming in (other.get("by_call_type", {}) or {}).items():
                target = self._usage["by_call_type"].setdefault(call_type, {})
                for key, value in incoming.items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        target[key] = target.get(key, 0) + value

    def _completion_kwargs(
        self,
        messages: list[dict[str, str]],
        timeout_seconds: float,
        max_tokens: int,
        *,
        json_mode: bool,
        thinking_mode: str | None,
        reasoning_effort: str | None = None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self._config["model"],
            "messages": messages,
            "temperature": float(self._config.get("temperature", 0.1)),
            "max_tokens": max_tokens,
            "timeout": timeout_seconds,
        }
        if json_mode and self._use_json_response_format:
            kwargs["response_format"] = {"type": "json_object"}
        if thinking_mode:
            kwargs["extra_body"] = {"thinking": {"type": thinking_mode}}
        if reasoning_effort is not None:
            kwargs.setdefault("extra_body", {})["reasoning_effort"] = reasoning_effort
        return kwargs


def require_final_content(response) -> str:
    choices = getattr(response, "choices", None)
    if not choices:
        raise RuntimeError("LLM response has no choices; no final answer was returned.")
    choice = choices[0]
    finish = getattr(choice, "finish_reason", None)
    usage = getattr(response, "usage", None)
    details = getattr(usage, "completion_tokens_details", None)
    counts = (f"completion_tokens={getattr(usage, 'completion_tokens', None)}, "
              f"reasoning_tokens={getattr(details, 'reasoning_tokens', None)}")
    if finish == "length":
        raise RuntimeError(f"LLM output truncated (finish_reason=length; {counts}). "
                           "Increase this call's max_tokens or reduce reasoning_effort; this is not an invalid strategy selection.")
    content = getattr(choice.message, "content", None)
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError(f"LLM returned empty final content (finish_reason={finish}; {counts}). "
                           "Reasoning text is not a final JSON answer.")
    return content


def resolve_api_key(config: dict[str, Any]) -> str:
    direct = str(config.get("api_key") or "").strip()
    if direct.startswith("${") and direct.endswith("}"):
        return os.environ.get(direct[2:-1].strip(), "").strip()
    if direct:
        return direct
    env_name = str(config.get("api_key_env") or "").strip()
    if not env_name:
        return ""
    from_environment = os.environ.get(env_name, "").strip()
    if from_environment:
        return from_environment
    if len(env_name) > 32:
        logger.warning("llm.api_key_env appears to contain a key; use llm.api_key or an environment variable name")
        return env_name
    return ""


def normalize_thinking_mode(value: Any) -> str | None:
    if isinstance(value, bool):
        return "enabled" if value else "disabled"
    text = str(value or "").strip().lower()
    if text in {"enabled", "disabled"}:
        return text
    return None


def numeric_config(
    config: dict[str, Any],
    key: str,
    default: float,
    fallback_key: str | None = None,
) -> float:
    value = config.get(key)
    if value is None and fallback_key:
        value = config.get(fallback_key)
    if value is None:
        return float(default)
    return float(value)


def extract_json_object(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("LLM response does not contain a JSON object.")
    return json.loads(text[start : end + 1])
