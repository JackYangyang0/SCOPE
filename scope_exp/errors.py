from __future__ import annotations

from typing import Any


class ScopeExpError(Exception):
    pass


class ConfigError(ScopeExpError):
    pass


class ProtocolError(ScopeExpError):
    pass


class GateError(ScopeExpError):
    pass


class InfraError(ScopeExpError): 
    def __init__(self, message: str, *, stage: str | None = None,
                 retryable: bool = True, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.stage = stage
        self.retryable = retryable
        self.details = details or {}

    def as_record(self) -> dict[str, Any]:
        return {
            "kind": "infrastructure",
            "message": str(self),
            "stage": self.stage,
            "retryable": self.retryable,
            "details": self.details,
        }


class ScienceFailure(ScopeExpError):
    def __init__(self, message: str, *, category: str,
                 details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.category = category
        self.details = details or {}

    def as_record(self) -> dict[str, Any]:
        return {
            "kind": "science_failure",
            "category": self.category,
            "message": str(self),
            "details": self.details,
        }

SCIENCE_CATEGORIES = (
    "no_candidate",
    "contract_violation", 
    "compile_failure", 
    "runtime_failure", 
    "numerical_failure",
    "safety_failure", 
    "budget_exhausted",
)

INFRA_CATEGORIES = (
    "api_unavailable",
    "api_timeout",
    "api_unknown_response",
    "runner_error",
    "disk_error",
    "permission_error",
    "external_interference",
    "toolchain_missing",
)
