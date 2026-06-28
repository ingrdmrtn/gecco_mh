"""Retry helper for OpenAI-compatible provider calls."""

from __future__ import annotations

import json
import time
from typing import Callable, TypeVar

T = TypeVar("T")

try:  # pragma: no cover - optional dependency import path
    from openai import (  # type: ignore
        APIConnectionError,
        APITimeoutError,
        InternalServerError,
        RateLimitError,
    )
except Exception:  # pragma: no cover - openai may be absent in tests
    _TRANSIENT_OPENAI_ERRORS: tuple[type[BaseException], ...] = ()
else:  # pragma: no cover - exercised only when openai is installed
    _TRANSIENT_OPENAI_ERRORS = (
        APIConnectionError,
        APITimeoutError,
        InternalServerError,
        RateLimitError,
    )


class LLMProviderRetryError(RuntimeError):
    """Raised when provider retries are exhausted."""


def retry_llm_provider_call(
    call: Callable[[], T],
    *,
    provider: str,
    model: str,
    operation: str,
    attempts: int = 3,
    backoff_seconds: float = 2.0,
) -> T:
    """Retry transient provider failures without exposing prompt content."""

    if attempts < 1:
        raise ValueError("attempts must be at least 1")

    transient_errors = (json.JSONDecodeError,) + _TRANSIENT_OPENAI_ERRORS
    last_exc: BaseException | None = None

    for attempt in range(1, attempts + 1):
        try:
            return call()
        except transient_errors as exc:
            last_exc = exc
            if attempt >= attempts:
                raise LLMProviderRetryError(
                    f"{operation} failed for provider={provider!r}, model={model!r} "
                    f"after {attempts} attempts; last error class: {exc.__class__.__name__}"
                ) from exc
            sleep_seconds = max(0.0, backoff_seconds) * (2 ** (attempt - 1))
            if sleep_seconds:
                time.sleep(sleep_seconds)

    assert last_exc is not None  # pragma: no cover - defensive only
    raise LLMProviderRetryError(
        f"{operation} failed for provider={provider!r}, model={model!r} "
        f"after {attempts} attempts; last error class: {last_exc.__class__.__name__}"
    ) from last_exc
