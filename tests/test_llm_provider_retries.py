"""Contract tests for OpenAI-compatible provider retry handling."""

from __future__ import annotations

import json

import pytest

from gecco.llm_provider_retries import LLMProviderRetryError, retry_llm_provider_call


def test_retry_helper_retries_json_decode_error_then_succeeds():
    calls = {"count": 0}

    def _call():
        calls["count"] += 1
        if calls["count"] == 1:
            raise json.JSONDecodeError("Expecting value", "", 0)
        return "ok"

    result = retry_llm_provider_call(
        _call,
        provider="openrouter",
        model="gpt-test",
        operation="candidate generation",
        attempts=3,
        backoff_seconds=0.0,
    )

    assert result == "ok"
    assert calls["count"] == 2


def test_retry_helper_raises_context_without_prompt_content():
    prompt_text = "secret prompt content"

    def _call():
        raise json.JSONDecodeError("Expecting value", prompt_text, 0)

    with pytest.raises(LLMProviderRetryError) as exc_info:
        retry_llm_provider_call(
            _call,
            provider="openrouter",
            model="gpt-test",
            operation="judge structured verdict synthesis",
            attempts=1,
            backoff_seconds=0.0,
        )

    message = str(exc_info.value)
    assert "openrouter" in message
    assert "gpt-test" in message
    assert "judge structured verdict synthesis" in message
    assert "JSONDecodeError" in message
    assert prompt_text not in message


def test_retry_helper_respects_attempts_one_fail_fast():
    calls = {"count": 0}

    def _call():
        calls["count"] += 1
        raise json.JSONDecodeError("Expecting value", "", 0)

    with pytest.raises(LLMProviderRetryError):
        retry_llm_provider_call(
            _call,
            provider="openrouter",
            model="gpt-test",
            operation="fallback generation",
            attempts=1,
            backoff_seconds=0.0,
        )

    assert calls["count"] == 1
