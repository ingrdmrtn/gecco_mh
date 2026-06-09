"""Phase 8 contract tests for provider registry dispatch."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from config.schema import load_config
from gecco.load_llms import model_loader
from gecco.load_llms import provider_registry
from gecco.run_gecco import GeCCoModelSearch


EXPECTED_PROVIDER_KEYS = {
    "gemini",
    "kcl",
    "llama",
    "openai",
    "opencode-go",
    "openrouter",
    "qwen",
    "r1",
    "vllm",
}


class _FakeResponses:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(output_text="openai response")


class _FakeChatCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="openrouter response"))]
        )


class _FakeModel:
    def __init__(self):
        self.responses = _FakeResponses()
        self.chat = SimpleNamespace(completions=_FakeChatCompletions())


def _make_search(provider: str, **llm_overrides):
    llm = SimpleNamespace(
        provider=provider,
        base_model="test-model",
        temperature=0.1,
        max_output_tokens=64,
        system_prompt="stay concise",
        **llm_overrides,
    )
    search = GeCCoModelSearch.__new__(GeCCoModelSearch)
    search.cfg = SimpleNamespace(llm=llm)
    return search


def test_provider_registry_resolves_exact_supported_keys():
    """The registry should expose the canonical exact provider keys only."""
    assert set(provider_registry.get_registered_provider_keys()) == EXPECTED_PROVIDER_KEYS

    openrouter = provider_registry.get_provider_spec("openrouter")
    assert openrouter.key == "openrouter"
    assert openrouter.label == "OpenRouter"
    assert openrouter.api_family == "openai_compatible"
    assert openrouter.structured_output_mode == "chat_json_schema_optional"
    assert openrouter.supports_system_prompt is True


def test_provider_registry_rejects_substring_provider_keys():
    """Substring-like provider names should fail with a registry error."""
    with pytest.raises(ValueError, match="Registered LLM providers"):
        provider_registry.get_provider_spec("my-openrouter-proxy")


def test_provider_registry_rejects_case_variant_provider_keys():
    """Case variants must fail because provider keys are exact registry keys."""
    with pytest.raises(ValueError, match="Registered LLM providers"):
        provider_registry.get_provider_spec("OpenAI")


def test_load_llm_uses_exact_registered_loader(monkeypatch):
    """load_llm() should delegate to the exact registered loader callable."""
    calls = []

    def fake_loader(model_name: str, **kwargs):
        calls.append((model_name, kwargs))
        return ("loaded-model", "loaded-tokenizer")

    monkeypatch.setitem(
        provider_registry.PROVIDER_REGISTRY,
        "opencode-go",
        provider_registry.ProviderSpec(
            key="opencode-go",
            label="OpenCode Zen",
            api_family="openai_compatible",
            prompt_family="open",
            structured_output_mode="chat_json_object",
            supports_system_prompt=True,
            loader=fake_loader,
        ),
    )

    model, tokenizer = model_loader.load_llm(
        "opencode-go",
        "test-model",
        base_url="https://example.invalid/v1",
    )

    assert (model, tokenizer) == ("loaded-model", "loaded-tokenizer")
    assert calls == [
        (
            "test-model",
            {"base_url": "https://example.invalid/v1"},
        )
    ]


def test_load_llm_does_not_route_opencode_go_by_substring(monkeypatch):
    """A substring-like provider name must not fall through to opencode-go."""
    monkeypatch.setitem(
        provider_registry.PROVIDER_REGISTRY,
        "opencode-go",
        provider_registry.ProviderSpec(
            key="opencode-go",
            label="OpenCode Zen",
            api_family="openai_compatible",
            prompt_family="open",
            structured_output_mode="chat_json_object",
            supports_system_prompt=True,
            loader=lambda *args, **kwargs: ("loaded-model", "loaded-tokenizer"),
        ),
    )

    with pytest.raises(ValueError, match="Registered LLM providers"):
        model_loader.load_llm("my-opencode-go-proxy", "test-model")


def test_generate_uses_provider_api_family_for_openrouter_chat():
    """OpenRouter should route through the OpenAI-compatible chat branch."""
    search = _make_search("openrouter", supports_json_schema=True)
    model = _FakeModel()

    result = search.generate(
        model=model,
        tokenizer=None,
        prompt="Explain the result.",
        response_schema={"type": "object", "properties": {"answer": {"type": "string"}}},
        system_prompt="Use the developer prompt.",
    )

    assert result == "openrouter response"
    assert model.responses.calls == []
    assert len(model.chat.completions.calls) == 1
    call = model.chat.completions.calls[0]
    assert call["model"] == "test-model"
    assert call["messages"][0]["role"] == "system"
    assert "response_format" in call
    assert call["extra_body"] == {"provider": {"require_parameters": True}}


def test_generate_rejects_unknown_provider_before_hf_fallback():
    """Unknown providers should fail before any HF-style fallback path can run."""
    search = _make_search("my-openrouter-proxy")

    with pytest.raises(ValueError, match="Registered LLM providers"):
        search.generate(
            model=_FakeModel(),
            tokenizer=None,
            prompt="Explain the result.",
            response_schema=None,
        )
