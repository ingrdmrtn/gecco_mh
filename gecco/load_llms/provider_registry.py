"""Canonical provider registry for GeCCo LLM backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


Loader = Callable[..., tuple[Any, Any]]


@dataclass(frozen=True, slots=True)
class ProviderSpec:
    """Metadata for one exact provider key."""

    key: str
    label: str
    api_family: str
    prompt_family: str
    structured_output_mode: str
    supports_system_prompt: bool
    loader: Loader


def _load_openai_client(model_name: str, **kwargs) -> tuple[Any, Any]:
    from gecco.load_llms.gpt_backend import load_gpt

    return load_gpt(model_name), None


def _load_gemini_client(model_name: str, **kwargs) -> tuple[Any, Any]:
    from gecco.load_llms.gemini_backend import load_gemini

    return load_gemini(model_name), None


def _load_vllm_client(model_name: str, **kwargs) -> tuple[Any, Any]:
    from gecco.load_llms.vllm_backend import load_vllm

    return load_vllm(model_name), None


def _load_kcl_client(model_name: str, **kwargs) -> tuple[Any, Any]:
    from gecco.load_llms.kcl_backend import load_kcl

    return load_kcl(model_name), None


def _load_openrouter_client(model_name: str, **kwargs) -> tuple[Any, Any]:
    from gecco.load_llms.openrouter_backend import load_openrouter

    return load_openrouter(model_name), None


def _load_opencode_go_client(model_name: str, **kwargs) -> tuple[Any, Any]:
    from gecco.load_llms.opencode_backend import load_opencode

    return load_opencode(model_name, base_url=kwargs.get("base_url")), None


def _load_hf_client(loader: Callable[[str], tuple[Any, Any]], model_name: str, **kwargs) -> tuple[Any, Any]:
    tokenizer, model = loader(model_name)
    return model, tokenizer


def _load_qwen_client(model_name: str, **kwargs) -> tuple[Any, Any]:
    from gecco.load_llms.qwen_backend import load_qwen

    return _load_hf_client(load_qwen, model_name, **kwargs)


def _load_r1_client(model_name: str, **kwargs) -> tuple[Any, Any]:
    from gecco.load_llms.r1_backend import load_r1

    return _load_hf_client(load_r1, model_name, **kwargs)


def _load_llama_client(model_name: str, **kwargs) -> tuple[Any, Any]:
    from gecco.load_llms.llama_backend import load_llama

    return _load_hf_client(load_llama, model_name, **kwargs)


PROVIDER_REGISTRY: dict[str, ProviderSpec] = {
    "openai": ProviderSpec(
        key="openai",
        label="OpenAI",
        api_family="openai",
        prompt_family="closed",
        structured_output_mode="responses_api",
        supports_system_prompt=True,
        loader=_load_openai_client,
    ),
    "gemini": ProviderSpec(
        key="gemini",
        label="Gemini",
        api_family="gemini",
        prompt_family="closed",
        structured_output_mode="response_schema",
        supports_system_prompt=True,
        loader=_load_gemini_client,
    ),
    "kcl": ProviderSpec(
        key="kcl",
        label="KCL",
        api_family="openai_compatible",
        prompt_family="closed",
        structured_output_mode="chat_json_object",
        supports_system_prompt=True,
        loader=_load_kcl_client,
    ),
    "vllm": ProviderSpec(
        key="vllm",
        label="vLLM",
        api_family="openai_compatible",
        prompt_family="open",
        structured_output_mode="chat_json_object",
        supports_system_prompt=True,
        loader=_load_vllm_client,
    ),
    "openrouter": ProviderSpec(
        key="openrouter",
        label="OpenRouter",
        api_family="openai_compatible",
        prompt_family="open",
        structured_output_mode="chat_json_schema_optional",
        supports_system_prompt=True,
        loader=_load_openrouter_client,
    ),
    "opencode-go": ProviderSpec(
        key="opencode-go",
        label="OpenCode Zen",
        api_family="openai_compatible",
        prompt_family="open",
        structured_output_mode="chat_json_object",
        supports_system_prompt=True,
        loader=_load_opencode_go_client,
    ),
    "qwen": ProviderSpec(
        key="qwen",
        label="Qwen",
        api_family="hf",
        prompt_family="open",
        structured_output_mode="schema_instructions",
        supports_system_prompt=False,
        loader=_load_qwen_client,
    ),
    "r1": ProviderSpec(
        key="r1",
        label="R1-Distilled",
        api_family="hf",
        prompt_family="open",
        structured_output_mode="schema_instructions",
        supports_system_prompt=False,
        loader=_load_r1_client,
    ),
    "llama": ProviderSpec(
        key="llama",
        label="LLaMA",
        api_family="hf",
        prompt_family="open",
        structured_output_mode="schema_instructions",
        supports_system_prompt=False,
        loader=_load_llama_client,
    ),
}


def get_registered_provider_keys() -> tuple[str, ...]:
    """Return the canonical provider keys in registry order."""
    return tuple(PROVIDER_REGISTRY.keys())


def get_provider_spec(provider: str) -> ProviderSpec:
    """Resolve one exact provider key to its canonical registry entry."""
    try:
        return PROVIDER_REGISTRY[provider]
    except KeyError as exc:
        raise ValueError(
            f"Unknown LLM provider: {provider}. Registered LLM providers: "
            f"{', '.join(get_registered_provider_keys())}"
        ) from exc
