# llm/model_loader.py


from gecco.load_llms.provider_registry import get_provider_spec


def load_llm(provider: str, model_name: str, **kwargs):
    """Return a ``(model, tokenizer)`` tuple for an exact provider key."""
    provider_spec = get_provider_spec(provider)
    return provider_spec.loader(model_name, **kwargs)
