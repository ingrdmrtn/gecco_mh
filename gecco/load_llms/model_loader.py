# llm/model_loader.py

def load_llm(provider: str, model_name: str, **kwargs):
    """Return (model, tokenizer) tuple for any provider."""
    provider = provider.lower()

    if provider in {"openai", "gpt"}:
        from gecco.load_llms.gpt_backend import load_gpt
        model = load_gpt(model_name)
        tokenizer = None
    elif "r1" in provider:
        from gecco.load_llms.r1_backend import load_r1
        tokenizer, model = load_r1(model_name)
    elif "qwen" in provider:
        from gecco.load_llms.qwen_backend import load_qwen
        tokenizer, model = load_qwen(model_name)
    elif "llama" in provider:
        from gecco.load_llms.llama_backend import load_llama
        tokenizer, model = load_llama(model_name)
    elif "gemini" in provider:
        from gecco.load_llms.gemini_backend import load_gemini
        model = load_gemini(model_name)
        tokenizer = None
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")

    return model, tokenizer
