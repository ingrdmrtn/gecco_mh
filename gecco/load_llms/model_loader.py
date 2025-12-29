# llm/model_loader.py
from gecco.load_llms.llama_backend import load_llama
from gecco.load_llms.qwen_backend import load_qwen
from gecco.load_llms.r1_backend import load_r1
from gecco.load_llms.gpt_backend import load_gpt
from gecco.load_llms.gemini_backend import load_gemini

def load_llm(provider: str, model_name: str, **kwargs):
    """Return (model, tokenizer) tuple for any provider."""
    provider = provider.lower()

    if provider in {"openai", "gpt"}:
        model = load_gpt(model_name)
        tokenizer = None
    elif "r1" in provider:
        tokenizer, model = load_r1(model_name)
    elif "qwen" in provider:
        tokenizer, model = load_qwen(model_name)
    elif "llama" in provider:
        tokenizer, model = load_llama(model_name)
    elif "gemini" in provider:
        model = load_gemini(model_name) 
        tokenizer = None
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")

    return model, tokenizer