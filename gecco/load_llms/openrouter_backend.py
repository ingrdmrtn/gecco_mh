import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


def load_openrouter(model_name: str):
    """Load an OpenAI-compatible client pointing at OpenRouter API.

    Requires OPENROUTER_API_KEY environment variable.
    Base URL defaults to OpenRouter's API endpoint.
    """
    print(f"[GeCCo] Initializing OpenRouter backend for model: {model_name}")
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENROUTER_API_KEY not found in environment or .env file. "
            "Get an API key from https://openrouter.ai"
        )
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    return OpenAI(base_url=base_url, api_key=api_key)
