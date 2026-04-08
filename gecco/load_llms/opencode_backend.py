import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


def load_opencode(model_name: str):
    """Load an OpenAI-compatible client pointing at OpenCode Zen API.

    Requires OPENCODE_API_KEY environment variable.
    Base URL defaults to OpenCode's API endpoint.
    """
    print(f"[GeCCo] Initializing OpenCode Zen backend for model: {model_name}")
    api_key = os.getenv("OPENCODE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENCODE_API_KEY not found in environment or .env file. "
            "Get an API key from https://opencode.ai"
        )
    base_url = os.getenv("OPENCODE_BASE_URL", "https://api.opencode.ai/v1")
    return OpenAI(base_url=base_url, api_key=api_key)
