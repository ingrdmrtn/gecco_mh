import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


def load_vllm(model_name: str):
    """Load an OpenAI-compatible client pointing at a vLLM server.

    Requires VLLM_BASE_URL environment variable (e.g. http://localhost:8000/v1).
    Optionally reads VLLM_API_KEY (defaults to "EMPTY" since vLLM doesn't
    require authentication by default).
    """
    print(f"[GeCCo] Initializing vLLM backend for model: {model_name}")
    base_url = os.getenv("VLLM_BASE_URL")
    if not base_url:
        raise EnvironmentError(
            "VLLM_BASE_URL not found in environment or .env file. "
            "Set it to your vLLM server address, e.g. http://localhost:8000/v1"
        )
    api_key = os.getenv("VLLM_API_KEY", "EMPTY")
    return OpenAI(base_url=base_url, api_key=api_key)
