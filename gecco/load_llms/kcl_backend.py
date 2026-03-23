import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

KCL_BASE_URL = "https://api.ai.create.kcl.ac.uk/v1"


def load_kcl(model_name: str):
    """Load an OpenAI-compatible client pointing at KCL's AI API.

    Requires KCL_API_KEY environment variable.
    Base URL defaults to https://api.ai.create.kcl.ac.uk/v1 but can be
    overridden with KCL_BASE_URL.
    """
    print(f"[GeCCo] Initializing KCL backend for model: {model_name}")
    api_key = os.getenv("KCL_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "KCL_API_KEY not found in environment or .env file. "
            "Get an API key from KCL's AI platform."
        )
    base_url = os.getenv("KCL_BASE_URL", KCL_BASE_URL)
    return OpenAI(base_url=base_url, api_key=api_key)
