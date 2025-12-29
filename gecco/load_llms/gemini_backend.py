import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

def load_gemini(model_name: str):
    print(f"[GeCCo] Initializing GEMINI model: {model_name}")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY not found in environment or .env file.")
    Gemini = genai.Client()
    return Gemini