# Adding OpenCode Zen and OpenRouter as LLM Backends

## Overview

This plan describes how to add OpenCode Zen and OpenRouter as new LLM provider backends in GeCCo. Both are OpenAI-compatible APIs, meaning they can be integrated using the same pattern as the existing `vllm_backend`.

## Why Add These Backends?

- **OpenCode Zen**: Provides access to various frontier models through OpenCode's infrastructure
- **OpenRouter**: A unified API offering access to models from multiple providers (Anthropic, OpenAI, Meta, etc.)

Both services offer advantages over self-hosted vLLM (no server maintenance) while providing access to a wider range of models.

## Structured Output Enforcement

Both OpenCode Zen and OpenRouter support OpenAI's `response_format` parameter for structured output. This ensures the LLM returns valid JSON, which is critical for GeCCo's model parsing pipeline.

**Enforcement level**: JSON mode (`response_format={"type": "json_object"}`)

This guarantees syntactically valid JSON but doesn't enforce schema structure. Schema validation is handled by the Pydantic layer (see `plans/pydantic_schema_enforcement.md`).

## Current Architecture

GeCCo uses a two-file pattern for each backend:

1. **`gecco/load_llms/<name>_backend.py`**: Contains a `load_<name>(model_name)` function that returns an API client
2. **`gecco/load_llms/model_loader.py`**: Dispatch function `load_llm(provider, model_name)` that routes to the appropriate backend based on the provider string

For generation, `gecco/run_gecco.py` contains a `generate()` method with if/elif branches for each provider type.

## Implementation Plan

### Step 1: Create Backend Files

Create two new files in `gecco/load_llms/`:

#### `gecco/load_llms/opencode_backend.py`

```python
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
        raise EnvironmentError("OPENCODE_API_KEY not found in environment or .env file.")
    base_url = os.getenv("OPENCODE_BASE_URL", "https://api.opencode.ai/v1")
    return OpenAI(base_url=base_url, api_key=api_key)
```

#### `gecco/load_llms/openrouter_backend.py`

```python
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
        raise EnvironmentError("OPENROUTER_API_KEY not found in environment or .env file.")
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    return OpenAI(base_url=base_url, api_key=api_key)
```

### Step 2: Update the Model Loader Dispatch

In `gecco/load_llms/model_loader.py`, add dispatch entries for the new providers.

Find the else clause that raises `ValueError` and add entries before it:

```python
elif "opencode" in provider:
    from gecco.load_llms.opencode_backend import load_opencode
    model = load_opencode(model_name)
    tokenizer = None

elif "openrouter" in provider:
    from gecco.load_llms.openrouter_backend import load_openrouter
    model = load_openrouter(model_name)
    tokenizer = None
```

### Step 3: Update the Generate Method

In `gecco/run_gecco.py`, modify the `generate()` method to handle the new providers.

On line 192, change:

```python
elif "vllm" in provider or "kcl" in provider:
```

to:

```python
elif "vllm" in provider or "kcl" in provider or "opencode" in provider or "openrouter" in provider:
```

Both OpenCode Zen and OpenRouter use the standard OpenAI chat completions API, just like vLLM and KCL. They support `response_format={"type": "json_object"}` for JSON mode.

### Step 3.5: Add Structured Output Helper Function

Add a helper function to `gecco/structured_output.py` for OpenAI-compatible JSON mode:

**After `get_vllm_response_format()` (around line 406), add:**

```python
def get_openai_compatible_response_format() -> dict:
    """
    JSON mode for any OpenAI-compatible API.
    
    Use this for providers that support JSON mode but not full JSON Schema:
    - vLLM (self-hosted)
    - KCL (internal)
    - OpenCode Zen
    - OpenRouter
    
    Ensures output is syntactically valid JSON. Schema enforcement is handled
    by the Pydantic validation layer (see pydantic_schema_enforcement.md).
    
    Returns
    -------
    dict
        OpenAI-compatible response format specification.
    """
    return {"type": "json_object"}
```

**Then update the vLLM/KCL/OpenCode/OpenRouter branch in `run_gecco.py` generate() method:**

```python
# Structured output via json_object mode
if getattr(self.cfg.llm, "structured_output", True):
    from gecco.structured_output import get_openai_compatible_response_format
    create_kwargs["response_format"] = get_openai_compatible_response_format()
```

This replaces the existing `get_vllm_response_format()` call, making it clear that all OpenAI-compatible providers use the same JSON mode enforcement.

### Step 4: Add Environment Variables to `.env.example`

Add the following to `gecco_tutorials/.env.example` (or create if it doesn't exist):

```
# OpenCode Zen
OPENCODE_API_KEY=your_opencode_api_key_here
OPENCODE_BASE_URL=https://api.opencode.ai/v1  # optional, usually not needed

# OpenRouter
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1  # optional, usually not needed
```

### Step 5: Documentation

Create simple, readable documentation for users:

#### `docs/backends/opencode_zen.md`

```markdown
# OpenCode Zen Backend

OpenCode Zen provides access to various LLM models through a unified API.

## Setup

1. Get your API key from [opencode.ai](https://opencode.ai)
2. Add to your `.env` file:
   ```
   OPENCODE_API_KEY=your_key_here
   ```

## Usage

In your YAML config:

```yaml
llm:
  provider: "opencode"
  base_model: "anthropic/claude-3-5-sonnet"  # or any model available on OpenCode
  temperature: 0.1
  max_tokens: 4096
```

## Available Models

See [OpenCode Zen model catalog](https://opencode.ai/models) for available models.
```

#### `docs/backends/openrouter.md`

```markdown
# OpenRouter Backend

OpenRouter provides unified access to models from multiple providers.

## Setup

1. Get your API key from [openrouter.ai](https://openrouter.ai)
2. Add to your `.env` file:
   ```
   OPENROUTER_API_KEY=your_key_here
   ```

## Usage

In your YAML config:

```yaml
llm:
  provider: "openrouter"
  base_model: "anthropic/claude-3-5-sonnet"  # or any model from their catalog
  temperature: 0.1
  max_tokens: 4096
```

## Available Models

OpenRouter hosts models from Anthropic, OpenAI, Meta, Mistral, and others. See [openrouter.ai/models](https://openrouter.ai/models) for the full catalog.
```

Also update the main `docs/backends/index.md` to list the new backends.

## Testing

To test the new backends:

1. Set environment variables:
   ```bash
   export OPENCODE_API_KEY=your_key
   export OPENROUTER_API_KEY=your_key
   ```

2. Run with test mode using the new provider:
   ```bash
   python scripts/run_gecco_distributed.py --config two_step_factors_distributed.yaml --test --provider opencode
   ```

## Files to Create/Modify

| File | Action |
|------|--------|
| `gecco/load_llms/opencode_backend.py` | Create |
| `gecco/load_llms/openrouter_backend.py` | Create |
| `gecco/load_llms/model_loader.py` | Modify |
| `gecco/run_gecco.py` | Modify (add provider dispatch, use `get_openai_compatible_response_format()`) |
| `gecco/structured_output.py` | Modify (add `get_openai_compatible_response_format()`) |
| `gecco_tutorials/.env.example` | Modify |
| `docs/backends/opencode_zen.md` | Create |
| `docs/backends/openrouter.md` | Create |
| `docs/backends/index.md` | Modify |

## Notes

- Both backends use OpenAI-compatible APIs and support `response_format={"type": "json_object"}` for JSON mode
- OpenRouter may require slightly different model naming conventions (e.g., `anthropic/claude-3-5-sonnet` instead of `claude-3-5-sonnet`)
- Rate limits and pricing vary by provider; check their respective dashboards
- JSON mode ensures syntactically valid JSON; schema validation is handled by Pydantic (see `plans/pydantic_schema_enforcement.md`)
