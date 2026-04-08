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
