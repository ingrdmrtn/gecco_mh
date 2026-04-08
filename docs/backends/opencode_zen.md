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
