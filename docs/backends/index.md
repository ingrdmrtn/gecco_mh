# LLM Backends

GeCCo supports multiple LLM providers for model generation:

## Supported Backends

- **OpenAI/GPT**: OpenAI's GPT models (o1, o3, GPT-4, etc.)
- **Gemini**: Google's Gemini models
- **vLLM**: Self-hosted models via vLLM
- **KCL**: King's College London internal API
- **OpenCode Zen**: Access to frontier models via OpenCode
- **OpenRouter**: Unified API for models from multiple providers
- **Hugging Face**: Local models (Llama, Qwen, DeepSeek-R1, etc.)

## Backend Documentation

- [OpenCode Zen](opencode_zen.md)
- [OpenRouter](openrouter.md)

## Configuration

Each backend is configured via the `llm` section in your YAML config:

```yaml
llm:
  provider: "openrouter"  # or "opencode", "vllm", etc.
  base_model: "anthropic/claude-3-5-sonnet"
  temperature: 0.1
  max_tokens: 4096
```

## Environment Variables

API keys and optional base URLs are read from environment variables or `.env` files:

- `OPENCODE_API_KEY` / `OPENCODE_BASE_URL`
- `OPENROUTER_API_KEY` / `OPENROUTER_BASE_URL`
- `KCL_API_KEY` / `KCL_BASE_URL`
- `VLLM_BASE_URL` / `VLLM_API_KEY`
- `OPENAI_API_KEY`
- `GEMINI_API_KEY`

See `.env.example` in `gecco_tutorials/` for a template.
