# Config Directory

This directory contains YAML configuration files for GeCCo experiments, along with a Pydantic schema for validation.

## Organization

```
config/
├── README.md                  # This file
├── __init__.py                # Empty package init
├── schema.py                  # Pydantic config schema + loader
├── two_step_factors/          # Active experiment configs
│   └── deepseekv4flash/       # Configs for DeepSeek V4 Flash model
│       ├── README.md          # Per-run config documentation
│       ├── judge_off.yaml
│       ├── judge_random.yaml
│       ├── judge_static_*.yaml
│       ├── judge_llm_*.yaml
│       └── judge_agent_*.yaml
└── archive/                   # Historical/backup configs
    ├── two_step_factors_gpt55.yaml
    ├── two_step_factors_gemini3flash_*.yaml
    ├── two_step_factors_deepseekv4pro.yaml
    └── ... (other retired configs)
```

## Config Sections

All configs share a common top-level structure validated by `schema.py`:

| Section | Purpose |
|---|---|
| `slurm` | HPC resource allocation (CPUs, memory, partition) |
| `loop` | Top-level search loop settings (iterations, clients) |
| `task` | Experiment name, description, and goal prompt |
| `data` | Input CSV path, columns, narrative template |
| `llm` | Model provider, system prompt, guardrails, code template |
| `judge` | Judge mode, context sections, output formatting |
| `evaluation` | Model fitting/evaluation (metric, optimizer, train/test split) |
| `validation` | Retry limits for model generation |
| `parameter_recovery` | Simulated parameter recovery diagnostics |
| `baseline` | Reference model code used for comparison |
| `feedback` | Feedback mechanism (llm) |
| `individual_differences_eval` | Self-report data for individual-differences analysis |
| `clients` | Per-client LLM configurations and personas |

## How Configs Are Loaded

Configs are loaded via `config.schema.load_config(path)`. The YAML is validated against `GeCCoConfig` (Pydantic model) with cross-field validation (e.g., judge mode + context compatibility).

## Active vs Archive

- **`two_step_factors/`**: Configs actively used for two-step decision-making experiments.
- **`archive/`**: Retired configs from earlier model runs (GPT, Gemini, etc.), kept for reproducibility.
