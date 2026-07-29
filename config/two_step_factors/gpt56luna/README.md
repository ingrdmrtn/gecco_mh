# GPT-5.6 Luna — Two-Step Factors Configs

This directory contains YAML configs for running the two-step decision-making task with GPT-5.6 Luna as the model proposer. The configs differ along three key dimensions: **judge mode**, **judge context**, and **client persona setup**.

## Judge Modes

The `judge.mode` field controls how the judge evaluates proposed models:

| Mode | Meaning |
|---|---|
| `off` | No judge; models are proposed and accepted as-is |
| `random` | Judge picks a random model to advance |
| `static` | Judge uses a fixed, pre-written evaluation prompt (no LLM calls) |
| `llm` | Judge uses an LLM call (same model) to synthesize a written evaluation |
| `agent` | Judge uses a tool-using agent to produce a structured evaluation |

## Judge Context

The `judge.context` fields control what information is included in the judge's prompt:

| Context field | Description |
|---|---|
| `attempted_models` | List of all previously attempted model names and descriptions |
| `performance` | Fitted parameter values, BIC, and NLL for each model |
| `best_model_code` | Full Python source code of the current best model |
| `diagnostic` | Model diagnostic analysis (posterior predictive checks, etc.) |
| `individual_differences` | Self-report questionnaire factor analysis |

## Persona Synthesis

When `judge.output.persona_synthesis: true`, the judge generates evaluations from multiple client personas (exploit, explore, minimal, hybrid, complex) to produce more diverse and targeted feedback.

## Config File Reference

### No Judge / Random Baselines

| File | `mode` | `context` | Notes |
|---|---|---|---|
| `judge_off.yaml` | `off` | none | No judge — disables evaluation entirely |
| `judge_random.yaml` | `random` | none | Random model selection baseline |

### Static Judge (no LLM calls for evaluation)

| File | `mode` | `context` | `persona_synthesis` | Notes |
|---|---|---|---|---|
| `judge_static_attempted.yaml` | `static` | attempted only | no | Minimal context — just lists models tried |
| `judge_static_attempted_performance.yaml` | `static` | attempted + performance | no | Add performance numbers |
| `judge_static_all_context.yaml` | `static` | attempted + performance + best_model_code + diagnostic | no | Full context, generic clients |
| `judge_static_all_context_persona.yaml` | `static` | attempted + performance + best_model_code + diagnostic | yes | Full context + persona synthesis |

### LLM Judge (LLM-call for evaluation text)

| File | `mode` | `context` | `persona_synthesis` | Notes |
|---|---|---|---|---|
| `judge_llm_attempted.yaml` | `llm` | attempted only | no | Minimal context |
| `judge_llm_attempted_performance_code.yaml` | `llm` | attempted + performance + best_model_code | no | Add performance + best code |
| `judge_llm_attempted_performance_code_diagnostic.yaml` | `llm` | all four | no | Full context, no persona |
| `judge_llm_all_context_persona.yaml` | `llm` | all four | yes | Full context + persona synthesis |

### Agent Judge (tool-using agent for structured evaluation)

| File | `mode` | `context` | `persona_synthesis` | Notes |
|---|---|---|---|---|
| `judge_agent_all_context.yaml` | `agent` | all four | no | Full context, generic clients |
| `judge_agent_all_context_persona.yaml` | `agent` | all four | yes | Full context + persona synthesis |

## Client Configuration

Configs with persona synthesis (`*_persona.yaml`) use five specialized client roles:

- **exploit** — refines the current best model with small, targeted changes
- **explore** — proposes structurally novel model architectures
- **minimal** — targets parsimony (max 3 parameters)
- **hybrid** — focuses on MB/MF hybrid architectures
- **complex** — proposes rich, high-parameter models (up to 10 params)

Configs without persona synthesis use five generic clients (`client1`–`client5`) with no specialization.
