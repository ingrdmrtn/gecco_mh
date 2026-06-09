# Tool-Using Judge Implementation Record

This document describes the orchestrated judge pipeline as it is currently implemented in the repository. It is intended to be an as-built reference for developers who need to understand the runtime flow, data model, configuration, artifacts, and known limitations.

## Scope

The current implementation adds four connected pieces:

1. A DuckDB-backed diagnostic store under `gecco/diagnostic_store/`
2. Posterior predictive checks under `gecco/offline_evaluation/ppc.py`
3. A tool-using LLM judge under `gecco/construct_feedback/tool_judge.py`
4. Integration points in `gecco/run_gecco.py`

The Phase 4 runtime uses the orchestrated judge pipeline as the only supported judge path.

## High-Level Runtime Flow

At runtime, the judge path works like this:

1. `run_gecco.py` optionally initializes parameter recovery, a reusable simulator for PPC, the diagnostic store, and the orchestrated judge helper stack.
2. For each generated model, GeCCo validates and optionally runs parameter recovery before fitting.
3. If fitting succeeds, GeCCo computes optional individual-differences analyses and optional PPC summaries.
4. The per-model result is written into the in-memory `iteration_results` list.
5. At the end of the iteration, the canonical JSON artifact is written to `results/{task}/bics/iter{N}{tag}_run{X}.json`.
6. The diagnostic store writes a derived relational view of that same iteration into `results/{task}/diagnostics.duckdb`.
7. On the next feedback step, if the tool judge is enabled, the judge queries the DuckDB store through read-only tools and synthesizes structured feedback.
8. The synthesized feedback is injected back into the prompt loop, and a judge trace file is written to `results/{task}/judge/iter{N}{tag}_run{X}.json`.

The JSON artifacts in `results/{task}/bics/` remain the canonical source of truth. The DuckDB database is derived and can be rebuilt.

## Integration in `run_gecco.py`

### Initialization

During initialization, `run_gecco.py` sets up the following optional components:

- `self.recovery_checker`
- `self._ppc_simulator`
- `self.diagnostic_store`
- `self.tool_judge`
- `self.ppc_enabled`
- `self.ppc_n_sims`

Important implementation details:

- PPC reuses the simulator created for parameter recovery. If `parameter_recovery.enabled` is false, the PPC path will not run even if `judge.ppc.enabled` is true.
- The diagnostic store is only created when `cfg.judge.diagnostic_store.enabled` is truthy.
- The judge helper stack uses the orchestrated pipeline for both distributed and single-worker runs.
- `judge.mode` is retired from the runtime config contract.

### Per-model evaluation flow

For each proposed model, the current implementation does the following in order:

1. Build a model spec for validation and parameter recovery.
2. If parameter recovery is enabled, run `self.recovery_checker.check(spec)`.
3. If recovery fails, append a `RECOVERY_FAILED` record to `iteration_results` and skip fitting.
4. If recovery passes, continue to fitting via `run_fit(...)`.
5. Optionally compute individual-differences results.
6. Optionally compute PPC results with `compute_ppc(...)`.
7. Append a result dict into `iteration_results`.

For successfully fit models, the result dict currently includes:

- `function_name`
- `metric_name`
- `metric_value`
- `param_names`
- `code_file`
- `recovery`
- `individual_differences`
- `code`
- `eval_metrics`
- `participant_n_trials`
- `parameter_values`
- `ppc` when PPC succeeds

The `recovery` field is important. It is now included for successful models so the diagnostic store can populate the `parameter_recovery` table for models the judge is most likely to inspect.

### Iteration write path

At the end of each iteration:

1. `iteration_results` is serialized to the canonical JSON file in `results/{task}/bics/`
2. `self.feedback.record_iteration(...)` updates the legacy feedback history
3. If a diagnostic store exists, `write_iteration(...)` persists a relational copy into DuckDB
4. If distributed mode is active, the registry is updated afterward

### Judge dispatch

When `self.best_model` exists, GeCCo chooses between two feedback paths:

- Tool-using path: `self.tool_judge.get_feedback(...)`
- Fallback path: `self.feedback.get_feedback(...)`

The tool-using path currently receives:

- `iteration`
- `run_idx`
- `tag`
- `best_model`
- `best_metric`

The `tag` propagation is intentional and required to make judge trace filenames unique in distributed or participant-specific runs.

## Diagnostic Store

### Package layout

The diagnostic store implementation lives in:

- `gecco/diagnostic_store/__init__.py`
- `gecco/diagnostic_store/schema.py`
- `gecco/diagnostic_store/store.py`
- `gecco/diagnostic_store/populate.py`
- `gecco/diagnostic_store/rebuild.py`
- `gecco/diagnostic_store/tools.py`

### Database file location

The database is created at:

`results/{task}/diagnostics.duckdb`

### Schema

The current schema version is `1`.

Tables currently implemented:

1. `schema_version`
2. `iterations`
3. `models`
4. `model_participants`
5. `parameter_recovery`
6. `individual_differences`
7. `ppc`
8. `validation_errors`

#### `iterations`

One row per `(run_idx, iteration, tag)` with:

- `iteration_id`
- `run_idx`
- `iteration`
- `client_id`
- `tag`
- `timestamp`
- `n_models_proposed`

#### `models`

One row per evaluated model candidate with:

- `model_id`
- `iteration_id`
- `run_idx`
- `iteration`
- `name`
- `code`
- `metric_name`
- `metric_value`
- `param_names`
- `status`

Current `status` values are derived from `metric_name`:

- `ok`
- `recovery_failed`
- `fit_error`
- `validation_error`

#### `model_participants`

One row per participant fit result with:

- `id`
- `model_id`
- `participant_idx`
- `bic`
- `n_trials`
- `params`

The `params` JSON object maps parameter name to fitted value for that participant.

#### `parameter_recovery`

One row per model with:

- `model_id`
- `passed`
- `mean_r`
- `n_successful`
- `per_param_r`
- `simulation_error`

For `RECOVERY_FAILED` rows, the store reconstructs a minimal recovery record from the iteration result if a full `recovery` payload is not present.

#### `individual_differences`

One row per model with:

- `model_id`
- `mean_r2`
- `max_r2`
- `best_param`
- `per_param_r2`
- `per_param_detail`

#### `ppc`

One row per participant-statistic combination with:

- `ppc_id`
- `model_id`
- `participant_id`
- `statistic_name`
- `condition`
- `observed`
- `simulated_mean`
- `simulated_q025`
- `simulated_q975`
- `n_sims`

#### `validation_errors`

One row per validation failure with:

- `error_id`
- `model_id`
- `error_type`
- `error_message`
- `error_details`

### Store write behavior

`DiagnosticStore.write_iteration(...)` writes one iteration at a time and performs the following operations:

1. Creates or reuses the `iterations` row
2. Inserts a `models` row for each item in `iteration_results`
3. Inserts per-participant fit rows into `model_participants`
4. Writes recovery data into `parameter_recovery`
5. Writes individual-differences results into `individual_differences`
6. Writes validation failures into `validation_errors`
7. Writes PPC rows into `ppc`

Timestamps written by the store are timezone-aware UTC ISO strings.

### Rebuild behavior

`gecco/diagnostic_store/rebuild.py` provides `rebuild_from_artifacts(...)`, which rebuilds the database from `results/{task}/bics/iter*.json` files.

Behavior:

- Deletes the existing `.duckdb` file when `overwrite=True`
- Scans `bics/` for iteration JSON files
- Parses iteration, tag, and run index from filenames
- Replays each iteration into a fresh `DiagnosticStore`

This is the disaster-recovery path when the database is missing, stale, or incompatible.

## Judge Tool Layer

The judge never queries DuckDB directly. It uses read-only tools implemented in `gecco/diagnostic_store/tools.py`.

Current tools:

1. `list_iterations`
2. `get_best_models`
3. `get_model`
4. `get_per_participant_fit`
5. `get_recovery`
6. `get_individual_differences`
7. `get_ppc`
8. `compare_models`
9. `get_parameter_distribution`
10. `search_models`
11. `get_bic_trajectory`

Implementation notes:

- Every tool takes a `DiagnosticStore` instance plus tool arguments.
- Results are plain Python dicts or lists that are JSON-serializable.
- JSON columns are hydrated back into Python objects before being returned where applicable.
- Tool dispatch is centralized in `dispatch_tool(store, tool_name, args)`.
- Unknown tools or tool exceptions return an error payload instead of raising.

### What the tools actually expose

The current tool layer supports the following kinds of questions:

- What iterations have run and how many models succeeded?
- What are the best models by metric value?
- What code and parameters belong to a specific model?
- How does fit vary across participants for a model?
- Did a model pass recovery and which parameters recover poorly?
- Does a model have strong individual-differences signal?
- Which PPC statistics fall outside the predictive interval?
- How do a few selected models compare side by side?
- What is the distribution of a specific parameter across participants?
- Which models contain a code fragment or parameter name?
- Is best-fit performance improving over iterations?

## Posterior Predictive Checks

PPCs are implemented in `gecco/offline_evaluation/ppc.py`.

### Preconditions

PPC only runs when all of the following are true:

1. `judge.ppc.enabled` is true
2. `parameter_recovery.enabled` is true, so a simulator exists
3. The fitted model returned non-empty `parameter_values`

If those conditions are not met, no PPC is computed.

### Simulation path

The implementation reuses the parameter recovery simulator rather than introducing a second simulator implementation.

For each participant:

1. Observed participant data is extracted from the behavioral dataframe
2. A set of summary statistics is computed on observed data
3. `n_sims` forward simulations are generated with `simulator.simulate_subject(...)`
4. The same statistics are computed on each simulated dataset
5. The simulated distribution is reduced to mean, 2.5th percentile, and 97.5th percentile
6. One record per participant-statistic pair is produced

### Statistics currently computed

The current implementation computes:

- Choice proportions per observed choice value
- Lag-1 stay probability
- Win-stay rate when a reward column is detected
- Lose-shift rate when a reward column is detected
- Choice proportion by low-cardinality condition columns among the configured input columns

Current implementation detail:

- The `condition` field in stored PPC records is always `None`
- Condition-specific information is encoded into `statistic_name`, for example `choice_prop_cond_<column>_<value>`

This means the database supports condition-aware PPC analysis, but the condition label is not currently split into its own column by the PPC generator.

### Output shape

`compute_ppc(...)` returns:

```python
{"records": [ ... ]}
```

Each record contains:

- `participant_id`
- `statistic_name`
- `condition`
- `observed`
- `simulated_mean`
- `simulated_q025`
- `simulated_q975`
- `n_sims`

## Tool-Using Judge

The judge implementation lives in `gecco/construct_feedback/tool_judge.py`.

### Public contract

The main public method is:

```python
ToolUsingJudge.get_feedback(
    iteration: int,
    run_idx: int = 0,
    tag: str = "",
    best_model: str | None = None,
    best_metric: float | None = None,
) -> JudgeVerdict
```

`JudgeVerdict` currently contains:

- `iteration`
- `per_angle`
- `key_recommendations`
- `synthesized_feedback`
- `tool_call_count`
- `wall_time_seconds`

`synthesized_feedback` is the field injected into the next prompt and is the compatibility layer with the older feedback system.

### Analytical framing

The judge prompt instructs the model to reason across six angles:

1. Statistical fit quality
2. Parameter identifiability
3. Predictive adequacy
4. Individual differences
5. Mechanistic or theoretical coherence
6. Coverage

These are prompt-level analytical lenses. They are not implemented as separate agents or independent passes with isolated tool budgets.

### Backend support

The current implementation detects the backend from `cfg.llm.provider`:

- OpenAI-compatible providers: native tool loop through `chat.completions.create(...)`
- Gemini providers: native tool loop through `generate_content(...)`
- Other providers or HuggingFace: no native tool loop, falls back to one-shot generation

### OpenAI-compatible tool loop

The `_OpenAIToolLoop` implementation:

1. Sends system prompt, user message, and tool schemas
2. Receives tool calls from the model
3. Dispatches each tool through `dispatch_tool(...)`
4. Appends tool results back into the message list
5. Repeats until the model stops calling tools or the tool budget is exhausted
6. If the tool budget is exhausted, asks the model for a final synthesis without additional tools

### Gemini tool loop

The `_GeminiToolLoop` implementation follows the same broad pattern using Gemini function declarations and function responses.

Known implementation detail:

- The current Gemini loop focuses on function calls and does not preserve any free-text parts that may accompany a tool-calling response. This does not break the loop, but it can discard intermediate natural-language analysis from mixed responses.

### Fallback mode

If no native tool loop is available, `_fallback_generate(...)` assembles a prompt using:

- The judge system prompt
- The current iteration context
- A minimal direct pull from the store: BIC trajectory and top models

This fallback does not use iterative tool calling.

### Structured verdict extraction

The judge uses a two-stage process:

1. First stage: run the tool loop and obtain a free-form analysis
2. Second stage: ask the model to reformat that analysis as a JSON verdict

If JSON parsing fails, the implementation falls back to treating the entire response as `synthesized_feedback` with empty `per_angle` and `key_recommendations`.

This makes the system robust to imperfect structured outputs, but it means some trace files may contain an unstructured final verdict if the formatting pass fails.

### Audit traces

Every judge invocation writes a JSON trace file to:

`results/{task}/judge/iter{N}{tag}_run{X}.json`

The trace payload currently includes:

- `iteration`
- `run_idx`
- `tag`
- `timestamp`
- `tool_call_count`
- `wall_time_seconds`
- `tool_call_trace`
- `per_angle`
- `key_recommendations`
- `synthesized_feedback`

Each `tool_call_trace` entry contains:

- `tool`
- `args`
- `result_summary`

The `result_summary` is truncated to the first 500 characters of the JSON-serialized tool result.

## Configuration

An example configuration block exists in `config/judge_tool_example.yaml`.

Current judge-specific settings used by the implementation:

- `judge.max_tool_calls`
- `judge.model` as an optional override
- `judge.diagnostic_store.enabled`
- `judge.ppc.enabled`
- `judge.ppc.n_sims`

Important implementation detail:

- There is no strict Pydantic schema for the judge block in `config/schema.py` at the moment
- The judge code accesses configuration values via `getattr(...)` on the loaded config namespace

## Artifact Summary

Current artifacts produced by the judge stack are:

- Canonical iteration JSON: `results/{task}/bics/iter{N}{tag}_run{X}.json`
- Derived DuckDB store: `results/{task}/diagnostics.duckdb`
- Judge trace JSON: `results/{task}/judge/iter{N}{tag}_run{X}.json`
- Prompt feedback text: `results/{task}/feedback/iter{N}{tag}_run{X}.txt`

The text feedback file is still written by the main loop after the orchestrated judge artifact has been produced.

## Differences From the Original Plan

The current implementation is close to the original plan, but not identical.

Notable differences in the code as built:

- The `models` table does not currently store `rationale` or `analysis` columns.
- PPC stores condition-aware statistics in `statistic_name`; the separate `condition` column is not actively populated by `compute_ppc(...)`.
- The tool schemas include a `metric` parameter on `get_best_models(...)`, but the current implementation ignores it and always orders by `metric_value` ascending.
- The judge does not isolate analytical angles into separate tool loops; the angles exist only in prompt instructions.
- Judge output parsing is tolerant and may degrade to free text if structured JSON extraction fails.
- The judge config is flexible but not schema-enforced in `config/schema.py`.

## Current Limitations and Risks

The most important current limitations are:

1. PPC depends on parameter recovery being enabled because it reuses that simulator.
2. Gemini mixed text plus function-call responses are not fully preserved.
3. The tool layer is read-only and does not support re-running analyses or fitting variants.
4. The database is append-oriented and does not currently perform deduplication if the same iteration is replayed into an existing store outside the normal rebuild flow.
5. The judge relies on model compliance for tool use and structured output; the implementation is robust to failures, but not immune to weak judge behavior.

## Practical Developer Notes

If you need to extend this system, the lowest-risk entry points are:

- Add new derived tables and write logic in `gecco/diagnostic_store/store.py`
- Add new read-only analysis tools in `gecco/diagnostic_store/tools.py`
- Extend PPC summary statistics in `gecco/offline_evaluation/ppc.py`
- Tighten structured output or backend behavior in `gecco/construct_feedback/tool_judge.py`

If the store looks wrong, inspect the canonical `bics/iter*.json` files first. The database is derived from those files, and `rebuild_from_artifacts(...)` is the intended recovery path.
