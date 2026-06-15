# Judge Individual Differences Context Plan

## 1. Summary

Add an explicit `judge.context.individual_differences` switch so reporting and investigation of participant-level fit heterogeneity and self-report regression results are optional. `judge.context.diagnostic` should continue to gate recovery, PPC, and block residual diagnostics, while individual-differences evidence is only exposed when the new flag is enabled. The main risk this plan prevents is individual-differences information leaking through prompts, static summaries, or agent tools when a run asks for diagnostics but not individual differences.

## 2. Scope

Allowed implementation files:
- `config/schema.py`
- `gecco/construct_feedback/tool_judge.py`
- `gecco/diagnostic_store/tools.py`
- `docs/index.html`
- `config/**/*.yaml`, only to add or set `judge.context.individual_differences` in runtime/example configs affected by this schema change

Allowed test files:
- `tests/test_phase3_config_validation.py`
- `tests/test_judge_mode_context.py`
- `tests/test_phase4_orchestrated_judge.py`, only if artifact/orchestration behavior needs a focused regression
- `tests/test_judge_orchestration.py` and `tests/test_judge_enhancements.py`, only to update fixtures broken by the new context field

Allowed verification artifacts:
- None required. Final response command output is sufficient.

Forbidden files/directories:
- `results/`, `results*/`, `data*/`, generated DuckDB files, notebooks, dashboard assets, and unrelated model/evaluation code
- Existing plan/review files except this plan
- Broad config restructuring outside `judge.context.individual_differences`

Production surfaces and entrypoints in scope:
- Config loading and validation: `JudgeContextConfig`, `JudgeConfig.validate_mode_context`, `judge_context_enabled`, `load_config`
- Feedback assembly: `_judge_context_flags`, `_build_active_prompt_angles`, `_build_judge_system_prompt`, `_JUDGE_USER_TEMPLATE`, `_build_static_diagnostic_summary`, `_apply_capability_postprocessing`, `ToolUsingJudge.get_feedback_analysis`
- Agent tools: `_enabled_agent_tool_names`, `get_judge_tool_names`, `get_judge_tool_schemas`, `dispatch_tool`, `get_individual_differences`, `get_participant_best_models`, `compare_models`, `get_parameter_distribution`
- Prompt layers: judge system prompt, judge user message, agent planning/tool-loop messages, and synthesis prompts if they mention diagnostic or individual-differences evidence
- Runtime configs under `config/` and docs examples that declare `judge.context`

Explicit non-goals:
- Do not change how individual-differences analyses are computed, stored, or evaluated.
- Do not change `individual_differences_eval` data loading or regression semantics.
- Do not add a compatibility layer for old configs beyond normal default `false` handling for the new context field.
- Do not redesign judge modes, persona synthesis, launch behavior, or diagnostic-store schema.

Worktree rules:
- Run `git status --short` before implementation and briefly classify tracked and untracked pre-existing changes, including `docs/index.html` and plan files if still present.
- Do not revert or modify unrelated pre-existing changes.
- If editing an already-dirty in-scope file, inspect the relevant hunks first and preserve unrelated user changes.
- Final changed-file review must include tracked and untracked files. Any changed file outside the allowed scope means implementation is incomplete unless the user approves the scope change.

## 3. Contracts

| Contract | Code path / symbol | Positive proof | Negative proof |
| --- | --- | --- | --- |
| Schema exposes a separate flag | `JudgeContextConfig`, `JudgeConfig.validate_mode_context` | Configs load with `individual_differences: true` for `static`, `llm`, and `agent`; `agent` with only this flag enabled is valid | `off` and `random` reject the new flag; unknown context fields still fail |
| Diagnostic context no longer includes individual differences | `_build_static_diagnostic_summary`, `_build_active_prompt_angles`, `_build_judge_system_prompt`, `_JUDGE_USER_TEMPLATE`, `_apply_capability_postprocessing` | With `diagnostic: true` and `individual_differences: false`, recovery/PPC/block residual content is still available | Static feedback, LLM prompts, and postprocessed feedback contain no `individual differences`, `R²`, `r2`, self-report, participant heterogeneity, or `get_individual_differences` wording |
| Individual-differences context gates both heterogeneity and regression | `_build_static_diagnostic_summary`, `get_feedback_analysis`, `get_judge_tool_names`, `get_judge_tool_schemas` | With `individual_differences: true`, static/LLM context may mention participant fit heterogeneity and regression/R² evidence; agent schemas include `get_individual_differences` and `get_participant_best_models` | With the flag disabled, `get_individual_differences`, `get_participant_best_models`, and any ID-bearing `compare_models` fields are not exposed through schemas or prompt/tool output |
| Dispatch cannot bypass the flag | `dispatch_tool`, `_enabled_agent_tool_names`, agent tool loops | Allowed ID tools can be dispatched when the flag is enabled | A forced LLM call to `get_individual_differences` or `get_participant_best_models` returns a forbidden-tool error when the flag is disabled |

## 4. Tests And Verification

Add or update tests:
- `tests/test_phase3_config_validation.py`: valid configs for the new field across `static`, `llm`, and `agent`; `agent` valid with only `individual_differences`; `off`/`random` invalid with `individual_differences: true`; unknown context keys still rejected.
- `tests/test_judge_mode_context.py`: update context fixtures to include `individual_differences`; add diagnostic-only-without-ID cases proving recovery/PPC/residual remain but ID/heterogeneity/regression text is absent.
- `tests/test_judge_mode_context.py`: add ID-only and diagnostic+ID cases proving actual OpenAI/Gemini tool schemas and actual prompt messages include the expected ID tools/text only when enabled.
- `tests/test_judge_mode_context.py`: add forced forbidden dispatch tests for `get_individual_differences` and `get_participant_best_models` when the flag is disabled.
- `tests/test_phase4_orchestrated_judge.py`: add or update only if persisted artifact assembly can append or preserve ID text outside `ToolUsingJudge` tests.

Commands to run with the `gecco_mh` conda environment:
- `conda run -n gecco_mh pytest tests/test_phase3_config_validation.py`
- `conda run -n gecco_mh pytest tests/test_judge_mode_context.py`
- `conda run -n gecco_mh pytest tests/test_phase4_orchestrated_judge.py tests/test_judge_orchestration.py tests/test_judge_enhancements.py`
- If YAML configs are edited, run a short `conda run -n gecco_mh python` script that loads each edited config via `config.schema.load_config`.

Manual review checks:
- Search active code and docs for `individual differences`, `individual_differences`, `R²`, `r2`, and `get_participant_best_models`; confirm each active exposure is gated by `context.individual_differences` or is computation/storage code outside judge reporting.
- Search `compare_models` output paths and tests to confirm ID regression fields cannot leak when only `diagnostic` or `performance` is enabled.
- Review captured LLM-message tests to confirm they inspect actual provider call payloads, not just helper return values.
- Compare final `git status --short` against the baseline classification.

## 5. Implementation Steps

1. Baseline and tighten scope
- Run `git status --short` and classify pre-existing tracked/untracked changes.
- Inspect current relevant hunks in any dirty in-scope files before editing.
- Confirm the final implementation file list remains within this plan.

2. Schema tests and schema edit
- Add failing config validation tests for `individual_differences` in `tests/test_phase3_config_validation.py`.
- Add `individual_differences: bool = False` to `JudgeContextConfig` and include it in `has_context` and validation messages in `config/schema.py`.
- Run `conda run -n gecco_mh pytest tests/test_phase3_config_validation.py` before continuing.

3. Static and LLM context gates
- Update `_judge_context_flags`, `_build_active_prompt_angles`, prompt text, and `_apply_capability_postprocessing` in `tool_judge.py`.
- Split `_build_static_diagnostic_summary` so recovery/PPC/block residuals follow `diagnostic`, while participant heterogeneity and self-report regression follow `individual_differences`.
- Add tests that capture static output and actual LLM messages for diagnostic-only, ID-only, and diagnostic+ID contexts.

4. Agent tool gating
- In `diagnostic_store/tools.py`, move `get_individual_differences` and `get_participant_best_models` behind the new context flag.
- Ensure `compare_models` does not return ID regression fields unless the flag is enabled; prefer a separate allowed variant or internal parameter over exposing ID fields in the diagnostic/performance path.
- Update schema filtering tests for OpenAI and Gemini and forbidden-dispatch tests.

5. Configs and docs
- Update `docs/index.html` to document `diagnostic` and `individual_differences` separately, including analytical angles, tool gating, and config examples.
- Update runtime/example YAML configs with explicit `individual_differences`. Preserve current “all context” behavior by setting it true only where the config name or intent indicates all context or individual-differences reporting.
- Load each edited YAML config with `load_config`.

6. Final verification
- Run the targeted pytest commands.
- Run the manual searches and final changed-file comparison.
- Report any skipped command, failed config load, or approved scope expansion in the final response.

## 6. Forbidden Patterns

- Do not leave individual-differences checks bundled under `context.diagnostic`.
- Do not rely on postprocessing after prompts or tools have already seen individual-differences data.
- Do not expose `get_participant_best_models` through performance-only or diagnostic-only agent schemas.
- Do not let `compare_models` leak `id_mean_r2`, `id_max_r2`, or `id_best_param` unless `context.individual_differences` is enabled.
- Do not test only helper functions when actual LLM provider payloads or tool schemas are the behavior under contract.
- Do not introduce hidden global context, environment switches, hardcoded config paths, or launch-behavior changes.
- Do not opportunistically refactor diagnostic computation, data loading, or result persistence.

## 7. Acceptance Checklist

Schema contract:
- `tests/test_phase3_config_validation.py` covers valid/invalid `individual_differences` context combinations.
- `conda run -n gecco_mh pytest tests/test_phase3_config_validation.py` passes.

Prompt/static contract:
- Static and LLM tests prove diagnostic-only output excludes individual-differences and heterogeneity text.
- Static and LLM tests prove ID-enabled output includes the intended heterogeneity/regression context.
- Actual captured LLM messages are asserted.

Agent contract:
- Tool-schema tests prove ID tools appear only with `context.individual_differences`.
- Forbidden-dispatch tests prove direct calls to ID tools are rejected when disabled.
- `compare_models` leakage is covered by a test or by removed ID fields.

Final checks:
- `conda run -n gecco_mh pytest tests/test_judge_mode_context.py` passes.
- `conda run -n gecco_mh pytest tests/test_phase4_orchestrated_judge.py tests/test_judge_orchestration.py tests/test_judge_enhancements.py` passes, or any skipped subset is justified.
- Edited YAML configs load through `config.schema.load_config`.
- Final `git status --short` contains only allowed files plus documented pre-existing unrelated changes.
- Manual searches find no active ungated individual-differences reporting path.
