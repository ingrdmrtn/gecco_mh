# Judge Mode Context Refactor Plan

## 1. Summary

Refactor judge configuration from a flat `capabilities` list into explicit `mode`, `context`, and `output` sections. The goal is to make feedback conditions unambiguous, fail fast for invalid combinations such as `mode: random` with context enabled, and gate agent tools so they cannot access information outside the configured context. The main risk this plan prevents is accidental information leakage between experimental conditions, especially tool access revealing performance, diagnostics, or code when those contexts are disabled.

## 2. Scope

Allowed implementation files:
- `config/schema.py`
- `gecco/construct_feedback/tool_judge.py`
- `gecco/construct_feedback/orchestrated.py`
- `gecco/diagnostic_store/tools.py`
- `gecco/feedback_coordinator.py`
- `gecco/run_gecco.py`
- `gecco/cli/run_judge_orchestrator.py`
- `docs/index.html`
- `config/**/*.yaml`, only where required to update production configs to the new judge schema

Allowed test files:
- `tests/test_phase3_config_validation.py`
- `tests/test_phase4_orchestrated_judge.py`
- `tests/test_judge_orchestration.py`
- `tests/test_judge_enhancements.py`
- New focused test file under `tests/` if these files become too broad, e.g. `tests/test_judge_mode_context.py`

Allowed verification artifacts:
- None required. Final response command output is sufficient if test output is short.

Forbidden files/directories:
- `results/`, `data*/`, generated DuckDB files, notebooks, dashboard assets, unrelated model/evaluation code, and unrelated docs.
- Do not edit existing untracked plan/review files unless the user explicitly approves.

Production surfaces and entrypoints in scope:
- Config loading and validation: `config.schema.load_config`, `JudgeConfig`, `GeCCoConfig.validate_judge_dependencies`.
- Feedback orchestration: `ToolUsingJudge.get_feedback_analysis`, `ToolUsingJudge.synthesize_for_persona`, `build_feedback_artifact`, `orchestrate_judge_feedback`.
- LLM prompt layers: judge system prompt, judge user message, tool schemas passed to tool loops, synthesis prompt, persona suffix handling.
- CLI/runtime launch paths that load configs and invoke judge feedback: `python -m gecco run local-client`, distributed/orchestrator paths, and `python -m gecco internal judge-orchestrate`.

Explicit non-goals:
- Do not change model generation, fitting, BIC computation, PPC computation, parameter recovery computation, or individual-differences computation.
- Do not redesign persona profile semantics beyond moving `persona_synthesis` under `judge.output`.
- Do not preserve the old flat `capabilities` behavior silently. Either reject old configs with a clear error or implement an explicit, tested deprecation translation. Prefer rejection plus updating repository configs unless compatibility is explicitly needed.

Worktree requirements:
- Run `git status --short` before implementation.
- Classify pre-existing tracked and untracked changes briefly before editing.
- Do not revert or modify unrelated pre-existing changes.
- After implementation, compare the final changed-file list against the allowed scope. Any out-of-scope file means the implementation is incomplete unless the user approves the scope change.

## 3. Contracts

### Contract A: Config Schema Is Explicit And Fails Fast

Code path/symbols:
- `JudgeConfig`, new `JudgeContextConfig`, new `JudgeOutputConfig`, `GeCCoConfig.validate_judge_dependencies`.

Required behavior:
- `judge.mode` accepts exactly `off`, `random`, `static`, `llm`, `agent`.
- `judge.context` contains only `attempted_models`, `performance`, `best_model_code`, `diagnostic` booleans.
- `judge.output` contains `persona_synthesis` boolean.
- `mode: off` and `mode: random` reject any enabled context and reject `persona_synthesis: true`.
- `mode: agent` requires at least one enabled context.
- `persona_synthesis: true` still requires multiple persona profiles or client persona guidance.

Positive proof:
- Validation tests load valid examples for each mode.

Negative proof:
- Tests assert `mode: random` with any context fails with an error mentioning random/context.
- Tests assert `mode: off` with any context fails.
- Tests assert `mode: agent` with all context disabled fails.
- Tests assert unknown old `capabilities` is either rejected with a clear migration message or translated by a tested compatibility layer.

### Contract B: Static Mode Provides Context Only

Code path/symbols:
- Deterministic context assembly currently related to `_build_summary_only_feedback` and best-model appendix in `build_feedback_artifact`.

Required behavior:
- `mode: static` returns enabled context directly.
- It must not call any LLM, fallback generation, synthesis generation, or diagnostic tools.
- It must not add conclusions, recommendations, or diagnostic prose beyond the deterministic context sections.
- `attempted_models: true` without `performance: true` must not contain BIC, metric values, status, rankings, or failed/success terms.

Positive proof:
- Tests capture static feedback for attempted-only, attempted+performance, and attempted+performance+best-code.

Negative proof:
- Monkeypatch LLM/fallback/tool methods to fail in static tests.
- Assert forbidden performance terms are absent from attempted-only static feedback.

### Contract C: Random Mode Has No Context

Code path/symbols:
- Random feedback branch in `ToolUsingJudge.get_feedback_analysis` and orchestration artifact builder.

Required behavior:
- `mode: random` returns only the fixed random feedback text.
- It must not include attempted models, performance, best-model code, diagnostics, previous verdict data, or persona synthesis.
- It must not call tools or LLMs.

Positive proof:
- Existing random feedback tests are updated to use `mode: random`.

Negative proof:
- Tests assert random mode rejects context in config validation.
- Tests assert random feedback artifact has no best-model code even if a caller passes best model data.

### Contract D: LLM Mode Uses Only Deterministic Context

Code path/symbols:
- Non-tool LLM/fallback path in `ToolUsingJudge.get_feedback_analysis`, synthesis prompt construction.

Required behavior:
- `mode: llm` builds deterministic context from enabled `judge.context`, then asks an LLM to summarize and recommend using only that context.
- It must not expose tool schemas or call diagnostic tools.
- Prompt messages must include the assembled context and must not include disabled context.

Positive proof:
- Test captures the actual messages passed to the LLM/fallback and verifies enabled context appears.

Negative proof:
- Monkeypatch tool loop and `dispatch_tool` path to fail if invoked.
- Test disabled `performance` context does not put BIC/metric text into the LLM prompt.

### Contract E: Agent Tool Access Is Context-Gated

Code path/symbols:
- `TOOL_SCHEMAS`, new context-gated tool schema selection, `_OpenAIToolLoop.run`, `_GeminiToolLoop.run`, dispatch path.

Required behavior:
- `mode: agent` passes only tools allowed by enabled context.
- `attempted_models` tools must not return performance, status, rankings, diagnostics, or code.
- `performance` tools may return BIC, trajectory, fit status, rankings, and counts, but not code or diagnostics.
- `best_model_code` enables code-returning tools or code appendices.
- `diagnostic` enables PPC, recovery, residuals, individual differences, participant heterogeneity, and parameter distribution tools.
- Dispatch must reject calls to tools outside the allowed context even if a malicious or confused LLM requests them.

Positive proof:
- Tests inspect actual tool schemas passed to the tool loop for each context combination.

Negative proof:
- Tests attempt to dispatch a forbidden tool under a limited context and assert it returns/reports a forbidden-tool error rather than data.
- Tests for `agent + attempted_models` confirm no metric/code/diagnostic tool schemas are present.

### Contract F: Prompt Layers Follow Mode And Context

Code path/symbols:
- `_build_judge_system_prompt`, `_JUDGE_USER_TEMPLATE` or replacement, `_build_synthesis_prompt`, persona suffix insertion, tool-loop message construction.

Required behavior:
- System, user, and synthesis prompts mention only enabled context areas.
- Diagnostic analysis instructions appear only when `context.diagnostic: true`.
- Code-building instructions or best-code appendices appear only when `context.best_model_code: true`.
- Persona-specific synthesis appears only when `output.persona_synthesis: true` and persona config is valid.

Positive proof:
- Tests capture actual messages sent to the LLM/tool loop and assert expected sections are present.

Negative proof:
- Tests assert disabled context terms do not appear in captured messages.

## 4. Tests And Verification

Add or update tests:
- `tests/test_phase3_config_validation.py`
  - Valid configs for `off`, `random`, `static`, `llm`, `agent`.
  - Invalid random/off with context.
  - Invalid agent with no context.
  - Invalid persona synthesis without persona config.
  - Old `capabilities` rejection or explicit translation behavior.
- `tests/test_judge_mode_context.py` or existing judge test file
  - Static attempted-only has names only and no metrics/status/recommendations.
  - Static attempted+performance includes both deterministic sections and no LLM/tool calls.
  - Random mode short-circuits and includes no context or code.
  - LLM mode captures actual prompt messages and proves disabled context is absent.
  - Agent mode captures actual tool schemas for attempted-only, performance-only, best-code, diagnostic, and full contexts.
  - Forbidden tool dispatch under limited context returns forbidden error.
- `tests/test_phase4_orchestrated_judge.py` or `tests/test_judge_orchestration.py`
  - Best-model code appendix obeys `context.best_model_code` across static, llm, and agent modes.
  - Persona synthesis obeys `judge.output.persona_synthesis`.

Commands to run using the `gecco_mh` conda environment:
- `conda run -n gecco_mh pytest tests/test_phase3_config_validation.py`
- `conda run -n gecco_mh pytest tests/test_judge_mode_context.py` if a new file is created
- `conda run -n gecco_mh pytest tests/test_phase4_orchestrated_judge.py tests/test_judge_orchestration.py tests/test_judge_enhancements.py`
- If runtime config YAML files are updated: `conda run -n gecco_mh python - <<'PY'` with a short script that loads each edited config via `config.schema.load_config`.

Manual review checks:
- Review all changed files and confirm they are within allowed scope.
- Search for `capabilities` and confirm remaining usages are either removed, migration-only, or test fixtures for rejection/translation.
- Search prompt builders for hardcoded diagnostic, performance, or code references that bypass context gates.
- Review `docs/index.html` and edited YAML configs for consistency with the new schema.

## 5. Implementation Steps

1. Baseline audit
   - Run `git status --short`.
   - Note pre-existing tracked/untracked files.
   - Inspect current judge schema, feedback orchestration, tool schemas, and tests.

2. Schema tests first
   - Add validation tests for new `mode`, `context`, and `output` shapes in `tests/test_phase3_config_validation.py`.
   - Implement `JudgeContextConfig`, `JudgeOutputConfig`, `JudgeMode`, and validation in `config/schema.py`.
   - Decide and test either old `capabilities` rejection or explicit deprecation translation before touching runtime behavior.

3. Deterministic context builder
   - Extract a single context assembly function in `tool_judge.py` for attempted models, performance, diagnostics, and best-code marker/appendix coordination.
   - Write tests for attempted-only and attempted+performance before wiring modes.
   - Ensure attempted-only uses safe queries that do not include metric/status/code fields.

4. Random and static modes
   - Update `get_feedback_analysis`, `synthesize_for_persona`, and `build_feedback_artifact` to use `judge.mode` and `judge.context`.
   - Update random/static tests to prove no LLM/tool calls occur.
   - Check that best-model code is appended only when `context.best_model_code` is true and mode is not `random`/`off`.

5. LLM mode
   - Route `mode: llm` through context assembly plus non-tool LLM synthesis.
   - Capture actual messages in tests and assert disabled context is absent.
   - Ensure no tool schemas are passed.

6. Agent context-gated tools
   - Split tool schemas into safe context groups or add a schema/filter function that returns only permitted tools.
   - Add dispatch-level enforcement for allowed tools per active context.
   - Update OpenAI and Gemini tool loops to receive the filtered schemas and allowed dispatch set.
   - Add tests that inspect actual tool schemas and forbidden dispatch behavior.

7. Prompt updates
   - Replace capability-based prompt branching with mode/context/output branching.
   - Ensure diagnostic angles are only present with `context.diagnostic`.
   - Ensure recommendations are part of `llm` and `agent` modes, not a separate capability.
   - Capture actual LLM/tool-loop messages in tests.

8. Orchestration and config surfaces
   - Update `orchestrated.py`, `feedback_coordinator.py`, `run_gecco.py`, and `run_judge_orchestrator.py` to use the new fields.
   - Update repository YAML configs that are production examples.
   - Update docs judge section and annotated examples.

9. Verification and final audit
   - Run the targeted pytest commands.
   - Load all edited YAML configs.
   - Run `git status --short` and compare final changed files to scope.
   - Search for stale `capabilities` behavior and context leaks.

## 6. Forbidden Patterns

- Do not keep `capabilities` as the hidden source of truth while adding new fields on top.
- Do not allow `mode: random` or `mode: off` to silently ignore enabled context; fail fast instead.
- Do not rely only on post-processing to remove forbidden information after tools have already exposed it.
- Do not pass all tool schemas to the agent and merely tell the LLM not to use forbidden tools.
- Do not use broad fallback behavior that turns invalid mode/context combinations into static or LLM mode.
- Do not write tests that mock away message construction or tool schema selection when those are the behavior under test.
- Do not add hardcoded config paths, hidden global active-context state, or mutable module-level context.
- Do not opportunistically refactor unrelated judge diagnostics, model fitting, data loading, or dashboard code.

## 7. Acceptance Checklist

Config schema:
- `conda run -n gecco_mh pytest tests/test_phase3_config_validation.py` passes.
- Invalid random/off/agent combinations fail with clear errors.
- Remaining `capabilities` behavior is deliberately rejected or explicitly translated with tests.

Static/random/LLM behavior:
- Static mode tests prove deterministic context only and no LLM/tool calls.
- Random mode tests prove no context, code, LLM calls, or tool calls.
- LLM mode tests capture actual messages and prove disabled context is absent.

Agent gating:
- Agent tests prove actual tool schemas are restricted by context.
- Dispatch-level tests prove forbidden tools cannot return data.
- Full agent mode with all context has attempted, performance, best-code, and diagnostic access.

Prompt and orchestration:
- Prompt tests prove diagnostic instructions only appear with `context.diagnostic`.
- Best-model code appendix appears only with `context.best_model_code` and never in random/off.
- Persona synthesis only runs when `judge.output.persona_synthesis` is true and persona config is valid.

Final review:
- Edited YAML configs load with `config.schema.load_config`.
- `docs/index.html` documents `mode`, `context`, and `output` rather than flat capabilities.
- `git status --short` changed-file list is within allowed scope or approved by the user.
- Any unverified risk is explicitly reported in the final response.
