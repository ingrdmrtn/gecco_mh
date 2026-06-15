# Judge Individual Differences Context Review

## 1. Findings

1. **Required YAML config updates are incomplete** (plan phase 5; acceptance checklist final checks).  
   Several `config/**/*.yaml` files that declare `judge.context` still omit the new explicit `individual_differences` field, despite the plan requiring runtime/example YAML configs to be updated with an explicit value. Examples:
   - `config/archive/two_step_factors_gpt55.yaml:159-164`
   - `config/archive/judge_tool_example.yaml:9-14`
   - `config/two_step_factors/deepseekv4flash/judge_llm_attempted.yaml:155-160`
   Only six config files currently contain `individual_differences:`. This leaves many examples relying on the default rather than documenting the new context split.

2. **Forced LLM-call proof is missing for the new forbidden ID tools** (contract: “Dispatch cannot bypass the flag”; tests plan lines 63-64).  
   `tests/test_judge_mode_context.py:1139-1167` directly calls `dispatch_tool()` for `get_individual_differences` and `get_participant_best_models`, but does not exercise an actual OpenAI/Gemini tool loop forced to request those tools. Existing actual-provider forced-call coverage at `tests/test_judge_mode_context.py:555-596` covers `get_best_models`, not the new ID tools required by this plan. This leaves the agent-loop path for forced ID calls under-proven.

3. **`compare_models` leakage proof is weak/incomplete** (contract: “Individual-differences context gates both heterogeneity and regression”; forbidden pattern: no `id_mean_r2`, `id_max_r2`, or `id_best_param` leak).  
   The implementation removes ID fields from `compare_models()` (`gecco/diagnostic_store/tools.py:649-665`), but I did not find a focused test asserting that `compare_models` output/tool traces lack `id_mean_r2`, `id_max_r2`, or `id_best_param` when only diagnostic/performance is enabled. The plan explicitly accepts either a test or removed fields; the fields appear removed, but the required proof is not captured in tests or a receipt available in the worktree.

4. **Implementation receipt and phase-gate evidence are not available in the repository** (plan phases 1, 2, 5, 6).  
   I found no implementation receipt for this plan. Therefore I could not verify the required baseline/final `git status --short`, phase-gated test run after schema edits, final targeted pytest commands, manual searches, or edited-YAML `load_config` checks. This is an evidence gap rather than a code-path failure.

## 2. Open Questions

- Is there an implementation receipt outside the repository that records baseline/final `git status --short`, command results, manual grep checks, and YAML load checks?

## 3. Verification Summary

Observed from code/tests:
- Schema flag exists and participates in mode/context validation: `config/schema.py:149-156`, `config/schema.py:214-235`.
- Static/LLM context construction separates diagnostic and ID summary inputs: `gecco/construct_feedback/tool_judge.py:302-423`, `gecco/construct_feedback/tool_judge.py:2248-2258`.
- Active prompt angles gate the ID angle on `context.individual_differences`: `gecco/construct_feedback/tool_judge.py:788-827`.
- Agent tool names gate `get_individual_differences` and `get_participant_best_models`: `gecco/diagnostic_store/tools.py:950-973`.
- `dispatch_tool()` enforces an allowlist when supplied: `gecco/diagnostic_store/tools.py:1493-1510`.
- OpenAI/Gemini schema tests include ID-only and diagnostic+ID cases: `tests/test_judge_mode_context.py:433-489`, `tests/test_judge_mode_context.py:494-552`.
- Static and OpenAI LLM prompt tests cover diagnostic-only and ID-only cases: `tests/test_judge_mode_context.py:964-1136`.

Still required/unverified:
- `conda run -n gecco_mh pytest tests/test_phase3_config_validation.py`
- `conda run -n gecco_mh pytest tests/test_judge_mode_context.py`
- `conda run -n gecco_mh pytest tests/test_phase4_orchestrated_judge.py tests/test_judge_orchestration.py tests/test_judge_enhancements.py`
- YAML `load_config` checks for every edited config.
- Manual searches for ID/R²/tool leakage.
- Baseline/final `git status --short` comparison and pre-existing dirty-file classification.

Contract rows verified from inspected implementation:
- Schema exposes a separate flag: substantially implemented; command proof unavailable.
- Diagnostic context no longer includes individual differences: substantially implemented for static and OpenAI LLM tests; Gemini prompt-negative proof for ID wording is less direct.
- Individual-differences context gates heterogeneity/regression: substantially implemented; compare_models proof weak as noted.
- Dispatch cannot bypass the flag: allowlist implemented; forced-ID LLM-loop proof missing.

## 4. Scope Audit

I could not run `git status --short` or `git diff`, so I cannot definitively distinguish pre-existing dirty files from files changed for this plan.

Observed in-scope files with relevant implementation/test content:
- `config/schema.py` — allowed.
- `gecco/construct_feedback/tool_judge.py` — allowed.
- `gecco/diagnostic_store/tools.py` — allowed.
- `docs/index.html` — allowed.
- `tests/test_phase3_config_validation.py` — allowed.
- `tests/test_judge_mode_context.py` — allowed.
- `config/two_step_factors/deepseekv4flash/*.yaml` — allowed by `config/**/*.yaml`, but only some relevant configs have explicit `individual_differences`.

Suspicious/incomplete scope item:
- Multiple `config/archive/*.yaml` and some runtime/example YAMLs declare `judge.context` without explicit `individual_differences`, contrary to the plan’s config/docs phase.
