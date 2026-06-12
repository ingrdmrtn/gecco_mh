# Review: judge-capability-refactor-plan-2

## 1. Findings

### High — Static/LLM diagnostic context calls diagnostic tools directly

- **Files:** `gecco/construct_feedback/tool_judge.py:270-283`, `gecco/construct_feedback/tool_judge.py:1927-1935`, `gecco/construct_feedback/tool_judge.py:2015-2023`
- **Violates:** Contract B (`static` must not call diagnostic tools); Contract D (`llm` must not call diagnostic tools); forbidden pattern against exposing forbidden information before gating.
- `_build_static_diagnostic_summary()` imports and calls diagnostic-store tool functions (`get_recovery`, `get_ppc`, `get_block_residuals`, `get_individual_differences`, plus performance-backed diagnostic helpers). Both `static` mode and the shared `llm`/`agent` prompt-context path call this helper when `context.diagnostic` is enabled.
- This means `static` and `llm` diagnostic feedback are produced by the same diagnostic tool layer the plan explicitly reserves/gates, rather than by deterministic non-tool context assembly.

### High — Recovery shortcut bypasses mode/context gates and can reuse old verdict content

- **Files:** `gecco/construct_feedback/tool_judge.py:1864-1894`, `gecco/construct_feedback/tool_judge.py:2292-2400`, `tests/test_phase4_orchestrated_judge.py:596-615`
- **Violates:** Contract B (static context only; no conclusions/recommendations), Contract D/F (disabled context must not appear in prompts/output), phase 4/5 mode gating.
- `get_feedback_analysis()` checks the recovery-failure shortcut before the `static` and LLM/agent context assembly branches. `_try_shortcut_from_recovery_failure()` then returns previous persisted `synthesized_feedback` with a recovery-failure note, regardless of the current mode or enabled context.
- The test at `tests/test_phase4_orchestrated_judge.py:596-615` codifies this behavior for `mode="static"`, so static mode can output previous verdict prose/recommendations rather than only deterministic enabled context.

### Medium — Documentation still describes retired capability/tool behavior

- **File:** `docs/index.html:500-518`, `docs/index.html:628-648`
- **Violates:** Manual review check to update docs for the new schema; forbidden/stale `capabilities` behavior review check.
- The docs still say post-processing is based on enabled `capabilities`, describe a universal 14-tool judge analysis path, mention `cited_models`, and introduce analytical angles with “when tools is enabled” / “enabled capabilities”. This conflicts with the new `mode`/`context`/`output` contract and could mislead users about static/LLM/random behavior.

## 2. Open Questions

None blocking.

## 3. Verification Summary

- I read the plan in full and reviewed the relevant implementation/test/docs paths by inspection.
- I did not independently rerun pytest or `git status` in this environment; an existing review artifact records a targeted pytest run as passing, but the issues above are code-path violations not addressed by passing tests.
- Contract rows verified by inspection: A mostly implemented; E mostly implemented for agent schema/dispatch gating.
- Contract rows not fully satisfied: B, D, F due the findings above; documentation/manual review check remains incomplete.

## 4. Scope Audit

- Changed files observed from the existing review artifact are within the plan’s allowed implementation/test/config/doc scope: `config/schema.py`, `gecco/construct_feedback/orchestrated.py`, `gecco/construct_feedback/tool_judge.py`, `gecco/diagnostic_store/tools.py`, `gecco/run_gecco.py`, `docs/index.html`, `config/**/*.yaml`, and allowed tests.
- Remaining suspicious/stale patterns: `docs/index.html` still contains capability-era prose; tests still include a static-mode recovery shortcut that conflicts with the new static contract.
