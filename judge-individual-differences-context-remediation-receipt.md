# Individual Differences Context Remediation Receipt

## Pre-flight
- `git status --short` baseline:
  - `M config/schema.py`
  - `M config/two_step_factors/deepseekv4flash/judge_agent_all_context.yaml`
  - `M config/two_step_factors/deepseekv4flash/judge_agent_all_context_persona.yaml`
  - `M config/two_step_factors/deepseekv4flash/judge_llm_all_context_persona.yaml`
  - `M config/two_step_factors/deepseekv4flash/judge_llm_attempted_performance_code.yaml`
  - `M config/two_step_factors/deepseekv4flash/judge_static_all_context.yaml`
  - `M config/two_step_factors/deepseekv4flash/judge_static_all_context_persona.yaml`
  - `M docs/index.html`
  - `M gecco/construct_feedback/tool_judge.py`
  - `M gecco/diagnostic_store/tools.py`
  - `M tests/test_judge_mode_context.py`
  - `M tests/test_phase3_config_validation.py`
  - `?? judge-individual-differences-context-plan.md`
  - `?? judge-individual-differences-context-review.md`
  - `?? uv-migration-plan.md`
- Pre-existing dirty files left untouched by this remediation: the 12 tracked files above plus the 3 untracked plan/review artifacts.

## Final status
- `git status --short` after remediation still contains the same pre-existing dirty files above, plus this remediation’s edits:
  - 27 YAML files under `config/**/*.yaml`
  - `docs/index.html`
  - `tests/test_judge_mode_context.py`
  - `judge-individual-differences-context-remediation-receipt.md`

## Files changed by this remediation
- 27 YAML configs with explicit `judge.context.individual_differences: false`:
  - `config/archive/judge_tool_example.yaml`
  - `config/archive/test_orchestrator.yaml`
  - `config/archive/two_step_factors_cmg.yaml`
  - `config/archive/two_step_factors_deepseekv4flash.yaml`
  - `config/archive/two_step_factors_deepseekv4pro.yaml`
  - `config/archive/two_step_factors_gemini31_flash_lite.yaml`
  - `config/archive/two_step_factors_gemini3flash.yaml`
  - `config/archive/two_step_factors_gemini3flash_capabilities_full.yaml`
  - `config/archive/two_step_factors_gemini3flash_capabilities_no_citations.yaml`
  - `config/archive/two_step_factors_gemini3flash_capabilities_no_diagnostics.yaml`
  - `config/archive/two_step_factors_gemini3flash_capabilities_no_recommendations.yaml`
  - `config/archive/two_step_factors_gemini3flash_capabilities_no_tools.yaml`
  - `config/archive/two_step_factors_gemini3flash_capabilities_no_tools_cmg.yaml`
  - `config/archive/two_step_factors_gemini3flash_capabilities_no_tools_cmg_smoke.yaml`
  - `config/archive/two_step_factors_gemini3flash_capabilities_random_feedback.yaml`
  - `config/archive/two_step_factors_gemini3flash_capabilities_summary_only.yaml`
  - `config/archive/two_step_factors_gemini3flash_generic.yaml`
  - `config/archive/two_step_factors_gpt54nano.yaml`
  - `config/archive/two_step_factors_gpt54nano_judge.yaml`
  - `config/archive/two_step_factors_gpt55.yaml`
  - `config/archive/two_step_factors_qwen36plus_judge.yaml`
  - `config/two_step_factors/deepseekv4flash/judge_llm_attempted.yaml`
  - `config/two_step_factors/deepseekv4flash/judge_llm_attempted_performance_code.yaml`
  - `config/two_step_factors/deepseekv4flash/judge_off.yaml`
  - `config/two_step_factors/deepseekv4flash/judge_random.yaml`
  - `config/two_step_factors/deepseekv4flash/judge_static_attempted.yaml`
  - `config/two_step_factors/deepseekv4flash/judge_static_attempted_performance.yaml`
- `docs/index.html`
- `tests/test_judge_mode_context.py`
- `judge-individual-differences-context-remediation-receipt.md`

## Verification
- `conda run -n gecco_mh pytest tests/test_phase3_config_validation.py` → 39 passed.
- `conda run -n gecco_mh pytest tests/test_judge_mode_context.py` → 40 passed.
- `conda run -n gecco_mh pytest tests/test_phase4_orchestrated_judge.py tests/test_judge_orchestration.py tests/test_judge_enhancements.py` → 30 passed.
- YAML audit/load script → `missing_count 0`; `loaded_count 26`; `skipped_count 1`.
  - Skipped file: `config/archive/judge_tool_example.yaml` (illustrative judge-only snippet; not a full `GeCCoConfig`).
- Docs path audit → all changed config paths exist (`config/archive/...` references and `config/two_step_factors/deepseekv4flash/judge_off.yaml`).

## Test coverage added
- New provider-loop forced-call tests in `tests/test_judge_mode_context.py` request forbidden `get_individual_differences` and `get_participant_best_models` calls under `individual_differences: false` for both OpenAI and Gemini mocks.
- The tests assert the loop trace records `Forbidden tool` and monkeypatch the ID tool functions to fail if executed.

## Scope / residual risk
- Changed files stayed within the allowed scope.
- No plan/review files, results/data, notebooks, dashboards, or unrelated code were edited.
- Residual risk: `config/archive/judge_tool_example.yaml` is intentionally illustrative and cannot be validated with `config.schema.load_config` because required top-level GeCCo sections are absent.
