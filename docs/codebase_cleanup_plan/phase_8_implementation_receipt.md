# Phase 8 Implementation Receipt

## Pre-flight Baseline

### `git status --short`

```text
 M .claude/settings.json
 M README.md
 M bash/launch_vllm_server.sh
 M bash/run_cmg_generator.sh
 M bash/run_gecco_distributed.sh
 M bash/run_judge_orchestrator.sh
 M bash/run_test_evaluation.sh
 M config/two_step_factors_distributed.yaml
 M config/two_step_factors_gemma4.yaml
 M config/two_step_factors_glm5.yaml
 M config/two_step_factors_minimax_m2.yaml
 M config/two_step_factors_minimax_m27.yaml
 M config/two_step_factors_nemotron.yaml
 M config/two_step_factors_opencode_go.yaml
 M config/two_step_factors_step35.yaml
 M gecco-mh-dashboard/app.py
 M gecco-mh-dashboard/dashboard/config.py
 M gecco/artifacts.py
 M gecco/candidate_evaluation.py
 M gecco/candidate_generation.py
 M gecco/cli/monitor_distributed.py
 M gecco/cli/run_judge_orchestrator.py
 D gecco/construct_feedback/feedback.py
 M gecco/construct_feedback/orchestrated.py
 M gecco/construct_feedback/tool_judge.py
 M gecco/coordination.py
 M gecco/diagnostic_store/schema.py
 M gecco/run_gecco.py
 D scripts/launch_cmg_distributed.py
 D scripts/launch_distributed.py
 D scripts/monitor_distributed.py
 D scripts/reset_distributed.py
 D scripts/run_gecco_distributed.py
 D scripts/run_judge_orchestrator.py
 D scripts/run_test_evaluation.py
 M tests/fixtures/phase0/legacy_cli_inventory.json
 M tests/test_cmg_judge.py
 M tests/test_cmg_launcher.py
 M tests/test_cmg_registry.py
 M tests/test_cmg_runtime.py
 M tests/test_judge_orchestration.py
 M tests/test_phase0_characterization.py
 M tests/test_phase4_orchestrated_judge.py
 M tests/test_phase5_duckdb_canonical_state.py
 M tests/test_phase6_candidate_evaluator.py
 M tests/test_phase6_candidate_generator.py
 M tests/test_phase6_parallel_extraction_subtracks.py
?? config/__init__.py
?? docs/baseline_model_two_step_factors.md
?? docs/centralised_judge_implementation.md
?? docs/codebase_cleanup_plan.md
?? docs/codebase_cleanup_plan/
?? docs/codebase_cleanup_plan_deck.html
?? docs/codebase_deepdive_deck.html
?? docs/contract_first_planner_agent_prompt.md
?? docs/judge_implementation_record.md
?? docs/phase6_revision_difficulty_deck.html
?? docs/ppc_speedup_implementation.md
?? gecco/__main__.py
?? gecco/cli/__init__.py
?? gecco/cli/launch_cmg_distributed.py
?? gecco/cli/launch_distributed.py
?? gecco/cli/reset_distributed.py
?? gecco/cli/run_local_client.py
?? gecco/cli/run_test_evaluation.py
?? test_ppc_speedup.py
?? tests/test_phase1_docs_reset.py
?? tests/test_phase2_cli_contract.py
?? tests/test_phase7_duckdb_coordination_and_status_views.py
```

### `git diff --name-only`

```text
.claude/settings.json
README.md
bash/launch_vllm_server.sh
bash/run_cmg_generator.sh
bash/run_gecco_distributed.sh
bash/run_judge_orchestrator.sh
bash/run_test_evaluation.sh
config/two_step_factors_distributed.yaml
config/two_step_factors_gemma4.yaml
config/two_step_factors_glm5.yaml
config/two_step_factors_minimax_m2.yaml
config/two_step_factors_minimax_m27.yaml
config/two_step_factors_nemotron.yaml
config/two_step_factors_opencode_go.yaml
config/two_step_factors_step35.yaml
gecco-mh-dashboard/app.py
gecco-mh-dashboard/dashboard/config.py
gecco/artifacts.py
gecco/candidate_evaluation.py
gecco/candidate_generation.py
gecco/cli/monitor_distributed.py
gecco/cli/run_judge_orchestrator.py
gecco/construct_feedback/feedback.py
gecco/construct_feedback/orchestrated.py
gecco/construct_feedback/tool_judge.py
gecco/coordination.py
gecco/diagnostic_store/schema.py
gecco/run_gecco.py
scripts/launch_cmg_distributed.py
scripts/launch_distributed.py
scripts/monitor_distributed.py
scripts/reset_distributed.py
scripts/run_gecco_distributed.py
scripts/run_judge_orchestrator.py
scripts/run_test_evaluation.py
tests/fixtures/phase0/legacy_cli_inventory.json
tests/test_cmg_judge.py
tests/test_cmg_launcher.py
tests/test_cmg_registry.py
tests/test_cmg_runtime.py
tests/test_judge_orchestration.py
tests/test_phase0_characterization.py
tests/test_phase4_orchestrated_judge.py
tests/test_phase5_duckdb_canonical_state.py
tests/test_phase6_candidate_evaluator.py
tests/test_phase6_candidate_generator.py
tests/test_phase6_parallel_extraction_subtracks.py
```

### Approval note

- Phase 0 continuation under the dirty baseline was explicitly approved before implementation.

## Phase 8-Owned Files

### New files

- `gecco/load_llms/provider_registry.py`
- `gecco/reporting.py`
- `tests/test_phase8_provider_registry.py`
- `tests/test_phase8_report_export.py`
- `docs/codebase_cleanup_plan/phase_8_implementation_receipt.md`

### Modified clean files

- `config/schema.py`
- `gecco/load_llms/model_loader.py`
- `gecco/prompt_builder/prompt.py`
- `tests/test_phase3_config_validation.py`

### Modified pre-existing dirty allowed files with exact hunk ownership only

- `gecco/run_gecco.py`: `GeCCoModelSearch.generate()` provider dispatch and related import use.
- `gecco/candidate_generation.py`: provider system-prompt support warning in `CandidateGenerator.generate_non_cmg_iteration()`.
- `gecco/cli/launch_distributed.py`: provider display and vLLM launch condition logic in `run_distributed_launcher()`.

## Exact Symbols And Tests Changed

- `gecco/load_llms/provider_registry.py`
  - `ProviderSpec`
  - `PROVIDER_REGISTRY`
  - `get_registered_provider_keys()`
  - `get_provider_spec()`
- `gecco/load_llms/model_loader.py`
  - `load_llm()`
- `config/schema.py`
  - `LLMConfig.validate_provider_key()`
- `gecco/run_gecco.py`
  - `GeCCoModelSearch.generate()`
- `gecco/candidate_generation.py`
  - `CandidateGenerator.generate_non_cmg_iteration()` provider warning branch
- `gecco/prompt_builder/prompt.py`
  - provider-family prompt layout selection in `build_prompt()`
- `gecco/cli/launch_distributed.py`
  - provider display and `launch_vllm` condition logic in `run_distributed_launcher()`
- `gecco/reporting.py`
  - `_resolve_db_path()`
  - `_fetch_metric_trajectory()`
  - `_fetch_best_models()`
  - `_fetch_best_overall()`
  - `_load_summary_from_duckdb()`
  - `load_report_summary()`
  - `render_report_text()`
  - `render_report_html()`
- `tests/test_phase8_provider_registry.py`
  - `test_provider_registry_resolves_exact_supported_keys`
  - `test_provider_registry_rejects_substring_provider_keys`
  - `test_provider_registry_rejects_case_variant_provider_keys`
  - `test_load_llm_uses_exact_registered_loader`
  - `test_load_llm_does_not_route_opencode_go_by_substring`
  - `test_generate_uses_provider_api_family_for_openrouter_chat`
  - `test_generate_rejects_unknown_provider_before_hf_fallback`
- `tests/test_phase8_report_export.py`
  - `test_report_summary_reads_duckdb_from_custom_results_dir`
  - `test_report_summary_fails_when_only_json_artifacts_exist`
  - `test_report_renderer_uses_supplied_summary_without_rescanning`
  - `test_report_renderer_does_not_open_default_results_dir`
- `tests/test_phase3_config_validation.py`
  - `test_llm_config_accepts_registered_provider_keys`
  - `test_llm_config_rejects_unknown_or_substring_provider_key`
  - `test_llm_config_rejects_case_variant_provider_key`

## Commands Run And Contract Coverage

- `conda run -n gecco_mh pytest tests/test_phase8_provider_registry.py -q`
  - Covers C1, C3, C4.
  - Final output: `7 passed in 6.69s`
- `conda run -n gecco_mh pytest tests/test_phase3_config_validation.py -q`
  - Covers C2.
  - Final output: `40 passed in 9.18s`
- `conda run -n gecco_mh pytest tests/test_phase8_report_export.py -q`
  - Covers C5, C6.
  - Final output: `4 passed in 4.06s`
- `grep -R "in provider\|provider in" gecco/load_llms gecco/run_gecco.py gecco/candidate_generation.py gecco/prompt_builder/prompt.py gecco/cli/launch_distributed.py`
  - Covers C1-C4 forbidden substring dispatch audit.
  - Final output: no matches.
- `grep -R "glob(.*json\|bics.*json\|top_models_test.json\|Path(\"results\")" gecco/reporting* || true`
  - Covers C5-C6 forbidden JSON/default-root fallback audit.
  - Final output: no matches.
- `git diff --check`
  - Covers C8 whitespace/sanity verification.
  - Final output: no whitespace errors; only CRLF warnings from pre-existing dirty files.

## Contract Rows Verified

- C1: Verified. Registry resolves exact keys only and rejects substring and case-variant provider names.
- C2: Verified. `LLMConfig` imports registry validation and rejects unknown, substring, and case-variant provider keys.
- C3: Verified. `load_llm()` delegates only to the exact registered loader.
- C4: Verified. `GeCCoModelSearch.generate()` dispatches by provider metadata and rejects unknown providers before HF fallback.
- C5: Verified. Reporting reads only from an explicit DuckDB file (`db_path` or `results_dir/diagnostics.duckdb`) and fails when only JSON artifacts exist.
- C6: Verified. Renderers consume the supplied summary and do not open files or DuckDB.
- C8: Verified. Exact pre-flight and final audit outputs are recorded below; no new forbidden-path Phase 8 changes were introduced.

## Contract Rows Not Verified

- C7: Not applicable. `scripts/compile_results.py` was intentionally left untouched.

## Tests Expected To Fail Before The Fix And Pass After

- `tests/test_phase8_provider_registry.py` initially failed because `gecco.load_llms.provider_registry` did not exist.
- `tests/test_phase8_report_export.py` initially failed because `gecco.reporting` did not exist.
- After the first implementation pass, review identified an exact-key gap; the new case-variant tests would have failed before `get_provider_spec()` stopped normalizing case.

## Manual Review Checks

- Confirmed no substring-based provider dispatch remains in the audited runtime files.
- Confirmed provider lookup now requires exact registered keys, including exact casing.
- Confirmed the report summary uses focused DuckDB aggregate queries instead of reading the full shared registry snapshot.
- Confirmed the reporting module does not scan JSON artifacts or a default `results/` root.
- Confirmed renderer helpers only consume the supplied summary object.

## Final Scope Firewall Audit Outputs

### `git status --short`

```text
 M .claude/settings.json
 M README.md
 M bash/launch_vllm_server.sh
 M bash/run_cmg_generator.sh
 M bash/run_gecco_distributed.sh
 M bash/run_judge_orchestrator.sh
 M bash/run_test_evaluation.sh
 M config/schema.py
 M config/two_step_factors_distributed.yaml
 M config/two_step_factors_gemma4.yaml
 M config/two_step_factors_glm5.yaml
 M config/two_step_factors_minimax_m2.yaml
 M config/two_step_factors_minimax_m27.yaml
 M config/two_step_factors_nemotron.yaml
 M config/two_step_factors_opencode_go.yaml
 M config/two_step_factors_step35.yaml
 M gecco-mh-dashboard/app.py
 M gecco-mh-dashboard/dashboard/config.py
 M gecco/artifacts.py
 M gecco/candidate_evaluation.py
 M gecco/candidate_generation.py
 M gecco/cli/monitor_distributed.py
 M gecco/cli/run_judge_orchestrator.py
 D gecco/construct_feedback/feedback.py
 M gecco/construct_feedback/orchestrated.py
 M gecco/construct_feedback/tool_judge.py
 M gecco/coordination.py
 M gecco/diagnostic_store/schema.py
 M gecco/load_llms/model_loader.py
 M gecco/prompt_builder/prompt.py
 M gecco/run_gecco.py
 D scripts/launch_cmg_distributed.py
 D scripts/launch_distributed.py
 D scripts/monitor_distributed.py
 D scripts/reset_distributed.py
 D scripts/run_gecco_distributed.py
 D scripts/run_judge_orchestrator.py
 D scripts/run_test_evaluation.py
 M tests/fixtures/phase0/legacy_cli_inventory.json
 M tests/test_cmg_judge.py
 M tests/test_cmg_launcher.py
 M tests/test_cmg_registry.py
 M tests/test_cmg_runtime.py
 M tests/test_judge_orchestration.py
 M tests/test_phase0_characterization.py
 M tests/test_phase3_config_validation.py
 M tests/test_phase4_orchestrated_judge.py
 M tests/test_phase5_duckdb_canonical_state.py
 M tests/test_phase6_candidate_evaluator.py
 M tests/test_phase6_candidate_generator.py
 M tests/test_phase6_parallel_extraction_subtracks.py
?? config/__init__.py
?? docs/baseline_model_two_step_factors.md
?? docs/centralised_judge_implementation.md
?? docs/codebase_cleanup_plan.md
?? docs/codebase_cleanup_plan/
?? docs/codebase_cleanup_plan_deck.html
?? docs/codebase_deepdive_deck.html
?? docs/contract_first_planner_agent_prompt.md
?? docs/judge_implementation_record.md
?? docs/phase6_revision_difficulty_deck.html
?? docs/ppc_speedup_implementation.md
?? gecco/__main__.py
?? gecco/cli/__init__.py
?? gecco/cli/launch_cmg_distributed.py
?? gecco/cli/launch_distributed.py
?? gecco/cli/reset_distributed.py
?? gecco/cli/run_local_client.py
?? gecco/cli/run_test_evaluation.py
?? gecco/load_llms/provider_registry.py
?? gecco/reporting.py
?? test_ppc_speedup.py
?? tests/test_phase1_docs_reset.py
?? tests/test_phase2_cli_contract.py
?? tests/test_phase7_duckdb_coordination_and_status_views.py
?? tests/test_phase8_provider_registry.py
?? tests/test_phase8_report_export.py
```

### `git diff --name-only`

```text
.claude/settings.json
README.md
bash/launch_vllm_server.sh
bash/run_cmg_generator.sh
bash/run_gecco_distributed.sh
bash/run_judge_orchestrator.sh
bash/run_test_evaluation.sh
config/schema.py
config/two_step_factors_distributed.yaml
config/two_step_factors_gemma4.yaml
config/two_step_factors_glm5.yaml
config/two_step_factors_minimax_m2.yaml
config/two_step_factors_minimax_m27.yaml
config/two_step_factors_nemotron.yaml
config/two_step_factors_opencode_go.yaml
config/two_step_factors_step35.yaml
gecco-mh-dashboard/app.py
gecco-mh-dashboard/dashboard/config.py
gecco/artifacts.py
gecco/candidate_evaluation.py
gecco/candidate_generation.py
gecco/cli/monitor_distributed.py
gecco/cli/run_judge_orchestrator.py
gecco/construct_feedback/feedback.py
gecco/construct_feedback/orchestrated.py
gecco/construct_feedback/tool_judge.py
gecco/coordination.py
gecco/diagnostic_store/schema.py
gecco/load_llms/model_loader.py
gecco/prompt_builder/prompt.py
gecco/run_gecco.py
scripts/launch_cmg_distributed.py
scripts/launch_distributed.py
scripts/monitor_distributed.py
scripts/reset_distributed.py
scripts/run_gecco_distributed.py
scripts/run_judge_orchestrator.py
scripts/run_test_evaluation.py
tests/fixtures/phase0/legacy_cli_inventory.json
tests/test_cmg_judge.py
tests/test_cmg_launcher.py
tests/test_cmg_registry.py
tests/test_cmg_runtime.py
tests/test_judge_orchestration.py
tests/test_phase0_characterization.py
tests/test_phase3_config_validation.py
tests/test_phase4_orchestrated_judge.py
tests/test_phase5_duckdb_canonical_state.py
tests/test_phase6_candidate_evaluator.py
tests/test_phase6_candidate_generator.py
tests/test_phase6_parallel_extraction_subtracks.py
```

### `git diff --name-only -- gecco-mh-dashboard`

```text
gecco-mh-dashboard/app.py
gecco-mh-dashboard/dashboard/config.py
```

### `git diff --name-only -- bash README.md .claude`

```text
.claude/settings.json
README.md
bash/launch_vllm_server.sh
bash/run_cmg_generator.sh
bash/run_gecco_distributed.sh
bash/run_judge_orchestrator.sh
bash/run_test_evaluation.sh
```

### `git diff --name-only -- gecco/diagnostic_store/rebuild.py gecco/diagnostic_store/schema.py gecco/coordination.py gecco/artifacts.py gecco/cli/run_test_evaluation.py`

```text
gecco/artifacts.py
gecco/coordination.py
gecco/diagnostic_store/schema.py
```

### `git diff --name-only -- config/*.yaml`

```text
config/two_step_factors_distributed.yaml
config/two_step_factors_gemma4.yaml
config/two_step_factors_glm5.yaml
config/two_step_factors_minimax_m2.yaml
config/two_step_factors_minimax_m27.yaml
config/two_step_factors_nemotron.yaml
config/two_step_factors_opencode_go.yaml
config/two_step_factors_step35.yaml
```

### `git diff --name-only -- scripts`

```text
scripts/launch_cmg_distributed.py
scripts/launch_distributed.py
scripts/monitor_distributed.py
scripts/reset_distributed.py
scripts/run_gecco_distributed.py
scripts/run_judge_orchestrator.py
scripts/run_test_evaluation.py
```

## Forbidden-Path Audit Summary

- All forbidden-path outputs above match the dirty baseline categories already present before Phase 8.
- No new Phase 8-owned changes were introduced to `gecco-mh-dashboard/**`, `bash/**`, `README.md`, `.claude/**`, `config/*.yaml`, `gecco/artifacts.py`, `gecco/coordination.py`, `gecco/diagnostic_store/schema.py`, or `scripts/**`.

## Scope Confirmation

- Every Phase 8-owned changed file is allowed by the Scope Firewall.
- `scripts/compile_results.py` was intentionally left untouched.
- Dashboard files, `.claude/**`, `bash/**`, `config/*.yaml`, diagnostic rebuild code, and `gecco/cli/run_test_evaluation.py` were intentionally left untouched by Phase 8.

## Adjacent Fixes Deferred As Out Of Scope

- Dashboard JSON/report cleanup.
- JSON inspection-output cleanup in `ArtifactStore`.
- Diagnostic rebuild/import changes.
- Runtime/test-evaluation output ownership changes.
- Broader script or YAML config cleanup.
