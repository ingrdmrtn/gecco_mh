# Cleanup Acceptance Gaps Review Fix Receipt

## Baseline

- Pre-flight `git status --short`:
  ```
  M .claude/settings.json
  M README.md
  M bash/launch_vllm_server.sh
  M bash/run_cmg_generator.sh
  M bash/run_gecco_distributed.sh
  M bash/run_judge_orchestrator.sh
  M bash/run_test_evaluation.sh
  M config/judge_tool_example.yaml
  M config/schema.py
  M config/test_orchestrator.yaml
  M config/two_step_factors_cmg.yaml
  M config/two_step_factors_deepseekv4flash.yaml
  M config/two_step_factors_deepseekv4pro.yaml
  M config/two_step_factors_distributed.yaml
  M config/two_step_factors_gemini31_flash_lite.yaml
  M config/two_step_factors_gemini3flash.yaml
  M config/two_step_factors_gemini3flash_generic.yaml
  D config/two_step_factors_gemini3flash_generic_lesion_complete.yaml
  D config/two_step_factors_gemini3flash_generic_lesion_no_citations.yaml
  D config/two_step_factors_gemini3flash_generic_lesion_no_diagnostics.yaml
  D config/two_step_factors_gemini3flash_generic_lesion_no_recommendations.yaml
  D config/two_step_factors_gemini3flash_generic_lesion_no_tools.yaml
  D config/two_step_factors_gemini3flash_generic_lesion_no_tools_cmg.yaml
  D config/two_step_factors_gemini3flash_generic_lesion_no_tools_cmg_smoke.yaml
  D config/two_step_factors_gemini3flash_generic_lesion_noise.yaml
  D config/two_step_factors_gemini3flash_generic_lesion_summary_only.yaml
  M config/two_step_factors_gemma4.yaml
  M config/two_step_factors_glm5.yaml
  M config/two_step_factors_gpt54nano.yaml
  M config/two_step_factors_gpt55.yaml
  M config/two_step_factors_minimax_m2.yaml
  M config/two_step_factors_minimax_m27.yaml
  M config/two_step_factors_nemotron.yaml
  M config/two_step_factors_opencode_go.yaml
  M config/two_step_factors_qwen36plus_judge.yaml
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
  M gecco/diagnostic_store/store.py
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
  ?? config/two_step_factors_gemini3flash_capabilities_full.yaml
  ?? config/two_step_factors_gemini3flash_capabilities_no_citations.yaml
  ?? config/two_step_factors_gemini3flash_capabilities_no_diagnostics.yaml
  ?? config/two_step_factors_gemini3flash_capabilities_no_recommendations.yaml
  ?? config/two_step_factors_gemini3flash_capabilities_no_tools.yaml
  ?? config/two_step_factors_gemini3flash_capabilities_no_tools_cmg.yaml
  ?? config/two_step_factors_gemini3flash_capabilities_no_tools_cmg_smoke.yaml
  ?? config/two_step_factors_gemini3flash_capabilities_random_feedback.yaml
  ?? config/two_step_factors_gemini3flash_capabilities_summary_only.yaml
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
- Pre-existing dirty files: all entries above were present before this plan's edits.

## Progress

- Baseline recorded before implementation.
- Migrated the receipt to the required `cleanup-acceptance-gaps-review-fix-receipt.md` path.

## Implementation

- Updated `gecco/construct_feedback/tool_judge.py` so no-recommendations modes no longer receive next-iteration/actionable wording in the shared judge prompt.
- Updated `tests/test_phase3_config_validation.py` to discover production configs from the full `config/*.yaml` surface with explicit non-production exclusions and to assert no actionable next-iteration wording leaks into the captured LLM-bound prompts.
- Removed the prior wrong receipt path after copying its baseline content into the required fix receipt path.

## Verification

- Phase-gate note for Step 2: before adding the new analysis-phase test, source inspection showed the prior prompt test only exercised `synthesize_for_persona()` and read `messages = chat_create.call_args.kwargs["messages"]`; it did not call `get_feedback_analysis()` or capture the analysis-phase LLM-bound prompt. I used that inspection result as the pre-fix rationale, then added the missing test.
- `conda run -n gecco_mh pytest tests/test_phase3_config_validation.py::test_get_feedback_analysis_captures_no_recommendations_analysis_messages -q`
  ```
  .                                                                        [100%]
  1 passed in 2.21s
  ```
- `conda run -n gecco_mh pytest tests/test_phase3_config_validation.py::test_orchestrated_persona_synthesis_captures_capability_limited_llm_messages -q`
  ```
  ..                                                                       [100%]
  2 passed in 2.24s
  ```
- `conda run -n gecco_mh pytest tests/test_phase3_config_validation.py -q`
  ```
  ............................................................             [100%]
  60 passed in 3.08s
  ```
- Final `git status --short`
  ```
   M .claude/settings.json
   M README.md
   M bash/launch_vllm_server.sh
   M bash/run_cmg_generator.sh
   M bash/run_gecco_distributed.sh
   M bash/run_judge_orchestrator.sh
   M bash/run_test_evaluation.sh
   M config/judge_tool_example.yaml
   M config/schema.py
   M config/test_orchestrator.yaml
   M config/two_step_factors_cmg.yaml
   M config/two_step_factors_deepseekv4flash.yaml
   M config/two_step_factors_deepseekv4pro.yaml
   M config/two_step_factors_distributed.yaml
   M config/two_step_factors_gemini31_flash_lite.yaml
   M config/two_step_factors_gemini3flash.yaml
   M config/two_step_factors_gemini3flash_generic.yaml
   D config/two_step_factors_gemini3flash_generic_lesion_complete.yaml
   D config/two_step_factors_gemini3flash_generic_lesion_no_citations.yaml
   D config/two_step_factors_gemini3flash_generic_lesion_no_diagnostics.yaml
   D config/two_step_factors_gemini3flash_generic_lesion_no_recommendations.yaml
   D config/two_step_factors_gemini3flash_generic_lesion_no_tools.yaml
   D config/two_step_factors_gemini3flash_generic_lesion_no_tools_cmg.yaml
   D config/two_step_factors_gemini3flash_generic_lesion_no_tools_cmg_smoke.yaml
   D config/two_step_factors_gemini3flash_generic_lesion_noise.yaml
   D config/two_step_factors_gemini3flash_generic_lesion_summary_only.yaml
   M config/two_step_factors_gemma4.yaml
   M config/two_step_factors_glm5.yaml
   M config/two_step_factors_gpt54nano.yaml
   M config/two_step_factors_gpt55.yaml
   M config/two_step_factors_minimax_m2.yaml
   M config/two_step_factors_minimax_m27.yaml
   M config/two_step_factors_nemotron.yaml
   M config/two_step_factors_opencode_go.yaml
   M config/two_step_factors_qwen36plus_judge.yaml
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
   M gecco/diagnostic_store/store.py
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
  ?? config/two_step_factors_gemini3flash_capabilities_full.yaml
  ?? config/two_step_factors_gemini3flash_capabilities_no_citations.yaml
  ?? config/two_step_factors_gemini3flash_capabilities_no_diagnostics.yaml
  ?? config/two_step_factors_gemini3flash_capabilities_no_recommendations.yaml
  ?? config/two_step_factors_gemini3flash_capabilities_no_tools.yaml
  ?? config/two_step_factors_gemini3flash_capabilities_no_tools_cmg.yaml
  ?? config/two_step_factors_gemini3flash_capabilities_no_tools_cmg_smoke.yaml
  ?? config/two_step_factors_gemini3flash_capabilities_random_feedback.yaml
  ?? config/two_step_factors_gemini3flash_capabilities_summary_only.yaml
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
- Final `git diff --name-only`
  ```
  warning: CRLF will be replaced by LF in config/two_step_factors_gemma4.yaml.
  The file will have its original line endings in your working directory
  warning: CRLF will be replaced by LF in config/two_step_factors_glm5.yaml.
  The file will have its original line endings in your working directory
  warning: CRLF will be replaced by LF in config/two_step_factors_minimax_m2.yaml.
  The file will have its original line endings in your working directory
  warning: CRLF will be replaced by LF in config/two_step_factors_minimax_m27.yaml.
  The file will have its original line endings in your working directory
  warning: CRLF will be replaced by LF in config/two_step_factors_nemotron.yaml.
  The file will have its original line endings in your working directory
  warning: CRLF will be replaced by LF in config/two_step_factors_opencode_go.yaml.
  The file will have its original line endings in your working directory
  warning: CRLF will be replaced by LF in config/two_step_factors_step35.yaml.
  The file will have its original line endings in your working directory
  warning: CRLF will be replaced by LF in gecco-mh-dashboard/app.py.
  The file will have its original line endings in your working directory
  warning: CRLF will be replaced by LF in gecco/load_llms/model_loader.py.
  The file will have its original line endings in your working directory
  .claude/settings.json
  README.md
  bash/launch_vllm_server.sh
  bash/run_cmg_generator.sh
  bash/run_gecco_distributed.sh
  bash/run_judge_orchestrator.sh
  bash/run_test_evaluation.sh
  config/judge_tool_example.yaml
  config/schema.py
  config/test_orchestrator.yaml
  config/two_step_factors_cmg.yaml
  config/two_step_factors_deepseekv4flash.yaml
  config/two_step_factors_deepseekv4pro.yaml
  config/two_step_factors_distributed.yaml
  config/two_step_factors_gemini31_flash_lite.yaml
  config/two_step_factors_gemini3flash.yaml
  config/two_step_factors_gemini3flash_generic.yaml
  config/two_step_factors_gemini3flash_generic_lesion_complete.yaml
  config/two_step_factors_gemini3flash_generic_lesion_no_citations.yaml
  config/two_step_factors_gemini3flash_generic_lesion_no_diagnostics.yaml
  config/two_step_factors_gemini3flash_generic_lesion_no_recommendations.yaml
  config/two_step_factors_gemini3flash_generic_lesion_no_tools.yaml
  config/two_step_factors_gemini3flash_generic_lesion_no_tools_cmg.yaml
  config/two_step_factors_gemini3flash_generic_lesion_no_tools_cmg_smoke.yaml
  config/two_step_factors_gemini3flash_generic_lesion_noise.yaml
  config/two_step_factors_gemini3flash_generic_lesion_summary_only.yaml
  config/two_step_factors_gemma4.yaml
  config/two_step_factors_glm5.yaml
  config/two_step_factors_gpt54nano.yaml
  config/two_step_factors_gpt55.yaml
  config/two_step_factors_minimax_m2.yaml
  config/two_step_factors_minimax_m27.yaml
  config/two_step_factors_nemotron.yaml
  config/two_step_factors_opencode_go.yaml
  config/two_step_factors_qwen36plus_judge.yaml
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
  gecco/diagnostic_store/store.py
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
- Final `git diff --name-only -- gecco-mh-dashboard`
  ```
  warning: CRLF will be replaced by LF in gecco-mh-dashboard/app.py.
  The file will have its original line endings in your working directory
  gecco-mh-dashboard/app.py
  gecco-mh-dashboard/dashboard/config.py
  ```
- Receipt path check
  ```
  $ ls "docs/codebase_cleanup_plan/cleanup-acceptance-gaps-review-fix-receipt.md"
  docs/codebase_cleanup_plan/cleanup-acceptance-gaps-review-fix-receipt.md
  $ ls "docs/codebase_cleanup_plan/cleanup-acceptance-gaps-review-receipt.md"
  ls: cannot access 'docs/codebase_cleanup_plan/cleanup-acceptance-gaps-review-receipt.md': No such file or directory
  ```

## Contract Coverage

- Capability-limited judge prompts: covered by the prompt-construction change in `gecco/construct_feedback/tool_judge.py`, the new analysis-phase prompt-capture assertion in `tests/test_phase3_config_validation.py`, and the synthesis-phase prompt-capture assertions in the same file.
- Lesion-free production config surface: covered by the dynamic `config/*.yaml` discovery and lesion-name assertions in `tests/test_phase3_config_validation.py`.
- Receipt evidence: covered by this receipt file at the required path.

## Files Changed For This Plan

- `docs/codebase_cleanup_plan/cleanup-acceptance-gaps-review-fix-receipt.md`
- `gecco/construct_feedback/tool_judge.py`
- `tests/test_phase3_config_validation.py`
- Deleted `docs/codebase_cleanup_plan/cleanup-acceptance-gaps-review-receipt.md`

## Final Audit

- The full untracked audit is captured in the exact final `git status --short` block above.
- No dashboard, scripts, or unrelated config files were edited by this plan.
- The wrong receipt path `docs/codebase_cleanup_plan/cleanup-acceptance-gaps-review-receipt.md` was removed as an implementation-owned artifact.

## Deferred

- All unrelated pre-existing worktree changes were intentionally left untouched.
