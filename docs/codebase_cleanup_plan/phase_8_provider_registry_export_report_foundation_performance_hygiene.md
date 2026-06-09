# Phase 8: Provider Registry, Export/Report Foundation, Performance Hygiene

## 1. Summary

Finish the cleanup by replacing substring-based LLM provider dispatch with exact registry-backed dispatch, and by adding a small DuckDB-backed report/export foundation that does not depend on runtime JSON artifacts.

The main risks this plan prevents are:

- A provider registry is added, but old substring checks still control generation or prompt behavior.
- Config validation still accepts typo or substring provider names.
- A new report/export path silently reads `bics/*.json` or other JSON artifacts instead of DuckDB.
- Phase 8 scope creep touches dashboard, broad scripts, migrations, or unrelated dirty files.

Apply proportionality: this phase should use a small number of focused tests and source audits. Do not add migrations, compatibility aliases, persisted cache manifests, broad report features, or dashboard cleanup unless the user explicitly expands scope.

Follow `docs/codebase_cleanup_plan/implementation_guardrails.md` while implementing this phase.

## 2. Pipeline Map

### Provider Dispatch Pipeline

1. Config file writes `llm.provider` and `llm.base_model`.
2. `config.schema.LLMConfig` validates `llm.provider` against exact registry keys.
3. `gecco.load_llms.provider_registry` owns the canonical provider keys and metadata.
4. `gecco.load_llms.model_loader.load_llm()` resolves the provider through the registry and calls the registered loader.
5. CLI routes call `load_llm()`:
   - `gecco/cli/run_gecco_distributed.py`
   - `gecco/cli/run_judge_orchestrator.py`
6. `GeCCoModelSearch.generate()` reads the same provider metadata to choose the generation API family and structured-output behavior.
7. Prompt generation and warnings read the same metadata instead of ad hoc string membership checks.

Producer-consumer handoff to protect: config validation, model loading, generation behavior, prompt behavior, and launch display must all use the same provider source of truth.

### Report/Export Pipeline

1. Runtime and diagnostic state is written to DuckDB by existing registry/store code.
2. New report/export foundation opens an explicit DuckDB path under a caller-provided results directory or DB path.
3. Summary/query helpers read DuckDB tables/views only.
4. Render/export helpers consume an in-memory summary object produced by the DuckDB reader.
5. Optional HTML/text/JSON-like serializable output may be produced from that in-memory summary, but the input source remains DuckDB.

Producer-consumer handoff to protect: report rendering must consume the summary returned by the DuckDB reader, not rescan `models/`, `bics/`, `feedback/`, `judge/`, or default `results/` directories.

### Performance Hygiene Pipeline

1. DuckDB summary queries should aggregate in DuckDB where practical.
2. Renderer functions should receive one summary object and must not rescan files or rerun expensive table scans per section.
3. If a cache is in-process only, no manifest/freshness mechanism is required.
4. If the implementation adds a persisted report cache, it must also add a freshness check tied to the source DuckDB file and schema version. Prefer not adding a persisted cache in this phase.

## 3. Contract Matrix

| Contract | Asset or interface | Canonical location or source of truth | Writer or producer | Readers or consumers | Required behavior | Validation function or check | Positive test | Negative test | Forbidden fallback or shortcut | Verification command or review check |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C1 | LLM provider registry | `gecco/load_llms/provider_registry.py` | Phase 8 implementation | `config.schema`, `model_loader`, `GeCCoModelSearch.generate`, prompt/generator helpers | Exact provider keys only; metadata includes loader, label, API family, structured-output mode, and system-prompt support where needed | `get_provider_spec(provider)` raises a specific `ValueError` for unknown keys | `test_provider_registry_resolves_exact_supported_keys` | `test_provider_registry_rejects_substring_provider_keys` | Any `"foo" in provider` dispatch, silent alias guessing, fallback to HF/default branch for unknown keys | `grep -R "in provider\|provider in" gecco/load_llms gecco/run_gecco.py gecco/candidate_generation.py gecco/prompt_builder/prompt.py gecco/cli/launch_distributed.py` reviewed for allowed non-dispatch uses only |
| C2 | Config provider validation | `config/schema.py` imports registry keys or shared validation helper | Config loader | All CLI/runtime callers of `load_config()` | `llm.provider` must be one exact registered key; production configs must load; typos/substrings fail before model loading | Pydantic validation through `load_config()` | `test_llm_config_accepts_registered_provider_keys` | `test_llm_config_rejects_unknown_or_substring_provider_key` | Duplicating a hand-written provider list that can drift from runtime registry | `conda run -n gecco_mh pytest tests/test_phase8_provider_registry.py tests/test_phase3_config_validation.py -q` |
| C3 | Provider-backed model loading | `gecco/load_llms/model_loader.py` delegates to registry | Registry loader specs | Distributed client and judge orchestrator CLIs | `load_llm(provider, model_name, **kwargs)` calls only the exact provider's registered loader and preserves existing `(model, tokenizer)` ordering | Loader tests with monkeypatched registered loader callables | `test_load_llm_uses_exact_registered_loader` | `test_load_llm_does_not_route_opencode_go_by_substring` | Keeping old `elif "opencode" in provider` style dispatch under or beside the registry | `conda run -n gecco_mh pytest tests/test_phase8_provider_registry.py -q` plus source grep C1 |
| C4 | Provider-backed generation metadata | `GeCCoModelSearch.generate()` consumes provider metadata, not substring checks | Provider registry | Candidate generation and judge generation calls | OpenAI Responses, Gemini, OpenAI-compatible chat, and HF-style paths are selected by `api_family`; OpenRouter JSON-schema behavior uses metadata/config without substring checks | Focused fake-client tests or source-level dispatch audit | `test_generate_uses_provider_api_family_for_openrouter_chat` | `test_generate_rejects_unknown_provider_before_hf_fallback` | Falling through to HF generation for typo providers; substring checks for labels or structured output | `conda run -n gecco_mh pytest tests/test_phase8_provider_registry.py -q`; grep C1 |
| C5 | DuckDB-backed report summary | New `gecco/reporting.py` or `gecco/reporting/summary.py` | Diagnostic/runtime DuckDB stores | New report/export renderers and tests | Reads from explicit DuckDB path or explicit results directory; returns a small serializable summary; uses no `bics/*.json` or directory scans as input | `load_report_summary(results_dir=..., db_path=...)` or equivalent | `test_report_summary_reads_duckdb_from_custom_results_dir` | `test_report_summary_fails_when_only_json_artifacts_exist` | JSON fallback, default `results/<task>` fallback, directory glob fallback | `conda run -n gecco_mh pytest tests/test_phase8_report_export.py -q`; `grep -R "glob(.*json\|bics.*json\|top_models_test.json" gecco/reporting*` |
| C6 | Report rendering handoff and performance hygiene | Report renderer consumes summary object | New report summary helper | HTML/text/export renderer | Renderer must not reread DuckDB or scan result directories; repeated sections use the already-loaded summary | Test with a sentinel/fake summary and no results files | `test_report_renderer_uses_supplied_summary_without_rescanning` | `test_report_renderer_does_not_open_default_results_dir` | Renderer calling `load_report_summary()` internally without explicit input; hidden module-level results path | `conda run -n gecco_mh pytest tests/test_phase8_report_export.py -q`; source grep for `Path("results")` in reporting module |
| C7 | Legacy JSON report path ownership | `scripts/compile_results.py`, if changed at all | Existing stale script | Human users only; not dashboard | Either left untouched, or replaced/deleted in favor of DuckDB-backed reporting with no JSON fallback. Do not make it a compatibility wrapper that accepts both JSON and DuckDB. | Changed-file audit and source grep | If changed: `test_compile_results_uses_duckdb_report_summary` | If changed: `test_compile_results_fails_without_duckdb_source` | Keeping JSON scan as fallback for convenience | `git diff --name-only -- scripts/compile_results.py`; if changed, `grep -n "glob(.*json\|json.load\|bics" scripts/compile_results.py` |
| C8 | Scope compliance and dirty baseline | Git worktree | Implementer | Reviewer | Phase 8-owned changes are limited to allowed files/symbols; pre-existing dirty files are not counted as Phase 8 evidence | Baseline `git status --short` and `git diff --name-only` before edits; final changed-file comparison | Baseline and final receipt sections exist | Forbidden path newly changed without explicit approval fails review | Modifying dashboard, broad config files, migrations, stale scripts, or unrelated phase tests because they mention providers/reporting | Review commands in Scope Firewall and final receipt |

## 4. Scope Firewall

### Allowed Implementation Files

Only these implementation files may be changed for Phase 8:

- `gecco/load_llms/model_loader.py`
- `gecco/load_llms/provider_registry.py` (new)
- `gecco/load_llms/__init__.py` only if needed to export registry helpers
- `config/schema.py` only within `LLMConfig` provider validation/imports
- `gecco/run_gecco.py` only within `GeCCoModelSearch.generate()` provider dispatch and directly related imports
- `gecco/candidate_generation.py` only within the system-prompt support warning that currently checks HF provider strings
- `gecco/prompt_builder/prompt.py` only within provider-category prompt selection
- `gecco/cli/launch_distributed.py` only within provider display/vLLM launch condition logic
- `gecco/reporting.py` or `gecco/reporting/**` (new)
- `scripts/compile_results.py` only if replacing/removing its JSON report implementation as part of C7; otherwise leave untouched

If an allowed file is already dirty, ownership is by exact hunk/symbol, not by whole file.

### Allowed Test Files

- `tests/test_phase8_provider_registry.py` (new)
- `tests/test_phase8_report_export.py` (new)
- `tests/test_phase3_config_validation.py` only for adding provider-validation cases or updating production-provider expectations
- Existing focused tests may be adjusted only when they fail because provider validation now rejects a previously accepted typo test fixture.

### Allowed Documentation Files

- `docs/codebase_cleanup_plan/phase_8_provider_registry_export_report_foundation_performance_hygiene.md`
- `docs/codebase_cleanup_plan/phase_8_implementation_receipt.md` (required final receipt)

### Explicitly Forbidden Files and Directories

- `gecco-mh-dashboard/**`
- `gecco/diagnostic_store/rebuild.py` unless the user explicitly expands scope to diagnostic import cleanup
- `gecco/diagnostic_store/schema.py` unless a report query needs an existing view bug fixed and the user approves
- `gecco/coordination.py` unless a report query exposes a Phase 7 contract violation and the user approves
- `gecco/artifacts.py` unless the user explicitly expands scope to runtime JSON inspection-output cleanup
- `gecco/cli/run_test_evaluation.py` unless the user explicitly expands scope to test-evaluation output ownership
- `bash/**`
- `config/*.yaml` unless a production provider key must be renamed and the user approves the specific file
- `README.md`
- `.claude/**`
- `scripts/*.py` except `scripts/compile_results.py` under C7
- unrelated tests from Phases 0-7, except the narrow `tests/test_phase3_config_validation.py` allowance above

### Tempting But Forbidden Adjacent Fixes

- Do not clean up dashboard JSON readers.
- Do not rewrite `diagnostic_store/rebuild.py`; it is an import/rebuild utility, not the new report foundation.
- Do not remove all JSON inspection outputs from `ArtifactStore`.
- Do not redesign `run_test_evaluation()` output ownership unless explicitly approved.
- Do not migrate existing result directories or databases.
- Do not add compatibility aliases beyond exact keys required by current configs.
- Do not add a new public CLI surface unless explicitly approved.
- Do not reformat broad files while editing a small provider block.

### What To Do If Scope Appears Insufficient

Stop and ask the user before touching a forbidden file or broadening a symbol-level allowance. A changed file outside the allowed lists makes implementation incomplete unless the user explicitly approves a scope change and the final receipt records that approval.

### Required Scope Review Commands

Run these before final acceptance and summarize the output in the receipt:

```bash
git status --short
git diff --name-only
git diff --name-only -- gecco-mh-dashboard
git diff --name-only -- bash README.md .claude
git diff --name-only -- gecco/diagnostic_store/rebuild.py gecco/diagnostic_store/schema.py gecco/coordination.py gecco/artifacts.py gecco/cli/run_test_evaluation.py
git diff --name-only -- config/*.yaml
git diff --name-only -- scripts
```

If any forbidden path was already dirty in the pre-flight baseline, compare final output against the baseline and report whether Phase 8 added new changes to that path.

## 5. Test Inventory

Only the named tests/checks below count as proof for this phase.

| Contract | Positive proof | Negative proof | Review/audit proof | Verification command |
| --- | --- | --- | --- | --- |
| C1 | `test_provider_registry_resolves_exact_supported_keys` | `test_provider_registry_rejects_substring_provider_keys` | Source grep for provider substring dispatch | `conda run -n gecco_mh pytest tests/test_phase8_provider_registry.py -q` |
| C2 | `test_llm_config_accepts_registered_provider_keys`; existing production config test still passes | `test_llm_config_rejects_unknown_or_substring_provider_key` | Confirm validation imports registry keys/shared helper | `conda run -n gecco_mh pytest tests/test_phase8_provider_registry.py tests/test_phase3_config_validation.py -q` |
| C3 | `test_load_llm_uses_exact_registered_loader` | `test_load_llm_does_not_route_opencode_go_by_substring` | Source grep for old `elif "..." in provider` loader dispatch | `conda run -n gecco_mh pytest tests/test_phase8_provider_registry.py -q` |
| C4 | `test_generate_uses_provider_api_family_for_openrouter_chat` | `test_generate_rejects_unknown_provider_before_hf_fallback` | Source grep for substring generation dispatch | `conda run -n gecco_mh pytest tests/test_phase8_provider_registry.py -q` |
| C5 | `test_report_summary_reads_duckdb_from_custom_results_dir` | `test_report_summary_fails_when_only_json_artifacts_exist` | Grep reporting module for JSON glob/default result fallback | `conda run -n gecco_mh pytest tests/test_phase8_report_export.py -q` |
| C6 | `test_report_renderer_uses_supplied_summary_without_rescanning` | `test_report_renderer_does_not_open_default_results_dir` | Grep reporting module for hidden `Path("results")` or directory scans | `conda run -n gecco_mh pytest tests/test_phase8_report_export.py -q` |
| C7 | If `scripts/compile_results.py` changes: `test_compile_results_uses_duckdb_report_summary` | If changed: `test_compile_results_fails_without_duckdb_source` | `git diff --name-only -- scripts/compile_results.py`; grep for JSON fallback | `conda run -n gecco_mh pytest tests/test_phase8_report_export.py -q` if changed |
| C8 | Pre-flight baseline recorded; final receipt lists new Phase 8 files | Forbidden-path audit shows no new forbidden changes | Scope review commands from Scope Firewall | Manual review plus `git diff --check` |

### Required Test Details

- `test_provider_registry_rejects_substring_provider_keys` must use at least one plausible old substring key, for example `"my-openrouter-proxy"`, and assert the specific provider-validation error mentions unknown or registered provider keys. It must not pass because an unrelated config field is missing.
- `test_load_llm_does_not_route_opencode_go_by_substring` must prove `opencode-go` is handled only if explicitly registered. Use monkeypatched registry loader callables; do not rely on real network clients or environment variables.
- `test_generate_rejects_unknown_provider_before_hf_fallback` must construct a minimal fake `GeCCoModelSearch` or call the relevant helper so that the failure would have fallen through to HF-style generation before the fix. Assert the specific unknown-provider error.
- `test_report_summary_reads_duckdb_from_custom_results_dir` must create a temporary non-default results directory and a DuckDB file there. The default `results/` directory must not be used or required.
- `test_report_summary_fails_when_only_json_artifacts_exist` must create a plausible `bics/iter0_run0.json` without a DuckDB source and assert a specific error. It must not pass because the temp results directory is missing.
- `test_report_renderer_does_not_open_default_results_dir` must use a sentinel default `results/` directory or monkeypatch `Path.open`/DuckDB connect narrowly enough to prove the renderer consumes the supplied summary only. Do not mock the report summary reader in the summary-reader tests.

## 6. Failure Cases To Prevent

- Provider typo `openrouter-local` is accepted because `"openrouter" in provider` still matches.
- `opencode-go` works by accidental substring matching instead of an explicit registered key.
- Config validation has one provider list and runtime dispatch has another list.
- `load_llm()` uses the registry, but `GeCCoModelSearch.generate()` still uses substring dispatch.
- Unknown providers fall through to the HF generation branch and fail later with a confusing tokenizer/model error.
- Prompt construction treats provider groups differently from generation because it keeps its old local list.
- Report summary tests pass because they mock the reader they are supposed to prove.
- Report export reads a stale `bics/*.json` file even though the DuckDB store is absent or different.
- Renderer functions rescan directories for each section instead of using one summary object.
- A stale `scripts/compile_results.py` change keeps JSON fallback for convenience.
- Dashboard files are modified because they mention reports or JSON.
- Dirty pre-existing files are mistaken for Phase 8-owned evidence.

## 7. Implementation Phases

### Phase 0: Pre-Flight Baseline And Scope Approval

#### Goal

Establish the dirty-worktree baseline before behavior changes.

#### Files Allowed To Change

- None.

#### Files Explicitly Forbidden

- All files.

#### Required Steps

1. Run:

   ```bash
   git status --short
   git diff --name-only
   ```

2. Record the exact output for the final receipt.
3. Identify forbidden paths already dirty.
4. If forbidden paths are dirty, stop and ask the user whether to continue under that dirty baseline unless the user has already explicitly approved continuing.
5. Do not revert or modify unrelated pre-existing changes.

#### Phase Gate

Do not proceed unless the dirty baseline is recorded and broad forbidden-path dirt is either absent or explicitly approved for baseline-only comparison.

#### Changed-File Audit

`git diff --name-only` must be unchanged by this phase.

#### Receipt Items

- Exact `git status --short` baseline.
- Exact `git diff --name-only` baseline.
- User approval note if continuing with forbidden paths already dirty.

### Phase 1: Provider Registry Contracts And Tests First

#### Goal

Write focused failing tests for exact provider keys, config validation, model loading, and generation dispatch.

#### Files Allowed To Change

- `tests/test_phase8_provider_registry.py`
- `tests/test_phase3_config_validation.py` only for provider-validation cases if necessary

#### Files Explicitly Forbidden

- Implementation files.
- Dashboard, scripts, YAML configs, and unrelated Phase 0-7 tests.

#### Required Implementation Details

- Add tests before provider implementation.
- Use fake loader functions and fake model clients; do not require API keys or network calls.
- Use non-default/minimal config files for validation tests so a provider error is isolated.

#### Tests To Add First

- `test_provider_registry_resolves_exact_supported_keys`
- `test_provider_registry_rejects_substring_provider_keys`
- `test_llm_config_accepts_registered_provider_keys`
- `test_llm_config_rejects_unknown_or_substring_provider_key`
- `test_load_llm_uses_exact_registered_loader`
- `test_load_llm_does_not_route_opencode_go_by_substring`
- `test_generate_uses_provider_api_family_for_openrouter_chat`
- `test_generate_rejects_unknown_provider_before_hf_fallback`

#### Negative Tests

The four negative tests above must fail for the intended reason before implementation, not because imports are missing. If the registry module does not exist yet, tests may initially fail on import, but after adding a minimal placeholder registry they must fail on behavior before the production fix is completed.

#### Phase Gate

Run:

```bash
conda run -n gecco_mh pytest tests/test_phase8_provider_registry.py -q
```

Expected behavior before implementation: new tests fail for missing registry behavior or old substring behavior.

Do not proceed unless failures map to C1-C4 and no implementation files were changed.

#### Changed-File Audit

Run:

```bash
git diff --name-only
git diff --name-only -- gecco-mh-dashboard bash README.md .claude config/*.yaml scripts
```

No new forbidden-path changes are allowed.

#### Receipt Items

- Test names added.
- Failure summary before implementation.
- Changed-file audit output.

### Phase 2: Implement Exact Provider Registry

#### Goal

Move provider ownership into a registry and remove provider substring dispatch from Phase 8-owned runtime paths.

#### Files Allowed To Change

- `gecco/load_llms/provider_registry.py`
- `gecco/load_llms/model_loader.py`
- `gecco/load_llms/__init__.py` only if needed
- `config/schema.py` only within `LLMConfig` provider validation/imports
- `gecco/run_gecco.py` only within `GeCCoModelSearch.generate()` provider dispatch and directly related imports
- `gecco/candidate_generation.py` only within provider system-prompt support warning
- `gecco/prompt_builder/prompt.py` only within provider-category prompt selection
- `gecco/cli/launch_distributed.py` only within provider display/vLLM condition logic

#### Files Explicitly Forbidden

- `config/*.yaml` unless user approves a specific provider-key rename.
- `gecco-mh-dashboard/**`
- Report/export files.
- Unrelated runner/evaluator/judge logic.

#### Files That May Look Related But Must Not Be Touched

- Backend implementation files such as `gpt_backend.py`, `openrouter_backend.py`, `opencode_backend.py`, unless a test proves their loader signature is incompatible with registry wiring.
- `structured_output.py`; use existing helpers.

#### Required Implementation Details

- Define a small provider spec, preferably a frozen dataclass or named tuple.
- Include exact keys required by existing supported behavior, including `opencode-go` if keeping the current config key.
- Config validation must import registry keys or a shared helper from the registry module. Do not duplicate a separate list.
- `load_llm()` must resolve the spec once and call the spec's loader.
- `GeCCoModelSearch.generate()` must branch on metadata such as `api_family`, not provider substrings.
- Prompt/generator checks must use metadata such as `supports_system_prompt` or `prompt_family`.
- Unknown providers must fail early with a clear error listing or referencing registered keys.

#### Tests To Add First

Already added in Phase 1.

#### Negative Tests

All C1-C4 negative tests must pass after implementation.

#### Phase Gate

Run:

```bash
conda run -n gecco_mh pytest tests/test_phase8_provider_registry.py -q
conda run -n gecco_mh pytest tests/test_phase3_config_validation.py -q
```

Run source audit:

```bash
grep -R "in provider\|provider in" gecco/load_llms gecco/run_gecco.py gecco/candidate_generation.py gecco/prompt_builder/prompt.py gecco/cli/launch_distributed.py
```

Expected behavior: tests pass; grep output has no substring-based dispatch. If grep finds non-dispatch text or comments, document why it is allowed or remove it.

Do not proceed unless C1-C4 are verified.

#### Changed-File Audit

Run:

```bash
git diff --name-only
git diff --name-only -- gecco-mh-dashboard bash README.md .claude config/*.yaml scripts
```

Every new changed file must be in the Phase 2 allowed list.

#### Receipt Items

- Registry keys implemented.
- Tests run and output summary.
- Grep output summary for provider dispatch.
- Changed-file audit output.

### Phase 3: Report/Export Foundation Tests First

#### Goal

Write focused tests proving the new report/export foundation reads DuckDB from explicit custom paths and does not fall back to JSON artifacts or default result roots.

#### Files Allowed To Change

- `tests/test_phase8_report_export.py`

#### Files Explicitly Forbidden

- Implementation files.
- Dashboard files.
- Existing report scripts.

#### Required Implementation Details

- Tests must create a temporary non-default results directory.
- Tests must create a real DuckDB store with minimal data, not mock away the reader being tested.
- Negative JSON-fallback test must create plausible JSON files so the old wrong behavior would pass if implemented.

#### Tests To Add First

- `test_report_summary_reads_duckdb_from_custom_results_dir`
- `test_report_summary_fails_when_only_json_artifacts_exist`
- `test_report_renderer_uses_supplied_summary_without_rescanning`
- `test_report_renderer_does_not_open_default_results_dir`

If `scripts/compile_results.py` will be changed under C7, also add:

- `test_compile_results_uses_duckdb_report_summary`
- `test_compile_results_fails_without_duckdb_source`

#### Negative Tests

The JSON-only and default-results tests must isolate their target failure modes and assert specific errors.

#### Phase Gate

Run:

```bash
conda run -n gecco_mh pytest tests/test_phase8_report_export.py -q
```

Expected behavior before implementation: tests fail for missing report module or missing DuckDB-only behavior.

Do not proceed unless failures map to C5-C7 and no implementation files were changed.

#### Changed-File Audit

Run forbidden-path checks from the Scope Firewall.

#### Receipt Items

- Test names added.
- Failure summary before implementation.
- Changed-file audit output.

### Phase 4: Implement DuckDB-Backed Report/Export Foundation

#### Goal

Add the smallest useful report/export foundation that reads DuckDB once and renders from an in-memory summary.

#### Files Allowed To Change

- `gecco/reporting.py` or `gecco/reporting/**`
- `scripts/compile_results.py` only if implementing C7 replacement/deletion with no JSON fallback
- `tests/test_phase8_report_export.py` only to adjust tests for final public function names

#### Files Explicitly Forbidden

- `gecco-mh-dashboard/**`
- `gecco/diagnostic_store/rebuild.py`
- `gecco/artifacts.py`
- `gecco/cli/run_test_evaluation.py`
- `gecco/diagnostic_store/schema.py` unless user-approved

#### Files That May Look Related But Must Not Be Touched

- Dashboard report/views/data-adapter files.
- JSON inspection-output writers.
- Diagnostic rebuild/import logic.
- Test-evaluation JSON output.

#### Required Implementation Details

- Provide a small summary reader with an explicit `results_dir` or `db_path` argument.
- Use DuckDB tables/views only. Prefer existing `DiagnosticStore` or direct DuckDB read-only connection.
- Do not scan `bics/`, `models/`, `feedback/`, or `judge/` as report inputs.
- Return a serializable summary object or plain dict containing only fields tests require, such as task/result name, iteration count, best models, and metric trajectory.
- Renderer functions must accept the summary object and return output text/HTML. They must not open files or DuckDB internally.
- Use one query or a small fixed set of aggregate queries. Do not loop over files or per-section rescans.
- Do not add persisted report caches. If a persisted cache is added despite this guidance, stop and add freshness checks before proceeding.

#### Tests To Add First

Already added in Phase 3.

#### Negative Tests

All C5-C7 negative tests must pass after implementation.

#### Phase Gate

Run:

```bash
conda run -n gecco_mh pytest tests/test_phase8_report_export.py -q
```

Run source audits:

```bash
grep -R "glob(.*json\|bics.*json\|top_models_test.json\|Path(\"results\")" gecco/reporting* || true
git diff --name-only -- gecco-mh-dashboard gecco/diagnostic_store/rebuild.py gecco/artifacts.py gecco/cli/run_test_evaluation.py
```

Expected behavior: tests pass; grep has no JSON/default-root fallback in the new reporting module; forbidden-path diff shows no Phase 8-owned changes.

Do not proceed unless C5-C7 are verified or explicitly marked not applicable because `scripts/compile_results.py` was left untouched.

#### Changed-File Audit

Run full Scope Firewall commands. Every new changed file must be allowed.

#### Receipt Items

- Public report helper names.
- Tests run and output summary.
- Source audit summaries.
- Whether `scripts/compile_results.py` was untouched, changed, or deferred.

### Phase 5: Final Integration And Review

#### Goal

Run focused verification, source audits, and write the implementation receipt.

#### Files Allowed To Change

- `docs/codebase_cleanup_plan/phase_8_implementation_receipt.md`
- Test or implementation files already changed in earlier phases only for fixing failures found by final verification

#### Files Explicitly Forbidden

- Any new file outside the allowed lists unless user-approved.

#### Required Implementation Details

- Do not add new work in this phase except fixes directly required by failing Phase 8 tests or audits.
- If desired behavior is already present, avoid unnecessary implementation edits and strengthen only missing tests/checks.

#### Phase Gate

Run:

```bash
conda run -n gecco_mh pytest tests/test_phase8_provider_registry.py -q
conda run -n gecco_mh pytest tests/test_phase8_report_export.py -q
conda run -n gecco_mh pytest tests/test_phase3_config_validation.py -q
git diff --check
```

Run all Scope Firewall review commands.

Do not call Phase 8 complete unless every contract row is verified or explicitly documented as not applicable with a reason.

#### Changed-File Audit

Final receipt must list:

- Files newly changed by Phase 8.
- Files already dirty before Phase 8 and untouched by Phase 8.
- Exact symbols or test names changed inside broad allowed files.
- Output summaries from every forbidden-path audit.

#### Receipt Items

Write `docs/codebase_cleanup_plan/phase_8_implementation_receipt.md`. Work is incomplete if this file is missing, vague, or omits a contract row.

## 8. Forbidden Patterns

- Hardcoded default paths such as `Path("results")` inside report readers/renderers.
- Hidden module-level global state as the only way to pass provider, path, DB, or config information.
- Accepting both correct DuckDB input and JSON fallback input for the new report foundation.
- Backward-compatibility fallbacks that mask provider typos or missing DuckDB files.
- Existence-only validation for structured config/provider behavior.
- Tests that only check command strings when runtime behavior matters.
- Tests that mock or monkeypatch the validator, reader, or writer whose behavior they are meant to prove.
- Tests that assert only that an error happened without asserting the specific contract error.
- Opportunistic refactors, formatting-only edits, documentation edits, dashboard edits, migrations, or cleanup files not required by the named contracts.
- Broad exception handling, compatibility wrappers, or fallback behavior added because it is convenient rather than required.
- Removing or rewriting suspicious adjacent behavior whose ownership cannot be proven from the pre-flight baseline.
- Substring provider dispatch, including `"openrouter" in provider`, `"opencode" in provider`, `"hf" in provider`, or equivalent partial matching.
- Duplicating registry key lists in config validation instead of sharing registry constants/helpers.
- Persisted report caches without a freshness check tied to the source DuckDB file and schema version.

## 9. Final Verification

Run these exact commands with the `gecco_mh` conda environment for Python/test commands:

```bash
conda run -n gecco_mh pytest tests/test_phase8_provider_registry.py -q
conda run -n gecco_mh pytest tests/test_phase8_report_export.py -q
conda run -n gecco_mh pytest tests/test_phase3_config_validation.py -q
git diff --check
git status --short
git diff --name-only
git diff --name-only -- gecco-mh-dashboard
git diff --name-only -- bash README.md .claude
git diff --name-only -- gecco/diagnostic_store/rebuild.py gecco/diagnostic_store/schema.py gecco/coordination.py gecco/artifacts.py gecco/cli/run_test_evaluation.py
git diff --name-only -- config/*.yaml
git diff --name-only -- scripts
grep -R "in provider\|provider in" gecco/load_llms gecco/run_gecco.py gecco/candidate_generation.py gecco/prompt_builder/prompt.py gecco/cli/launch_distributed.py
grep -R "glob(.*json\|bics.*json\|top_models_test.json\|Path(\"results\")" gecco/reporting* || true
```

### Implementation Verification Receipt

The implementer must write `docs/codebase_cleanup_plan/phase_8_implementation_receipt.md` with:

- Exact tests and commands run.
- Which contract rows each command covers.
- Any contract rows not verified and why.
- Tests expected to fail before the fix and pass after the fix.
- Manual review checks performed, including grep output summaries for forbidden patterns.
- Pre-flight `git status --short` summary.
- Pre-flight `git diff --name-only` baseline.
- Final changed-file list for this plan.
- Files newly changed by this plan versus already dirty before this plan.
- Exact symbols, test names, fixtures, classes, or code blocks changed when scope was narrower than whole-file level.
- Confirmation that every changed file is listed as allowed by the Scope Firewall.
- Output summaries from forbidden-path audits.
- Any tempting adjacent fixes intentionally deferred because they were out of scope.
- Explicit note if `scripts/compile_results.py`, `gecco/cli/run_test_evaluation.py`, dashboard files, or diagnostic rebuild code were intentionally left untouched.

Work is incomplete if the receipt is missing, vague, or omits a contract row.

## 10. Residual Risks

- This plan does not prove full end-to-end LLM generation across all real providers. It uses fake clients and exact registry contracts to keep tests fast and deterministic.
- This plan does not remove all JSON inspection outputs. It only prevents the new report/export foundation from depending on them.
- This plan does not clean dashboard JSON/report behavior.
- This plan does not migrate existing results directories, existing DuckDB files, or historical JSON artifacts.
- This plan does not redesign `run_test_evaluation()` output ownership unless the user explicitly expands scope.
- Source greps may need human review for false positives in comments or harmless text. Any allowed false positive must be documented in the receipt.
