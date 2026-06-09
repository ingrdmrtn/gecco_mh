# Phase 8 Review Findings

## 1. Findings

### High: Provider registry accepts non-exact case variants

- Violates: C1, C2; Forbidden Patterns; Phase 2 required exact provider keys only.
- References: `gecco/load_llms/provider_registry.py:174-178`, `config/schema.py:65-69`.
- Issue: `get_provider_spec()` lowercases the supplied provider before lookup, so unregistered case variants such as `OpenAI` are accepted even though the registry key is `openai` and the plan requires exact registered keys only.
- Proof observed: `conda run -n gecco_mh python -c "from gecco.load_llms.provider_registry import get_provider_spec; print(get_provider_spec('OpenAI').key)"` printed `openai`.
- Test gap: `tests/test_phase8_provider_registry.py:80-83` and `tests/test_phase3_config_validation.py:168-182` reject substring-like providers but do not reject case variants, so C1/C2 exactness is not fully proven.

```python
# gecco/load_llms/provider_registry.py
key = provider.lower()
return PROVIDER_REGISTRY[key]
```

### High: C8 receipt is incomplete and cannot prove scope/dirty-baseline compliance

- Violates: C8; Phase 0 receipt items; Phase 5 receipt items; Implementation Verification Receipt.
- References: `docs/codebase_cleanup_plan/phase_8_implementation_receipt.md:3-6`, `docs/codebase_cleanup_plan/phase_8_implementation_receipt.md:53-61`.
- Issue: The plan requires exact pre-flight `git status --short` and exact pre-flight `git diff --name-only` output, plus final changed-file lists and forbidden-path audit outputs. The receipt only summarizes that forbidden paths were dirty and says the diff “matched the same pre-existing dirty set”. Because many forbidden paths are currently dirty, the review cannot verify whether those changes were pre-existing, Phase 8-owned, or changed further during Phase 8.
- Current forbidden-path diffs observed: `gecco-mh-dashboard/**`, `bash/**`, `README.md`, `.claude/**`, `config/*.yaml`, `gecco/artifacts.py`, `gecco/coordination.py`, `gecco/diagnostic_store/schema.py`, and multiple `scripts/*.py` paths.
- Action required: Add exact baseline outputs and exact final forbidden-path audit outputs, or otherwise provide a comparable baseline artifact. Without that, C8 remains unverifiable.

### Medium: Report summary does not aggregate directly in DuckDB and reads full registry state

- Violates: Performance Hygiene Pipeline; Phase 4 required implementation details for small fixed aggregate queries where practical.
- References: `gecco/reporting.py:75-87`, `gecco/coordination.py:180-293`.
- Issue: `load_report_summary()` calls `SharedRegistry.read()` and then summarizes the full registry snapshot in Python. `SharedRegistry.read()` loads unrelated state including `runtime_tried_param_sets`, `runtime_client_entries`, `runtime_candidate_generations`, `runtime_generator_status`, and `runtime_judge_iterations`, not just the iteration/result aggregates required by the report summary.
- Risk: The report foundation avoids JSON fallback, but it can still load substantially more DuckDB state than the summary requires and does not demonstrate DuckDB-side aggregation for best models/trajectory.
- Action required: Query only the tables/columns needed for the returned summary, preferably with DuckDB aggregation/windowing for best-per-iteration and best-overall rows.

## 2. Open Questions

- None blocking review confidence.

## 3. Verification Summary

Commands/checks observed during review:

- `conda run -n gecco_mh pytest tests/test_phase8_provider_registry.py -q` -> `6 passed`.
- `conda run -n gecco_mh pytest tests/test_phase8_report_export.py -q` -> `4 passed`.
- `conda run -n gecco_mh pytest tests/test_phase3_config_validation.py -q` -> `39 passed`.
- `git diff --check` -> no whitespace errors; CRLF warnings were emitted for several dirty files.
- `grep -R "in provider\|provider in" gecco/load_llms gecco/run_gecco.py gecco/candidate_generation.py gecco/prompt_builder/prompt.py gecco/cli/launch_distributed.py` -> no output.
- `grep -R "glob(.*json\|bics.*json\|top_models_test.json\|Path(\"results\")" gecco/reporting* || true` -> no output.
- Scope Firewall audit commands were run and showed many currently dirty forbidden paths.

Contract rows verified:

- C3: `load_llm()` delegates to the registered exact-key loader for matching lower-case keys, and substring-like `opencode-go` variants are rejected by tests.
- C4: `GeCCoModelSearch.generate()` uses registry `api_family` metadata and rejects unknown substring-like providers before HF fallback.
- C5: New reporting reads from explicit DuckDB paths and rejects JSON-only artifacts in the covered test case.
- C6: Renderers consume supplied summaries and do not open files/DuckDB in the covered test case.
- C7: `scripts/compile_results.py` appears not changed by Phase 8 implementation; current script changes are other deleted script paths, not `scripts/compile_results.py`.

Contract rows not fully verified:

- C1: Not fully verified because non-exact case variants are accepted.
- C2: Not fully verified because config validation inherits the non-exact case acceptance.
- C8: Not verified because the receipt omits exact baseline and final audit outputs needed to distinguish Phase 8 changes from pre-existing dirty files.

## 4. Scope Audit

Phase 8-owned files claimed in receipt and Scope Firewall status:

- `gecco/load_llms/provider_registry.py` -> allowed.
- `gecco/reporting.py` -> allowed.
- `tests/test_phase8_provider_registry.py` -> allowed.
- `tests/test_phase8_report_export.py` -> allowed.
- `config/schema.py` -> allowed only within `LLMConfig` provider validation/imports; observed provider-validation change is within scope.
- `gecco/load_llms/model_loader.py` -> allowed.
- `gecco/prompt_builder/prompt.py` -> allowed only within provider-category prompt selection; observed provider metadata usage is within scope.
- `tests/test_phase3_config_validation.py` -> allowed only for provider-validation cases or production-provider expectations; provider-validation additions are within scope.
- `gecco/run_gecco.py` -> allowed only within `GeCCoModelSearch.generate()` provider dispatch/imports; observed provider metadata usage is within scope.
- `gecco/candidate_generation.py` -> allowed only within system-prompt support warning; observed provider metadata usage is within scope.
- `gecco/cli/launch_distributed.py` -> allowed only within provider display/vLLM launch condition logic; observed provider metadata usage is within scope.
- `docs/codebase_cleanup_plan/phase_8_implementation_receipt.md` -> required and allowed, but incomplete as noted above.

Currently dirty forbidden or suspicious paths:

- `.claude/settings.json` -> forbidden by Scope Firewall; receipt claims pre-existing but exact baseline proof is missing.
- `README.md` -> forbidden; receipt claims pre-existing but exact baseline proof is missing.
- `bash/**` -> forbidden; receipt claims pre-existing but exact baseline proof is missing.
- `config/*.yaml` -> forbidden unless explicitly approved; receipt claims pre-existing but exact baseline proof is missing.
- `gecco-mh-dashboard/**` -> forbidden; receipt claims pre-existing but exact baseline proof is missing.
- `gecco/artifacts.py`, `gecco/coordination.py`, `gecco/diagnostic_store/schema.py` -> forbidden or approval-gated; receipt claims pre-existing but exact baseline proof is missing.
- `scripts/*.py` deletions -> forbidden except `scripts/compile_results.py` under C7; receipt claims scripts were not edited by Phase 8, but exact baseline proof is missing.

Forbidden patterns checked:

- Provider substring dispatch grep returned no matches in the Phase 8-audited paths.
- Reporting JSON/default-results fallback grep returned no matches in `gecco/reporting*`.
- Non-exact provider key normalization remains present via `provider.lower()` in the registry.
