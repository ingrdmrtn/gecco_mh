# Phase 3 remediation: Config validation follow-up

## Purpose

Close the remaining gaps in Phase 3 so production configs load cleanly, existing lesion configs keep their current behaviour, and validation stays strict without introducing a broad capability-permission framework.

## Relationship to original Phase 3 plan

This is a follow-up to `docs/codebase_cleanup_plan/phase_3_config_validation.md`.

The original plan aimed to:

- make runtime loading respect the schema,
- replace lesion-first config naming with additive `judge.capabilities`,
- fail fast on invalid combinations,
- delete old lesion names/shims,
- ensure empty `judge.capabilities: []` yields explicit empty feedback.

A review showed the tests pass, but the implementation still has important mismatches with production configs and some lesion behaviours are not yet restored.

## Current gaps to fix

1. `config/schema.py` currently requires `TaskConfig.instructions`, but production configs use `task.goal` and runtime also uses `task.goal`; production configs fail `load_config()` because `instructions` is missing.
2. `config/two_step_factors_gemini3flash_generic.yaml` has no `judge.capabilities`, so schema loading would default it to `[]` and produce empty feedback. It should represent the full/non-lesioned judge condition.
3. Runtime currently only checks `tools` and `best_model_code` capabilities. Other declared capabilities exist, but most are not actually consumed.
4. `random_feedback` exists in the schema and `lesion_noise` config, but has no runtime behaviour.
5. `gecco/construct_feedback/judge_lesion.py` is dead code and should be deleted once replacement behaviours are covered.
6. Existing tests are synthetic and do not load production configs.

## Non-goals

- Do not build a broad/full capability permission system.
- Do not redesign the judge pipeline.
- Do not add new config families beyond what is needed for the current configs.
- Do not add `instructions` to every production config unless a real runtime consumer is introduced.
- Do not preserve dead compatibility shims once behaviour is covered by explicit capability logic.

## Capability behaviour target

The goal is to make the current configs behave according to their intended Phase 3 lesion/full-judge meanings, because today's runtime behaviour is partly broken.

| Config / capability set | Expected behaviour |
|---|---|
| `judge.capabilities: []` | Explicit empty trace and explicit empty feedback; no substantive judge output. |
| Full/non-lesioned config (`two_step_factors_gemini3flash_generic.yaml`) with `attempted_models_overview`, `performance_summary`, `best_model_code`, `recommendations`, `mechanistic_coherence`, `tools`, `citations`, `coverage` | Normal judge feedback with diagnostics, citations, recommendations, model-code appendix, and summary content. No `random_feedback`. |
| `random_feedback` only | Deterministic generic/noise feedback only. No tool calls, no normal judge synthesis. Prefer a fixed string over a new schema field. |
| No `tools` | Avoid diagnostic tool calls, but still allow non-tool feedback from precomputed summary text if other substantive capabilities are present. |
| `attempted_models_overview` + `performance_summary` only (`summary_only`) | Concise quantitative/progress summary only. No recommendations, citations, mechanistic analysis, diagnostics, or best-model code unless explicitly enabled. |
| No `recommendations` | `key_recommendations: []` and no recommendation / next-step section in `synthesized_feedback`. |
| No `citations` | Remove model IDs, model names, and iteration-specific citations from synthesized feedback. |
| No `coverage` | Avoid PPC/residual/diagnostic-detail sections and other diagnostic-depth content tied to coverage. |
| `best_model_code` present | Continue to control whether best-model code is appended. |

Notes:

- Keep the behaviour tied to the existing configs and current runtime shape.
- Prefer helper functions in `gecco/construct_feedback/tool_judge.py` over resurrecting `JudgeLesion`.

## TDD-first implementation plan

### Chunk A: Align schema with real runtime config

1. Write tests first that production configs load through `load_config()`.
2. Cover at least these representative configs:
   - generic full,
   - complete/empty,
   - noise,
   - no_tools,
   - no_recommendations,
   - no_citations,
   - no_diagnostics,
   - summary_only,
   - CMG variant if present.
3. Update `TaskConfig` so `instructions` is optional with an empty default, or remove it if that is clearly safe.
4. Prefer the optional/default-empty approach because it is the least disruptive and matches the runtime use of `task.goal`.
5. Do not backfill `instructions` into every config.

### Chunk B: Make the full/generic config explicit

1. Add an explicit full capability list to `config/two_step_factors_gemini3flash_generic.yaml`.
2. Define the full set in the config and in tests as:
   - `attempted_models_overview`
   - `performance_summary`
   - `best_model_code`
   - `recommendations`
   - `mechanistic_coherence`
   - `tools`
   - `citations`
   - `coverage`
3. Exclude `random_feedback` from the full config.
4. Apply the same explicit full list to any other full/non-lesion judge config that should produce normal judge feedback.

### Chunk C: Restore existing lesion behaviours using capability-aware runtime logic

1. Keep the runtime focused on current configs only.
2. `[]` should remain an explicit empty feedback case.
3. Add deterministic handling for `random_feedback`.
4. Make `no_tools` skip tool-based diagnostics, but still allow summary-based or synthesized content from other capabilities.
5. Make `summary_only` produce concise quantitative/progress feedback only.
6. Make `no_recommendations` suppress recommendation content and return `key_recommendations: []`.
7. Make `no_citations` scrub model IDs/names/iteration-specific citations from synthesized feedback.
8. Make `no_diagnostics` suppress PPC/residual/diagnostic-detail sections.
9. Preserve `best_model_code` behaviour.
10. Keep helper functions small and deterministic; prefer additions to `tool_judge.py`.
11. Once helper coverage is in place, delete `gecco/construct_feedback/judge_lesion.py`.

### Chunk D: Tighten validation without over-engineering

1. Add a duplicate-capability validation test.
2. Add validation for `random_feedback` combinations.
3. Recommended rule: `random_feedback` must be the only capability, because mixing it with substantive feedback is ambiguous.
4. Keep the existing `persona_synthesis` dependency test as-is.
5. Avoid a full dependency graph unless current configs require it.

### Chunk E: Tests and cleanup

1. Add tests for production config loading.
2. Add unit tests for capability post-processing helpers without calling real LLMs.
3. Update tests for empty capabilities so the trace remains explicit.
4. Delete the dead `judge_lesion.py` only after replacement tests pass.

## Implementation chunks

### Chunk A — Schema/runtime alignment

Dependencies: none.

Work:

- Make `TaskConfig.instructions` optional/default-empty.
- Add loading tests for production configs.
- Confirm `load_config()` succeeds for configs that rely on `task.goal`.

### Chunk B — Explicit full config

Dependencies: Chunk A.

Work:

- Update `config/two_step_factors_gemini3flash_generic.yaml` with the full capability list.
- Mirror the same list in any other non-lesioned config that should behave normally.

### Chunk C — Capability-aware feedback shaping

Dependencies: Chunks A and B.

Work:

- Add deterministic helper functions in `tool_judge.py` for post-processing/suppression.
- Cover `random_feedback`, summary-only, no-citations, no-recommendations, and no-diagnostics behaviour.
- Keep tool usage gated only where needed.
- Remove dead lesion code after helpers are covered.

### Chunk D — Validation tightening

Dependencies: Chunk C.

Work:

- Add duplicate-capability validation coverage.
- Add `random_feedback` exclusivity validation.
- Preserve current persona dependency validation.

### Chunk E — Cleanup

Dependencies: Chunks C and D.

Work:

- Delete `gecco/construct_feedback/judge_lesion.py` only after all replacement tests are green.

## Detailed file-by-file guidance

### `config/schema.py`

- Make `TaskConfig.instructions` optional with a default empty value.
- Keep the schema aligned with production configs that use `task.goal`.
- Avoid adding new schema fields unless they are required by a tested runtime behaviour.

### `config/two_step_factors_gemini3flash_generic.yaml`

- Add the explicit full capability list.
- Make sure this config represents the non-lesioned judge condition, not the empty-capability condition.

### Other config YAMLs under `config/`

- Update any existing full/non-lesioned config that should behave normally.
- Keep lesion configs as small deltas from the full config where practical.

### `gecco/construct_feedback/tool_judge.py`

- Add small helper functions for:
  - empty-capability handling,
  - random feedback output,
  - recommendation suppression,
  - citation scrubbing,
  - diagnostic suppression,
  - capability post-processing that does not duplicate best-model-code appending owned by `run_gecco.py`.
- Keep functions deterministic and easy to test.

### `gecco/run_gecco.py`

- Keep the existing `best_model_code` capability append behaviour here unless the implementer chooses a cleaner helper.
- Do not duplicate best-model-code appending in `tool_judge.py`.
- Ensure empty capabilities still skip feedback cleanly.
- Ensure capability post-processing in `tool_judge.py` and final feedback assembly here do not fight each other.

### `gecco/construct_feedback/judge_lesion.py`

- Remove this file after its behaviours are fully replaced and tests pass.

### Config loading / CLI / runtime tests

- Add production-config load coverage.
- Add capability-behaviour tests for the specific existing lesion configs.
- Add tests for explicit empty traces.

## Tests to add or update

1. Production config loading tests that call `load_config()` on real YAML files.
2. A test that the generic/full config loads and exposes the explicit full capability set.
3. A test for empty `judge.capabilities: []` that checks trace and feedback are explicit.
4. A test for `random_feedback` only.
5. A test for `summary_only` content suppression.
6. A test for `no_tools` ensuring no diagnostic tool calls are made.
7. A test for `no_recommendations` returning an empty recommendation list.
8. A test for `no_citations` scrubbing model identifiers / iteration citations.
9. A test for `no_diagnostics` suppressing diagnostic sections.
10. A validation test for duplicate capabilities.
11. A validation test for invalid `random_feedback` combinations.

## Manual verification commands

Use the `gecco_mh` conda environment for all Python commands:

- `conda run -n gecco_mh python -m pytest tests/<relevant_test_file>.py`
- `conda run -n gecco_mh python -m pytest tests/<relevant_test_file_1>.py tests/<relevant_test_file_2>.py`
- `conda run -n gecco_mh python -m pytest`

Run the smallest relevant set after each chunk, then run the full suite before cleanup.

## Acceptance criteria

- Production configs load successfully through `load_config()`.
- The generic full config explicitly declares the full non-lesioned capability set.
- Existing lesion configs keep their intended behaviour.
- Empty `judge.capabilities: []` yields explicit empty feedback and trace.
- `random_feedback` is deterministic and isolated.
- Duplicate capabilities are rejected.
- Invalid `random_feedback` mixes are rejected.
- Dead lesion code is removed after replacement coverage exists.

## What not to do

- Do not introduce a broad capability permission system.
- Do not rename the entire config model just to satisfy a single field mismatch.
- Do not add `instructions` to configs as a workaround for the schema bug.
- Do not preserve dead shims once equivalent behaviour exists.
- Do not call real LLMs in unit tests.
- Do not make lesion behaviour rely on ad hoc branching that is hard to test.

## Risks and mitigations

- **Risk:** Optionalizing `instructions` may hide future mismatches.
  - **Mitigation:** Add load tests against real configs and keep schema/runtime alignment explicit.
- **Risk:** Scrubbing citations too aggressively may remove useful summary text.
  - **Mitigation:** Target only model IDs, model names, and iteration-specific citations tied to the lesioned configs.
- **Risk:** `random_feedback` could become ambiguous when mixed with other capabilities.
  - **Mitigation:** Validate it as an exclusive mode.
- **Risk:** Deleting `judge_lesion.py` too early may break hidden paths.
  - **Mitigation:** Remove it only after helper tests and config-loading tests pass.

## Suggested implementation order

1. Fix `TaskConfig.instructions` and add real config-loading tests.
2. Make the generic config explicit with the full capability set.
3. Implement deterministic helper logic for existing lesion configs.
4. Add validation for duplicates and `random_feedback` exclusivity.
5. Update tests for empty traces and no-citation/no-recommendation/no-diagnostics cases.
6. Delete `gecco/construct_feedback/judge_lesion.py` after all tests pass.
