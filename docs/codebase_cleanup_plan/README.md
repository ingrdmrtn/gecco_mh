# Cleanup plan overview

## Goal

Refactor the repository into a smaller, clearer, deletion-first system that a junior developer can follow end to end.

## Workflow overview

- Read this README first. It is the whole-plan overview, not a phase file.
- Each linked phase file is one scoped work package with its own boundaries and tests.
- If you are assigned one phase file, implement only that phase unless you are explicitly assigned more.
- “Can run in parallel with” means parallel ownership is possible; it does not mean one implementer should automatically do both phases.

## How to execute this plan

Follow these rules from the start:

- Work phase by phase; do not try to do the whole cleanup at once.
- Start with Phase 0 and freeze current behavior first.
- Write characterization tests first, exactly as the TDD strategy says, before major refactors or deletions.
- Breaking changes are allowed, but do not delete old behavior until the replacement path is covered by tests and docs are updated.
- Remove old judge lesion config names once the new capability model is in place and tested; do not keep long-term compatibility wrappers.
- Delete old `scripts/*.py` entrypoints once the new CLI routes are covered by tests; do not preserve them as long-term wrappers.
- Dashboard code is out of scope for this cleanup and should not be touched.
- Ignore untracked files unless they are directly relevant to this cleanup; do not modify unrelated untracked files.
- Create a new git branch before starting, and use staged commits to keep the work grouped into small, reviewable units.

## How to assign work

- Assign one phase file per implementer unless you are intentionally grouping multiple phases.
- Require every implementer to read this README before starting their phase file.
- Use separate branches/PRs, or staged commits per phase/subtrack where practical.
- For any parallel phase pair, keep ownership separate unless the coordinator explicitly combines them.

## Core decisions

- DuckDB is the canonical runtime state store.
- JSON is not the runtime source of truth.
- Runtime JSON outputs are not part of the target architecture.
- The judge path is orchestrated-only, even for a single generator/worker.
- The judge uses explicit additive capabilities/add-ins; the default is no substantive feedback unless capabilities are enabled.
- Dashboard work is out of scope for this plan.
- Old names, shims, and legacy code should be deleted once the replacement path is tested.

## Target architecture summary

The target system has six main pieces:

1. Validated config loading through Pydantic.
2. One public CLI family (`gecco ...`) that routes to modules instead of doing the work itself.
3. One orchestrated judge pipeline with explicit capabilities/add-ins and post-processors.
4. DuckDB as the canonical runtime state and coordination layer.
5. Small extracted services for context, artifacts, generation, evaluation, feedback, and distributed coordination.
6. An explicit provider registry instead of substring-based dispatch.

## Judge capability model summary

The target judge surface is additive: configure `judge.capabilities` explicitly, and treat built-in profiles as documentation/examples rather than a runtime schema abstraction.

| Capability | Key constraint |
| --- | --- |
| `attempted_models_overview` | Lists only types/families/features of models tried; no performance scores, rankings, diagnostics, or success/failure labels. |
| `performance_summary` | Summarizes observed outcomes without replacing the full evidence trace. |
| `best_model_code` | Surfaces the chosen code artifact only when enabled. |
| `recommendations` | Separate from mechanistic explanation; do not collapse into `mechanistic_coherence`. |
| `mechanistic_coherence` | Separate capability for explanation/coherence analysis. |
| `tools` | Enables agentic judge behavior using read-only access to DuckDB/artifacts only; no mutation, fitting, generation, or external side effects. |
| `citations` | Emits citations only from available evidence/artifacts. |
| `persona_synthesis` | Requires multiple configured personas or explicit persona profiles; invalid combinations must fail validation. |
| `coverage` | Reports coverage over the configured evidence set. |
| `random_feedback` | Explicit experimental replacement for old `noise` behavior. |

Default behavior: `judge.capabilities: []` produces no substantive feedback, but the run still writes an explicit empty `FeedbackArtifact`/trace showing that no capabilities were enabled.

Non-agentic behavior is a judge configuration constraint, not a capability: when `tools` is absent, the judge receives preassembled evidence bundles from the orchestrator/DuckDB and cannot query tools.

Future result extraction/export work is separate from runtime JSON outputs; if it exists later, it should read from DuckDB, not replace it.

## Non-goals

- Do not add hidden legacy fallbacks.
- Do not keep compatibility wrappers just to ease the transition.
- Do not introduce JSON as runtime state.
- Do not implement dashboard work.
- Do not rewrite model semantics unless a phase explicitly says so.
- Do not require `judge.lesion` or any lesion-first runtime schema in the target architecture.
- Do not require expensive fitting, HBI, or PPC in the default unit suite.

## Coding standards / implementation workflow

- Use type annotations for new Python code.
- Use Google-style docstrings for public functions/classes.
- Format with Black.
- Prefer small modules with one responsibility.
- Keep tests fast and deterministic.
- Use the `gecco_mh` conda environment for any future Python/test commands.
- Create a new git branch before starting this cleanup work.
- Use staged commits to keep changes grouped into small, reviewable units.
- Prefer committing by phase or subtrack rather than one large commit.

## Phase dependencies / parallel workstreams

| Phase | Depends on | Can run in parallel with |
| --- | --- | --- |
| 0 Freeze behavior | none | none |
| 1 Docs reset | 0 | 2 |
| 2 CLI consolidation | 0 | 1 |
| 3 Config validation | 2 | prep for 4 and 5 |
| 4 Orchestrated judge | 3 | 5 once schema/config decisions are stable |
| 5 DuckDB canonical state | 3 | 4 once schema/config decisions are stable |
| 6 Service extraction | 3 and 5 | internal parallel subtracks |
| 7 DuckDB coordination/status views | 5 and 6 | none |
| 8 Provider registry/export/perf | 7 | none |

Parallel rule of thumb:

- Start with Phase 0.
- After Phase 0, Phases 1 and 2 can proceed in parallel.
- Phase 3 depends on Phase 2.
- Phase 4 depends on Phase 3.
- Phase 5 depends on Phase 3 and can proceed alongside Phase 4 after schema/config decisions are stable.
- Phase 6 depends on Phases 3 and 5, and can split into internal parallel subtracks.
- Phase 7 depends on Phases 5 and 6.
- Phase 8 depends on Phase 7.

## TDD strategy

Use this order:

1. Characterization tests for current behavior.
2. Contract tests for the new interfaces.
3. Deletion of old code only after the replacement is protected by tests.

Important test rule:

- Characterization tests come before major refactors or deletions.
- Contract tests should prove the validated config model, CLI routing, orchestrated judge, DuckDB state, and extracted services.
- Fast/smoke tests may exercise full capability coverage with mocks/stubs when helpful.
- Expensive fitting, HBI, and PPC should stay out of the default unit suite.

## What to delete / what not to do / test categories / acceptance criteria

### What to delete

Delete by category, after replacement tests exist:

- Stale scripts and script-only entrypoints.
- Old config names and shims.
- JSON runtime artifact paths and any code that treats them as canonical.
- Legacy/manual judge implementation after historical parity is proven.
- `judge.lesion` compatibility wrappers or lesion-first runtime schema support.
- Registry JSON reliance.
- Substring-based provider dispatch.

These are categories to inspect, not a promise that every filename listed earlier is deleted.

### What not to do

- No hidden legacy fallback.
- No compatibility wrappers.
- No JSON runtime state.
- No dashboard work.
- No expensive default unit tests.
- No broad model semantics rewrite just to change architecture.
- No `judge.lesion` runtime schema or compatibility shim.

### Test categories

- Characterization tests
- CLI contract tests
- Config contract tests
- Judge contract tests
- DuckDB state contract tests
- Service extraction tests
- Concurrency/coordination tests
- Provider registry tests
- Export/report foundation tests
- Lightweight performance smoke tests

### Acceptance criteria

The cleanup is complete when:

- Docs match the active code paths.
- The CLI surface is unified.
- Config loading is validated.
- The judge is orchestrated-only and additive-capability based.
- DuckDB is canonical for runtime state and artifacts.
- No runtime JSON source-of-truth remains.
- Phase 6 services are split and tested.
- DuckDB coordination works under the chosen transaction strategy.
- Provider dispatch is explicit.
- `judge.lesion` is gone, with no compatibility wrappers.
- Dashboard remains out of scope for this cleanup.

## Phase files

- [Phase 0: Freeze current behavior](phase_0_freeze_current_behavior.md)
- [Phase 1: Docs reset](phase_1_docs_reset.md)
- [Phase 2: CLI consolidation](phase_2_cli_consolidation.md)
- [Phase 3: Config validation](phase_3_config_validation.md)
- [Phase 4: Orchestrated-only judge rationalization](phase_4_orchestrated_only_judge_rationalization.md)
- [Phase 5: DuckDB canonical state](phase_5_duckdb_canonical_state.md)
- [Phase 6: Parallel extraction subtracks](phase_6_parallel_extraction_subtracks.md)
- [Phase 7: DuckDB coordination and status views](phase_7_duckdb_coordination_and_status_views.md)
- [Phase 8: Provider registry, export/report foundation, performance hygiene](phase_8_provider_registry_export_report_foundation_performance_hygiene.md)
