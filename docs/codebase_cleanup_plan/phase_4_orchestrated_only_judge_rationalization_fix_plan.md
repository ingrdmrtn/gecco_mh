# Phase 4 Remediation Plan: Orchestrated-only judge rationalization (review fixes)

## 1) Title and context

This is a remedial plan for Phase 4 after review.

Goal: keep the single orchestrated judge pipeline, fix the review findings, and preserve the Phase 4 acceptance criteria from the original plan.

## 2) Relationship to the original Phase 4 plan

| Original plan item | Original intent | What this fix plan does |
|---|---|---|
| Purpose | Use one judge pipeline everywhere | Keeps the single orchestrated path; removes shortcut/schema inconsistencies |
| Target pipeline | `JudgeCapabilities -> EvidenceBundle -> Analysis/Synthesis -> PostProcessors -> FeedbackArtifact` | Preserves the same pipeline shape and ensures recovery/short-worker paths still end as a canonical `FeedbackArtifact` |
| Implementation task: route all judge behavior through the orchestrated path | No legacy/manual judge path | Keeps the orchestrated path; fixes the remaining bypasses and fan-out issues |
| Implementation task: default judge produces no substantive feedback unless explicit capabilities are enabled | Explicit no-feedback behavior | Preserves explicit empty/no-feedback traces and prevents accidental empty persona output |
| Implementation task: non-agentic judge configs receive preassembled evidence and cannot query tools | Orchestrator-owned evidence | Keeps this scope unchanged; no broader refactor |
| Deletion task: remove legacy/manual feedback code only after parity | No hidden fallback path | Avoids reintroducing legacy wrappers or alternate judge APIs |
| Acceptance: single orchestrated pipeline handles production and single-worker cases | One path for all runs | Adds the missing metadata, persona fallback, and capability-gated routing so both cases behave consistently |
| Acceptance: empty `judge.capabilities: []` produces an explicit no-feedback trace | Clear empty-result behavior | Verifies the trace remains explicit and does not get synthesized accidentally |
| Acceptance: full judge is expressible as composed capability set | Capability composition works | Preserves CMG/generator compatibility while gating fan-out with `persona_synthesis` |

## 3) Current findings to fix

1. Short-circuit recovery currently loses canonical metadata/provenance, so the artifact can no longer explain why it short-circuited.
2. Some recovery feedback is malformed because dict-like data is being stringified into user feedback instead of being kept as structured fields.
3. Local/single-worker runs can end up with empty persona feedback when the requested persona is missing; this must fall back to a default, never silently empty.
4. Persona fan-out is not properly gated by `persona_synthesis`, so synthesis can happen when it should not.
5. Schema asymmetry exists because `_save_trace` writes a competing shape instead of the canonical `FeedbackArtifact` shape in the shortcut path.
6. Test coverage is missing for the exact recovery, fallback, persona-routing, and trace-shape cases above.

## 4) Guiding constraints

- Keep the single orchestrated pipeline.
- No legacy fallback path.
- No `judge.lesion` wrapper.
- Do not introduce a broad `EvidenceBundle` refactor.
- Use type annotations and docstrings for any Python additions.
- Format Python changes with Black.
- Keep the scope targeted to Phase 4 fixes only.

## 5) TDD plan

Write the failing tests first, then implement the code.

### Test file 1: `tests/test_phase4_orchestrated_judge.py`

#### `test_recovery_failure_preserves_metadata_per_angle_and_recommendations`
- Arrange: simulate a recovery-failure path with prior verdict data that includes metadata, per-angle analysis, and recommendations.
- Act: run the orchestrated judge pipeline.
- Assert: the returned `FeedbackArtifact` keeps structured `metadata`, and the shortcut `analysis_data` carries either `metadata` plus a structured `verdict_payloads`/`shortcut_verdict_payload` with `per_angle` and `key_recommendations`, so `build_feedback_artifact()` can populate canonical fields without synthesis; feedback text must not contain `{'default':` or `"default":`.

#### `test_feedback_artifact_serialization_preserves_metadata`
- Arrange: build a `FeedbackArtifact` with metadata present.
- Act: serialize it through the same helper used by the orchestrated flow.
- Assert: metadata survives round-trip serialization exactly as structured data.

#### `test_local_single_worker_without_client_id_uses_default_feedback`
- Arrange: use a config with `cfg.clients` present but no `client_id` for the current worker.
- Act: run the local/single-worker path.
- Assert: the run uses default feedback, not empty feedback.

#### `test_persona_synthesis_fanout_only_runs_when_enabled`
- Arrange: configure multiple personas with `persona_synthesis` disabled.
- Act: run the orchestrated judge pipeline.
- Assert: no persona fan-out occurs.

#### `test_persona_synthesis_fanout_runs_for_cmg_generator_compatibility`
- Arrange: enable CMG/generator compatibility and keep `persona_synthesis` enabled.
- Act: run the orchestrated judge pipeline.
- Assert: generator persona synthesis still works.

#### `test_empty_capabilities_produces_explicit_no_feedback_trace`
- Arrange: set `judge.capabilities: []`.
- Act: run the judge.
- Assert: the trace is explicit no-feedback, and synthesis is skipped.

#### `test_shortcut_path_does_not_write_competing_trace_schema`
- Arrange: trigger the recovery short-circuit path.
- Act: inspect the persisted trace.
- Assert: `_save_trace` is not used for the shortcut, or the shortcut trace matches the canonical `FeedbackArtifact` schema exactly.

### Test file 2: `tests/test_judge_orchestration.py`

#### `test_orchestrator_uses_shared_orchestrated_runner`
- Arrange: create a normal orchestrated config.
- Act: run the orchestrator.
- Assert: it uses the shared orchestrated runner and not an alternate judge path.

#### `test_build_feedback_artifact_returns_canonical_json_shape`
- Arrange: provide representative analysis data.
- Act: call `build_feedback_artifact`.
- Assert: the result is canonical JSON with structured fields, including metadata.

### Test file 3: `tests/test_cmg_judge.py`

#### `test_cmg_short_circuit_feedback_keyed_by_generator`
- Arrange: use CMG settings.
- Act: run the short-circuit path.
- Assert: feedback is keyed by the generator persona and remains structured.

#### `test_non_cmg_short_circuit_feedback_keyed_by_default`
- Arrange: use non-CMG settings.
- Act: run the short-circuit path.
- Assert: feedback is keyed by default.

> Note: keep the existing test names where they already exist; update them in place rather than duplicating names across files.

### Test file 4: `tests/test_judge_enhancements.py`

#### `test_store_tools_expose_block_residuals_and_participant_best_models`
- Keep this as-is unless the Phase 4 cleanup accidentally regresses neighboring judge behavior.

### Run order for TDD

1. Run the new/targeted tests first and confirm they fail:
   - `conda run -n gecco_mh pytest tests/test_phase4_orchestrated_judge.py -q`
   - `conda run -n gecco_mh pytest tests/test_judge_orchestration.py -q`
   - `conda run -n gecco_mh pytest tests/test_cmg_judge.py -q`
2. Implement one chunk at a time.
3. Re-run the same tests after each chunk.

## 6) Implementation chunks, vertical and actionable

### Chunk A: Canonical short-circuit artifact metadata

**Goal:** preserve provenance and structured recovery output in the shortcut path.

**Files and symbols:**
- `gecco/construct_feedback/orchestrated.py`
  - `FeedbackArtifact`
  - `build_feedback_artifact`
  - `run_orchestrated_judge_pipeline`
- `gecco/construct_feedback/tool_judge.py`
  - `get_feedback_analysis`
  - `_try_shortcut_from_recovery_failure`
  - `_load_previous_verdict`

**Work:**
- Add `metadata: dict[str, Any]` to `FeedbackArtifact`.
- Make `build_feedback_artifact` copy `analysis_data["metadata"]` into the artifact.
- In the recovery shortcut, `analysis_data` should carry `metadata` plus structured `verdict_payloads`/`shortcut_verdict_payload` with `per_angle` and `key_recommendations`, so `build_feedback_artifact()` can fill canonical fields without synthesis.
- `run_orchestrated_judge_pipeline()` should use persona-keyed `analysis_data["synthesized_feedback"]` when present, or wrap plain text as `{"default": text}`; it must never stringify a synthesized_feedback dict.
- Remove or avoid `_save_trace` in the shortcut path.

### Chunk B: Persona routing and capability gate

**Goal:** gate synthesis by capability while keeping CMG/generator compatibility.

**Files and symbols:**
- `gecco/construct_feedback/orchestrated.py`
  - `run_orchestrated_judge_pipeline`
  - `FeedbackArtifact.feedback_for_persona`
- `gecco/run_gecco.py`

**Work:**
- Add a small helper such as `_resolve_synthesis_personas(cfg) -> dict[str, Any | None]`.
- Behavior: CMG enabled => generator persona; elif `persona_synthesis` capability enabled and clients exist => all clients; else => default only.
- This means local/single-worker runs with `cfg.clients` but no `persona_synthesis` still produce a default artifact.
- For `feedback_for_persona`, prefer the requested persona, then `default`; if neither exists and the feedback dict is non-empty, raise a clear `ValueError` rather than silently returning empty.

### Chunk C: Registry/orchestrator consistency

**Goal:** keep the orchestrator path canonical even when no runnable model is present.

**Work:**
- Keep any no-runnable-model behavior explicit and canonical.
- If the no-runnable-model fallback remains direct-to-registry temporarily, it must still create/persist a canonical `FeedbackArtifact` or include a test documenting why it is not judge feedback; recommended option: canonical static artifact.
- Do not add a hidden fallback or a separate judge module.

### Chunk D: Clean up trace/schema asymmetry

**Goal:** stop the shortcut path from competing with the canonical schema.

**Work:**
- Treat `_save_trace` as non-shortcut-only legacy plumbing.
- Do not use `_save_trace` for recovery shortcuts.
- If `_save_trace` remains, keep it aligned with the canonical artifact shape.

### Chunk E: Docs update

**Goal:** record the review-driven fix.

**Work:**
- Update the Phase 4 implementation record or nearby cleanup docs with the final behavior changes.

## 7) Step-by-step instructions for a junior developer

1. Read the original Phase 4 plan and this remediation plan together.
2. Add the tests first.
3. Run:
   - `conda run -n gecco_mh pytest tests/test_phase4_orchestrated_judge.py -q`
4. Fix Chunk A until the recovery tests pass.
5. Run:
   - `conda run -n gecco_mh pytest tests/test_cmg_judge.py -q`
6. Fix Chunk B until persona routing tests pass.
7. Run:
   - `conda run -n gecco_mh pytest tests/test_judge_orchestration.py -q`
8. Fix any remaining trace-shape or canonical-path issues.
9. Run the full targeted set:
   - `conda run -n gecco_mh pytest tests/test_phase4_orchestrated_judge.py tests/test_judge_orchestration.py tests/test_cmg_judge.py -q`
10. Format Python files with Black before finishing.

## 8) What not to do

- Do not reintroduce `gecco.construct_feedback.feedback.py`.
- Do not add a `judge.lesion` compatibility wrapper.
- Do not query tools when tools capability is absent.
- Do not stringify synthesized feedback dicts into user feedback.
- Do not silently return empty feedback for missing personas.
- Do not create a broad `EvidenceBundle` refactor.

## 9) Acceptance checklist

- [ ] Recovery short-circuit preserves metadata/provenance.
- [ ] Recovery output stays structured and is not dict-stringified.
- [ ] Local/single-worker missing-client cases use default feedback.
- [ ] Persona fan-out is gated by `persona_synthesis`.
- [ ] CMG/generator compatibility still works.
- [ ] Empty capabilities produce explicit no-feedback traces.
- [ ] Shortcut traces use one canonical schema.
- [ ] Phase 4 scope stays narrow.
