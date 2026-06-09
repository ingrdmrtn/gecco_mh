# Phase 4 Remediation: hard orchestrated-only judge pipeline

## Purpose

Remediate Phase 4 so the judge path is hard orchestrated-only everywhere, including single-worker runs.

This file is a new remediation plan related to the original Phase 4 plan:
`docs/codebase_cleanup_plan/phase_4_orchestrated_only_judge_rationalization.md`.

## Relationship to the original Phase 4 plan

The original Phase 4 target pipeline is still the intended shape:

`JudgeCapabilities -> EvidenceBundle -> Analysis/Synthesis -> PostProcessors -> FeedbackArtifact`

This remediation plan keeps that target, but changes the migration policy:

- Phase 4 is now **hard orchestrated-only**.
- Single-worker runs must use the same orchestrated judge path as distributed runs.
- The old “keep parity before deletion” wording is superseded by an explicit product decision: legacy manual/feedback behavior is retired intentionally and does not need to be resurrected.

## Scope decisions

- Do **not** modify the original plan file; create only this new remediation plan.
- Do **not** add a new service, daemon, or complex architecture.
- Prefer small extracted helpers/functions/classes only where needed.
- Do **not** touch unrelated CLI migration work or other phase files.
- Keep the judge pipeline shape centered on the orchestrated runner, registry, evidence bundle, synthesis, post-processors, and feedback artifact.

## Current review findings to fix

1. Single-worker execution still bypasses orchestrator/registry and calls `ToolUsingJudge.get_feedback()` locally when no shared registry exists.
2. `best_model_code` is appended in `gecco/run_gecco.py` after the judge verdict/artifact, so the judge JSON artifact does not match the final consumed feedback.
3. `config/schema.py` still exposes `mode: Literal["manual", "tool_using"] = "manual"` even though runtime no longer supports manual mode.
4. Existing tests still validate transitional behavior such as `orchestrated=False` instead of the hard orchestrated-only contract.

## Final behaviour after the review fixes

- Recovery shortcuts now persist one canonical `FeedbackArtifact` shape with structured `metadata`, `per_angle`, and `key_recommendations`.
- Shortcut feedback remains persona-keyed structured data and is never created by stringifying a feedback dict.
- Persona fan-out now runs only when `persona_synthesis` is enabled, except for the CMG generator compatibility path.
- Local or single-worker runs without a matching persona now fall back to default feedback rather than silently returning an empty string.
- No-runnable-model orchestrator fallbacks persist a canonical static artifact instead of writing registry-only judge output.

## Files/modules likely involved

- `gecco/run_gecco.py`
- `gecco/cli/run_judge_orchestrator.py`
- `config/schema.py`
- `gecco/construct_feedback/*`
- `tests/test_cmg_judge.py`
- `tests/test_judge_orchestration.py`
- `tests/test_judge_enhancements.py`
- `tests/test_phase4_orchestrated_judge.py`

## TDD order

Write or update the tests below before changing implementation.

### Test set 1: config contract cleanup

- Add tests that reject `mode: manual`.
- Add tests that reject `mode: tool_using` and any other unknown judge modes, unless the implementer deliberately uses a deprecated-field validator with a clearer rejection message.
- Add tests that confirm the remaining config contract no longer advertises legacy manual mode.
- Add tests that update/remove existing YAML config keys currently set to `mode: "tool_using"` if schema cleanup rejects the field.

### Test set 2: unified feedback artifact contract

- Add tests that the helper builds a complete `FeedbackArtifact` JSON shape.
- Add tests that both orchestrated and local/single-worker execution use the same artifact builder/persistence path.

### Test set 3: best_model_code placement

- Add tests that `best_model_code` is included inside the judge pipeline/artifact output.
- Add tests that `run_gecco.py` does not mutate the feedback artifact by string concatenation after the judge returns.

### Test set 4: single-worker orchestration parity

- Add tests that single-worker runs still go through the orchestrated analysis/synthesis/post-processing path.
- Add tests that absence of a shared registry falls back to an in-process/local orchestrated runner, not a direct `get_feedback()` bypass.

### Test set 5: phase 4 hard contract

- Update Phase 4 tests so they assert `orchestrated=True` as the target behavior.
- Remove transitional assertions that expect the old non-orchestrated path.

## Implementation chunks

### A. Config contract cleanup

**Tests to write first**

- Add config validation tests for rejecting `manual`.
- Add config validation tests for rejecting unknown modes.
- Add a regression test ensuring stale docs/comments do not describe `manual` as supported runtime behavior.

**Implementation notes**

- Retire `judge.mode` from the public contract entirely; the final state should reject any `judge.mode` key, including current `mode: "tool_using"` YAML keys, because the field is retired rather than narrowed.
- Make the schema fail fast for `mode: manual`, `mode: "tool_using"`, and any other unknown values.
- Keep the error message clear and user-facing.
- Update directly stale comments/docstrings only where they would mislead readers.

**Affected files**

- `config/schema.py`
- Any directly stale docstrings/comments referenced by the config contract
- Config-focused tests under `tests/`
- Existing YAML configs that still set `mode: "tool_using"`

**Expected result**

- The runtime no longer accepts `judge.mode` as a runtime configuration field.
- The schema reflects the actual orchestrated-only runtime.

**What not to do**

- Do not introduce a compatibility shim for manual mode.
- Do not keep dead enum/union values around “just in case.”

### B. Feedback artifact contract

**Tests to write first**

- Add tests for a single helper that constructs the canonical `FeedbackArtifact` JSON shape.
- Add tests that the same helper is used by both orchestrated and single-worker/local flows.
- Add tests for artifact persistence/serialization shape, including required fields.

**Implementation notes**

- Introduce or standardize one helper responsible for building and persisting the `FeedbackArtifact`.
- Make that helper the only place that defines the canonical JSON shape.
- Keep the helper small and explicit so both orchestration paths can reuse it.

**Affected files**

- `gecco/construct_feedback/*` or a small adjacent helper module
- `gecco/run_gecco.py` only for calling the helper, not defining the artifact shape
- `gecco/cli/run_judge_orchestrator.py`
- Artifact/serialization tests under `tests/`

**Expected result**

- Orchestrated and local runs produce the same feedback artifact shape.

**What not to do**

- Do not duplicate artifact-building logic in `run_gecco.py`.
- Do not create a new persistence subsystem.

### C. Move `best_model_code` into the judge pipeline/artifact

**Tests to write first**

- Add tests that `best_model_code` is present in the judge-produced artifact or post-processor output.
- Add tests that the final feedback seen by downstream code already contains `best_model_code` before `run_gecco.py` returns.

**Implementation notes**

- Move `best_model_code` handling into the judge pipeline where the verdict/artifact is assembled.
- Remove post-return string concatenation in `run_gecco.py`.
- Keep the artifact as the source of truth for the final judge output.

**Affected files**

- `gecco/run_gecco.py`
- `gecco/cli/run_judge_orchestrator.py`
- Judge/post-processor code under `gecco/construct_feedback/*`
- Output-shape tests under `tests/`

**Expected result**

- The JSON artifact matches the actual feedback consumed by the rest of the system.

**What not to do**

- Do not patch the final string in the runner after the judge has finished.
- Do not add duplicate `best_model_code` copies in multiple layers.

### D. Make single-worker runs use the orchestrated path

**Tests to write first**

- Add tests that single-worker execution still creates/uses the orchestrated analysis/synthesis/post-processing path.
- Add tests that a missing shared registry triggers an in-process/local orchestrated runner/helper.
- Add regression tests covering the current bypass in `gecco/run_gecco.py`.

**Implementation notes**

- Replace the direct `ToolUsingJudge.get_feedback()` local bypass.
- Use the same orchestrated helper stack for both distributed and single-worker execution.
- If the shared registry is unavailable, route through an in-process/local orchestrated runner/helper rather than falling back to a legacy direct call.

**Affected files**

- `gecco/run_gecco.py`
- `gecco/cli/run_judge_orchestrator.py`
- Orchestration/helper code under `gecco/construct_feedback/*`
- Single-worker and orchestration tests under `tests/`

**Expected result**

- There is one judge path for both distributed and single-worker runs.

**What not to do**

- Do not reintroduce a special single-worker judge shortcut.
- Do not add a second “local-only” feedback implementation.

### E. Retarget Phase 4 tests to the hard orchestrated-only contract

**Tests to write first**

- Update existing Phase 4 tests so the target assertion is `orchestrated=True`.
- Remove or rewrite tests that only prove transitional compatibility.
- Add assertions that the new contract is stable for both single-worker and distributed execution.

**Implementation notes**

- Keep the tests focused on the final architecture, not the migration story.
- Preserve useful coverage, but change the expected path to the hard orchestrated-only behavior.

**Affected files**

- `tests/test_cmg_judge.py`
- `tests/test_judge_orchestration.py`
- `tests/test_judge_enhancements.py`
- `tests/test_phase4_orchestrated_judge.py`

**Expected result**

- The suite verifies the target state, not transitional behavior.

**What not to do**

- Do not keep `orchestrated=False` as an expected end-state in Phase 4 tests.
- Do not preserve tests whose only purpose is legacy parity.

### F. Remove dead fallback/gating code after tests pass

**Tests to write first**

- Confirm the revised tests fail before cleanup and pass after cleanup.
- Add a final regression test if needed to ensure no fallback path remains reachable.

**Implementation notes**

- Delete dead gating/fallback branches only after the new tests are green.
- Keep the cleanup narrow and directly related to Phase 4 judge flow.

**Affected files**

- `gecco/run_gecco.py`
- `config/schema.py`
- Any now-unused judge fallback helpers or branches

**Expected result**

- The codebase contains only the hard orchestrated judge path.

**What not to do**

- Do not perform broad refactors unrelated to the judge pipeline.
- Do not delete code until the updated tests prove it is unused.

## Validation commands

Run these from the repository root:

```bash
conda run -n gecco_mh pytest tests/test_cmg_judge.py tests/test_judge_orchestration.py tests/test_judge_enhancements.py tests/test_phase4_orchestrated_judge.py -q
conda run -n gecco_mh pytest tests -q
conda run -n gecco_mh python -m compileall gecco tests
```

## Coding standards for this remediation

- Any new Python function/helper/class must use type annotations.
- Public or non-trivial helper functions should have Google-style docstrings.
- Keep comments focused on why the orchestration/feedback-artifact path exists, not obvious line-by-line comments.
- Format changed Python files with Black.
- Use the `gecco_mh` conda environment for all Python/test commands.

## Final acceptance criteria

- Single-worker and distributed runs both use the same orchestrated judge pipeline.
- `judge.mode` is no longer accepted as runtime configuration; stale manual/tool_using docs are removed or corrected.
- `FeedbackArtifact` is built and persisted through one canonical helper/path.
- `best_model_code` is part of the judge pipeline/artifact before the runner returns.
- Phase 4 tests assert the hard orchestrated-only contract.
- No legacy manual/feedback parity path remains.
