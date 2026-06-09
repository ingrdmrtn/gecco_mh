# Phase 4: Orchestrated-only judge rationalization

## Purpose

Use one judge pipeline everywhere.

## Target pipeline

- `JudgeCapabilities -> EvidenceBundle -> Analysis/Synthesis -> PostProcessors -> FeedbackArtifact`

## Depends on

- Phase 3.

## Can run in parallel with

- Phase 5 after the config contract exists.

## Likely files/modules to inspect

- `gecco/construct_feedback/feedback.py`
- `gecco/construct_feedback/tool_judge.py`
- `gecco/construct_feedback/judge_lesion.py`
- `tests/test_cmg_judge.py`
- `tests/test_judge_orchestration.py`
- `tests/test_judge_enhancements.py`

## Tests to write first

- Judge contract tests for evidence completeness and trace shape.
- Parity tests that capture the historical lesion/ablation behavior needed for migration only.

## Implementation tasks

- Route all judge behavior through the orchestrated path.
- Make the default judge produce no substantive feedback unless explicit capabilities are enabled.
- Non-agentic judge configurations receive preassembled evidence bundles from the orchestrator/DuckDB and cannot query tools.
- Keep experimental behavior in explicit capabilities, add-ins, or post-processors.
- Treat a full judge as a composition of capabilities, not a special default mode.
- If parity fails, add an explicit capability composition that reproduces the needed historical behavior.

## Deletion tasks

- Remove legacy/manual feedback code only after parity tests prove the orchestrated judge can reproduce the required historical behavior.
- Delete the old implementation after parity passes.

## Nuanced legacy-judge policy

- First write parity/characterization tests for the needed historical lesion/manual behavior.
- Then implement the behavior as explicit capabilities/add-ins in the orchestrated system if parity is missing.
- Only then delete the old path.
- No hidden fallback path.
- No `judge.lesion` runtime compatibility wrapper.

## Acceptance criteria

- A single orchestrated pipeline handles both production and single-worker cases.
- Empty `judge.capabilities: []` produces an explicit no-feedback trace.
- The full judge is expressible as a composed capability set.
