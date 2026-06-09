# Phase 6: Parallel extraction subtracks

## Purpose

Split `GeCCoModelSearch` into smaller services behind tests.

## Depends on

- Phases 3 and 5.

## Parallel workstreams

This phase is intentionally parallel once Phases 3 and 5 are in place.

### A. `RunContext` and validated config/path ownership

Depends on:

- Phase 3 config validation.
- Phase 5 DuckDB decision.

Likely modules to create/change:

- `gecco/run_context.py` or equivalent small context module.
- `gecco/coordination.py` if path ownership stays there.

Tests to write first:

- Unit tests for path ownership.
- Tempdir lifecycle tests.
- Invalid config rejection tests.

Implementation tasks:

- Move path resolution and ownership into a single context object.
- Make tempdir and output-path handling explicit.

Deletion tasks:

- Remove path ownership from the monolith once the context owns it.

Parallel note:

- Can start in parallel with B after Phases 3 and 5.

### B. `ArtifactStore` / DuckDB writer-reader contracts

Depends on:

- Phase 5 table/schema decisions.

Likely modules to create/change:

- `gecco/artifacts.py` or a similar focused store module.
- `gecco/diagnostic_store/store.py`

Tests to write first:

- Contract tests for write/read round trips.
- Transaction boundary tests.

Implementation tasks:

- Encapsulate artifact persistence behind a small store interface.
- Ensure DuckDB writes and reads are the canonical path.

Deletion tasks:

- Remove direct monolith-owned write/read code once the store exists.

Parallel note:

- Can start in parallel with A once the table schema and file layout are agreed.

### C. `CandidateGenerator` extraction

Depends on:

- A.

Likely modules to create/change:

- A small generator service module under `gecco/`.

Tests to write first:

- Generator contract tests.
- Tiny fixture tests with stubbed LLM backends.

Implementation tasks:

- Move candidate generation out of `GeCCoModelSearch`.
- Keep the generator pure with explicit inputs and outputs.

Deletion tasks:

- Remove generator logic from the monolith after the service is covered.

Parallel note:

- Can run after A, while D and F are being prepared.

### D. `CandidateEvaluator` extraction

Depends on:

- A, B, and C.

Likely modules to create/change:

- A small evaluator service module under `gecco/`.

Tests to write first:

- Evaluator contract tests.
- Stubbed fitting/diagnostic tests.

Implementation tasks:

- Move candidate scoring/evaluation behind a dedicated interface.
- Keep fitting calls stub-friendly in unit tests.

Deletion tasks:

- Remove evaluator logic from the monolith after parity is covered.

Parallel note:

- Can proceed after A/B/C are in place.

### E. `FeedbackCoordinator` / orchestrated judge adapter

Depends on:

- A, B, and D.

Likely modules to create/change:

- `gecco/construct_feedback/feedback.py`
- `gecco/construct_feedback/tool_judge.py`
- `gecco/construct_feedback/judge_lesion.py`

Tests to write first:

- Parity tests against required historical lesion/manual behavior.
- Tests for explicit capability, add-in, and post-processor handling.

Implementation tasks:

- Encapsulate judge orchestration behind one coordinator.
- Make all experiment behavior explicit via `judge.capabilities`.

Deletion tasks:

- Delete the legacy/manual judge path only after parity is proven.

Parallel note:

- Can run after A/B/D.

### F. `DistributedCoordinator` / DuckDB coordination

Depends on:

- A and B.

Likely modules to create/change:

- `gecco/coordination.py`
- `gecco/diagnostic_store/store.py`

Tests to write first:

- Concurrent client tests.
- Status-view consistency tests.
- Explicit transaction strategy tests.

Implementation tasks:

- Move distributed coordination into a small dedicated service.
- Make status reads come from DuckDB views.

Deletion tasks:

- Remove distributed coordination code from the monolith after the new service is covered.

Parallel note:

- Can proceed in parallel with C/D/E once A/B are stable.

## Exit criteria

- `GeCCoModelSearch` no longer owns these responsibilities directly.
- Each extracted service has its own contract tests.
