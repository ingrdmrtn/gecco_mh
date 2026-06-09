# Implementation Guardrails for Cleanup Plans

Use this guide alongside any phase or fix plan in this cleanup. Its purpose is to prevent plans from being implemented as thin wrappers, compatibility shims, or partial refactors that satisfy surface wording while leaving the old ownership model intact.

This guide applies especially when a plan says to extract services, remove monolith ownership, make DuckDB canonical, delete legacy behavior, or simplify runtime architecture.

## Core Rule

Passing tests is not enough. The implementation must also satisfy the ownership and deletion intent of the plan.

A cleanup task is not complete when a new module exists. It is complete when the old responsibility has moved, the old duplicate path is deleted, and tests prove the replacement path directly.

## Read Before Coding

Before changing code, read:

- The whole-plan overview for global constraints and non-goals.
- The assigned phase plan.
- Any follow-up fix plan for that phase.
- The current implementation around the affected files.

Then write down the intended ownership change in one sentence before implementing. For example:

```text
After this change, GeCCoModelSearch orchestrates candidate evaluation but CandidateEvaluator owns fitting, repair-loop boundaries, and finalization.
```

If that sentence cannot be made precise, the plan needs clarification before implementation.

## Avoid Wrapper Extraction

Wrapper extraction is the most common failure mode in cleanup work.

Wrapper extraction means creating a new class or module that only delegates back to the old owner. This makes the code look refactored while leaving the actual behavior in the monolith.

### Bad Pattern

```python
service.run(
    generate=self.generate_models,
    evaluate=self._fit_candidate_model,
    finalize=self._finalize_iteration_results,
)
```

This is usually not a real extraction. The old object still owns the important behavior.

### Better Pattern

```python
service = CandidateEvaluationService(
    fitting_backend=fitting_backend,
    artifact_store=artifact_store,
    registry=registry,
)
service.evaluate_iteration(request)
```

External dependencies may be injected. Monolith private methods should not be injected as service collaborators unless the plan explicitly says this is a temporary intermediate step and also defines when it must be removed.

## Define Ownership Explicitly

For every extracted service, the implementation should make these points clear:

- What the service owns now.
- What the previous owner no longer owns.
- What inputs the service requires.
- What outputs the service returns.
- What persistence or state mutation the service is allowed to perform.
- What dependencies are external backends versus old monolith behavior.

If the old monolith still contains the same business logic after the service is introduced, the extraction is incomplete.

## Turn Deletion Into a Gate

Do not treat deletion tasks as optional cleanup after the real work. In this cleanup, deletion is part of the real work.

For every replacement path:

1. Write characterization tests for current behavior if behavior must be preserved.
2. Write contract tests for the new owner.
3. Route production code through the new owner.
4. Delete the old duplicate path.
5. Run tests that prove the new path is active.

The task is not done at step 3.

## Make Forbidden Patterns Testable

Plans should not only say what to build. They should also say what must no longer exist.

Examples of forbidden patterns that can be checked:

- An extracted service imports the monolith module.
- A coordinator has a fallback import from an old runtime path.
- A runtime path writes JSON before writing DuckDB state.
- A monolith method still performs direct artifact path construction after `RunContext` or `ArtifactStore` exists.
- A service test constructs the old monolith to exercise the new service.
- A private monolith method is passed into an extracted service as its main behavior.

Where practical, add tests or simple source checks that fail when these patterns return.

## Prefer Real Service Tests Over Mock-Only Tests

Mocks are useful for external systems. They are not enough to prove a service boundary.

Use real instances for the service under test and for small local collaborators where possible:

- Real `RunContext` with a temporary project root.
- Real `ArtifactStore` with a temporary results directory.
- Real DuckDB-backed store or registry when the contract is about runtime state.
- Fake LLM/fitting backends that return deterministic data.

Avoid tests that only prove a mocked collaborator was called. Those tests often pass even when the old architecture is still intact.

## Distinguish External Dependencies From Old Owners

It is acceptable to stub expensive or external dependencies:

- LLM calls.
- Model fitting.
- Network APIs.
- Long-running diagnostics.

It is not acceptable to treat the old monolith as the stub for the extracted service.

For example, a fake fitting backend is fine. Passing `GeCCoModelSearch._fit_candidate_model` into an evaluator service means the evaluator does not own evaluation yet.

## Keep Runtime State Canonical

When a plan says DuckDB is canonical, implementation should reflect that in code and tests.

Do:

- Write runtime state through DuckDB-backed stores first.
- Read coordination/status/runtime state from DuckDB-backed stores.
- Treat any remaining JSON as non-canonical inspection output only when the plan allows it.

Do not:

- Preserve JSON as a runtime source of truth.
- Add JSON fallbacks for compatibility.
- Let tests assert runtime correctness by reading JSON when DuckDB is meant to be canonical.

## Do Not Add Hidden Compatibility Paths

Avoid hidden fallbacks such as:

- `try old_path except use_new_path`
- dynamically importing old modules if new dependencies are missing
- preserving old config names without explicit migration scope
- keeping old CLI/script paths as wrappers unless the phase explicitly allows a short-lived transition

If a fallback is needed for persisted data, shipped behavior, or external consumers, document that explicitly in the plan and add a deletion point.

## Keep Scope Narrow

When implementing a phase, do only that phase.

Do not use a service extraction task to also implement:

- dashboard changes
- export/report features
- provider registry work
- broad model semantics changes
- new expensive default tests
- unrelated CLI cleanup

If adjacent work is discovered, write it down as a follow-up instead of folding it into the current phase.

## Review Checklist Before Calling Work Complete

Before marking a plan complete, answer these questions:

- Does the new service own real behavior, or does it mostly call back into the old owner?
- Can the service be tested without constructing the monolith?
- Are private monolith methods still injected into the service as core behavior?
- Has the old duplicate logic been deleted?
- Do tests prove the production route uses the new owner?
- Are tests too mock-heavy to catch ownership mistakes?
- Does any hidden fallback keep old behavior alive?
- Does runtime state follow the canonical store required by the plan?
- Did the implementation stay inside the assigned phase scope?

If any answer is unclear, do not treat the phase as complete.

## What Good Completion Looks Like

A robust cleanup implementation usually has these properties:

- The old monolith is smaller in the areas assigned by the plan.
- New services have explicit request/response shapes or clear method signatures.
- Tests exercise services directly with realistic local collaborators.
- Production orchestration delegates to the new services.
- Old direct ownership code is deleted, not merely bypassed.
- There are no hidden legacy fallbacks.
- The implementation can be explained in terms of ownership, not just files changed.

## Suggested Handoff Format

When handing a plan to a developer, include this short instruction:

```text
Follow `docs/codebase_cleanup_plan/implementation_guardrails.md` while implementing this plan. In particular, avoid wrapper extraction: the new service must own the behavior, tests must prove it without constructing the old monolith, and the old duplicate path must be deleted once covered.
```
