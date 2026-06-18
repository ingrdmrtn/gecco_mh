# Dashboard remediation plan

Source of truth for the review findings in `dashboard-rework-review.md`.

## Summary

Fix three blockers:
1. Preserve `source_db + model_id + split` identity for `individual_differences` loading/joining.
2. Prove `load_registry_snapshot()` uses `SharedRegistry.open_existing(results_dir / "shared_registry.duckdb").read()` with no JSON fallback replacing canonical registry loading.
3. Record a complete implementation receipt covering phase approvals, validation evidence, dirty-file/scope classification, and the browser-launch gap.

## Pipeline map

- Review findings -> remediation plan -> targeted code/test changes -> implementation receipt -> final verification.
- Keep each phase independently checkable; do not advance if the phase gate is not satisfied.

## Contract matrix

| Row | Contract | Evidence |
| --- | --- | --- |
| C1 | `individual_differences` retains `source_db + model_id + split` identity through load/join paths | Test coverage for identity-preserving load/join behavior |
| C2 | `load_registry_snapshot()` calls `SharedRegistry.open_existing(results_dir / "shared_registry.duckdb").read()` | Direct proof in tests; no JSON fallback replacing canonical load path |
| C3 | Implementation receipt exists and captures approvals, validation, dirty-file/scope classification, and browser-launch gap | Updated receipt artifact with phase-by-phase evidence |

## Scope firewall

Allowed implementation files for the remediation work:
- `gecco-mh-dashboard/dashboard/data_adapter.py`
- `gecco-mh-dashboard/dashboard/history_store.py`
- `tests/test_dashboard_compatibility.py`
- `tests/test_dashboard_artifacts.py`
- `tests/test_dashboard_foundation.py`
- `tests/test_dashboard_models.py`
- `tests/test_dashboard_clients.py`
- `tests/test_dashboard_overview.py`
- `tests/test_dashboard_judge.py`
- `tests/test_dashboard_integration.py` (if present)
- `dashboard-implementation-receipt.md`

Forbidden scope:
- Any other dashboard code, unrelated tests, docs, or generated artifacts.
- Any browser automation or UI launch changes; the browser-launch gap is to be documented, not fixed in this plan.

## Test and verification inventory

Validation commands:
- `python -m pytest tests/test_dashboard_compatibility.py`
- `python -m pytest tests/test_dashboard_artifacts.py`
- `python -m pytest tests/test_dashboard_models.py`
- `python -m pytest tests/test_dashboard_foundation.py`
- `python -m pytest tests/test_dashboard_clients.py`
- `python -m pytest tests/test_dashboard_overview.py`
- `python -m pytest tests/test_dashboard_judge.py`
- `python -m pytest`

Use the smallest relevant subset first; run the full suite only for final verification.

## Implementation phases

### Phase 1: Identity preservation

- Update the `individual_differences` load/join path so `source_db`, `model_id`, and `split` remain the stable identity.
- Add or adjust tests that fail if rows are merged or re-keyed incorrectly.
- Gate: do not proceed until identity is proven end-to-end.

### Phase 2: Canonical registry loading

- Add proof that `load_registry_snapshot()` opens `results_dir / "shared_registry.duckdb"` through `SharedRegistry.open_existing(...).read()`.
- Remove/avoid any JSON fallback that would replace canonical registry loading.
- Gate: do not proceed until the test explicitly captures the canonical path.

### Phase 3: Implementation receipt

- Update `dashboard-implementation-receipt.md` with:
  - phase approvals,
  - validation evidence,
  - dirty-file/scope classification,
  - browser-launch gap note.
- Gate: receipt must match the actual verification run.

## Forbidden patterns

- Do not broaden scope to unrelated dashboard cleanup.
- Do not add alternate registry loaders or silent fallback behavior.
- Do not claim browser-launch verification if no browser launch was performed.
- Do not edit files outside the allowed scope firewall.

## Final verification

- Re-run the relevant tests above.
- Confirm the receipt reflects the executed commands and results.
- Confirm every touched file is inside the allowed scope firewall.
- Confirm the plan’s findings are closed without introducing new fallback behavior.
