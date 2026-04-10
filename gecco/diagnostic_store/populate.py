"""
Convenience wrapper so callers don't need to instantiate DiagnosticStore
directly when they only want to write one iteration.
"""

from __future__ import annotations

from gecco.diagnostic_store.store import DiagnosticStore


def write_iteration_result(
    store: DiagnosticStore,
    iteration: int,
    run_idx: int,
    iteration_results: list[dict],
    ppc_results: dict | None = None,
    tag: str = "",
    client_id: str | None = None,
) -> None:
    """Write one iteration's results to *store*.

    This is a thin pass-through to :meth:`DiagnosticStore.write_iteration`
    and exists so import paths are symmetric with :mod:`rebuild`.
    """
    store.write_iteration(
        iteration=iteration,
        run_idx=run_idx,
        iteration_results=iteration_results,
        ppc_results=ppc_results,
        tag=tag,
        client_id=client_id,
    )
