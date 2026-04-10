"""
Diagnostic store for GeCCo — a DuckDB-backed, queryable record of every
model evaluated across iterations.  The store is a *derived* artifact;
the canonical source of truth remains the per-iteration JSON files in
``results/{task}/bics/``.  If the store is lost or the schema changes it
can be rebuilt via :func:`gecco.diagnostic_store.rebuild.rebuild_from_artifacts`.
"""

from gecco.diagnostic_store.store import DiagnosticStore

__all__ = ["DiagnosticStore"]
