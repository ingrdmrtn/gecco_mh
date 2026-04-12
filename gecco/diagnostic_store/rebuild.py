"""
Rebuild a DiagnosticStore from the canonical JSON artifacts in
``results/{task}/bics/``.

Use this when:
* The ``.duckdb`` file is missing or corrupted.
* The schema was changed and a full rebuild is needed.

Usage::

    from gecco.diagnostic_store.rebuild import rebuild_from_artifacts
    store = rebuild_from_artifacts("results/two_step_factors")
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from gecco.diagnostic_store.store import DiagnosticStore


_ITER_FILENAME_RE = re.compile(
    r"iter(?P<iteration>\d+)(?:_client(?P<client_id>.+?))?_run(?P<run_idx>\d+)"
    r"(?:_participant\w+)?\.json$"
)


def _detect_split(filename: str) -> str:
    """Detect split from filename: _val -> 'val', _test -> 'test', else 'train'."""
    if "_val" in filename:
        return "val"
    if "_test" in filename:
        return "test"
    return "train"


def rebuild_from_artifacts(
    results_dir: str | Path,
    db_path: str | None = None,
    overwrite: bool = True,
) -> DiagnosticStore:
    """Rebuild the diagnostic store from JSON artifact files.

    Parameters
    ----------
    results_dir:
        Path to the task results directory (contains ``bics/``).
    db_path:
        Where to write the rebuilt ``.duckdb`` file.  Defaults to
        ``{results_dir}/diagnostics.duckdb``.
    overwrite:
        If True, delete an existing database before rebuilding.

    Returns
    -------
    DiagnosticStore
        The newly rebuilt store (left open).
    """
    results_dir = Path(results_dir)
    bics_dir = results_dir / "bics"

    if not bics_dir.exists():
        raise FileNotFoundError(f"bics directory not found: {bics_dir}")

    if db_path is None:
        db_path = str(results_dir / "diagnostics.duckdb")

    if overwrite and Path(db_path).exists():
        Path(db_path).unlink()

    store = DiagnosticStore(db_path)

    json_files = sorted(bics_dir.glob("iter*.json"))
    if not json_files:
        print(f"[rebuild] No iter*.json files found in {bics_dir}")
        return store

    print(f"[rebuild] Processing {len(json_files)} artifact files from {bics_dir}")

    for json_file in json_files:
        m = _ITER_FILENAME_RE.search(json_file.name)
        if not m:
            print(f"[rebuild] Skipping unrecognised filename: {json_file.name}")
            continue

        iteration = int(m.group("iteration"))
        run_idx = int(m.group("run_idx"))
        client_id = m.group("client_id") or None
        tag = f"_client{client_id}" if client_id else ""

        with open(json_file) as f:
            try:
                iteration_results = json.load(f)
            except json.JSONDecodeError as e:
                print(f"[rebuild] JSON parse error in {json_file.name}: {e}")
                continue

        split = _detect_split(json_file.name)

        store.write_iteration(
            iteration=iteration,
            run_idx=run_idx,
            iteration_results=iteration_results,
            tag=tag,
            client_id=client_id,
        )
        print(
            f"[rebuild]   iter={iteration} run={run_idx} client='{client_id}' split='{split}' "
            f"({len(iteration_results)} models)"
        )

    top_models_test_file = results_dir / "bics" / "top_models_test.json"
    if top_models_test_file.exists():
        print(f"[rebuild] Processing top_models_test.json")
        with open(top_models_test_file) as f:
            try:
                top_models = json.load(f)
            except json.JSONDecodeError as e:
                print(f"[rebuild] JSON parse error in top_models_test.json: {e}")
            else:
                for entry in top_models:
                    store.write_top_model_test(entry)
                print(f"[rebuild]   Wrote {len(top_models)} test entries")

    print("[rebuild] Done.")
    return store
