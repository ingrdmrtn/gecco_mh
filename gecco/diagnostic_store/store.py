"""
DiagnosticStore — thin wrapper around a DuckDB file that exposes
write and query helpers used by the rest of the diagnostic_store package.
"""

from __future__ import annotations

import threading

import orjson
from pathlib import Path
from typing import Any

try:
    import duckdb
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "duckdb is required for the diagnostic store. "
        "Install it with: pip install duckdb"
    ) from exc

from gecco.diagnostic_store.schema import create_schema


class DiagnosticStore:
    """Thread-safe DuckDB-backed store for GeCCo diagnostics.

    Parameters
    ----------
    db_path:
        Path to the ``.duckdb`` file.  Passing ``':memory:'`` creates an
        in-memory database (useful for testing).
    """

    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        self._lock = threading.Lock()
        self._conn = duckdb.connect(self.db_path)
        create_schema(self._conn)

    # ------------------------------------------------------------------ #
    # Low-level helpers
    # ------------------------------------------------------------------ #

    def execute(self, sql: str, params: list | None = None) -> Any:
        """Execute *sql* with an exclusive lock and return the result."""
        with self._lock:
            if params is not None:
                result = self._conn.execute(sql, params)
            else:
                result = self._conn.execute(sql)
            return result

    def fetchall(self, sql: str, params: list | None = None) -> list[dict]:
        """Execute *sql* and return rows as a list of dicts."""
        with self._lock:
            if params is not None:
                cursor = self._conn.execute(sql, params)
            else:
                cursor = self._conn.execute(sql)
            cols = [d[0] for d in cursor.description]
            return [dict(zip(cols, row)) for row in cursor.fetchall()]

    def fetchone(self, sql: str, params: list | None = None) -> dict | None:
        """Execute *sql* and return the first row as a dict, or None."""
        rows = self.fetchall(sql, params)
        return rows[0] if rows else None

    # ------------------------------------------------------------------ #
    # Upsert helpers
    # ------------------------------------------------------------------ #

    def _get_or_create_iteration(
        self,
        run_idx: int,
        iteration: int,
        tag: str,
        client_id: str | None,
        timestamp: str | None,
        n_models_proposed: int,
    ) -> int:
        """Return the iteration_id, creating the row if absent.
        
        Must be called within a transaction and with the lock held.
        """
        tag = tag or ""
        row = self._conn.execute(
            "SELECT iteration_id FROM iterations "
            "WHERE run_idx=? AND iteration=? AND tag=?",
            [run_idx, iteration, tag],
        ).fetchone()
        if row:
            return row[0]
        self._conn.execute(
            "INSERT INTO iterations "
            "(run_idx, iteration, client_id, tag, timestamp, n_models_proposed) "
            "VALUES (?,?,?,?,?,?)",
            [run_idx, iteration, client_id, tag, timestamp, n_models_proposed],
        )
        row = self._conn.execute(
            "SELECT iteration_id FROM iterations "
            "WHERE run_idx=? AND iteration=? AND tag=?",
            [run_idx, iteration, tag],
        ).fetchone()
        return row[0]

    def _insert_model(
        self,
        iteration_id: int,
        run_idx: int,
        iteration: int,
        name: str,
        code: str | None,
        metric_name: str | None,
        metric_value: float | None,
        param_names: list,
        status: str,
        mean_nll: float | None = None,
        split: str = "train",
    ) -> int:
        """Insert a model row and return its model_id.
        
        Pre-allocates model_id from sequence to avoid post-INSERT SELECT.
        Must be called within a transaction and with the lock held.
        """
        # Pre-allocate model_id from sequence (Opt-C)
        model_id = self._conn.execute("SELECT nextval('models_id_seq')").fetchone()[0]
        
        self._conn.execute(
            "INSERT INTO models "
            "(model_id, iteration_id, run_idx, iteration, name, code, metric_name, "
            " metric_value, mean_nll, split, param_names, status) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            [
                model_id,
                iteration_id,
                run_idx,
                iteration,
                name,
                code,
                metric_name,
                metric_value,
                mean_nll,
                split,
                orjson.dumps(param_names).decode(),
                status,
            ],
        )
        return model_id

    # ------------------------------------------------------------------ #
    # Public write interface
    # ------------------------------------------------------------------ #

    def write_iteration(
        self,
        iteration: int,
        run_idx: int,
        iteration_results: list[dict],
        ppc_results: dict | None = None,
        tag: str = "",
        client_id: str | None = None,
        timestamp: str | None = None,
    ) -> None:
        """Persist one iteration's results to the store.

        Parameters
        ----------
        iteration:
            Zero-based iteration index.
        run_idx:
            Index of the independent run (distributed mode).
        iteration_results:
            The list of per-model dicts produced by the GeCCo model loop
            (same structure written to ``bics/iterN_runX.json``).
        ppc_results:
            Optional dict mapping model name → PPC output from
            :func:`gecco.offline_evaluation.ppc.compute_ppc`.
        tag:
            File tag used to disambiguate distributed/individual runs.
        client_id:
            Client identifier in distributed mode.
        timestamp:
            ISO timestamp string; defaults to current time.
        """
        from datetime import datetime, timezone

        if timestamp is None:
            timestamp = datetime.now(timezone.utc).isoformat()

        ppc_results = ppc_results or {}
        n_models = len(iteration_results)

        with self._lock:
            # Make iteration ingestion idempotent: clear any existing rows
            # for this (run_idx, iteration, tag) before inserting fresh data.
            # We must do this outside the transaction because DuckDB checks
            # foreign key constraints immediately and the subquery-based deletes
            # don't affect constraint checking within the same transaction.
            tag_clean = tag or ""
            iteration_row = self._conn.execute(
                "SELECT iteration_id FROM iterations "
                "WHERE run_idx=? AND iteration=? AND tag=?",
                [run_idx, iteration, tag_clean],
            ).fetchone()
            
            if iteration_row:
                iteration_id = iteration_row[0]
                # Get model IDs to delete
                model_rows = self._conn.execute(
                    "SELECT model_id FROM models WHERE iteration_id=?",
                    [iteration_id],
                ).fetchall()
                model_ids = [r[0] for r in model_rows]
                
                if model_ids:
                    placeholders = ','.join(['?' for _ in model_ids])
                    # Delete from child tables first
                    for table in ['model_participants', 'parameter_recovery', 
                                  'individual_differences', 'ppc', 
                                  'block_residuals', 'validation_errors']:
                        self._conn.execute(
                            f"DELETE FROM {table} WHERE model_id IN ({placeholders})",
                            model_ids,
                        )
                    # Delete from models
                    self._conn.execute(
                        f"DELETE FROM models WHERE model_id IN ({placeholders})",
                        model_ids,
                    )
                # Delete the iteration row itself
                self._conn.execute(
                    "DELETE FROM iterations WHERE iteration_id=?",
                    [iteration_id],
                )
            
            # Begin explicit transaction for inserts (Opt-A)
            self._conn.execute("BEGIN TRANSACTION")
            try:
                iteration_id = self._get_or_create_iteration(
                    run_idx, iteration, tag, client_id, timestamp, n_models
                )

                # Batch accumulators for bulk inserts (Opt-B)
                participant_rows = []
                ppc_rows = []
                block_residual_rows = []

                for result in iteration_results:
                    name = result.get("function_name", "unknown")
                    metric_name = result.get("metric_name")
                    metric_value = result.get("metric_value")
                    param_names = result.get("param_names") or []
                    code = result.get("code")

                    # Derive status from metric_name
                    if metric_name in ("RECOVERY_FAILED", "FIT_ERROR", "VALIDATION_ERROR"):
                        status = metric_name.lower()
                        metric_value_store = None
                    else:
                        status = "ok"
                        metric_value_store = (
                            float(metric_value)
                            if metric_value is not None and metric_value != float("inf")
                            else None
                        )

                    model_id = self._insert_model(
                        iteration_id=iteration_id,
                        run_idx=run_idx,
                        iteration=iteration,
                        name=name,
                        code=code,
                        metric_name=metric_name,
                        metric_value=metric_value_store,
                        param_names=param_names,
                        status=status,
                        mean_nll=result.get("mean_nll"),
                        split="train",
                    )

                    # ---- per-participant data (train split) - accumulate for batch insert ----
                    eval_metrics = result.get("eval_metrics") or []
                    param_values = result.get("parameter_values") or []
                    n_trials_list = result.get("participant_n_trials") or []
                    per_participant_nll = result.get("per_participant_nll") or []

                    for idx, bic_val in enumerate(eval_metrics):
                        params_i = param_values[idx] if idx < len(param_values) else []
                        n_trials_i = n_trials_list[idx] if idx < len(n_trials_list) else None
                        nll_i = (
                            per_participant_nll[idx] if idx < len(per_participant_nll) else None
                        )
                        params_dict = {}
                        for pi, pname in enumerate(param_names):
                            if pi < len(params_i):
                                params_dict[pname] = float(params_i[pi])
                        participant_rows.append([
                            model_id,
                            idx,
                            float(bic_val) if bic_val is not None else None,
                            nll_i,
                            n_trials_i,
                            orjson.dumps(params_dict).decode(),
                        ])

                    # ---- parameter recovery ----
                    recovery = result.get("recovery")
                    if recovery is None and metric_name == "RECOVERY_FAILED":
                        # Reconstruct a minimal recovery record from the result dict
                        recovery = {
                            "passed": False,
                            "mean_r": result.get("recovery_r", 0.0),
                            "n_successful": result.get("recovery_n_successful", 0),
                            "per_param_r": result.get("recovery_per_param", {}),
                            "simulation_error": result.get("simulation_error"),
                        }
                    if recovery is not None:
                        self._conn.execute(
                            "INSERT OR REPLACE INTO parameter_recovery "
                            "(model_id, passed, mean_r, n_successful, per_param_r, simulation_error) "
                            "VALUES (?,?,?,?,?,?)",
                            [
                                model_id,
                                bool(recovery.get("passed", False)),
                                recovery.get("mean_r"),
                                recovery.get("n_successful"),
                                orjson.dumps(recovery.get("per_param_r", {})).decode(),
                                recovery.get("simulation_error"),
                            ],
                        )

                    # ---- individual differences (train split) ----
                    id_results = result.get("individual_differences")
                    if id_results:
                        self._conn.execute(
                            "INSERT OR REPLACE INTO individual_differences "
                            "(model_id, mean_r2, max_r2, best_param, per_param_r2, per_param_detail, split) "
                            "VALUES (?,?,?,?,?,?,?)",
                            [
                                model_id,
                                id_results.get("mean_r2"),
                                id_results.get("max_r2"),
                                id_results.get("best_param"),
                                orjson.dumps(id_results.get("per_param_r2", {})).decode(),
                                orjson.dumps(id_results.get("per_param_detail", {})).decode(),
                                "train",
                            ],
                        )

                    # ---- val split: write second model entry if val metrics present ----
                    val_metric_value = result.get("val_metric_value")
                    val_mean_nll = result.get("val_mean_nll")
                    val_model_id = None
                    if val_metric_value is not None:
                        val_model_id = self._insert_model(
                            iteration_id=iteration_id,
                            run_idx=run_idx,
                            iteration=iteration,
                            name=name,
                            code=code,
                            metric_name=metric_name,
                            metric_value=float(val_metric_value)
                            if val_metric_value != float("inf")
                            else None,
                            param_names=param_names,
                            status="ok",
                            mean_nll=val_mean_nll,
                            split="val",
                        )

                        # ---- val individual differences ----
                        val_id_results = result.get("val_individual_differences")
                        if val_id_results:
                            self._conn.execute(
                                "INSERT OR REPLACE INTO individual_differences "
                                "(model_id, mean_r2, max_r2, best_param, per_param_r2, per_param_detail, split) "
                                "VALUES (?,?,?,?,?,?,?)",
                                [
                                    val_model_id,
                                    val_id_results.get("mean_r2"),
                                    val_id_results.get("max_r2"),
                                    val_id_results.get("best_param"),
                                    orjson.dumps(val_id_results.get("per_param_r2", {})).decode(),
                                    orjson.dumps(val_id_results.get("per_param_detail", {})).decode(),
                                    "val",
                                ],
                            )

                    # ---- validation errors ----
                    if metric_name == "VALIDATION_ERROR":
                        self._conn.execute(
                            "INSERT INTO validation_errors "
                            "(model_id, error_type, error_message, error_details) "
                            "VALUES (?,?,?,?)",
                            [
                                model_id,
                                result.get("error_type"),
                                result.get("error_message"),
                                orjson.dumps(result.get("error_details", {})).decode(),
                            ],
                        )

                    # ---- PPC - accumulate for batch insert ----
                    ppc_data = ppc_results.get(name)
                    if ppc_data:
                        records = (
                            ppc_data if isinstance(ppc_data, list) else ppc_data.get("records", [])
                        )
                        for rec in records:
                            ppc_rows.append([
                                model_id,
                                str(rec.get("participant_id", "")),
                                rec.get("statistic_name", ""),
                                rec.get("condition"),
                                rec.get("observed"),
                                rec.get("simulated_mean"),
                                rec.get("simulated_q025"),
                                rec.get("simulated_q975"),
                                rec.get("n_sims"),
                            ])

                    # ---- block residuals - accumulate for batch insert ----
                    block_residuals = result.get("block_residuals")
                    if block_residuals:
                        records = (
                            block_residuals
                            if isinstance(block_residuals, list)
                            else block_residuals.get("records", [])
                        )
                        for rec in records:
                            block_residual_rows.append([
                                model_id,
                                str(rec.get("participant_id", "")),
                                rec.get("block_idx"),
                                rec.get("block_start"),
                                rec.get("block_end"),
                                rec.get("mean_nll_per_trial"),
                                rec.get("n_trials"),
                            ])

                # Bulk insert all accumulated rows (Opt-B)
                # DuckDB evaluates nextval() per row in executemany, so this is safe.
                if participant_rows:
                    self._conn.executemany(
                        "INSERT INTO model_participants "
                        "(id, model_id, participant_idx, bic, nll, n_trials, params) "
                        "VALUES (nextval('model_participants_id_seq'),?,?,?,?,?,?)",
                        participant_rows,
                    )

                if ppc_rows:
                    self._conn.executemany(
                        "INSERT INTO ppc "
                        "(model_id, participant_id, statistic_name, condition, "
                        " observed, simulated_mean, simulated_q025, simulated_q975, n_sims) "
                        "VALUES (?,?,?,?,?,?,?,?,?)",
                        ppc_rows,
                    )

                if block_residual_rows:
                    self._conn.executemany(
                        "INSERT INTO block_residuals "
                        "(model_id, participant_id, block_idx, block_start, block_end, "
                        " mean_nll_per_trial, n_trials) "
                        "VALUES (?,?,?,?,?,?,?)",
                        block_residual_rows,
                    )

                # Commit transaction (Opt-A)
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    def write_top_model_test(self, entry: dict) -> None:
        """Write a top model test evaluation entry.

        Parameters
        ----------
        entry:
            Dict with keys: model_name, val_nll, test_mean_BIC, test_mean_NLL,
            test_individual_BIC, test_individual_NLL, test_individual_differences.
        """
        model_name = entry.get("model_name", "unknown")
        val_nll = entry.get("val_nll")
        test_mean_bic = entry.get("test_mean_BIC")
        test_mean_nll = entry.get("test_mean_NLL")
        test_individual_bic = entry.get("test_individual_BIC", [])
        test_individual_nll = entry.get("test_individual_NLL", [])
        test_id_results = entry.get("test_individual_differences")

        self.execute(
            "INSERT INTO models "
            "(iteration_id, run_idx, iteration, name, code, metric_name, "
            " metric_value, mean_nll, split, param_names, status) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            [
                None,
                0,
                -1,
                model_name,
                None,
                "BIC",
                test_mean_bic,
                test_mean_nll,
                "test",
                orjson.dumps([]).decode(),
                "ok",
            ],
        )
        row = self.fetchone(
            "SELECT model_id FROM models WHERE name=? AND split='test' ORDER BY model_id DESC LIMIT 1",
            [model_name],
        )
        if row is None:
            return
        model_id = row["model_id"]

        if test_id_results:
            self.execute(
                "INSERT OR REPLACE INTO individual_differences "
                "(model_id, mean_r2, max_r2, best_param, per_param_r2, per_param_detail, split) "
                "VALUES (?,?,?,?,?,?,?)",
                [
                    model_id,
                    test_id_results.get("mean_r2"),
                    test_id_results.get("max_r2"),
                    test_id_results.get("best_param"),
                    orjson.dumps(test_id_results.get("per_param_r2", {})).decode(),
                    orjson.dumps(test_id_results.get("per_param_detail", {})).decode(),
                    "test",
                ],
            )

    def close(self) -> None:
        """Close the underlying DuckDB connection."""
        with self._lock:
            self._conn.close()
