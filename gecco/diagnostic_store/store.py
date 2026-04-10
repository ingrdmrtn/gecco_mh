"""
DiagnosticStore — thin wrapper around a DuckDB file that exposes
write and query helpers used by the rest of the diagnostic_store package.
"""

from __future__ import annotations

import json
import threading
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
        """Return the iteration_id, creating the row if absent."""
        tag = tag or ""
        row = self.fetchone(
            "SELECT iteration_id FROM iterations "
            "WHERE run_idx=? AND iteration=? AND tag=?",
            [run_idx, iteration, tag],
        )
        if row:
            return row["iteration_id"]
        self.execute(
            "INSERT INTO iterations "
            "(run_idx, iteration, client_id, tag, timestamp, n_models_proposed) "
            "VALUES (?,?,?,?,?,?)",
            [run_idx, iteration, client_id, tag, timestamp, n_models_proposed],
        )
        row = self.fetchone(
            "SELECT iteration_id FROM iterations "
            "WHERE run_idx=? AND iteration=? AND tag=?",
            [run_idx, iteration, tag],
        )
        return row["iteration_id"]

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
    ) -> int:
        """Insert a model row and return its model_id."""
        self.execute(
            "INSERT INTO models "
            "(iteration_id, run_idx, iteration, name, code, metric_name, "
            " metric_value, param_names, status) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            [
                iteration_id,
                run_idx,
                iteration,
                name,
                code,
                metric_name,
                metric_value,
                json.dumps(param_names),
                status,
            ],
        )
        row = self.fetchone(
            "SELECT model_id FROM models "
            "WHERE iteration_id=? AND name=? "
            "ORDER BY model_id DESC LIMIT 1",
            [iteration_id, name],
        )
        return row["model_id"]

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

        iteration_id = self._get_or_create_iteration(
            run_idx, iteration, tag, client_id, timestamp, n_models
        )

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
            )

            # ---- per-participant data ----
            eval_metrics = result.get("eval_metrics") or []
            param_values = result.get("parameter_values") or []
            n_trials_list = result.get("participant_n_trials") or []

            for idx, bic_val in enumerate(eval_metrics):
                params_i = param_values[idx] if idx < len(param_values) else []
                n_trials_i = n_trials_list[idx] if idx < len(n_trials_list) else None
                params_dict = {}
                for pi, pname in enumerate(param_names):
                    if pi < len(params_i):
                        params_dict[pname] = float(params_i[pi])
                self.execute(
                    "INSERT INTO model_participants "
                    "(id, model_id, participant_idx, bic, n_trials, params) "
                    "VALUES (nextval('model_participants_id_seq'),?,?,?,?,?)",
                    [
                        model_id,
                        idx,
                        float(bic_val) if bic_val is not None else None,
                        n_trials_i,
                        json.dumps(params_dict),
                    ],
                )

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
                self.execute(
                    "INSERT OR REPLACE INTO parameter_recovery "
                    "(model_id, passed, mean_r, n_successful, per_param_r, simulation_error) "
                    "VALUES (?,?,?,?,?,?)",
                    [
                        model_id,
                        bool(recovery.get("passed", False)),
                        recovery.get("mean_r"),
                        recovery.get("n_successful"),
                        json.dumps(recovery.get("per_param_r", {})),
                        recovery.get("simulation_error"),
                    ],
                )

            # ---- individual differences ----
            id_results = result.get("individual_differences")
            if id_results:
                self.execute(
                    "INSERT OR REPLACE INTO individual_differences "
                    "(model_id, mean_r2, max_r2, best_param, per_param_r2, per_param_detail) "
                    "VALUES (?,?,?,?,?,?)",
                    [
                        model_id,
                        id_results.get("mean_r2"),
                        id_results.get("max_r2"),
                        id_results.get("best_param"),
                        json.dumps(id_results.get("per_param_r2", {})),
                        json.dumps(id_results.get("per_param_detail", {})),
                    ],
                )

            # ---- validation errors ----
            if metric_name == "VALIDATION_ERROR":
                self.execute(
                    "INSERT INTO validation_errors "
                    "(model_id, error_type, error_message, error_details) "
                    "VALUES (?,?,?,?)",
                    [
                        model_id,
                        result.get("error_type"),
                        result.get("error_message"),
                        json.dumps(result.get("error_details", {})),
                    ],
                )

            # ---- PPC ----
            ppc_data = ppc_results.get(name)
            if ppc_data:
                self._write_ppc(model_id, ppc_data)

            # ---- block residuals ----
            block_residuals = result.get("block_residuals")
            if block_residuals:
                self._write_block_residuals(model_id, block_residuals)

    def _write_ppc(self, model_id: int, ppc_data: dict) -> None:
        """Write PPC rows for a single model.

        *ppc_data* is the structure returned by
        :func:`gecco.offline_evaluation.ppc.compute_ppc`:
        a dict mapping ``(participant_id, statistic_name, condition)`` → stat dict,
        or a list of flat stat records.
        """
        records = ppc_data if isinstance(ppc_data, list) else ppc_data.get("records", [])
        for rec in records:
            self.execute(
                "INSERT INTO ppc "
                "(model_id, participant_id, statistic_name, condition, "
                " observed, simulated_mean, simulated_q025, simulated_q975, n_sims) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                [
                    model_id,
                    str(rec.get("participant_id", "")),
                    rec.get("statistic_name", ""),
                    rec.get("condition"),
                    rec.get("observed"),
                    rec.get("simulated_mean"),
                    rec.get("simulated_q025"),
                    rec.get("simulated_q975"),
                    rec.get("n_sims"),
                ],
            )

    def _write_block_residuals(self, model_id: int, block_data: dict) -> None:
        """Write block residual rows for a single model."""
        records = (
            block_data if isinstance(block_data, list)
            else block_data.get("records", [])
        )
        for rec in records:
            self.execute(
                "INSERT INTO block_residuals "
                "(model_id, participant_id, block_idx, block_start, block_end, "
                " mean_nll_per_trial, n_trials) "
                "VALUES (?,?,?,?,?,?,?)",
                [
                    model_id,
                    str(rec.get("participant_id", "")),
                    rec.get("block_idx"),
                    rec.get("block_start"),
                    rec.get("block_end"),
                    rec.get("mean_nll_per_trial"),
                    rec.get("n_trials"),
                ],
            )

    def close(self) -> None:
        """Close the underlying DuckDB connection."""
        with self._lock:
            self._conn.close()
