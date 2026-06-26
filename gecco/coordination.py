"""
Distributed coordination for parallel GeCCo search clients.

Runtime coordination state is stored canonically in DuckDB. A lightweight
advisory lock file enforces a single-writer strategy across client processes,
and every mutation runs inside an explicit transaction.
"""

from __future__ import annotations

import fcntl
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

try:
    import duckdb
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "duckdb is required for distributed coordination. "
        "Install it with: pip install duckdb"
    ) from exc

from gecco.diagnostic_store.schema import create_schema
from gecco.sentry_init import capture_coordination_error
from gecco.utils import TimestampedConsole

console = TimestampedConsole()


class SharedRegistry:
    """DuckDB-backed coordination registry for distributed GeCCo clients."""

    def __init__(self, registry_path: str | Path):
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = self.db_path_for_registry(self.registry_path)
        self.lock_path = Path(f"{self.db_path}.lock")
        self.lock_path.touch(exist_ok=True)
        self._initialise_store()

    @classmethod
    def open_existing(cls, registry_path: str | Path) -> "SharedRegistry":
        """Open an existing registry without initialising schema.

        Args:
            registry_path: Path to the registry file used to derive the DuckDB file.

        Returns:
            A registry handle that can be used for read-only access.
        """
        instance = cls.__new__(cls)
        instance.registry_path = Path(registry_path)
        instance.db_path = cls.db_path_for_registry(instance.registry_path)
        instance.lock_path = Path(f"{instance.db_path}.lock")
        return instance

    @staticmethod
    def db_path_for_registry(registry_path: str | Path) -> Path:
        """Return the canonical DuckDB path for a registry path."""
        path = Path(registry_path)
        return path.with_suffix(".duckdb")

    @staticmethod
    def _empty_registry() -> dict[str, Any]:
        return {
            "global_best": None,
            "baseline": None,
            "abort": None,
            "tried_param_sets": [],
            "client_entries": {},
            "iteration_history": [],
            "candidate_generations": {},
            "generator_status": {},
            "judge_iterations": {},
        }

    @staticmethod
    def _client_key(client_id: Any) -> str:
        return str(client_id)

    @staticmethod
    def _restore_client_id(value: Any) -> Any:
        if isinstance(value, str) and value.lstrip("-").isdigit():
            try:
                return int(value)
            except ValueError:
                return value
        return value

    @staticmethod
    def _to_json_text(value: Any) -> str:
        return json.dumps(value)

    @staticmethod
    def _from_json_value(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        return value

    def _initialise_store(self) -> None:
        def _create(connection):
            create_schema(connection)

        self._with_connection(
            write=True,
            operation="initialise",
            callback=_create,
            initialise_schema=True,
        )

    def _with_connection(
        self,
        *,
        write: bool,
        operation: str,
        callback,
        initialise_schema: bool = False,
    ):
        lock_file = None
        connection = None
        try:
            if write or self.lock_path.exists():
                lock_file = open(self.lock_path, "a+")
                lock_mode = fcntl.LOCK_EX if write else fcntl.LOCK_SH
                fcntl.flock(lock_file.fileno(), lock_mode)

            connect_kwargs = {} if write else {"read_only": True}
            connection = duckdb.connect(str(self.db_path), **connect_kwargs)
            try:
                if initialise_schema:
                    create_schema(connection)
                if write:
                    connection.execute("BEGIN TRANSACTION")
                result = callback(connection)
                if write:
                    connection.execute("COMMIT")
                return result
            except Exception as exc:
                if write:
                    try:
                        connection.execute("ROLLBACK")
                    except Exception:
                        pass
                capture_coordination_error(error=exc, operation=operation)
                raise
            finally:
                if connection is not None:
                    connection.close()
        finally:
            if lock_file is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                lock_file.close()

    def _fetchone(self, sql: str, params: list[Any] | None = None) -> dict[str, Any] | None:
        def _query(conn):
            cursor = conn.execute(sql, params or [])
            row = cursor.fetchone()
            if row is None:
                return None
            columns = [desc[0] for desc in cursor.description]
            return dict(zip(columns, row))

        return self._with_connection(write=False, operation="fetchone", callback=_query)

    def _fetchall(self, sql: str, params: list[Any] | None = None) -> list[dict[str, Any]]:
        def _query(conn):
            cursor = conn.execute(sql, params or [])
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]

        return self._with_connection(write=False, operation="fetchall", callback=_query)

    def read(self) -> dict[str, Any]:
        """Read the current registry state from DuckDB."""

        def _read(conn):
            data = self._empty_registry()

            global_best = conn.execute(
                "SELECT metric_value, model_code, param_names, client_id, iteration "
                "FROM runtime_global_best WHERE singleton = 1"
            ).fetchone()
            if global_best is not None:
                data["global_best"] = {
                    "metric_value": global_best[0],
                    "model_code": global_best[1],
                    "param_names": self._from_json_value(global_best[2]) or [],
                    "client_id": self._restore_client_id(global_best[3]),
                    "iteration": global_best[4],
                }

            baseline = conn.execute(
                "SELECT function_name, executable_function_name, metric_name, metric_value, "
                "param_names, eval_metrics, mean_r2, max_r2, best_param, per_param_r2, "
                "code, val_mean_nll "
                "FROM runtime_baseline WHERE singleton = 1"
            ).fetchone()
            if baseline is not None:
                data["baseline"] = {
                    "function_name": baseline[0],
                    "executable_function_name": baseline[1],
                    "metric_name": baseline[2],
                    "metric_value": baseline[3],
                    "param_names": self._from_json_value(baseline[4]) or [],
                    "eval_metrics": self._from_json_value(baseline[5]) or [],
                    "mean_r2": baseline[6],
                    "max_r2": baseline[7],
                    "best_param": baseline[8],
                    "per_param_r2": self._from_json_value(baseline[9]) or {},
                    "code": baseline[10],
                    "val_mean_nll": baseline[11],
                }

            for row in conn.execute(
                "SELECT param_set FROM runtime_tried_param_sets ORDER BY param_key"
            ).fetchall():
                data["tried_param_sets"].append(self._from_json_value(row[0]))

            for row in conn.execute(
                "SELECT client_id, last_iteration, best_metric, status, updated_at, "
                "had_runnable_model, activity "
                "FROM runtime_client_entries ORDER BY client_id"
            ).fetchall():
                data["client_entries"][row[0]] = {
                    "last_iteration": row[1],
                    "best_metric": row[2],
                    "status": row[3],
                    "updated_at": row[4],
                    "had_runnable_model": row[5],
                    "activity": row[6],
                }

            for row in conn.execute(
                "SELECT client_id, iteration, results, status, had_runnable_model "
                "FROM runtime_iteration_history ORDER BY created_at, client_id"
            ).fetchall():
                history_entry = {
                    "client_id": self._restore_client_id(row[0]),
                    "iteration": row[1],
                    "results": self._from_json_value(row[2]) or [],
                }
                if row[3] is not None:
                    history_entry["status"] = row[3]
                if row[4] is not None:
                    history_entry["had_runnable_model"] = row[4]
                data["iteration_history"].append(history_entry)

            for row in conn.execute(
                "SELECT iteration, candidates, generated_by, timestamp "
                "FROM runtime_candidate_generations ORDER BY iteration"
            ).fetchall():
                data["candidate_generations"][str(row[0])] = {
                    "candidates": self._from_json_value(row[1]) or [],
                    "generated_by": self._restore_client_id(row[2]),
                    "timestamp": row[3],
                }

            for row in conn.execute(
                "SELECT iteration, client_id, status, n_candidates, error, updated_at "
                "FROM runtime_generator_status ORDER BY iteration"
            ).fetchall():
                entry = {
                    "client_id": self._restore_client_id(row[1]),
                    "status": row[2],
                    "n_candidates": row[3],
                    "updated_at": row[5],
                }
                if row[4] is not None:
                    entry["error"] = row[4]
                data["generator_status"][str(row[0])] = entry

            for row in conn.execute(
                "SELECT iteration, synthesized_feedback, verdict, failed, error, timestamp "
                "FROM runtime_judge_iterations ORDER BY iteration"
            ).fetchall():
                entry = {
                    "synthesized_feedback": self._from_json_value(row[1]) or {},
                    "verdict": self._from_json_value(row[2]) or {},
                    "timestamp": row[5],
                }
                if row[3]:
                    entry["failed"] = True
                    entry["error"] = row[4]
                data["judge_iterations"][str(row[0])] = entry

            abort_row = conn.execute(
                "SELECT client_id, iteration, reason, status, created_at "
                "FROM runtime_abort WHERE singleton = 1"
            ).fetchone()
            if abort_row is not None:
                data["abort"] = {
                    "client_id": self._restore_client_id(abort_row[0]),
                    "iteration": abort_row[1],
                    "reason": abort_row[2],
                    "status": abort_row[3],
                    "created_at": abort_row[4],
                }

            return data

        return self._with_connection(write=False, operation="read", callback=_read)

    def update(
        self,
        client_id,
        iteration,
        results,
        best_model=None,
        best_metric=None,
        param_names=None,
        tried_param_sets=None,
        status="running",
        had_runnable_model=None,
    ):
        """Atomically merge this client's iteration results into DuckDB."""

        client_key = self._client_key(client_id)
        timestamp = datetime.now().isoformat()
        tried_param_sets = tried_param_sets or []

        serializable_results = []
        for result in results:
            entry = {
                "function_name": result.get("function_name", ""),
                "metric_name": result.get("metric_name", "BIC"),
                "metric_value": result.get("metric_value", float("inf")),
                "param_names": result.get("param_names", []),
                "code": result.get("code", ""),
            }
            if result.get("error"):
                entry["error"] = result["error"]
            if result.get("recovery_r") is not None:
                entry["recovery_r"] = result["recovery_r"]
            if result.get("recovery_per_param"):
                entry["recovery_per_param"] = result["recovery_per_param"]
            if result.get("eval_metrics"):
                entry["eval_metrics"] = result.get("eval_metrics")
            if result.get("participant_n_trials"):
                entry["participant_n_trials"] = result.get("participant_n_trials")
            id_res = result.get("individual_differences")
            if id_res and isinstance(id_res, dict):
                entry["individual_differences"] = {
                    "mean_r2": id_res.get("mean_r2"),
                    "max_r2": id_res.get("max_r2"),
                    "best_param": id_res.get("best_param"),
                    "per_param_r2": id_res.get("per_param_r2"),
                    "summary_text": id_res.get("summary_text", ""),
                }
                entry["mean_r2"] = id_res.get("mean_r2")
                entry["max_r2"] = id_res.get("max_r2")
                entry["best_param"] = id_res.get("best_param")
                entry["per_param_r2"] = id_res.get("per_param_r2")
            if result.get("val_metric_value") is not None:
                entry["val_metric_value"] = result["val_metric_value"]
                entry["val_mean_nll"] = result.get("val_mean_nll")
                entry["val_eval_metrics"] = result.get("val_eval_metrics", [])
                entry["val_per_participant_nll"] = result.get(
                    "val_per_participant_nll", []
                )
            val_id = result.get("val_individual_differences")
            if val_id and isinstance(val_id, dict):
                entry["val_individual_differences"] = {
                    "mean_r2": val_id.get("mean_r2"),
                    "max_r2": val_id.get("max_r2"),
                    "best_param": val_id.get("best_param"),
                    "per_param_r2": val_id.get("per_param_r2"),
                }
            serializable_results.append(entry)

        def _update(conn):
            existing = conn.execute(
                "SELECT activity FROM runtime_client_entries WHERE client_id = ?",
                [client_key],
            ).fetchone()
            activity = existing[0] if existing is not None else None
            history_row = conn.execute(
                "SELECT created_at FROM runtime_iteration_history "
                "WHERE client_id = ? AND iteration = ?",
                [client_key, iteration],
            ).fetchone()
            existing_created_at = (
                history_row[0] if history_row is not None and history_row[0] else timestamp
            )

            conn.execute(
                "INSERT OR REPLACE INTO runtime_client_entries "
                "(client_id, last_iteration, best_metric, status, updated_at, had_runnable_model, activity) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    client_key,
                    iteration,
                    best_metric,
                    status,
                    timestamp,
                    had_runnable_model,
                    activity,
                ],
            )

            conn.execute(
                "INSERT OR REPLACE INTO runtime_iteration_history "
                "(client_id, iteration, results, status, had_runnable_model, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    client_key,
                    iteration,
                    self._to_json_text(serializable_results),
                    status,
                    had_runnable_model,
                    existing_created_at,
                    timestamp,
                ],
            )

            for param_set in tried_param_sets:
                param_key = self._to_json_text(param_set)
                conn.execute(
                    "INSERT OR IGNORE INTO runtime_tried_param_sets (param_key, param_set) "
                    "VALUES (?, ?)",
                    [param_key, param_key],
                )

            if best_metric is not None and best_model is not None:
                current = conn.execute(
                    "SELECT metric_value FROM runtime_global_best WHERE singleton = 1"
                ).fetchone()
                if current is None or current[0] is None or best_metric < current[0]:
                    conn.execute(
                        "INSERT OR REPLACE INTO runtime_global_best "
                        "(singleton, metric_value, model_code, param_names, client_id, iteration) "
                        "VALUES (1, ?, ?, ?, ?, ?)",
                        [
                            best_metric,
                            best_model,
                            self._to_json_text(param_names or []),
                            client_key,
                            iteration,
                        ],
                    )

        self._with_connection(write=True, operation="update", callback=_update)

    def set_client_status(self, client_id, status, activity=None):
        """Update a client's status without writing a new iteration row."""

        client_key = self._client_key(client_id)

        def _set(conn):
            existing = conn.execute(
                "SELECT last_iteration, best_metric, had_runnable_model, activity "
                "FROM runtime_client_entries WHERE client_id = ?",
                [client_key],
            ).fetchone()
            timestamp = datetime.now().isoformat()
            if existing is None:
                conn.execute(
                    "INSERT INTO runtime_client_entries "
                    "(client_id, last_iteration, best_metric, status, updated_at, had_runnable_model, activity) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    [client_key, None, None, status, timestamp, None, activity],
                )
            else:
                conn.execute(
                    "UPDATE runtime_client_entries SET status = ?, updated_at = ?, activity = COALESCE(?, activity) "
                    "WHERE client_id = ?",
                    [status, timestamp, activity, client_key],
                )

        self._with_connection(write=True, operation="set-client-status", callback=_set)

    def get_max_iteration(self):
        """Return the highest iteration number across all clients, or -1."""
        row = self._fetchone(
            "SELECT MAX(iteration) AS max_iteration FROM runtime_iteration_history"
        )
        return row["max_iteration"] if row and row["max_iteration"] is not None else -1

    def get_max_iteration_for_client(self, client_id):
        """Return highest fully completed iteration for this client, or -1."""
        client_key = self._client_key(client_id)
        row = self._fetchone(
            "SELECT MAX(iteration) AS max_iteration FROM runtime_iteration_history "
            "WHERE client_id = ? AND status IN ('complete', 'complete_no_success')",
            [client_key],
        )
        return row["max_iteration"] if row and row["max_iteration"] is not None else -1

    def get_max_generator_iteration(self, client_id):
        """Return highest completed CMG generator iteration for this client, or -1."""
        row = self._fetchone(
            "SELECT MAX(gs.iteration) AS max_iteration "
            "FROM runtime_generator_status gs "
            "JOIN runtime_candidate_generations cg ON cg.iteration = gs.iteration "
            "WHERE gs.client_id = ? AND gs.status = 'complete'",
            [self._client_key(client_id)],
        )
        return row["max_iteration"] if row and row["max_iteration"] is not None else -1

    def set_activity(self, client_id, activity):
        """Update a client's current activity without pushing results."""

        def _set(conn):
            existing = conn.execute(
                "SELECT last_iteration, best_metric, status, had_runnable_model "
                "FROM runtime_client_entries WHERE client_id = ?",
                [self._client_key(client_id)],
            ).fetchone()
            if existing is None:
                conn.execute(
                    "INSERT INTO runtime_client_entries "
                    "(client_id, last_iteration, best_metric, status, updated_at, had_runnable_model, activity) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    [
                        self._client_key(client_id),
                        None,
                        None,
                        None,
                        datetime.now().isoformat(),
                        None,
                        activity,
                    ],
                )
            else:
                conn.execute(
                    "UPDATE runtime_client_entries SET activity = ?, updated_at = ? WHERE client_id = ?",
                    [activity, datetime.now().isoformat(), self._client_key(client_id)],
                )

        self._with_connection(write=True, operation="set-activity", callback=_set)

    def set_baseline(self, baseline_result):
        """Write baseline result to the canonical runtime store."""

        def _set(conn):
            id_res = baseline_result.get("individual_differences") or {}
            conn.execute(
                "INSERT OR REPLACE INTO runtime_baseline "
                "(singleton, function_name, executable_function_name, metric_name, metric_value, "
                "param_names, eval_metrics, mean_r2, max_r2, best_param, per_param_r2, code, "
                "val_mean_nll) "
                "VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    baseline_result.get("function_name", "baseline_model"),
                    baseline_result.get("executable_function_name"),
                    baseline_result.get("metric_name", "BIC"),
                    baseline_result.get("metric_value"),
                    self._to_json_text(baseline_result.get("param_names", [])),
                    self._to_json_text(baseline_result.get("eval_metrics", [])),
                    id_res.get("mean_r2"),
                    id_res.get("max_r2"),
                    id_res.get("best_param"),
                    self._to_json_text(id_res.get("per_param_r2", {})),
                    baseline_result.get("code"),
                    baseline_result.get("val_mean_nll"),
                ],
            )

        self._with_connection(write=True, operation="set-baseline", callback=_set)

    def count_clients_at_iteration(self, iteration: int) -> int:
        """Count the number of distinct clients that wrote results for an iteration."""
        row = self._fetchone(
            "SELECT COUNT(*) AS n_clients FROM runtime_iteration_history WHERE iteration = ?",
            [iteration],
        )
        return int(row["n_clients"]) if row else 0

    def wait_for_iteration(
        self,
        iteration: int,
        n_expected: int,
        timeout_seconds: float,
        poll_seconds: float = 2.0,
    ) -> int:
        """Poll until at least *n_expected* clients have written iteration results."""
        start_time = time.time()
        while True:
            self.raise_if_aborted()
            count = self.count_clients_at_iteration(iteration)
            if count >= n_expected:
                console.print(
                    f"[green]Iteration {iteration} complete: {count}/{n_expected} clients "
                    f"in {time.time() - start_time:.1f}s[/]"
                )
                return count

            self.raise_if_aborted()
            elapsed = time.time() - start_time
            if elapsed >= timeout_seconds:
                console.print(
                    f"[yellow]Iteration {iteration} timeout: got {count}/{n_expected} clients "
                    f"after {elapsed:.1f}s, proceeding with available results[/]"
                )
                return count

            time.sleep(poll_seconds)

    def count_clients_complete(self, iteration: int) -> int:
        """Count clients who completed an iteration (not retrying)."""
        row = self._fetchone(
            "SELECT COUNT(*) AS n_clients "
            "FROM runtime_iteration_history "
            "WHERE iteration = ? AND status IN ('complete', 'complete_no_success')",
            [iteration],
        )
        return int(row["n_clients"]) if row else 0

    def count_clients_with_models(self, iteration: int) -> int:
        """Count clients who produced at least one runnable model."""
        row = self._fetchone(
            "SELECT COUNT(*) AS n_clients "
            "FROM runtime_iteration_history "
            "WHERE iteration = ? AND COALESCE(had_runnable_model, FALSE)",
            [iteration],
        )
        return int(row["n_clients"]) if row else 0

    def wait_for_clients_complete(
        self,
        iteration: int,
        n_expected: int,
        timeout_seconds: float,
        poll_seconds: float = 5.0,
    ) -> int:
        """Poll until *n_expected* clients have completed an iteration."""
        start_time = time.time()
        while True:
            self.raise_if_aborted()
            count = self.count_clients_complete(iteration)
            if count >= n_expected:
                console.print(
                    f"[green]Iteration {iteration} complete: {count}/{n_expected} clients "
                    f"in {time.time() - start_time:.1f}s[/]"
                )
                return count

            self.raise_if_aborted()
            elapsed = time.time() - start_time
            if elapsed >= timeout_seconds:
                console.print(
                    f"[yellow]Iteration {iteration} timeout: got {count}/{n_expected} complete clients "
                    f"after {elapsed:.1f}s, proceeding with available results[/]"
                )
                return count

            time.sleep(poll_seconds)

    def set_judge_feedback(
        self,
        iteration: int,
        synthesized_feedback: str | dict,
        verdict_payload: dict,
    ) -> None:
        """Store the shared judge verdict for an iteration."""

        feedback_dict = (
            {"default": synthesized_feedback}
            if isinstance(synthesized_feedback, str)
            else synthesized_feedback
        )

        def _set(conn):
            conn.execute(
                "INSERT OR REPLACE INTO runtime_judge_iterations "
                "(iteration, synthesized_feedback, verdict, failed, error, timestamp) "
                "VALUES (?, ?, ?, FALSE, NULL, ?)",
                [
                    iteration,
                    self._to_json_text(feedback_dict),
                    self._to_json_text(verdict_payload),
                    datetime.now().isoformat(),
                ],
            )

        self._with_connection(write=True, operation="set-judge-feedback", callback=_set)

    def set_judge_failure(self, iteration: int, error: str) -> None:
        """Write an explicit failure entry for an iteration."""

        def _set(conn):
            conn.execute(
                "INSERT OR REPLACE INTO runtime_judge_iterations "
                "(iteration, synthesized_feedback, verdict, failed, error, timestamp) "
                "VALUES (?, ?, ?, TRUE, ?, ?)",
                [
                    iteration,
                    self._to_json_text({}),
                    self._to_json_text({}),
                    error,
                    datetime.now().isoformat(),
                ],
            )

        self._with_connection(write=True, operation="set-judge-failure", callback=_set)

    def request_abort(
        self,
        *,
        client_id: Any,
        reason: str,
        iteration: int | None = None,
        status: str = "failed",
    ) -> None:
        """Persist a shared abort request for distributed clients."""

        def _set(conn):
            conn.execute(
                "INSERT OR IGNORE INTO runtime_abort "
                "(singleton, client_id, iteration, reason, status, created_at) "
                "VALUES (1, ?, ?, ?, ?, ?)",
                [
                    self._client_key(client_id),
                    iteration,
                    reason,
                    status,
                    datetime.now().isoformat(),
                ],
            )

        self._with_connection(write=True, operation="request-abort", callback=_set)

    def get_abort(self) -> Optional[dict]:
        """Return the active abort record, if present."""

        row = self._fetchone(
            "SELECT client_id, iteration, reason, status, created_at "
            "FROM runtime_abort WHERE singleton = 1",
        )
        if row is None:
            return None
        return {
            "client_id": self._restore_client_id(row["client_id"]),
            "iteration": row["iteration"],
            "reason": row["reason"],
            "status": row["status"],
            "created_at": row["created_at"],
        }

    def raise_if_aborted(self) -> None:
        """Raise if any client has requested a shared abort."""

        abort = self.get_abort()
        if abort is None:
            return None

        iteration = abort.get("iteration")
        iteration_fragment = f" at iteration {iteration}" if iteration is not None else ""
        message = (
            f"Distributed run aborted by client {abort.get('client_id')}{iteration_fragment}: "
            f"{abort.get('reason', 'unknown reason')}"
        )
        console.print(f"[red]{message}[/]")
        raise RuntimeError(message)

    def get_judge_feedback(self, iteration: int) -> Optional[dict]:
        """Retrieve the stored judge verdict for an iteration."""
        row = self._fetchone(
            "SELECT synthesized_feedback, verdict, failed, error, timestamp "
            "FROM runtime_judge_iterations WHERE iteration = ?",
            [iteration],
        )
        if row is None:
            return None
        result = {
            "synthesized_feedback": self._from_json_value(row["synthesized_feedback"]) or {},
            "verdict": self._from_json_value(row["verdict"]) or {},
            "timestamp": row["timestamp"],
        }
        if row["failed"]:
            result["failed"] = True
            result["error"] = row["error"]
        return result

    def get_judge_feedback_for_persona(
        self, iteration: int, persona_name: str, fallback: str = "default"
    ) -> Optional[dict]:
        """Retrieve persona-specific judge feedback for an iteration."""
        verdict_dict = self.get_judge_feedback(iteration)
        if verdict_dict is None:
            return None

        feedback_dict = verdict_dict.get("synthesized_feedback", {})
        if isinstance(feedback_dict, str):
            feedback_dict = {"default": feedback_dict}

        persona_feedback = feedback_dict.get(
            persona_name, feedback_dict.get(fallback, "")
        )
        return {
            "synthesized_feedback": persona_feedback,
            "verdict": verdict_dict.get("verdict", {}),
            "timestamp": verdict_dict.get("timestamp"),
        }

    def wait_for_judge_feedback(
        self,
        iteration: int,
        timeout_seconds: float,
        poll_seconds: float = 2.0,
    ) -> Optional[dict]:
        """Poll until judge feedback is available for an iteration, or timeout."""
        start_time = time.time()
        while True:
            self.raise_if_aborted()
            feedback = self.get_judge_feedback(iteration)
            if feedback is not None:
                elapsed = time.time() - start_time
                if feedback.get("failed"):
                    console.print(
                        f"[red]Orchestrated judge reported failure for iteration {iteration} "
                        f"after {elapsed:.1f}s: {feedback.get('error', 'unknown')}[/]"
                    )
                else:
                    console.print(
                        f"[green]Received judge feedback for iteration {iteration} "
                        f"in {elapsed:.1f}s[/]"
                    )
                return feedback

            self.raise_if_aborted()
            elapsed = time.time() - start_time
            if elapsed >= timeout_seconds:
                console.print(
                    f"[yellow]Timeout waiting for judge feedback (iteration {iteration}) "
                    f"after {elapsed:.1f}s[/]"
                )
                return None

            time.sleep(poll_seconds)

    def set_candidate_models(self, iteration, candidates, generated_by):
        """Publish generated candidates for one iteration."""

        def _set(conn):
            existing = conn.execute(
                "SELECT candidates, generated_by, timestamp "
                "FROM runtime_candidate_generations WHERE iteration = ?",
                [iteration],
            ).fetchone()
            if existing is not None:
                console.print(
                    f"[yellow]Candidates already exist for iteration {iteration}, not overwriting[/]"
                )
                return {
                    "candidates": self._from_json_value(existing[0]) or [],
                    "generated_by": self._restore_client_id(existing[1]),
                    "timestamp": existing[2],
                }

            entry = {
                "candidates": candidates,
                "generated_by": generated_by,
                "timestamp": datetime.now().isoformat(),
            }
            conn.execute(
                "INSERT INTO runtime_candidate_generations "
                "(iteration, candidates, generated_by, timestamp) VALUES (?, ?, ?, ?)",
                [
                    iteration,
                    self._to_json_text(candidates),
                    self._client_key(generated_by),
                    entry["timestamp"],
                ],
            )
            return entry

        return self._with_connection(
            write=True,
            operation="set-candidate-models",
            callback=_set,
        )

    def get_candidate_models(self, iteration) -> Optional[dict]:
        """Return published candidates for an iteration, or None."""
        row = self._fetchone(
            "SELECT candidates, generated_by, timestamp "
            "FROM runtime_candidate_generations WHERE iteration = ?",
            [iteration],
        )
        if row is None:
            return None
        return {
            "candidates": self._from_json_value(row["candidates"]) or [],
            "generated_by": self._restore_client_id(row["generated_by"]),
            "timestamp": row["timestamp"],
        }

    def wait_for_candidate_models(
        self,
        iteration,
        timeout_seconds: float,
        poll_seconds: float = 2.0,
    ) -> Optional[dict]:
        """Poll until candidate models are available or timeout."""
        start_time = time.time()
        while True:
            self.raise_if_aborted()
            result = self.get_candidate_models(iteration)
            if result is not None:
                elapsed = time.time() - start_time
                console.print(
                    f"[green]Received candidate models for iteration {iteration} "
                    f"in {elapsed:.1f}s[/]"
                )
                return result

            self.raise_if_aborted()
            elapsed = time.time() - start_time
            if elapsed >= timeout_seconds:
                console.print(
                    f"[yellow]Timeout waiting for candidate models (iteration {iteration}) "
                    f"after {elapsed:.1f}s[/]"
                )
                return None

            time.sleep(poll_seconds)

    def update_candidate_model(self, iteration, index, candidate):
        """Overwrite one candidate after evaluator repair."""

        def _update(conn):
            row = conn.execute(
                "SELECT candidates, generated_by, timestamp "
                "FROM runtime_candidate_generations WHERE iteration = ?",
                [iteration],
            ).fetchone()
            if row is None:
                raise ValueError(f"No candidate generation entry for iteration {iteration}")

            candidates = self._from_json_value(row[0]) or []
            for i, existing in enumerate(candidates):
                if existing.get("index") == index:
                    candidates[i] = candidate
                    conn.execute(
                        "UPDATE runtime_candidate_generations SET candidates = ? WHERE iteration = ?",
                        [self._to_json_text(candidates), iteration],
                    )
                    return True

            raise ValueError(f"No candidate with index {index} in iteration {iteration}")

        return self._with_connection(
            write=True,
            operation="update-candidate-model",
            callback=_update,
        )

    def set_generator_status(
        self, iteration, client_id, status, n_candidates=None, error=None
    ):
        """Record generator progress separately from evaluator results."""

        def _set(conn):
            conn.execute(
                "INSERT OR REPLACE INTO runtime_generator_status "
                "(iteration, client_id, status, n_candidates, error, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                [
                    iteration,
                    self._client_key(client_id),
                    status,
                    n_candidates,
                    error,
                    datetime.now().isoformat(),
                ],
            )

        self._with_connection(write=True, operation="set-generator-status", callback=_set)


def apply_client_profile(cfg, profile_name):
    """
    Apply a named client profile's overrides to the config.

    Profiles are defined in the YAML config under a `clients:` section.
    Each profile can override LLM settings and add extra guardrails.

    Special fields:
    - system_prompt_suffix: appended to cfg.llm.system_prompt
    - extra_guardrails: appended to cfg.llm.guardrails list
    - All other fields: direct override on cfg.llm
    """
    clients = _mapping_get(cfg, "clients")
    if not clients:
        raise ValueError("No 'clients' section in config")

    profile = _mapping_get(clients, profile_name)
    if profile is None:
        available = list(clients.keys()) if isinstance(clients, dict) else list(vars(clients).keys())
        raise ValueError(
            f"Client profile '{profile_name}' not found. "
            f"Available profiles: {available}"
        )

    llm_overrides = _mapping_get(profile, "llm")
    if llm_overrides:
        # Append suffix to system prompt
        suffix = _mapping_get(llm_overrides, "system_prompt_suffix")
        if suffix:
            cfg.llm.system_prompt = cfg.llm.system_prompt.rstrip() + "\n\n" + suffix

        # Append extra guardrails
        extra = _mapping_get(llm_overrides, "extra_guardrails")
        if extra:
            if not hasattr(cfg.llm, "guardrails") or cfg.llm.guardrails is None:
                cfg.llm.guardrails = []
            cfg.llm.guardrails.extend(extra)

        # Override other LLM fields directly
        skip = {"system_prompt_suffix", "extra_guardrails"}
        llm_items = llm_overrides.items() if isinstance(llm_overrides, dict) else vars(llm_overrides).items()
        for key, val in llm_items:
            if key not in skip:
                setattr(cfg.llm, key, val)

    console.print(f"[dim]Applied client profile '[cyan]{profile_name}[/]'[/]")


def _mapping_get(obj, key, default=None):
    """Read a key from either a mapping or an attribute container."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)
