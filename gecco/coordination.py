"""
Distributed coordination for parallel GeCCo search clients.

Uses a shared JSON registry file on the filesystem for coordination.
Advisory file locking (fcntl.flock) ensures safe concurrent access.
"""

import fcntl
import json
import os
import tempfile
import time
from pathlib import Path
from datetime import datetime
from typing import Optional

from gecco.sentry_init import capture_coordination_error
from gecco.utils import TimestampedConsole
from rich.console import Console

console = TimestampedConsole()


class SharedRegistry:
    """
    Filesystem-based coordination registry for distributed GeCCo clients.

    Each client reads/writes a shared JSON file. File locking prevents
    corruption from concurrent access.

    Registry format:
    {
        "global_best": {
            "metric_value": float,
            "model_code": str,
            "param_names": [...],
            "client_id": int,
            "iteration": int
        },
        "tried_param_sets": [[...], ...],
        "client_entries": {
            "0": {"last_iteration": int, "best_metric": float, "status": str},
            ...
        },
        "iteration_history": [
            {
                "client_id": int,
                "iteration": int,
                "results": [
                    {"function_name": str, "metric_value": float,
                     "param_names": [...], "code": str}
                ]
            },
            ...
        ]
    }
    """

    def __init__(self, registry_path):
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize empty registry if it doesn't exist
        if not self.registry_path.exists():
            self._atomic_write(self._empty_registry())

    @staticmethod
    def _empty_registry():
        return {
            "global_best": None,
            "baseline": None,
            "tried_param_sets": [],
            "client_entries": {},
            "iteration_history": [],
        }

    def read(self):
        """Read the current registry state with shared (read) lock."""
        try:
            with open(self.registry_path, "r") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                try:
                    data = json.load(f)
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            return data
        except (json.JSONDecodeError, FileNotFoundError):
            return self._empty_registry()

    def _atomic_write(self, data):
        """Write data atomically using a temp file + os.replace."""
        fd, tmp_path = tempfile.mkstemp(
            dir=str(self.registry_path.parent),
            suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp_path, str(self.registry_path))
        except Exception as e:
            capture_coordination_error(error=e, operation="atomic-write")
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

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
    ):
        """
        Atomically merge this client's iteration results into the registry.

        Uses an exclusive lock around read-modify-write.
        """
        with open(self.registry_path, "a+") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.seek(0)
                content = f.read()
                if content.strip():
                    data = json.loads(content)
                else:
                    data = self._empty_registry()

                # Update client entry
                data["client_entries"][str(client_id)] = {
                    "last_iteration": iteration,
                    "best_metric": best_metric,
                    "status": status,
                    "updated_at": datetime.now().isoformat(),
                }

                # Append iteration history (strip non-serializable fields)
                serializable_results = []
                for r in results:
                    entry = {
                        "function_name": r.get("function_name", ""),
                        "metric_name": r.get("metric_name", "BIC"),
                        "metric_value": r.get("metric_value", float("inf")),
                        "param_names": r.get("param_names", []),
                        "code": r.get("code", ""),
                    }
                    # Include error message for failed models
                    if r.get("error"):
                        entry["error"] = r["error"]
                    # Include recovery stats for failed recovery checks
                    if r.get("recovery_r") is not None:
                        entry["recovery_r"] = r["recovery_r"]
                    if r.get("recovery_per_param"):
                        entry["recovery_per_param"] = r["recovery_per_param"]
                    # Include per-participant eval metrics for fit quality analysis
                    eval_metrics = r.get("eval_metrics")
                    if eval_metrics:
                        entry["eval_metrics"] = eval_metrics
                    # Include per-participant trial counts for chance-level detection
                    participant_n_trials = r.get("participant_n_trials")
                    if participant_n_trials:
                        entry["participant_n_trials"] = participant_n_trials
                    # Include individual differences R² if available
                    id_res = r.get("individual_differences")
                    if id_res and isinstance(id_res, dict):
                        entry["individual_differences"] = {
                            "mean_r2": id_res.get("mean_r2"),
                            "max_r2": id_res.get("max_r2"),
                            "best_param": id_res.get("best_param"),
                            "per_param_r2": id_res.get("per_param_r2"),
                            "summary_text": id_res.get("summary_text", ""),
                        }
                        # Also store at top level for monitor dashboard
                        entry["mean_r2"] = id_res.get("mean_r2")
                        entry["max_r2"] = id_res.get("max_r2")
                        entry["best_param"] = id_res.get("best_param")
                        entry["per_param_r2"] = id_res.get("per_param_r2")
                    # Include validation metrics for test evaluation (Step 9)
                    if r.get("val_metric_value") is not None:
                        entry["val_metric_value"] = r["val_metric_value"]
                        entry["val_mean_nll"] = r.get("val_mean_nll")
                        entry["val_eval_metrics"] = r.get("val_eval_metrics", [])
                        entry["val_per_participant_nll"] = r.get(
                            "val_per_participant_nll", []
                        )
                    val_id = r.get("val_individual_differences")
                    if val_id and isinstance(val_id, dict):
                        entry["val_individual_differences"] = {
                            "mean_r2": val_id.get("mean_r2"),
                            "max_r2": val_id.get("max_r2"),
                            "best_param": val_id.get("best_param"),
                            "per_param_r2": val_id.get("per_param_r2"),
                        }
                    serializable_results.append(entry)

                # Replace existing entry for same (client_id, iteration), or append
                new_entry = {
                    "client_id": client_id,
                    "iteration": iteration,
                    "results": serializable_results,
                }
                replaced = False
                for idx, existing_entry in enumerate(data["iteration_history"]):
                    if (
                        existing_entry.get("client_id") == client_id
                        and existing_entry.get("iteration") == iteration
                    ):
                        data["iteration_history"][idx] = new_entry
                        replaced = True
                        break
                if not replaced:
                    data["iteration_history"].append(new_entry)

                # Merge tried param sets (deduplicate)
                if tried_param_sets:
                    existing = {tuple(s) for s in data["tried_param_sets"]}
                    for ps in tried_param_sets:
                        key = tuple(ps)
                        if key not in existing:
                            data["tried_param_sets"].append(ps)
                            existing.add(key)

                # Update global best if this client has a better model
                if best_metric is not None and best_model is not None:
                    current_best = data["global_best"]
                    if (
                        current_best is None
                        or best_metric < current_best["metric_value"]
                    ):
                        data["global_best"] = {
                            "metric_value": best_metric,
                            "model_code": best_model,
                            "param_names": param_names or [],
                            "client_id": client_id,
                            "iteration": iteration,
                        }

                self._atomic_write(data)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def get_max_iteration(self):
        """Return the highest iteration number across all clients, or -1 if none."""
        data = self.read()
        max_iter = -1
        for entry in data.get("iteration_history", []):
            it = entry.get("iteration")
            if it is not None and it > max_iter:
                max_iter = it
        return max_iter

    def set_activity(self, client_id, activity):
        """Update a client's current activity without pushing results."""
        with open(self.registry_path, "a+") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.seek(0)
                content = f.read()
                if content.strip():
                    data = json.loads(content)
                else:
                    data = self._empty_registry()

                entry = data["client_entries"].get(str(client_id), {})
                entry["activity"] = activity
                entry["updated_at"] = datetime.now().isoformat()
                data["client_entries"][str(client_id)] = entry

                self._atomic_write(data)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def mark_complete(self, client_id):
        """Mark a client as complete in the registry."""
        with open(self.registry_path, "a+") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.seek(0)
                content = f.read()
                if content.strip():
                    data = json.loads(content)
                else:
                    data = self._empty_registry()

                if str(client_id) in data["client_entries"]:
                    data["client_entries"][str(client_id)]["status"] = "complete"
                    data["client_entries"][str(client_id)]["updated_at"] = (
                        datetime.now().isoformat()
                    )

                self._atomic_write(data)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def set_baseline(self, baseline_result):
        """Write baseline result to the registry under the 'baseline' key."""
        with open(self.registry_path, "a+") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.seek(0)
                content = f.read()
                if content.strip():
                    data = json.loads(content)
                else:
                    data = self._empty_registry()

                # Store a serializable subset (no numpy arrays)
                data["baseline"] = {
                    "function_name": baseline_result.get(
                        "function_name", "baseline_model"
                    ),
                    "metric_name": baseline_result.get("metric_name", "BIC"),
                    "metric_value": baseline_result.get("metric_value"),
                    "param_names": baseline_result.get("param_names", []),
                    "eval_metrics": baseline_result.get("eval_metrics", []),
                }
                # Include individual differences if available
                id_res = baseline_result.get("individual_differences")
                if id_res and isinstance(id_res, dict):
                    data["baseline"]["mean_r2"] = id_res.get("mean_r2")
                    data["baseline"]["max_r2"] = id_res.get("max_r2")
                    data["baseline"]["best_param"] = id_res.get("best_param")
                    data["baseline"]["per_param_r2"] = id_res.get("per_param_r2")

                self._atomic_write(data)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def count_clients_at_iteration(self, iteration: int) -> int:
        """
        Count the number of distinct clients that have written results for an iteration.

        Returns the count of unique client_ids in iteration_history with the given iteration.
        """
        data = self.read()
        seen_clients = set()
        for entry in data.get("iteration_history", []):
            if entry.get("iteration") == iteration:
                client_id = entry.get("client_id")
                if client_id is not None:
                    seen_clients.add(client_id)
        return len(seen_clients)

    def wait_for_iteration(
        self,
        iteration: int,
        n_expected: int,
        timeout_seconds: float,
        poll_seconds: float = 2.0,
    ) -> int:
        """
        Poll until at least n_expected clients have written iteration results, or timeout.

        Polls count_clients_at_iteration every poll_seconds until:
        - count >= n_expected (returns count immediately), or
        - timeout_seconds elapses (returns count at that time)

        Returns the count of clients who contributed to the iteration.
        """
        start_time = time.time()
        while True:
            count = self.count_clients_at_iteration(iteration)
            if count >= n_expected:
                console.print(
                    f"[green]Iteration {iteration} complete: {count}/{n_expected} clients "
                    f"in {time.time() - start_time:.1f}s[/]"
                )
                return count

            elapsed = time.time() - start_time
            if elapsed >= timeout_seconds:
                console.print(
                    f"[yellow]Iteration {iteration} timeout: got {count}/{n_expected} clients "
                    f"after {elapsed:.1f}s, proceeding with available results[/]"
                )
                return count

            time.sleep(poll_seconds)

    def set_judge_feedback(
        self,
        iteration: int,
        synthesized_feedback: str | dict,  # R2: dict keyed by persona name
        verdict_payload: dict,
    ) -> None:
        """
        Store the shared judge verdict for an iteration.

        Parameters
        ----------
        iteration : int
            Iteration number
        synthesized_feedback : str or dict
            If str: single global feedback (backward compat)
            If dict: per-persona feedback {persona_name: feedback_str, ...}
        verdict_payload : dict
            Metadata about the verdict

        The verdict is stored in a new top-level 'judge_iterations' dict keyed by iteration.
        """
        with open(self.registry_path, "a+") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.seek(0)
                content = f.read()
                if content.strip():
                    data = json.loads(content)
                else:
                    data = self._empty_registry()

                # Initialize judge_iterations if not present
                if "judge_iterations" not in data:
                    data["judge_iterations"] = {}

                # R2: Normalize feedback to dict format
                if isinstance(synthesized_feedback, str):
                    feedback_dict = {"default": synthesized_feedback}
                else:
                    feedback_dict = synthesized_feedback

                # Store the verdict with timestamp
                data["judge_iterations"][str(iteration)] = {
                    "synthesized_feedback": feedback_dict,
                    "verdict": verdict_payload,
                    "timestamp": datetime.now().isoformat(),
                }

                self._atomic_write(data)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def get_judge_feedback(self, iteration: int) -> Optional[dict]:
        """
        Retrieve the stored judge verdict for an iteration (full dict with per-persona feedback).

        Returns the verdict dict (with 'synthesized_feedback' dict and 'verdict' keys)
        or None if not yet written.
        """
        data = self.read()
        judge_iterations = data.get("judge_iterations", {})
        return judge_iterations.get(str(iteration))

    def get_judge_feedback_for_persona(
        self, iteration: int, persona_name: str, fallback: str = "default"
    ) -> Optional[dict]:
        """
        R2: Retrieve persona-specific judge feedback for an iteration.

        Parameters
        ----------
        iteration : int
            Iteration number
        persona_name : str
            Name of the persona (e.g. 'exploit', 'explore', 'diverse')
        fallback : str
            Fallback key if persona not found; defaults to 'default'

        Returns the full verdict dict but with synthesized_feedback narrowed to the
        persona-specific text (or fallback if persona not found).
        """
        verdict_dict = self.get_judge_feedback(iteration)
        if verdict_dict is None:
            return None

        feedback_dict = verdict_dict.get("synthesized_feedback", {})
        # Ensure feedback is dict format (backward compat with old str-based storage)
        if isinstance(feedback_dict, str):
            feedback_dict = {"default": feedback_dict}

        # Get persona-specific feedback or fall back
        persona_feedback = feedback_dict.get(persona_name, feedback_dict.get(fallback, ""))

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
        """
        Client-side helper: poll until judge feedback is available for an iteration, or timeout.

        Polls get_judge_feedback every poll_seconds until:
        - feedback is available (returns the dict), or
        - timeout_seconds elapses (returns None)

        Used by clients to wait for the orchestrator's shared verdict.
        """
        start_time = time.time()
        while True:
            feedback = self.get_judge_feedback(iteration)
            if feedback is not None:
                elapsed = time.time() - start_time
                console.print(
                    f"[green]Received judge feedback for iteration {iteration} "
                    f"in {elapsed:.1f}s[/]"
                )
                return feedback

            elapsed = time.time() - start_time
            if elapsed >= timeout_seconds:
                console.print(
                    f"[yellow]Timeout waiting for judge feedback (iteration {iteration}) "
                    f"after {elapsed:.1f}s[/]"
                )
                return None

            time.sleep(poll_seconds)


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
    clients = getattr(cfg, "clients", None)
    if not clients:
        raise ValueError("No 'clients' section in config")

    profile = getattr(clients, profile_name, None)
    if not profile:
        available = [k for k in vars(clients).keys()] if clients else []
        raise ValueError(
            f"Client profile '{profile_name}' not found. "
            f"Available profiles: {available}"
        )

    llm_overrides = getattr(profile, "llm", None)
    if llm_overrides:
        # Append suffix to system prompt
        suffix = getattr(llm_overrides, "system_prompt_suffix", None)
        if suffix:
            cfg.llm.system_prompt = cfg.llm.system_prompt.rstrip() + "\n\n" + suffix

        # Append extra guardrails
        extra = getattr(llm_overrides, "extra_guardrails", None)
        if extra:
            if not hasattr(cfg.llm, "guardrails") or cfg.llm.guardrails is None:
                cfg.llm.guardrails = []
            cfg.llm.guardrails.extend(extra)

        # Override other LLM fields directly
        skip = {"system_prompt_suffix", "extra_guardrails"}
        for key, val in vars(llm_overrides).items():
            if key not in skip:
                setattr(cfg.llm, key, val)

    console.print(f"[dim]Applied client profile '[cyan]{profile_name}[/]'[/]")
