"""
Distributed coordination for parallel GeCCo search clients.

Uses a shared JSON registry file on the filesystem for coordination.
Advisory file locking (fcntl.flock) ensures safe concurrent access.
"""

import fcntl
import json
import os
import tempfile
from pathlib import Path
from datetime import datetime

from rich.console import Console

console = Console()


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
        except Exception:
            # Clean up temp file on failure
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def update(self, client_id, iteration, results, best_model=None,
               best_metric=None, param_names=None, tried_param_sets=None,
               status="running"):
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
                    if r.get("recovery_n_successful") is not None:
                        entry["recovery_n_successful"] = r["recovery_n_successful"]
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
                    serializable_results.append(entry)

                # Replace existing entry for same (client_id, iteration), or append
                new_entry = {
                    "client_id": client_id,
                    "iteration": iteration,
                    "results": serializable_results,
                }
                replaced = False
                for idx, existing_entry in enumerate(data["iteration_history"]):
                    if (existing_entry.get("client_id") == client_id
                            and existing_entry.get("iteration") == iteration):
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
                    if current_best is None or best_metric < current_best["metric_value"]:
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
                    data["client_entries"][str(client_id)]["updated_at"] = datetime.now().isoformat()

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
                    "function_name": baseline_result.get("function_name", "baseline_model"),
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
