"""Distributed coordination helpers for GeCCo search clients."""

from __future__ import annotations

from typing import Any


class DistributedCoordinator:
    """Coordinate registry synchronisation and runtime state updates."""

    def sync_from_registry(self, *, search: Any) -> None:
        """Merge shared registry state into the local search object."""

        if search.shared_registry is None:
            return

        data = search.shared_registry.read()

        global_best = data.get("global_best")
        if global_best and global_best["metric_value"] < search.best_metric:
            search.best_metric = global_best["metric_value"]
            search.best_model = global_best["model_code"]
            search.best_params = global_best["param_names"]

        existing = {tuple(s) for s in search.tried_param_sets}
        for ps in data.get("tried_param_sets", []):
            key = tuple(ps)
            if key not in existing:
                search.tried_param_sets.append(ps)
                existing.add(key)

        all_history = data.get("iteration_history", [])
        new_entries = all_history[search._merged_history_count :]
        for entry in new_entries:
            if entry.get("client_id") == search.client_id:
                continue
            search.feedback.history.append(
                {
                    "iteration": entry["iteration"],
                    "results": entry["results"],
                    "client_id": entry.get("client_id"),
                }
            )
        search._merged_history_count = len(all_history)

    def set_activity(self, *, search: Any, activity: str) -> None:
        """Publish the current activity if distributed coordination is enabled."""

        if search.shared_registry is None:
            return
        search.shared_registry.set_activity(search.client_id, activity)

    def update_registry(
        self,
        *,
        search: Any,
        iteration: int,
        results: list[dict[str, Any]],
        status: str = "running",
        had_runnable_model: bool | None = None,
    ) -> None:
        """Publish the current iteration state to the shared registry."""

        if search.shared_registry is None:
            return
        search.shared_registry.update(
            client_id=search.client_id,
            iteration=iteration,
            results=results,
            best_model=search.best_model,
            best_metric=search.best_metric,
            param_names=search.best_params,
            tried_param_sets=search.tried_param_sets,
            status=status,
            had_runnable_model=had_runnable_model,
        )

    def start_iteration(self, *, search: Any, cmg_cfg: Any | None) -> int:
        """Return the iteration index to resume from."""

        if (
            search.shared_registry is not None
            and search.client_id is not None
            and cmg_cfg is not None
            and search._cmg_is_generator(cmg_cfg)
        ):
            max_existing = search.shared_registry.get_max_generator_iteration(search.client_id)
        elif search.shared_registry is not None and search.client_id is not None:
            max_existing = search.shared_registry.get_max_iteration_for_client(search.client_id)
        elif search.shared_registry is not None:
            max_existing = search.shared_registry.get_max_iteration()
        else:
            max_existing = -1

        return max_existing + 1 if max_existing >= 0 else 0
