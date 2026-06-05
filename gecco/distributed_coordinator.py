"""Distributed coordination helpers for GeCCo search clients."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class DistributedSyncResult:
    """Updated local state after synchronising from the shared registry."""

    best_metric: float
    best_model: str | None
    best_params: list[Any]
    tried_param_sets: list[list[Any]]
    feedback_history: list[dict[str, Any]]
    merged_history_count: int


class DistributedCoordinator:
    """Coordinate registry synchronisation and runtime state updates."""

    def sync_from_registry(
        self,
        *,
        shared_registry: Any,
        best_metric: float,
        best_model: str | None,
        best_params: list[Any],
        tried_param_sets: list[list[Any]],
        feedback_history: list[dict[str, Any]],
        merged_history_count: int,
        client_id: Any,
    ) -> DistributedSyncResult:
        """Merge shared registry state into the local runtime state."""

        if shared_registry is None:
            return DistributedSyncResult(
                best_metric=best_metric,
                best_model=best_model,
                best_params=best_params,
                tried_param_sets=tried_param_sets,
                feedback_history=feedback_history,
                merged_history_count=merged_history_count,
            )

        data = shared_registry.read()

        global_best = data.get("global_best")
        if global_best and global_best["metric_value"] < best_metric:
            best_metric = global_best["metric_value"]
            best_model = global_best["model_code"]
            best_params = global_best["param_names"]

        existing = {tuple(s) for s in tried_param_sets}
        for ps in data.get("tried_param_sets", []):
            key = tuple(ps)
            if key not in existing:
                tried_param_sets.append(ps)
                existing.add(key)

        all_history = data.get("iteration_history", [])
        new_entries = all_history[merged_history_count:]
        for entry in new_entries:
            if entry.get("client_id") == client_id:
                continue
            feedback_history.append(
                {
                    "iteration": entry["iteration"],
                    "results": entry["results"],
                    "client_id": entry.get("client_id"),
                }
            )
        merged_history_count = len(all_history)

        return DistributedSyncResult(
            best_metric=best_metric,
            best_model=best_model,
            best_params=best_params,
            tried_param_sets=tried_param_sets,
            feedback_history=feedback_history,
            merged_history_count=merged_history_count,
        )

    def set_activity(self, *, shared_registry: Any, client_id: Any, activity: str) -> None:
        """Publish the current activity if distributed coordination is enabled."""

        if shared_registry is None:
            return
        shared_registry.set_activity(client_id, activity)

    def update_registry(
        self,
        *,
        shared_registry: Any,
        client_id: Any,
        iteration: int,
        results: list[dict[str, Any]],
        best_model: str | None,
        best_metric: float,
        best_params: list[Any],
        tried_param_sets: list[list[Any]],
        status: str = "running",
        had_runnable_model: bool | None = None,
    ) -> None:
        """Publish the current iteration state to the shared registry."""

        if shared_registry is None:
            return
        shared_registry.update(
            client_id=client_id,
            iteration=iteration,
            results=results,
            best_model=best_model,
            best_metric=best_metric,
            param_names=best_params,
            tried_param_sets=tried_param_sets,
            status=status,
            had_runnable_model=had_runnable_model,
        )

    def start_iteration(
        self,
        *,
        shared_registry: Any,
        client_id: Any,
        cmg_cfg: Any | None,
        is_generator: bool,
    ) -> int:
        """Return the iteration index to resume from."""

        if (
            shared_registry is not None
            and client_id is not None
            and cmg_cfg is not None
            and is_generator
        ):
            max_existing = shared_registry.get_max_generator_iteration(client_id)
        elif shared_registry is not None and client_id is not None:
            max_existing = shared_registry.get_max_iteration_for_client(client_id)
        elif shared_registry is not None:
            max_existing = shared_registry.get_max_iteration()
        else:
            max_existing = -1

        return max_existing + 1 if max_existing >= 0 else 0
