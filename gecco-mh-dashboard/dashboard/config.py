from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class DashboardConfig:
    default_task: str = "two_step_factors"
    default_refresh_seconds: int = 10
    default_max_history_points: int = 1000


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_results_dir(task_name: str) -> Path:
    return project_root() / "results" / task_name


def available_tasks() -> List[str]:
    """Return task names that have a shared_registry.json in results/."""
    results = project_root() / "results"
    if not results.is_dir():
        return []
    return sorted(
        d.name for d in results.iterdir()
        if d.is_dir() and (d / "shared_registry.json").exists()
    )
