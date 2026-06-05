"""Run context and path ownership helpers for GeCCo.

The context owns the resolved project paths, results layout, and temporary
directory lifecycle so that ``GeCCoModelSearch`` does not need to manage these
concerns directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from gecco.tempdirs import configure_temp_dirs


@dataclass(slots=True)
class RunContext:
    """Resolved runtime paths for a single GeCCo search run.

    Attributes:
        project_root: Repository root used for relative artefacts.
        results_dir: Resolved run output directory.
        temp_root: Parent directory for per-run temporary files.
        tempdir: Managed temporary directory for the current run.
    """

    project_root: Path
    results_dir: Path
    temp_root: Path
    tempdir: TemporaryDirectory
    diagnostics_path: Path | None = None
    fit_type: str = "group"
    task_name: str = ""
    client_id: Any = None
    _artifact_subdirs: tuple[str, ...] = field(default=("models", "bics", "feedback"), init=False)

    @classmethod
    def from_cfg(
        cls,
        cfg: Any,
        *,
        project_root: str | Path | None = None,
        client_id: Any = None,
    ) -> "RunContext":
        """Resolve and create the runtime path layout from a config object.

        Args:
            cfg: Loaded runtime config.
            project_root: Optional repository root override.
            client_id: Optional distributed client identifier.

        Returns:
            A fully initialised run context.

        Raises:
            ValueError: If required config fields are missing or invalid.
        """

        if not hasattr(cfg, "task") or not getattr(cfg.task, "name", ""):
            raise ValueError("cfg.task.name is required to resolve run paths")
        if not hasattr(cfg, "evaluation"):
            raise ValueError("cfg.evaluation is required to resolve run paths")

        fit_type = getattr(cfg.evaluation, "fit_type", "group")
        if fit_type not in {"group", "individual"}:
            raise ValueError("cfg.evaluation.fit_type must be 'group' or 'individual'")

        resolved_root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[1]
        task_name = str(cfg.task.name)
        results_dir = resolved_root / "results" / task_name
        if fit_type == "individual":
            results_dir = resolved_root / "results" / f"{task_name}_individual"

        temp_root = resolved_root / "tmp"
        configure_temp_dirs(resolved_root)
        temp_root.mkdir(parents=True, exist_ok=True)
        tempdir = TemporaryDirectory(dir=temp_root)

        context = cls(
            project_root=resolved_root,
            results_dir=results_dir,
            temp_root=temp_root,
            tempdir=tempdir,
            fit_type=fit_type,
            task_name=task_name,
            client_id=client_id,
        )
        context.ensure_layout()
        context.diagnostics_path = context.default_diagnostics_path()
        return context

    def ensure_layout(self) -> None:
        """Create the standard results directory layout."""

        self.results_dir.mkdir(parents=True, exist_ok=True)
        for subdir in self._artifact_subdirs:
            (self.results_dir / subdir).mkdir(parents=True, exist_ok=True)

    @property
    def is_individual(self) -> bool:
        """Return whether this run uses individual fitting."""

        return self.fit_type == "individual"

    def _participant_suffix(self, participant: str | None = None) -> str:
        """Return the participant filename suffix for individual runs."""

        if not self.is_individual or participant is None:
            return ""
        return f"_participant{participant}"

    def candidate_model_path(
        self,
        *,
        iteration: int,
        run_idx: int,
        tag: str = "",
        participant: str | None = None,
    ) -> Path:
        """Return the canonical candidate model file path."""

        suffix = self._participant_suffix(participant)
        return self.results_dir / "models" / f"iter{iteration}{tag}_run{run_idx}{suffix}.txt"

    def iteration_results_path(
        self,
        *,
        iteration: int,
        run_idx: int,
        tag: str = "",
        participant: str | None = None,
    ) -> Path:
        """Return the canonical iteration-results file path."""

        suffix = self._participant_suffix(participant)
        return self.results_dir / "bics" / f"iter{iteration}{tag}_run{run_idx}{suffix}.json"

    def feedback_path(
        self,
        *,
        iteration: int,
        run_idx: int,
        tag: str = "",
        participant: str | None = None,
    ) -> Path:
        """Return the canonical feedback file path."""

        suffix = self._participant_suffix(participant)
        return self.results_dir / "feedback" / f"iter{iteration}{tag}_run{run_idx}{suffix}.txt"

    def default_diagnostics_path(self) -> Path:
        """Return the canonical diagnostics DuckDB file path."""

        shard = f"_{self.client_id}" if self.client_id else ""
        return self.results_dir / f"diagnostics{shard}.duckdb"

    @property
    def tempdir_path(self) -> Path:
        """Return the managed temporary directory path."""

        return Path(self.tempdir.name)

    def close(self) -> None:
        """Clean up the managed temporary directory."""

        self.tempdir.cleanup()

    def __enter__(self) -> "RunContext":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
