"""Characterization tests for Phase 0 cleanup work."""

import json
from pathlib import Path

from config.schema import load_config
from gecco.coordination import SharedRegistry
from gecco.run_gecco import GeCCoModelSearch
from scripts.reset_distributed import (
    ARTIFACT_DIRS,
    BASELINE_FILES,
    REGISTRY_FILES,
    get_results_dir,
)


FIXTURES_DIR = Path(__file__).parent / "fixtures" / "phase0"


def test_load_config_uses_recursive_namespaces_without_validation():
    """Freeze the current namespace-based config loading behaviour."""
    cfg = load_config(str(FIXTURES_DIR / "legacy_load_config.yaml"))

    assert cfg.task.name == "freeze_task"
    assert cfg.judge.orchestrated == "yes please"
    assert cfg.judge.barrier.orchestrator_wait_seconds == "120"
    assert cfg.custom_section.nested.extra_flag is True
    assert cfg.custom_section.nested.labels == ["alpha", "beta"]


def test_phase0_inventory_matches_current_legacy_entrypoints():
    """Freeze the legacy CLI script names and reset artefact layout."""
    inventory = json.loads((FIXTURES_DIR / "legacy_cli_inventory.json").read_text())

    for relative_path in inventory["legacy_cli_scripts"]:
        assert (Path(__file__).resolve().parents[1] / relative_path).exists()

    assert ARTIFACT_DIRS == inventory["reset_artifact_dirs"]
    assert REGISTRY_FILES == inventory["registry_files"]
    assert BASELINE_FILES == inventory["baseline_files"]


def test_get_results_dir_keeps_individual_suffix_behaviour():
    """Freeze the distributed results directory naming convention."""

    class _Cfg:
        class task:
            name = "phase0_task"

        class evaluation:
            fit_type = "individual"

    assert get_results_dir(_Cfg()) == Path("results") / "phase0_task_individual"


def test_shared_registry_normalises_string_feedback_to_default_persona(tmp_path):
    """Freeze the default-persona feedback storage format."""
    registry = SharedRegistry(str(tmp_path / "registry.json"))
    registry.set_judge_feedback(
        iteration=2,
        synthesized_feedback="Keep exploring simpler variants.",
        verdict_payload={"best_bic": 99.1},
    )

    result = registry.get_judge_feedback(iteration=2)
    assert result is not None
    assert result["synthesized_feedback"] == {
        "default": "Keep exploring simpler variants."
    }


def test_shared_registry_returns_persona_specific_feedback_with_fallback(tmp_path):
    """Freeze persona-specific feedback lookup semantics."""
    registry = SharedRegistry(str(tmp_path / "registry.json"))
    registry.set_judge_feedback(
        iteration=3,
        synthesized_feedback={
            "explore": "Increase diversity.",
            "default": "Hold course.",
        },
        verdict_payload={},
    )

    explore_feedback = registry.get_judge_feedback_for_persona(3, "explore")
    fallback_feedback = registry.get_judge_feedback_for_persona(3, "missing")

    assert explore_feedback is not None
    assert explore_feedback["synthesized_feedback"] == "Increase diversity."
    assert fallback_feedback is not None
    assert fallback_feedback["synthesized_feedback"] == "Hold course."


def test_file_tag_uses_client_suffix_only_when_distributed():
    """Freeze current artefact filename tagging behaviour."""
    distributed_search = GeCCoModelSearch.__new__(GeCCoModelSearch)
    distributed_search.client_id = 7

    single_search = GeCCoModelSearch.__new__(GeCCoModelSearch)
    single_search.client_id = None

    assert distributed_search._file_tag() == "_client7"
    assert single_search._file_tag() == ""
