"""Phase 6 non-CMG orchestration tests for ``run_n_shots``."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from gecco.artifacts import ArtifactStore
from gecco.candidate_evaluation import BestModelState, CandidateEvaluator
from gecco.candidate_generation import CandidateGenerator
from gecco.diagnostic_store.store import DiagnosticStore
from gecco.run_context import RunContext
from gecco.run_gecco import GeCCoModelSearch


def test_run_n_shots_non_cmg_uses_extracted_services_and_retries_once(tmp_path: Path):
    """Non-CMG iterations should flow through the generator and evaluator services."""

    cfg = SimpleNamespace(
        loop=SimpleNamespace(max_iterations=1),
        evaluation=SimpleNamespace(fit_type="group", metric="bic"),
        llm=SimpleNamespace(models_per_iteration=1),
        judge=None,
        validation=SimpleNamespace(max_syntax_retries=1),
        task=SimpleNamespace(name="phase6_non_cmg"),
        data=SimpleNamespace(input_columns=[]),
        clients={},
    )
    run_context = RunContext.from_cfg(cfg, project_root=tmp_path)
    diagnostic_store = DiagnosticStore(tmp_path / "phase6_non_cmg.duckdb")
    artifact_store = ArtifactStore(run_context, diagnostic_store)
    generator = CandidateGenerator(artifact_store)
    evaluator = CandidateEvaluator(artifact_store)

    generator.generate_models = MagicMock(
        return_value=(
            "def model_a():\n    return 0",
            [
                {
                    "name": "model_a",
                    "code": "def model_a():\n    return 0",
                    "parameters": [],
                }
            ],
        )
    )
    generator.generate_models_naive = MagicMock(
        side_effect=AssertionError("naive generation should not be used")
    )
    fit_call_count = {"count": 0}

    def _fit_candidate_model(*args, **kwargs):
        fit_call_count["count"] += 1
        if fit_call_count["count"] == 1:
            return (
                {
                    "function_name": "model_a",
                    "metric_name": "VALIDATION_ERROR",
                    "metric_value": float("inf"),
                    "param_names": [],
                    "code": "def model_a():\n    return 0",
                    "error_type": "syntax",
                    "error_message": "bad syntax",
                    "error_details": {},
                },
                False,
            )
        return (
            {
                "function_name": "model_a",
                "metric_name": "BIC",
                "metric_value": 1.0,
                "param_names": [],
                "code": "def model_a():\n    return 0",
                "parameter_values": [],
            },
            False,
        )

    evaluator.fit_candidate_model = MagicMock(side_effect=_fit_candidate_model)
    evaluator.finalize_iteration_results = MagicMock(
        wraps=evaluator.finalize_iteration_results
    )

    search = GeCCoModelSearch.__new__(GeCCoModelSearch)
    search.cfg = cfg
    search.df = SimpleNamespace()
    search.model = object()
    search.tokenizer = object()
    search.prompt_builder = SimpleNamespace(
        build_input_prompt=MagicMock(return_value="prompt text")
    )
    search.client_id = None
    search.shared_registry = None
    search.best_state = BestModelState()
    search.best_model = None
    search.best_metric = float("inf")
    search.best_params = []
    search.best_param_names = []
    search.best_param_values = None
    search.best_iter = -1
    search.best_id_results = None
    search.tried_param_sets = []
    search.feedback = SimpleNamespace(history=[], record_iteration=MagicMock())
    search.tool_judge = None
    search._merged_history_count = 0
    search.recovery_checker = None
    search.id_eval_data = None
    search.ppc_enabled = False
    search._ppc_simulator = None
    search.ppc_n_sims = 100
    search.block_residuals_enabled = False
    search.block_residuals_n_blocks = 10
    search.run_context = run_context
    search.results_dir = run_context.results_dir
    search.artifact_store = artifact_store
    search.candidate_generator = generator
    search.candidate_evaluator = evaluator
    search.generate = MagicMock(side_effect=AssertionError("legacy generation path used"))
    search._cmg_config = MagicMock(return_value=None)
    search._sync_from_registry = MagicMock()
    search._file_tag = MagicMock(return_value="")
    search._set_activity = MagicMock()
    search._update_registry = MagicMock()
    search.distributed_coordinator = MagicMock(start_iteration=MagicMock(return_value=0))
    search.run_n_shots = GeCCoModelSearch.run_n_shots.__get__(search, GeCCoModelSearch)

    for legacy_name in (
        "generate_models",
        "generate_models_naive",
        "_fit_candidate_model",
        "_finalize_iteration_results",
    ):
        setattr(
            search,
            legacy_name,
            MagicMock(side_effect=AssertionError(f"{legacy_name} should not be used")),
        )

    search.run_n_shots(0, None)

    assert generator.generate_models.call_count == 2
    assert evaluator.fit_candidate_model.call_count == 2
    assert evaluator.finalize_iteration_results.call_count == 1
    assert search.best_metric == 1.0
    assert search.best_model == "def model_a():\n    return 0"
    assert search.feedback.record_iteration.call_count == 1
    assert diagnostic_store.fetchone("SELECT COUNT(*) AS n FROM iterations") == {"n": 1}
    assert diagnostic_store.fetchone("SELECT COUNT(*) AS n FROM models") == {"n": 1}
    assert not (run_context.results_dir / "bics" / "iter0_run0.json").exists()

    diagnostic_store.close()
    run_context.close()
