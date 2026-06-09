"""Tests for CMG runtime validation and evaluator index mapping."""

from unittest.mock import MagicMock, patch
from types import SimpleNamespace

import pytest

from gecco.artifacts import ArtifactStore
from gecco.candidate_evaluation import CandidateEvaluator
from gecco.candidate_generation import CandidateGenerator
from gecco.coordination import SharedRegistry
from gecco.run_context import RunContext
from gecco.run_gecco import GeCCoModelSearch


def _make_cmg_cfg(enabled=True, generator_client="generator", n_models=2):
    return SimpleNamespace(
        enabled=enabled,
        generator_client=generator_client,
        n_models=n_models,
    )


def _make_search(client_id=None, cfg=None):
    mock_cfg = cfg or SimpleNamespace(
        judge=SimpleNamespace(capabilities=[]),
        clients={},
    )
    search = MagicMock(spec=GeCCoModelSearch)
    search.client_id = client_id
    search.cfg = mock_cfg
    search.shared_registry = MagicMock()

    # Attach real (unmocked) methods under test
    search._cmg_config = GeCCoModelSearch._cmg_config.__get__(search, GeCCoModelSearch)
    search._cmg_is_generator = GeCCoModelSearch._cmg_is_generator.__get__(search, GeCCoModelSearch)
    search._cmg_evaluator_index = GeCCoModelSearch._cmg_evaluator_index.__get__(search, GeCCoModelSearch)
    return search


# --- _cmg_config ---

def test_cmg_config_disabled():
    cfg = SimpleNamespace(centralized_model_generation=None)
    search = _make_search(cfg=cfg)
    assert search._cmg_config() is None


def test_cmg_config_enabled():
    cmg = _make_cmg_cfg()
    cfg = SimpleNamespace(
        centralized_model_generation=cmg,
        judge=SimpleNamespace(capabilities=[]),
    )
    search = _make_search(cfg=cfg)
    result = search._cmg_config()
    assert result is not None
    assert result.enabled is True


# --- _cmg_is_generator ---

def test_cmg_is_generator_matches():
    cmg = _make_cmg_cfg(generator_client="generator")
    search = _make_search(client_id="generator")
    assert search._cmg_is_generator(cmg) is True


def test_cmg_is_generator_mismatch():
    cmg = _make_cmg_cfg(generator_client="generator")
    search = _make_search(client_id="evaluator_1")
    assert search._cmg_is_generator(cmg) is False


def test_cmg_generator_profile_lookup_supports_dict_backed_clients(tmp_path):
    """CMG generator lookup should honour dict-backed client profiles."""
    artifact_store = MagicMock()
    artifact_store.write_candidate_artifacts.return_value = tmp_path / "candidate.py"
    generator = CandidateGenerator(artifact_store)
    prompt_builder = MagicMock()
    prompt_builder.build_naive_prompt.return_value = "naive prompt"
    prompt_builder.build_input_prompt.return_value = "translated prompt"

    generate_text = MagicMock(return_value="psychological hypothesis")

    cfg = SimpleNamespace(
        llm=SimpleNamespace(provider="mock-provider"),
        clients={
            "generator": {
                "naive_ideation": {
                    "enabled": True,
                    "persona": "dict persona",
                    "translation_preamble": "dict translation",
                }
            }
        },
    )

    with patch("gecco.candidate_generation.get_provider_spec") as mock_provider_spec:
        mock_provider_spec.return_value = SimpleNamespace(supports_system_prompt=True)
        with patch.object(
            CandidateGenerator,
            "generate_models",
            autospec=True,
            return_value=("code", [{"name": "cognitive_model1"}]),
        ) as mock_generate_models:
            result = generator.generate_non_cmg_iteration(
                iteration=0,
                run_idx=0,
                feedback="",
                n_models=1,
                cfg=cfg,
                tag="",
                prompt_builder=prompt_builder,
                generate_text=generate_text,
                model=MagicMock(),
                tokenizer=MagicMock(),
                client_id="generator",
            )

    assert generate_text.call_args.kwargs["system_prompt"] == "dict persona"
    prompt_builder.build_input_prompt.assert_called_once_with(
        feedback_text="",
        naive_idea="psychological hypothesis",
        translation_preamble="dict translation",
        n_models=1,
        force_include_feedback=False,
    )
    assert mock_generate_models.call_args.kwargs["prompt"] == "translated prompt"
    assert result.code_text == "code"





# --- _cmg_evaluator_index ---

def test_evaluator_index_zero():
    cmg = _make_cmg_cfg(n_models=2)
    search = _make_search(client_id=0)
    assert search._cmg_evaluator_index(cmg) == 0


def test_evaluator_index_one():
    cmg = _make_cmg_cfg(n_models=2)
    search = _make_search(client_id=1)
    assert search._cmg_evaluator_index(cmg) == 1


def test_evaluator_index_out_of_range():
    cmg = _make_cmg_cfg(n_models=2)
    search = _make_search(client_id=2)
    assert search._cmg_evaluator_index(cmg) is None


def test_evaluator_index_negative():
    cmg = _make_cmg_cfg(n_models=2)
    search = _make_search(client_id=-1)
    assert search._cmg_evaluator_index(cmg) is None


def test_evaluator_index_non_numeric():
    cmg = _make_cmg_cfg(n_models=2)
    search = _make_search(client_id="generator")
    assert search._cmg_evaluator_index(cmg) is None


def test_evaluator_index_none_n_models():
    cmg = _make_cmg_cfg(n_models=None)
    search = _make_search(client_id=0)
    assert search._cmg_evaluator_index(cmg) is None


# --- _validate_repaired_func_name ---

@pytest.fixture
def evaluator_with_store(tmp_path):
    """Return a CandidateEvaluator with a minimal artefact boundary."""

    cfg = SimpleNamespace(
        task=SimpleNamespace(name="cmg_runtime"),
        evaluation=SimpleNamespace(metric="bic"),
    )
    run_context = RunContext.from_cfg(cfg, project_root=tmp_path)
    evaluator = CandidateEvaluator(ArtifactStore(run_context))
    try:
        yield evaluator
    finally:
        run_context.close()


def test_validate_func_name_valid(evaluator_with_store):
    """Valid code defining the expected function with proper params should pass."""
    code = """
@njit
def cognitive_model1(action_1, state, action_2, reward, model_parameters):
    '''Example model.
    Bounds:
    alpha: [0, 1]
    beta: [0, 10]
    '''
    alpha, beta = model_parameters
    n_trials = len(action_1)
    nll = 0.0
    for t in range(n_trials):
        nll -= np.log(1.0 / 2)
    return nll
"""
    assert evaluator_with_store._validate_repaired_func_name(code, "cognitive_model1") is True


def test_validate_func_name_wrong_name(evaluator_with_store):
    """Code defining a different function name should fail."""
    code = """
@njit
def cognitive_model2(action_1, state, action_2, reward, model_parameters):
    '''Model 2.
    Bounds:
    alpha: [0, 1]
    beta: [0, 10]
    '''
    alpha, beta = model_parameters
    n_trials = len(action_1)
    nll = 0.0
    for t in range(n_trials):
        nll -= np.log(1.0 / 2)
    return nll
"""
    assert evaluator_with_store._validate_repaired_func_name(code, "cognitive_model1") is False


def test_validate_func_name_name_in_comment_only(evaluator_with_store):
    """Code with expected name only in a comment should fail."""
    code = """
# This is cognitive_model1
@njit
def cognitive_model2(action_1, state, action_2, reward, model_parameters):
    '''Model 2.
    Bounds:
    alpha: [0, 1]
    beta: [0, 10]
    '''
    alpha, beta = model_parameters
    n_trials = len(action_1)
    nll = 0.0
    for t in range(n_trials):
        nll -= np.log(1.0 / 2)
    return nll
"""
    assert evaluator_with_store._validate_repaired_func_name(code, "cognitive_model1") is False


def test_validate_func_name_syntax_error(evaluator_with_store):
    """Malformed code should fail."""
    code = """
@njit
def cognitive_model1(action_1, state, action_2, reward
    return 0.0
"""
    assert evaluator_with_store._validate_repaired_func_name(code, "cognitive_model1") is False


# --- _validate_cmg_runtime (numeric generator rejection) ---

def _make_runtime_validatable(client_id, cfg):
    search = MagicMock(spec=GeCCoModelSearch)
    search.client_id = client_id
    search.cfg = cfg
    search.shared_registry = MagicMock()
    search._validate_cmg_runtime = GeCCoModelSearch._validate_cmg_runtime.__get__(
        search, GeCCoModelSearch
    )
    return search


def test_runtime_numeric_generator_rejected():
    """generator_client that is a numeric string should be rejected."""
    cmg = _make_cmg_cfg(generator_client="0")
    cfg = SimpleNamespace(
        centralized_model_generation=cmg,
        judge=SimpleNamespace(capabilities=[]),
    )
    search = _make_runtime_validatable(client_id=0, cfg=cfg)
    with pytest.raises(ValueError, match="named profile"):
        search._validate_cmg_runtime(cmg)


def test_runtime_named_generator_accepted():
    """generator_client that is a name should pass validation."""
    cmg = _make_cmg_cfg(generator_client="generator")
    cfg = SimpleNamespace(
        centralized_model_generation=cmg,
        judge=SimpleNamespace(capabilities=[]),
    )
    search = _make_runtime_validatable(client_id="generator", cfg=cfg)
    # Should not raise
    search._validate_cmg_runtime(cmg)


def test_runtime_missing_judge_configuration():
    """CMG requires validated judge configuration, not a retired flag."""
    cmg = _make_cmg_cfg(generator_client="generator")
    cfg = SimpleNamespace(
        centralized_model_generation=cmg,
        judge=None,
    )
    search = _make_runtime_validatable(client_id="generator", cfg=cfg)
    with pytest.raises(ValueError, match="judge configuration"):
        search._validate_cmg_runtime(cmg)


def test_runtime_missing_shared_registry():
    """CMG requires a shared registry."""
    cmg = _make_cmg_cfg(generator_client="generator")
    cfg = SimpleNamespace(
        centralized_model_generation=cmg,
        judge=SimpleNamespace(capabilities=[]),
    )
    search = MagicMock(spec=GeCCoModelSearch)
    search.cfg = cfg
    search.shared_registry = None
    search._validate_cmg_runtime = GeCCoModelSearch._validate_cmg_runtime.__get__(
        search, GeCCoModelSearch
    )
    with pytest.raises(ValueError, match="shared registry"):
        search._validate_cmg_runtime(cmg)


# --- _fit_candidate_model empty code (Chunk 3) ---

def test_empty_code_returns_validation_error(tmp_path):
    """Empty candidate code should return a VALIDATION_ERROR result, not None."""
    cfg = SimpleNamespace(
        evaluation=SimpleNamespace(fit_type="group"),
        task=SimpleNamespace(name="cmg_runtime"),
        data=SimpleNamespace(input_columns=[]),
    )
    run_context = RunContext.from_cfg(cfg, project_root=tmp_path)
    artifact_store = ArtifactStore(run_context)
    evaluator = CandidateEvaluator(artifact_store)

    model_dict = {
        "func_name": "cognitive_model1",
        "name": "test_model",
        "code": "",
        "parameters": [],
    }
    result, should_stop = evaluator.fit_candidate_model(
        model_dict=model_dict,
        model_idx=0,
        n_models=1,
        it=0,
        run_idx=0,
        tag="",
        model_file=artifact_store.candidate_model_path(iteration=0, run_idx=0, tag=""),
        baseline_bic=None,
        df=SimpleNamespace(),
        cfg=cfg,
    )
    assert result is not None
    assert result["metric_name"] == "VALIDATION_ERROR"
    assert result["error_type"] == "empty_code"
    assert "No code provided" in result["error_message"]
    assert should_stop is False
    run_context.close()


# --- build_prompt force_include_feedback (Chunk 2) ---

def test_repair_prompt_forces_feedback():
    """Repair prompt must include feedback even when llm.include_feedback is False."""
    cfg = SimpleNamespace(
        task=SimpleNamespace(
            name="test",
            description="desc",
            goal="Propose {models_per_iteration} models: {model_names}",
        ),
        llm=SimpleNamespace(
            provider="openai",
            models_per_iteration=1,
            include_feedback=False,
            system_prompt="sys",
            template_model="template",
            guardrails=[],
            structured_output=False,
        ),
        evaluation=SimpleNamespace(fit_type="group"),
    )
    data_text = "trial 1: choice=A"
    data = None
    feedback = "Previous models failed because..."

    from gecco.prompt_builder.prompt import build_prompt

    prompt = build_prompt(cfg, data_text, data, feedback_text=feedback, force_include_feedback=True)
    assert "Feedback from previous iterations" in prompt
    assert feedback in prompt


def test_normal_prompt_respects_include_feedback_false():
    """Normal prompt should omit feedback when llm.include_feedback is False and not forced."""
    cfg = SimpleNamespace(
        task=SimpleNamespace(
            name="test",
            description="desc",
            goal="Propose {models_per_iteration} models: {model_names}",
        ),
        llm=SimpleNamespace(
            provider="openai",
            models_per_iteration=1,
            include_feedback=False,
            system_prompt="sys",
            template_model="template",
            guardrails=[],
            structured_output=False,
        ),
        evaluation=SimpleNamespace(fit_type="group"),
    )
    data_text = "trial 1: choice=A"
    data = None
    feedback = "Previous models failed because..."

    from gecco.prompt_builder.prompt import build_prompt

    prompt = build_prompt(cfg, data_text, data, feedback_text=feedback, force_include_feedback=False)
    assert "Feedback from previous iterations" not in prompt
    assert feedback not in prompt


# --- run_n_shots resume branches (Chunks 1 & 2) ---

def test_run_n_shots_generator_resume_uses_generator_helper():
    """Generator client should use get_max_generator_iteration for resume."""
    cmg = _make_cmg_cfg(generator_client="generator", n_models=2)
    cfg = SimpleNamespace(
        task=SimpleNamespace(name="test"),
        loop=SimpleNamespace(max_iterations=0),
        centralized_model_generation=cmg,
        judge=SimpleNamespace(capabilities=[], barrier=SimpleNamespace(client_wait_seconds=1)),
        evaluation=SimpleNamespace(fit_type="group", metric="bic"),
        llm=SimpleNamespace(provider="openai", models_per_iteration=1),
        clients=SimpleNamespace(),
    )

    search = MagicMock(spec=GeCCoModelSearch)
    search.cfg = cfg
    search.client_id = "generator"
    search.shared_registry = MagicMock()
    search.shared_registry.get_max_generator_iteration.return_value = 0
    search.df = MagicMock()
    search.df_val = None
    search.best_model = None
    search.best_metric = float("inf")
    search.best_iter = -1
    search.best_params = []
    search.feedback = MagicMock()
    search.feedback.history = []
    search.feedback.get_feedback.return_value = ""
    search.feedback.record_iteration = MagicMock()
    search.tried_param_sets = []
    search._file_tag.return_value = ""
    search._sync_from_registry = MagicMock()
    search._set_activity = MagicMock()
    search._update_registry = MagicMock()
    search._cmg_config = GeCCoModelSearch._cmg_config.__get__(search, GeCCoModelSearch)
    search._cmg_is_generator = GeCCoModelSearch._cmg_is_generator.__get__(search, GeCCoModelSearch)
    search._cmg_evaluator_index = GeCCoModelSearch._cmg_evaluator_index.__get__(search, GeCCoModelSearch)
    search._validate_cmg_runtime = GeCCoModelSearch._validate_cmg_runtime.__get__(search, GeCCoModelSearch)
    search._require_distributed_coordinator = GeCCoModelSearch._require_distributed_coordinator.__get__(search, GeCCoModelSearch)
    search._run_cmg_generator_iteration = MagicMock()
    search._run_cmg_evaluator_iteration = MagicMock()
    search.distributed_coordinator = MagicMock(start_iteration=MagicMock(return_value=1))
    search.results_dir = MagicMock()

    search.run_n_shots = GeCCoModelSearch.run_n_shots.__get__(search, GeCCoModelSearch)
    search.run_n_shots(0, None)

    search.distributed_coordinator.start_iteration.assert_called_once_with(
        shared_registry=search.shared_registry,
        client_id="generator",
        cmg_cfg=cmg,
        is_generator=True,
    )


def test_run_n_shots_evaluator_resume_uses_per_client_helper():
    """Numeric evaluator client should use get_max_iteration_for_client for resume."""
    cmg = _make_cmg_cfg(generator_client="generator", n_models=2)
    cfg = SimpleNamespace(
        task=SimpleNamespace(name="test"),
        loop=SimpleNamespace(max_iterations=0),
        centralized_model_generation=cmg,
        judge=SimpleNamespace(capabilities=[], barrier=SimpleNamespace(client_wait_seconds=1)),
        evaluation=SimpleNamespace(fit_type="group", metric="bic"),
        llm=SimpleNamespace(provider="openai", models_per_iteration=1),
        clients=SimpleNamespace(),
    )

    search = MagicMock(spec=GeCCoModelSearch)
    search.cfg = cfg
    search.client_id = 0
    search.shared_registry = MagicMock()
    search.shared_registry.get_max_iteration_for_client.return_value = 0
    search.df = MagicMock()
    search.df_val = None
    search.best_model = None
    search.best_metric = float("inf")
    search.best_iter = -1
    search.best_params = []
    search.feedback = MagicMock()
    search.feedback.history = []
    search.feedback.get_feedback.return_value = ""
    search.feedback.record_iteration = MagicMock()
    search.tried_param_sets = []
    search._file_tag.return_value = ""
    search._sync_from_registry = MagicMock()
    search._set_activity = MagicMock()
    search._update_registry = MagicMock()
    search._cmg_config = GeCCoModelSearch._cmg_config.__get__(search, GeCCoModelSearch)
    search._cmg_is_generator = GeCCoModelSearch._cmg_is_generator.__get__(search, GeCCoModelSearch)
    search._cmg_evaluator_index = GeCCoModelSearch._cmg_evaluator_index.__get__(search, GeCCoModelSearch)
    search._validate_cmg_runtime = GeCCoModelSearch._validate_cmg_runtime.__get__(search, GeCCoModelSearch)
    search._require_distributed_coordinator = GeCCoModelSearch._require_distributed_coordinator.__get__(search, GeCCoModelSearch)
    search._run_cmg_generator_iteration = MagicMock()
    search._run_cmg_evaluator_iteration = MagicMock()
    search.distributed_coordinator = MagicMock(start_iteration=MagicMock(return_value=1))
    search.results_dir = MagicMock()

    search.run_n_shots = GeCCoModelSearch.run_n_shots.__get__(search, GeCCoModelSearch)
    search.run_n_shots(0, None)

    search.distributed_coordinator.start_iteration.assert_called_once_with(
        shared_registry=search.shared_registry,
        client_id=0,
        cmg_cfg=cmg,
        is_generator=False,
    )


def test_run_n_shots_respects_max_iterations_on_resume():
    """Resumed runs should not exceed configured max_iterations total count."""
    cmg = _make_cmg_cfg(generator_client="generator", n_models=2)
    cfg = SimpleNamespace(
        task=SimpleNamespace(name="test"),
        loop=SimpleNamespace(max_iterations=2),
        centralized_model_generation=cmg,
        judge=SimpleNamespace(capabilities=[], barrier=SimpleNamespace(client_wait_seconds=1)),
        evaluation=SimpleNamespace(fit_type="group", metric="bic"),
        llm=SimpleNamespace(provider="openai", models_per_iteration=1),
        clients=SimpleNamespace(),
    )

    search = MagicMock(spec=GeCCoModelSearch)
    search.cfg = cfg
    search.client_id = 0
    search.shared_registry = MagicMock()
    # Client has completed iteration 0, so resume should start at iteration 1
    search.shared_registry.get_max_iteration_for_client.return_value = 0
    search.df = MagicMock()
    search.df_val = None
    search.best_model = None
    search.best_metric = float("inf")
    search.best_iter = -1
    search.best_params = []
    search.feedback = MagicMock()
    search.feedback.history = []
    search.feedback.get_feedback.return_value = ""
    search.feedback.record_iteration = MagicMock()
    search.tried_param_sets = []
    search._file_tag.return_value = ""
    search._sync_from_registry = MagicMock()
    search._set_activity = MagicMock()
    search._update_registry = MagicMock()
    search._cmg_config = GeCCoModelSearch._cmg_config.__get__(search, GeCCoModelSearch)
    search._cmg_is_generator = GeCCoModelSearch._cmg_is_generator.__get__(search, GeCCoModelSearch)
    search._cmg_evaluator_index = GeCCoModelSearch._cmg_evaluator_index.__get__(search, GeCCoModelSearch)
    search._validate_cmg_runtime = GeCCoModelSearch._validate_cmg_runtime.__get__(search, GeCCoModelSearch)
    search._require_distributed_coordinator = GeCCoModelSearch._require_distributed_coordinator.__get__(search, GeCCoModelSearch)
    # Track which iterations the evaluator processes
    processed_iterations = []

    def mock_evaluator(it, run_idx, feedback, cmg_cfg, baseline_bic):
        processed_iterations.append(it)

    search._run_cmg_evaluator_iteration = mock_evaluator
    search._run_cmg_generator_iteration = MagicMock()
    search.distributed_coordinator = MagicMock(start_iteration=MagicMock(return_value=1))
    search.results_dir = MagicMock()

    search.run_n_shots = GeCCoModelSearch.run_n_shots.__get__(search, GeCCoModelSearch)
    search.run_n_shots(0, None)

    # start_iteration returns 1, so the client should only process iteration 1.
    assert processed_iterations == [1]


def test_run_n_shots_preserves_terminal_no_success_status(tmp_path):
    """CMG evaluator exit should not overwrite terminal no-success status."""

    cmg = _make_cmg_cfg(generator_client="generator", n_models=1)
    cfg = SimpleNamespace(
        task=SimpleNamespace(name="cmg_runtime"),
        loop=SimpleNamespace(max_iterations=1),
        centralized_model_generation=cmg,
        judge=SimpleNamespace(capabilities=[], barrier=SimpleNamespace(client_wait_seconds=1)),
        evaluation=SimpleNamespace(fit_type="group", metric="bic"),
        llm=SimpleNamespace(provider="openai", models_per_iteration=1),
        clients=SimpleNamespace(),
    )

    search = MagicMock(spec=GeCCoModelSearch)
    search.cfg = cfg
    search.client_id = 0
    search.shared_registry = SharedRegistry(tmp_path / "shared_registry.duckdb")
    search.shared_registry.mark_complete = MagicMock()
    search.distributed_coordinator = MagicMock(start_iteration=MagicMock(return_value=0))
    search.df = SimpleNamespace()
    search.df_val = None
    search.best_model = None
    search.best_metric = float("inf")
    search.best_iter = -1
    search.best_params = []
    search.feedback = MagicMock()
    search.feedback.history = []
    search.feedback.record_iteration = MagicMock()
    search.tried_param_sets = []
    search.results_dir = tmp_path / "results"
    search._file_tag = MagicMock(return_value="")
    search._sync_from_registry = MagicMock()
    search._set_activity = MagicMock()
    search._sync_best_attrs_from_state = MagicMock()
    search._cmg_config = GeCCoModelSearch._cmg_config.__get__(search, GeCCoModelSearch)
    search._cmg_is_generator = GeCCoModelSearch._cmg_is_generator.__get__(search, GeCCoModelSearch)
    search._cmg_evaluator_index = GeCCoModelSearch._cmg_evaluator_index.__get__(search, GeCCoModelSearch)
    search._validate_cmg_runtime = GeCCoModelSearch._validate_cmg_runtime.__get__(search, GeCCoModelSearch)
    search._require_distributed_coordinator = GeCCoModelSearch._require_distributed_coordinator.__get__(search, GeCCoModelSearch)
    search._run_cmg_generator_iteration = MagicMock()

    def _fake_run_cmg_evaluator_iteration(it, run_idx, feedback, cmg_cfg, baseline_bic):
        search.shared_registry.update(
            client_id=0,
            iteration=it,
            results=[
                {
                    "function_name": "model_a",
                    "metric_name": "VALIDATION_ERROR",
                    "metric_value": float("inf"),
                    "param_names": [],
                    "code": "def model_a():\n    return 0",
                }
            ],
            status="complete_no_success",
            had_runnable_model=False,
        )

    search._run_cmg_evaluator_iteration = _fake_run_cmg_evaluator_iteration
    search.run_n_shots = GeCCoModelSearch.run_n_shots.__get__(search, GeCCoModelSearch)

    search.run_n_shots(0, None)

    snapshot = search.shared_registry.read()

    assert snapshot["client_entries"]["0"]["status"] == "complete_no_success"
    assert snapshot["client_entries"]["0"]["had_runnable_model"] is False
    search.shared_registry.mark_complete.assert_not_called()


# --- _is_cmg_repairable_error (Chunk 2) ---

def _make_search_with_repairable(tmp_path):
    cfg = SimpleNamespace(
        data=SimpleNamespace(input_columns=["action_1", "state", "action_2", "reward"]),
        llm=SimpleNamespace(
            abstract_base_model="""\
class CognitiveModelBase:
    pass

def make_cognitive_model(model_cls):
    def cognitive_model(action_1, state, action_2, reward, model_parameters):
        return model_cls().compute_nll(action_1, state, action_2, reward, model_parameters)
    return cognitive_model
""",
        ),
    )
    run_context = RunContext.from_cfg(
        SimpleNamespace(task=SimpleNamespace(name="cmg_runtime"), evaluation=SimpleNamespace(metric="bic")),
        project_root=tmp_path,
    )
    evaluator = CandidateEvaluator(ArtifactStore(run_context))
    return evaluator, cfg, run_context


def test_recovery_simulation_failure_is_repairable(tmp_path):
    """RECOVERY_FAILED with simulation_error and 0 successes should be repairable."""
    evaluator, _, run_context = _make_search_with_repairable(tmp_path)
    result = {
        "metric_name": "RECOVERY_FAILED",
        "simulation_error": "TypeError: bad operand type for unary -: 'NoneType'",
        "recovery_n_successful": 0,
    }
    assert evaluator._is_repairable_error(result) is True
    run_context.close()


def test_poor_recovery_is_not_repairable(tmp_path):
    """RECOVERY_FAILED without simulation_error and with some successes is not repairable."""
    evaluator, _, run_context = _make_search_with_repairable(tmp_path)
    result = {
        "metric_name": "RECOVERY_FAILED",
        "simulation_error": None,
        "recovery_n_successful": 50,
        "recovery_r": 0.1,
    }
    assert evaluator._is_repairable_error(result) is False
    run_context.close()


def test_validation_error_is_repairable(tmp_path):
    """VALIDATION_ERROR should still be repairable."""
    evaluator, _, run_context = _make_search_with_repairable(tmp_path)
    assert evaluator._is_repairable_error({"metric_name": "VALIDATION_ERROR"}) is True
    run_context.close()


def test_fit_error_is_repairable(tmp_path):
    """FIT_ERROR should still be repairable."""
    evaluator, _, run_context = _make_search_with_repairable(tmp_path)
    assert evaluator._is_repairable_error({"metric_name": "FIT_ERROR"}) is True
    run_context.close()


def test_none_result_is_not_repairable(tmp_path):
    """None result should not be repairable."""
    evaluator, _, run_context = _make_search_with_repairable(tmp_path)
    assert evaluator._is_repairable_error(None) is False
    run_context.close()


# --- _smoke_test_model_return_value (Chunk 4) ---

def test_smoke_test_catches_none(tmp_path):
    """A model returning None should produce an error string."""
    evaluator, cfg, run_context = _make_search_with_repairable(tmp_path)

    def bad_model(action_1, state, action_2, reward, model_parameters):
        return None

    spec = SimpleNamespace(
        func=bad_model,
        param_names=["alpha"],
        bounds={"alpha": [0, 1]},
    )
    error = evaluator._smoke_test_model_return_value(spec, cfg)
    assert error is not None
    assert "returned None" in error
    run_context.close()


def test_smoke_test_catches_non_numeric(tmp_path):
    """A model returning a non-numeric string should produce an error string."""
    evaluator, cfg, run_context = _make_search_with_repairable(tmp_path)

    def bad_model(action_1, state, action_2, reward, model_parameters):
        return "not a number"

    spec = SimpleNamespace(
        func=bad_model,
        param_names=["alpha"],
        bounds={"alpha": [0, 1]},
    )
    error = evaluator._smoke_test_model_return_value(spec, cfg)
    assert error is not None
    assert "non-numeric" in error
    run_context.close()


def test_smoke_test_catches_non_finite(tmp_path):
    """A model returning inf should produce an error string."""
    evaluator, cfg, run_context = _make_search_with_repairable(tmp_path)

    def bad_model(action_1, state, action_2, reward, model_parameters):
        return float("inf")

    spec = SimpleNamespace(
        func=bad_model,
        param_names=["alpha"],
        bounds={"alpha": [0, 1]},
    )
    error = evaluator._smoke_test_model_return_value(spec, cfg)
    assert error is not None
    assert "non-finite" in error
    run_context.close()


def test_smoke_test_accepts_numeric_return(tmp_path):
    """A model returning a finite numeric value should pass the smoke test."""
    evaluator, cfg, run_context = _make_search_with_repairable(tmp_path)

    def good_model(action_1, state, action_2, reward, model_parameters):
        return 1.23

    spec = SimpleNamespace(
        func=good_model,
        param_names=["alpha"],
        bounds={"alpha": [0, 1]},
    )
    error = evaluator._smoke_test_model_return_value(spec, cfg)
    assert error is None
    run_context.close()


def test_validate_func_name_class_based_candidate_uses_context(evaluator_with_store):
    """Class-based repaired candidates should validate with config-backed context."""

    cfg = SimpleNamespace(
        llm=SimpleNamespace(
            abstract_base_model="""\
class CognitiveModelBase:
    pass

def make_cognitive_model(model_cls):
    def wrapper(action_1, state, action_2, reward, model_parameters):
        return model_cls().compute_nll(action_1, state, action_2, reward, model_parameters)
    return wrapper
"""
        )
    )
    code = """
class ParticipantModel1(CognitiveModelBase):
    \"\"\"Bounds:\nalpha: [0, 1]\"\"\"

    def compute_nll(self, action_1, state, action_2, reward, model_parameters):
        alpha, = model_parameters
        return float(alpha)

@njit
def cognitive_model1(action_1, state, action_2, reward, model_parameters):
    return ParticipantModel1().compute_nll(
        action_1,
        state,
        action_2,
        reward,
        model_parameters,
    )
"""

    assert evaluator_with_store._validate_repaired_func_name(
        code,
        "cognitive_model1",
        cfg=cfg,
        structured_params=[{"name": "alpha", "lower_bound": 0, "upper_bound": 1}],
    ) is True
