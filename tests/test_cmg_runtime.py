"""Tests for CMG runtime validation and evaluator index mapping."""

from unittest.mock import MagicMock, patch
from types import SimpleNamespace

import pytest

from gecco.run_gecco import GeCCoModelSearch


def _make_cmg_cfg(enabled=True, generator_client="generator", n_models=2):
    return SimpleNamespace(
        enabled=enabled,
        generator_client=generator_client,
        n_models=n_models,
    )


def _make_search(client_id=None, cfg=None):
    mock_cfg = cfg or SimpleNamespace(
        judge=SimpleNamespace(orchestrated=True),
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
    cfg = SimpleNamespace(centralized_model_generation=cmg, judge=SimpleNamespace(orchestrated=True))
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
def search_with_cfg():
    """Return a GeCCoModelSearch-like object with a minimal cfg for build_model_spec."""
    cfg = SimpleNamespace(
        evaluation=SimpleNamespace(metric="bic"),
    )
    search = MagicMock(spec=GeCCoModelSearch)
    search.cfg = cfg
    search._validate_repaired_func_name = GeCCoModelSearch._validate_repaired_func_name.__get__(
        search, GeCCoModelSearch
    )
    return search


def test_validate_func_name_valid(search_with_cfg):
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
    assert search_with_cfg._validate_repaired_func_name(code, "cognitive_model1") is True


def test_validate_func_name_wrong_name(search_with_cfg):
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
    assert search_with_cfg._validate_repaired_func_name(code, "cognitive_model1") is False


def test_validate_func_name_name_in_comment_only(search_with_cfg):
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
    assert search_with_cfg._validate_repaired_func_name(code, "cognitive_model1") is False


def test_validate_func_name_syntax_error(search_with_cfg):
    """Malformed code should fail."""
    code = """
@njit
def cognitive_model1(action_1, state, action_2, reward
    return 0.0
"""
    assert search_with_cfg._validate_repaired_func_name(code, "cognitive_model1") is False


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
        judge=SimpleNamespace(orchestrated=True),
    )
    search = _make_runtime_validatable(client_id=0, cfg=cfg)
    with pytest.raises(ValueError, match="named profile"):
        search._validate_cmg_runtime(cmg)


def test_runtime_named_generator_accepted():
    """generator_client that is a name should pass validation."""
    cmg = _make_cmg_cfg(generator_client="generator")
    cfg = SimpleNamespace(
        centralized_model_generation=cmg,
        judge=SimpleNamespace(orchestrated=True),
    )
    search = _make_runtime_validatable(client_id="generator", cfg=cfg)
    # Should not raise
    search._validate_cmg_runtime(cmg)


def test_runtime_missing_judge_orchestrated():
    """CMG requires judge.orchestrated to be True."""
    cmg = _make_cmg_cfg(generator_client="generator")
    cfg = SimpleNamespace(
        centralized_model_generation=cmg,
        judge=SimpleNamespace(orchestrated=False),
    )
    search = _make_runtime_validatable(client_id="generator", cfg=cfg)
    with pytest.raises(ValueError, match="judge.orchestrated"):
        search._validate_cmg_runtime(cmg)


def test_runtime_missing_shared_registry():
    """CMG requires a shared registry."""
    cmg = _make_cmg_cfg(generator_client="generator")
    cfg = SimpleNamespace(
        centralized_model_generation=cmg,
        judge=SimpleNamespace(orchestrated=True),
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

def test_empty_code_returns_validation_error():
    """Empty candidate code should return a VALIDATION_ERROR result, not None."""
    search = MagicMock(spec=GeCCoModelSearch)
    search.cfg = SimpleNamespace(evaluation=SimpleNamespace(metric="bic"))
    search.recovery_checker = None
    search.df = None
    search.df_val = None
    search.id_eval_data = None
    search._ppc_simulator = None
    search.ppc_enabled = False
    search.block_residuals_enabled = False
    search.best_metric = float("inf")
    search.tried_param_sets = []
    search._set_activity = MagicMock()
    search.results_dir = MagicMock()

    search._fit_candidate_model = GeCCoModelSearch._fit_candidate_model.__get__(
        search, GeCCoModelSearch
    )

    model_dict = {
        "func_name": "cognitive_model1",
        "name": "test_model",
        "code": "",
        "parameters": [],
    }
    result, should_stop = search._fit_candidate_model(
        model_dict=model_dict,
        model_idx=0,
        n_models=1,
        it=0,
        run_idx=0,
        tag="",
        model_file="dummy.txt",
        baseline_bic=None,
    )
    assert result is not None
    assert result["metric_name"] == "VALIDATION_ERROR"
    assert result["error_type"] == "empty_code"
    assert "No code provided" in result["error_message"]
    assert should_stop is False


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
        judge=SimpleNamespace(orchestrated=True, barrier=SimpleNamespace(client_wait_seconds=1)),
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
    search._run_cmg_generator_iteration = MagicMock()
    search._run_cmg_evaluator_iteration = MagicMock()
    search.results_dir = MagicMock()

    search.run_n_shots = GeCCoModelSearch.run_n_shots.__get__(search, GeCCoModelSearch)
    search.run_n_shots(0, None)

    search.shared_registry.get_max_generator_iteration.assert_called_once_with("generator")
    search.shared_registry.get_max_iteration_for_client.assert_not_called()


def test_run_n_shots_evaluator_resume_uses_per_client_helper():
    """Numeric evaluator client should use get_max_iteration_for_client for resume."""
    cmg = _make_cmg_cfg(generator_client="generator", n_models=2)
    cfg = SimpleNamespace(
        task=SimpleNamespace(name="test"),
        loop=SimpleNamespace(max_iterations=0),
        centralized_model_generation=cmg,
        judge=SimpleNamespace(orchestrated=True, barrier=SimpleNamespace(client_wait_seconds=1)),
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
    search._run_cmg_generator_iteration = MagicMock()
    search._run_cmg_evaluator_iteration = MagicMock()
    search.results_dir = MagicMock()

    search.run_n_shots = GeCCoModelSearch.run_n_shots.__get__(search, GeCCoModelSearch)
    search.run_n_shots(0, None)

    search.shared_registry.get_max_iteration_for_client.assert_called_once_with(0)
    search.shared_registry.get_max_generator_iteration.assert_not_called()


def test_run_n_shots_respects_max_iterations_on_resume():
    """Resumed runs should not exceed configured max_iterations total count."""
    cmg = _make_cmg_cfg(generator_client="generator", n_models=2)
    cfg = SimpleNamespace(
        task=SimpleNamespace(name="test"),
        loop=SimpleNamespace(max_iterations=2),
        centralized_model_generation=cmg,
        judge=SimpleNamespace(orchestrated=True, barrier=SimpleNamespace(client_wait_seconds=1)),
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
    # Track which iterations the evaluator processes
    processed_iterations = []

    def mock_evaluator(it, run_idx, feedback, cmg_cfg, baseline_bic):
        processed_iterations.append(it)

    search._run_cmg_evaluator_iteration = mock_evaluator
    search._run_cmg_generator_iteration = MagicMock()
    search.results_dir = MagicMock()

    search.run_n_shots = GeCCoModelSearch.run_n_shots.__get__(search, GeCCoModelSearch)
    search.run_n_shots(0, None)

    # max_iterations=2, completed iteration 0, so should only process iteration 1
    assert processed_iterations == [1]


# --- _is_cmg_repairable_error (Chunk 2) ---

def _make_search_with_repairable():
    cfg = SimpleNamespace(
        data=SimpleNamespace(input_columns=["action_1", "state", "action_2", "reward"]),
    )
    search = MagicMock(spec=GeCCoModelSearch)
    search.cfg = cfg
    search._is_cmg_repairable_error = GeCCoModelSearch._is_cmg_repairable_error.__get__(
        search, GeCCoModelSearch
    )
    search._smoke_test_model_return_value = GeCCoModelSearch._smoke_test_model_return_value.__get__(
        search, GeCCoModelSearch
    )
    return search


def test_recovery_simulation_failure_is_repairable():
    """RECOVERY_FAILED with simulation_error and 0 successes should be repairable."""
    search = _make_search_with_repairable()
    result = {
        "metric_name": "RECOVERY_FAILED",
        "simulation_error": "TypeError: bad operand type for unary -: 'NoneType'",
        "recovery_n_successful": 0,
    }
    assert search._is_cmg_repairable_error(result) is True


def test_poor_recovery_is_not_repairable():
    """RECOVERY_FAILED without simulation_error and with some successes is not repairable."""
    search = _make_search_with_repairable()
    result = {
        "metric_name": "RECOVERY_FAILED",
        "simulation_error": None,
        "recovery_n_successful": 50,
        "recovery_r": 0.1,
    }
    assert search._is_cmg_repairable_error(result) is False


def test_validation_error_is_repairable():
    """VALIDATION_ERROR should still be repairable."""
    search = _make_search_with_repairable()
    assert search._is_cmg_repairable_error({"metric_name": "VALIDATION_ERROR"}) is True


def test_fit_error_is_repairable():
    """FIT_ERROR should still be repairable."""
    search = _make_search_with_repairable()
    assert search._is_cmg_repairable_error({"metric_name": "FIT_ERROR"}) is True


def test_none_result_is_not_repairable():
    """None result should not be repairable."""
    search = _make_search_with_repairable()
    assert search._is_cmg_repairable_error(None) is False


# --- _smoke_test_model_return_value (Chunk 4) ---

def test_smoke_test_catches_none():
    """A model returning None should produce an error string."""
    search = _make_search_with_repairable()

    def bad_model(action_1, state, action_2, reward, model_parameters):
        return None

    spec = SimpleNamespace(
        func=bad_model,
        param_names=["alpha"],
        bounds={"alpha": [0, 1]},
    )
    error = search._smoke_test_model_return_value(spec)
    assert error is not None
    assert "returned None" in error


def test_smoke_test_catches_non_numeric():
    """A model returning a non-numeric string should produce an error string."""
    search = _make_search_with_repairable()

    def bad_model(action_1, state, action_2, reward, model_parameters):
        return "not a number"

    spec = SimpleNamespace(
        func=bad_model,
        param_names=["alpha"],
        bounds={"alpha": [0, 1]},
    )
    error = search._smoke_test_model_return_value(spec)
    assert error is not None
    assert "non-numeric" in error


def test_smoke_test_catches_non_finite():
    """A model returning inf should produce an error string."""
    search = _make_search_with_repairable()

    def bad_model(action_1, state, action_2, reward, model_parameters):
        return float("inf")

    spec = SimpleNamespace(
        func=bad_model,
        param_names=["alpha"],
        bounds={"alpha": [0, 1]},
    )
    error = search._smoke_test_model_return_value(spec)
    assert error is not None
    assert "non-finite" in error


def test_smoke_test_accepts_numeric_return():
    """A model returning a finite numeric value should pass the smoke test."""
    search = _make_search_with_repairable()

    def good_model(action_1, state, action_2, reward, model_parameters):
        return 1.23

    spec = SimpleNamespace(
        func=good_model,
        param_names=["alpha"],
        bounds={"alpha": [0, 1]},
    )
    error = search._smoke_test_model_return_value(spec)
    assert error is None
