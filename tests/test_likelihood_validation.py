"""Contract tests for likelihood validation (pre-fit static + post-fit)."""

from __future__ import annotations

import pytest

from gecco.offline_evaluation.likelihood_validation import (
    InvalidLikelihoodError,
    ValidationResult,
    validate_likelihood_static,
    validate_likelihood_post_fit,
    validation_error_result,
    is_valid_likelihood_result,
)


# ========================================================================
# Static validation
# ========================================================================


class TestStaticValidation:
    """Static pre-fit checks — no execution, pure AST/code analysis."""

    VALID_TWO_STEP = """
def cognitive_model(stimulus, action, reward, params):
    alpha, beta = params
    n_trials = len(stimulus)
    Q = 0.0
    log_lik = 0.0
    for t in range(n_trials):
        # compute prediction error
        pe = reward[t] - Q
        # compute choice probability (logit)
        log_prob = -beta * abs(pe)
        prob = 2.0 * 1.0 / (1.0 + math.exp(-beta * abs(pe)))
        # accumulate log likelihood
        log_lik += math.log(prob + 1e-10)
        # update Q after observing outcome
        Q = Q + alpha * pe
    return -log_lik
"""

    # Valid action-conditioned likelihood: action[t] is used in the
    # probability computation (selecting which arm's probability to
    # compute) BEFORE the log_lik accumulation, but this is the correct
    # first step of computing the likelihood of the observed choice.
    VALID_ACTION_CONDITIONED = """
def cognitive_model(stimulus, action, reward, params):
    alpha, beta = params
    n_trials = len(stimulus)
    Q = [0.0, 0.0]
    log_lik = 0.0
    for t in range(n_trials):
        # compute probability for the observed action using action[t]
        q_selected = Q[action[t]]
        q_other = Q[1 - action[t]]
        prob = 1.0 / (1.0 + math.exp(beta * (q_other - q_selected)))
        # accumulate log likelihood for the observed choice
        log_lik += math.log(prob + 1e-10)
        # update Q values after observing outcome
        pe = reward[t] - q_selected
        Q[action[t]] = Q[action[t]] + alpha * pe
    return -log_lik
"""

    RETURN_ZERO = """
def cognitive_model(stimulus, action, reward, params):
    return 0.0
"""

    RETURN_ZERO_INT = """
def cognitive_model(stimulus, action, reward, params):
    return 0
"""

    CHOICE_LEAKAGE = """
def cognitive_model(stimulus, action, reward, params):
    alpha, beta = params
    n_trials = len(stimulus)
    Q = 0.0
    log_lik = 0.0
    for t in range(n_trials):
        # Use action before computing likelihood — leaks the answer
        if action[t] == 1:
            Q = Q + alpha * (reward[t] - Q)
        # likelihood comes after the state update using action
        pe = reward[t] - Q
        prob = 1.0 / (1.0 + math.exp(-beta * pe))
        log_lik += math.log(prob + 1e-10)
    return -log_lik
"""

    # Action-driven state update before probability computation.
    # Unlike VALID_ACTION_CONDITIONED where action[t] only selects
    # which probability to compute, this model uses action[t] to
    # update Q values BEFORE computing the probability, which
    # leaks the correct answer.
    ACTION_DRIVEN_STATE_UPDATE = """
def cognitive_model(stimulus, action, reward, params):
    alpha, beta = params
    n_trials = len(stimulus)
    Q = [0.0, 0.0]
    log_lik = 0.0
    for t in range(n_trials):
        # State update using action BEFORE probability (leakage)
        Q[action[t]] = Q[action[t]] + alpha * (reward[t] - Q[action[t]])
        pe = reward[t] - Q[action[t]]
        prob = 1.0 / (1.0 + math.exp(-beta * pe))
        log_lik += math.log(prob + 1e-10)
    return -log_lik
"""

    NON_ACCUMULATED = """
def cognitive_model(stimulus, action, reward, params):
    alpha, beta = params
    n_trials = len(stimulus)
    Q = 0.0
    for t in range(n_trials):
        pe = reward[t] - Q
        prob = 1.0 / (1.0 + math.exp(-beta * pe))
        log_lik = math.log(prob + 1e-10)  # = not +=
        Q = Q + alpha * pe
    return -log_lik
"""

    def test_static_rejects_return_zero(self):
        result = validate_likelihood_static(self.RETURN_ZERO, "cognitive_model")
        assert not result.passed
        assert result.error_type == "constant_likelihood"

    def test_static_rejects_return_zero_int(self):
        result = validate_likelihood_static(self.RETURN_ZERO_INT, "cognitive_model")
        assert not result.passed
        assert result.error_type == "constant_likelihood"

    def test_static_rejects_choice_leakage(self):
        result = validate_likelihood_static(self.CHOICE_LEAKAGE, "cognitive_model")
        assert not result.passed
        assert result.error_type == "choice_leakage"

    def test_static_rejects_action_driven_state_update(self):
        """Action-driven state update before likelihood still fails."""
        result = validate_likelihood_static(
            self.ACTION_DRIVEN_STATE_UPDATE, "cognitive_model"
        )
        assert not result.passed
        assert result.error_type == "choice_leakage"

    def test_static_rejects_non_accumulated_nll(self):
        result = validate_likelihood_static(self.NON_ACCUMULATED, "cognitive_model")
        assert not result.passed
        assert result.error_type == "non_additive_nll"

    def test_static_accepts_valid_two_step(self):
        result = validate_likelihood_static(self.VALID_TWO_STEP, "cognitive_model")
        assert result.passed

    def test_static_accepts_action_conditioned_likelihood(self):
        """Valid action-conditioned likelihood with action[t] in prob computation
        must NOT be rejected as choice leakage."""
        result = validate_likelihood_static(
            self.VALID_ACTION_CONDITIONED, "cognitive_model"
        )
        assert result.passed, (
            f"Expected valid action-conditioned likelihood to pass, "
            f"got error_type={result.error_type}: {result.error_message}"
        )

    # ---- Suffixed choice-column tests ---- #

    SUFFIXED_CHOICE_LEAKAGE = """
def cognitive_model(stimulus, action_1, action_2, reward, params):
    alpha, beta = params
    n_trials = len(stimulus)
    Q = [0.0, 0.0]
    log_lik = 0.0
    for t in range(n_trials):
        # Use suffixed choice column in state update BEFORE likelihood — leaks answer
        Q[action_1[t]] = Q[action_1[t]] + alpha * (reward[t] - Q[action_1[t]])
        pe = reward[t] - Q[action_1[t]]
        prob = 1.0 / (1.0 + math.exp(-beta * pe))
        log_lik += math.log(prob + 1e-10)
    return -log_lik
"""

    # Valid: suffixed choice column used only in probability computation,
    # which is the correct first step of a likelihood calculation.
    VALID_SUFFIXED_ACTION_CONDITIONED = """
def cognitive_model(stimulus, action_1, action_2, reward, params):
    alpha, beta = params
    n_trials = len(stimulus)
    Q_1 = 0.0
    Q_2 = 0.0
    log_lik = 0.0
    for t in range(n_trials):
        # Use action_1[t] to select which Q for probability computation
        q_selected = Q_1 if action_1[t] == 1 else Q_2
        q_other = Q_2 if action_1[t] == 1 else Q_1
        prob = 1.0 / (1.0 + math.exp(beta * (q_other - q_selected)))
        log_lik += math.log(prob + 1e-10)
        # Update after likelihood
        pe = reward[t] - q_selected
        if action_1[t] == 1:
            Q_1 = Q_1 + alpha * pe
        else:
            Q_2 = Q_2 + alpha * pe
    return -log_lik
"""

    # Explicit kernel[action_1[t]] += ... pattern from the plan
    KERNEL_SUFFIXED_LEAKAGE = """
def cognitive_model(stimulus, action_1, reward, params):
    alpha, beta = params
    n_trials = len(stimulus)
    Q = [0.0, 0.0]
    kernel = [0.0, 0.0]
    log_lik = 0.0
    for t in range(n_trials):
        # Leak: kernel updated using action_1[t] before log-likelihood
        kernel[action_1[t]] += alpha * (reward[t] - Q[action_1[t]])
        pe = reward[t] - Q[action_1[t]]
        prob = 1.0 / (1.0 + math.exp(-beta * pe))
        log_lik += math.log(prob + 1e-10)
    return -log_lik
"""

    def test_static_rejects_suffixed_choice_leakage(self):
        """action_1[t] used in state update before likelihood is rejected."""
        result = validate_likelihood_static(
            self.SUFFIXED_CHOICE_LEAKAGE, "cognitive_model"
        )
        assert not result.passed
        assert result.error_type == "choice_leakage"

    def test_static_rejects_kernel_suffixed_leakage(self):
        """kernel[action_1[t]] += ... before likelihood is rejected."""
        result = validate_likelihood_static(
            self.KERNEL_SUFFIXED_LEAKAGE, "cognitive_model"
        )
        assert not result.passed
        assert result.error_type == "choice_leakage"

    def test_static_accepts_suffixed_choice_conditioned_likelihood(self):
        """Valid action_1[t] usage in probability computation must NOT be
        rejected as choice leakage."""
        result = validate_likelihood_static(
            self.VALID_SUFFIXED_ACTION_CONDITIONED, "cognitive_model"
        )
        assert result.passed, (
            f"Expected valid suffixed action-conditioned likelihood to pass, "
            f"got error_type={result.error_type}: {result.error_message}"
        )

    def test_static_handles_missing_func(self):
        code = "def other_func(x):\n    return 0.0\n"
        result = validate_likelihood_static(code, "cognitive_model")
        assert not result.passed
        assert result.error_type == "parse_error"

    def test_static_handles_syntax_error(self):
        code_bad = "def cognitive_model(x):\n    return \\n"
        result = validate_likelihood_static(code_bad, "cognitive_model")
        assert not result.passed


# ========================================================================
# Post-fit validation
# ========================================================================


class TestPostFitValidation:

    def test_post_fit_rejects_all_zero_nll(self):
        result = validate_likelihood_post_fit(
            per_participant_nll=[0.0, 0.0, 0.0],
            mean_nll=0.0,
            func_name="test_model",
            n_participants=3,
        )
        assert not result.passed
        assert result.error_type == "degenerate_nll"

    def test_post_fit_rejects_negative_nll(self):
        result = validate_likelihood_post_fit(
            per_participant_nll=[1.0, -0.5, 2.0],
            mean_nll=0.833,
            func_name="test_model",
            n_participants=3,
        )
        assert not result.passed
        assert result.error_type == "degenerate_nll"

    def test_post_fit_rejects_tiny_per_choice_nll(self):
        """Per-choice NLL below threshold is rejected even when
        mean total NLL is above 1e-6."""
        result = validate_likelihood_post_fit(
            per_participant_nll=[1e-5],   # total NLL = 1e-5 (> 1e-6)
            mean_nll=1e-5,
            func_name="test_model",
            n_participants=1,
            participant_n_trials=[10000],  # per_choice = 1e-9 (< 1e-8)
        )
        assert not result.passed
        assert result.error_type == "degenerate_nll"
        assert result.error_details.get("mean_per_choice_nll", 1) < 1e-8

    def test_post_fit_rejects_tiny_nll_without_trial_counts(self):
        """Without trial counts, per-participant NLL is used directly
        as per-choice NLL, so tiny values are still rejected."""
        result = validate_likelihood_post_fit(
            per_participant_nll=[1e-8, 1e-9, 1e-10],
            mean_nll=1e-9,
            func_name="test_model",
            n_participants=3,
        )
        assert not result.passed
        assert result.error_type == "degenerate_nll"

    def test_post_fit_accepts_large_total_nll_with_many_trials(self):
        """A large total NLL with many trials should produce reasonable
        per-choice NLL and pass."""
        result = validate_likelihood_post_fit(
            per_participant_nll=[50.0, 60.0],
            mean_nll=55.0,
            func_name="test_model",
            n_participants=2,
            participant_n_trials=[100, 100],
        )
        assert result.passed

    def test_post_fit_rejects_missing_data(self):
        result = validate_likelihood_post_fit(
            per_participant_nll=None,
            mean_nll=None,
            func_name="test_model",
            n_participants=0,
        )
        assert not result.passed
        assert result.error_type == "degenerate_nll"

    def test_post_fit_accepts_valid_nll(self):
        result = validate_likelihood_post_fit(
            per_participant_nll=[50.0, 60.0, 55.0],
            mean_nll=55.0,
            func_name="test_model",
            n_participants=3,
        )
        assert result.passed


# ========================================================================
# Result helpers
# ========================================================================


class TestResultHelpers:

    def test_validation_error_result(self):
        error = InvalidLikelihoodError(
            message="Constant zero NLL",
            error_type="constant_likelihood",
            details={"func_name": "m", "snippet": "return 0.0"},
        )
        result = validation_error_result(error)
        assert result["metric_name"] == "VALIDATION_ERROR"
        assert result["metric_value"] == float("inf")
        assert result["error_type"] == "constant_likelihood"
        assert result["error_message"] == "Constant zero NLL"
        assert result["error_details"] == {"func_name": "m", "snippet": "return 0.0"}

    def test_is_valid_likelihood_result_positive(self):
        assert is_valid_likelihood_result(
            {"metric_name": "BIC", "metric_value": 100.0}
        )

    def test_is_valid_likelihood_result_validation_error(self):
        assert not is_valid_likelihood_result(
            {"metric_name": "VALIDATION_ERROR", "metric_value": float("inf")}
        )

    def test_is_valid_likelihood_result_missing_metric(self):
        assert not is_valid_likelihood_result({})
