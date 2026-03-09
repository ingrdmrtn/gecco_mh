"""
Tests for the scipy-based HBI implementation.

Includes parameter recovery tests using a simple 2-armed bandit RL model.
"""

import numpy as np
import pytest
from dataclasses import dataclass
from typing import List, Dict, Any

from gecco.model_fitting.hbi_scipy import (
    to_unbounded,
    to_bounded,
    log_det_jacobian,
    params_to_unbounded,
    params_to_bounded,
    numerical_hessian,
    optimize_subject_map,
    run_hbi_scipy,
)


# --- Simple RL Model for Testing ---


def simple_rl_model(choices, rewards, model_parameters):
    """
    Simple 2-armed bandit Q-learning with softmax.

    Bounds:
    alpha: [0, 1]
    beta: [0, 10]
    """
    alpha, beta = model_parameters
    n_trials = len(choices)
    Q = np.zeros(2)
    nll = 0.0
    for t in range(n_trials):
        # Softmax policy
        exp_q = np.exp(beta * (Q - np.max(Q)))  # numerically stable
        probs = exp_q / np.sum(exp_q)
        c = int(choices[t])
        nll -= np.log(probs[c] + 1e-10)
        # Value update
        r = rewards[t]
        Q[c] += alpha * (r - Q[c])
    return nll


@dataclass
class MockModelSpec:
    """Minimal ModelSpec for testing."""
    name: str
    func: Any
    param_names: List[str]
    bounds: Dict[str, List[float]]


def create_rl_model_spec():
    return MockModelSpec(
        name="simple_rl",
        func=simple_rl_model,
        param_names=["alpha", "beta"],
        bounds={"alpha": [0.0, 1.0], "beta": [0.0, 10.0]},
    )


def simulate_subject(alpha, beta, n_trials, rng):
    """Simulate data from a single subject with known parameters."""
    Q = np.zeros(2)
    choices = np.zeros(n_trials, dtype=np.float64)
    rewards = np.zeros(n_trials, dtype=np.float64)

    # Two arms with different reward probabilities
    reward_probs = [0.3, 0.7]

    for t in range(n_trials):
        # Softmax choice
        exp_q = np.exp(beta * (Q - np.max(Q)))
        probs = exp_q / np.sum(exp_q)

        c = rng.choice(2, p=probs)
        r = float(rng.random() < reward_probs[c])

        choices[t] = c
        rewards[t] = r

        # Update
        Q[c] += alpha * (r - Q[c])

    return choices, rewards


def _sample_beta_scaled(a, b, lb, ub, rng):
    """Sample from Beta(a, b) scaled to [lb, ub]."""
    return lb + rng.beta(a, b) * (ub - lb)


def _sample_truncnorm(mean, std, lb, ub, rng):
    """Sample from truncated normal clipped to [lb, ub]."""
    return np.clip(rng.normal(mean, std), lb, ub)


def generate_synthetic_dataset(
    n_subjects, n_trials,
    alpha_dist="normal", alpha_params=None,
    beta_dist="normal", beta_params=None,
    seed=42,
):
    """
    Generate synthetic data from known individual-level parameters.

    Parameters
    ----------
    alpha_dist : str
        Distribution for alpha: "normal" or "beta".
    alpha_params : dict
        For "normal": {"mean": float, "std": float}
        For "beta": {"a": float, "b": float}
    beta_dist : str
        Distribution for beta (inverse temperature): "normal" or "beta".
    beta_params : dict
        For "normal": {"mean": float, "std": float}
        For "beta": {"a": float, "b": float}
    """
    rng = np.random.default_rng(seed)
    alpha_params = alpha_params or {"mean": 0.3, "std": 0.12}
    beta_params = beta_params or {"mean": 3.0, "std": 0.8}

    true_alphas = []
    true_betas = []
    participant_data = []

    for _ in range(n_subjects):
        if alpha_dist == "beta":
            alpha = _sample_beta_scaled(alpha_params["a"], alpha_params["b"], 0.01, 0.99, rng)
        else:
            alpha = _sample_truncnorm(alpha_params["mean"], alpha_params["std"], 0.01, 0.99, rng)

        if beta_dist == "beta":
            beta_val = _sample_beta_scaled(beta_params["a"], beta_params["b"], 0.1, 9.9, rng)
        else:
            beta_val = _sample_truncnorm(beta_params["mean"], beta_params["std"], 0.1, 9.9, rng)

        true_alphas.append(alpha)
        true_betas.append(beta_val)
        choices, rewards = simulate_subject(alpha, beta_val, n_trials, rng)
        participant_data.append([choices, rewards])

    return participant_data, np.array(true_alphas), np.array(true_betas)


# --- Tests ---


class TestParameterTransforms:
    def test_roundtrip_unit_interval(self):
        """to_bounded(to_unbounded(x)) should return x for [0, 1] bounds."""
        for x in [0.1, 0.3, 0.5, 0.7, 0.9]:
            y = to_unbounded(x, 0.0, 1.0)
            x_back = to_bounded(y, 0.0, 1.0)
            assert abs(x - x_back) < 1e-6, f"Roundtrip failed: {x} -> {y} -> {x_back}"

    def test_roundtrip_wider_interval(self):
        """Roundtrip for [0, 10] bounds."""
        for x in [0.5, 2.0, 5.0, 8.0, 9.5]:
            y = to_unbounded(x, 0.0, 10.0)
            x_back = to_bounded(y, 0.0, 10.0)
            assert abs(x - x_back) < 1e-6

    def test_midpoint_maps_to_zero(self):
        """Midpoint of [lb, ub] should map to 0 in unbounded space."""
        assert abs(to_unbounded(0.5, 0.0, 1.0)) < 1e-6
        assert abs(to_unbounded(5.0, 0.0, 10.0)) < 1e-6

    def test_vector_roundtrip(self):
        """params_to_bounded(params_to_unbounded(x)) should return x."""
        bounds = [(0.0, 1.0), (0.0, 10.0)]
        x = np.array([0.3, 3.0])
        y = params_to_unbounded(x, bounds)
        x_back = params_to_bounded(y, bounds)
        np.testing.assert_allclose(x, x_back, atol=1e-6)

    def test_jacobian_positive(self):
        """Log det Jacobian should be finite for reasonable values."""
        for y in [-2.0, -1.0, 0.0, 1.0, 2.0]:
            ldj = log_det_jacobian(y, 0.0, 1.0)
            assert np.isfinite(ldj)


class TestNumericalHessian:
    def test_quadratic(self):
        """Hessian of f(x) = 0.5 * x^T A x should be A."""
        A = np.array([[2.0, 0.5], [0.5, 3.0]])

        def f(x):
            return 0.5 * x @ A @ x

        H = numerical_hessian(f, np.array([1.0, 1.0]))
        np.testing.assert_allclose(H, A, atol=1e-4)

    def test_diagonal(self):
        """Hessian of sum of squares should be diagonal."""
        def f(x):
            return 2.0 * x[0]**2 + 5.0 * x[1]**2

        H = numerical_hessian(f, np.array([0.0, 0.0]))
        np.testing.assert_allclose(H, np.diag([4.0, 10.0]), atol=1e-4)


class TestSingleSubjectMAP:
    def test_recover_parameters(self):
        """MAP optimization should recover parameters for a single subject."""
        rng = np.random.default_rng(123)
        true_alpha, true_beta = 0.3, 3.0
        choices, rewards = simulate_subject(true_alpha, true_beta, 200, rng)

        bounds_list = [(0.0, 1.0), (0.0, 10.0)]
        prior_mean = np.zeros(2)
        prior_precision = np.ones(2) * 1.0  # weak prior

        theta_map, _, _, log_post = optimize_subject_map(
            simple_rl_model,
            [choices, rewards],
            prior_mean,
            prior_precision,
            bounds_list,
            prior_mean.copy(),
        )

        # Transform back to bounded space
        recovered = params_to_bounded(theta_map, bounds_list)
        assert abs(recovered[0] - true_alpha) < 0.15, (
            f"Alpha recovery failed: true={true_alpha}, recovered={recovered[0]}"
        )
        assert abs(recovered[1] - true_beta) < 2.0, (
            f"Beta recovery failed: true={true_beta}, recovered={recovered[1]}"
        )
        assert np.isfinite(log_post)


RECOVERY_CONFIGS = [
    pytest.param(
        "normal",
        {"mean": 0.3, "std": 0.12},
        "normal",
        {"mean": 3.0, "std": 0.8},
        id="normal_low_alpha_moderate_beta",
    ),
    pytest.param(
        "normal",
        {"mean": 0.6, "std": 0.15},
        "normal",
        {"mean": 5.0, "std": 1.2},
        id="normal_high_alpha_high_beta",
    ),
    pytest.param(
        "beta",
        {"a": 2.0, "b": 5.0},   # skewed low, mean ≈ 0.29
        "normal",
        {"mean": 3.0, "std": 0.8},
        id="beta_skewed_alpha_normal_beta",
    ),
    pytest.param(
        "beta",
        {"a": 5.0, "b": 2.0},   # skewed high, mean ≈ 0.71
        "beta",
        {"a": 3.0, "b": 3.0},   # symmetric, mean ≈ 0.5 → scaled to ~5.0
        id="beta_high_alpha_beta_symmetric_beta",
    ),
]


class TestHBIParameterRecovery:

    @pytest.mark.parametrize(
        "alpha_dist,alpha_params,beta_dist,beta_params",
        RECOVERY_CONFIGS,
    )
    def test_parameter_recovery(
        self, alpha_dist, alpha_params, beta_dist, beta_params
    ):
        """HBI should recover individual parameters (Pearson r) across distributions."""
        participant_data, true_alphas, true_betas = generate_synthetic_dataset(
            n_subjects=30,
            n_trials=300,
            alpha_dist=alpha_dist,
            alpha_params=alpha_params,
            beta_dist=beta_dist,
            beta_params=beta_params,
            seed=42,
        )
        spec = create_rl_model_spec()

        result = run_hbi_scipy(
            participant_data=participant_data,
            model_specs=[spec],
            max_iter=30,
            tol=1e-4,
            n_starts=2,
        )

        recovered = result.parameters[0]  # (2, N)
        alpha_r = np.corrcoef(true_alphas, recovered[0, :])[0, 1]
        beta_r = np.corrcoef(true_betas, recovered[1, :])[0, 1]

        assert alpha_r > 0.5, (
            f"Alpha Pearson r too low: {alpha_r:.3f} "
            f"(dist={alpha_dist}, params={alpha_params})"
        )
        assert beta_r > 0.5, (
            f"Beta Pearson r too low: {beta_r:.3f} "
            f"(dist={beta_dist}, params={beta_params})"
        )

    def test_hbi_convergence(self):
        """HBI should converge within max_iter."""
        participant_data, _, _ = generate_synthetic_dataset(
            n_subjects=30, n_trials=300, seed=42,
        )
        spec = create_rl_model_spec()

        result = run_hbi_scipy(
            participant_data=participant_data,
            model_specs=[spec],
            max_iter=50,
            tol=1e-4,
            n_starts=1,
        )

        assert result.converged, (
            f"HBI did not converge in {result.n_iterations} iterations"
        )

    def test_single_model_responsibilities(self):
        """With single model, all responsibilities should be 1.0."""
        participant_data, _, _ = generate_synthetic_dataset(
            n_subjects=30, n_trials=300, seed=42,
        )
        spec = create_rl_model_spec()

        result = run_hbi_scipy(
            participant_data=participant_data,
            model_specs=[spec],
            max_iter=10,
            n_starts=1,
        )

        np.testing.assert_allclose(result.responsibilities, 1.0, atol=1e-6)


class TestHBIModelComparison:
    def test_correct_model_preferred(self):
        """When data is generated from model A, HBI should prefer model A."""
        # Generate data from simple RL model
        participant_data, _, _ = generate_synthetic_dataset(
            n_subjects=20,
            n_trials=150,
            alpha_params={"mean": 0.3, "std": 0.12},
            beta_params={"mean": 3.0, "std": 0.8},
            seed=99,
        )

        # Model A: correct model (alpha, beta)
        spec_a = create_rl_model_spec()

        # Model B: misspecified - random responding (just beta, alpha fixed at 0)
        def random_model(choices, rewards, model_parameters):
            """Random responding model.
            Bounds:
            beta: [0, 10]
            """
            (beta,) = model_parameters
            n_trials = len(choices)
            nll = 0.0
            for t in range(n_trials):
                # No learning, always 50/50
                nll -= np.log(0.5)
            return nll

        spec_b = MockModelSpec(
            name="random",
            func=random_model,
            param_names=["beta"],
            bounds={"beta": [0.0, 10.0]},
        )

        result = run_hbi_scipy(
            participant_data=participant_data,
            model_specs=[spec_a, spec_b],
            max_iter=30,
            tol=1e-4,
            n_starts=1,
        )

        # Model A should have higher frequency and exceedance probability
        assert result.model_frequency[0] > result.model_frequency[1], (
            f"Model A frequency ({result.model_frequency[0]:.3f}) should be > "
            f"Model B ({result.model_frequency[1]:.3f})"
        )
        assert result.exceedance_prob[0] > 0.7, (
            f"Model A exceedance prob too low: {result.exceedance_prob[0]:.3f}"
        )


class TestRLModel:
    def test_model_returns_finite(self):
        """The test RL model should return finite NLL for valid parameters."""
        rng = np.random.default_rng(0)
        choices, rewards = simulate_subject(0.3, 3.0, 100, rng)
        nll = simple_rl_model(choices, rewards, np.array([0.3, 3.0]))
        assert np.isfinite(nll)
        assert nll > 0  # NLL should be positive

    def test_better_params_lower_nll(self):
        """True parameters should yield lower NLL than random ones."""
        rng = np.random.default_rng(0)
        true_alpha, true_beta = 0.3, 3.0
        choices, rewards = simulate_subject(true_alpha, true_beta, 500, rng)

        nll_true = simple_rl_model(choices, rewards, np.array([true_alpha, true_beta]))
        nll_random = simple_rl_model(choices, rewards, np.array([0.5, 0.1]))

        assert nll_true < nll_random, (
            f"True params NLL ({nll_true:.2f}) should be < random ({nll_random:.2f})"
        )
