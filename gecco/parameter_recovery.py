"""
Parameter recovery check for candidate cognitive models.

Before fitting a model to the full dataset, this module runs a quick
parameter recovery test: simulate synthetic data from the model using
known "true" parameters, fit the model to recover those parameters,
and reject models where recovery (Pearson r) is below a threshold.

Uses the "incremental NLL trick" to extract per-trial choice probabilities
from model functions that only return total NLL.
"""

import time
from abc import ABC, abstractmethod

import numpy as np
from scipy.optimize import minimize
from scipy.stats import pearsonr
from rich.console import Console

from gecco.offline_evaluation.utils import ModelSpec

console = Console()


# ============================================================
# Task simulator interface
# ============================================================

class TaskSimulator(ABC):
    """Abstract base class for task-specific data simulation."""

    @abstractmethod
    def simulate_subject(self, model_func, true_params, n_trials):
        """Simulate one subject's behavioral data.

        Uses the model function to generate choice probabilities
        (via the incremental NLL trick) and the task environment
        to generate environmental responses.

        Parameters
        ----------
        model_func : callable
            Model function with signature (col1, col2, ..., params) -> NLL.
        true_params : np.array
            True parameter values for simulation.
        n_trials : int
            Number of trials to simulate.

        Returns
        -------
        list of np.array
            Data columns in the order expected by model_func.
        """

    @abstractmethod
    def get_input_columns(self):
        """Return ordered list of input column names."""


def _extract_choice_probs(model_func, data_arrays, trial_idx,
                          choice_col_idx, n_options, params):
    """Extract choice probabilities using the incremental NLL trick.

    Evaluates the model with each possible choice value at `trial_idx`
    and normalises the resulting likelihoods to obtain probabilities.

    Parameters
    ----------
    model_func : callable
        Model function: (*data_columns, params) -> NLL.
    data_arrays : list of np.array
        Current data arrays (will be sliced to trial_idx+1).
    trial_idx : int
        Which trial to extract probabilities for.
    choice_col_idx : int
        Index into data_arrays for the choice column to vary.
    n_options : int
        Number of discrete choice options (e.g. 2 for binary).
    params : array-like
        Model parameters.

    Returns
    -------
    np.array
        Normalised choice probabilities of shape (n_options,).
    """
    log_likes = np.zeros(n_options)

    for opt in range(n_options):
        data_copy = [col[:trial_idx + 1].copy() for col in data_arrays]
        data_copy[choice_col_idx][trial_idx] = opt
        nll = model_func(*data_copy, params)
        log_likes[opt] = -nll  # convert NLL to log-likelihood

    # Subtract max for numerical stability before exp
    log_likes -= np.max(log_likes)
    probs = np.exp(log_likes)
    prob_sum = probs.sum()
    if prob_sum <= 0 or not np.isfinite(prob_sum):
        # Fall back to uniform if numerical issues
        return np.ones(n_options) / n_options
    probs /= prob_sum
    return probs


# ============================================================
# Two-step task simulator
# ============================================================

class TwoStepSimulator(TaskSimulator):
    """Simulator for the two-step decision task.

    Task structure:
    - Stage 1: Choose between 2 options (spaceships)
    - Transition: Probabilistic mapping from choice to state (planet)
    - Stage 2: Choose between 2 options (aliens) on the reached state
    - Reward: Binary, from drifting reward probabilities

    Data columns (in model function order):
    action_1, state, action_2, reward
    """

    def __init__(self, transition_probs=None, reward_drift_sd=0.025,
                 reward_bounds=(0.25, 0.75)):
        if transition_probs is not None:
            self.transition_probs = np.asarray(transition_probs)
        else:
            self.transition_probs = np.array([[0.7, 0.3], [0.3, 0.7]])
        self.reward_drift_sd = reward_drift_sd
        self.reward_bounds = reward_bounds

    def get_input_columns(self):
        return ["choice_1", "state", "choice_2", "reward"]

    def _generate_reward_walks(self, n_trials):
        """Generate drifting reward probabilities (Gaussian random walk).

        Returns
        -------
        np.array of shape (2, 2, n_trials)
            reward_probs[state][action_2][trial]
        """
        lb, ub = self.reward_bounds
        reward_probs = np.zeros((2, 2, n_trials))
        for s in range(2):
            for a in range(2):
                reward_probs[s, a, 0] = np.random.uniform(lb, ub)
                for t in range(1, n_trials):
                    reward_probs[s, a, t] = np.clip(
                        reward_probs[s, a, t - 1]
                        + np.random.normal(0, self.reward_drift_sd),
                        lb, ub,
                    )
        return reward_probs

    def simulate_subject(self, model_func, true_params, n_trials):
        """Simulate one subject using the incremental NLL trick.

        Per trial:
        1. Extract p(choice_1) via NLL trick -> sample choice_1
        2. Sample state from transition_probs[choice_1]
        3. Extract p(choice_2 | state) via NLL trick -> sample choice_2
        4. Sample reward from drifting reward_probs[state][choice_2]
        """
        # Column indices in model function signature
        COL_A1 = 0   # choice_1 / action_1
        COL_S = 1     # state
        COL_A2 = 2    # choice_2 / action_2
        COL_R = 3     # reward

        # Initialize data arrays
        action_1 = np.zeros(n_trials, dtype=int)
        state = np.zeros(n_trials, dtype=int)
        action_2 = np.zeros(n_trials, dtype=int)
        reward = np.zeros(n_trials, dtype=int)

        reward_probs = self._generate_reward_walks(n_trials)

        data_arrays = [action_1, state, action_2, reward]

        for t in range(n_trials):
            # --- Stage 1: extract p(choice_1) and sample ---
            probs_a1 = _extract_choice_probs(
                model_func, data_arrays, t, COL_A1, 2, true_params
            )
            a1 = np.random.choice(2, p=probs_a1)
            action_1[t] = a1

            # --- Environment: determine state from transition ---
            s = np.random.choice(2, p=self.transition_probs[a1])
            state[t] = s

            # --- Stage 2: extract p(choice_2 | state) and sample ---
            probs_a2 = _extract_choice_probs(
                model_func, data_arrays, t, COL_A2, 2, true_params
            )
            a2 = np.random.choice(2, p=probs_a2)
            action_2[t] = a2

            # --- Environment: determine reward ---
            r = np.random.binomial(1, reward_probs[s, a2, t])
            reward[t] = r

        return [action_1, state, action_2, reward]


# ============================================================
# Simulator registry
# ============================================================

_SIMULATOR_REGISTRY = {
    "two_step": TwoStepSimulator,
}


def get_simulator(recovery_cfg):
    """Create a TaskSimulator from config.

    Parameters
    ----------
    recovery_cfg : SimpleNamespace
        The parameter_recovery config section. Must have a `simulator`
        field naming a built-in simulator. May have `simulator_config`
        with kwargs for the simulator constructor.

    Returns
    -------
    TaskSimulator
    """
    name = getattr(recovery_cfg, "simulator", "two_step")
    cls = _SIMULATOR_REGISTRY.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown simulator '{name}'. "
            f"Available: {list(_SIMULATOR_REGISTRY.keys())}"
        )
    kwargs = {}
    sim_cfg = getattr(recovery_cfg, "simulator_config", None)
    if sim_cfg is not None:
        kwargs = vars(sim_cfg) if hasattr(sim_cfg, "__dict__") else dict(sim_cfg)
    return cls(**kwargs)


# ============================================================
# Parameter recovery checker
# ============================================================

class ParameterRecoveryChecker:
    """Checks whether a model's parameters can be recovered from
    simulated data.

    For each simulated subject:
    1. Sample true parameters from uniform(bounds)
    2. Simulate behavioral data using the TaskSimulator
    3. Fit the model to simulated data using L-BFGS-B
    4. Compare recovered vs true parameters

    The model passes if the mean Pearson r across parameters
    exceeds the threshold.
    """

    def __init__(self, simulator, n_subjects=50, n_trials=100,
                 threshold=0.5, n_fitting_starts=3):
        self.simulator = simulator
        self.n_subjects = n_subjects
        self.n_trials = n_trials
        self.threshold = threshold
        self.n_fitting_starts = n_fitting_starts

    def check(self, spec):
        """Run parameter recovery check on a model.

        Parameters
        ----------
        spec : ModelSpec
            Compiled model specification with func, param_names, bounds.

        Returns
        -------
        dict
            {
                "passed": bool,
                "mean_r": float,
                "per_param_r": dict[str, float],
                "n_successful": int,
                "elapsed_seconds": float,
            }
        """
        t0 = time.time()
        param_names = spec.param_names
        bounds_dict = spec.bounds
        n_params = len(param_names)

        if n_params == 0:
            return {
                "passed": False,
                "mean_r": 0.0,
                "per_param_r": {},
                "n_successful": 0,
                "elapsed_seconds": time.time() - t0,
            }

        # Build ordered bounds list
        bounds_list = [bounds_dict[p] for p in param_names]

        true_params_all = []
        recovered_params_all = []

        for subj in range(self.n_subjects):
            try:
                # Sample true parameters
                true_params = self._sample_params(bounds_list)

                # Simulate data
                sim_data = self.simulator.simulate_subject(
                    spec.func, true_params, self.n_trials
                )

                # Fit model to simulated data
                recovered = self._fit_subject(
                    spec.func, sim_data, bounds_list
                )

                if recovered is not None:
                    true_params_all.append(true_params)
                    recovered_params_all.append(recovered)

            except Exception:
                # Model crashed during simulation or fitting — skip subject
                continue

        n_successful = len(true_params_all)

        # Need at least 5 successful subjects for meaningful correlation
        if n_successful < 5:
            return {
                "passed": False,
                "mean_r": 0.0,
                "per_param_r": {p: 0.0 for p in param_names},
                "n_successful": n_successful,
                "elapsed_seconds": time.time() - t0,
            }

        true_arr = np.array(true_params_all)       # (n_successful, n_params)
        rec_arr = np.array(recovered_params_all)    # (n_successful, n_params)

        per_param_r = {}
        for p_idx, p_name in enumerate(param_names):
            true_col = true_arr[:, p_idx]
            rec_col = rec_arr[:, p_idx]

            # Check for zero variance (all same value after recovery)
            if np.std(true_col) < 1e-10 or np.std(rec_col) < 1e-10:
                per_param_r[p_name] = 0.0
            else:
                r, _ = pearsonr(true_col, rec_col)
                per_param_r[p_name] = float(r) if np.isfinite(r) else 0.0

        mean_r = float(np.mean(list(per_param_r.values())))
        passed = mean_r >= self.threshold

        elapsed = time.time() - t0
        status_color = "green" if passed else "yellow"
        status_text = "PASSED" if passed else "FAILED"
        console.print(
            f"    [{status_color}]Recovery {status_text}[/] "
            f"(mean r={mean_r:.2f}, {n_successful}/{self.n_subjects} subjects, "
            f"{elapsed:.1f}s)"
        )
        # Per-parameter breakdown
        param_strs = [
            f"{name}: r={r_val:.2f}" for name, r_val in per_param_r.items()
        ]
        console.print(f"    [dim]Per-parameter: {', '.join(param_strs)}[/]")

        return {
            "passed": passed,
            "mean_r": mean_r,
            "per_param_r": per_param_r,
            "n_successful": n_successful,
            "elapsed_seconds": elapsed,
        }

    @staticmethod
    def _sample_params(bounds_list):
        """Sample parameters uniformly from bounds."""
        return np.array([
            np.random.uniform(lb, ub) for lb, ub in bounds_list
        ])

    def _fit_subject(self, model_func, sim_data, bounds_list):
        """Fit model to a single subject's simulated data.

        Uses L-BFGS-B with multiple random starts.

        Returns
        -------
        np.array or None
            Recovered parameter values, or None if all starts failed.
        """
        best_nll = np.inf
        best_params = None

        for _ in range(self.n_fitting_starts):
            x0 = self._sample_params(bounds_list)
            try:
                result = minimize(
                    lambda params: model_func(*sim_data, params),
                    x0=x0,
                    method="L-BFGS-B",
                    bounds=bounds_list,
                    options={"maxiter": 200, "ftol": 1e-8},
                )
                if result.fun < best_nll and np.isfinite(result.fun):
                    best_nll = result.fun
                    best_params = result.x.copy()
            except Exception:
                continue

        return best_params
