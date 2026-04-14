"""
Parameter recovery check for candidate cognitive models.

Before fitting a model to the full dataset, this module runs a quick
parameter recovery test: simulate synthetic data from the model using
known "true" parameters, fit the model to recover those parameters,
and reject models where recovery (Pearson r) is below a threshold.

Uses the "incremental NLL trick" to extract per-trial choice probabilities
from model functions that only return total NLL.

Simulation is parallelised across subjects. Fitting uses HBI (the same
hierarchical Bayesian inference used for the real data), which has its
own internal parallelisation across subjects and cores.
"""

import os
import time
from abc import ABC, abstractmethod

import numpy as np
from scipy.stats import pearsonr
from rich.console import Console

from gecco.offline_evaluation.utils import ModelSpec
from gecco.utils import TimestampedConsole

console = TimestampedConsole()


# ============================================================
# Task simulator interface
# ============================================================

class TaskSimulator(ABC):
    """Abstract base class for task-specific data simulation."""

    @abstractmethod
    def simulate_subject(self, model_func, true_params, n_trials, rng=None):
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
        rng : np.random.Generator, optional
            Random number generator for reproducibility.

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
    end = trial_idx + 1

    # Create slices (views, no allocation) rather than full copies
    sliced = [col[:end] for col in data_arrays]

    # Save the original value at the trial index
    original_val = data_arrays[choice_col_idx][trial_idx]

    for opt in range(n_options):
        # Mutate the underlying array; sliced[choice_col_idx] is a view so it sees the change
        data_arrays[choice_col_idx][trial_idx] = opt
        nll = model_func(*sliced, params)
        log_likes[opt] = -nll  # convert NLL to log-likelihood

    # Restore the original value
    data_arrays[choice_col_idx][trial_idx] = original_val

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

    def _generate_reward_walks(self, n_trials, rng):
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
                reward_probs[s, a, 0] = rng.uniform(lb, ub)
                for t in range(1, n_trials):
                    reward_probs[s, a, t] = np.clip(
                        reward_probs[s, a, t - 1]
                        + rng.normal(0, self.reward_drift_sd),
                        lb, ub,
                    )
        return reward_probs

    def simulate_subject(self, model_func, true_params, n_trials, rng=None):
        """Simulate one subject using the incremental NLL trick.

        Per trial:
        1. Extract p(choice_1) via NLL trick -> sample choice_1
        2. Sample state from transition_probs[choice_1]
        3. Extract p(choice_2 | state) via NLL trick -> sample choice_2
        4. Sample reward from drifting reward_probs[state][choice_2]
        """
        if rng is None:
            rng = np.random.default_rng()

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

        reward_probs = self._generate_reward_walks(n_trials, rng)

        data_arrays = [action_1, state, action_2, reward]

        for t in range(n_trials):
            # --- Stage 1: extract p(choice_1) and sample ---
            probs_a1 = _extract_choice_probs(
                model_func, data_arrays, t, COL_A1, 2, true_params
            )
            a1 = int(rng.random() < probs_a1[1])
            action_1[t] = a1

            # --- Environment: determine state from transition ---
            s = int(rng.random() < self.transition_probs[a1][1])
            state[t] = s

            # --- Stage 2: extract p(choice_2 | state) and sample ---
            probs_a2 = _extract_choice_probs(
                model_func, data_arrays, t, COL_A2, 2, true_params
            )
            a2 = int(rng.random() < probs_a2[1])
            action_2[t] = a2

            # --- Environment: determine reward ---
            r = rng.binomial(1, reward_probs[s, a2, t])
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
# Worker function for parallel simulation
# ============================================================

def _simulate_one_subject(model_func, simulator, n_trials, bounds_list, seed):
    """Simulate a single subject in a worker process.

    Returns
    -------
    tuple (true_params, sim_data) or None on failure.
    """
    rng = np.random.default_rng(seed)

    # Sample true parameters
    true_params = np.array([
        rng.uniform(lb, ub) for lb, ub in bounds_list
    ])

    # Simulate data
    sim_data = simulator.simulate_subject(
        model_func, true_params, n_trials, rng=rng
    )

    return (true_params, sim_data)


# ============================================================
# Parameter recovery checker
# ============================================================

class ParameterRecoveryChecker:
    """Checks whether a model's parameters can be recovered from
    simulated data.

    For each simulated subject:
    1. Sample true parameters from uniform(bounds)
    2. Simulate behavioral data using the TaskSimulator

    Then fit ALL simulated subjects as a group using HBI (the same
    hierarchical Bayesian inference used for real data evaluation).
    This tests recovery under the actual fitting procedure and
    leverages HBI's built-in parallelisation across subjects.

    The model passes if the mean Pearson r across parameters
    exceeds the threshold.
    """

    def __init__(self, simulator, n_subjects=50, n_trials=100,
                 threshold=0.5, n_fitting_starts=3, n_jobs=-1):
        self.simulator = simulator
        self.n_subjects = n_subjects
        self.n_trials = n_trials
        self.threshold = threshold
        self.n_fitting_starts = n_fitting_starts
        self.n_jobs = n_jobs

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

        # --- Step 1: Simulate all subjects (parallelised) ---
        seed_seq = np.random.SeedSequence()
        seeds = seed_seq.spawn(self.n_subjects)

        console.print(
            f"    [dim]Simulating {self.n_subjects} subjects "
            f"({self.n_trials} trials each)...[/]"
        )
        t_sim = time.time()

        sim_results, sim_error = self._simulate_all(spec.func, bounds_list, seeds)

        # Collect successful simulations
        true_params_all = []
        participant_data = []
        for res in sim_results:
            if res is not None:
                true_params_all.append(res[0])
                participant_data.append(res[1])

        n_simulated = len(true_params_all)
        console.print(
            f"    [dim]Simulation complete: {n_simulated}/{self.n_subjects} "
            f"subjects ({time.time() - t_sim:.1f}s)[/]"
        )

        if n_simulated < 5:
            return {
                "passed": False,
                "mean_r": 0.0,
                "per_param_r": {p: 0.0 for p in param_names},
                "n_successful": n_simulated,
                "elapsed_seconds": time.time() - t0,
                "simulation_error": sim_error,
            }

        # --- Step 2: Fit all subjects as a group using HBI ---
        console.print(
            f"    [dim]Fitting {n_simulated} simulated subjects with HBI...[/]"
        )
        t_fit = time.time()

        try:
            recovered_params = self._fit_group_hbi(
                spec, participant_data
            )
        except Exception as e:
            console.print(f"    [yellow]HBI fitting failed: {e}[/]")
            return {
                "passed": False,
                "mean_r": 0.0,
                "per_param_r": {p: 0.0 for p in param_names},
                "n_successful": 0,
                "elapsed_seconds": time.time() - t0,
            }

        console.print(
            f"    [dim]HBI fitting complete ({time.time() - t_fit:.1f}s)[/]"
        )

        # --- Step 3: Compute Pearson r per parameter ---
        true_arr = np.array(true_params_all)   # (N, n_params)
        rec_arr = recovered_params              # (N, n_params)

        per_param_r = {}
        for p_idx, p_name in enumerate(param_names):
            true_col = true_arr[:, p_idx]
            rec_col = rec_arr[:, p_idx]

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
            f"(mean r={mean_r:.2f}, {n_simulated}/{self.n_subjects} subjects, "
            f"{elapsed:.1f}s)"
        )
        param_strs = [
            f"{name}: r={r_val:.2f}" for name, r_val in per_param_r.items()
        ]
        console.print(f"    [dim]Per-parameter: {', '.join(param_strs)}[/]")

        return {
            "passed": passed,
            "mean_r": mean_r,
            "per_param_r": per_param_r,
            "n_successful": n_simulated,
            "elapsed_seconds": elapsed,
            "simulation_error": None,
        }

    def _simulate_all(self, model_func, bounds_list, seeds):
        """Simulate all subjects, parallelised across cores.

        Returns
        -------
        tuple[list, str | None]
            (results, first_error_str) where first_error_str is the first
            exception message encountered, or None if all succeeded.
        """
        n_workers = self._resolve_n_jobs()

        if n_workers > 1:
            return self._simulate_parallel(
                model_func, bounds_list, seeds, n_workers
            )
        else:
            return self._simulate_sequential(
                model_func, bounds_list, seeds
            )

    def _simulate_sequential(self, model_func, bounds_list, seeds):
        """Simulate all subjects sequentially."""
        results = []
        first_error = None
        for seed in seeds:
            try:
                res = _simulate_one_subject(
                    model_func, self.simulator, self.n_trials,
                    bounds_list, seed,
                )
                results.append(res)
            except Exception as e:
                if first_error is None:
                    first_error = f"{type(e).__name__}: {e}"
                    console.print(f"    [yellow]Simulation error: {e}[/]")
                results.append(None)
        return results, first_error

    def _simulate_parallel(self, model_func, bounds_list, seeds, n_workers):
        """Simulate all subjects in parallel.

        Uses loky (which handles exec'd functions via cloudpickle).
        Falls back to sequential if parallelisation fails.
        """
        from concurrent.futures import as_completed

        try:
            from loky import get_reusable_executor
            executor = get_reusable_executor(max_workers=n_workers)
        except ImportError:
            from concurrent.futures import ProcessPoolExecutor
            executor = ProcessPoolExecutor(max_workers=n_workers)

        # Submit first job as a test — if pickling fails, fall back
        # to sequential rather than failing all 50 silently
        test_fut = executor.submit(
            _simulate_one_subject,
            model_func, self.simulator, self.n_trials,
            bounds_list, seeds[0],
        )
        try:
            test_result = test_fut.result(timeout=120)
        except Exception as e:
            parallel_err = f"{type(e).__name__}: {e}"
            console.print(
                f"    [yellow]Parallel simulation failed ({parallel_err}), "
                f"falling back to sequential...[/]"
            )
            return self._simulate_sequential(model_func, bounds_list, seeds)

        # First subject succeeded — submit the rest
        futures = {}
        results = [None] * len(seeds)
        results[0] = test_result

        for idx, seed in enumerate(seeds[1:], start=1):
            fut = executor.submit(
                _simulate_one_subject,
                model_func, self.simulator, self.n_trials,
                bounds_list, seed,
            )
            futures[fut] = idx

        first_error = None
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                results[idx] = fut.result()
            except Exception as e:
                if first_error is None:
                    first_error = f"{type(e).__name__}: {e}"
                    console.print(f"    [yellow]Simulation error (subject {idx}): {e}[/]")
                results[idx] = None

        return results, first_error

    def _fit_group_hbi(self, spec, participant_data):
        """Fit all simulated subjects as a group using HBI.

        Returns
        -------
        np.array of shape (N, n_params)
            Recovered parameters for each subject (bounded space).
        """
        from gecco.model_fitting.hbi_scipy import run_hbi_scipy

        hbi_result = run_hbi_scipy(
            participant_data=participant_data,
            model_specs=[spec],
            max_iter=50,
            tol=1e-5,
            n_starts=self.n_fitting_starts,
            n_jobs=self.n_jobs,
        )

        # hbi_result.parameters[0] is (D, N) for model 0
        # Transpose to (N, D) to match true_params_all layout
        return hbi_result.parameters[0].T

    def _resolve_n_jobs(self):
        """Resolve n_jobs respecting SLURM allocation."""
        n_jobs = self.n_jobs
        if n_jobs == -1:
            slurm_cpus = (
                os.environ.get("SLURM_CPUS_PER_TASK")
                or os.environ.get("SLURM_CPUS_ON_NODE")
            )
            if slurm_cpus is not None:
                n_jobs = int(slurm_cpus)
            else:
                n_jobs = os.cpu_count() or 1
        return min(n_jobs, self.n_subjects)
