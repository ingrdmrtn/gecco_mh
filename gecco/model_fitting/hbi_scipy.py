"""
Hierarchical Bayesian Inference (HBI) using scipy optimization.

Implements the CBM (Computational Bayesian Model comparison) algorithm
for hierarchical parameter estimation and model comparison, using scipy
for optimization and numerical Hessians (no JAX dependency).

Works with arbitrary callable model functions (e.g., LLM-generated models)
that follow the ModelSpec interface.
"""

import os
import numpy as np
import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional, Callable
from scipy.optimize import minimize
from scipy.special import psi, gammaln
from concurrent.futures import as_completed
from loky import get_reusable_executor

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, MofNCompleteColumn
from rich.table import Table
from rich.panel import Panel

from gecco.utils import TimestampedConsole

logger = logging.getLogger("hbi_scipy")
console = TimestampedConsole()


# --- Data Structures ---


@dataclass
class GaussianGammaDistribution:
    """Group-level prior/posterior for mean and precision of parameters."""
    a: np.ndarray       # mean, shape (D,)
    beta: float         # pseudo-count for mean precision
    sigma: np.ndarray   # scale of Gamma, shape (D,)
    nu: float           # shape of Gamma
    Etau: np.ndarray    # E[tau] = nu / sigma
    Elogtau: np.ndarray # E[log tau] = psi(nu) - log(sigma)
    logG: float         # log normalizing constant


@dataclass
class DirichletDistribution:
    """Model frequency distribution."""
    limInf: bool
    alpha: np.ndarray   # concentration parameters, shape (K,)
    Elogm: np.ndarray   # E[log m_k]
    logC: float         # log normalizing constant


@dataclass
class HBIResult:
    """Output of HBI fitting."""
    parameters: List[np.ndarray]                          # per-model params in bounded space, each (D_k, N)
    parameters_unbounded: List[np.ndarray]                # per-model params in unbounded space
    responsibilities: np.ndarray                          # (K, N) responsibility matrix
    group_mean: List[np.ndarray]                          # per-model group means in unbounded space
    group_precision: List[np.ndarray]                     # per-model group precisions
    model_frequency: np.ndarray                           # (K,) model frequencies
    exceedance_prob: np.ndarray                           # (K,) exceedance probabilities
    protected_exceedance_prob: Optional[np.ndarray] = None
    per_subject_nll: Optional[np.ndarray] = None          # (K, N) NLL at MAP estimates
    converged: bool = False
    n_iterations: int = 0


# --- Parameter Transforms ---


def to_unbounded(x, lb, ub):
    """Transform from bounded [lb, ub] to unbounded (-inf, inf) using logit."""
    x = np.asarray(x, dtype=np.float64)
    x_01 = (x - lb) / (ub - lb)
    x_01 = np.clip(x_01, 1e-8, 1.0 - 1e-8)
    return np.log(x_01 / (1.0 - x_01))


def to_bounded(y, lb, ub):
    """Transform from unbounded to bounded [lb, ub] using sigmoid."""
    y = np.asarray(y, dtype=np.float64)
    x_01 = 1.0 / (1.0 + np.exp(-y))
    return lb + x_01 * (ub - lb)


def log_det_jacobian(y, lb, ub):
    """Log |det(dx/dy)| for the sigmoid transform. Used as Jacobian correction."""
    y = np.asarray(y, dtype=np.float64)
    s = 1.0 / (1.0 + np.exp(-y))
    return np.log(ub - lb) + np.log(s) + np.log(1.0 - s)


def params_to_unbounded(theta_bounded, bounds_list):
    """Transform a vector of bounded parameters to unbounded space."""
    return np.array([
        float(to_unbounded(theta_bounded[i], lb, ub))
        for i, (lb, ub) in enumerate(bounds_list)
    ])


def params_to_bounded(theta_unbounded, bounds_list):
    """Transform a vector of unbounded parameters to bounded space."""
    return np.array([
        float(to_bounded(theta_unbounded[i], lb, ub))
        for i, (lb, ub) in enumerate(bounds_list)
    ])


# --- Numerical Hessian ---


def numerical_hessian(f, x, eps=1e-5):
    """Compute Hessian via central finite differences."""
    n = len(x)
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            x_pp = x.copy(); x_pp[i] += eps; x_pp[j] += eps
            x_pm = x.copy(); x_pm[i] += eps; x_pm[j] -= eps
            x_mp = x.copy(); x_mp[i] -= eps; x_mp[j] += eps
            x_mm = x.copy(); x_mm[i] -= eps; x_mm[j] -= eps
            H[i, j] = (f(x_pp) - f(x_pm) - f(x_mp) + f(x_mm)) / (4 * eps**2)
            H[j, i] = H[i, j]
    return H


# --- Individual MAP Optimization ---


def optimize_subject_map(
    nll_func: Callable,
    data_columns: List[np.ndarray],
    prior_mean: np.ndarray,
    prior_precision: np.ndarray,
    bounds_list: List[Tuple[float, float]],
    theta_init: np.ndarray,
    maxiter: int = 200,
    ftol: float = 1e-10,
    compute_hessian: bool = True,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Find MAP estimate for a single subject under a single model.

    Args:
        nll_func: Model NLL function: nll_func(*data_columns, params) -> float
        data_columns: List of arrays, one per input column
        prior_mean: Prior mean in unbounded space, shape (D,)
        prior_precision: Diagonal prior precision in unbounded space, shape (D,)
        bounds_list: [(lb, ub), ...] for each parameter
        theta_init: Starting point in unbounded space, shape (D,)
        maxiter: Maximum optimizer iterations (lower for warm-start EM steps)
        ftol: Function tolerance for optimizer convergence
        compute_hessian: Whether to compute the numerical Hessian (can skip for speed)

    Returns:
        theta_map: MAP estimate in unbounded space
        hessian_inv_diag: Diagonal of inverse Hessian at MAP
        log_det_hessian: log |H| at MAP
        log_posterior: log posterior at MAP (log_lik + log_prior + log_jacobian)
    """
    D = len(theta_init)

    def neg_log_posterior(theta_unbounded):
        # Transform to bounded space
        theta_bounded = params_to_bounded(theta_unbounded, bounds_list)

        # NLL from model
        try:
            nll = float(nll_func(*data_columns, theta_bounded))
        except Exception:
            return 1e10

        if not np.isfinite(nll):
            return 1e10

        # Gaussian prior in unbounded space
        diff = theta_unbounded - prior_mean
        neg_log_prior = 0.5 * np.sum(diff**2 * prior_precision)

        # Jacobian correction (add because this is negative log posterior)
        jac = sum(
            float(log_det_jacobian(theta_unbounded[i], lb, ub))
            for i, (lb, ub) in enumerate(bounds_list)
        )

        return nll + neg_log_prior - jac

    # Optimize
    result = minimize(
        neg_log_posterior,
        theta_init,
        method="L-BFGS-B",
        options={"maxiter": maxiter, "ftol": ftol},
    )
    theta_map = result.x

    if compute_hessian:
        # Compute Hessian at MAP
        H = numerical_hessian(neg_log_posterior, theta_map)

        # Regularize and invert
        H_reg = H + np.eye(D) * 1e-6
        try:
            H_inv = np.linalg.inv(H_reg)
            sign, logdet = np.linalg.slogdet(H_reg)
            if sign <= 0:
                logdet = D * np.log(1e-6)
        except np.linalg.LinAlgError:
            H_inv = np.eye(D) * 1e6
            logdet = D * np.log(1e-6)

        hessian_inv_diag = np.diag(H_inv)
    else:
        hessian_inv_diag = None
        logdet = None

    log_posterior = -neg_log_posterior(theta_map)

    return theta_map, hessian_inv_diag, logdet, log_posterior


# --- Parallel Helpers ---


def _fit_subject_init(nll_func, data_cols, prior_mean, prior_prec, bounds_list, n_starts, Dk):
    """Fit a single subject during initialization (multi-start)."""
    best_log_post = -np.inf
    best_stats = (np.zeros(Dk), np.zeros(Dk), 0.0, -np.inf)

    start_points = [prior_mean.copy()]
    for _ in range(n_starts - 1):
        start_points.append(np.random.randn(Dk) * 0.5)

    for start_val in start_points:
        th_map, h_inv_diag, logdet, log_post = optimize_subject_map(
            nll_func, data_cols, prior_mean, prior_prec, bounds_list, start_val,
        )
        if log_post > best_log_post:
            best_log_post = log_post
            best_stats = (th_map, h_inv_diag, logdet, log_post)

    return best_stats


def _fit_subject_em(nll_func, data_cols, prior_mean, prior_precision, bounds_list,
                    th_init, maxiter=50, ftol=1e-6, compute_hessian=True):
    """Fit a single subject during an EM iteration."""
    return optimize_subject_map(
        nll_func, data_cols, prior_mean, prior_precision, bounds_list, th_init,
        maxiter=maxiter, ftol=ftol, compute_hessian=compute_hessian,
    )


def _fit_subjects_batch_init(nll_func, batch_data, batch_indices, prior_mean,
                             prior_prec, bounds_list, n_starts, Dk):
    """Fit a batch of subjects during initialization (multi-start)."""
    results = {}
    for i, n in enumerate(batch_indices):
        results[n] = _fit_subject_init(
            nll_func, batch_data[i], prior_mean, prior_prec, bounds_list, n_starts, Dk,
        )
    return results


def _fit_subjects_batch_em(nll_func, batch_data, batch_indices, prior_mean,
                           prior_precision, bounds_list, batch_theta_inits,
                           maxiter=50, ftol=1e-6, compute_hessian=True):
    """Fit a batch of subjects during an EM iteration."""
    results = {}
    for i, n in enumerate(batch_indices):
        results[n] = optimize_subject_map(
            nll_func, batch_data[i], prior_mean, prior_precision, bounds_list,
            batch_theta_inits[i], maxiter=maxiter, ftol=ftol,
            compute_hessian=compute_hessian,
        )
    return results


# --- HBI Update Functions ---


def hbi_sumstats(
    responsibilities: np.ndarray,
    theta: List[np.ndarray],
    Ainvdiag: List[np.ndarray],
):
    """Calculate sufficient statistics weighted by responsibilities."""
    K, _ = responsibilities.shape
    weighted_means = [None] * K
    weighted_variances = [None] * K
    effective_counts = np.zeros(K)

    for k in range(K):
        resp_k = responsibilities[k, :]
        Nk = float(resp_k.sum()) + 1e-16
        effective_counts[k] = Nk
        theta_k = theta[k]      # (Dk, N)
        Ainvdiag_k = Ainvdiag[k]

        mean_k = np.sum(theta_k * resp_k[np.newaxis, :], axis=1, keepdims=True) / Nk
        term1 = (
            np.sum((theta_k**2 + Ainvdiag_k) * resp_k[np.newaxis, :], axis=1, keepdims=True) / Nk
        )
        var_k = term1 - mean_k**2
        var_k = np.maximum(var_k, 1e-10)

        weighted_means[k] = mean_k
        weighted_variances[k] = var_k

    return effective_counts, weighted_means, weighted_variances


def hbi_qmutau(
    prior_mu_tau: List[GaussianGammaDistribution],
    effective_counts,
    weighted_means,
    weighted_variances,
):
    """Update variational posterior q(mu, tau) for group-level mean and precision."""
    K = len(effective_counts)
    qmutau_out = []

    for k in range(K):
        a0 = prior_mu_tau[k].a
        beta0 = prior_mu_tau[k].beta
        nu0 = prior_mu_tau[k].nu
        sigma0 = prior_mu_tau[k].sigma

        Nk = effective_counts[k]
        tb = weighted_means[k]    # (Dk, 1)
        Sd = weighted_variances[k]

        beta = beta0 + Nk
        a = (beta0 * a0[:, np.newaxis] + Nk * tb) / beta
        a = a.flatten()

        nu = nu0 + 0.5 * Nk

        diff = tb.flatten() - a0.flatten()
        term_sd = Nk * Sd.flatten()
        sigma = sigma0 + 0.5 * (
            term_sd + (Nk * beta0 / (Nk + beta0)) * diff**2
        )
        sigma = np.maximum(sigma, 1e-10)

        Elogtau = psi(nu) - np.log(sigma)
        Etau = nu / sigma
        logG = np.sum(-gammaln(nu) + nu * np.log(sigma))

        qmutau_out.append(
            GaussianGammaDistribution(
                a=a, beta=beta, sigma=sigma, nu=nu,
                Etau=Etau, Elogtau=Elogtau, logG=logG,
            )
        )
    return qmutau_out


def hbi_qm(prior_model_freq: DirichletDistribution, effective_counts):
    """Update variational posterior q(m) for model frequencies."""
    alpha0 = prior_model_freq.alpha
    alpha = alpha0 + effective_counts
    alpha_star = np.sum(alpha)

    Elogm = psi(alpha) - psi(alpha_star)
    logC = gammaln(alpha_star) - np.sum(gammaln(alpha))

    return DirichletDistribution(
        limInf=False, alpha=alpha, Elogm=Elogm, logC=logC
    )


def hbi_qHZ(
    post_mu_tau,
    post_model_freq,
    qh_log_posterior,
    qh_logdet,
    subject_params_unbounded,
    hessian_inv_diag_per_model,
):
    """Update responsibilities r_kn (q(z_n=k)) using Laplace approximation."""
    K, N = qh_log_posterior.shape
    log_rho = np.zeros((K, N))

    for k in range(K):
        Dk = len(post_mu_tau[k].a)
        ElogdetT = np.sum(post_mu_tau[k].Elogtau)
        beta = post_mu_tau[k].beta
        Etau = post_mu_tau[k].Etau

        logdetET = np.sum(np.log(Etau))
        lambda_val = 0.5 * ElogdetT - 0.5 * logdetET - 0.5 * Dk / beta
        shift = 0.5 * Dk * np.log(2 * np.pi) + lambda_val + post_model_freq.Elogm[k]

        log_rho[k, :] = qh_log_posterior[k, :] - 0.5 * qh_logdet[k, :] + shift

    # Normalize responsibilities
    max_log_rho = np.max(log_rho, axis=0)
    log_rho_stable = log_rho - max_log_rho
    responsibilities = np.exp(log_rho_stable) / np.sum(np.exp(log_rho_stable), axis=0)

    return responsibilities


def compute_exceedance_probabilities(alpha: np.ndarray, n_samples: int = 100000):
    """Compute exceedance probabilities from Dirichlet posterior."""
    samples = np.random.dirichlet(alpha, size=n_samples)
    best_indices = np.argmax(samples, axis=1)
    xp = np.bincount(best_indices, minlength=len(alpha)) / n_samples
    return xp


# --- Main HBI Function ---


def run_hbi_scipy(
    participant_data: List[List[np.ndarray]],
    model_specs: List[Any],
    max_iter: int = 50,
    tol: float = 1e-5,
    n_starts: int = 3,
    n_jobs: int = -1,
    model_comparison: bool = False,
    hyperprior_beta: float = 1.0,
    hyperprior_nu: float = 0.5,
    hyperprior_sigma: float = 0.01,
) -> HBIResult:
    """
    Run Hierarchical Bayesian Inference using scipy optimization.

    When model_comparison=False (default), runs in parameter estimation mode:
    fits a single model with empirical Bayes shrinkage. This is faster because
    it skips Hessian computation, responsibility updates, and model frequency
    tracking. If multiple model_specs are provided, model_comparison is
    automatically enabled.

    When model_comparison=True, runs full CBM: computes Hessians for Laplace
    approximation, updates model responsibilities, and tracks model frequencies
    and exceedance probabilities.

    Args:
        participant_data: List of N participants, each a list of input column arrays.
        model_specs: List of K ModelSpec objects, each with .func, .param_names, .bounds
        max_iter: Maximum EM iterations
        tol: Convergence tolerance
        n_starts: Number of random restarts for initial fitting
        n_jobs: Number of parallel jobs (-1 = all cores, 1 = sequential)
        model_comparison: Enable full model comparison (Hessians, responsibilities,
            exceedance probabilities). Auto-enabled when len(model_specs) > 1.
        hyperprior_beta: Prior pseudo-count for mean precision
        hyperprior_nu: Prior shape for precision Gamma
        hyperprior_sigma: Prior scale for precision Gamma

    Returns:
        HBIResult with fitted parameters, responsibilities, model frequencies, etc.
    """
    # Respect SLURM allocation if available
    if n_jobs == -1:
        slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK") or os.environ.get("SLURM_CPUS_ON_NODE")
        if slurm_cpus is not None:
            n_jobs = int(slurm_cpus)

    N = len(participant_data)
    K = len(model_specs)

    # Auto-enable model comparison when multiple models are provided
    if K > 1:
        model_comparison = True

    # Resolve n_jobs
    if n_jobs == -1:
        n_jobs = os.cpu_count() or 1
    n_jobs = min(n_jobs, N)  # no point having more workers than subjects

    mode_str = "Model Comparison" if model_comparison else "Parameter Estimation"
    console.rule(f"[bold blue]HBI: {mode_str}")
    console.print(
        f"  Models: [cyan]{K}[/]  Subjects: [cyan]{N}[/]  "
        f"Max iterations: [cyan]{max_iter}[/]  Workers: [cyan]{n_jobs}[/]"
    )

    # Extract bounds as list of tuples for each model
    bounds_per_model = []
    param_counts = []
    for spec in model_specs:
        bl = [(spec.bounds[p][0], spec.bounds[p][1]) for p in spec.param_names]
        bounds_per_model.append(bl)
        param_counts.append(len(spec.param_names))

    # Initialize responsibilities uniformly
    responsibilities = np.ones((K, N)) / K

    # Initialize group-level priors
    prior_mu_tau = []
    for k in range(K):
        Dk = param_counts[k]
        # Prior mean at midpoint of bounds in unbounded space
        a0 = np.array([
            float(to_unbounded((lb + ub) / 2.0, lb, ub))
            for lb, ub in bounds_per_model[k]
        ])
        sigma0 = np.ones(Dk) * hyperprior_sigma
        nu0 = hyperprior_nu
        Etau0 = nu0 / sigma0
        Elogtau0 = psi(nu0) - np.log(sigma0)
        logG0 = np.sum(-gammaln(nu0) + nu0 * np.log(sigma0))

        prior_mu_tau.append(
            GaussianGammaDistribution(
                a=a0, beta=hyperprior_beta, sigma=sigma0, nu=nu0,
                Etau=Etau0, Elogtau=Elogtau0, logG=logG0,
            )
        )

    prior_model_freq = DirichletDistribution(
        limInf=False,
        alpha=np.ones(K),
        Elogm=psi(np.ones(K)) - psi(K),
        logC=0.0,
    )

    # --- Initial individual fits ---
    subject_params_unbounded = []
    hessian_inv_diag_per_model = []

    # Create reusable process pool (uses cloudpickle, handles exec'd functions)
    executor = get_reusable_executor(max_workers=n_jobs)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as init_progress:
        for k in range(K):
            Dk = param_counts[k]
            prior_mean_init = np.zeros(Dk)
            prior_prec_init = np.ones(Dk) * 5.0

            init_task = init_progress.add_task(
                f"[cyan]Init {model_specs[k].name} ({Dk} params, {n_starts} starts)",
                total=N,
            )

            theta_k = np.zeros((Dk, N))
            hess_k = np.zeros((Dk, N))

            # Batch subjects into n_jobs chunks to reduce process pool overhead
            all_indices = list(range(N))
            chunks = [all_indices[i::n_jobs] for i in range(n_jobs)]
            chunks = [c for c in chunks if c]  # remove empty

            futures = {}
            for chunk in chunks:
                batch_data = [participant_data[n] for n in chunk]
                f = executor.submit(
                    _fit_subjects_batch_init,
                    model_specs[k].func,
                    batch_data,
                    chunk,
                    prior_mean_init,
                    prior_prec_init,
                    bounds_per_model[k],
                    n_starts,
                    Dk,
                )
                futures[f] = chunk

            for future in as_completed(futures):
                batch_results = future.result()
                for n, (th, h, _, _) in batch_results.items():
                    theta_k[:, n] = th
                    hess_k[:, n] = h
                    init_progress.advance(init_task)

            subject_params_unbounded.append(theta_k)
            hessian_inv_diag_per_model.append(hess_k)

    # --- Main EM Loop ---
    log_posterior_matrix = np.zeros((K, N))
    log_det_hessian_matrix = np.zeros((K, N))
    converged = False

    # Hessian only needed for model comparison (Laplace approximation)
    hessian_freq = 3  # recompute every N iterations when needed

    # Track convergence: group mean change for param estimation,
    # effective count change for model comparison
    old_group_means = [np.zeros_like(prior_mu_tau[k].a) for k in range(K)]
    old_effective_counts = np.zeros(K)

    # Early stopping: stop if mean log-posterior doesn't improve for `patience` iterations
    best_mean_log_post = -np.inf
    patience = 5
    no_improve_count = 0

    # Skip-stable-subjects: track per-subject parameter change
    skip_threshold = 1e-4
    min_iters_before_skip = 3
    subject_skip_mask = [np.zeros(N, dtype=bool) for _ in range(K)]

    def _parallel_em_step(k, do_hessian, em_maxiter, em_ftol, skip_mask=None):
        """Run parallelized E-step for model k with batching."""
        prior_mean = post_mu_tau[k].a
        prior_precision = post_mu_tau[k].Etau

        # Determine which subjects to optimize
        if skip_mask is not None and not do_hessian:
            active = [n for n in range(N) if not skip_mask[n]]
        else:
            active = list(range(N))

        if not active:
            return {}

        # Batch active subjects into n_jobs chunks
        n_batches = min(n_jobs, len(active))
        chunks = [active[i::n_batches] for i in range(n_batches)]
        chunks = [c for c in chunks if c]

        futures = {}
        for chunk in chunks:
            batch_data = [participant_data[n] for n in chunk]
            batch_thetas = [subject_params_unbounded[k][:, n] for n in chunk]
            f = executor.submit(
                _fit_subjects_batch_em,
                model_specs[k].func,
                batch_data,
                chunk,
                prior_mean,
                prior_precision,
                bounds_per_model[k],
                batch_thetas,
                em_maxiter,
                em_ftol,
                do_hessian,
            )
            futures[f] = chunk

        all_results = {}
        for future in as_completed(futures):
            all_results.update(future.result())
        return all_results

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("[dim]{task.fields[status]}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        em_task = progress.add_task(
            "[cyan]EM iterations", total=max_iter, status=""
        )

        for it in range(max_iter):
            # Step 1: Summary statistics
            effective_counts, weighted_means, weighted_variances = hbi_sumstats(
                responsibilities, subject_params_unbounded, hessian_inv_diag_per_model
            )

            # Step 2: Update group hyperparameters
            post_mu_tau = hbi_qmutau(prior_mu_tau, effective_counts, weighted_means, weighted_variances)
            if model_comparison:
                post_model_freq = hbi_qm(prior_model_freq, effective_counts)

            # Step 3: Re-fit each subject with updated priors (parallelized)
            is_last = (it == max_iter - 1)
            do_hessian = model_comparison and ((it % hessian_freq == 0) or is_last)

            # Adaptive optimizer tolerance: loose early, tight later
            if it < 3:
                em_maxiter, em_ftol = 20, 1e-4
            elif it < 10:
                em_maxiter, em_ftol = 35, 1e-5
            else:
                em_maxiter, em_ftol = 50, 1e-6

            new_subject_params = [p.copy() for p in subject_params_unbounded]
            new_hessian_inv_diag = [h.copy() for h in hessian_inv_diag_per_model]

            # Use skip mask for subjects that have stabilized
            use_skip = it >= min_iters_before_skip
            n_skipped = 0

            for k in range(K):
                skip = subject_skip_mask[k] if use_skip else None
                results = _parallel_em_step(k, do_hessian, em_maxiter, em_ftol, skip)
                if use_skip and skip is not None:
                    n_skipped += int(skip.sum())
                for n, (th_map, h_inv_diag, logdet, log_post) in results.items():
                    new_subject_params[k][:, n] = th_map
                    if do_hessian:
                        new_hessian_inv_diag[k][:, n] = h_inv_diag
                        log_det_hessian_matrix[k, n] = logdet
                    log_posterior_matrix[k, n] = log_post

            # Update skip mask: track per-subject parameter change
            for k in range(K):
                for n in range(N):
                    delta_n = np.max(np.abs(
                        new_subject_params[k][:, n] - subject_params_unbounded[k][:, n]
                    ))
                    subject_skip_mask[k][n] = delta_n < skip_threshold

            subject_params_unbounded = new_subject_params
            hessian_inv_diag_per_model = new_hessian_inv_diag

            # Early stopping: check if mean log-posterior is improving
            mean_log_post = float(np.mean(log_posterior_matrix))
            if mean_log_post > best_mean_log_post + 1e-6:
                best_mean_log_post = mean_log_post
                no_improve_count = 0
            else:
                no_improve_count += 1

            if no_improve_count >= patience:
                progress.update(
                    em_task, completed=max_iter,
                    status=f"[bold yellow]Early stop[/] (no improvement for {patience} iters)",
                )
                converged = True
                break

            # Step 4: Update responsibilities (only for model comparison)
            if model_comparison:
                responsibilities = hbi_qHZ(
                    post_mu_tau,
                    post_model_freq,
                    log_posterior_matrix,
                    log_det_hessian_matrix,
                    subject_params_unbounded,
                    hessian_inv_diag_per_model,
                )

            # Check convergence
            if model_comparison:
                # Converge on effective counts (model assignment)
                delta = np.sum(np.abs(effective_counts - old_effective_counts))
                freq_str = ", ".join(f"{c:.1f}" for c in effective_counts)
                hess_str = " [H]" if do_hessian else ""
                status_str = f"delta={delta:.2e}  counts=[{freq_str}]{hess_str}" if it > 0 else f"counts=[{freq_str}]{hess_str}"
            else:
                # Converge on group mean (parameter estimates)
                delta = sum(
                    float(np.max(np.abs(post_mu_tau[k].a - old_group_means[k])))
                    for k in range(K)
                )
                skip_str = f"  skipped={n_skipped}/{N*K}" if n_skipped > 0 else ""
                status_str = f"delta={delta:.2e}{skip_str}" if it > 0 else ""

            if it > 0:
                progress.update(em_task, advance=1, status=status_str)
                if delta < tol:
                    # For model comparison, recompute Hessians at convergence
                    if model_comparison and not do_hessian:
                        for k in range(K):
                            results = _parallel_em_step(k, True)
                            for n, (_, h_inv_diag, logdet, _) in results.items():
                                hessian_inv_diag_per_model[k][:, n] = h_inv_diag
                                log_det_hessian_matrix[k, n] = logdet

                    progress.update(
                        em_task, completed=max_iter,
                        status=f"[bold green]Converged[/] (iter {it + 1})",
                    )
                    converged = True
                    break
            else:
                progress.update(em_task, advance=1, status=status_str)

            old_effective_counts = effective_counts.copy()
            old_group_means = [post_mu_tau[k].a.copy() for k in range(K)]

    if not converged:
        console.print(
            f"[bold yellow]Warning:[/] HBI did not converge in {max_iter} iterations"
        )

    # --- Compute per-subject NLL at final MAP estimates ---
    per_subject_nll = np.zeros((K, N))
    for k in range(K):
        for n in range(N):
            theta_bounded = params_to_bounded(
                subject_params_unbounded[k][:, n], bounds_per_model[k]
            )
            try:
                nll = float(model_specs[k].func(*participant_data[n], theta_bounded))
            except Exception:
                nll = 1e10
            per_subject_nll[k, n] = nll

    # --- Transform parameters to bounded space ---
    params_bounded = []
    for k in range(K):
        theta_bounded_k = np.zeros_like(subject_params_unbounded[k])
        for n in range(N):
            theta_bounded_k[:, n] = params_to_bounded(
                subject_params_unbounded[k][:, n], bounds_per_model[k]
            )
        params_bounded.append(theta_bounded_k)

    # Model comparison outputs
    if model_comparison:
        xp = compute_exceedance_probabilities(post_model_freq.alpha)
        model_freq = post_model_freq.alpha / np.sum(post_model_freq.alpha)
    else:
        xp = np.ones(K)
        model_freq = np.ones(K) / K

    # Summary table
    table = Table(title="HBI Results", show_header=True, header_style="bold cyan")
    table.add_column("Model", style="bold")
    table.add_column("Params", justify="right")
    if model_comparison:
        table.add_column("Frequency", justify="right")
        table.add_column("Exceedance Prob", justify="right")
    table.add_column("Mean NLL", justify="right")
    for k in range(K):
        row = [model_specs[k].name, str(param_counts[k])]
        if model_comparison:
            row.extend([
                f"{model_freq[k]:.3f}",
                f"{xp[k]:.3f}",
            ])
        row.append(f"{np.mean(per_subject_nll[k]):.2f}")
        table.add_row(*row)
    console.print(table)
    console.rule("[bold blue]HBI Complete")

    return HBIResult(
        parameters=params_bounded,
        parameters_unbounded=subject_params_unbounded,
        responsibilities=responsibilities,
        group_mean=[q.a for q in post_mu_tau],
        group_precision=[q.Etau for q in post_mu_tau],
        model_frequency=model_freq,
        exceedance_prob=xp,
        per_subject_nll=per_subject_nll,
        converged=converged,
        n_iterations=it + 1,
    )
