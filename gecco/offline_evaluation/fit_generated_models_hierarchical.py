"""
Hierarchical fitting wrapper that bridges the run_fit() interface
with the scipy-based HBI implementation.
"""

import time
import numpy as np

from rich.console import Console

from gecco.offline_evaluation.utils import build_model_spec
from gecco.offline_evaluation.evaluation_functions import aic as _aic, bic as _bic
from gecco.model_fitting.hbi_scipy import run_hbi_scipy

console = Console()


def run_fit_hierarchical(
    df,
    code_text,
    cfg,
    expected_func_name="cognitive_model",
    max_iter=50,
    tol=1e-5,
    n_starts=3,
    n_jobs=-1,
):
    """
    Fit an LLM-generated model hierarchically using HBI.

    Same interface as run_fit() but uses hierarchical Bayesian inference
    for parameter estimation (empirical Bayes shrinkage).

    Parameters
    ----------
    df : pd.DataFrame
        Behavioral dataset to fit on.
    code_text : str
        LLM-generated model code.
    cfg : object
        Experiment configuration.
    expected_func_name : str
        Function name to extract from code.
    max_iter : int
        Max EM iterations for HBI.
    tol : float
        Convergence tolerance.
    n_starts : int
        Number of random restarts for initial fitting.

    Returns
    -------
    dict
        Compatible with run_fit() output:
        {metric_name, metric_value, param_names, model_name,
         parameter_values, eval_metrics, hbi_result}
    """
    # Build model spec
    spec = build_model_spec(
        code_text,
        expected_func_name=expected_func_name,
        cfg=cfg,
    )

    data_cfg = cfg.data
    participants = df[data_cfg.id_column].unique()
    input_cols = list(data_cfg.input_columns)
    N = len(participants)

    console.print(
        f"[bold]HBI fitting[/] [cyan]{expected_func_name}[/] "
        f"to [cyan]{N}[/] participants "
        f"(params: {', '.join(spec.param_names)})"
    )

    # Extract per-participant data
    participant_data = []
    participant_n_trials = []
    for p in participants:
        df_p = df[df[data_cfg.id_column] == p].reset_index(drop=True)
        cols = [df_p[c].to_numpy() for c in input_cols]
        participant_data.append(cols)
        participant_n_trials.append(len(df_p))

    t0 = time.time()

    # Run HBI with single model
    hbi_result = run_hbi_scipy(
        participant_data=participant_data,
        model_specs=[spec],
        max_iter=max_iter,
        tol=tol,
        n_starts=n_starts,
        n_jobs=n_jobs,
    )

    elapsed = time.time() - t0
    console.print(
        f"  Completed in [cyan]{elapsed:.1f}s[/] "
        f"({hbi_result.n_iterations} iterations, "
        f"converged={hbi_result.converged})"
    )

    # Extract results for the single model (index 0)
    params_bounded = hbi_result.parameters[0]  # (D, N)
    nll_per_subject = hbi_result.per_subject_nll[0]  # (N,)
    n_params = len(spec.param_names)

    # Compute per-participant metrics
    metric_name = cfg.evaluation.metric.upper()
    metric_map = {"AIC": _aic, "BIC": _bic}
    metric_func = metric_map.get(metric_name, _bic)

    eval_metrics = []
    for i in range(N):
        nll = nll_per_subject[i]
        m = metric_func(nll, n_params, participant_n_trials[i])
        eval_metrics.append(float(m))

    mean_metric = float(np.mean(eval_metrics))
    console.print(f"  Mean {metric_name} = [bold]{mean_metric:.2f}[/]")

    # Format parameter_values as list of arrays (one per participant)
    parameter_values = [params_bounded[:, n] for n in range(N)]

    return {
        "metric_name": metric_name,
        "metric_value": mean_metric,
        "param_names": list(spec.param_names),
        "model_name": spec.name,
        "parameter_values": parameter_values,
        "eval_metrics": eval_metrics,
        "participant_n_trials": participant_n_trials,
        "hbi_result": hbi_result,
    }
