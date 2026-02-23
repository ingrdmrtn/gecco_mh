# engine/run_fit.py
import time
import numpy as np
from datetime import datetime
from scipy.optimize import minimize
from gecco.offline_evaluation.utils import build_model_spec
from gecco.offline_evaluation.evaluation_functions import aic as _aic, bic as _bic

def _log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}")

def run_fit(df, code_text, cfg, expected_func_name="cognitive_model"):
    """
    Compile an LLM-generated cognitive model, fit it to participant data,
    and return fit statistics (AIC/BIC) across participants.

    Parameters
    ----------
    df : pd.DataFrame
        The behavioral dataset to fit on.
    code_text : str
        The full LLM-generated model code (function definition).
    cfg : SimpleNamespace
        Full experiment configuration (already loaded via load_config()).
    expected_func_name : str
        Name of the function to extract and fit (default = "cognitive_model").

    Returns
    -------
    dict
        {
            "metric_name": str,
            "metric_value": float,
            "param_names": list,
            "model_name": str
        }
    """
    # --- Build model specification from code ---
    spec = build_model_spec(
        code_text, 
        expected_func_name=expected_func_name,
        cfg=cfg  # Pass config so it can extract base class
    )

    # --- Prepare metric function ---
    metric_name = cfg.evaluation.metric.upper()
    metric_map = {"AIC": _aic, "BIC": _bic}
    metric_func = metric_map.get(metric_name, _bic)

    # --- Load the model function from code ---
    model_func =  spec.func

    # --- Data + fit configuration ---
    data_cfg = cfg.data
    participants = df[data_cfg.id_column].unique()
    print(spec.param_names)
    parameter_bounds = list(spec.bounds.values())

    n_starts = getattr(cfg.evaluation, "n_starts", 10)

    eval_metrics = []
    parameter_estimates = []
    n_participants = len(participants)
    t0_fit = time.time()
    _log(f"[GeCCo] Fitting {expected_func_name} to {n_participants} participants ({n_starts} starts each)")

    # --- Fit per participant ---
    for pi, p in enumerate(participants):
        df_p = df[df[data_cfg.id_column] == p].reset_index(drop=True)

        # Extract task-relevant input columns dynamically
        input_cols = list(data_cfg.input_columns)
        inputs = [df_p[c].to_numpy() for c in input_cols]

        min_ll = np.inf
        best_parameter_values = []
        for _ in range(n_starts):
            x0 = [np.random.uniform(lo, hi) for lo, hi in parameter_bounds]
            res = minimize(
                lambda x: float(model_func(*inputs, x)),
                x0,
                method="L-BFGS-B",
                bounds=parameter_bounds,
            )
            if res.fun < min_ll:
                min_ll = res.fun
                best_parameter_values = res.x

        eval_metrics.append(metric_func(min_ll, len(parameter_bounds), len(df_p)))
        parameter_estimates.append(best_parameter_values)

        if (pi + 1) % 10 == 0 or (pi + 1) == n_participants:
            elapsed = time.time() - t0_fit
            _log(f"[GeCCo] Fitted {pi+1}/{n_participants} participants ({elapsed:.0f}s elapsed)")

    # --- Aggregate results ---
    mean_metric = float(np.mean(eval_metrics))
    total_fit_time = time.time() - t0_fit
    _log(f"[GeCCo] Mean {metric_name} = {mean_metric:.2f} (fitting took {total_fit_time:.1f}s)")

    return {
        "metric_name": metric_name,
        "metric_value": mean_metric,
        "param_names": list(spec.param_names),
        "model_name": spec.name,
        "parameter_values": parameter_estimates,
        "eval_metrics": eval_metrics

    }