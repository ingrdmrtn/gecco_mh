import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from pathlib import Path
from datetime import datetime


def _log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}")


def load_id_data(cfg):
    """Load individual differences data from the path specified in config."""
    id_cfg = cfg.individual_differences_eval
    data_path = Path(id_cfg.data_path)

    # Resolve relative paths from project root
    if not data_path.is_absolute():
        project_root = Path(__file__).resolve().parents[2]
        data_path = project_root / data_path

    df = pd.read_csv(data_path)
    _log(f"[GeCCo] Loaded individual differences data: {len(df)} rows from {data_path}")
    return df


def evaluate_individual_differences(fit_result, df, cfg, id_data=None):
    """
    Regress fitted model parameters on individual difference measures.

    For each model parameter, runs: param_i ~ predictors + covariates
    using OLS and collects R² values.

    Parameters
    ----------
    fit_result : dict
        Output from run_fit(), must contain 'parameter_values' and 'param_names'.
    df : pd.DataFrame
        Behavioural data used for fitting (to extract participant IDs).
    cfg : SimpleNamespace
        Config with individual_differences_eval section.
    id_data : pd.DataFrame, optional
        Pre-loaded individual differences data. If None, loads from config path.

    Returns
    -------
    dict with keys: mean_r2, per_param_r2, summary_text
    """
    id_cfg = cfg.individual_differences_eval
    predictors = list(id_cfg.predictors)
    covariates = list(getattr(id_cfg, "covariates", []))
    all_features = predictors + covariates

    # Load ID data if not provided
    if id_data is None:
        id_data = load_id_data(cfg)

    # Get participant IDs in the same order as parameter_values
    # behavioral_id_column: column in behavioural data to join on (defaults to data.id_column)
    # This is needed when the fitting ID (e.g., numeric "participant") differs from the
    # join key shared with the self-report data (e.g., string "subject_id")
    fitting_id_col = cfg.data.id_column
    join_id_col = getattr(id_cfg, "behavioral_id_column", fitting_id_col)
    id_data_id_col = id_cfg.id_column
    participants = df[fitting_id_col].unique()

    # Build parameter DataFrame (one row per participant)
    param_names = fit_result["param_names"]
    param_values = fit_result["parameter_values"]
    param_df = pd.DataFrame(param_values, columns=param_names)

    # Get the join key for each participant (one value per participant)
    if join_id_col != fitting_id_col:
        # The join column may not be in the eval DataFrame (dropped by load_data),
        # so load the mapping directly from the original data file
        data_path = Path(cfg.data.path)
        if not data_path.is_absolute():
            project_root = Path(__file__).resolve().parents[2]
            data_path = project_root / data_path
        full_df = pd.read_csv(data_path, usecols=[fitting_id_col, join_id_col]).drop_duplicates()
        id_map = dict(zip(full_df[fitting_id_col], full_df[join_id_col].astype(str)))
        join_ids = [id_map[p] for p in participants]
    else:
        join_ids = [str(p) for p in participants]
    param_df["_participant_id"] = join_ids

    # Prepare ID data for merge
    id_data = id_data.copy()
    id_data["_participant_id"] = id_data[id_data_id_col].astype(str)

    # Inner join
    merged = param_df.merge(id_data, on="_participant_id", how="inner")

    n_dropped = len(participants) - len(merged)
    if n_dropped > 0:
        _log(f"[GeCCo] Warning: {n_dropped} participants dropped during ID merge "
             f"({len(merged)} remaining)")

    if len(merged) < 5:
        _log("[GeCCo] Too few participants for individual differences regression")
        return {
            "mean_r2": 0.0,
            "per_param_r2": {p: 0.0 for p in param_names},
            "summary_text": "Individual differences analysis skipped: too few participants after merge.",
        }

    # Check that all feature columns exist
    missing_cols = [c for c in all_features if c not in merged.columns]
    if missing_cols:
        raise ValueError(
            f"Columns {missing_cols} not found in individual differences data. "
            f"Available: {list(id_data.columns)}"
        )

    # Build feature matrix
    X = merged[all_features].values.astype(float)

    # Drop rows with NaN in features
    valid_mask = ~np.isnan(X).any(axis=1)
    if valid_mask.sum() < 5:
        _log("[GeCCo] Too few valid rows after removing NaNs")
        return {
            "mean_r2": 0.0,
            "per_param_r2": {p: 0.0 for p in param_names},
            "summary_text": "Individual differences analysis skipped: too many missing values.",
        }
    X = X[valid_mask]

    # Run regression for each parameter
    per_param_r2 = {}
    summary_lines = ["Individual Differences Analysis (R² from param ~ questionnaire regressions):"]

    for pname in param_names:
        y = merged[pname].values.astype(float)[valid_mask]

        # Skip if no variance
        if np.std(y) < 1e-10:
            per_param_r2[pname] = 0.0
            summary_lines.append(f"  - {pname}: R² = 0.00 (no variance in parameter)")
            continue

        reg = LinearRegression().fit(X, y)
        r2 = reg.score(X, y)
        r2 = max(0.0, r2)  # Clip negative R² (possible with few samples)
        per_param_r2[pname] = r2
        summary_lines.append(f"  - {pname}: R² = {r2:.3f}")

    mean_r2 = float(np.mean(list(per_param_r2.values()))) if per_param_r2 else 0.0
    summary_lines.append(f"  Mean R² across parameters: {mean_r2:.3f}")

    summary_text = "\n".join(summary_lines)
    _log(f"[GeCCo] Individual differences: mean R² = {mean_r2:.3f}")

    return {
        "mean_r2": mean_r2,
        "per_param_r2": per_param_r2,
        "summary_text": summary_text,
    }
