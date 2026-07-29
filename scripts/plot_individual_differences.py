#!/usr/bin/env python
"""
Plot fitted cognitive-model parameters against individual-difference measures.

The pipeline's own individual-differences analysis
(gecco/offline_evaluation/individual_differences.py) fits a plain
LinearRegression on raw features, so it can only detect straight-line,
additive relationships. That is a poor match for developmental data: an
inverted-U in age -- the central finding of Eckstein et al. (2022), where
mid-teens peak -- produces a near-flat best-fitting line and an R^2 near zero.

This script plots each parameter against each predictor and fits BOTH a linear
and a quadratic trend, reporting both R^2 values so you can see directly
whether the linear assumption is costing you signal.

Usage:
  # Best model from the run's registry, on the eval split
  python scripts/plot_individual_differences.py --config eckstein_probswitch_gpt54nano.yaml --registry

  # The configured baseline model instead
  python scripts/plot_individual_differences.py --config eckstein_probswitch_gpt54nano.yaml --baseline

  # A specific model file, on the held-out test split
  python scripts/plot_individual_differences.py --config eckstein_probswitch_gpt54nano.yaml \
      --code model.py --func-name cognitive_model1 --split test
"""

import argparse
import json
import re
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib

matplotlib.use("Agg")  # headless: HPC nodes have no display
import matplotlib.pyplot as plt

from config.schema import load_config
from gecco.prepare_data.io import load_data, split_by_participant
from gecco.offline_evaluation.fit_generated_models import run_fit_hierarchical
from gecco.offline_evaluation.individual_differences import load_id_data
from gecco.run_gecco import extract_model_func_name

# Reuse the registry loader and split logic rather than reimplementing them,
# so this script can never drift from what the pipeline actually fits.
from test_fit_model import load_code_from_registry, get_eval_test_split

from rich.console import Console
from rich.table import Table

console = Console()

# Chart chrome and the two-series categorical palette. Validated with the
# dataviz palette checker (light surface #fcfcfb): lightness band, chroma
# floor, CVD separation (worst pair dE 24.7 protan), normal-vision floor
# (dE 33.6) and >=3:1 contrast all pass.
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
SERIES_LINEAR = "#2a78d6"     # slot 1, blue
SERIES_QUADRATIC = "#eb6834"  # slot 2, orange

MAX_CATEGORICAL_LEVELS = 5    # at or below this, treat a predictor as discrete


def r_squared(y, y_hat):
    """Plain coefficient of determination, guarded against zero variance."""
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot < 1e-12:
        return 0.0
    ss_res = float(np.sum((y - y_hat) ** 2))
    return 1.0 - ss_res / ss_tot


def fit_trends(x, y):
    """Fit linear and quadratic trends, returning curves and their R^2."""
    order = np.argsort(x)
    xs = x[order]
    grid = np.linspace(float(xs.min()), float(xs.max()), 200)

    lin_coef = np.polyfit(x, y, 1)
    lin_r2 = r_squared(y, np.polyval(lin_coef, x))

    # A quadratic needs at least 3 distinct x values to be identified
    if len(np.unique(x)) >= 3:
        quad_coef = np.polyfit(x, y, 2)
        quad_r2 = r_squared(y, np.polyval(quad_coef, x))
        quad_curve = np.polyval(quad_coef, grid)
    else:
        quad_coef, quad_r2, quad_curve = None, None, None

    return {
        "grid": grid,
        "linear_curve": np.polyval(lin_coef, grid),
        "linear_r2": lin_r2,
        "quadratic_curve": quad_curve,
        "quadratic_r2": quad_r2,
        # Vertex of the parabola, i.e. where a peak or trough sits
        "peak_at": (
            -quad_coef[1] / (2 * quad_coef[0])
            if quad_coef is not None and abs(quad_coef[0]) > 1e-12
            else None
        ),
    }


def build_parameter_frame(fit_result, df, cfg):
    """Join fitted per-participant parameters to the individual-difference data.

    Mirrors evaluate_individual_differences: parameter_values follows
    df[id_column].unique(), and the join key may be a different column
    (subject_id) than the fitting id (participant).
    """
    id_cfg = cfg.individual_differences_eval
    fitting_id_col = cfg.data.id_column
    join_id_col = getattr(id_cfg, "behavioral_id_column", fitting_id_col)

    participants = df[fitting_id_col].unique()
    param_df = pd.DataFrame(
        fit_result["parameter_values"], columns=fit_result["param_names"]
    )

    if join_id_col != fitting_id_col:
        data_path = Path(cfg.data.path)
        if not data_path.is_absolute():
            data_path = Path(__file__).resolve().parents[1] / data_path
        full_df = pd.read_csv(
            data_path, usecols=[fitting_id_col, join_id_col]
        ).drop_duplicates()
        id_map = dict(zip(full_df[fitting_id_col], full_df[join_id_col].astype(str)))
        join_ids = [id_map[p] for p in participants]
    else:
        join_ids = [str(p) for p in participants]

    param_df["_participant_id"] = join_ids

    id_data = load_id_data(cfg).copy()
    id_data["_participant_id"] = id_data[id_cfg.id_column].astype(str)

    merged = param_df.merge(id_data, on="_participant_id", how="inner")
    n_dropped = len(participants) - len(merged)
    if n_dropped:
        console.print(
            f"[yellow]{n_dropped} participants dropped in the ID merge "
            f"({len(merged)} remaining)[/]"
        )
    return merged, list(fit_result["param_names"])


def style_axis(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis="y", color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRIDLINE)
    ax.tick_params(colors=INK_MUTED, labelsize=8, length=3)


def plot_grid(merged, param_names, features, out_path, title):
    """One panel per (parameter, feature); continuous features get trend fits."""
    n_rows, n_cols = len(param_names), len(features)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.4 * n_cols + 1.0, 2.7 * n_rows + 1.2),
        squeeze=False,
        layout="constrained",
    )
    fig.patch.set_facecolor(SURFACE)

    rng = np.random.default_rng(0)
    stats_rows = []
    drew_trend = False

    for i, pname in enumerate(param_names):
        for j, feat in enumerate(features):
            ax = axes[i][j]
            style_axis(ax)

            pair = merged[[pname, feat]].apply(pd.to_numeric, errors="coerce").dropna()
            x = pair[feat].to_numpy(dtype=float)
            y = pair[pname].to_numpy(dtype=float)

            if len(x) < 5 or np.std(x) < 1e-12:
                ax.text(
                    0.5, 0.5, "insufficient data",
                    ha="center", va="center", transform=ax.transAxes,
                    color=INK_MUTED, fontsize=9,
                )
                if i == 0:
                    ax.set_title(feat, color=INK_PRIMARY, fontsize=10, pad=8)
                if j == 0:
                    ax.set_ylabel(pname, color=INK_PRIMARY, fontsize=10)
                continue

            levels = np.unique(x)
            is_categorical = len(levels) <= MAX_CATEGORICAL_LEVELS

            if is_categorical:
                # Discrete predictor (e.g. Gender): jittered strip + group means,
                # since a scatter and a trend line would both be misleading.
                jitter = rng.uniform(-0.12, 0.12, size=len(x))
                ax.scatter(
                    x + jitter, y, s=26, color=INK_MUTED, alpha=0.5,
                    linewidths=0, zorder=2,
                )
                for lv in levels:
                    vals = y[x == lv]
                    ax.plot(
                        [lv - 0.22, lv + 0.22], [vals.mean()] * 2,
                        color=SERIES_LINEAR, linewidth=2, zorder=3,
                    )
                ax.set_xticks(levels)
                ax.text(
                    0.03, 0.95, f"n = {len(x)}\ngroup means",
                    transform=ax.transAxes, va="top", ha="left",
                    color=INK_SECONDARY, fontsize=8,
                )
                stats_rows.append((pname, feat, len(x), None, None, None))
            else:
                trends = fit_trends(x, y)
                ax.scatter(
                    x, y, s=30, color=INK_MUTED, alpha=0.45,
                    linewidths=0, zorder=2,
                )
                ax.plot(
                    trends["grid"], trends["linear_curve"],
                    color=SERIES_LINEAR, linewidth=2, zorder=4,
                    label="linear" if not drew_trend else None,
                )
                if trends["quadratic_curve"] is not None:
                    ax.plot(
                        trends["grid"], trends["quadratic_curve"],
                        color=SERIES_QUADRATIC, linewidth=2, zorder=3,
                        label="quadratic" if not drew_trend else None,
                    )
                drew_trend = True

                q_r2 = trends["quadratic_r2"]
                label = f"n = {len(x)}\nlinear R² = {trends['linear_r2']:.3f}"
                if q_r2 is not None:
                    label += f"\nquadratic R² = {q_r2:.3f}"
                ax.text(
                    0.03, 0.95, label,
                    transform=ax.transAxes, va="top", ha="left",
                    color=INK_SECONDARY, fontsize=8,
                )
                stats_rows.append(
                    (pname, feat, len(x), trends["linear_r2"], q_r2, trends["peak_at"])
                )

            if i == 0:
                ax.set_title(feat, color=INK_PRIMARY, fontsize=10, pad=8)
            if j == 0:
                ax.set_ylabel(pname, color=INK_PRIMARY, fontsize=10)

    handles, labels = [], []
    for ax_row in axes:
        for ax in ax_row:
            h, l = ax.get_legend_handles_labels()
            handles += h
            labels += l
    if handles:
        fig.legend(
            handles, labels, loc="outside lower center", ncol=2,
            frameon=False, labelcolor=INK_SECONDARY, fontsize=9,
        )

    fig.suptitle(title, color=INK_PRIMARY, fontsize=12)
    fig.savefig(out_path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return stats_rows


def main():
    parser = argparse.ArgumentParser(
        description="Plot fitted parameters against individual-difference measures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", required=True, help="Config YAML name (in config/)")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--registry", action="store_true", help="Best model from the registry")
    src.add_argument("--baseline", action="store_true", help="The configured baseline model")
    src.add_argument("--code", type=str, help="Path to a .py file with the model")
    parser.add_argument("--model-name", type=str, default=None, help="Registry model name")
    parser.add_argument("--model-index", type=int, default=None, help="Registry model index")
    parser.add_argument("--func-name", type=str, default=None, help="Override function name")
    parser.add_argument("--split", choices=["eval", "test"], default="eval")
    parser.add_argument("--out", type=str, default=None, help="Output PNG path")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    cfg = load_config(project_root / "config" / args.config)

    if not hasattr(cfg, "individual_differences_eval"):
        console.print("[bold red]Config has no individual_differences_eval section[/]")
        sys.exit(1)

    results_dir = project_root / "results" / cfg.task.name

    # --- Resolve the model code ---
    if args.code:
        code = Path(args.code).read_text()
        source = args.code
    elif args.baseline:
        baseline_cfg = getattr(cfg, "baseline", None)
        code = getattr(baseline_cfg, "model", None) if baseline_cfg else None
        if not code:
            code = getattr(cfg.llm, "template_model", None)
        source = "config baseline"
    else:
        code = load_code_from_registry(
            results_dir, model_name=args.model_name, model_index=args.model_index
        )
        source = f"registry ({results_dir.name})"

    if not code:
        console.print("[bold red]No model code resolved[/]")
        sys.exit(1)

    func_name = args.func_name or extract_model_func_name(code)
    console.print(f"[dim]Model:[/] {func_name}  [dim]from[/] {source}")

    # --- Data and split, exactly as the pipeline builds them ---
    df = load_data(cfg.data.path, cfg.data.input_columns)
    splits = split_by_participant(df, cfg.data.id_column, cfg.data.splits)
    df_eval, df_test = get_eval_test_split(df, splits["prompt"], cfg)
    df_fit = df_eval if args.split == "eval" else df_test
    console.print(
        f"[dim]Split:[/] {args.split}  "
        f"({df_fit[cfg.data.id_column].nunique()} participants)"
    )

    # --- Fit ---
    fit_result = run_fit_hierarchical(
        df_fit, code, cfg=cfg, expected_func_name=func_name
    )

    merged, param_names = build_parameter_frame(fit_result, df_fit, cfg)

    id_cfg = cfg.individual_differences_eval
    features = list(id_cfg.predictors) + list(getattr(id_cfg, "covariates", []))
    missing = [f for f in features if f not in merged.columns]
    if missing:
        console.print(f"[bold red]Features not in the ID data: {missing}[/]")
        sys.exit(1)

    out_path = Path(args.out) if args.out else (
        results_dir / f"individual_differences_{func_name}_{args.split}.png"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stats_rows = plot_grid(
        merged,
        param_names,
        features,
        out_path,
        f"{func_name} — parameters vs individual differences ({args.split} split)",
    )

    # --- Report: linear vs quadratic is the point of the exercise ---
    table = Table(title="Linear vs quadratic fit", header_style="bold")
    table.add_column("Parameter")
    table.add_column("Predictor")
    table.add_column("n", justify="right")
    table.add_column("linear R²", justify="right")
    table.add_column("quadratic R²", justify="right")
    table.add_column("Δ R²", justify="right")
    table.add_column("peak at", justify="right")
    for pname, feat, n, lin, quad, peak in stats_rows:
        if lin is None:
            table.add_row(pname, feat, str(n), "—", "—", "—", "categorical")
            continue
        delta = (quad - lin) if quad is not None else None
        table.add_row(
            pname, feat, str(n),
            f"{lin:.3f}",
            f"{quad:.3f}" if quad is not None else "—",
            f"{delta:+.3f}" if delta is not None else "—",
            f"{peak:.1f}" if peak is not None else "—",
        )
    console.print(table)
    console.print(
        "\n[dim]A large positive Δ R² means the pipeline's linear-only "
        "individual-differences analysis is understating that relationship. "
        "Check whether 'peak at' falls inside the observed predictor range — "
        "a vertex far outside it means the quadratic is bending, not peaking.[/]"
    )
    console.print(f"\nSaved plot to [bold]{out_path}[/]")


if __name__ == "__main__":
    main()
