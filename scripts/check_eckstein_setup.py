#!/usr/bin/env python
"""
Preflight checks for running GeCCo on a new dataset, without calling an LLM.

Runs three offline checks against a config:

  1. Data + splits + narrative rendering (exactly as run_gecco_distributed.py does)
  2. Baseline model compilation (and writes the code out so it can be fed to
     scripts/test_fit_model.py for a timed HBI fit)
  3. Task simulator sanity + parameter recovery on the baseline model

Usage:
    python scripts/check_eckstein_setup.py --config eckstein_probswitch_gpt54nano.yaml

    # Skip the (slow) recovery fit, just check simulator dynamics:
    python scripts/check_eckstein_setup.py --config eckstein_probswitch_gpt54nano.yaml --sim-only

    # Smaller/faster recovery check:
    python scripts/check_eckstein_setup.py --config eckstein_probswitch_gpt54nano.yaml \
        --recovery-subjects 20 --recovery-trials 130
"""

import argparse
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.schema import load_config
from gecco.prepare_data.io import load_data, split_by_participant
from gecco.prepare_data.data2text import get_data2text_function

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def check_data_and_narrative(cfg):
    """Step 1: load data, reproduce the splits, render the prompt narrative."""
    console.rule("[bold]1. Data, splits and narrative")

    data_cfg = cfg.data
    df = load_data(data_cfg.path, data_cfg.input_columns)
    console.print(
        f"Loaded [cyan]{len(df)}[/] rows, "
        f"[cyan]{df[data_cfg.id_column].nunique()}[/] participants "
        f"from [dim]{data_cfg.path}[/]"
    )
    console.print(f"Columns: {list(df.columns)}")

    for col in data_cfg.input_columns:
        vals = np.unique(df[col].values)
        preview = vals if len(vals) <= 10 else f"{vals[:5]} ... {vals[-5:]}"
        console.print(f"  [dim]{col}[/]: {len(vals)} unique -> {preview}")

    splits = split_by_participant(df, data_cfg.id_column, data_cfg.splits)
    df_prompt = splits["prompt"]

    # Same eval/test split as run_gecco_distributed.py
    eval_test_proportion = getattr(cfg.evaluation, "eval_test_split", 0.7)
    non_prompt_ids = sorted(
        set(df[data_cfg.id_column].unique())
        - set(df_prompt[data_cfg.id_column].unique())
    )
    np.random.seed(getattr(cfg.evaluation, "split_seed", 42))
    np.random.shuffle(non_prompt_ids)
    split_idx = int(len(non_prompt_ids) * eval_test_proportion)
    eval_ids, test_ids = non_prompt_ids[:split_idx], non_prompt_ids[split_idx:]

    table = Table(title="Data Split", show_header=True, header_style="bold")
    table.add_column("Split")
    table.add_column("Participants", justify="right")
    table.add_column("Trials", justify="right")
    table.add_row(
        "Prompt",
        str(df_prompt[data_cfg.id_column].nunique()),
        str(len(df_prompt)),
    )
    table.add_row(
        "Eval",
        str(len(eval_ids)),
        str(len(df[df[data_cfg.id_column].isin(eval_ids)])),
    )
    table.add_row(
        "Test",
        str(len(test_ids)),
        str(len(df[df[data_cfg.id_column].isin(test_ids)])),
    )
    console.print(table)

    data2text = get_data2text_function(data_cfg.data2text_function)
    metadata = getattr(getattr(cfg, "metadata", None), "flag", False)
    data_text = data2text(
        df_prompt,
        id_col=data_cfg.id_column,
        template=data_cfg.narrative_template,
        fit_type=getattr(cfg.evaluation, "fit_type", "group"),
        metadata=getattr(cfg.metadata, "narrative_template", None) if metadata else None,
        max_trials=getattr(data_cfg, "max_prompt_trials", None),
        value_mappings=getattr(data_cfg, "value_mappings", None),
    )
    console.print(
        Panel(
            data_text[:1200] + ("\n[...]" if len(data_text) > 1200 else ""),
            title=f"Narrative sent to the LLM ({len(data_text)} chars total)",
            border_style="blue",
        )
    )
    return df


def check_baseline(cfg, out_path):
    """Step 2: compile the baseline model and write it out for test_fit_model.py."""
    console.rule("[bold]2. Baseline model")

    from gecco.offline_evaluation.utils import build_model_spec

    baseline_cfg = getattr(cfg, "baseline", None)
    code = getattr(baseline_cfg, "model", None) if baseline_cfg else None
    if not code:
        code = getattr(cfg.llm, "template_model", None)
    if not code:
        console.print("[yellow]No baseline.model or llm.template_model in config[/]")
        return None, None

    match = re.search(r"def\s+(\w+)\s*\(", code)
    func_name = match.group(1) if match else "cognitive_model"

    spec = build_model_spec(code, expected_func_name=func_name, cfg=cfg)
    console.print(
        f"Compiled [bold]{func_name}[/] — parameters: "
        f"{', '.join(f'{p}{spec.bounds[p]}' for p in spec.param_names)}"
    )

    Path(out_path).write_text(code)
    console.print(f"Baseline code written to [dim]{out_path}[/]")
    console.print(
        "\nTime an HBI fit on the eval split with:\n"
        f"  [cyan]python scripts/test_fit_model.py --config {cfg._config_name} "
        f"--code {out_path} --func-name {func_name} --split eval[/]\n"
    )
    return spec, func_name


def check_simulator(cfg, spec, n_subjects, n_trials, sim_only):
    """Step 3: simulator dynamics sanity check, then parameter recovery."""
    console.rule("[bold]3. Simulator and parameter recovery")

    recovery_cfg = getattr(cfg, "parameter_recovery", None)
    if recovery_cfg is None or not getattr(recovery_cfg, "enabled", False):
        console.print("[yellow]parameter_recovery.enabled is false — skipping[/]")
        return

    from gecco.parameter_recovery import (
        ParameterRecoveryChecker,
        get_simulator,
    )

    simulator = get_simulator(recovery_cfg)
    console.print(
        f"Simulator: [bold]{type(simulator).__name__}[/] "
        f"(columns {simulator.get_input_columns()})"
    )

    expected_cols = list(cfg.data.input_columns)
    if simulator.get_input_columns() != expected_cols:
        console.print(
            f"[bold red]MISMATCH:[/] simulator columns "
            f"{simulator.get_input_columns()} != data.input_columns {expected_cols}"
        )

    # --- Dynamics sanity check on a handful of simulated subjects ---
    rng = np.random.default_rng(0)
    bounds_list = [spec.bounds[p] for p in spec.param_names]
    accs, block_lens = [], []
    for _ in range(5):
        true_params = np.array([rng.uniform(lb, ub) for lb, ub in bounds_list])
        cols = simulator.simulate_subject(spec.func, true_params, n_trials, rng=rng)
        action, reward = cols[0], cols[1]
        accs.append(float(np.mean(reward)))
        # Block structure is implicit; approximate from reward runs
        switches = np.flatnonzero(np.diff(action) != 0)
        block_lens.append(n_trials / max(len(switches), 1))

    console.print(
        f"5 simulated subjects: mean reward rate = [cyan]{np.mean(accs):.3f}[/] "
        f"(real data: 0.560 overall, 0.767 when correct box chosen)"
    )
    console.print(
        f"  mean trials between choice switches = [cyan]{np.mean(block_lens):.1f}[/]"
    )

    if sim_only:
        console.print("[dim]--sim-only: skipping the recovery fit[/]")
        return

    checker = ParameterRecoveryChecker(
        simulator=simulator,
        n_subjects=n_subjects,
        n_trials=n_trials,
        threshold=getattr(recovery_cfg, "threshold", 0.5),
        n_fitting_starts=getattr(recovery_cfg, "n_fitting_starts", 3),
        n_jobs=getattr(recovery_cfg, "n_jobs", -1),
    )
    t0 = time.time()
    result = checker.check(spec)
    console.print(
        f"Recovery finished in {time.time() - t0:.1f}s — "
        f"passed={result['passed']}, mean r={result['mean_r']:.3f}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Offline preflight checks for a GeCCo dataset config",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Config YAML file name (in config/)")
    parser.add_argument("--baseline-out", type=str, default=None,
                        help="Where to write the baseline model code "
                             "(default: /tmp/baseline_<task>.py)")
    parser.add_argument("--recovery-subjects", type=int, default=None,
                        help="Override parameter_recovery.n_subjects")
    parser.add_argument("--recovery-trials", type=int, default=None,
                        help="Override parameter_recovery.n_trials")
    parser.add_argument("--sim-only", action="store_true",
                        help="Check simulator dynamics but skip the recovery fit")
    parser.add_argument("--skip-recovery", action="store_true",
                        help="Skip step 3 entirely")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "config" / args.config
    if not config_path.exists():
        console.print(f"[bold red]ERROR:[/] config not found: {config_path}")
        sys.exit(1)

    cfg = load_config(config_path)
    cfg._config_name = args.config  # only used for the printed hint

    console.print(
        Panel(
            f"[bold]Config:[/] {args.config}\n"
            f"[bold]Task:[/] {cfg.task.name}\n"
            f"[bold]Data:[/] {cfg.data.path}",
            title="GeCCo dataset preflight",
            style="blue",
        )
    )

    check_data_and_narrative(cfg)

    baseline_out = args.baseline_out or f"/tmp/baseline_{cfg.task.name}.py"
    spec, _ = check_baseline(cfg, baseline_out)

    if spec is not None and not args.skip_recovery:
        recovery_cfg = getattr(cfg, "parameter_recovery", None)
        n_subjects = args.recovery_subjects or getattr(recovery_cfg, "n_subjects", 50)
        n_trials = args.recovery_trials or getattr(recovery_cfg, "n_trials", 100)
        check_simulator(cfg, spec, n_subjects, n_trials, args.sim_only)

    console.rule("[bold green]Preflight complete")


if __name__ == "__main__":
    main()
