"""Internal CLI route for test evaluation post-processing."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from config.schema import load_config
from gecco.coordination import SharedRegistry
from gecco.cli.run_gecco_distributed import _split_prompt_train_test
from gecco.offline_evaluation.fit_generated_models import (
    run_fit_hierarchical as run_fit,
)
from gecco.offline_evaluation.utils import build_model_spec
from gecco.prepare_data.io import load_data
from gecco.tempdirs import configure_temp_dirs


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def register_parser(subparsers) -> argparse.ArgumentParser:
    """Register the internal test evaluation subcommand."""
    parser = subparsers.add_parser(
        "test-evaluation", help=argparse.SUPPRESS, description=argparse.SUPPRESS
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--write-store", action="store_true")
    parser.set_defaults(handler=main)
    return parser


def load_splits(cfg):
    """Replicate the exact split logic from the distributed client."""
    data_cfg = cfg.data
    df = load_data(data_cfg.path, data_cfg.input_columns)
    _, _, df_test, _, _ = _split_prompt_train_test(df, data_cfg, cfg.evaluation)
    return df_test


def _selection_metric_field(cfg) -> str:
    """Return the development metric field to rank candidates by."""
    metric = str(getattr(cfg.evaluation, "metric", "BIC")).lower()
    return "mean_nll" if "nll" in metric else "metric_value"


def _selection_metric_value(candidate: dict, cfg) -> float | None:
    """Return a finite ranking score from the chosen development metric."""
    field = _selection_metric_field(cfg)
    value = candidate.get(field)
    if value is None:
        return None
    try:
        if not np.isfinite(value):
            return None
    except TypeError:
        return None
    return float(value)


def collect_candidates(registry, cfg):
    """Collect unique candidates with finite development scores."""
    data = registry.read()
    candidates = {}
    selection_field = _selection_metric_field(cfg)
    for entry in data.get("iteration_history", []):
        client_id = entry.get("client_id")
        iteration = entry.get("iteration")
        for result in entry.get("results", []):
            score = _selection_metric_value(result, cfg)
            if score is None:
                continue
            name = result.get("function_name", "")
            code = result.get("code", "")
            candidate_key = (
                client_id,
                iteration,
                name,
                hashlib.sha256(code.encode("utf-8")).hexdigest() if code else "",
            )
            candidate = {
                "client_id": client_id,
                "iteration": iteration,
                "function_name": name,
                "executable_function_name": result.get("executable_function_name")
                or result.get("func_name"),
                "code": code,
                "selection_metric_name": selection_field,
                "selection_metric_value": score,
                "param_names": result.get("param_names", []),
            }
            if candidate_key not in candidates or score < candidates[candidate_key][
                "selection_metric_value"
            ]:
                candidates[candidate_key] = candidate
    return sorted(
        candidates.values(),
        key=lambda candidate: (
            candidate["selection_metric_value"],
            str(candidate.get("client_id", "")),
            -1 if candidate.get("iteration") is None else int(candidate.get("iteration")),
            candidate.get("function_name", ""),
        ),
    )


def _resolve_executable_function_name(candidate, cfg):
    """Resolve the callable name to execute for a candidate record."""
    explicit_name = candidate.get("executable_function_name") or candidate.get("func_name")
    if explicit_name:
        return explicit_name

    code = candidate.get("code", "")
    if not code:
        return candidate.get("function_name", "")

    display_name = candidate.get("function_name") or "cognitive_model"
    try:
        spec = build_model_spec(code, expected_func_name=display_name, cfg=cfg)
    except Exception:
        return display_name
    return getattr(spec.func, "__name__", display_name)


def fit_one_on_test(candidate, df_test, cfg, id_eval_data=None):
    """Fit a single candidate model on the test split."""
    func_name = candidate["function_name"]
    expected_func_name = _resolve_executable_function_name(candidate, cfg)
    code = candidate["code"]
    if not code:
        return None
    try:
        fit_res = run_fit(df_test, code, cfg=cfg, expected_func_name=expected_func_name)
    except Exception as exc:
        print(f"[test] skipping {func_name}: {exc}")
        return None

    entry = {
        "model_name": func_name,
        "selection_metric_name": candidate.get("selection_metric_name"),
        "selection_metric_value": candidate.get("selection_metric_value"),
        "val_nll": candidate.get("val_mean_nll"),
        "test_mean_BIC": float(fit_res["metric_value"]),
        "test_mean_NLL": float(fit_res["mean_nll"]),
        "test_individual_BIC": fit_res["eval_metrics"],
        "test_individual_NLL": fit_res["per_participant_nll"],
        "test_individual_differences": None,
    }
    if id_eval_data is not None and hasattr(cfg, "individual_differences_eval"):
        try:
            from gecco.offline_evaluation.individual_differences import (
                evaluate_individual_differences,
            )

            id_results = evaluate_individual_differences(
                fit_res, df_test, cfg, id_data=id_eval_data
            )
            entry["test_individual_differences"] = {
                "mean_r2": id_results.get("mean_r2"),
                "max_r2": id_results.get("max_r2"),
                "best_param": id_results.get("best_param"),
                "per_param_r2": id_results.get("per_param_r2"),
                "per_param_detail": id_results.get("per_param_detail", {}),
            }
        except Exception as exc:
            print(f"[test] Individual differences eval failed for {func_name}: {exc}")
    return entry


def _format_optional_float(value) -> str:
    if value is None:
        return "n/a"
    try:
        if not np.isfinite(value):
            return "n/a"
    except TypeError:
        return "n/a"
    return f"{float(value):.2f}"


def run_test_evaluation(
    *, config: str, results_dir: str, write_store: bool = False
) -> int | None:
    """Run the test evaluation post-processing pipeline."""
    configure_temp_dirs(PROJECT_ROOT, prefix="test-eval")

    config_path = config
    if not os.path.isabs(config_path) and not config_path.endswith(".yaml"):
        config_path = config_path + ".yaml"
    if not os.path.isabs(config_path):
        config_path = PROJECT_ROOT / "config" / config_path
    cfg = load_config(config_path)

    resolved_results_dir = Path(results_dir)
    registry_path = resolved_results_dir / "shared_registry.duckdb"
    if not registry_path.exists():
        print(f"[test] ERROR: Registry not found: {registry_path}")
        raise SystemExit(1)
    registry = SharedRegistry.open_existing(registry_path)

    id_eval_data = None
    if hasattr(cfg, "individual_differences_eval"):
        try:
            from gecco.offline_evaluation.individual_differences import load_id_data

            id_eval_data = load_id_data(cfg)
            print("[test] Loaded individual differences data")
        except Exception as exc:
            print(f"[test] Warning: Could not load individual differences data: {exc}")

    df_test = load_splits(cfg)
    print(f"[test] Loaded test split: {len(df_test)} rows")

    candidates = collect_candidates(registry, cfg)
    print(f"[test] Found {len(candidates)} unique candidates with valid development scores")

    n_top = getattr(cfg.evaluation, "n_test_models", 10)
    top = candidates[:n_top]
    print(f"[test] Will evaluate top {len(top)} models on test split")

    baseline = registry.read().get("baseline")
    if baseline and baseline.get("code"):
        baseline_score = _selection_metric_value(baseline, cfg)
        top = [
            {
                "client_id": "baseline",
                "iteration": -1,
                "function_name": baseline.get("function_name", "baseline_model"),
                "code": baseline["code"],
                "selection_metric_name": _selection_metric_field(cfg),
                "selection_metric_value": baseline_score,
                "val_mean_nll": baseline.get("val_mean_nll", float("nan")),
                "param_names": baseline.get("param_names", []),
                "executable_function_name": baseline.get("executable_function_name"),
            }
        ] + top
        print("[test] Added baseline to evaluation list")

    results = []
    for cand in top:
        entry = fit_one_on_test(cand, df_test, cfg, id_eval_data=id_eval_data)
        if entry is not None:
            results.append(entry)
            selection_text = _format_optional_float(entry.get("selection_metric_value"))
            val_nll_text = _format_optional_float(entry.get("val_nll"))
            print(
                f"[test] {entry['model_name']}: score={selection_text}, val_nll={val_nll_text}, "
                f"test_BIC={entry['test_mean_BIC']:.2f}, test_NLL={entry['test_mean_NLL']:.2f}"
            )

    out_path = resolved_results_dir / "bics" / "top_models_test.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as file_obj:
        json.dump(results, file_obj, indent=2)
    print(f"[test] Wrote {len(results)} entries to {out_path}")

    if write_store:
        try:
            from gecco.diagnostic_store.store import DiagnosticStore

            db_path = resolved_results_dir / "diagnostics.duckdb"
            store = DiagnosticStore(str(db_path))
            for entry in results:
                store.write_top_model_test(entry)
            store.close()
            print(f"[test] Wrote {len(results)} entries to diagnostic store: {db_path}")
        except Exception as exc:
            print(f"[test] Warning: Could not write to diagnostic store: {exc}")
            print("[test] Re-run with --write-store to persist the diagnostic store")
    else:
        print(
            "[test] Diagnostic store persistence is disabled; pass --write-store "
            "to persist the diagnostic store"
        )
    return None


def main(args: argparse.Namespace) -> int | None:
    """Run the test evaluation command from parsed CLI arguments."""
    return run_test_evaluation(
        config=args.config,
        results_dir=args.results_dir,
        write_store=args.write_store,
    )
