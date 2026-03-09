"""
Compile GeCCo results into a formatted HTML report.

Usage:
    python scripts/compile_results.py --results_dir results/two_step_factors
    python scripts/compile_results.py --results_dir results/two_step_factors --output report.html
"""

import argparse
import json
import re
from pathlib import Path
from datetime import datetime


def load_bic_files(bics_dir):
    """Load all BIC JSON files, separating iteration results from best-model files."""
    iteration_results = {}
    best_results = {}

    for f in sorted(bics_dir.glob("*.json")):
        name = f.stem
        with open(f) as fh:
            data = json.load(fh)

        if name.startswith("best_bic_"):
            best_results[name] = data
        elif name.startswith("iter"):
            # Parse iter and run from filename like "iter0_run0" or "iter0_run0_participant1"
            iteration_results[name] = data

    return iteration_results, best_results


def load_text_files(directory, extension="txt"):
    """Load all text files from a directory."""
    files = {}
    if not directory.exists():
        return files
    for f in sorted(directory.glob(f"*.{extension}")):
        files[f.stem] = f.read_text()
    return files


def load_param_files(param_dir):
    """Load parameter CSV files."""
    params = {}
    if not param_dir.exists():
        return params
    for f in sorted(param_dir.glob("*.csv")):
        params[f.stem] = f.read_text()
    return params


def extract_iter_run(name):
    """Extract (iteration, run) from a filename like 'iter2_run0'."""
    m = re.match(r"iter(\d+)_run(\d+)", name)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


def build_bic_trajectory(iteration_results):
    """Build a per-run BIC trajectory table."""
    runs = {}
    for name, data in iteration_results.items():
        it, run = extract_iter_run(name)
        if it is None:
            continue
        if run not in runs:
            runs[run] = []
        if isinstance(data, list):
            for model in data:
                runs[run].append({
                    "iteration": it,
                    "function": model.get("function_name", "?"),
                    "metric": model.get("metric_name", "BIC"),
                    "value": model.get("metric_value", None),
                    "params": model.get("param_names", []),
                })
        elif isinstance(data, dict) and "bic" in data:
            runs[run].append({
                "iteration": it,
                "function": "best",
                "metric": "BIC",
                "value": data["bic"],
                "params": [],
            })
    return runs


def escape_html(text):
    """Escape HTML special characters."""
    return (text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;"))


def generate_html(results_dir, iteration_results, best_results,
                  models, feedback, best_models, params):
    """Generate the full HTML report."""

    task_name = results_dir.name
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    trajectory = build_bic_trajectory(iteration_results)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>GeCCo Results — {escape_html(task_name)}</title>
<style>
  body {{ font-family: 'Segoe UI', system-ui, sans-serif; max-width: 960px; margin: 2em auto; padding: 0 1em; color: #1a1a1a; background: #fafafa; }}
  h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 0.3em; }}
  h2 {{ color: #2c3e50; margin-top: 2em; border-bottom: 1px solid #ddd; padding-bottom: 0.2em; }}
  h3 {{ color: #34495e; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
  th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
  th {{ background: #3498db; color: white; }}
  tr:nth-child(even) {{ background: #f2f2f2; }}
  tr:hover {{ background: #e8f4fd; }}
  .best {{ background: #d4edda !important; font-weight: bold; }}
  pre {{ background: #282c34; color: #abb2bf; padding: 1em; border-radius: 6px; overflow-x: auto; font-size: 0.85em; line-height: 1.4; }}
  .feedback-block {{ background: #fff; border-left: 4px solid #3498db; padding: 1em; margin: 1em 0; white-space: pre-wrap; font-size: 0.9em; }}
  .summary-box {{ background: #eaf2f8; border: 1px solid #3498db; border-radius: 6px; padding: 1em; margin: 1em 0; }}
  .metric {{ font-size: 1.8em; font-weight: bold; color: #2980b9; }}
  .param-tag {{ display: inline-block; background: #ecf0f1; border-radius: 3px; padding: 2px 8px; margin: 2px; font-size: 0.85em; }}
  .empty {{ color: #999; font-style: italic; }}
  .meta {{ color: #777; font-size: 0.85em; }}
</style>
</head>
<body>

<h1>GeCCo Results: {escape_html(task_name)}</h1>
<p class="meta">Generated: {now} &nbsp;|&nbsp; Results directory: <code>{escape_html(str(results_dir))}</code></p>
"""

    # --- Summary box ---
    if best_results:
        best_bic = min(v["bic"] for v in best_results.values() if "bic" in v)
        n_iterations = len(iteration_results)
        n_runs = len(set(extract_iter_run(k)[1] for k in iteration_results if extract_iter_run(k)[1] is not None))
        html += f"""
<div class="summary-box">
  <p><span class="metric">{best_bic:.2f}</span> &nbsp; Best BIC</p>
  <p>{n_iterations} iteration file(s) across {n_runs} run(s)</p>
</div>
"""
    elif not iteration_results:
        html += '<div class="summary-box"><p class="empty">No BIC results found.</p></div>\n'

    # --- BIC Trajectory ---
    html += "<h2>BIC Trajectory</h2>\n"
    if trajectory:
        for run_idx in sorted(trajectory.keys()):
            entries = sorted(trajectory[run_idx], key=lambda x: (x["iteration"], x["function"]))
            html += f"<h3>Run {run_idx}</h3>\n"
            html += "<table><tr><th>Iteration</th><th>Model</th><th>Metric</th><th>Value</th><th>Parameters</th></tr>\n"

            best_val = min((e["value"] for e in entries if e["value"] is not None), default=None)
            for e in entries:
                val = e["value"]
                val_str = f"{val:.2f}" if val is not None else "—"
                is_best = val is not None and val == best_val
                cls = ' class="best"' if is_best else ''
                param_tags = " ".join(f'<span class="param-tag">{escape_html(p)}</span>' for p in e["params"])
                html += f'<tr{cls}><td>{e["iteration"]}</td><td>{escape_html(e["function"])}</td><td>{e["metric"]}</td><td>{val_str}</td><td>{param_tags or "—"}</td></tr>\n'
            html += "</table>\n"
    else:
        html += '<p class="empty">No iteration results to display.</p>\n'

    # --- Best Models ---
    html += "<h2>Best Model Code</h2>\n"
    if best_models:
        for name, code in best_models.items():
            label = name.replace("best_model_", "Run ").replace("_", " ")
            html += f"<h3>{escape_html(label)}</h3>\n"
            html += f"<pre>{escape_html(code)}</pre>\n"
    else:
        html += '<p class="empty">No best model files found.</p>\n'

    # --- Parameters ---
    html += "<h2>Best Parameters</h2>\n"
    if params:
        for name, csv_text in params.items():
            label = name.replace("best_params_", "Run ").replace("_", " ")
            lines = csv_text.strip().split("\n")
            if lines:
                headers = lines[0].split(",")
                html += f"<h3>{escape_html(label)}</h3>\n"
                html += "<table><tr>" + "".join(f"<th>{escape_html(h)}</th>" for h in headers) + "</tr>\n"
                for row in lines[1:21]:  # Show first 20 rows
                    cols = row.split(",")
                    html += "<tr>" + "".join(f"<td>{escape_html(c)}</td>" for c in cols) + "</tr>\n"
                if len(lines) > 21:
                    html += f'<tr><td colspan="{len(headers)}" style="text-align:center; color:#999;">... {len(lines)-21} more rows</td></tr>\n'
                html += "</table>\n"
    else:
        html += '<p class="empty">No parameter files found.</p>\n'

    # --- Feedback ---
    html += "<h2>Feedback History</h2>\n"
    if feedback:
        for name in sorted(feedback.keys(), key=lambda n: (extract_iter_run(n))):
            it, run = extract_iter_run(name)
            label = f"Iteration {it}, Run {run}" if it is not None else name
            html += f"<h3>{escape_html(label)}</h3>\n"
            html += f'<div class="feedback-block">{escape_html(feedback[name])}</div>\n'
    else:
        html += '<p class="empty">No feedback files found.</p>\n'

    # --- Raw Model Outputs ---
    non_best_models = {k: v for k, v in models.items() if not k.startswith("best_model")}
    if non_best_models:
        html += "<h2>All Model Outputs</h2>\n"
        for name in sorted(non_best_models.keys(), key=lambda n: extract_iter_run(n)):
            it, run = extract_iter_run(name)
            label = f"Iteration {it}, Run {run}" if it is not None else name
            html += f"<h3>{escape_html(label)}</h3>\n"
            html += f"<details><summary>Show full LLM output</summary><pre>{escape_html(non_best_models[name])}</pre></details>\n"

    html += """
</body>
</html>
"""
    return html


def main():
    parser = argparse.ArgumentParser(description="Compile GeCCo results into an HTML report")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Path to results directory (e.g. results/two_step_factors)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output HTML file path (default: <results_dir>/report.html)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = Path(__file__).resolve().parents[1] / results_dir

    if not results_dir.exists():
        print(f"Error: results directory not found: {results_dir}")
        return

    # Load everything
    bics_dir = results_dir / "bics"
    models_dir = results_dir / "models"
    feedback_dir = results_dir / "feedback"
    param_dir = results_dir / "parameters"

    iteration_results, best_results = {}, {}
    if bics_dir.exists():
        iteration_results, best_results = load_bic_files(bics_dir)

    models = load_text_files(models_dir) if models_dir.exists() else {}
    best_models = {k: v for k, v in models.items() if k.startswith("best_model")}
    feedback = load_text_files(feedback_dir) if feedback_dir.exists() else {}
    params = load_param_files(param_dir)

    # Generate report
    html = generate_html(results_dir, iteration_results, best_results,
                         models, feedback, best_models, params)

    output_path = Path(args.output) if args.output else results_dir / "report.html"
    output_path.write_text(html)
    print(f"Report saved to: {output_path}")


if __name__ == "__main__":
    main()
