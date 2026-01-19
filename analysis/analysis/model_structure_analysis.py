#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Rule-based mechanistic comparison of aligned cognitive model families
WITH theory-driven plots.

Inputs:
- Baseline models (no age): all_individual_models.txt
- Age-augmented models:    all_individual_age_models.txt

Outputs:
- model_mechanism_fingerprints.csv
- age_delta_by_model.csv
- model_delta_summaries.txt
- plots/
    - arbitration_shift_matrix.png
    - control_vs_representation.png
    - mechanism_reallocation.png

No LLMs. Pure AST + pattern rules.
"""

from __future__ import annotations
import ast, re, os, csv, argparse
from dataclasses import dataclass, asdict
from typing import Dict, List, Any

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]


# =========================
# Utilities
# =========================

def read_text(path):
    project_root = Path(__file__).resolve().parents[0]
    with open(f"{project_root}/{path}", "r", encoding="utf-8") as f:
        return f.read()

def norm(s):
    return re.sub(r"\s+", " ", s.lower())


# =========================
# Fingerprint definition
# =========================

@dataclass
class Fingerprint:
    model_name: str
    family: str

    # Core
    uses_rl: int = 0
    uses_wm: int = 0

    # Arbitration
    arbitration_type: str = "unknown"
    arbitration_dynamic: int = 0

    # Control mechanisms
    pe_gating: int = 0
    confidence_gating: int = 0
    availability_gating: int = 0
    lapse_noise: int = 0

    # Representation mechanisms
    wm_capacity: int = 0
    wm_interference: int = 0
    wm_forgetting: int = 0

    # Load effects
    load_effect: int = 0

    # Age effects
    age_effect: int = 0
    age_capacity: int = 0
    age_arbitration: int = 0
    age_forgetting: int = 0
    age_lapse: int = 0


# =========================
# Feature extraction
# =========================

class Extractor(ast.NodeVisitor):
    def __init__(self, src):
        self.txt = norm(src)

    def has(self, *patterns):
        return any(p in self.txt for p in patterns)

    def extract(self, name, family):
        f = Fingerprint(model_name=name, family=family)

        f.uses_rl = int(self.has("q[s", "delta = r -"))
        f.uses_wm = int(self.has("w[s", "wm_", "stored_action", "mem_act", "counts"))

        # Arbitration types
        if self.has("stored_action", "wm_cache", "has_mem"):
            f.arbitration_type = "availability"
            f.arbitration_dynamic = 1
            f.availability_gating = 1
        elif self.has("conf =", "conf_scaled", "sorted_w"):
            f.arbitration_type = "confidence_gated"
            f.arbitration_dynamic = 1
            f.confidence_gating = 1
        elif self.has("abs(pe", "pe_mag", "pe_sensitivity", "gate_slope"):
            f.arbitration_type = "pe_gated"
            f.arbitration_dynamic = 1
            f.pe_gating = 1
        elif self.has("wm_weight * p_wm"):
            f.arbitration_type = "fixed"

        # Representation
        f.wm_capacity = int(self.has("k_base", "k_eff", "capacity"))
        f.wm_interference = int(self.has("mean_other", "interference", "misbind"))
        f.wm_forgetting = int(self.has("decay", "forget", "leak"))

        # Noise / lapses
        f.lapse_noise = int(self.has("lapse", "slip", "noise"))

        # Load
        f.load_effect = int(self.has("nS", "set_size", "3.0 / nS"))

        # Age
        f.age_effect = int(self.has("age_group", "age_pen", "older"))
        f.age_capacity = int(f.age_effect and f.wm_capacity)
        f.age_arbitration = int(f.age_effect and f.arbitration_dynamic)
        f.age_forgetting = int(f.age_effect and f.wm_forgetting)
        f.age_lapse = int(f.age_effect and f.lapse_noise)

        return f


# =========================
# Parsing model files
# =========================

def parse_models(text, family):
    tree = ast.parse(text)
    models = {}

    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and re.match(r"p\d+_model", node.name):
            src = ast.get_source_segment(text, node)
            ex = Extractor(src)
            models[node.name] = ex.extract(node.name, family)

    return models


# =========================
# Delta + summaries
# =========================

def compute_delta(base, age):
    d = {"model_name": base.model_name}
    for k in asdict(base):
        if k in ("model_name", "family"):
            continue
        b, a = getattr(base, k), getattr(age, k)
        if isinstance(b, int):
            d[k] = a - b
        elif isinstance(b, str):
            d[k] = "" if a == b else f"{b} → {a}"
    return d


def summarize(base, age, d):
    lines = [f"{base.model_name}:"]
    if d["arbitration_type"]:
        lines.append(f"  • Arbitration shift: {d['arbitration_type']}")
    if age.age_capacity:
        lines.append("  • Adds age-dependent WM capacity limit")
    if age.age_arbitration:
        lines.append("  • Adds age-dependent control/arbitration bias")
    if age.age_forgetting:
        lines.append("  • Adds age-dependent WM forgetting")
    if age.age_lapse:
        lines.append("  • Adds age-dependent lapse/noise")
    if len(lines) == 1:
        lines.append("  • No qualitative architectural change detected")
    return "\n".join(lines)


# =========================
# Plotting
# =========================

def plot_arbitration_shift(df, outdir):
    base = df[df.family == "baseline"]
    age  = df[df.family == "age"]

    m = base.merge(age, on="model_name", suffixes=("_base", "_age"))
    ct = pd.crosstab(m["arbitration_type_base"], m["arbitration_type_age"])

    plt.figure(figsize=(6, 5))
    plt.imshow(ct, cmap="Blues")
    plt.xticks(range(len(ct.columns)), ct.columns, rotation=45)
    plt.yticks(range(len(ct.index)), ct.index)
    plt.colorbar(label="Number of models")
    plt.xlabel("Age-dependent arbitration")
    plt.ylabel("Baseline arbitration")
    plt.title("Arbitration regime shifts with age")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "arbitration_shift_matrix.png"))
    plt.close()


def plot_control_vs_representation(delta, outdir):
    control = (
        (delta["age_arbitration"] > 0) |
        (delta["age_lapse"] > 0)
    ).astype(int)

    rep = (
        (delta["age_capacity"] > 0) |
        (delta["age_forgetting"] > 0)
    ).astype(int)

    # Count quadrants
    counts = {
        (0, 0): 0,
        (1, 0): 0,
        (0, 1): 0,
        (1, 1): 0,
    }

    for c, r in zip(control, rep):
        counts[(c, r)] += 1

    fig, ax = plt.subplots(figsize=(5, 5))

    for (c, r), n in counts.items():
        ax.scatter(c, r, s=300)
        ax.text(c, r, str(n), ha="center", va="center", fontsize=14, color="white")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["No control change", "Control change"])
    ax.set_yticklabels(["No representation change", "Representation change"])

    ax.set_xlabel("Control-level age effects")
    ax.set_ylabel("Representation-level age effects")
    ax.set_title("Where age enters the cognitive architecture\n(count of models)")

    ax.set_xlim(-0.3, 1.3)
    ax.set_ylim(-0.3, 1.3)
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "control_vs_representation.png"))
    plt.close()


def plot_mechanism_reallocation(delta, outdir):
    mechs = {
        "wm_capacity": "WM capacity",
        "wm_interference": "WM interference",
        "wm_forgetting": "WM forgetting",
        "arbitration_dynamic": "Dynamic arbitration",
    }

    rows = []
    for k, label in mechs.items():
        rows.append({
            "mechanism": label,
            "added": (delta[k] > 0).sum(),
            "removed": (delta[k] < 0).sum()
        })

    df = pd.DataFrame(rows).set_index("mechanism")

    df.plot(kind="bar", figsize=(7, 4))
    plt.ylabel("Number of models")
    plt.title("Mechanism reallocation with age")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "mechanism_reallocation.png"))
    plt.close()


# =========================
# Main
# =========================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="all_individual_models.txt")
    ap.add_argument("--age", default="all_individual_age_models.txt")
    project_root = Path(__file__).resolve().parents[0]

    ap.add_argument("--outdir", default=project_root)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    project_root = Path(__file__).resolve().parents[1]
    plotdir = f"{project_root}/plots/"
    os.makedirs(plotdir, exist_ok=True)

    base_models = parse_models(read_text(args.baseline), "baseline")
    age_models  = parse_models(read_text(args.age), "age")

    names = sorted(set(base_models) & set(age_models))

    fingerprints = []
    deltas = []
    summaries = []

    for n in names:
        b, a = base_models[n], age_models[n]
        fingerprints += [asdict(b), asdict(a)]
        d = compute_delta(b, a)
        deltas.append(d)
        summaries.append(summarize(b, a, d))

    pd.DataFrame(fingerprints).to_csv(
        os.path.join(args.outdir, "model_mechanism_fingerprints.csv"),
        index=False
    )

    delta_df = pd.DataFrame(deltas)
    delta_df.to_csv(
        os.path.join(args.outdir, "age_delta_by_model.csv"),
        index=False
    )

    with open(os.path.join(args.outdir, "model_delta_summaries.txt"), "w") as f:
        f.write("\n\n".join(summaries))

    # Plots
    fp_df = pd.DataFrame(fingerprints)
    plot_arbitration_shift(fp_df, plotdir)
    plot_control_vs_representation(delta_df, plotdir)
    plot_mechanism_reallocation(delta_df, plotdir)

    print("Done.")
    print(f"Results written to: {args.outdir}")
    print(f"Plots written to: {plotdir}")


if __name__ == "__main__":
    main()


print('stop')