#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Exhaustive, rule-based mechanistic comparison of aligned cognitive model families.

✔ No LLMs
✔ Closed-world exhaustive taxonomy
✔ Deterministic
✔ IDE-safe
✔ Fails loudly if a new architecture appears

INPUTS (default, relative to this file):
- all_individual_models.txt
- all_individual_age_models.txt

OUTPUTS:
- model_mechanism_fingerprints.csv
- age_delta_by_model.csv
- model_delta_summaries.txt
- plots/
    - arbitration_shift_matrix.png
    - control_vs_representation.png
    - mechanism_reallocation.png
"""

from __future__ import annotations
import ast, re, os, argparse
from dataclasses import dataclass, asdict
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Paths / utilities
# ============================================================

ROOT = Path(__file__).resolve().parent

def read_text(rel_path: str) -> str:
    with open(ROOT / rel_path, "r", encoding="utf-8") as f:
        return f.read()

def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower())


# ============================================================
# Fingerprint definition
# ============================================================

@dataclass
class Fingerprint:
    model_name: str
    family: str

    # Core presence
    uses_rl: int = 0
    uses_wm: int = 0

    # Arbitration / control (EXHAUSTIVE)
    arbitration_type: str = ""

    # Control mechanisms
    dynamic_control: int = 0
    parametric_control: int = 0
    integrated_control: int = 0

    # Representation mechanisms
    wm_capacity: int = 0
    wm_interference: int = 0
    wm_forgetting: int = 0

    # Age flags
    age_effect: int = 0
    age_control: int = 0
    age_representation: int = 0


# ============================================================
# Feature extractor
# ============================================================


class Extractor:
    def __init__(self, src: str):
        self.txt = norm(src)

    def has_any(self, *patterns):
        return any(p in self.txt for p in patterns)

    def has_all(self, *patterns):
        return all(p in self.txt for p in patterns)

    def has(self, *patterns) -> bool:
        return any(p in self.txt for p in patterns)

    def _diagnose_unclassified(self, model_name: str, family: str) -> None:
        txt = self.txt

        # Minimal token probes (extend if you want)
        probes = {
            "has_Q": self.has("q[s"),
            "has_W": self.has("w[s", "wm_", "stored_action", "wm_cache", "counts"),
            "has_policy_total": self.has("p_total"),
            "has_p_rl": self.has("p_rl"),
            "has_p_wm": self.has("p_wm"),
            "has_wm_weight": self.has("wm_weight"),
            "has_gate_words": self.has("gate", "gating", "sigmoid"),
            "has_conf": self.has("conf", "confidence"),
            "has_pe": self.has("pe", "prediction error", "delta = r -"),
            "has_abs_pe": self.has("abs(pe", "pe_mag"),
            "has_surprise": self.has("surprise"),
            "has_avail": self.has("has_mem", "stored_action", "wm_cache"),
            "has_integrate": self.has("q + w", "beta * q +", "logits ="),
            "has_parametric": self.has("beta_eff", "alpha_eff", "uncert"),
            "has_softmax": self.has("softmax"),
            "has_epsilon": self.has("epsilon"),
            "has_bonus_bias": self.has("bonus", "bias", "wm_bonus"),
        }

        # Print a compact report
        print("\n" + "=" * 80)
        print("❌ UNCLASSIFIED MODEL")
        print(f"Model:  {model_name}")
        print(f"Family: {family}")
        print("-" * 80)
        print("Probe flags:")
        for k, v in probes.items():
            print(f"  {k:16s}: {v}")

        # Print arbitration-relevant lines (best effort)
        print("-" * 80)
        print("Relevant lines (token hits):")

        # If you still have original source, pass it in; otherwise this is best effort.
        # Here we reconstruct line-ish chunks by splitting on ';' and ' for ' etc.
        raw = re.sub(r"\\n", "\n", txt)
        raw_lines = raw.split("\n")
        tokens = [
            "p_total", "p_rl", "p_wm", "wm_weight", "gate", "sigmoid",
            "conf", "confidence", "pe", "abs(pe", "surprise",
            "stored_action", "wm_cache", "has_mem",
            "logits", "softmax", "epsilon", "bonus", "bias",
            "beta_eff", "alpha_eff", "uncert", "q + w", "beta * q +",
        ]
        hits = 0
        for line in raw_lines:
            if any(t in line for t in tokens):
                print("  " + line.strip())
                hits += 1
                if hits >= 60:
                    print("  ... (truncated)")
                    break

        print("=" * 80 + "\n")

        # -------------------------
        # Exhaustive arbitration classifier
        # -------------------------

    def classify_arbitration(self) -> str:
        """
        Mutually exclusive & collectively exhaustive over THIS codebase.
        Order matters. Do not reorder without understanding implications.
        """

        # ------------------------------------------------------------
        # 0. Learning-only control
        # WM affects learning dynamics but NEVER enters the policy
        # ------------------------------------------------------------
        if (
                self.has("q[s") and
                self.has("alpha", "learn", "update") and
                not self.has("p_wm", "wm_weight", "logits", "p =", "softmax")
        ):
            return "learning_only_control"

        # ------------------------------------------------------------
        # 1. RL only
        # WM exists at most as bookkeeping, never affects learning or choice
        # ------------------------------------------------------------
        if self.has("q[s") and not self.has("wm_", "w[s", "stored_action"):
            return "rl_only"

        # ------------------------------------------------------------
        # 2. WM only
        # ------------------------------------------------------------
        if self.has("w[s") and not self.has("q[s"):
            return "wm_only"

        # ------------------------------------------------------------
        # 3. Structural WM dominance
        # WM always overrides RL when available
        # ------------------------------------------------------------
        if self.has("stored_action", "wm_cache") and not self.has("wm_weight", "gate"):
            return "structural_wm_dominant"

        # ------------------------------------------------------------
        # 4. Fixed mixture
        # Constant WM–RL weighting, no trial-wise modulation
        # ------------------------------------------------------------
        if self.has_all("p_total", "p_wm", "p_rl") and self.has_any("wm_weight", "w_weight"):
            # If wm_weight is computed as a function of PE/conf/etc, it's gated; otherwise fixed.
            if self.has_any("wm_weight =") and self.has_any("sigmoid", "exp(") and self.has_any("pe", "conf",
                                                                                                "confidence",
                                                                                                "surprise", "abs("):
                # We'll let the gated rules below catch it.
                pass
            else:
                return "fixed_mixture"

        # ------------------------------------------------------------
        # 5. Signal-gated arbitration
        # ------------------------------------------------------------
        if self.has_any("wm_weight") and self.has_any("conf", "confidence") and self.has_any("exp(", "sigmoid"):
            return "confidence_gated"

        if self.has_any("wm_weight") and self.has_any("pe =", "abs(pe", "pe_mag", "delta = r -") and self.has_any(
                "exp(", "sigmoid"):
            # If abs/unsigned signal appears, call it surprise; else PE
            if self.has_any("abs(pe", "pe_mag"):
                return "surprise_gated"
            return "pe_gated"
        if self.has("surprise") and self.has("wm_weight"):
            return "surprise_gated"

        # ------------------------------------------------------------
        # 6. Availability-based arbitration
        # ------------------------------------------------------------
        if self.has("has_mem", "stored_action") and self.has("if"):
            return "availability_gated"

        # ------------------------------------------------------------
        # 7. Policy-space mixing (implicit arbitration)
        # Mixing happens AFTER softmax, no explicit gate
        # ------------------------------------------------------------
        if (
                self.has("p_rl", "p_wm") and
                self.has("+", "*", "epsilon") and
                not self.has("wm_weight", "gate", "sigmoid")
        ):
            return "policy_mixing"

        # ------------------------------------------------------------
        # 8. Parametric control
        # WM modulates learning rate or temperature
        # ------------------------------------------------------------
        if self.has("beta_eff", "alpha_eff", "uncert", "confidence"):
            return "parametric_control"

        # ------------------------------------------------------------
        # 9. Integrated values
        # WM and RL fused in value or logit space
        # ------------------------------------------------------------
        if self.has("q + w", "beta * q +", "logits ="):
            return "integrated_values"

        # ------------------------------------------------------------
        # CLOSED-WORLD GUARANTEE
        # ------------------------------------------------------------

        self._diagnose_unclassified(model_name=getattr(self, "_model_name", "UNKNOWN"),
                                    family=getattr(self, "_family", "UNKNOWN"))

        raise RuntimeError(
            "Unclassifiable arbitration structure detected.\n"
            "This indicates a genuinely new control architecture."
        )

    # -------------------------
    # Full fingerprint extraction
    # -------------------------

    def extract(self, name: str, family: str) -> Fingerprint:
        self._model_name = name
        self._family = family

        f = Fingerprint(model_name=name, family=family)

        f.uses_rl = int(self.has("q[s", "delta = r -"))
        f.uses_wm = int(self.has("w[s", "wm_", "stored_action", "counts"))

        f.arbitration_type = self.classify_arbitration()

        # Control style
        f.dynamic_control = int("gated" in f.arbitration_type)
        f.parametric_control = int(f.arbitration_type == "parametric_control")
        f.integrated_control = int(f.arbitration_type == "integrated_values")

        # Representation
        f.wm_capacity = int(self.has("k_base", "k_eff", "capacity"))
        f.wm_interference = int(self.has("mean_other", "misbind", "interference"))
        f.wm_forgetting = int(self.has("decay", "forget", "leak"))

        # Age
        f.age_effect = int(self.has("age_group", "age_pen", "older"))
        f.age_control = int(f.age_effect and f.dynamic_control)
        f.age_representation = int(
            f.age_effect and (f.wm_capacity or f.wm_forgetting)
        )

        return f


# ============================================================
# Parsing
# ============================================================

def parse_models(text: str, family: str) -> dict[str, Fingerprint]:
    tree = ast.parse(text)
    out = {}

    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and re.fullmatch(r"p\d+_model", node.name):
            src = ast.get_source_segment(text, node)
            ex = Extractor(src)
            try:
                out[node.name] = ex.extract(node.name, family)
            except RuntimeError as e:
                print("\n❌ Arbitration classification failed")
                print("Model:", node.name)
                print("Family:", family)
                print("---- MODEL SOURCE (truncated) ----")
                src = ast.get_source_segment(text, node)
                print(src[:1500])  # first 1500 chars
                print("--------------------------------")
                raise

    return out


# ============================================================
# Delta + summaries
# ============================================================

def compute_delta(b: Fingerprint, a: Fingerprint) -> dict:
    d = {"model_name": b.model_name}
    for k in asdict(b):
        if k in ("model_name", "family"):
            continue
        bv, av = getattr(b, k), getattr(a, k)
        if isinstance(bv, int):
            d[k] = av - bv
        else:
            d[k] = "" if av == bv else f"{bv} → {av}"
    return d


def summarize(b: Fingerprint, a: Fingerprint, d: dict) -> str:
    lines = [f"{b.model_name}:"]
    if d["arbitration_type"]:
        lines.append(f"  • Control regime shift: {d['arbitration_type']}")
    if a.age_control:
        lines.append("  • Age modifies control allocation")
    if a.age_representation:
        lines.append("  • Age modifies memory representation")
    if len(lines) == 1:
        lines.append("  • No qualitative structural change")
    return "\n".join(lines)


# ============================================================
# Plotting
# ============================================================

def plot_arbitration_shift(fp: pd.DataFrame, outdir: Path):
    b = fp[fp.family == "baseline"]
    a = fp[fp.family == "age"]
    m = b.merge(a, on="model_name", suffixes=("_base", "_age"))

    ct = pd.crosstab(m["arbitration_type_base"], m["arbitration_type_age"])

    plt.figure(figsize=(7, 6))
    plt.imshow(ct, cmap="Blues")
    plt.xticks(range(len(ct.columns)), ct.columns, rotation=45, ha="right")
    plt.yticks(range(len(ct.index)), ct.index)
    plt.colorbar(label="Number of models")
    plt.xlabel("Age-dependent control regime")
    plt.ylabel("Baseline control regime")
    plt.title("Control regime transitions with age")
    plt.tight_layout()
    plt.savefig(outdir / "arbitration_shift_matrix.png")
    plt.close()


def plot_control_vs_representation(delta: pd.DataFrame, outdir: Path):
    control = (delta["age_control"] > 0).astype(int)
    rep = (delta["age_representation"] > 0).astype(int)

    counts = {(0,0):0, (1,0):0, (0,1):0, (1,1):0}
    for c, r in zip(control, rep):
        counts[(c,r)] += 1

    plt.figure(figsize=(5,5))
    for (c,r), n in counts.items():
        plt.scatter(c, r, s=400)
        plt.text(c, r, str(n), ha="center", va="center",
                 fontsize=14, color="white")

    plt.xticks([0,1], ["No control change", "Control change"])
    plt.yticks([0,1], ["No representation change", "Representation change"])
    plt.xlabel("Control-level age effects")
    plt.ylabel("Representation-level age effects")
    plt.title("Where age enters the architecture (model counts)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outdir / "control_vs_representation.png")
    plt.close()


def plot_mechanism_reallocation(delta: pd.DataFrame, outdir: Path):
    rows = []
    for mech, label in [
        ("dynamic_control", "Dynamic control"),
        ("parametric_control", "Parametric control"),
        ("integrated_control", "Integrated control"),
        ("wm_capacity", "WM capacity"),
        ("wm_interference", "WM interference"),
        ("wm_forgetting", "WM forgetting"),
    ]:
        rows.append({
            "mechanism": label,
            "added": (delta[mech] > 0).sum(),
            "removed": (delta[mech] < 0).sum()
        })

    df = pd.DataFrame(rows).set_index("mechanism")
    df.plot(kind="bar", figsize=(8,4))
    plt.ylabel("Number of models")
    plt.title("Mechanism reallocation with age")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(outdir / "mechanism_reallocation.png")
    plt.close()


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="all_individual_models.txt")
    ap.add_argument("--age", default="all_individual_age_models.txt")

    project_root = Path(__file__).resolve().parents[1]

    args = ap.parse_args()

    outdir = project_root
    plotdir = outdir / "plots"
    outdir.mkdir(exist_ok=True)
    plotdir.mkdir(exist_ok=True)

    base = parse_models(read_text(args.baseline), "baseline")
    age  = parse_models(read_text(args.age), "age")

    names = sorted(set(base) & set(age))

    fps, deltas, summaries = [], [], []

    for n in names:
        b, a = base[n], age[n]
        fps += [asdict(b), asdict(a)]
        d = compute_delta(b, a)
        deltas.append(d)
        summaries.append(summarize(b, a, d))

    fp_df = pd.DataFrame(fps)
    delta_df = pd.DataFrame(deltas)

    fp_df.to_csv(outdir / "model_mechanism_fingerprints.csv", index=False)
    delta_df.to_csv(outdir / "age_delta_by_model.csv", index=False)

    with open(outdir / "model_delta_summaries.txt", "w") as f:
        f.write("\n\n".join(summaries))

    plot_arbitration_shift(fp_df, plotdir)
    plot_control_vs_representation(delta_df, plotdir)
    plot_mechanism_reallocation(delta_df, plotdir)

    print("✔ Analysis complete")
    print(f"Results: {outdir}")
    print(f"Plots:   {plotdir}")


if __name__ == "__main__":
    main()
