"""
Preprocess Xia et al. (2021) "butterfly" task data for GeCCo.

The butterfly task is a probabilistic stimulus-response learning task: on each
trial one of four butterflies (stimulus `s`) is shown, the participant chooses
one of two flowers (action `a`), and receives reward `r` (probabilistic).

Input is a SINGLE file with every participant's trials stacked (one row per
trial, identified by `sID`), so no concatenation is needed — this script only
reshapes it into GeCCo's conventions.

Reads:
  - data_xia_2021_butterfly.csv  (in this directory)

Writes:
  - ../preprocessed_xia_2021_butterfly.csv          (trial data for GeCCo)
  - ../preprocessed_xia_2021_butterfly_demographics.csv  (one row per participant)

Run from this directory:
    python preprocess_xia.py
"""

import os
import pandas as pd

# =========================================================================
# Configuration
# =========================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_FILE = os.path.join(SCRIPT_DIR, "data_xia_2021_butterfly.csv")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "..", "preprocessed_xia_2021_butterfly.csv")
DEMOGRAPHICS_FILE = os.path.join(
    SCRIPT_DIR, "..", "preprocessed_xia_2021_butterfly_demographics.csv"
)

# Per-trial columns kept for GeCCo. s/a/r are what the model fits; rt/acc are
# covariates. Demographics are split out into a separate per-participant file.
TRIAL_COLUMNS = ["s", "a", "r", "acc", "rt"]
DEMOGRAPHIC_COLUMNS = [
    "age", "sex", "pds", "t1", "t3", "t4",
    "group", "agegroup", "pdsgroup", "t1group",
]


def main():
    print("=" * 60)
    print("Preprocessing Xia et al. (2021) butterfly data for GeCCo")
    print("=" * 60)

    df = pd.read_csv(RAW_FILE)
    print(f"Loaded {len(df)} rows, {df['sID'].nunique()} participants")

    # --- Identifiers ---------------------------------------------------------
    df["subject_id"] = df["sID"]
    # Contiguous 0-based integer participant id (sID order preserved)
    df["participant"] = pd.factorize(df["sID"])[0]

    # --- Missed-response trials (a == -1 / r == -1) are KEPT and left as -1 ---
    # matching the study2 convention; the model fitting step skips them (they
    # can't be modelled). Only valid actions are 0-indexed so -1 stays -1.
    n_missed = int((df["a"] == -1).sum())
    print(f"Missed-response trials kept and flagged as -1: {n_missed}")

    # --- 0-index stimulus, action, and trial --------------------------------
    df["s"] = df["s"].astype(int) - 1          # 1..4 -> 0..3 (always shown)
    df["a"] = df["a"].astype(int)
    df.loc[df["a"] != -1, "a"] -= 1            # 1,2 -> 0,1; missed stays -1
    df["r"] = df["r"].astype(int)              # 0/1; missed stays -1
    df["trial"] = df["trial"].astype(int) - 1  # 1..120 -> 0..119

    # --- Trial output: identifiers first, then trial, then task columns ------
    lead = ["participant", "subject_id", "trial"]
    trial_df = df[lead + TRIAL_COLUMNS].copy()

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    trial_df.to_csv(OUTPUT_FILE, index=False)

    # --- Demographics output: one row per participant ------------------------
    demo_cols = ["participant", "subject_id"] + [
        c for c in DEMOGRAPHIC_COLUMNS if c in df.columns
    ]
    demo_df = df[demo_cols].drop_duplicates(subset="subject_id").reset_index(drop=True)
    demo_df.to_csv(DEMOGRAPHICS_FILE, index=False)

    # --- Report --------------------------------------------------------------
    print("\nPost-clean unique values:")
    for col in ["s", "a", "r"]:
        print(f"  {col}: {sorted(trial_df[col].unique())}")
    print(f"\nTrial data saved to: {OUTPUT_FILE}")
    print(f"  Shape: {trial_df.shape}")
    print(f"  Participants: {trial_df['participant'].nunique()}")
    print(f"  Mean trials/participant: {trial_df.groupby('participant').size().mean():.1f}")
    print(f"  Fraction reward: {trial_df['r'].mean():.3f}")
    print(f"\nDemographics saved to: {DEMOGRAPHICS_FILE}")
    print(f"  Shape: {demo_df.shape}")


if __name__ == "__main__":
    main()
