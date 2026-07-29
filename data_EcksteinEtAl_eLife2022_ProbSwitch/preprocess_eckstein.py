"""
Preprocess Eckstein et al. (eLife 2022) ProbSwitch data for GeCCo.

The task is a single-stage 2-armed probabilistic reversal-learning bandit:
on each trial the participant chooses one of two boxes and either gets a
reward or not. The rewarded side reverses unpredictably across blocks.

Reads:
  - PS_*.csv (one raw trial-level file per participant, in this directory)

Outputs:
  - ../data/preprocessed_eckstein.csv  (one row per trial per participant)

Output format mirrors data_g_2019/preprocess_study2.py: identifier columns
(participant, subject_id) first, then a 0-indexed trial column, then the
task columns GeCCo fits to (selected_box, reward) plus a few covariates.

Run from this directory:
    python preprocess_eckstein.py
"""

import os
import glob
import pandas as pd

# =========================================================================
# Configuration
# =========================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_TASK_DIR = SCRIPT_DIR                 # PS_*.csv live alongside this script
RAW_GLOB = "PS_*.csv"
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "..", "data", "preprocessed_eckstein.csv")

# Restrict to the paper's analysis sample. 306 participants have trial files,
# but Eckstein et al. (2022, Dev Cogn Neurosci) analysed N=291 — exactly the
# subjects present in the subject-level summary table, so that table defines the
# inclusion set. The 15 extras are the authors' exclusions. This also brings the
# age range in line with the published 8-30 (7.8-30.0 for the 291, vs 7.8-50.0
# for all 306), which matters because the individual-differences analysis
# regresses fitted parameters on age.
ANALYSIS_SAMPLE_FILE = os.path.join(
    SCRIPT_DIR, "EcksteinEtAl_eLife2022_ProbSwitch.csv"
)
RESTRICT_TO_ANALYSIS_SAMPLE = True

# Columns kept from each raw file. selected_box + reward are what GeCCo fits;
# the rest are useful covariates. Lag features (outcome_*_back, choice_*_back)
# are dropped — the model is meant to learn dynamics from the sequence.
KEEP_COLUMNS = [
    "TrialID",
    "selected_box",
    "reward",
    "block",
    "switch_trial",
    "trialsinceswitch",
    "RT",
    "correct_box",
]


# =========================================================================
# Step 1: Load and combine per-participant task files
# =========================================================================
def load_analysis_sample():
    """Subject ids the paper analysed, taken from the subject-level summary."""
    if not RESTRICT_TO_ANALYSIS_SAMPLE:
        return None
    summary = pd.read_csv(ANALYSIS_SAMPLE_FILE)
    ids = set(summary["sID"].astype(int))
    print(f"Analysis sample: {len(ids)} subjects from {os.path.basename(ANALYSIS_SAMPLE_FILE)}")
    return ids


def combine_task_files(raw_dir, raw_glob, analysis_sample=None):
    files = sorted(glob.glob(os.path.join(raw_dir, raw_glob)))
    print(f"Found {len(files)} participant files matching {raw_glob}")

    frames = []
    n_excluded = 0
    # participant is a contiguous 0-based index, assigned after exclusions so
    # there are no gaps (splits index into sorted unique ids).
    next_participant = 0
    for file_path in files:
        # Subject id is the number in the filename, e.g. PS_100.csv -> 100
        subj_id = os.path.basename(file_path).replace("PS_", "").replace(".csv", "")

        if analysis_sample is not None and int(subj_id) not in analysis_sample:
            n_excluded += 1
            continue

        df = pd.read_csv(file_path)

        missing = [c for c in KEEP_COLUMNS if c not in df.columns]
        if missing:
            print(f"  Warning: {os.path.basename(file_path)} missing {missing}, skipping")
            continue

        df = df[KEEP_COLUMNS].copy()
        df["subject_id"] = subj_id
        df["participant"] = next_participant
        next_participant += 1
        frames.append(df)

    if n_excluded:
        print(f"  Excluded {n_excluded} subjects not in the paper's analysis sample")

    combined = pd.concat(frames, ignore_index=True)
    print(f"Combined {len(frames)} participants, {len(combined)} total trials")
    return combined


# =========================================================================
# Step 2: Clean and remap values
# =========================================================================
def clean_values(df):
    # selected_box and reward must be integer. Missed-response trials (non-numeric
    # "NA") are KEPT but flagged as -1, matching the study2 convention; the model
    # fitting step is responsible for skipping them (they can't be modelled).
    for col in ["selected_box", "reward"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(-1).astype(int)

    n_missed = int((df["selected_box"] == -1).sum())
    print(f"  Missed-response trials kept and flagged as -1: {n_missed}")

    # 0-index trials to match the study2 convention (TrialID starts at 1)
    df["trial"] = pd.to_numeric(df["TrialID"], errors="coerce").astype(int) - 1
    df = df.drop(columns=["TrialID"])

    # Reorder: identifiers first, then trial, then task/covariate columns
    lead = ["participant", "subject_id", "trial"]
    rest = [c for c in df.columns if c not in lead]
    df = df[lead + rest]
    return df


# =========================================================================
# Main
# =========================================================================
def main():
    print("=" * 60)
    print("Preprocessing Eckstein et al. (2022) ProbSwitch data for GeCCo")
    print("=" * 60)

    analysis_sample = load_analysis_sample()
    df = combine_task_files(RAW_TASK_DIR, RAW_GLOB, analysis_sample=analysis_sample)

    print("\nCleaning values...")
    df = clean_values(df)

    print("\nPost-clean unique values:")
    for col in ["selected_box", "reward"]:
        print(f"  {col}: {sorted(df[col].unique())}")

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"\nFinal output saved to: {OUTPUT_FILE}")
    print(f"Shape: {df.shape}")
    print(f"Participants: {df['participant'].nunique()}")
    print(f"Mean trials/participant: {df.groupby('participant').size().mean():.1f}")
    print(f"Fraction reward: {df['reward'].mean():.3f}")


if __name__ == "__main__":
    main()
