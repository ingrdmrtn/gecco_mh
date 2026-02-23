"""
Preprocess Gillan et al. Study 2 data for GeCCo with transdiagnostic factors.

Reads:
  - self_report_study2.csv (1,413 participants, 9 questionnaires + 3 factors)
  - twostep_data_study2_individual_csv/ (1,712 raw task files)

Outputs:
  - ../data/two_step_gillan_2016_study2_transdiagnostic.csv

Run from the data_g_2019/ directory:
    python preprocess_study2.py
"""

import os
import pandas as pd
import numpy as np
from io import StringIO
from scipy.optimize import minimize

# =========================================================================
# Configuration
# =========================================================================
SELF_REPORT_FILE = "self_report_study2.csv"
RAW_TASK_DIR = "twostep_data_study2_individual_csv"
OUTPUT_FILE = "../data/two_step_gillan_2016_study2_transdiagnostic.csv"

COLUMN_NAMES = [
    "trial_num",
    "drift_1",
    "drift_2",
    "drift_3",
    "drift_4",
    "stage_1_response",
    "stage_1_selected_stimulus",
    "stage_1_rt",
    "transition",
    "stage_2_response",
    "stage_2_selected_stimulus",
    "stage_2_state",
    "stage_2_rt",
    "reward",
    "redundant",
]


# =========================================================================
# Step 1: Load and normalize self-report data
# =========================================================================
def load_self_report(path):
    df = pd.read_csv(path, index_col=0)
    print(f"Loaded self-report data: {df.shape[0]} participants")

    # Rename factors
    df = df.rename(columns={
        "Factor1": "ad_raw",
        "Factor2": "cit_raw",
        "Factor3": "sw_raw",
    })

    # Min-max normalize each factor to [0, 1]
    for raw_col, norm_col in [("ad_raw", "ad"), ("cit_raw", "cit"), ("sw_raw", "sw")]:
        col_min = df[raw_col].min()
        col_max = df[raw_col].max()
        df[norm_col] = (df[raw_col] - col_min) / (col_max - col_min)

    print(f"Factor ranges after normalization:")
    for col in ["ad", "cit", "sw"]:
        print(f"  {col}: [{df[col].min():.3f}, {df[col].max():.3f}]")

    return df


# =========================================================================
# Step 2: Match participants to task files
# =========================================================================
def match_participants(self_report_df, raw_task_dir):
    task_files = [f for f in os.listdir(raw_task_dir) if f.endswith(".csv")]
    task_subj_ids = {f.replace(".csv", "") for f in task_files}
    report_subj_ids = set(self_report_df["subj"].values)

    matched = report_subj_ids & task_subj_ids
    missing_task = report_subj_ids - task_subj_ids
    extra_task = task_subj_ids - report_subj_ids

    print(f"\nParticipant matching:")
    print(f"  Self-report participants: {len(report_subj_ids)}")
    print(f"  Task files: {len(task_subj_ids)}")
    print(f"  Matched: {len(matched)}")
    print(f"  In self-report but no task file: {len(missing_task)}")
    print(f"  Task file but no self-report: {len(extra_task)}")

    return sorted(matched)


# =========================================================================
# Step 3: Clean a single raw task file
# =========================================================================
def clean_task_file(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()

    # Find the LAST instructionLoop line
    last_marker_idx = -1
    for i, line in enumerate(lines):
        if "instructionLoop" in line:
            last_marker_idx = i

    if last_marker_idx == -1:
        print(f"  Warning: No instructionLoop marker in {os.path.basename(file_path)}")
        return None

    # Extract data lines after the last marker
    # Skip blank lines and non-data lines (some files have extra metadata)
    data_lines = []
    for line in lines[last_marker_idx + 1:]:
        stripped = line.strip()
        if stripped and stripped[0].isdigit():
            data_lines.append(line)

    if not data_lines:
        print(f"  Warning: No data after marker in {os.path.basename(file_path)}")
        return None

    data_str = "".join(data_lines)
    df = pd.read_csv(StringIO(data_str), header=None, names=COLUMN_NAMES)
    return df


# =========================================================================
# Step 4 & 5: Combine all participants and merge with factor scores
# =========================================================================
def combine_and_merge(matched_ids, raw_task_dir, self_report_df):
    frames = []
    skipped = 0

    for idx, subj_id in enumerate(matched_ids):
        file_path = os.path.join(raw_task_dir, f"{subj_id}.csv")
        df = clean_task_file(file_path)

        if df is None:
            skipped += 1
            continue

        df["subject_id"] = subj_id
        df["participant"] = idx

        # Reorder: identifiers first
        cols = ["participant", "subject_id"] + [c for c in df.columns if c not in ["participant", "subject_id"]]
        df = df[cols]
        frames.append(df)

        if (idx + 1) % 200 == 0:
            print(f"  Processed {idx + 1}/{len(matched_ids)} participants...")

    print(f"\n  Total participants processed: {len(frames)}")
    print(f"  Skipped: {skipped}")

    combined = pd.concat(frames, ignore_index=True)

    # Merge with self-report data
    survey_cols = ["subj", "stai_total", "sds_total", "oci_total", "lsas_total",
                   "bis_total", "scz_total", "aes_total", "eat_total", "audit_total",
                   "ad", "cit", "sw"]
    survey_subset = self_report_df[survey_cols].copy()
    survey_subset = survey_subset.rename(columns={"subj": "subject_id"})

    merged = pd.merge(combined, survey_subset, on="subject_id", how="left")

    missing = merged["ad"].isna().sum()
    if missing > 0:
        print(f"  Warning: {missing} rows have missing factor scores")
    else:
        print("  All rows matched with factor scores")

    return merged


# =========================================================================
# Step 6: Remap values
# =========================================================================
def remap_values(df):
    response_map = {"left": 0, "right": 1}
    state_map = {2: 0, 3: 1}

    # 0-index trials
    if "trial_num" in df.columns:
        df["trial_num"] = df["trial_num"] - 1

    if "stage_1_response" in df.columns:
        df["stage_1_response"] = df["stage_1_response"].replace(response_map)

    if "stage_2_response" in df.columns:
        df["stage_2_response"] = df["stage_2_response"].replace(response_map)

    if "stage_2_state" in df.columns:
        df["stage_2_state"] = pd.to_numeric(df["stage_2_state"], errors="coerce").fillna(-1).astype(int)
        df["stage_2_state"] = df["stage_2_state"].replace(state_map)

    rename_map = {
        "stage_1_response": "choice_1",
        "stage_2_response": "choice_2",
        "stage_2_state": "state",
        "trial_num": "trial",
    }
    df = df.rename(columns=rename_map)

    # Convert to int — any remaining non-numeric values become -1
    for col in ["choice_1", "choice_2", "state"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(-1).astype(int)

    # Verify
    print("\nPost-remap unique values:")
    for col in ["choice_1", "choice_2", "state"]:
        if col in df.columns:
            print(f"  {col}: {sorted(df[col].unique())}")

    return df


# =========================================================================
# Step 7: Baseline hybrid model fitting
# =========================================================================
def hybrid_model(action_1, state, action_2, reward, model_parameters):
    """
    Hybrid Model-Based/Model-Free with Perseveration and Eligibility Traces.
    7 parameters: learning_rate, learning_rate_2, beta, beta_2, w, lambd, perseveration
    """
    learning_rate, learning_rate_2, beta, beta_2, w, lambd, perseveration = model_parameters
    n_trials = len(action_1)

    transition_matrix = np.array([[0.7, 0.3], [0.3, 0.7]])
    prev_action_indicator = np.zeros(2)
    p_choice_1 = np.zeros(n_trials)
    p_choice_2 = np.zeros(n_trials)
    q_stage1_mf = np.zeros(2)
    q_stage2_mf = np.zeros((2, 2))

    for trial in range(n_trials):
        max_q_stage2 = np.max(q_stage2_mf, axis=1)
        q_stage1_mb = transition_matrix @ max_q_stage2
        q_stage1_combined = w * q_stage1_mb + (1 - w) * q_stage1_mf
        q_stage1_with_pers = q_stage1_combined + perseveration * prev_action_indicator

        exp_q1 = np.exp(beta * q_stage1_with_pers)
        probs_1 = exp_q1 / np.sum(exp_q1)
        p_choice_1[trial] = probs_1[action_1[trial]]

        state_idx = state[trial]
        exp_q2 = np.exp(beta_2 * q_stage2_mf[state_idx])
        probs_2 = exp_q2 / np.sum(exp_q2)
        p_choice_2[trial] = probs_2[action_2[trial]]

        delta_stage1 = q_stage2_mf[state_idx, action_2[trial]] - q_stage1_mf[action_1[trial]]
        q_stage1_mf[action_1[trial]] += learning_rate * delta_stage1

        delta_stage2 = reward[trial] - q_stage2_mf[state_idx, action_2[trial]]
        q_stage2_mf[state_idx, action_2[trial]] += learning_rate_2 * delta_stage2
        q_stage1_mf[action_1[trial]] += lambd * learning_rate * delta_stage2

        prev_action_indicator.fill(0)
        prev_action_indicator[action_1[trial]] = 1

    eps = 1e-10
    log_loss = -(np.sum(np.log(p_choice_1 + eps)) + np.sum(np.log(p_choice_2 + eps)))
    return log_loss


def fit_hybrid_model(choice1, state, choice2, reward, trials):
    nreps = 10
    llh_min = np.inf
    bounds = [[0, 1], [0, 1], [0.1, 10], [0.1, 10], [0, 1], [0, 1], [0, 1]]
    best_params = None

    for _ in range(nreps):
        x0 = [np.random.uniform(b[0], b[1]) for b in bounds]
        res = minimize(
            lambda params: hybrid_model(choice1, state, choice2, reward, params),
            x0,
            method="L-BFGS-B",
            bounds=bounds,
        )
        if res.fun < llh_min:
            llh_min = res.fun
            best_params = res.x

    n_params = len(bounds)
    n_trials = len(trials)
    bic = (2 * llh_min) + (n_params * np.log(n_trials))

    if np.isinf(bic) or np.isnan(bic):
        bic = -4 * np.log(0.5) * len(choice1)

    return bic, best_params


def add_baseline_bic(df):
    print("\n" + "=" * 40)
    print("Fitting baseline hybrid model to each participant...")
    print("=" * 40)

    for pid, group in df.groupby("participant"):
        # Skip participants with invalid trials (all -1)
        valid = group[group["choice_1"] != -1]
        if len(valid) < 10:
            print(f"  Participant {pid}: too few valid trials ({len(valid)}), using default BIC")
            df.loc[df["participant"] == pid, "baseline_bic"] = -4 * np.log(0.5) * len(group)
            continue

        choice1 = valid["choice_1"].to_numpy().astype(int)
        state_arr = valid["state"].to_numpy().astype(int)
        choice2 = valid["choice_2"].to_numpy().astype(int)
        reward_arr = valid["reward"].to_numpy().astype(int)
        trials = valid["trial"].to_numpy()

        bic, _ = fit_hybrid_model(choice1, state_arr, choice2, reward_arr, trials)
        df.loc[df["participant"] == pid, "baseline_bic"] = bic

        if (pid + 1) % 100 == 0:
            print(f"  Fitted {pid + 1} participants... (BIC = {bic:.1f})")

    print("Baseline fitting complete.")
    return df


# =========================================================================
# Main
# =========================================================================
def main():
    print("=" * 60)
    print("Preprocessing Study 2 data for GeCCo (transdiagnostic factors)")
    print("=" * 60)

    # Step 1: Load self-report
    self_report = load_self_report(SELF_REPORT_FILE)

    # Step 2: Match participants
    matched_ids = match_participants(self_report, RAW_TASK_DIR)

    # Step 3-5: Clean, combine, merge
    print("\nCleaning and combining task files...")
    df = combine_and_merge(matched_ids, RAW_TASK_DIR, self_report)
    print(f"Combined data shape: {df.shape}")

    # Step 6: Remap values
    print("\nRemapping values...")
    df = remap_values(df)

    # Save intermediate (before baseline fitting, which is slow)
    intermediate_file = "preprocessed_study2_no_baseline.csv"
    df.to_csv(intermediate_file, index=False)
    print(f"\nSaved intermediate file: {intermediate_file}")

    # Step 7: Fit baseline model
    df = add_baseline_bic(df)

    # Save final output
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nFinal output saved to: {OUTPUT_FILE}")
    print(f"Shape: {df.shape}")
    print(f"Participants: {df['participant'].nunique()}")
    print(f"\nFactor score summary:")
    print(df[["ad", "cit", "sw"]].describe())


if __name__ == "__main__":
    main()
