import shutil
import os
import pandas as pd
import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution

# 1. Load the dataset
df = pd.read_csv('self_report_study1.csv')

# 2. Define bins based on fixed ranges
# Low: 0-20
# Medium: 21-40
# High: 41-60
low_group = df[(df['oci_total'] > 0) & (df['oci_total'] <= 20)]
medium_group = df[(df['oci_total'] > 20) & (df['oci_total'] <= 40)]
high_group = df[(df['oci_total'] > 40) & (df['oci_total'] <= 60)]

print(
    f"Counts - 0-20: {len(low_group)}, 21-40: {len(medium_group)}, 41-60: {len(high_group)}")

# 3. Select 15 participants from each group
n_select = 15

if len(low_group) >= n_select and len(medium_group) >= n_select and len(high_group) >= n_select:
    sampled_low = low_group.sample(n=n_select, random_state=42)
    sampled_medium = medium_group.sample(n=n_select, random_state=42)
    sampled_high = high_group.sample(n=n_select, random_state=42)

    # 4. Create the new dataset
    selected_participants = pd.concat(
        [sampled_low, sampled_medium, sampled_high])

    # Add group labels
    selected_participants['group_bin'] = (
        ['0-20'] * n_select +
        ['21-40'] * n_select +
        ['41-60'] * n_select
    )

    # 5. Save the new dataset
    output_filename = 'selected_participants_bins.csv'
    selected_participants.to_csv(output_filename, index=False)

    print(f"Selection successful. Dataset saved to {output_filename}")
    print(selected_participants[['subj.x', 'oci_total', 'group_bin']].head())

else:
    print("Error: Not enough participants in one or more bins to sample 15.")


# 6. Copy associated files for selected participants
# ---------------------------------------------------------
source_folder = 'twostep_data_study1'       # Where the original files are
destination_folder = 'selected_twostep_data'  # New folder to create
participants_file = 'selected_participants_bins.csv'  # List of IDs to copy
file_extension = '.csv'                     # Extension of your data files
# ---------------------------------------------------------


def copy_selected_files():
    # 1. Load the list of selected participants
    if not os.path.exists(participants_file):
        print(f"Error: Could not find {participants_file}")
        return

    df = pd.read_csv(participants_file)
    subject_ids = df['subj.x'].unique()

    print(f"Found {len(subject_ids)} participants to copy.")

    # 2. Create the new directory if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
        print(f"Created new directory: {destination_folder}")
    else:
        print(f"Directory already exists: {destination_folder}")

    copied_count = 0
    missing_count = 0

    # 3. Copy files
    for subj_id in subject_ids:
        # Construct filenames
        filename = f"{subj_id}{file_extension}"
        src_path = os.path.join(source_folder, filename)
        dst_path = os.path.join(destination_folder, filename)

        if os.path.exists(src_path):
            try:
                shutil.copy2(src_path, dst_path)
                copied_count += 1
            except Exception as e:
                print(f"Error copying {filename}: {e}")
        else:
            print(f"Warning: File not found for subject {subj_id}")
            missing_count += 1

    # 4. Summary
    print("\n" + "="*40)
    print("Process Complete")
    print(f"Files successfully copied: {copied_count}")
    print(f"Files missing: {missing_count}")
    print(f"Location: {os.path.abspath(destination_folder)}")
    print("="*40)


# Run the file copying function
copy_selected_files()

# process the selected files as needed

# Configuration
# ---------------------------------------------------------
# Folder with the raw participant files
input_folder = 'selected_twostep_data'
output_folder = 'cleaned_twostep_data'   # Folder to save the cleaned files
# ---------------------------------------------------------

# Define the new column names based on your description
column_names = [
    'trial_num',                  # A
    'drift_1',                    # B
    'drift_2',                    # C
    'drift_3',                    # D
    'drift_4',                    # E
    'stage_1_response',           # F
    'stage_1_selected_stimulus',  # G
    'stage_1_rt',                 # H
    'transition',                 # I
    'stage_2_response',           # J
    'stage_2_selected_stimulus',  # K
    'stage_2_state',              # L
    'stage_2_rt',                 # M
    'reward',                     # N
    'redundant'                   # O
]


def clean_participant_file(file_path, save_path):
    try:
        # 1. Read the raw file content line by line
        with open(file_path, 'r') as f:
            lines = f.readlines()

        # 2. Find the start of the data
        # "Task data starts the row after the one that lists 'twostep_instruct_9' in column C."
        start_index = -1
        for i, line in enumerate(lines):
            if 'twostep_instruct_9' in line:
                start_index = i
                break

        if start_index == -1:
            print(
                f"Skipping {os.path.basename(file_path)}: 'twostep_instruct_9' marker not found.")
            return

        # 3. Extract data lines (skip the marker line itself)
        data_lines = lines[start_index + 1:]

        # 4. Create a DataFrame
        # We use a temporary buffer to read the list of strings as a CSV
        from io import StringIO
        data_str = "".join(data_lines)

        # Read without header since the file doesn't have a proper header row for the data
        df = pd.read_csv(StringIO(data_str), header=None, names=column_names)

        # 5. Save cleaned data
        df.to_csv(save_path, index=False)
        # print(f"Cleaned: {os.path.basename(file_path)}") # Optional: print progress

    except Exception as e:
        print(f"Error processing {os.path.basename(file_path)}: {e}")


def save_cleaned_files():
    # Create output directory
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output directory: {output_folder}")

    # Process all CSV files in the input folder
    files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]
    print(f"Found {len(files)} files to process.")

    for filename in files:
        src_path = os.path.join(input_folder, filename)
        dst_path = os.path.join(output_folder, filename)

        clean_participant_file(src_path, dst_path)

    print("\nProcessing complete.")
    print(f"Cleaned files are located in: {os.path.abspath(output_folder)}")


# Run the function
save_cleaned_files()

# Configuration
# ---------------------------------------------------------
# Folder containing the CLEANED data files (output from the previous script)
input_folder = 'cleaned_twostep_data'

# The file containing the list of selected participants (to ensure correct indexing order)
participants_list_file = 'selected_participants_bins.csv'

# Output file name
output_file = 'combined_data.csv'
# ---------------------------------------------------------


def combine_files():
    # 1. Load the participant list to establish the 0-44 index order
    if not os.path.exists(participants_list_file):
        print(
            f"Error: {participants_list_file} not found. Cannot determine participant order.")
        return

    participants_df = pd.read_csv(participants_list_file)

    # Ensure we have the subject ID column (assuming 'subj.x')
    if 'subj.x' not in participants_df.columns:
        print("Error: 'subj.x' column not found in participants list.")
        return

    subject_ids = participants_df['subj.x'].tolist()
    print(f"Loaded list of {len(subject_ids)} participants.")

    combined_frames = []
    files_missing = 0

    # 2. Iterate through participants by index (0 to 44)
    for index, subj_id in enumerate(subject_ids):
        file_name = f"{subj_id}.csv"
        file_path = os.path.join(input_folder, file_name)

        if os.path.exists(file_path):
            try:
                # Load the cleaned data
                df = pd.read_csv(file_path)

                # Add the requested columns
                df['subject_id'] = subj_id
                df['participant'] = index  # 0 to 44

                # Reorder columns to put identifiers first (optional, but nice)
                cols = ['participant', 'subject_id'] + \
                    [c for c in df.columns if c not in [
                        'participant', 'subject_id']]
                df = df[cols]

                combined_frames.append(df)
            except Exception as e:
                print(f"Error reading {file_name}: {e}")
        else:
            print(
                f"Warning: File not found for participant {index} ({subj_id})")
            files_missing += 1

    # 3. Concatenate and Save
    if combined_frames:
        final_df = pd.concat(combined_frames, ignore_index=True)
        final_df.to_csv(output_file, index=False)

        print("\n" + "="*40)
        print("Combination Complete")
        print(f"Total participants merged: {len(combined_frames)}")
        print(f"Files missing: {files_missing}")
        print(f"Final data shape: {final_df.shape}")
        print(f"Saved to: {output_file}")
        print("="*40)
    else:
        print("No data was combined.")


# Run the combination function
combine_files()

# Configuration
# ---------------------------------------------------------
# The combined task data file from the previous step
input_data_file = 'combined_data.csv'

# The self-report survey data file
survey_file = 'self_report_study1.csv'

# The output file name
output_file = 'combined_data_with_scores.csv'
# ---------------------------------------------------------


def append_scores():
    # 1. Load the combined task data
    if not os.path.exists(input_data_file):
        print(f"Error: {input_data_file} not found.")
        return

    print(f"Loading task data from {input_data_file}...")
    df_task = pd.read_csv(input_data_file)
    print(f"Task data shape: {df_task.shape}")

    # 2. Load the self-report survey data
    if not os.path.exists(survey_file):
        print(f"Error: {survey_file} not found.")
        return

    print(f"Loading survey data from {survey_file}...")
    df_survey = pd.read_csv(survey_file)

    # 3. Prepare the survey data for merging
    # Select only the columns we need: ID and the scores
    # The ID column in the survey file is 'subj.x'
    survey_cols = ['subj.x', 'stai_total', 'sds_total', 'oci_total']

    # Check if these columns exist
    if not set(survey_cols).issubset(df_survey.columns):
        print(
            f"Error: One or more columns {survey_cols} not found in survey file.")
        return

    df_survey_subset = df_survey[survey_cols].copy()

    # Rename 'subj.x' to 'subject_id' to match the column in our task data
    df_survey_subset.rename(columns={'subj.x': 'subject_id'}, inplace=True)

    # 4. Merge the datasets
    # We use a 'left' merge to keep all rows from the task data and add matching survey info
    print("Merging data...")
    df_merged = pd.merge(df_task, df_survey_subset,
                         on='subject_id', how='left')

    # 5. Verification
    # Check if we have any NaNs in the new columns (which would indicate missing IDs in survey data)
    missing_scores = df_merged['stai_total'].isna().sum()
    if missing_scores > 0:
        print(
            f"Warning: {missing_scores} rows have missing scores (ID mismatch).")
    else:
        print("All rows successfully matched with survey scores.")

    # 6. Save the final file
    df_merged.to_csv(output_file, index=False)

    print("\n" + "="*40)
    print("Process Complete")
    print(f"Final data shape: {df_merged.shape}")
    print(f"Saved to: {output_file}")
    print("="*40)

    # Show a preview
    print(df_merged[['subject_id', 'participant',
          'stai_total', 'sds_total', 'oci_total']].head())


# Run the function to append oci scores
append_scores()


# Configuration
# ---------------------------------------------------------
input_file = 'combined_data_with_scores.csv'
output_file = 'preprocessed_data.csv'
# ---------------------------------------------------------


def process_data():
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    df = pd.read_csv(input_file)
    print(f"Loaded data shape: {df.shape}")

    # 1. Verify and Print Initial Unique Values
    print("\n--- Pre-processing Unique Values ---")
    cols_to_check = ['stage_1_response', 'stage_2_response', 'stage_2_state']
    for col in cols_to_check:
        if col in df.columns:
            print(f"{col}: {df[col].unique()}")
        else:
            print(f"Warning: Column {col} not found in dataset.")

    # 2. Remap Values
    # We use a dictionary for mapping. Values not in the dict (like -1) remain unchanged.
    response_map = {'left': 0, 'right': 1}
    state_map = {2: 0, 3: 1}

    # make trial start from 0 instead of 1
    if 'trial_num' in df.columns:
        df['trial_num'] = df['trial_num'] - 1

    # Apply mappings
    # We convert to numeric, coercing errors to NaN if something unexpected appears,
    # but since we want to keep -1, we'll use .replace() which preserves unmapped values.

    if 'stage_1_response' in df.columns:
        df['stage_1_response'] = df['stage_1_response'].replace(response_map)

    if 'stage_2_response' in df.columns:
        df['stage_2_response'] = df['stage_2_response'].replace(response_map)

    if 'stage_2_state' in df.columns:
        df['stage_2_state'] = df['stage_2_state'].replace(state_map)

    # 3. Rename Columns
    rename_map = {
        'stage_1_response': 'choice_1',
        'stage_2_response': 'choice_2',
        'stage_2_state': 'state',
        'stai_total': 'stai',
        'sds_total': 'sds',
        'oci_total': 'oci',
        'trial_num': 'trial'
    }

    df.rename(columns=rename_map, inplace=True)

    # normalize oci scores to 0-1 range by dividing by 60
    max_oci_score = 60.0
    if 'oci' in df.columns:
        df['oci'] = df['oci'] / max_oci_score

    # 4. Verify and Save
    print("\n--- Post-processing Unique Values ---")
    new_cols_to_check = ['choice_1', 'choice_2', 'state']
    for col in new_cols_to_check:
        if col in df.columns:
            print(f"{col}: {df[col].unique()}")

    df.to_csv(output_file, index=False)
    print("\n" + "="*40)
    print(f"Processing Complete.")
    print(f"Renamed columns: {list(rename_map.values())}")
    print(f"Saved to: {output_file}")
    print("="*40)
    print(df.head())

# Run the processing function
process_data()


# Configuration
# ---------------------------------------------------------
input_file = 'preprocessed_data.csv'
output_file = 'two_step_gillan_2016_ocibalanced.csv'
# ---------------------------------------------------------


def hybrid_model(action_1, state, action_2, reward, model_parameters):
    """
    Hybrid Model-Based and Model-Free Learning with Perseveration and Eligibility Traces.

    Parameters:
    - learning_rate: Learning rate for stage 1 MF updates
    - learning_rate_2: Learning rate for stage 2 MF updates
    - beta: Inverse temperature for stage 1 choices
    - beta_2: Inverse temperature for stage 2 choices
    - w: Weight on model-based vs. model-free values (stage 1)
    - lambd: Eligibility trace weight
    - perseveration: Tendency to repeat previous stage 1 action
    """

    # Unpack parameters
    learning_rate, learning_rate_2, beta, beta_2, w, lambd, perseveration = model_parameters
    n_trials = len(action_1)

    # Transition model: spaceship -> planet
    transition_matrix = np.array([[0.7, 0.3], [0.3, 0.7]])

    # Perseveration indicator: 1 if action was repeated at stage 1
    prev_action_indicator = np.zeros(2)

    # Store choice probabilities
    p_choice_1 = np.zeros(n_trials)
    p_choice_2 = np.zeros(n_trials)

    # Initialize Q-values
    q_stage1_mf = np.zeros(2)          # Model-free Q-values for spaceship choices
    q_stage2_mf = np.zeros((2, 2))     # Model-free Q-values for state × action

    for trial in range(n_trials):
        # ----- Stage 1 -----
        max_q_stage2 = np.max(q_stage2_mf, axis=1)  # max Q for each planet
        q_stage1_mb = transition_matrix @ max_q_stage2  # model-based Q-values

        q_stage1_combined = w * q_stage1_mb + (1 - w) * q_stage1_mf
        q_stage1_with_pers = q_stage1_combined + perseveration * prev_action_indicator

        exp_q1 = np.exp(beta * q_stage1_with_pers)
        probs_1 = exp_q1 / np.sum(exp_q1)
        p_choice_1[trial] = probs_1[action_1[trial]]

        # ----- Stage 2 -----
        state_idx = state[trial]
        exp_q2 = np.exp(beta_2 * q_stage2_mf[state_idx])
        probs_2 = exp_q2 / np.sum(exp_q2)
        p_choice_2[trial] = probs_2[action_2[trial]]

        # ----- Learning -----

        # TD error for stage 1 (bootstrapped from stage 2)
        delta_stage1 = q_stage2_mf[state_idx, action_2[trial]] - q_stage1_mf[action_1[trial]]
        q_stage1_mf[action_1[trial]] += learning_rate * delta_stage1

        # TD error for stage 2
        delta_stage2 = reward[trial] - q_stage2_mf[state_idx, action_2[trial]]
        q_stage2_mf[state_idx, action_2[trial]] += learning_rate_2 * delta_stage2

        # Eligibility trace for stage 1
        q_stage1_mf[action_1[trial]] += lambd * learning_rate * delta_stage2

        # ----- Perseveration update -----
        prev_action_indicator.fill(0)
        prev_action_indicator[action_1[trial]] = 1

    eps = 1e-10
    log_loss = -(np.sum(np.log(p_choice_1 + eps)) + np.sum(np.log(p_choice_2 + eps)))

    return log_loss

def fit_hybrid_model(choice1, choice2_state, choice2, reward, trials, df_participant):

    nreps = 10
    llh_min_hybrid = np.inf
    model_parameter_bounds_hybrid = [[0, 1], [0, 1], [0.1, 10], [0.1, 10], [0, 1], [0, 1], [0, 1]]

    for rep in range(nreps):
        initial_guess_hybrid = [np.random.uniform(model_parameter_bounds_hybrid[i][0], model_parameter_bounds_hybrid[i][1])
                                for i in
                                range(len(model_parameter_bounds_hybrid))]
        res_hybrid = minimize(
            lambda params: hybrid_model(choice1, choice2_state, choice2, reward, params),
            # Pass params, not initial_guess
            initial_guess_hybrid,
            method='L-BFGS-B',
            bounds=model_parameter_bounds_hybrid)
        if res_hybrid.fun < llh_min_hybrid:
            llh_min_hybrid = res_hybrid.fun
            best_params_hybrid_current = res_hybrid.x

    bic_hybrid = (2 * llh_min_hybrid) + (len(model_parameter_bounds_hybrid) * np.log(len(trials)))

    if np.isinf(bic_hybrid) or np.isnan(bic_hybrid):
        bic_hybrid = -4 * np.log(0.5) * len(choice1)

    return bic_hybrid, best_params_hybrid_current

def add_baseline_to_ocibalanced(input_file, output_file):
    # load data
    df = pd.read_csv(input_file)
    results = []
    print("\n" + "="*40)
    print("Starting to fit participants for baseline model...")
    print("="*40 + "\n")
    for participant_id, df_participant in df.groupby('participant'):
        print(f"Fitting participant {participant_id}")
        choice1 = df_participant['choice_1'].to_numpy()
        choice2_state = df_participant['state'].to_numpy()
        choice2 = df_participant['choice_2'].to_numpy()
        reward = df_participant['reward'].to_numpy()
        trials = df_participant['trial'].to_numpy()
        bic_hybrid, best_params_hybrid_current = fit_hybrid_model(choice1, choice2_state, choice2, reward, trials, df_participant)
        # add the results bics to the original data as a column called baseline_bic
        df.loc[df['participant'] == participant_id, 'baseline_bic'] = bic_hybrid

    df.to_csv(output_file, index=False)

    print("\n" + "="*40)
    print("All participants fitted and baseline BICs added.")
    print(f"Saved updated data to: {output_file}")
    print("="*40)


# Run the function to add baseline BICs
add_baseline_to_ocibalanced(input_file, output_file)