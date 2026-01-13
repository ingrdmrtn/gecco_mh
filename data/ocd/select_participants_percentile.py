import pandas as pd

# 1. Load the dataset
# Ensure the file 'self_report_study1.csv' is in the current directory
df = pd.read_csv('self_report_study1.csv')

# 2. Determine cutoffs based on 33rd and 66th percentiles
# quantiles are calculated dynamically from the data
low_cutoff = df['oci_total'].quantile(0.33)
high_cutoff = df['oci_total'].quantile(0.66)

print(f"Low cutoff (33%): {low_cutoff}")
print(f"High cutoff (66%): {high_cutoff}")

# 3. Define the groups
# Low: <= 33rd percentile
# Medium: > 33rd and <= 66th percentile
# High: > 66th percentile
low_group = df[df['oci_total'] <= low_cutoff]
medium_group = df[(df['oci_total'] > low_cutoff) &
                  (df['oci_total'] <= high_cutoff)]
high_group = df[df['oci_total'] > high_cutoff]

# 4. Select 15 participants from each group
# We use a fixed random_state for reproducibility
n_select = 15
# Check if we have enough participants
if len(low_group) >= n_select and len(medium_group) >= n_select and len(high_group) >= n_select:
    sampled_low = low_group.sample(n=n_select, random_state=42)
    sampled_medium = medium_group.sample(n=n_select, random_state=42)
    sampled_high = high_group.sample(n=n_select, random_state=42)

    # 5. Create the new dataset
    # Concatenate the selected samples
    selected_participants = pd.concat(
        [sampled_low, sampled_medium, sampled_high])

    # Optional: Add a 'group' column to indicate which group they belong to
    selected_participants['group'] = (
        ['Low'] * n_select +
        ['Medium'] * n_select +
        ['High'] * n_select
    )

    # 6. Save the new dataset
    output_filename = 'selected_participants_45.csv'
    selected_participants.to_csv(output_filename, index=False)

    print(f"Successfully selected {len(selected_participants)} participants.")
    print(f"New dataset saved to: {output_filename}")
    print(selected_participants[['subj.x', 'oci_total', 'group']].head())

else:
    print("Error: Not enough participants in one or more groups to select 15.")
    print(
        f"Counts - Low: {len(low_group)}, Medium: {len(medium_group)}, High: {len(high_group)}")
