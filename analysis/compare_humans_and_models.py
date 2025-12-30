import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the datasets
df_humans = pd.read_csv('ppcs_humans.csv')
df_group = pd.read_csv('ppcs_group_metadata.csv')
df_individual = pd.read_csv('ppcs_individual_stai.csv')

# Define the columns of interest
cols = ['prob_stay_common_rewarded', 'prob_stay_rare_rewarded', 
        'prob_stay_common_not_rewarded', 'prob_stay_rare_not_rewarded']

# Calculate means for each dataset
means_humans = [df_humans[col].mean() for col in cols]
means_group = [df_group[col].mean() for col in cols]
means_individual = [df_individual[col].mean() for col in cols]

# Create subplots with shared y-axis
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

# Define labels
labels = ['common/r', 'rare/r', 'common/nr', 'rare/nr']
x = np.arange(len(labels))

# Plot for Humans
# Using default color (C0) implicitly by not specifying color
axes[0].bar(x, means_humans)
axes[0].set_title('Humans')
axes[0].set_xticks(x)
axes[0].set_xticklabels(labels)
axes[0].set_ylim(0, 1.0) # Explicitly setting ylim as these are probabilities

# Plot for Gecco Group Metadata
axes[1].bar(x, means_group)
axes[1].set_title('Gecco Group Metadata')
axes[1].set_xticks(x)
axes[1].set_xticklabels(labels)

# Plot for Gecco Individual
axes[2].bar(x, means_individual)
axes[2].set_title('Gecco Individual')
axes[2].set_xticks(x)
axes[2].set_xticklabels(labels)

# Adjust layout and save
plt.tight_layout()
plt.savefig('comparison_plot.png')