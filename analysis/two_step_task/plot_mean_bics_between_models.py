import matplotlib.pyplot as plt
import numpy as np
import os, sys, re, glob, numpy as np, pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from pathlib import Path
from config.schema import load_config
from group_bmc_analysis_oci import load_baseline_bics, load_gecco_bics_group, load_gecco_bics
project_root = Path(__file__).resolve().parents[2]

# Configuration
configs = ['two_step_psychiatry_group_ocd_maxsetting.yaml',
           'two_step_psychiatry_individual_function_gemini-3-pro_ocd_maxsetting.yaml',
           'two_step_psychiatry_group_metadata_ocd_maxsetting.yaml',
           'two_step_psychiatry_individual_ocd_function_gemini-3-pro_ocd_maxsetting.yaml']

labels = [
    "Hybrid\n(Baseline)",
    "Group\nw/o",
    "Indiv\nw/o",
    "Group\nw/ OCD",
    "Indiv\nw/ OCD",
]
# add (max setting) to the labels
short_labels = [label + " \n (max setting)" for label in labels[1:]]
short_labels = [labels[0]] + short_labels
means = []
sems = []
for idx, config in enumerate(configs):
    print(f"Using config: {config}")
    cfg = load_config(project_root / "config" / config)
    base_dir = os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(base_dir, 'data', 'two_step_gillan_2016_ocibalanced.csv')
    bics_dir = os.path.join(
    base_dir, 'results', f'{cfg.task.name}{'_individual' if cfg.evaluation.fit_type == 'individual' else ''}', 'bics')
    output_dir = os.path.join(base_dir, 'results', f'{cfg.task.name}{'_individual' if cfg.evaluation.fit_type == 'individual' else ''}')

    participants = list(range(14, 45))  # Participants 14-44
    # Load data
    print("Loading data...")
    baseline_bics = load_baseline_bics(data_path, participants)
    if idx == 0:
        gecco_bics = means.append(np.mean(list(baseline_bics.values())))
        sems.append(np.std(list(baseline_bics.values())) / np.sqrt(len(participants)))
    
    if 'individual' in cfg.task.name:
        gecco_bics = load_gecco_bics(bics_dir, participants)
    else:
        gecco_bics = load_gecco_bics_group(bics_dir, participants)

    # Calculate mean BICs
    mean_bics = np.mean(list(gecco_bics.values()))
    sem_bics = np.std(list(gecco_bics.values())) / np.sqrt(len(participants))

    # store the mean and sem in a list for later plotting
    means.append(mean_bics)
    sems.append(sem_bics)


x_pos = np.arange(len(short_labels))

# Create the plot
fig, ax = plt.subplots(figsize=(10, 6))

# Create bars with default color and error bars
bars = ax.bar(x_pos, means, yerr=sems, align='center', ecolor='black', capsize=10)

# Add labels and title
ax.set_ylabel('Mean BIC')
ax.set_xticks(x_pos)
ax.set_xticklabels(short_labels)
ax.set_title('Comparison of Mean BICs for Two-Step Psychiatry Task with Dataset curated on OCI (Gemini 3 pro)')

# Set y-axis limits
# ax.set_ylim(350, 400) # on stai based oci results
ax.set_ylim(350, 460) # on ocir based oci results

# Despine: Remove top and right spines, keep left and bottom
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(True)
ax.spines['bottom'].set_visible(True)

# Save the plot as a PNG file
plt.tight_layout()
plt.savefig(f'{project_root}/analysis/two_step_task/bic_comparison_oci{"_maxsetting" if any("maxsetting" in config for config in configs) else ""}.png', dpi=300, bbox_inches='tight')

# Optional: Show the plot as well
plt.show()