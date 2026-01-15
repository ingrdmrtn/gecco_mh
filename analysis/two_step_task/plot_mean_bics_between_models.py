import matplotlib.pyplot as plt
import numpy as np
import os, sys, re, glob, numpy as np, pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]

# Data
models = [
    "Hybrid (baseline)",
    "Group level w/o",
    "Individual level w/o OCD",
    "Group level w/ OCD",
    "Individual level w/ OCD"
]

## OCD on the stai dataset with Gemini 3 pro
# means = [
#     378.5052689999665,
#     375.80318299804406,
#     370.41006481450654,
#     379.2258787724597,
#     373.4467593649533
# ]
# sems = [
#     16.38777030114607,
#     16.41641315270264,
#     16.553885675320746,
#     16.613507998650697,
#     16.387121178842772
# ]

## OCD on ocir dataset with Gemini 3 pro
means = [
    427.32151136529444,
    424.5148078772101,
    416.3312142472037,
    421.0889810917275,
    418.82188896852665 
    
]
sems = [
    14.57560762763633,
    14.075866812851562,
    14.55011412599917,
    14.467979744373332,
    14.718324496029659
    
]



# Short labels for the x-axis
short_labels = [
    "Hybrid\n(Baseline)",
    "Group\nw/o",
    "Indiv\nw/o", 
    "Group\nw/ OCD",
    "Indiv\nw/ OCD"
]

x_pos = np.arange(len(models))

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
ax.set_ylim(400, 460) # on ocir based oci results

# Despine: Remove top and right spines, keep left and bottom
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(True)
ax.spines['bottom'].set_visible(True)

# Save the plot as a PNG file
plt.tight_layout()
plt.savefig(f'{project_root}/analysis/two_step_task/bic_comparison_ocir_gemini_3_pro.png', dpi=300, bbox_inches='tight')

# Optional: Show the plot as well
plt.show()