import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import sys
from pathlib import Path

# ==========================================
# 1. SETUP & STYLE
# ==========================================
def set_figure_style():
    """Sets a clean, publication-ready style."""
    sns.set_context("paper", font_scale=1.4)
    sns.set_style("ticks")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['figure.dpi'] = 150

set_figure_style()

# Helper function to parse list strings
def parse_list(x):
    try:
        if pd.isna(x): return []
        if isinstance(x, list): return x
        return ast.literal_eval(x)
    except:
        try:
            clean = x.strip("[]").replace("'", "").replace('"', '').replace('\n', ' ')
            if ',' in clean:
                return [s.strip() for s in clean.split(',') if s.strip()]
            return [s.strip() for s in clean.split() if s.strip()]
        except:
            return []

# ==========================================
# 2. CATEGORIZATION LOGIC (Updated)
# ==========================================
# def get_category(mech_name):
#     m = str(mech_name).lower()
    
#     # 1. Control (Arbitration between systems)
#     if any(k in m for k in ['independent-mb-strength', 'control', 'arbitration']): 
#         return 'Control'
    
#     # 2. Valuation (Subjective perception of outcomes)
#     if any(k in m for k in ['valu', 'loss', 'util', 'reward', 'sens']): 
#         return 'Valuation'
        
#     # 3. Learning (Update rules, plasticity)
#     if any(k in m for k in ['learn', 'updat', 'trans', 'risk', 'alien', 'delta', 'rate', 'plasticity', 'precision']): 
#         return 'Learning'
    
#     # 4. Perseveration (Habits, stickiness)
#     if any(k in m for k in ['stick', 'pers', 'habit', 'stay', 'rigidity']): 
#         return 'Perseveration'
    
#     # 5. Memory (Decay, forgetting)
#     if any(k in m for k in ['decay', 'forget', 'mem', 'trace']): 
#         return 'Memory'
    
#     # 6. Exploration (Choice policy parameters)
#     if any(k in m for k in ['bias', 'explor', 'temp', 'noise', 'conf', 'mixture']): 
#         return 'Exploration'
    
#     return 'Other'
def get_category(mech_name):
    m = str(mech_name).lower()
    
    # 1. Control
    # if any(k in m for k in ['independent-mb-strength', 'control', 'arbitration']): 
    #     return 'Control'
    
    # 2. Perseveration (Habits, stickiness) - Keep high priority
    if any(k in m for k in ['stick', 'pers', 'habit', 'stay', 'rigidity']): 
        return 'Perseveration'
    
    # 3. Memory (Decay, forgetting) - MOVE UP to catch 'value-decay'
    if any(k in m for k in ['decay', 'forget', 'mem', 'trace']): 
        return 'Memory'
    
    # 4. Learning (Update rules) - MOVE UP to catch 'risk-sensitive-learning'
    if any(k in m for k in ['learn', 'updat', 'trans', 'risk', 'alien', 'delta', 'rate', 'plasticity', 'precision''valu', 'loss', 'util', 'reward', 'sens', 'independent-mb-strength', 'control', 'arbitration']): 
        return 'Learning'
    
    # # 5. Valuation (Subjective perception) - Move down so it only catches pure valuation items
    # if any(k in m for k in ['valu', 'loss', 'util', 'reward', 'sens']): 
    #     return 'Valuation'
    
    # 6. Exploration
    if any(k in m for k in ['bias', 'explor', 'temp', 'noise', 'conf', 'mixture']): 
        return 'Exploration'
    
    return 'Other'

# ==========================================
# 3. DATA LOADING & PROCESSING
# ==========================================
# Load datasets
project_root = Path(__file__).resolve().parents[2]
df_psych = pd.read_csv(project_root / 'results' / 'two_step_psychiatry_individual_oci_function_ocibalanced_maxsetting_individual' / 'gecco_baseline_comparison.csv')
df_ind = pd.read_csv(project_root / 'results' / 'two_step_psychiatry_individual_function_ocibalanced_maxsetting_individual' / 'gecco_baseline_comparison.csv')

# Calculate BIC Improvements (Baseline - Best Model)
df_ind['bic_imp_ind'] = df_ind['baseline_bic'] - df_ind['best_model_bic']
df_psych['bic_imp_psych'] = df_psych['baseline_bic'] - df_psych['best_model_bic']

# Parse Mechanism Lists
df_ind['mech_ind'] = df_ind['unique_mechanisms'].apply(parse_list)
df_psych['mech_psych'] = df_psych['unique_mechanisms'].apply(parse_list)

# Merge datasets on 'participant'
df_merged = pd.merge(
    df_ind[['participant', 'bic_imp_ind', 'mech_ind', 'best_model_bic']],
    df_psych[['participant', 'bic_imp_psych', 'mech_psych', 'best_model_bic', 'oci']],
    on='participant',
    suffixes=('_ind', '_psych')
)

# Calculate Performance Difference
# Positive = Individual Model Fits Better
# Negative = Psychiatry Model Fits Better
df_merged['bic_diff'] = df_merged['bic_imp_ind'] - df_merged['bic_imp_psych']

# ==========================================
# 4. PLOTTING FIGURES
# ==========================================

# --- Figure 1: Model Comparison Scatter (The "Horse Race") ---
plt.figure(figsize=(8, 8))
plt.plot([-20, 160], [-20, 160], color='gray', linestyle='--', alpha=0.5, zorder=0)

scatter = plt.scatter(
    df_merged['bic_imp_psych'], 
    df_merged['bic_imp_ind'], 
    c=df_merged['oci'], 
    cmap='viridis', 
    s=100, 
    edgecolor='white', 
    linewidth=1.5,
    alpha=0.9
)
plt.colorbar(scatter, label='OCI Score')
plt.xlabel('Improvement: PIndividual Gecco w/ OCI Model')
plt.ylabel('Improvement: Individual Gecco w/o OCI Model')
plt.title('Model Fit Comparison')
plt.axis('equal') 
# plt.text(10, 100, 'Individual Gecco w/o OCI Wins', fontsize=12, fontweight='bold', color='#377eb8')
# plt.text(100, 10, 'Individual Gecco w/ OCI Wins', fontsize=12, fontweight='bold', color='#4daf4a')
plt.tight_layout()
plt.savefig('Fig11_Model_Comparison_Scatter.png', dpi=300)
plt.show()


# --- Figure 2: The "Waterfall" (Sorted Difference) ---
df_sorted = df_merged.sort_values('bic_diff', ascending=True)
colors = ['#4daf4a' if x < 0 else '#377eb8' for x in df_sorted['bic_diff']]

plt.figure(figsize=(10, 8))
plt.barh(range(len(df_sorted)), df_sorted['bic_diff'], color=colors, edgecolor='none')
plt.axvline(0, color='black', linewidth=1)
plt.yticks(range(len(df_sorted)), df_sorted['participant'], fontsize=9)
plt.xlabel('$\Delta$BIC (Individual Gecco w/o OCI - Individual Gecco w/ OCI)')
plt.ylabel('Participant ID')
plt.title('Fit Difference per Participant\n(Blue = Ind w/o OCI Better, Green = Ind w/ OCI Better)')
plt.tight_layout()
plt.savefig('Fig12_Waterfall_Difference.png', dpi=300)
plt.show()


# --- Figure 3: OCI vs Preference Trend ---
plt.figure(figsize=(8, 6))
sns.regplot(
    data=df_merged, 
    x='oci', 
    y='bic_diff', 
    scatter_kws={'s': 80, 'alpha': 0.7, 'color': 'gray'},
    line_kws={'color': 'black', 'linestyle': '--'}
)
plt.axhline(0, color='black', linewidth=0.5, alpha=0.5)
plt.ylabel('$\Delta$BIC (Individual Gecco w/o OCI - Individual Gecco w/ OCI)')
plt.xlabel('OCI Score')
plt.title('Does OCI Severity Predict Model Preference?')
plt.tight_layout()
plt.savefig('Fig13_OCI_vs_Preference.png', dpi=300)
plt.show()


# ==========================================
# 5. MECHANISM ANALYSIS
# ==========================================

# Calculate Removed/Added for every participant
def get_diffs(row):
    set_ind = set(row['mech_ind'])
    set_psych = set(row['mech_psych'])
    removed = list(set_ind - set_psych)  # In Ind but NOT Psych
    added = list(set_psych - set_ind)    # In Psych but NOT Ind
    return pd.Series({'removed': removed, 'added': added})

diff_df = df_merged.apply(get_diffs, axis=1)
df_analysis = pd.concat([df_merged, diff_df], axis=1)

# --- Figure 4: Category Turnover (New Analysis) ---
results = []
for _, row in df_analysis.iterrows():
    for r in row['removed']:
        results.append({'Type': 'Removed', 'Category': get_category(r)})
    for a in row['added']:
        results.append({'Type': 'Added', 'Category': get_category(a)})

df_changes = pd.DataFrame(results)

if not df_changes.empty:
    counts = df_changes.groupby(['Type', 'Category']).size().reset_index(name='Count')
    pivot = counts.pivot(index='Category', columns='Type', values='Count').fillna(0)
    
    # Ensure specific order of columns
    order = ['Learning', 'Memory', 'Perseveration', 'Exploration']#, 'Valuation', 'Control']
    pivot = pivot.reindex([c for c in order if c in pivot.index])
    
    pivot.plot(kind='bar', color=['#4daf4a', '#e41a1c'], width=0.8, figsize=(10, 6))
    plt.title('Mechanism Turnover by Category\n(Psychiatry vs Individual)')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Change Type', labels=['Added (in Individual Gecco w/ OCI)', 'Removed (from Individual Gecco w/o OCI)'])
    plt.tight_layout()
    plt.savefig('Fig14_Category_Turnover.png', dpi=300)
    plt.show()


# --- Figure 5: Top Mechanisms Gained and Lost (Specific Mechanisms) ---
all_removed = [item for sublist in df_analysis['removed'] for item in sublist]
all_added = [item for sublist in df_analysis['added'] for item in sublist]

removed_counts = pd.Series(all_removed).value_counts().head(8)
added_counts = pd.Series(all_added).value_counts().head(8)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.barplot(x=removed_counts.values, y=removed_counts.index, ax=axes[0], palette='Reds_r')
axes[0].set_title('Top Mechanisms REMOVED\n(Found in Individual Gecco w/o OCI, Lost in Individual Gecco w/ OCI)')
axes[0].set_xlabel('Frequency')

sns.barplot(x=added_counts.values, y=added_counts.index, ax=axes[1], palette='Greens_r')
axes[1].set_title('Top Mechanisms ADDED\n(Found in Individual Gecco w/ OCI Only)')
axes[1].set_xlabel('Frequency')

plt.tight_layout()
plt.savefig('Fig15_Mechanism_Gains_Losses.png', dpi=300)
plt.show()


# --- Figure 6: Replacement Heatmap ---
exchanges = []
for idx, row in df_analysis.iterrows():
    for r in row['removed']:
        for a in row['added']:
            exchanges.append({'Removed': r, 'Added': a})

df_exchanges = pd.DataFrame(exchanges)

if not df_exchanges.empty:
    # Filter for top mechanisms
    top_removed = df_exchanges['Removed'].value_counts().head(5).index
    top_added = df_exchanges['Added'].value_counts().head(5).index
    
    df_subset = df_exchanges[
        df_exchanges['Removed'].isin(top_removed) & 
        df_exchanges['Added'].isin(top_added)
    ]
    
    if not df_subset.empty:
        heatmap_data = pd.crosstab(df_subset['Removed'], df_subset['Added'])
        plt.figure(figsize=(10, 8))
        sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='Purples', cbar_kws={'label': 'Count'})
        plt.title('Mechanism Replacement Matrix\n(What replaces what?)')
        plt.ylabel('Removed (Individual Gecco w/o OCI Mechanism)')
        plt.xlabel('Added (Individual Gecco w/ OCI Mechanism)')
        plt.tight_layout()
        plt.savefig('Fig16_Replacement_Heatmap.png', dpi=300)
        plt.show()