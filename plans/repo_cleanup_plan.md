# Repository Cleanup Plan

## Background

The `gecco_mh` repository contains code for running cognitive model discovery experiments using Large Language Models (LLMs). Over time, the repository has accumulated multiple experiment configurations, analysis scripts, and data files from various research projects.

**Current Situation:** Only one configuration file is actively being used:
- `config/two_step_factors_distributed.yaml`

**Goal:** Remove all unnecessary files and directories to streamline the repository while preserving the working configuration and its dependencies.

**Note:** The `analysis/` directory is in `.gitignore` and does not need to be cleaned up.

---

## Active Configuration Analysis

The file `config/two_step_factors_distributed.yaml` specifies:

1. **Task:** Two-step decision-making task with factor analysis
2. **Data Sources:**
   - `data_g_2019/preprocessed_study2_no_baseline.csv` (main behavioral data)
   - `data_g_2019/self_report_study2.csv` (questionnaire data)
3. **LLM Provider:** Gemini (via Google GenAI API)
4. **Distributed Mode:** Uses SLURM array jobs with client profiles for parallel model search
5. **Parameter Recovery:** Enabled with `two_step` simulator

---

## Files and Directories to REMOVE

### Category 1: Unused Experiment Configurations (37 files)

**Location:** `config/`

**Reason:** These are configuration files for different experiments or older versions that are no longer needed. Only `two_step_factors_distributed.yaml` is in use.

**Files to Remove (36 total):**
```
config/decision_making.yaml
config/decision_making_openmodels.yaml
config/rlwm.yaml
config/rlwm_individual.yaml
config/rlwm_individual_age.yaml
config/two_step.yaml
config/two_step_factors.yaml
config/two_step_local.yaml
config/two_step_psychiatry.yaml
config/two_step_psychiatry_baseline_individual.yaml
config/two_step_psychiatry_group.yaml
config/two_step_psychiatry_group_metadata_ocd.yaml
config/two_step_psychiatry_group_metadata_ocd_maxsetting.yaml
config/two_step_psychiatry_group_metadata_stai.yaml
config/two_step_psychiatry_group_ocd.yaml
config/two_step_psychiatry_group_ocd_maxsetting.yaml
config/two_step_psychiatry_individual_function_gemini-3-pro.yaml
config/two_step_psychiatry_individual_function_gemini-3-pro_ocd.yaml
config/two_step_psychiatry_individual_function_gemini-3-pro_ocd_maxsetting.yaml
config/two_step_psychiatry_individual_ocd_function_gemini-3-pro.yaml
config/two_step_psychiatry_individual_ocd_function_gemini-3-pro_ocd.yaml
config/two_step_psychiatry_individual_ocd_function_gemini-3-pro_ocd_maxsetting.yaml
config/two_step_psychiatry_individual_stai_class.yaml
config/two_step_psychiatry_individual_stai_class_gemini-2.5-flash.yaml
config/two_step_psychiatry_individual_stai_class_gemini-2.5-pro.yaml
config/two_step_psychiatry_individual_stai_class_v2.yaml
config/two_step_psychiatry_individual_stai_function_gemini-3-pro.yaml
config/two_step_study2_factors_ddm_free.yaml
config/two_step_study2_factors_ddm_linked.yaml
config/two_step_study2_factors_group_interpretive.yaml
config/two_step_study2_factors_group_named.yaml
config/two_step_study2_factors_group_named_rt.yaml
config/two_step_task_group_metadata_stai_largecontext.yaml
config/two_step_task_group_metadata_stai_matchiterations.yaml
config/two_step_vllm_example.yaml
```

**Keep:**
```
config/two_step_factors_distributed.yaml  (ACTIVE CONFIG)
config/schema.py                           (Config validation - required by code)
```

---

### Category 2: Unused Data Files and Directories

**Location:** `data/` and `data_g_2019/`

**Reason:** The active configuration only uses data from `data_g_2019/`. All other data files are from different experiments or older data versions.

**Files/Directories to Remove:**
```
data/ocd/                                   (Entire directory - different experiment)
data/multi_attribute_decision_making.csv   (Different task)
data/rlwm.csv                              (Different task)
data/two_step_data.csv                     (Older data version)
data/two_step_gillan_2016.csv              (Older data version)
data/two_step_gillan_2016_ocibalanced.csv  (Older data version)
data/standardize_data.py                   (Data preprocessing script for old data)
```

**Keep:**
```
data_g_2019/                               (Entire directory - ACTIVE DATA)
  ├── preprocessed_study2_no_baseline.csv  (Used by config)
  ├── self_report_study2.csv               (Used by config for individual differences)
  ├── individual_items_study2.csv          (Related data)
  ├── weights_study2.csv                   (Related data)
  ├── preprocess_study2.py                 (Preprocessing script)
  ├── twostep_data_study1_individual_csv/  (Raw data directory)
  ├── twostep_data_study2_individual_csv/  (Raw data directory)
  ├── self_report_study1.csv               (Study 1 data)
  └── README.md                            (Documentation)
```

---

### Category 3: Unused Analysis Scripts

**Location:** `analysis/`

**Status: SKIP - `analysis/` is in `.gitignore`, no action needed.**

---

### Category 4: Unused Demo and Experiment Scripts

**Location:** `scripts/`

**Reason:** Most scripts are specific demos or experiment runners for configurations that are no longer in use. Only distributed runner scripts need to be preserved.

**Files to Remove:**
```
scripts/decision_making_demo.py
scripts/rlwm_demo.py
scripts/rlwm_individual_difference_demo.py
scripts/rlwm_individual_difference_demo_age.py
scripts/two_step_demo.py
scripts/two_step_group_metadata_stai.py
scripts/two_step_individual_function.py
scripts/two_step_individual_stai_class.py
scripts/two_step_psychiatry_group.py
scripts/individual_differences_2step_task_demo.py
scripts/posthoc_model_simulations_gecco_class.py
```

**Files to Keep:**
```
scripts/run_gecco_distributed.py     (Main entry point for distributed runs)
scripts/launch_distributed.py        (SLURM script generator)
scripts/monitor_distributed.py      (Monitors running distributed jobs)
scripts/reset_distributed.py        (Resets shared registry)
scripts/compile_results.py          (Compiles results from multiple clients)
scripts/test_fit_model.py           (Testing/debugging tool)
```

---

### Category 5: Dashboard Application

**Location:** `gecco-mh-dashboard/`

**Reason:** This is a Streamlit dashboard for monitoring experiments. Not needed for the core experiment workflow.

**Remove:** Entire directory
```
gecco-mh-dashboard/
```

---

### Category 6: Unused Bash Scripts

**Location:** `bash/`

**Reason:** Only the distributed runner scripts are needed. Others are for running different configurations or analysis workflows.

**Files to Remove:**
```
bash/run_analysis.sh
bash/run_gecco.sh
bash/run_gecco_group.sh
bash/run_gecco_individual.sh
```

**Files to Keep:**
```
bash/run_gecco_distributed.sh    (SLURM submission script for distributed runs)
bash/launch_vllm_server.sh       (Launches vLLM server - may be needed)
```

---

### Category 7: Unused LLM Backend Files

**Location:** `gecco/load_llms/`

**Reason:** The active config uses `provider: "gemini"`. The `vllm` provider may also be used when running distributed jobs. Other backends (OpenAI/GPT, R1, Qwen, Llama, KCL) are not currently configured.

**Files to Remove:**
```
gecco/load_llms/gpt_backend.py    (OpenAI/GPT API)
gecco/load_llms/r1_backend.py     (R1 model)
gecco/load_llms/qwen_backend.py   (Qwen model)
gecco/load_llms/llama_backend.py  (Llama model)
gecco/load_llms/kcl_backend.py    (KCL deployment)
```

**Files to Keep:**
```
gecco/load_llms/__init__.py
gecco/load_llms/model_loader.py    (Required - routes to appropriate backend)
gecco/load_llms/gemini_backend.py  (Used by active config)
gecco/load_llms/vllm_backend.py    (Potentially used in distributed mode)
```

---

### Category 8: Test Files

**Location:** `tests/`

**Action: KEEP** — `tests/test_hbi.py` contains comprehensive, valid tests for the core HBI model-fitting functionality (parameter recovery, model comparison, MAP estimation). Do not remove.

---

## Core Code to KEEP

The following directories and files are essential for the `two_step_factors_distributed.yaml` configuration and must be preserved:

### GeCCo Core Package
```
gecco/__init__.py
gecco/baseline.py                         (Fits baseline model)
gecco/coordination.py                     (Distributed registry)
gecco/parameter_recovery.py               (Parameter recovery checks)
gecco/run_gecco.py                        (Main search loop)
gecco/structured_output.py                 (JSON parsing for LLM responses)
gecco/utils.py                            (General utilities)

gecco/construct_feedback/
  ├── __init__.py
  └── feedback.py                         (Generates feedback for LLM)

gecco/load_llms/
  ├── __init__.py
  ├── model_loader.py                     (Backend router)
  ├── gemini_backend.py                   (Gemini API)
  └── vllm_backend.py                     (vLLM server)

gecco/model_fitting/
  ├── __init__.py
  └── hbi_scipy.py                        (Hierarchical Bayesian fitting)

gecco/offline_evaluation/
  ├── __init__.py
  ├── data_structures.py
  ├── evaluation_functions.py
  ├── fit_generated_models.py
  ├── fit_generated_models_hierarchical.py
  ├── individual_differences.py
  └── utils.py

gecco/prepare_data/
  ├── __init__.py
  ├── data2text.py                        (Converts data to narrative)
  └── io.py                               (Data loading)

gecco/prompt_builder/
  ├── __init__.py
  ├── guardrails.py                       (Model constraints)
  ├── prompt.py                           (Prompt construction)
  └── simulation_prompt.py               (Simulation prompts)
```

### Configuration
```
config/schema.py                           (Config validation)
config/two_step_factors_distributed.yaml   (Active config)
```

### Scripts
```
scripts/run_gecco_distributed.py
scripts/launch_distributed.py
scripts/monitor_distributed.py
scripts/reset_distributed.py
scripts/compile_results.py
scripts/test_fit_model.py
```

### Bash Scripts
```
bash/run_gecco_distributed.sh
bash/launch_vllm_server.sh
```

### Data
```
data_g_2019/   (Entire directory)
```

### Results
```
results/two_step_factors/   (Existing results - preserve)
```

### Project Files
```
AGENTS.md        (Instructions for AI agents)
README.md        (Project documentation)
requirements.txt (Python dependencies)
```

---

## Execution Plan

### Step 1: Create Backup
```bash
# Create a full backup before deleting anything
cd /mnt/c/Users/tobyw/OneDrive-KCL/projects/
tar -czf gecco_mh_backup_$(date +%Y%m%d).tar.gz gecco_mh/
```

### Step 2: Remove Unused Configs
```bash
cd /mnt/c/Users/tobyw/OneDrive-KCL/projects/gecco_mh

# Remove unused config files (keeping schema.py and the active config)
find config/ -name "*.yaml" ! -name "two_step_factors_distributed.yaml" -exec rm {} \;
```

### Step 3: Remove Unused Data
```bash
# Remove data directory contents (not the directory itself, in case needed later)
rm -rf data/ocd/
rm data/multi_attribute_decision_making.csv
rm data/rlwm.csv
rm data/two_step_data.csv
rm data/two_step_gillan_2016.csv
rm data/two_step_gillan_2016_ocibalanced.csv
rm data/standardize_data.py

# Verify data_g_2019/ remains intact
ls -la data_g_2019/
```

### Step 4: Analysis Files
**SKIP** — `analysis/` is in `.gitignore`, no action needed.

### Step 5: Remove Unused Scripts
```bash
cd scripts/
rm decision_making_demo.py
rm rlwm_demo.py
rm rlwm_individual_difference_demo.py
rm rlwm_individual_difference_demo_age.py
rm two_step_demo.py
rm two_step_group_metadata_stai.py
rm two_step_individual_function.py
rm two_step_individual_stai_class.py
rm two_step_psychiatry_group.py
rm individual_differences_2step_task_demo.py
rm posthoc_model_simulations_gecco_class.py
cd ../
```

### Step 6: Remove Dashboard
```bash
rm -rf gecco-mh-dashboard/
```

### Step 7: Remove Unused Bash Scripts
```bash
cd bash/
rm run_analysis.sh
rm run_gecco.sh
rm run_gecco_group.sh
rm run_gecco_individual.sh
cd ../
```

### Step 8: Remove Unused LLM Backends
```bash
cd gecco/load_llms/
rm gpt_backend.py
rm r1_backend.py
rm qwen_backend.py
rm llama_backend.py
rm kcl_backend.py
cd ../../
```

### Step 9: Clean up model_loader.py
Remove the dead conditional import branches for deleted providers (gpt, r1, qwen, llama, kcl) from `gecco/load_llms/model_loader.py`.

### Step 10: Optionally remove reference PDF
```bash
# A reference PDF exists at the repo root - can be removed to reduce repo size:
rm "Gillan et al. - 2016 - Characterizing a psychiatric symptom dimension rel.pdf"
```

---

## Verification Steps

After cleanup, verify the active configuration still works:

### 1. Check File Structure
```bash
# Verify critical files exist
ls config/two_step_factors_distributed.yaml
ls config/schema.py
ls -R gecco/
ls -R data_g_2019/
ls scripts/run_gecco_distributed.py
```

### 2. Test Configuration Loading
```bash
# Activate environment
conda activate gecco_mh

# Test that config loads correctly
python -c "from config.schema import load_config; from pathlib import Path; cfg = load_config(Path('config/two_step_factors_distributed.yaml')); print(f'Config loaded: {cfg.task.name}')"
```

### 3. Test Data Loading
```bash
# Test that data files are accessible
python -c "from gecco.prepare_data.io import load_data; df = load_data('data_g_2019/preprocessed_study2_no_baseline.csv', ['choice_1', 'state', 'choice_2', 'reward']); print(f'Loaded {len(df)} rows')"
```

### 4. Test Core Modules
```bash
# Test that core modules import correctly
python -c "
from gecco.run_gecco import GeCCoModelSearch
from gecco.baseline import fit_baseline_if_needed
from gecco.coordination import SharedRegistry
from gecco.parameter_recovery import get_simulator
from gecco.load_llms.model_loader import load_llm
from gecco.offline_evaluation.fit_generated_models import run_fit_hierarchical
print('All core modules imported successfully')
"
```

### 5. Dry Run (Optional)
```bash
# If you want to verify the full pipeline works, run a small test
# (Only do this if you have GPU access and want to spend compute)
python scripts/run_gecco_distributed.py \
  --config two_step_factors_distributed.yaml \
  --test \
  --client-id 0
```

---

## Estimated Disk Space Savings

| Category | Files | Estimated Size |
|----------|-------|----------------|
| Config files | 37 | ~200 KB |
| Data files | 7 | ~3 MB |
| Data directories | 2 | ~10 MB |
| Analysis files | 20 | ~200 KB |
| Analysis directory | 1 | ~100 KB |
| Scripts | 11 | ~100 KB |
| Dashboard | 1 dir | ~10 KB |
| Bash scripts | 4 | ~10 KB |
| LLM backends | 5 | ~30 KB |
| Tests | 2 | ~20 KB |
| **Total** | **~90 files** | **~14 MB** |

Note: Data files are relatively small, but the cleanup reduces complexity significantly. The `data/ocd/` directory contains the largest files.

---

## Post-Cleanup Recommendations

1. **Update README.md** to reflect the new structure and single active configuration
2. **Update AGENTS.md** if needed to reflect any changes to project structure
3. **Commit changes** to git with a clear message:
   ```
   git add -A
   git commit -m "Clean up repository: remove unused configs, data, and scripts
   
   - Remove 37 unused config files (keeping only two_step_factors_distributed.yaml)
   - Remove unused data files and ocd/ directory
   - Remove rlwm/ task analysis directory
   - Remove unused scripts and analysis files
   - Remove gecco-mh-dashboard/
   - Remove unused LLM backends (gpt, r1, qwen, llama, kcl)
   - Keep core gecco/ package and active configuration"
   ```

---

## Risk Mitigation

- **Backup created in Step 1** ensures no data is permanently lost
- **File-by-file verification** allows stopping at any point if issues arise
- **Incremental removal** by category makes rollback easier
- **Core code preservation** ensures the active configuration remains functional

If any step fails or produces unexpected results:
1. Stop immediately
2. Check error messages
3. Restore from backup if critical files were affected
4. Investigate the dependency before proceeding

---

## Files Summary

**Total files to remove:** ~90 files/directories
**Total files to keep:** ~50 files (core code + active config + data)

This cleanup will reduce repository complexity by approximately 65% while preserving all functionality needed for the active `two_step_factors_distributed.yaml` configuration.