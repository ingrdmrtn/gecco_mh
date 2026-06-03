# Plan: Add a single-stage bandit task to GeCCo

## Context

The user wants to run GeCCo on a different cognitive task: a single-stage 2-armed reversal-learning bandit (75% reward on the rewarded side, 0% on the other; rewarded side switches unpredictably across blocks; switches only follow rewarded trials, so a reward followed by no-reward is diagnostic of a possible reversal). Today every dataset and config in the repo is the two-step task (`choice_1, state, choice_2, reward`); the bandit task has only `(choice, reward)` per trial — fundamentally different shape.

The raw file (per the screenshot) has columns `RT, selected_box, reward, correct_box, sID, p_right, correct, model_name, TrialID, rewardversion, ACC, switch_trial, block, trialsincesw, outcome_1_t, outcome_2_t, choice_left, choice_1_back, choice_2_back` and includes `NA` cells for some columns. It needs preprocessing into a clean CSV before GeCCo can ingest it.

The good news: the fitting harness drives the model signature off `data.input_columns` ([gecco/offline_evaluation/fit_generated_models.py:75-84](gecco/offline_evaluation/fit_generated_models.py#L75-L84) — `inputs = [df_p[c].to_numpy() for c in input_cols]; model_func(*inputs, x)`), and the `narrative` data2text function ([gecco/prepare_data/data2text.py:45-115](gecco/prepare_data/data2text.py#L45-L115)) already accepts arbitrary templates and value_mappings. So the bandit task is **config-only** in `gecco/` — no library code needs to change as long as parameter recovery stays off.

(Per user direction, README cleanup is deferred — not in scope for this plan.)

---

## What changes

### 1. Preprocessor — `data_g_2019/preprocess_bandit.py` (new)

Modeled on [data_g_2019/preprocess_study2.py](data_g_2019/preprocess_study2.py) but much simpler since the raw file already looks tabular. Responsibilities:

- Load the raw CSV (path configurable at the top of the script — default placeholder e.g. `bandit_raw.csv`).
- Keep only the columns GeCCo needs plus a few useful covariates: `sID, TrialID, block, selected_box, reward, switch_trial, trialsincesw, RT, correct`. Drop the lag features (`outcome_1_t`, `outcome_2_t`, `choice_1_back`, `choice_2_back`) — the LLM model is supposed to learn dynamics from the sequence, not pre-computed lags.
- Coerce `selected_box` and `reward` to integer `{0, 1}`. Rows where either is `NA` get dropped (these are missed-response trials and the existing `narrative` template can't render them).
- Rename for harness compatibility: `TrialID → trial`, then 0-index it (`trial = trial - 1`) to match the convention in [preprocess_study2.py:185-187](data_g_2019/preprocess_study2.py#L185-L187).
- Add an integer `participant` column derived from `sID` (`pd.factorize(sID)[0]`) — the launcher's split logic uses integer participant ids; `sID` stays as a string identifier.
- Sanity-check and print: number of participants, mean trials per participant, fraction reward.
- Write to `data_g_2019/bandit_reversal.csv`.

The harness then reads this file via `data.path` in the YAML.

### 2. New config — `config/bandit_reversal_gpt54nano.yaml` (new)

Copy [config/two_step_factors_gpt54nano.yaml](config/two_step_factors_gpt54nano.yaml) verbatim and change the blocks below. Everything not shown (slurm, loop, evaluation, validation, feedback, client profiles) carries over unchanged.

```yaml
task:
  name: "bandit_reversal_gpt54nano_v1"
  description: |
    On each trial the participant chose one of two boxes. The chosen box
    either revealed a reward (gold coin) or was empty. The probability of
    reward was 75% on the rewarded side and 0% on the non-rewarded side.
    The rewarded side reversed unpredictably across blocks. Switches only
    occurred after rewarded trials, so a reward followed by no-reward is
    diagnostic of a possible reversal.

data:
  max_prompt_trials: 30
  path: "data_g_2019/bandit_reversal.csv"
  id_column: "participant"
  input_columns: ["selected_box", "reward"]
  data2text_function: "narrative"
  narrative_template: |
    The participant chose box {selected_box} and received reward = {reward}.
  value_mappings:
    selected_box:
      "0": "L"
      "1": "R"
  splits:
    prompt: "[1:3]"
```

```yaml
llm:
  # system_prompt updated to mention "two-armed bandit reversal-learning task"
  # instead of "two-step decision-making task"
  guardrails:
    # ...keep the @njit and numerical-stability guardrails unchanged...
    - "Take as input: `action, reward, model_parameters`."   # <-- replaces the 4-arg line

  template_model: |
    @njit
    def cognitive_model(action, reward, model_parameters):
        """Bandit Q-learning template — illustrates format only.
        Bounds: alpha [0,1], beta [0,10]
        """
        alpha, beta = model_parameters
        n_trials = len(action)
        q = np.zeros(2)
        p_choice = np.zeros(n_trials)
        for trial in range(n_trials):
            logits = beta * q
            logits = logits - logits.max()
            exp_q = np.exp(logits)
            probs = exp_q / np.sum(exp_q)
            p_choice[trial] = probs[action[trial]]
            delta = reward[trial] - q[action[trial]]
            q[action[trial]] += alpha * delta
        eps = 1e-10
        return -np.sum(np.log(p_choice + eps))
```

```yaml
baseline:
  model: |
    @njit
    def q_learning_perseveration(action, reward, model_parameters):
        """3-param baseline: alpha [0,1], beta [0,10], persev [-1,1]."""
        alpha, beta, persev = model_parameters
        n_trials = len(action)
        q = np.zeros(2)
        prev = np.zeros(2)
        p_choice = np.zeros(n_trials)
        for t in range(n_trials):
            logits = beta * q + persev * prev
            logits = logits - logits.max()
            exp_q = np.exp(logits)
            probs = exp_q / np.sum(exp_q)
            p_choice[t] = probs[action[t]]
            delta = reward[t] - q[action[t]]
            q[action[t]] += alpha * delta
            prev[:] = 0.0
            prev[action[t]] = 1.0
        eps = 1e-10
        return -np.sum(np.log(p_choice + eps))

parameter_recovery:
  enabled: false   # <-- disabled for v1; no bandit simulator in gecco/

# individual_differences_eval block: REMOVED (no matching self-report data)
```

### 3. Client profile cleanup

The `clients:` block in the source config has a `hybrid` profile whose prompt explicitly references model-based vs. model-free arbitration — meaningless for a single-stage bandit. Replace its `system_prompt_suffix` with bandit-relevant guidance (e.g. "explore mixtures of recency-weighted and Bayesian-update value estimates, or arbitration between different learning rates"), or drop the profile. Recommend dropping; the remaining profiles (`exploit`, `explore`, `minimal`, `complex`, `bayesian`, `latent_state_inference`) all transfer cleanly — `bayesian` and `latent_state_inference` are particularly apt for a reversal-learning task.

### 4. What does NOT change

- No edits to anything under `gecco/` — the harness is fully config-driven for this signature change.
- No edits to `bash/run_gecco_distributed.sh` or `scripts/launch_distributed.py` — they pass `--config` through.
- No README edits (deferred per user).

---

## Files

- **New:** `data_g_2019/preprocess_bandit.py`
- **New (output of preprocessor):** `data_g_2019/bandit_reversal.csv`
- **New:** `config/bandit_reversal_gpt54nano.yaml`

The user will need to drop the raw data file in place (path configured at the top of the preprocessor) before running.

---

## Verification

1. **Preprocessor runs and produces sane data:**
   ```bash
   cd /Users/k2582994/Documents/GitHub/gecco_mh/data_g_2019
   python preprocess_bandit.py
   # expect: "N participants: ...", "mean trials/participant: ...",
   #         "fraction reward: ~0.4-0.5", "wrote bandit_reversal.csv"
   ```

2. **Data + narrative dry run** (no LLM call, no SLURM):
   ```bash
   cd /Users/k2582994/Documents/GitHub/gecco_mh
   python -c "
   from config.schema import load_config
   from gecco.prepare_data.io import load_data
   from gecco.prepare_data.data2text import get_data2text_function
   cfg = load_config('config/bandit_reversal_gpt54nano.yaml')
   df = load_data(cfg.data.path, cfg.data.input_columns + [cfg.data.id_column, 'trial'])
   fn = get_data2text_function(cfg.data.data2text_function)
   print(fn(df.head(20), id_col=cfg.data.id_column,
           template=cfg.data.narrative_template, fit_type='group',
           value_mappings=cfg.data.value_mappings)[:600])
   "
   ```
   Expect "Participant 0: ... The participant chose box L and received reward = 1." lines.

3. **Baseline model fits without errors** — confirms the 2-arg signature flows through:
   ```bash
   python scripts/launch_distributed.py \
       --config bandit_reversal_gpt54nano.yaml \
       --conda-env gecco_mh \
       --profiles minimal \
       --dry-run            # inspect the sbatch first
   ```
   Then re-run without `--dry-run` for a real SLURM submission, and tail the log:
   ```bash
   tail -f logs/gecco-client-*_0.out
   ```
   Look for "Fitted N/N participants" and a finite mean-BIC line from the baseline.
