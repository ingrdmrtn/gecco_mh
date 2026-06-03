# Plan: Add an efficacy-prediction task (passive observation + sparse 1–7 ratings)

## Context

Third task type for GeCCo. Each trial, the participant **passively observes** a binary outcome (efficacy / no efficacy). Every 3–4 trials, they are prompted to predict how efficacious upcoming trials will be on a 1–7 Likert scale. There is no choice; the modelling target is the **rating sequence**, predicted from outcome history.

This is structurally novel for GeCCo:
- Two interleaved observation streams (dense binary outcomes, sparse 1–7 ratings).
- The DV is the rating, not a choice.
- Per user direction, the LLM search should explore **three link functions** between latent belief and observed rating: ordinal-thresholded, categorical-softmax, and Gaussian. Best handled with three client profiles that each bias the search toward one link.

The narrative + fitting harness are already config-driven ([gecco/offline_evaluation/fit_generated_models.py:75-84](../gecco/offline_evaluation/fit_generated_models.py#L75-L84) does `model_func(*inputs, x)` with `inputs = [df_p[c].to_numpy() for c in input_cols]`; [gecco/prepare_data/data2text.py:45-115](../gecco/prepare_data/data2text.py#L45-L115) renders any template with arbitrary `value_mappings`). So this task is **config-only** in `gecco/` — no library code needed, as long as parameter recovery stays disabled.

(Per previous direction: README cleanup deferred. Parameter recovery disabled for v1. Raw data needs preprocessing.)

---

## What changes

### 1. Preprocessor — `data_g_2019/preprocess_efficacy.py` (new)

- Load raw file (path placeholder at top of script — user drops file in place).
- Produce **one row per trial** with columns: `participant, subject_id, trial, outcome, rating`.
  - `outcome` ∈ `{0, 1}` (1 = efficacy, 0 = no efficacy).
  - `rating` is int in `{1..7}` on rating trials, `-1` on non-rating trials. Use `-1` (not NaN) so the column stays integer for `@njit` compatibility.
- Add integer `participant` via `pd.factorize(sID)[0]`; keep the raw id as `subject_id`.
- 0-index `trial` to match the convention in [preprocess_study2.py:185-187](../data_g_2019/preprocess_study2.py#L185-L187).
- Drop rows where `outcome` is missing.
- Sanity print: N participants, mean trials/participant, fraction of trials with rating, mean rating among rating trials.
- Output: `data_g_2019/efficacy_prediction.csv`.

### 2. New config — `config/efficacy_prediction_gpt54nano.yaml`

Copy [config/two_step_factors_gpt54nano.yaml](../config/two_step_factors_gpt54nano.yaml) and replace these blocks. Slurm/loop/evaluation/validation/feedback carry over unchanged.

```yaml
task:
  name: "efficacy_prediction_gpt54nano_v1"
  description: |
    On each trial, the participant passively observes a binary outcome:
    efficacy (1) or no efficacy (0). Every 3–4 trials, they are prompted
    to predict how efficacious the upcoming trials will be on a 1-to-7
    Likert scale (1 = not at all, 7 = extremely). Outcomes do not depend
    on the participant's predictions. The participant's task is to track
    the underlying efficacy rate and report calibrated predictions.

data:
  max_prompt_trials: 60          # larger than bandit; rating events are sparse
  path: "data_g_2019/efficacy_prediction.csv"
  id_column: "participant"
  input_columns: ["outcome", "rating"]
  data2text_function: "narrative"
  narrative_template: |
    Trial {trial}: outcome = {outcome}. Predicted efficacy rating = {rating}.
  value_mappings:
    outcome:
      "0": "no efficacy"
      "1": "efficacy"
    rating:
      "-1": "(not asked)"
      "1": "1/7"
      "2": "2/7"
      "3": "3/7"
      "4": "4/7"
      "5": "5/7"
      "6": "6/7"
      "7": "7/7"
  splits:
    prompt: "[1:3]"
```

The existing `narrative` data2text handles the sentinel cleanly: it stringifies via `str(int(vals[col]))` ([data2text.py:79-83](../gecco/prepare_data/data2text.py#L79-L83)), so `"-1"` maps to `(not asked)` and rating trials map to `5/7` etc. Non-rating lines read "Predicted efficacy rating = (not asked)" — slightly verbose, but no new code.

```yaml
llm:
  system_prompt: |
    You are a senior postdoctoral researcher in computational cognitive
    modelling. You are given participant data from an *efficacy
    prediction* task: the participant passively observes binary outcomes
    (efficacy / no efficacy) and is periodically prompted to predict
    upcoming efficacy on a 1–7 Likert scale. There is no choice in this
    task — you are modelling the rating sequence.

    Propose cognitive models that compute, on each trial, a latent
    belief about the efficacy rate (e.g. running average, Beta-Bernoulli
    posterior, leaky integrator, change-point inference) and map that
    belief to a 1–7 rating using one of three link functions:
      (a) ordinal-thresholded — 6 cutpoints on the latent scale yield 7
          ordered categories via a cumulative-Gaussian or cumulative-
          logistic CDF;
      (b) categorical softmax over 7 classes from latent values;
      (c) Gaussian: rating ~ Normal(mu, sigma), with mu a deterministic
          function of latent belief.

    The likelihood is summed only over trials where rating[t] != -1.

  guardrails:
    # ...numerical-stability + @njit guardrails carry over unchanged...
    - "Take as input: `outcome, rating, model_parameters`."
    - "Ratings are integers in {1,...,7}. The sentinel -1 indicates no rating prompt on that trial — skip those trials in the likelihood sum."
    - "Across the search, explore all three link functions: ordinal-thresholded, categorical-softmax, and Gaussian."

  template_model: |
    @njit
    def cognitive_model(outcome, rating, model_parameters):
        """Beta-Bernoulli belief + Gaussian link (illustration only — do not reuse logic).
        Bounds: alpha_prior [0.5, 20], beta_prior [0.5, 20], sigma [0.3, 3.0]
        """
        a_prior, b_prior, sigma = model_parameters
        n_trials = len(outcome)
        a, b = a_prior, b_prior
        nll = 0.0
        eps = 1e-10
        for t in range(n_trials):
            if rating[t] != -1:
                mean = a / (a + b)
                pred = 1.0 + 6.0 * mean        # map [0,1] -> [1,7]
                resid = (rating[t] - pred) / sigma
                nll += 0.5 * resid * resid + np.log(sigma + eps)
            if outcome[t] == 1:
                a += 1.0
            else:
                b += 1.0
        return nll
```

```yaml
baseline:
  model: |
    @njit
    def running_average_gaussian(outcome, rating, model_parameters):
        """3-param baseline: alpha [0,1] (learning rate), scale [0.1, 6], sigma [0.3, 3.0]."""
        alpha, scale, sigma = model_parameters
        n_trials = len(outcome)
        belief = 0.5
        nll = 0.0
        eps = 1e-10
        for t in range(n_trials):
            if rating[t] != -1:
                pred = 1.0 + scale * belief    # roughly maps [0,1] -> [1,1+scale]
                if pred < 1.0: pred = 1.0
                if pred > 7.0: pred = 7.0
                resid = (rating[t] - pred) / sigma
                nll += 0.5 * resid * resid + np.log(sigma + eps)
            belief += alpha * (outcome[t] - belief)
        return nll

parameter_recovery:
  enabled: false                  # no efficacy-task simulator yet; deferred

# individual_differences_eval block: REMOVED
```

### 3. Client profiles biased toward each link function

Replace the `clients:` block. Three link-specific profiles ensure the parallel search covers all three; keep `exploit` and `explore` for general refinement.

```yaml
clients:
  ordinal:
    llm:
      system_prompt_suffix: |
        Use the ordinal-thresholded link. Introduce 6 free cutpoints
        c_1 < c_2 < ... < c_6 on the latent belief scale (enforce
        ordering by cumulatively summing positive deltas). Place rating
        r at probability P(c_{r-1} < latent + noise <= c_r) using a
        cumulative-Gaussian or cumulative-logistic CDF. Sum NLL over
        rating trials only.
      extra_guardrails:
        - "Enforce monotone ordering of cutpoints by reparameterising as cumulative sums of positive deltas (e.g. softplus)."

  categorical:
    llm:
      system_prompt_suffix: |
        Use a categorical-softmax link. From the latent belief, produce
        a 7-vector of logits (e.g. logit_k = -beta * (latent - k/8)^2
        for k in 1..7, or any parameterisation that depends on latent
        belief and on free parameters) and apply softmax. Take the log
        of the chosen-rating probability, summed over rating trials.

  gaussian:
    llm:
      system_prompt_suffix: |
        Use a Gaussian link: rating ~ Normal(mu, sigma), where mu is a
        deterministic function of latent belief and sigma is a free
        parameter. Clamp mu to [1, 7] to avoid pathological NLL at the
        scale endpoints. Sum NLL over rating trials only.

  exploit:
    llm:
      system_prompt_suffix: |
        Refine the best model so far. Make small, targeted changes to
        the belief-update rule (learning rate, prior, change-point
        sensitivity) while keeping the link function unchanged.

  explore:
    llm:
      system_prompt_suffix: |
        Be creative about the belief representation: change-point
        inference with a hazard rate; volatility-tracking Kalman
        filters; leaky integrators with separate gains for confirming
        vs. surprising outcomes; ratings anchored on recent history
        only (recency-weighted). Mix and match with any link function.
      extra_guardrails:
        - "The belief-update rule should differ from any previously proposed model."
```

### 4. What does NOT change

- No edits under `gecco/` — `narrative` + the fitting harness handle the new shape via config alone.
- No edits to launcher / sbatch — they pass `--config` through.
- README untouched (deferred).

---

## Files

- **New:** `data_g_2019/preprocess_efficacy.py`
- **New (preprocessor output):** `data_g_2019/efficacy_prediction.csv`
- **New:** `config/efficacy_prediction_gpt54nano.yaml`

The user provides the raw file; path goes at the top of `preprocess_efficacy.py`.

---

## Verification

1. **Preprocessor:**
   ```bash
   cd /Users/k2582994/Documents/GitHub/gecco_mh/data_g_2019
   python preprocess_efficacy.py
   # expect: N participants, ~25-35% of trials with rating != -1,
   # mean rating in [3, 5]
   ```

2. **Narrative dry run** (no LLM, no SLURM):
   ```bash
   cd /Users/k2582994/Documents/GitHub/gecco_mh
   python -c "
   from config.schema import load_config
   from gecco.prepare_data.io import load_data
   from gecco.prepare_data.data2text import get_data2text_function
   cfg = load_config('config/efficacy_prediction_gpt54nano.yaml')
   df = load_data(cfg.data.path, cfg.data.input_columns + [cfg.data.id_column, 'trial'])
   fn = get_data2text_function(cfg.data.data2text_function)
   print(fn(df.head(20), id_col=cfg.data.id_column,
           template=cfg.data.narrative_template, fit_type='group',
           value_mappings=cfg.data.value_mappings))
   "
   ```
   Expect lines mixing `outcome = efficacy. Predicted efficacy rating = (not asked).` with `... = 5/7.` on rating trials.

3. **Baseline fits cleanly** — confirms the new signature and rating-only NLL flow:
   ```bash
   python scripts/launch_distributed.py \
       --config efficacy_prediction_gpt54nano.yaml \
       --conda-env gecco_mh \
       --profiles gaussian \
       --dry-run
   # then drop --dry-run; tail logs/gecco-client-*_0.out and look for finite mean BIC
   ```

4. **Coverage of all three link functions** — after a few iterations, inspect `results/efficacy_prediction_gpt54nano_v1/models/` and confirm at least one model exists from each of the `ordinal`, `categorical`, and `gaussian` profiles (file names include the client id, per [readme_open.md:216](../readme_open.md#L216)).
