# Multi-Reviewer Panel for GeCCo Cognitive Model Search

## Context: Two-Sided Multi-Agent Architecture

GeCCo is evolving from a single-agent loop into a **two-sided multi-agent system**. The proposal side and the evaluation side each use multiple specialist agents working in parallel:

### What's already built: Multi-Agent Proposal (distributed clients)

The distributed GeCCo pipeline (`scripts/run_gecco_distributed.py`, `gecco/coordination.py`) already parallelizes **model generation** across multiple SLURM job array clients. Each client runs the full GeCCo loop with a different search strategy:

```
         vLLM Server (GPU node)
                |  HTTP
     +----------+----------+
     |          |          |
 Client 0    Client 1   Client 2
 (exploit)   (explore)  (diverse)
     |          |          |
     +----+-----+-----+---+
          |
   SharedRegistry (JSON)
   - global_best model
   - tried_param_sets (deduplicated)
   - iteration_history (all clients)
```

- **Diversity mechanism:** Different client profiles (exploit, explore, diverse, minimal, hybrid, bayesian, complex) use different temperatures + system prompt suffixes
- **Coordination:** `SharedRegistry` on shared filesystem with file locking
- **Cross-pollination:** Each client syncs history from all others via `_sync_from_registry()`, so the `FeedbackGenerator` / `LLMFeedbackGenerator` sees models from every client when building trajectory, landscape, and factor coverage analyses

### What this plan adds: Multi-Agent Evaluation (reviewer panel)

We are now building the **mirror image** for the evaluation side. Instead of one feedback signal (fit metrics), each proposed model gets evaluated by a panel of 5 specialist reviewer agents, each with domain-specific expertise:

```
         Per Client, Per Iteration:
         
         LLM proposes model(s)
                |
         Deterministic checks
         (code runs, recovery, fitting)
                |
         ┌──────┼──────┬──────┬──────┐
         TH     NI     FQ     BD*    MC
       theory  numeric fitting bayes  comparison
         └──────┼──────┴──────┴──────┘
                |                    * BD only for Bayesian fitting
         PanelReport (structured verdicts)
                |
         → Saved to results/{task}/reviews/
         → Attached to iteration_results["review"]
         → Flows into feedback for next iteration
         → Shared across clients via SharedRegistry
```

### How the two sides connect

The two multi-agent systems compose naturally because they share the same coordination infrastructure:

| | Proposal Side (built) | Evaluation Side (this plan) |
|---|---|---|
| **Agents** | Client profiles (exploit, explore, diverse, etc.) | Reviewer specialists (TH, NI, FQ, BD, MC) |
| **Parallelism** | SLURM job array across nodes | ThreadPoolExecutor within a single client |
| **Coordination** | SharedRegistry JSON on filesystem | PanelReport attached to iteration_results |
| **Cross-pollination** | `_sync_from_registry()` merges all clients' history into `feedback.history` | Review verdicts flow through the same `feedback.history` → visible to all clients |
| **Diversity** | Different temperatures + prompt suffixes | Different domain vocabularies + failure pattern catalogs |
| **Output** | Candidate models with fit metrics | Structured verdicts with evidence-backed issues |
| **Feeds into** | `FeedbackGenerator.get_feedback()` → next iteration prompt | Same path: review data in `iteration_results` → `feedback.history` → next prompt |

**The key integration:** When Client 0 (exploit) reviews a model and attaches the PanelReport to its `iteration_results`, that review data gets pushed to the SharedRegistry via `_update_registry()`. When Client 1 (explore) syncs via `_sync_from_registry()`, it pulls Client 0's iteration history — including the review verdicts. So the LLMFeedbackGenerator on Client 1 can see that Client 0's model was flagged for "Parameter Soup" by the Theory Specialist, and guide Client 1's next proposal accordingly.

**Core principle:** Generation and evaluation must be separate. The proposing LLM never reviews its own output. The reviewer agents are distinct LLM calls with specialist system prompts — optionally using a completely different model/provider.

---

## Full Pipeline View (Single Iteration, Single Client)

```
1. _sync_from_registry()
   └─ Pull global_best, tried_params, iteration_history (incl. reviews) from all clients

2. feedback.get_feedback()
   ├─ Level 1: BIC trajectory (all clients)
   ├─ Level 2: Model landscape ranked table (all clients)
   ├─ Level 3: Factor coverage + stagnation (all clients)
   ├─ Level 4: Fit quality per participant
   ├─ Level 5: Individual differences R²
   ├─ Level 6: Cross-model subgroup comparison
   ├─ NEW Level 7: Expert Review Panel summary (all clients' recent reviews)
   └─ LLM judge synthesizes all levels into guidance

3. prompt_builder.build_input_prompt(feedback)
   └─ Task + data + template + guardrails + feedback (now includes reviewer verdicts)

4. generate_models(prompt)
   └─ LLM proposes N candidate models (structured JSON)

5. Per-model deterministic evaluation:
   ├─ Parameter recovery check (optional)
   ├─ Model fitting (L-BFGS-B / HBI)
   ├─ Individual differences regression (optional)
   └─ Record to iteration_results

6. *** NEW: ReviewPanel.review_model() for each fitted model ***
   ├─ Activate relevant specialists (skip BD for MLE)
   ├─ Run in parallel via ThreadPoolExecutor
   ├─ Aggregate into PanelReport
   └─ Attach report to iteration_results["review"]

7. feedback.record_iteration()
   └─ Stores iteration_results (now including reviews) in history

8. _update_registry()
   └─ Push iteration_results (including reviews) to SharedRegistry
       → Other clients will see these reviews on their next sync
```

The review panel is **opt-in** via config (`review_panel.enabled: true`) and does not break the existing single-agent or multi-client pipeline.

---

## New Module: `gecco/review_panel/`

### File Structure

```
gecco/review_panel/
    __init__.py           # exports ReviewPanel, ReviewVerdict, PanelReport
    schemas.py            # dataclass definitions for review output
    base_reviewer.py      # BaseReviewer with shared LLM call logic
    reviewers.py          # 5 specialist subclasses
    orchestrator.py       # ReviewPanel: activation, parallel dispatch, aggregation
    prompts.py            # System prompts + failure pattern catalogs
```

---

## Step 1: Review Output Schema (`schemas.py`)

```python
@dataclass
class ReviewIssue:
    pattern_name: str        # Named failure pattern (e.g. "Bare Exponentiation")
    severity: str            # "critical" | "major" | "minor" | "info"
    description: str
    evidence: str            # Line numbers, parameter values, diagnostics
    recommendation: str

@dataclass
class ReviewVerdict:
    reviewer_id: str         # "TH" | "NI" | "FQ" | "BD" | "MC"
    reviewer_name: str
    verdict: str             # "pass" | "revise" | "reject"
    confidence: float        # 0-1
    issues: List[ReviewIssue]
    summary: str
    escalations: List[str]   # Cross-domain flags for other specialists

@dataclass
class PanelReport:
    model_name: str
    reviewers_activated: List[str]
    verdicts: List[ReviewVerdict]
    consensus_verdict: str   # worst-of rule: any "reject" → reject, any "revise" → revise
    critical_issues: List[ReviewIssue]
    feedback_text: str       # Pre-formatted for prompt injection
```

Both `ReviewVerdict` and `PanelReport` get `to_dict()` / `from_dict()` for JSON serialization.

---

## Step 2: Specialist Prompts (`prompts.py`)

Each specialist defined as a dict with:
- `identity`: Under 50 tokens (the "15-year practitioner" test)
- `vocabulary`: Rich domain terms that activate deep expert knowledge
- `failure_patterns`: Named patterns with description + fix
- `review_template`: Instructions for structured JSON output

**5 specialists** (full text from user's specification):

| ID | Name | Domain | Activation |
|----|------|--------|------------|
| TH | Cognitive Theory Specialist | Plausibility, process models, construct validity | Always |
| NI | Numerical Integrity Specialist | Log-likelihood stability, constraints, NaN | Always |
| FQ | Fitting & Recovery Specialist | Multi-start, recovery, overfitting, identifiability | Always |
| BD | Bayesian Diagnostics Specialist | R-hat, divergences, ESS, funnels, PPC | Only when `fitting_method` involves Bayesian/HBI |
| MC | Model Comparison & Selection Specialist | Metric agreement, mimicry, parsimony, qualitative fit | Always |

---

## Step 3: Base Reviewer (`base_reviewer.py`)

```python
class BaseReviewer:
    def __init__(self, reviewer_id, config, generate_fn, cfg):
        # config = entry from REVIEWER_CONFIGS
        # generate_fn = reference to GeCCoModelSearch.generate() — avoids duplicating
        #   provider dispatch logic. Called with (model, tokenizer, prompt) but we
        #   temporarily swap cfg.llm.system_prompt to the reviewer's identity prompt.
        self.generate_fn = generate_fn

    def review(self, context: dict) -> ReviewVerdict:
        prompt = self._build_review_prompt(context)
        raw = self._call_llm(prompt)
        return self._parse_verdict(raw)

    def _call_llm(self, prompt: str) -> str:
        # Temporarily override cfg.llm.system_prompt with reviewer identity,
        # call self.generate_fn(model, tokenizer, prompt), then restore.
        # This wraps the existing generate() with no duplication.

    def _build_review_prompt(self, context: dict) -> str:
        # Assembles: vocabulary + failure patterns + model code
        # + fit results + review instructions + JSON output schema
        # (identity goes in system_prompt, not user prompt)

    def _parse_verdict(self, raw: str) -> ReviewVerdict:
        # JSON parse with regex fallback (reuse patterns from structured_output.py)
```

**Decision:** BaseReviewer wraps `GeCCoModelSearch.generate()` by reference rather than duplicating provider dispatch. The orchestrator receives the generate function at init time. When a separate reviewer LLM is configured (`review_panel.reviewer_provider`), the orchestrator loads it via `load_llm()` and passes a different model/tokenizer to the wrapped call.

**Review context dict** passed to each reviewer:
- `model_code`: Python source of the candidate
- `model_name`: display name
- `fit_results`: from `run_fit()` — metric_value, param_names, parameter_values, eval_metrics, participant_n_trials
- `recovery_results`: if available — mean_r, per_param_r
- `id_results`: individual differences R-squared if available
- `data_summary`: condensed dataset description
- `iteration_history`: top-N previous models with metrics (from feedback.history)
- `fitting_method`: "mle" | "hbi" | "bayesian"

---

## Step 4: Specialist Subclasses (`reviewers.py`)

Each inherits `BaseReviewer` and may override `_build_review_prompt()` to emphasize domain-specific context:

- **CognitiveTheoryReviewer** — emphasizes parameter-to-construct mapping, includes task description
- **NumericalIntegrityReviewer** — focuses on model code, includes parameter bounds
- **FittingRecoveryReviewer** — focuses on fit results + recovery stats, includes n_starts config
- **BayesianDiagnosticsReviewer** — focuses on MCMC diagnostics (when available)
- **ModelComparisonReviewer** — includes iteration history landscape, all models from current iteration

---

## Step 5: Orchestrator (`orchestrator.py`)

```python
class ReviewPanel:
    def __init__(self, model, tokenizer, cfg):
        # Optionally load separate reviewer LLM if cfg.review_panel.reviewer_provider set
        self.reviewers = self._build_reviewers()

    def review_model(self, context: dict) -> PanelReport:
        activated = self._get_activated_reviewers(context)
        # Parallel via concurrent.futures.ThreadPoolExecutor
        # (codebase is synchronous; ThreadPoolExecutor handles I/O-bound API calls)
        verdicts = self._run_parallel(activated, context)
        # Optional: cross-agent escalation pass
        if self.cfg_panel.escalation:
            verdicts = self._handle_escalations(verdicts, context)
        return self._build_report(context, verdicts)

    def _get_activated_reviewers(self, context):
        # BD only when fitting_method is bayesian/hbi
        # Respect cfg.review_panel.disabled_reviewers

    def format_for_feedback(self, report: PanelReport) -> str:
        # Structured text block for prompt injection
```

**Consensus rule:** worst-of — any "reject" → reject, any "revise" → revise, all "pass" → pass.

**Error handling:** A reviewer failure (timeout, parse error) produces a fallback verdict with `verdict="error"` and does not block the pipeline.

---

## Step 6: Integration into `run_gecco.py`

### In `__init__()` (after line 83):
```python
self.review_panel = None
if hasattr(cfg, 'review_panel') and getattr(cfg.review_panel, 'enabled', False):
    from gecco.review_panel import ReviewPanel
    self.review_panel = ReviewPanel(model, tokenizer, cfg)
    (self.results_dir / "reviews").mkdir(parents=True, exist_ok=True)
```

### In `run_n_shots()` — after line 629 (after per-model loop, before `self._set_activity("saving results")`):
```python
# --- Review panel (optional) ---
if self.review_panel is not None:
    for result in iteration_results:
        if result.get("metric_name") in ("FIT_ERROR", "RECOVERY_FAILED"):
            continue
        context = self._build_review_context(result, iteration_results, it)
        report = self.review_panel.review_model(context)
        result["review"] = report.to_dict()

    # Save reviews
    review_file = self.results_dir / "reviews" / f"iter{it}{tag}_run{run_idx}.json"
    with open(review_file, "w") as f:
        json.dump([r.get("review") for r in iteration_results if "review" in r], f, indent=2)
```

### New helper method `_build_review_context()`:
Assembles context dict from fit result + recovery data + iteration history + data summary.

### Cross-client review sharing (no additional code needed):

The review data flows across clients automatically through the existing coordination infrastructure:

1. **Attach:** `result["review"] = report.to_dict()` adds review to `iteration_results`
2. **Record locally:** `self.feedback.record_iteration(it, iteration_results)` stores it in `self.feedback.history`
3. **Push to registry:** `self._update_registry(it, iteration_results)` writes the full `iteration_results` (including `"review"` keys) to `SharedRegistry.iteration_history`
4. **Pull by other clients:** `self._sync_from_registry()` merges remote `iteration_history` entries into `self.feedback.history`
5. **Surface in feedback:** `_build_review_summary()` reads reviews from `self.feedback.history` — which now includes reviews from all clients

This means Client 1 (explore) can see that Client 0's (exploit) model was flagged by the Theory Specialist for "Parameter Soup", and the feedback generator can steer Client 1 away from the same structural issue. The review panel amplifies the cross-pollination that the distributed proposal system already provides.

---

## Step 7: Feedback Integration (`construct_feedback/feedback.py`)

### In `FeedbackGenerator.get_feedback()` — add new section:
```python
review_text = self._build_review_summary()
if review_text:
    parts.append(f"## Expert Review Panel Feedback\n{review_text}")
```

### New method `_build_review_summary()`:
- Pulls `"review"` key from most recent iteration results in `self.history`
- Formats each reviewer's verdict + critical issues
- Highlights cross-agent escalations

### In `LLMFeedbackGenerator`:
- Include review panel output in the judge prompt context so the LLM judge can synthesize reviewer findings.

---

## Step 8: Config Extension

Since `load_config()` uses `dict_to_namespace()` (not Pydantic validation), no schema changes are strictly required — just add YAML keys:

```yaml
review_panel:
  enabled: true
  disabled_reviewers: []          # e.g. ["BD"] for MLE-only
  parallel: true
  escalation: false
  reviewer_provider: null         # null = use same LLM as proposer
  reviewer_model: null            # null = use same model
  max_reviewer_tokens: 2048
```

Optionally add `ReviewPanelConfig` to `config/schema.py` for documentation/validation.

---

## Critical Files to Modify

| File | Change |
|------|--------|
| `gecco/review_panel/__init__.py` | **NEW** — exports |
| `gecco/review_panel/schemas.py` | **NEW** — ReviewIssue, ReviewVerdict, PanelReport |
| `gecco/review_panel/prompts.py` | **NEW** — 5 specialist prompt configs |
| `gecco/review_panel/base_reviewer.py` | **NEW** — BaseReviewer with LLM dispatch |
| `gecco/review_panel/reviewers.py` | **NEW** — 5 specialist subclasses |
| `gecco/review_panel/orchestrator.py` | **NEW** — ReviewPanel orchestrator |
| `gecco/run_gecco.py` | Add review panel init (~line 83) and review step (~line 631) |
| `gecco/construct_feedback/feedback.py` | Add `_build_review_summary()` and integrate into `get_feedback()` |
| `config/schema.py` | Optional: add `ReviewPanelConfig` |

---

## Implementation Order

1. `schemas.py` — no dependencies
2. `prompts.py` — no dependencies (user's full spec transcribed)
3. `base_reviewer.py` — depends on schemas, reuses LLM dispatch pattern from `run_gecco.py:85-282`
4. `reviewers.py` — depends on base_reviewer + prompts
5. `orchestrator.py` — depends on reviewers + schemas
6. `__init__.py` — exports
7. `run_gecco.py` — integration (init + review step + `_build_review_context()`)
8. `feedback.py` — add review summary section

---

## Verification

1. **Unit test:** Create a mock model/tokenizer, run `ReviewPanel.review_model()` with sample context, verify PanelReport structure
2. **Integration test:** Run `scripts/run_gecco_distributed.py --test --config <yaml_with_review_panel>` with the small Qwen model and verify:
   - Reviews saved to `results/{task}/reviews/`
   - Feedback text includes "Expert Review Panel" section
   - Pipeline completes without error
3. **Output inspection:** Check that reviewer verdicts cite specific evidence (line numbers, parameter values) and use named failure patterns
4. **Selective activation:** Run with `disabled_reviewers: ["BD"]` and verify BD is not called; run with HBI fitting and verify BD is activated

---

## Design Decisions

- **ThreadPoolExecutor over asyncio:** The entire codebase is synchronous. Converting to async would be invasive. ThreadPoolExecutor parallelizes I/O-bound API calls without restructuring.
- **Wrap existing generate(), don't duplicate:** BaseReviewer holds a reference to `GeCCoModelSearch.generate()` and swaps the system prompt. No provider dispatch duplication.
- **Feedback only, no active gating:** Reviewer verdicts feed into the next iteration's prompt as structured feedback. They do NOT influence model selection in the current iteration. This is safer — avoids discarding valid models due to reviewer hallucination, and lets the researcher inspect reviewer output before it drives decisions.
- **Separate reviewer LLM (optional):** Allows using a cheaper/faster model for review (e.g., Gemini Flash) while keeping a stronger model for generation. Enforces generation/evaluation separation.
- **Worst-of consensus:** Conservative — a single "reject" rejects the model. Appropriate for scientific validation where false positives are costly.
- **No HuggingFace parallelism:** Local HF models are not thread-safe. When `provider` is HF, reviewers run serially. This is detected automatically.
