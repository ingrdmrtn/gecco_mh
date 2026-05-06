# LangGraph Orchestration for GeCCo Multi-Agent Pipeline

## Why LangGraph?

The current GeCCo pipeline is a procedural `for` loop with conditionals scattered through `run_n_shots()`. As we add the reviewer panel, the control flow becomes:

- Sequential stages with conditional branching (skip BD, skip recovery)
- Fan-out/fan-in parallelism (5 reviewers)
- Cycles (iteration loop with early stopping)
- Cross-client state (SharedRegistry sync)

LangGraph's `StateGraph` makes this explicit: each stage is a node, control flow is edges, and shared state flows through a typed `State` object. It also gives us built-in support for parallel node execution, conditional routing, and checkpointing.

---

## State Schema

```python
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, START, END

class GeCCoState(TypedDict):
    # --- Iteration control ---
    iteration: int
    run_idx: int
    max_iterations: int
    client_id: str
    stop: bool

    # --- Shared data (persists across iterations) ---
    cfg: dict
    df: dict                          # serialized dataframe
    data_summary: str
    best_model: Optional[str]
    best_metric: float
    best_params: list
    tried_param_sets: list
    feedback_history: list            # list of iteration result dicts

    # --- Per-iteration working data (reset each cycle) ---
    feedback_text: str
    prompt: str
    raw_generation: str
    parsed_models: list               # list of model dicts
    current_model_idx: int
    iteration_results: list           # accumulates per-model results
    current_fit_result: Optional[dict]
    current_recovery_result: Optional[dict]

    # --- Review panel ---
    review_reports: list              # PanelReport dicts for this iteration
```

---

## Graph Schematic

```
                        ┌─────────────────────────────────────────┐
                        │              ITERATION CYCLE             │
                        │                                         │
 START ──► sync_registry ──► build_feedback ──► build_prompt      │
                                                     │            │
                                                generate_models   │
                                                     │            │
                                                parse_response    │
                                                     │            │
                                            ┌── next_model ◄──┐  │
                                            │        │        │  │
                                            │   ┌────┴────┐   │  │
                                            │   │ recovery │   │  │
                                            │   │  check?  │   │  │
                                            │   └────┬────┘   │  │
                                            │   yes/ │ \skip  │  │
                                            │      ▼         │  │
                                            │  run_recovery   │  │
                                            │   pass/ │ \fail │  │
                                            │        ▼       │  │
                                            │    fit_model ◄──┘  │
                                            │        │           │
                                            │   eval_individual  │
                                            │   _differences     │
                                            │        │           │
                                            │  record_result     │
                                            │        │           │
                                            │   more_models? ────┘
                                            │        │ no
                                            │        ▼
                                            │  ┌─ review_panel ─┐
                                            │  │                │
                                            │  │  ┌──┬──┬──┬──┐│
                                            │  │  TH NI FQ BD MC│  (parallel)
                                            │  │  └──┴──┴──┴──┘│
                                            │  │                │
                                            │  │ aggregate_     │
                                            │  │ verdicts       │
                                            │  └────────────────┘
                                            │        │
                                            │  save_iteration
                                            │        │
                                            │  update_registry
                                            │        │
                                            │  should_continue? ─── no ──► END
                                            │        │ yes
                                            └────────┘
```

---

## Node Definitions

```python
from langgraph.graph import StateGraph, START, END


# ── 1. Registry sync ────────────────────────────────────────────
def sync_registry(state: GeCCoState) -> dict:
    """Pull global_best, tried_params, iteration_history from SharedRegistry.
    Merges cross-client results into feedback_history."""
    # Currently: GeCCoModelSearch._sync_from_registry()
    ...
    return {"best_model": ..., "tried_param_sets": ..., "feedback_history": ...}


# ── 2. Feedback ─────────────────────────────────────────────────
def build_feedback(state: GeCCoState) -> dict:
    """Build multi-level feedback from history (trajectory, landscape,
    fit quality, R-squared, review summaries from all clients)."""
    # Currently: FeedbackGenerator.get_feedback() / LLMFeedbackGenerator.get_feedback()
    ...
    return {"feedback_text": feedback}


# ── 3. Prompt ───────────────────────────────────────────────────
def build_prompt(state: GeCCoState) -> dict:
    """Assemble task + data + template + guardrails + feedback into LLM prompt."""
    # Currently: PromptBuilderWrapper.build_input_prompt()
    ...
    return {"prompt": prompt}


# ── 4. Generate ─────────────────────────────────────────────────
def generate_models(state: GeCCoState) -> dict:
    """Call LLM (any provider) to propose N candidate models."""
    # Currently: GeCCoModelSearch.generate_models()
    ...
    return {"raw_generation": text, "parsed_models": models, "current_model_idx": 0}


# ── 5. Parse ────────────────────────────────────────────────────
def parse_response(state: GeCCoState) -> dict:
    """Extract structured model dicts from raw LLM output. JSON + regex fallback."""
    # Currently: parse_model_response() + optional reflection
    ...
    return {"parsed_models": parsed}


# ── 6. Model loop entry ────────────────────────────────────────
def next_model(state: GeCCoState) -> dict:
    """Load the next model from parsed_models for evaluation."""
    ...
    return {"current_model_idx": state["current_model_idx"]}


# ── 7. Parameter recovery ──────────────────────────────────────
def run_recovery(state: GeCCoState) -> dict:
    """Simulate from known params, refit, check correlation.
    Sets current_recovery_result with pass/fail + per-param r."""
    # Currently: ParameterRecoveryChecker.check()
    ...
    return {"current_recovery_result": recovery}


# ── 8. Model fitting ───────────────────────────────────────────
def fit_model(state: GeCCoState) -> dict:
    """Fit model to eval data via multi-start L-BFGS-B or HBI.
    Returns metric_value, param_names, parameter_values, eval_metrics."""
    # Currently: run_fit() / run_fit_hierarchical()
    ...
    return {"current_fit_result": fit_res}


# ── 9. Individual differences ──────────────────────────────────
def eval_individual_differences(state: GeCCoState) -> dict:
    """Regress fitted parameters on questionnaire measures. Returns R-squared."""
    # Currently: evaluate_individual_differences()
    ...
    return {"current_fit_result": {**state["current_fit_result"], "id_results": id_res}}


# ── 10. Record result ──────────────────────────────────────────
def record_result(state: GeCCoState) -> dict:
    """Append current model's fit result to iteration_results. Update best if improved."""
    ...
    return {"iteration_results": updated, "best_model": ..., "best_metric": ...}


# ── 11. Review panel (5 specialists in parallel) ───────────────
def review_theory(state: GeCCoState) -> dict:
    """TH: Cognitive theory plausibility review."""
    ...
    return {"review_reports": [verdict_th]}

def review_numerical(state: GeCCoState) -> dict:
    """NI: Numerical integrity review."""
    ...
    return {"review_reports": [verdict_ni]}

def review_fitting(state: GeCCoState) -> dict:
    """FQ: Fitting & recovery quality review."""
    ...
    return {"review_reports": [verdict_fq]}

def review_bayesian(state: GeCCoState) -> dict:
    """BD: Bayesian diagnostics review (conditional)."""
    ...
    return {"review_reports": [verdict_bd]}

def review_comparison(state: GeCCoState) -> dict:
    """MC: Model comparison & selection review."""
    ...
    return {"review_reports": [verdict_mc]}

def aggregate_verdicts(state: GeCCoState) -> dict:
    """Compile all reviewer verdicts into PanelReport per model.
    Consensus rule: worst-of. Attach to iteration_results."""
    ...
    return {"iteration_results": updated_with_reviews}


# ── 12. Save & sync ────────────────────────────────────────────
def save_iteration(state: GeCCoState) -> dict:
    """Save iteration_results to bics/, reviews/, models/ files.
    Record in feedback_history."""
    ...
    return {"feedback_history": updated}

def update_registry(state: GeCCoState) -> dict:
    """Push iteration_results (including reviews) to SharedRegistry.
    Other clients will see these on their next sync_registry call."""
    ...
    return {}
```

---

## Conditional Routing Functions

```python
def should_run_recovery(state: GeCCoState) -> str:
    """Route to recovery check or straight to fitting."""
    if state["cfg"].get("parameter_recovery", {}).get("enabled", False):
        return "run_recovery"
    return "fit_model"

def recovery_passed(state: GeCCoState) -> str:
    """After recovery: continue to fitting or skip model."""
    if state["current_recovery_result"]["passed"]:
        return "fit_model"
    return "record_result"  # record as RECOVERY_FAILED, move to next model

def more_models(state: GeCCoState) -> str:
    """After recording a result: loop back for next model or proceed to review."""
    if state["current_model_idx"] + 1 < len(state["parsed_models"]):
        return "next_model"
    return "review_panel"   # all models evaluated, run reviews

def should_activate_bayesian(state: GeCCoState) -> str:
    """BD reviewer only activates for Bayesian/HBI fitting methods."""
    method = state["cfg"].get("evaluation", {}).get("fitting_method", "mle")
    if "bayes" in method.lower() or "hbi" in method.lower():
        return "review_bayesian"
    return "skip_bayesian"

def should_continue(state: GeCCoState) -> str:
    """After an iteration: loop or stop."""
    if state["stop"] or state["iteration"] >= state["max_iterations"]:
        return END
    return "sync_registry"  # next iteration
```

---

## Graph Assembly

```python
def build_gecco_graph() -> StateGraph:
    graph = StateGraph(GeCCoState)

    # ── Register nodes ──────────────────────────────────────
    graph.add_node("sync_registry", sync_registry)
    graph.add_node("build_feedback", build_feedback)
    graph.add_node("build_prompt", build_prompt)
    graph.add_node("generate_models", generate_models)
    graph.add_node("parse_response", parse_response)
    graph.add_node("next_model", next_model)
    graph.add_node("run_recovery", run_recovery)
    graph.add_node("fit_model", fit_model)
    graph.add_node("eval_individual_differences", eval_individual_differences)
    graph.add_node("record_result", record_result)

    # Review panel nodes (run in parallel via fan-out)
    graph.add_node("review_theory", review_theory)
    graph.add_node("review_numerical", review_numerical)
    graph.add_node("review_fitting", review_fitting)
    graph.add_node("review_bayesian", review_bayesian)
    graph.add_node("review_comparison", review_comparison)
    graph.add_node("aggregate_verdicts", aggregate_verdicts)

    graph.add_node("save_iteration", save_iteration)
    graph.add_node("update_registry", update_registry)

    # ── Sequential edges (main pipeline) ────────────────────
    graph.add_edge(START, "sync_registry")
    graph.add_edge("sync_registry", "build_feedback")
    graph.add_edge("build_feedback", "build_prompt")
    graph.add_edge("build_prompt", "generate_models")
    graph.add_edge("generate_models", "parse_response")
    graph.add_edge("parse_response", "next_model")

    # ── Per-model evaluation (conditional + loop) ───────────
    graph.add_conditional_edges("next_model", should_run_recovery,
                                {"run_recovery": "run_recovery", "fit_model": "fit_model"})
    graph.add_conditional_edges("run_recovery", recovery_passed,
                                {"fit_model": "fit_model", "record_result": "record_result"})
    graph.add_edge("fit_model", "eval_individual_differences")
    graph.add_edge("eval_individual_differences", "record_result")
    graph.add_conditional_edges("record_result", more_models,
                                {"next_model": "next_model", "review_panel": "review_theory"})

    # ── Review panel (parallel fan-out → fan-in) ────────────
    # All reviewers receive the same state; their outputs merge
    # into review_reports via LangGraph's state reducer (list append).
    #
    # Fan-out: record_result → [TH, NI, FQ, MC] + conditionally BD
    graph.add_edge("record_result", "review_theory")    # via more_models routing
    graph.add_edge("record_result", "review_numerical")
    graph.add_edge("record_result", "review_fitting")
    graph.add_edge("record_result", "review_comparison")
    graph.add_conditional_edges("record_result", should_activate_bayesian,
                                {"review_bayesian": "review_bayesian",
                                 "skip_bayesian": "aggregate_verdicts"})

    # Fan-in: all reviewers → aggregate
    graph.add_edge("review_theory", "aggregate_verdicts")
    graph.add_edge("review_numerical", "aggregate_verdicts")
    graph.add_edge("review_fitting", "aggregate_verdicts")
    graph.add_edge("review_bayesian", "aggregate_verdicts")
    graph.add_edge("review_comparison", "aggregate_verdicts")

    # ── Post-review pipeline ────────────────────────────────
    graph.add_edge("aggregate_verdicts", "save_iteration")
    graph.add_edge("save_iteration", "update_registry")
    graph.add_conditional_edges("update_registry", should_continue,
                                {"sync_registry": "sync_registry", END: END})

    return graph.compile()
```

---

## Invocation

```python
# Single client entry point
initial_state: GeCCoState = {
    "iteration": 0,
    "run_idx": 0,
    "max_iterations": cfg.loop.max_iterations,
    "client_id": client_id,
    "stop": False,
    "cfg": cfg_dict,
    "df": df.to_dict(),
    "data_summary": data_text,
    "best_model": None,
    "best_metric": float("inf"),
    "best_params": [],
    "tried_param_sets": [],
    "feedback_history": [],
    "feedback_text": "",
    "prompt": "",
    "raw_generation": "",
    "parsed_models": [],
    "current_model_idx": 0,
    "iteration_results": [],
    "current_fit_result": None,
    "current_recovery_result": None,
    "review_reports": [],
}

app = build_gecco_graph()
final_state = app.invoke(initial_state)
```

---

## How This Maps to the Existing Codebase

| LangGraph Node | Current Code Location | Function/Method |
|---|---|---|
| `sync_registry` | `run_gecco.py:366-412` | `GeCCoModelSearch._sync_from_registry()` |
| `build_feedback` | `feedback.py:697-778` | `FeedbackGenerator.get_feedback()` |
| `build_prompt` | `prompt.py:2-148` | `build_prompt()` / `PromptBuilderWrapper` |
| `generate_models` | `run_gecco.py:284-358` | `GeCCoModelSearch.generate_models()` |
| `parse_response` | `structured_output.py:178-213` | `parse_model_response()` |
| `run_recovery` | `parameter_recovery.py` | `ParameterRecoveryChecker.check()` |
| `fit_model` | `fit_generated_models.py:13-113` | `run_fit()` / `run_fit_hierarchical()` |
| `eval_individual_differences` | `individual_differences.py:28-174` | `evaluate_individual_differences()` |
| `review_*` | **NEW** `review_panel/reviewers.py` | `BaseReviewer.review()` |
| `aggregate_verdicts` | **NEW** `review_panel/orchestrator.py` | `ReviewPanel._build_report()` |
| `save_iteration` | `run_gecco.py:631-642` | File saves + `feedback.record_iteration()` |
| `update_registry` | `run_gecco.py:420-433` | `GeCCoModelSearch._update_registry()` |

---

## What LangGraph Gives Us Over the Current Procedural Loop

1. **Explicit control flow** — The graph IS the documentation. No need to trace through 250 lines of `run_n_shots()` to understand the pipeline.
2. **Native parallel fan-out** — Reviewer agents run in parallel without manual ThreadPoolExecutor management. LangGraph handles it.
3. **Conditional routing as first-class** — `should_run_recovery`, `should_activate_bayesian`, `should_continue` are named functions, not nested `if` blocks.
4. **Checkpointing** — LangGraph can checkpoint state between nodes. If a client crashes mid-iteration, it resumes from the last completed node instead of restarting the iteration.
5. **Observability** — LangGraph Studio provides visual tracing of which nodes executed, what state looked like at each step, and where failures occurred.
6. **Composability** — The review panel subgraph can be developed and tested independently, then composed into the main graph.

## What It Costs

1. **New dependency** — `langgraph` + `langchain-core`
2. **State serialization** — Everything in `GeCCoState` must be JSON-serializable (no raw dataframes, model objects, or compiled functions in state)
3. **Refactoring** — Each node must be a pure-ish function of `(state) -> partial state update`, which means extracting logic out of the `GeCCoModelSearch` class methods
4. **Learning curve** — Team needs to understand LangGraph's state reducer semantics (especially for the `review_reports` list merge)

---

## Migration Path

This is NOT an all-or-nothing rewrite. Incremental approach:

1. **Phase 1:** Build the review panel as described in the main plan (ThreadPoolExecutor, plugs into existing loop). This works today with no new dependencies.

2. **Phase 2:** Wrap the existing `run_n_shots()` logic in LangGraph nodes that call the same underlying functions. The graph becomes a thin orchestration layer over the existing code.

3. **Phase 3:** Gradually move state management into `GeCCoState`. Replace `self.best_model` tracking with state flow. Replace the `for it in range(...)` loop with the graph's `should_continue` → `sync_registry` cycle.

Phase 1 is the immediate plan. This document shows where Phase 2-3 would go when the complexity justifies it.
