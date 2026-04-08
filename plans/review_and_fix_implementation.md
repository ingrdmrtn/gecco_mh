# Implementation Plan: Review-and-Fix for Cognitive Model Generation

## Overview

Replace the current "self-critique reflection" with a structured **review-and-fix** cycle:

```
Current:  Generate → Reflect (returns fixed code) → Fit
Proposed: Generate → Review (returns comments) → Fix (returns revised code) → Fit
```

## Architecture

### Current Flow
```
Generate models (JSON) 
  → LLM reviews models and returns FIXED CODE directly
  → Fit revised models
```

### Proposed Flow
```
Generate models (JSON)
  → Reviewer LLM reviews models and returns STRUCTURED COMMENTS
  → SAME generating agent sees comments and attempts to fix
  → Fit revised models
```

**Key Design Decision**: The generating agent (not a separate model) sees review comments and attempts fixes. This preserves the agent's understanding of its own intent.

---

## Files to Modify

| File | Change |
|------|--------|
| `config/two_step_factors_distributed.yaml` | Add `reviewer` config section |
| `gecco/structured_output.py` | Add `build_review_prompt()`, `build_fix_prompt()`, `parse_review_response()`, `get_review_schema()` |
| `gecco/run_gecco.py` | Modify `generate_models()` to use review-and-fix, add `_save_review()` method |
| `config/schema.py` | Add `ReviewerConfig` dataclass |

---

## Implementation Details

### 1. Configuration: `config/two_step_factors_distributed.yaml`

Add a new `reviewer` section under `llm`:

```yaml
llm:
  # ... existing config ...
  
  reviewer:
    enabled: true  # Set to false to disable review-and-fix
    
    persona: |
      You are an expert code reviewer for computational cognitive models.
      Your PhD is in computational neuroscience with 10+ years of experience
      debugging reinforcement learning models, likelihood functions, and
      hierarchical Bayesian models. You identify bugs BEFORE code execution.
    
    focus_areas:
      - numerical_stability
      - rl_correctness  
      - initialization
      - parameter_bounds
      - choice_indexing
      - likelihood_correctness
    
    issue_types:
      critical: "Will cause runtime errors or NaN values"
      warning: "May cause numerical issues in edge cases"
      info: "Minor improvement or style suggestion"
```

---

### 2. `gecco/structured_output.py`

#### 2.1 Add Review Schema

```python
def get_review_schema() -> dict:
    """JSON schema for structured review output."""
    return {
        "type": "object",
        "properties": {
            "reviews": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "model_name": {"type": "string"},
                        "overall_assessment": {
                            "type": "string",
                            "enum": ["passes", "minor_issues", "major_issues", "critical_failure"]
                        },
                        "issues": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "type": {
                                        "type": "string",
                                        "enum": [
                                            "numerical_stability",
                                            "rl_correctness",
                                            "uninitialized_variable",
                                            "parameter_bounds_mismatch",
                                            "indexing_error",
                                            "missing_initialization",
                                            "incorrect_update",
                                            "log_likelihood_error",
                                        ]
                                    },
                                    "location": {"type": "string"},
                                    "severity": {"type": "string", "enum": ["critical", "warning", "info"]},
                                    "description": {"type": "string"},
                                    "suggested_fix": {"type": "string"}
                                },
                                "required": ["type", "severity", "description"]
                            }
                        }
                    },
                    "required": ["model_name", "overall_assessment", "issues"]
                }
            }
        },
        "required": ["reviews"]
    }
```

#### 2.2 Add `build_review_prompt()`

```python
def build_review_prompt(models: List[Dict], guardrails: list = None,
                        persona: str = None, focus_areas: list = None) -> str:
    """
    Build a structured code review prompt for cognitive models.
    
    Returns review comments (NOT corrected code) in JSON format.
    """
    # ... implementation ...
```

#### 2.3 Add `build_fix_prompt()`

```python
def build_fix_prompt(original_models: List[Dict], review: Dict,
                     guardrails: list = None) -> str:
    """
    Build a prompt asking the generating agent to fix models based on review.
    
    Returns corrected models in the same JSON format as generation.
    """
    # ... implementation ...
```

#### 2.4 Add `parse_review_response()`

```python
def parse_review_response(text: str) -> Dict:
    """Parse structured review JSON from LLM response."""
    # ... implementation ...
```

---

### 3. `gecco/run_gecco.py`

#### 3.1 Modify `generate_models()` (lines 284-358)

Replace the current reflection block with review-and-fix:

```python
def generate_models(self, prompt):
    """
    Generate cognitive models with structured output, parsing,
    and optional review-and-fix cycle.
    """
    # ... initial generation ...
    
    # --- Review-and-Fix cycle ---
    reviewer_config = getattr(self.cfg.llm, "reviewer", {})
    if reviewer_config.get("enabled", False) and models:
        # Review phase
        # Fix phase (if issues found)
        # ...
    
    return raw_text, models
```

#### 3.2 Add `_save_review()` method

```python
def _save_review(self, review: Dict):
    """Save review comments to disk for debugging."""
    # ... implementation ...
```

---

### 4. Config Validation: `config/schema.py`

Add `ReviewerConfig` dataclass:

```python
@dataclass
class ReviewerConfig:
    enabled: bool = True
    persona: Optional[str] = None
    focus_areas: Optional[List[str]] = None
    issue_types: Optional[Dict[str, str]] = None

@dataclass  
class LLMConfig:
    # ... existing fields ...
    reviewer: Optional[ReviewerConfig] = None
```

---

## Focus Areas for Cognitive Models

The review checks domain-specific issues:

### numerical_stability
- Softmax overflow: exp(beta * Q) can overflow for beta > 5 with Q > 2
- Log of zero/negative: adding epsilon inside log, not outside
- Division by zero: probability denominators must not be zero
- NaN propagation: initialize arrays to 0.0, not empty

### rl_correctness
- Learning rate updates: Q += alpha * delta, not Q = alpha * delta
- TD error sign: reward - Q[current] (not Q[current] - reward)
- Eligibility traces: lambda * alpha * delta_2 updates Q1, not just Q2
- Model-based values: transition_matrix @ max_Q_stage2 computes MB values

### initialization
- All Q-values must be initialized before the trial loop
- Stage-2 Q-values need shape (n_states, n_actions)
- Perseveration/eligibility variables reset between trials

### parameter_bounds
- Inverse temperature (beta): typically [0, 10] or [0, 30], not [0, 1]
- Learning rates: [0, 1]
- MB/MF weights: [0, 1]
- Docstrings must match actual bounds in code

### choice_indexing
- p_choice[t] = probs[action[t]] (index by OBSERVED action)
- State indexing: use current trial's state, not state[t+1]
- Array shapes: Q[stage, action] or Q[stage][action] consistently

### likelihood_correctness
- Return NEGATIVE log-likelihood (minimization objective)
- Sum over ALL choice probabilities (stage 1 AND stage 2)
- log(p_choice + eps) inside sum, not outside

---

## Key Design Points

1. **Same LLM generates fixes**: The generating agent (not a separate model) sees review comments and attempts fixes. This preserves the agent's understanding of its own intent.

2. **Structured review format**: JSON schema ensures parseable, actionable comments rather than verbose prose.

3. **Task-specific focus**: Different cognitive tasks can have different `focus_areas` in config (e.g., Bayesian tasks focus on prior specification, RL tasks focus on update rules).

4. **Reviews persist**: Saved to `results/<task>/reviews/` for debugging why certain fixes were attempted.

5. **Graceful degradation**: If review or fix parsing fails, fall back to original models.

---

## Testing

After implementation:

1. **Test review generation**: Run with `reviewer.enabled: true` and check `results/<task>/reviews/` for JSON files
2. **Test fix application**: Verify models are corrected when issues found
3. **Test graceful degradation**: Set invalid review schema, verify fallback to original models
4. **Test disable flag**: Set `reviewer.enabled: false`, verify no review happens

---

## Rollback

If issues arise:

1. Set `reviewer.enabled: false` in config
2. The code falls back to the original (no review) behavior
3. No structural changes needed elsewhere

---

## Future Enhancements

1. **Per-task focus areas**: Different tasks load different `focus_areas` from config
2. **Review effectiveness logging**: Track how often fixes improve BIC vs. make it worse
3. **Multi-pass review**: Iteratively review until all critical issues resolved
4. **Integration with Pydantic validation**: Use Pydantic for structural checks, reviewer for semantic checks