"""
Shared likelihood-validity gate for generated cognitive models.

Provides static (pre-fit) and post-fit validation to reject models whose
likelihood objective is constant, degenerate, leaky (non-additive), or
produces invalid numeric outputs.  Both train-evaluation and test-evaluation
paths use the same validator and raise the same ``InvalidLikelihoodError``.
"""

from __future__ import annotations

import ast
import dataclasses
import math
from typing import Any


class InvalidLikelihoodError(Exception):
    """Raised when a model's likelihood objective is structurally invalid.

    Attributes
    ----------
    message:
        Human-readable description of the failure.
    error_type:
        Machine-readable category, e.g. ``"constant_likelihood"``,
        ``"choice_leakage"``, ``"non_additive_nll"``,
        ``"degenerate_nll"``.
    details:
        Optional structured payload (snippet, line numbers, etc.).
    """

    def __init__(
        self,
        message: str,
        error_type: str = "invalid_likelihood",
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.error_type = error_type
        self.details = details or {}


# --------------------------------------------------------------------------- #
# Validation result helpers
# --------------------------------------------------------------------------- #


@dataclasses.dataclass
class ValidationResult:
    """Outcome of a likelihood-validation check."""

    passed: bool
    error_type: str | None = None
    error_message: str | None = None
    error_details: dict[str, Any] | None = None


def validation_error_result(
    error: InvalidLikelihoodError,
) -> dict[str, Any]:
    """Convert an ``InvalidLikelihoodError`` into a standard result dict.

    The returned dict is compatible with the ``metric_name='VALIDATION_ERROR'``
    contract used by ``CandidateEvaluator.fit_candidate_model`` and
    ``fit_one_on_test``.
    """
    return {
        "metric_name": "VALIDATION_ERROR",
        "metric_value": float("inf"),
        "error_type": error.error_type,
        "error_message": error.message,
        "error_details": error.details,
    }


def is_valid_likelihood_result(result: dict[str, Any]) -> bool:
    """Return True when a result dict represents a valid fitted model."""
    return result.get("metric_name") != "VALIDATION_ERROR" and result.get(
        "metric_value", float("inf")
    ) != float("inf")


# --------------------------------------------------------------------------- #
# Static validation (pre-fit)
# --------------------------------------------------------------------------- #


def _extract_function_body(code: str, func_name: str) -> str | None:
    """Return the source of the function *func_name* from *code*, or ``None``."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            lines = code.splitlines()
            start = node.lineno - 1 if node.lineno else 0
            end = node.end_lineno if node.end_lineno else len(lines)
            return "\n".join(lines[start:end])
    return None


def _has_return_zero(body: str) -> bool:
    """Check for ``return 0.0`` or equivalent constant-zero return."""
    try:
        tree = ast.parse(body)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Return) and node.value is not None:
            # Literal zero (0, 0.0)
            if isinstance(node.value, ast.Constant) and node.value.value == 0:
                return True
            if isinstance(node.value, ast.UnaryOp) and isinstance(
                node.value.op, ast.USub
            ):
                if (
                    isinstance(node.value.operand, ast.Constant)
                    and node.value.operand.value == 0
                ):
                    return True
        # Also check for `return 0` without decimal
    return False


def _is_probability_assignment(stmt: ast.AST) -> bool:
    """Return True if *stmt* assigns a probability/log-probability variable.

    Probability variables are those whose name indicates they hold a
    probability or log-probability value that feeds into the likelihood
    computation.  Assignments to these variables are allowed to read the
    observed choice column because selecting/computing the probability of
    the observed action is the first step of a valid likelihood calculation.
    """
    PROBABILITY_NAMES = {
        "prob", "probs",
        "log_prob", "log_probs", "logprob", "logprobs", "logp",
        "likelihood", "log_likelihood",
    }
    if isinstance(stmt, ast.Assign):
        for target in stmt.targets:
            name = None
            if isinstance(target, ast.Name):
                name = target.id.lower()
            elif isinstance(target, ast.Subscript) and isinstance(
                target.value, ast.Name
            ):
                name = target.value.id.lower()
            if name and (name in PROBABILITY_NAMES or name.startswith("prob_")):
                # Exclude log_lik assignments (handled separately)
                if "log_lik" in name:
                    return False
                return True
    return False


def _is_choice_indexed_assignment(stmt: ast.AST) -> bool:
    """Return True if an assignment's target indexes state by a choice variable.

    This catches patterns like ``Q[action[t]] = ...`` or
    ``kernel[action[t]] += ...`` where the observed choice drives which
    part of the latent state is updated *before* the likelihood computation.
    """
    if isinstance(stmt, ast.Assign):
        for target in stmt.targets:
            # Check if the target itself is a subscript whose slice
            # reads a choice column (e.g. Q[action[t]])
            if isinstance(target, ast.Subscript) and _contains_choice_read(target):
                return True
    if isinstance(stmt, ast.AugAssign):
        if isinstance(stmt.target, ast.Subscript) and _contains_choice_read(stmt.target):
            return True
    return False


def _is_choice_conditioned_control_flow(stmt: ast.AST) -> bool:
    """Return True if a control flow statement's condition reads a choice column.

    This catches patterns like ``if action[t] == 1: Q = ...`` where the
    observed choice determines which state-update branch executes.
    """
    if isinstance(stmt, (ast.If, ast.While)):
        return _contains_choice_read(stmt.test)
    return False


def _has_choice_leakage_before_likelihood(body: str) -> bool:
    """Check for choice-column access *before* a log-likelihood assignment.

    This detects patterns where the model uses the observed choice
    (``action[t]``, ``choice[t]``, etc.) inside the trial loop to update
    internal state *before* computing the log-likelihood of that choice,
    which creates a circular dependency and leaks the answer.

    Two specific leakage patterns are detected:

    1. **Choice-indexed assignment**: the observed choice drives which
       element of a state array is updated (e.g. ``Q[action[t]] = ...``,
       ``kernel[action[t]] += ...``).

    2. **Choice-conditioned control flow**: the observed choice determines
       which branch of an ``if``/``while`` is taken, leading to
       action-dependent state updates (e.g. ``if action[t] == 1: ...``).

    Legitimate reads of the observed choice that are part of a probability
    or log-probability computation (e.g. ``prob = probs[action[t]]``) are
    *not* flagged.
    """
    try:
        tree = ast.parse(body)
    except SyntaxError:
        return False

    # Find all for-loops over trials
    for node in ast.walk(tree):
        if not isinstance(node, ast.For):
            continue
        # Look inside the loop body
        loop_body = node.body
        likelihood_assigned = False
        choice_access_before = False

        for stmt in loop_body:
            # Check if this statement assigns log_likelihood
            if _is_log_likelihood_assignment(stmt):
                likelihood_assigned = True
                break
            # Probability assignment with choice read is valid
            if _is_probability_assignment(stmt):
                continue
            # Leakage: choice-indexed state update (e.g. Q[action[t]] = ...)
            if _is_choice_indexed_assignment(stmt):
                choice_access_before = True
                break
            # Leakage: control flow conditioned on choice (e.g. if action[t]==1)
            if _is_choice_conditioned_control_flow(stmt):
                choice_access_before = True
                break

        if choice_access_before:
            return True

    return False


def _is_log_likelihood_assignment(stmt: ast.AST) -> bool:
    """Return True if *stmt* assigns a variable containing 'log_lik'.

    Handles both plain assignment (``log_lik = ...``) and augmented
    assignment (``log_lik += ...``).
    """
    if isinstance(stmt, ast.Assign):
        for target in stmt.targets:
            if isinstance(target, ast.Name) and "log_lik" in target.id.lower():
                return True
            if isinstance(target, ast.Subscript):
                # e.g., log_lik[t] = ...
                if (
                    isinstance(target.value, ast.Name)
                    and "log_lik" in target.value.id.lower()
                ):
                    return True
    if isinstance(stmt, ast.AugAssign):
        if isinstance(stmt.target, ast.Name) and "log_lik" in stmt.target.id.lower():
            return True
        if isinstance(stmt.target, ast.Subscript):
            if (
                isinstance(stmt.target.value, ast.Name)
                and "log_lik" in stmt.target.value.id.lower()
            ):
                return True
    return False


_CHOICE_BASE_NAMES = frozenset({
    "action", "actions",
    "choice", "choices",
    "key_press", "key_presses",
})


def _is_choice_column_name(name: str) -> bool:
    """Return True if *name* refers to an observed-choice column.

    Matches exact base names (``action``, ``choice``, etc.) **and**
    suffixed variants (``action_1``, ``action_2``, ``choice_1``,
    ``key_press_left``, etc.) so that patterns like
    ``kernel[action_1[t]] += ...`` are detected as choice reads.
    """
    lower = name.lower()
    if lower in _CHOICE_BASE_NAMES:
        return True
    for base in _CHOICE_BASE_NAMES:
        if lower.startswith(base + "_"):
            return True
    return False


def _contains_choice_read(node: ast.AST) -> bool:
    """Return True if the AST node references a choice-column variable."""
    for child in ast.walk(node):
        if isinstance(child, ast.Subscript):
            # e.g., action[t], choice[t], action_1[t]
            if isinstance(child.value, ast.Name) and _is_choice_column_name(
                child.value.id
            ):
                return True
        if isinstance(child, ast.Name):
            # Could also be a direct reference to a choice array
            if _is_choice_column_name(child.id):
                return True
    return False


def _has_non_accumulated_nll(body: str) -> bool:
    """Check for ``log_lik = ...`` inside loop *without* accumulation.

    A valid model accumulates per-trial log-likelihood (e.g.
    ``log_lik += ...`` or ``neg_log_lik += ...``) and returns the
    sum.  If a fresh ``log_lik = ...`` assignment overwrites the
    variable each iteration, only the last trial's value is returned.
    """
    try:
        tree = ast.parse(body)
    except SyntaxError:
        return False

    for node in ast.walk(tree):
        if not isinstance(node, ast.For):
            continue
        for stmt in node.body:
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name) and target.id.lower() in {
                        "log_lik",
                        "log_likelihood",
                        "nll",
                        "neg_log_lik",
                        "neg_log_likelihood",
                    }:
                        # Plain assignment (=) inside loop = non-accumulated
                        if not isinstance(stmt, ast.AugAssign):
                            return True
    return False


def validate_likelihood_static(
    code: str,
    func_name: str = "cognitive_model",
) -> ValidationResult:
    """Run static (pre-fit) likelihood-validity checks on model *code*.

    Checks performed:
    1. Constant-zero return (``return 0.0``) → ``constant_likelihood``.
    2. Choice leakage before log-likelihood assignment → ``choice_leakage``.
    3. Non-accumulated per-trial log-likelihood → ``non_additive_nll``.

    Parameters
    ----------
    code:
        Source code of the model function.
    func_name:
        Name of the function to inspect.

    Returns
    -------
    ValidationResult
        ``passed=True`` when all checks pass.
    """
    body = _extract_function_body(code, func_name)
    if body is None:
        return ValidationResult(
            passed=False,
            error_type="parse_error",
            error_message=f"Could not extract function body for {func_name}",
            error_details={"func_name": func_name},
        )

    # Check 1: constant zero return
    if _has_return_zero(body):
        return ValidationResult(
            passed=False,
            error_type="constant_likelihood",
            error_message=(
                f"Model '{func_name}' returns a constant 0.0 — "
                "the negative log-likelihood is degenerate and non-informative."
            ),
            error_details={"func_name": func_name, "snippet": "return 0.0"},
        )

    # Check 2: choice leakage before likelihood assignment
    if _has_choice_leakage_before_likelihood(body):
        return ValidationResult(
            passed=False,
            error_type="choice_leakage",
            error_message=(
                f"Model '{func_name}' accesses the observed choice column "
                "before computing the log-likelihood, creating a circular "
                "dependency that leaks the correct answer."
            ),
            error_details={"func_name": func_name},
        )

    # Check 3: non-accumulated NLL
    if _has_non_accumulated_nll(body):
        return ValidationResult(
            passed=False,
            error_type="non_additive_nll",
            error_message=(
                f"Model '{func_name}' assigns log_lik = ... inside the trial "
                "loop without accumulation (e.g., use += instead of =). "
                "Only the last trial's value will be returned."
            ),
            error_details={"func_name": func_name},
        )

    return ValidationResult(passed=True)


# --------------------------------------------------------------------------- #
# Post-fit validation
# --------------------------------------------------------------------------- #


def validate_likelihood_post_fit(
    per_participant_nll: list[float] | None,
    mean_nll: float | None,
    *,
    func_name: str = "cognitive_model",
    n_participants: int = 0,
    participant_n_trials: list[int] | None = None,
) -> ValidationResult:
    """Validate likelihood numeric outputs after fitting.

    Checks performed:
    1. All-zero participant NLLs → ``degenerate_nll`` (constant zero).
    2. Any negative participant NLL → ``degenerate_nll`` (invalid negative).
    3. Implausibly tiny per-choice NLL → ``degenerate_nll``, computed on a
       per-choice basis using participant trial counts when available.

    Parameters
    ----------
    per_participant_nll:
        List of per-participant negative log-likelihood values.  May be
        ``None`` if the fit produced no participant-level data.
    mean_nll:
        Mean NLL across participants.  May be ``None``.
    func_name:
        Model name for error messages.
    n_participants:
        Number of participants expected (for context in error messages).
    participant_n_trials:
        List of trial counts per participant, used to compute per-choice
        NLL.  When ``None`` or shorter than *per_participant_nll*, the
        total NLL is used as a fallback estimate.

    Returns
    -------
    ValidationResult
        ``passed=True`` when all checks pass.
    """
    if not per_participant_nll and mean_nll is None:
        return ValidationResult(
            passed=False,
            error_type="degenerate_nll",
            error_message=(
                f"Model '{func_name}' produced no per-participant NLL data "
                "and no mean NLL."
            ),
            error_details={"func_name": func_name, "n_participants": n_participants},
        )

    if per_participant_nll:
        # Check 1: all-zero NLLs
        if all(v == 0.0 for v in per_participant_nll):
            return ValidationResult(
                passed=False,
                error_type="degenerate_nll",
                error_message=(
                    f"Model '{func_name}' returned zero NLL for all "
                    f"{len(per_participant_nll)} participants — the "
                    "likelihood is degenerate (constant zero)."
                ),
                error_details={
                    "func_name": func_name,
                    "per_participant_nll": per_participant_nll,
                },
            )

        # Check 2: any negative NLL
        negative = [(i, v) for i, v in enumerate(per_participant_nll) if v < 0.0]
        if negative:
            return ValidationResult(
                passed=False,
                error_type="degenerate_nll",
                error_message=(
                    f"Model '{func_name}' returned negative NLL for "
                    f"{len(negative)}/{len(per_participant_nll)} participants "
                    "(NLL cannot be negative)."
                ),
                error_details={
                    "func_name": func_name,
                    "negative_participants": negative[:10],
                    "per_participant_nll": per_participant_nll,
                },
            )

        # Check 3: implausibly tiny per-choice NLL
        # Use trial counts where available for per-choice computation;
        # fall back to total NLL / trial-guess heuristic when counts
        # are missing.
        per_choice_nll_values: list[float] = []
        for i, nll in enumerate(per_participant_nll):
            if participant_n_trials and i < len(participant_n_trials):
                n_trials = participant_n_trials[i]
                if n_trials and n_trials > 0:
                    per_choice_nll_values.append(nll / n_trials)
                else:
                    per_choice_nll_values.append(nll)
            else:
                per_choice_nll_values.append(nll)

        if per_choice_nll_values:
            min_per_choice = min(per_choice_nll_values)
            mean_per_choice = sum(per_choice_nll_values) / len(per_choice_nll_values)
            if mean_per_choice < 1e-8:
                return ValidationResult(
                    passed=False,
                    error_type="degenerate_nll",
                    error_message=(
                        f"Model '{func_name}' returned a mean per-choice NLL "
                        f"of {mean_per_choice:.2e} which is implausibly small "
                        "(< 1e-8). The likelihood is likely degenerate "
                        "(near-zero variance)."
                    ),
                    error_details={
                        "func_name": func_name,
                        "mean_nll": mean_nll,
                        "mean_per_choice_nll": mean_per_choice,
                        "min_per_choice_nll": min_per_choice,
                        "per_participant_nll": per_participant_nll,
                        "participant_n_trials": participant_n_trials,
                    },
                )

    return ValidationResult(passed=True)
