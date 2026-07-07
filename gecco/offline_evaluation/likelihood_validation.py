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


def _has_constant_numeric_return(body: str) -> bool:
    """Check for a return statement that always yields the same numeric constant.

    This catches ``return 0.0``, ``return 0``, ``return 0.5``, ``return -0.1``,
    and any other literal numeric return.  It intentionally does **not** catch
    ``return variable_name`` or ``return -log_lik``.
    """
    try:
        tree = ast.parse(body)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Return) and node.value is not None:
            # Literal constant (int or float)
            if isinstance(node.value, ast.Constant) and isinstance(
                node.value.value, (int, float)
            ):
                return True
            # Unary negation of a literal constant  e.g. return -0.5
            if isinstance(node.value, ast.UnaryOp) and isinstance(
                node.value.op, ast.USub
            ):
                if isinstance(node.value.operand, ast.Constant) and isinstance(
                    node.value.operand.value, (int, float)
                ):
                    return True
    return False


def _body_contains_loop(body: str) -> bool:
    """Return True when *body* contains at least one for-loop or while-loop."""
    try:
        tree = ast.parse(body)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, (ast.For, ast.While)):
            return True
    return False


_ACCUMULATION_NAMES = frozenset({
    "log_lik", "log_likelihood",
    "nll", "neg_log_lik", "neg_log_likelihood",
})

# Names that indicate likelihood-related computation when they appear
# in the RHS of an assignment to a likelihood variable.  This includes
# probability names (prob, log_prob, etc.) and function names (log).
_LIKELIHOOD_SHAPED_RHS_NAMES = frozenset({
    "prob", "probs",
    "log_prob", "log_probs", "logprob", "logprobs",
    "logp",
    "likelihood",
    "log",
})


def _is_likelihood_shaped_expr(node: ast.AST, target_name: str | None = None) -> bool:
    """Return True when *node* is a genuinely likelihood-shaped expression.

    An expression counts as likelihood-shaped if it:

    1. **Self-reference**: references the same likelihood variable on the
       RHS (e.g. ``nll = nll + ...``), indicating additive accumulation.
    2. **Log call**: contains a call to ``log``, ``np.log``, ``math.log``,
       etc.
    3. **Probability/likelihood reference**: references a variable whose
       name suggests it holds a probability, log-probability, or likelihood
       value (``prob``, ``log_prob``, ``likelihood``, etc.).

    Parameters
    ----------
    node:
        The RHS expression AST node to inspect.
    target_name:
        The name of the target variable (used for self-reference detection).
        When ``None``, self-reference is not checked.
    """
    if target_name:
        target_lower = target_name.lower()
    else:
        target_lower = None

    for child in ast.walk(node):
        # Self-reference — the RHS reads the same accumulation variable
        if target_lower and isinstance(child, ast.Name):
            if child.id.lower() == target_lower:
                return True

        # Call to a log function: log(...), np.log(...), math.log(...)
        if isinstance(child, ast.Call):
            func = child.func
            if isinstance(func, ast.Attribute) and func.attr == "log":
                return True
            if isinstance(func, ast.Name) and func.id == "log":
                return True

        # Reference to a probability / likelihood variable name
        if isinstance(child, ast.Name):
            name_lower = child.id.lower()
            if target_lower and name_lower == target_lower:
                continue  # already counted as self-reference above
            if name_lower in _LIKELIHOOD_SHAPED_RHS_NAMES:
                return True

    return False


def _has_likelihood_accumulation(body: str) -> bool:
    """Return True when *body* contains evidence of likelihood accumulation.

    Accepted evidence:
      - Augmented assignment (``+=``) to a likelihood variable
        (``log_lik``, ``nll``, ``neg_log_lik``, etc.).
      - Plain assignment to a likelihood variable whose RHS is genuinely
        likelihood-shaped — involves log/probability/likelihood terms or
        is self-referential/additive (e.g. ``log_lik = log_lik + ...``,
        ``nll = math.log(prob)``).  Plain parameter-only assignments
        like ``nll = alpha + beta`` do **not** count.
      - A call to ``np.sum`` or ``numpy.sum`` whose argument contains
        ``log`` (e.g. ``np.sum(np.log(probs))``).

    Plain initialization assignments like ``nll = 0.0`` or
    ``log_lik = 0.0`` are **not** counted as accumulation evidence;
    they are merely variable initialisation.
    """
    try:
        tree = ast.parse(body)
    except SyntaxError:
        return False

    for node in ast.walk(tree):
        # Augmented assignment of a likelihood variable
        if isinstance(node, ast.AugAssign):
            if isinstance(node.target, ast.Name):
                if node.target.id.lower() in _ACCUMULATION_NAMES:
                    return True
            if isinstance(node.target, ast.Subscript):
                if (
                    isinstance(node.target.value, ast.Name)
                    and node.target.value.id.lower() in _ACCUMULATION_NAMES
                ):
                    return True
        # Plain assignment to a likelihood variable
        # (catches cases like ``log_lik = log_lik + ...`` which is
        #  semantically additive but syntactically a plain assignment).
        # Skip initialisation to a literal constant (e.g. nll = 0.0).
        # Tightened: only count as evidence when the RHS is genuinely
        # likelihood-shaped (log/probability terms or self-referential).
        # Plain parameter-only assignments like ``nll = alpha + beta``
        # are rejected.
        if isinstance(node, ast.Assign):
            _rhs_is_literal = (
                isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, (int, float))
            )
            if _rhs_is_literal:
                continue
            for target in node.targets:
                if isinstance(target, ast.Name):
                    if target.id.lower() in _ACCUMULATION_NAMES:
                        if _is_likelihood_shaped_expr(node.value, target_name=target.id):
                            return True
                if isinstance(target, ast.Subscript):
                    if (
                        isinstance(target.value, ast.Name)
                        and target.value.id.lower() in _ACCUMULATION_NAMES
                    ):
                        if _is_likelihood_shaped_expr(node.value, target_name=target.value.id):
                            return True

    # Check for np.sum(np.log(...)) pattern
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            # np.sum(...) or numpy.sum(...)
            func = node.func
            is_np_sum = (
                (isinstance(func, ast.Attribute)
                 and func.attr == "sum"
                 and isinstance(func.value, ast.Name)
                 and func.value.id in ("np", "numpy"))
            )
            if is_np_sum and node.args:
                inner = node.args[0]
                # Check if the argument to np.sum contains a call to log
                if isinstance(inner, ast.Call):
                    inner_func = inner.func
                    if (isinstance(inner_func, ast.Attribute)
                            and inner_func.attr == "log"
                            and isinstance(inner_func.value, ast.Name)
                            and inner_func.value.id in ("np", "numpy")):
                        return True
                    if (isinstance(inner_func, ast.Name)
                            and inner_func.id == "log"):
                        return True
                # Also check if the argument contains numpy.log or math.log
                # via any nested call
                for child in ast.walk(inner):
                    if isinstance(child, ast.Call):
                        child_func = child.func
                        if (isinstance(child_func, ast.Attribute)
                                and child_func.attr == "log"):
                            return True

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
    1. Constant numeric return (``return 0.0``, ``return 0.5``, etc.)
       → ``constant_likelihood``.
    2. No likelihood-objective evidence (no accumulation of ``log_lik``,
       ``nll``, etc., and no ``np.sum(np.log(...))`` pattern).
    3. Choice leakage before log-likelihood assignment → ``choice_leakage``.
    4. Non-accumulated per-trial log-likelihood → ``non_additive_nll``.

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

    # Check 1: constant numeric return (any literal int/float, not just 0)
    if _has_constant_numeric_return(body):
        return ValidationResult(
            passed=False,
            error_type="constant_likelihood",
            error_message=(
                f"Model '{func_name}' returns a constant numeric value — "
                "the negative log-likelihood is degenerate and non-informative."
            ),
            error_details={"func_name": func_name, "snippet": "constant numeric return"},
        )

    # Check 2: likelihood-objective evidence.
    # Reject any model (with or without a loop) that lacks likelihood
    # accumulation or vectorised-sum evidence.  Plain initialisation
    # assignments (nll = 0.0, log_lik = 0.0) do not count as evidence.
    if not _has_likelihood_accumulation(body):
        return ValidationResult(
            passed=False,
            error_type="constant_likelihood",
            error_message=(
                f"Model '{func_name}' has no likelihood accumulation "
                "evidence (no log_lik += ..., nll += ..., "
                "neg_log_lik += ..., or np.sum(np.log(...)) pattern). "
                "The model cannot produce an informative likelihood."
            ),
            error_details={"func_name": func_name, "snippet": "no likelihood accumulation evidence"},
        )

    # Check 3: choice leakage before likelihood assignment
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

    # Check 4: non-accumulated NLL
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
    min_per_choice_nll: float = 0.01,
) -> ValidationResult:
    """Validate likelihood numeric outputs after fitting.

    Checks performed:
    1. All-zero participant NLLs → ``degenerate_nll`` (constant zero).
    2. Any negative participant NLL → ``degenerate_nll`` (invalid negative).
    3. Implausibly tiny per-choice NLL → ``degenerate_nll``, computed on a
       per-choice basis using participant trial counts when available.
       The per-choice threshold is configurable via *min_per_choice_nll*;
       set to ``0.0`` or ``None`` to disable the tiny-NLL check.

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
    min_per_choice_nll:
        Minimum acceptable mean per-choice NLL.  Mean per-choice NLL values
        below this threshold are rejected as degenerate.  Set to ``0.0`` or
        ``None`` to disable this check.  Default ``0.01``.

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

        if per_choice_nll_values and min_per_choice_nll and min_per_choice_nll > 0:
            min_per_choice = min(per_choice_nll_values)
            mean_per_choice = sum(per_choice_nll_values) / len(per_choice_nll_values)
            if mean_per_choice < min_per_choice_nll:
                return ValidationResult(
                    passed=False,
                    error_type="degenerate_nll",
                    error_message=(
                        f"Model '{func_name}' returned a mean per-choice NLL "
                        f"of {mean_per_choice:.4f} which is below the minimum "
                        f"threshold ({min_per_choice_nll}). The likelihood is "
                        "likely degenerate (near-zero variance)."
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
