"""
Cognitive Library - Class-Based Model Primitives
==================================================

Extracted from 27 participant-specific cognitive models.
Each primitive represents a reusable cognitive mechanism.

MECHANISM CATEGORIES:
  - model_based: Forward planning using transition matrix
  - perseveration: Tendency to repeat previous actions
  - win_stay: Reward-dependent action repetition
  - asymmetric_learning: Different learning rates for pos/neg outcomes
  - memory_decay: Forgetting of unchosen options

STAI MODULATION PATTERNS:
  - multiplicative: param * stai
  - inverse_linear: param * (1 - stai)
  - additive_boost: param * (1 + stai)
  - inverse_division: param / (1 + stai)
"""

import numpy as np
from typing import Tuple, Optional, Callable


# ============================================================
# Core Helper Functions
# ============================================================

def softmax(values: np.ndarray, beta: float) -> np.ndarray:
    """Numerically stable softmax."""
    centered = values - np.max(values)
    exp_vals = np.exp(beta * centered)
    return exp_vals / np.sum(exp_vals)


def compute_mb_values(q_stage2: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Compute model-based values: Q_MB = T @ max(Q2)."""
    max_q2 = np.max(q_stage2, axis=1)
    return T @ max_q2


# ============================================================
# STAI Modulation Functions
# ============================================================

def stai_multiplicative(param: float, stai: float) -> float:
    """Scale parameter by STAI: higher anxiety = larger effect."""
    return param * stai


def stai_inverse_linear(param: float, stai: float) -> float:
    """Scale parameter inversely: higher anxiety = smaller effect."""
    return param * (1.0 - stai)


def stai_additive_boost(param: float, stai: float) -> float:
    """Boost parameter by STAI: higher anxiety = larger effect."""
    return param * (1.0 + stai)


def stai_inverse_division(param: float, stai: float) -> float:
    """Divide by (1 + STAI): higher anxiety = smaller effect."""
    return param / (1.0 + stai)


# ============================================================
# Policy Stage 1 Primitives
# ============================================================

def policy_stage1_perseveration_multiplicative(
    q_stage1: np.ndarray,
    beta: float,
    bonus: float,
    stai: float,
    last_action: Optional[int]
) -> np.ndarray:
    """
    Policy Stage 1: perseveration
    STAI modulation: multiplicative
    Used by: p20, p21, p22, p27, p29... (11 total)
    """
    q_modified = q_stage1.copy()
    
    if last_action is not None:
        effective_bonus = stai_multiplicative(bonus, stai)
        q_modified[last_action] += effective_bonus
    
    return softmax(q_modified, beta)


def policy_stage1_perseveration_stai_first(
    q_stage1: np.ndarray,
    beta: float,
    bonus: float,
    stai: float,
    last_action: Optional[int]
) -> np.ndarray:
    """
    Policy Stage 1: perseveration
    STAI modulation: stai_first
    Used by: p26, p36, p40, p41, p44 (5 total)
    """
    q_modified = q_stage1.copy()
    
    if last_action is not None:
        effective_bonus = bonus
        q_modified[last_action] += effective_bonus
    
    return softmax(q_modified, beta)


def policy_stage1_perseveration(
    q_stage1: np.ndarray,
    beta: float,
    bonus: float,
    stai: float,
    last_action: Optional[int]
) -> np.ndarray:
    """
    Policy Stage 1: perseveration
    Used by: p18, p19, p38 (3 total)
    """
    q_modified = q_stage1.copy()
    
    if last_action is not None:
        effective_bonus = bonus
        q_modified[last_action] += effective_bonus
    
    return softmax(q_modified, beta)


def policy_stage1_model_based_perseveration_inverse_linear(
    q_stage1: np.ndarray,
    q_stage2: np.ndarray,
    T: np.ndarray,
    beta: float,
    w: float,
    pers: float,
    stai: float,
    last_action: Optional[int]
) -> np.ndarray:
    """
    Policy Stage 1: model_based, perseveration
    STAI modulation: inverse_linear
    Used by: p24 (1 total)
    """
    # Compute model-based values
    q_mb = compute_mb_values(q_stage2, T)
    
    # Anxiety reduces model-based control
    w_eff = w * (1.0 - stai)
    
    # Mix MF and MB
    q_net = (1.0 - w_eff) * q_stage1 + w_eff * q_mb
    
    # Add perseveration
    if last_action is not None:
        q_net = q_net.copy()
        q_net[last_action] += pers
    
    return softmax(q_net, beta)


def policy_stage1_perseveration_additive_boost(
    q_stage1: np.ndarray,
    beta: float,
    bonus: float,
    stai: float,
    last_action: Optional[int]
) -> np.ndarray:
    """
    Policy Stage 1: perseveration
    STAI modulation: additive_boost
    Used by: p25 (1 total)
    """
    q_modified = q_stage1.copy()
    
    if last_action is not None:
        effective_bonus = stai_additive_boost(bonus, stai)
        q_modified[last_action] += effective_bonus
    
    return softmax(q_modified, beta)


def policy_stage1_perseveration_win_stay_stai_first(
    q_stage1: np.ndarray,
    beta: float,
    bonus: float,
    stai: float,
    last_action: Optional[int],
    last_reward: Optional[float]
) -> np.ndarray:
    """
    Policy Stage 1: perseveration, win_stay
    STAI modulation: stai_first
    Used by: p28 (1 total)
    """
    q_modified = q_stage1.copy()
    
    if last_reward == 1.0 and last_action is not None:
        effective_bonus = stai * bonus  # Anxiety amplifies win-stay
        q_modified[last_action] += effective_bonus
    
    return softmax(q_modified, beta)


def policy_stage1_perseveration_win_stay_multiplicative(
    q_stage1: np.ndarray,
    beta: float,
    bonus: float,
    stai: float,
    last_action: Optional[int],
    last_reward: Optional[float]
) -> np.ndarray:
    """
    Policy Stage 1: perseveration, win_stay
    STAI modulation: multiplicative
    Used by: p35 (1 total)
    """
    q_modified = q_stage1.copy()
    
    if last_reward == 1.0 and last_action is not None:
        effective_bonus = stai * bonus  # Anxiety amplifies win-stay
        q_modified[last_action] += effective_bonus
    
    return softmax(q_modified, beta)


def policy_stage1_model_based_inverse_linear(
    q_stage1: np.ndarray,
    q_stage2: np.ndarray,
    T: np.ndarray,
    beta: float,
    w: float,
    stai: float
) -> np.ndarray:
    """
    Policy Stage 1: model_based
    STAI modulation: inverse_linear
    Used by: p37 (1 total)
    """
    # Compute model-based values
    q_mb = compute_mb_values(q_stage2, T)
    
    # Anxiety reduces model-based control
    w_eff = w * (1.0 - stai)
    
    # Mix MF and MB
    q_net = (1.0 - w_eff) * q_stage1 + w_eff * q_mb
    
    return softmax(q_net, beta)


def policy_stage1_model_based(
    q_stage1: np.ndarray,
    q_stage2: np.ndarray,
    T: np.ndarray,
    beta: float,
    w: float,
    stai: float
) -> np.ndarray:
    """
    Policy Stage 1: model_based
    Used by: p43 (1 total)
    """
    # Compute model-based values
    q_mb = compute_mb_values(q_stage2, T)
    
    # Anxiety reduces model-based control
    w_eff = w * (1.0 - stai)
    
    # Mix MF and MB
    q_net = (1.0 - w_eff) * q_stage1 + w_eff * q_mb
    
    return softmax(q_net, beta)



# ============================================================
# Value Update Primitives
# ============================================================

def value_update_standard(
    q_stage1: np.ndarray,
    q_stage2: np.ndarray,
    alpha: float,
    action_1: int,
    state: int,
    action_2: int,
    reward: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Value Update: standard TD
    Used by: p18, p20, p25, p33, p34... (10 total)
    """
    # Stage 2 TD update
    delta_2 = reward - q_stage2[state, action_2]
    q_stage2 = q_stage2.copy()
    q_stage2[state, action_2] += alpha * delta_2
    
    # Stage 1 TD update
    delta_1 = q_stage2[state, action_2] - q_stage1[action_1]
    q_stage1 = q_stage1.copy()
    q_stage1[action_1] += alpha * delta_1
    
    return q_stage1, q_stage2


def value_update_asymmetric_learning(
    q_stage1: np.ndarray,
    q_stage2: np.ndarray,
    alpha_pos: float,
    alpha_neg: float,
    action_1: int,
    state: int,
    action_2: int,
    reward: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Value Update: asymmetric_learning
    Used by: p23 (1 total)
    """
    # Stage 2 TD update with asymmetric learning
    delta_2 = reward - q_stage2[state, action_2]
    lr_2 = alpha_pos if delta_2 >= 0 else alpha_neg
    
    q_stage2 = q_stage2.copy()
    q_stage2[state, action_2] += lr_2 * delta_2
    
    # Stage 1 TD update
    delta_1 = q_stage2[state, action_2] - q_stage1[action_1]
    lr_1 = alpha_pos if delta_1 >= 0 else alpha_neg
    
    q_stage1 = q_stage1.copy()
    q_stage1[action_1] += lr_1 * delta_1
    
    return q_stage1, q_stage2


def value_update_model_based(
    q_stage1: np.ndarray,
    q_stage2: np.ndarray,
    alpha: float,
    action_1: int,
    state: int,
    action_2: int,
    reward: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Value Update: model_based
    Used by: p29 (1 total)
    """
    # Stage 2 TD update
    delta_2 = reward - q_stage2[state, action_2]
    q_stage2 = q_stage2.copy()
    q_stage2[state, action_2] += alpha * delta_2
    
    # Stage 1 TD update
    delta_1 = q_stage2[state, action_2] - q_stage1[action_1]
    q_stage1 = q_stage1.copy()
    q_stage1[action_1] += alpha * delta_1
    
    return q_stage1, q_stage2



# ============================================================
# Init Model Primitives
# ============================================================

def init_model_perseveration_multiplicative(param: float, stai: float) -> float:
    """Initialize model with STAI modulation: multiplicative
    Used by: p19, p38 (2 total)
    """
    return param * stai


def init_model_perseveration_additive_boost(param: float, stai: float) -> float:
    """Initialize model with STAI modulation: additive_boost
    Used by: p18 (1 total)
    """
    return param


def init_model_asymmetric_learning_stai_first(param: float, stai: float) -> float:
    """Initialize model with STAI modulation: stai_first
    Used by: p23 (1 total)
    """
    return param


def init_model_model_based(param: float, stai: float) -> float:
    """Initialize model with STAI modulation: none
    Used by: p37 (1 total)
    """
    return param


def init_model_model_based_inverse_linear(param: float, stai: float) -> float:
    """Initialize model with STAI modulation: inverse_linear
    Used by: p43 (1 total)
    """
    return param * (1.0 - stai)


def init_model_perseveration(param: float, stai: float) -> float:
    """Initialize model with STAI modulation: none
    Used by: p44 (1 total)
    """
    return param



# ============================================================
# Post-Trial Primitives
# ============================================================

def post_trial_memory_decay(
    q_stage1: np.ndarray,
    q_stage2: np.ndarray,
    decay_rate: float,
    action_1: int,
    state: int,
    action_2: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Post-trial processing: memory_decay
    Used by: p42 (1 total)
    """
    q_stage1 = q_stage1.copy()
    q_stage2 = q_stage2.copy()
    
    # Decay unchosen options
    unchosen_1 = 1 - action_1
    q_stage1[unchosen_1] *= (1.0 - decay_rate)
    
    unchosen_2 = 1 - action_2
    q_stage2[state, unchosen_2] *= (1.0 - decay_rate)
    
    # Decay unvisited state
    unvisited_state = 1 - state
    q_stage2[unvisited_state, :] *= (1.0 - decay_rate)
    
    return q_stage1, q_stage2


def post_trial_eligibility_trace_memory_decay_perseveration(
    q_stage1: np.ndarray,
    q_stage2: np.ndarray,
    decay_rate: float,
    action_1: int,
    state: int,
    action_2: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Post-trial processing: eligibility_trace, memory_decay, perseveration
    Used by: p44 (1 total)
    """
    q_stage1 = q_stage1.copy()
    q_stage2 = q_stage2.copy()
    
    # Decay unchosen options
    unchosen_1 = 1 - action_1
    q_stage1[unchosen_1] *= (1.0 - decay_rate)
    
    unchosen_2 = 1 - action_2
    q_stage2[state, unchosen_2] *= (1.0 - decay_rate)
    
    # Decay unvisited state
    unvisited_state = 1 - state
    q_stage2[unvisited_state, :] *= (1.0 - decay_rate)
    
    return q_stage1, q_stage2

