"""Cognitive Modeling Library v4.0

Automatically extracted primitives for two-step task RL models.
Patterns extracted:
- STAI modulation (anxiety scaling)
- Decay/forgetting mechanisms
- Model-based/model-free hybrid computations
- Softmax choice probabilities
- TD learning updates
- Perseveration/stickiness biases
"""

import numpy as np

def calc_expr_0(epsilon, p1_seq, p2_seq):
    return (np.sum(np.log((epsilon + p1_seq))) + np.sum(np.log((epsilon + p2_seq))))

def extract_stai_scalar(stai_arr):
    return float(stai_arr[0])

def calc_expr_2(beta, q_centered):
    return np.exp((beta * q_centered))

def max_q2_per_state(Q2):
    return np.max(Q2, axis=1)

def calc_expr_4(alpha, delta2):
    return (alpha * delta2)

def calc_expr_5(logits):
    return np.exp(logits)

def get_max_value(logits):
    return np.max(logits)

def calc_expr_7(q1, q1_mb, w_mb):
    return (((1.0 - w_mb) * q1) + (q1_mb * w_mb))

def calc_expr_8(exp_q):
    return (exp_q / np.sum(exp_q))

def calc_expr_9(q1, q1_mb, w_mb_eff):
    return (((1.0 - w_mb_eff) * q1) + (q1_mb * w_mb_eff))

def calc_expr_10(alpha, delta1):
    return (alpha * delta1)

def get_max_value_1(q1):
    return (q1 - np.max(q1))

def decay_complement(decay):
    return (1.0 - decay)

def get_max_value_2(q2_row):
    return (q2_row - np.max(q2_row))

def stai_scale_half(stai_arr):
    return (0.5 * stai_arr)

def init_choice_probs(a1_seq):
    num_trials = len(a1_seq)
    p1_seq = np.zeros(num_trials)
    p2_seq = np.zeros(num_trials)
    return num_trials, p1_seq, p2_seq

def compute_mb_values(Q2, T):
    max_q2 = max_q2_per_state(Q2)
    q1_mb = (T @ max_q2)
    return max_q2, q1_mb

def record_stage2(a2, p2_seq, probs2, r_seq, t):
    p2_seq[t] = probs2[a2]
    r = r_seq[t]
    return r

def td_update_q2(Q2, a2, alpha, r, s):
    delta2 = (r - Q2[(s, a2)])
    Q2[(s, a2)] += calc_expr_4(alpha, delta2)
    return delta2

def init_q_values():
    Q2 = np.zeros((2, 2))
    T = np.array([[0.7, 0.3], [0.3, 0.7]])
    return Q2, T

def init_trial_vars(a1_seq):
    (num_trials, p1_seq, p2_seq) = init_choice_probs(a1_seq)
    q1 = np.zeros(2)
    return num_trials, p1_seq, p2_seq, q1

def mechanism_21(p1_seq, p2_seq):
    epsilon = 1e-12
    nll = (- calc_expr_0(epsilon, p1_seq, p2_seq))
    return epsilon, nll

def record_stage1(a1, p1_seq, probs1, s_seq, t):
    p1_seq[t] = probs1[a1]
    s = s_seq[t]
    return s

def get_actions(a1_seq, a2_seq, t):
    a1 = int(a1_seq[t])
    a2 = int(a2_seq[t])
    return a1, a2

def td_update_q1(Q2, a1, a2, alpha, q1, s):
    target1 = Q2[(s, a2)]
    delta1 = (target1 - q1[a1])
    q1[a1] += calc_expr_10(alpha, delta1)
    return delta1, target1

def get_actions_1(a1_seq, a2_seq, t):
    a1 = a1_seq[t]
    a2 = a2_seq[t]
    return a1, a2

def init_q_values_1(Q2, T):
    bias = np.zeros(2)
    (max_q2, q1_mb) = compute_mb_values(Q2, T)
    return bias, max_q2, q1_mb

def init_choice_probs_1(num_trials):
    p1_seq = np.zeros(num_trials)
    p2_seq = np.zeros(num_trials)
    return p1_seq, p2_seq

def mechanism_28(Q2, a2, alpha, p2_seq, probs2, r_seq, s, t):
    r = record_stage2(a2, p2_seq, probs2, r_seq, t)
    delta2 = td_update_q2(Q2, a2, alpha, r, s)
    return delta2, r

def td_update_q1_1(Q2, a1, a2, alpha, q1, s):
    delta1 = (Q2[(s, a2)] - q1[a1])
    q1[a1] += calc_expr_10(alpha, delta1)
    return delta1

def mechanism_30(a1_seq):
    last_a1 = None
    (num_trials, p1_seq, p2_seq) = init_choice_probs(a1_seq)
    return last_a1, num_trials, p1_seq, p2_seq

def extract_stai(a1_seq, stai_arr):
    (num_trials, p1_seq, p2_seq, q1) = init_trial_vars(a1_seq)
    stai_arr = extract_stai_scalar(stai_arr)
    return num_trials, p1_seq, p2_seq, q1, stai_arr

def calc_expr_32(stai_arr):
    return (stai_scale_half(stai_arr) + 0.5)

def calc_expr_33(probs1):
    return (probs1 / np.sum(probs1))

def calc_expr_34(probs2):
    return (probs2 / np.sum(probs2))

def calc_expr_35(epsilon, exp_q):
    return (exp_q / (epsilon + np.sum(exp_q)))

