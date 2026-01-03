import numpy as np, pandas as pd

def compute_stay_probabilities(choice_1, state, reward):
    choice_1 = np.asarray(choice_1)
    state = np.asarray(state)
    reward = np.asarray(reward)

    # stay from trial t -> t+1
    stay = (choice_1[1:] == choice_1[:-1])   # length T-1

    # common / rare transition on previous trial (t-1)
    common_transition = (
        ((choice_1 == 0) & (state == 0)) |
        ((choice_1 == 1) & (state == 1))
    )[:-1]                                   # length T-1

    rare_transition = ~common_transition     # complementary

    # reward on previous trial (t-1)
    rewarded = (reward[:-1] == 1)
    not_rewarded = ~rewarded                 # since 0/1

    def stay_prob(condition):
        return np.mean(stay[condition]) if np.any(condition) else np.nan

    prob_stay_common_rewarded     = stay_prob(common_transition & rewarded)
    prob_stay_common_not_rewarded = stay_prob(common_transition & not_rewarded)
    prob_stay_rare_rewarded       = stay_prob(rare_transition & rewarded)
    prob_stay_rare_not_rewarded   = stay_prob(rare_transition & not_rewarded)

    return [
        prob_stay_common_rewarded,
        prob_stay_rare_rewarded,
        prob_stay_common_not_rewarded,
        prob_stay_rare_not_rewarded,
    ]
