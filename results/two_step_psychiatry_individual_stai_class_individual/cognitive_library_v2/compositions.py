"""
Model Compositions
==================

Patterns of how primitives combine to form complete cognitive models.
These represent common "cognitive strategies" for the two-step task.
"""

from primitives import *

# Common cognitive strategy patterns:
#
# Pattern: ('helper::softmax', 'modulation::multiplicative', 'policy::perseveration_bonus', 'value_update::td_stage1', 'value_update::td_stage2')
# Used by: ['p19', 'p20', 'p21', 'p22', 'p26', 'p27', 'p29', 'p30', 'p31', 'p32', 'p33', 'p34', 'p36', 'p38', 'p39', 'p41']
#
# Pattern: ('helper::softmax', 'modulation::multiplicative', 'policy::perseveration_bonus', 'policy::win_stay_bonus', 'value_update::td_stage1', 'value_update::td_stage2')
# Used by: ['p28', 'p35', 'p40']
#
# Pattern: ('helper::compute_mb_values', 'helper::softmax', 'modulation::inverse_linear', 'policy::mb_mf_mixture', 'value_update::td_stage1', 'value_update::td_stage2')
# Used by: ['p37', 'p43']
#
# Pattern: ('helper::softmax', 'modulation::inverse_division', 'policy::perseveration_bonus', 'value_update::td_stage1', 'value_update::td_stage2')
# Used by: ['p18']
#
# Pattern: ('helper::softmax', 'modulation::multiplicative', 'value_update::asymmetric_td')
# Used by: ['p23']
#
# Pattern: ('helper::compute_mb_values', 'helper::softmax', 'modulation::inverse_linear', 'policy::mb_mf_mixture', 'policy::perseveration_bonus', 'value_update::td_stage1', 'value_update::td_stage2')
# Used by: ['p24']
#
# Pattern: ('helper::softmax', 'modulation::custom', 'policy::perseveration_bonus', 'value_update::td_stage1', 'value_update::td_stage2')
# Used by: ['p25']
#
# Pattern: ('decay::memory_decay', 'helper::softmax', 'modulation::multiplicative', 'value_update::td_stage1', 'value_update::td_stage2')
# Used by: ['p42']
#
# Pattern: ('decay::eligibility_trace', 'decay::memory_decay', 'helper::softmax', 'modulation::multiplicative', 'value_update::td_stage1', 'value_update::td_stage2')
# Used by: ['p44']
#

# ============================================================
# COMPOSITION TEMPLATES
# ============================================================

# --- INVERSE_DIVISION STAI MODULATION ---
# 1 participants

# --- MULTIPLICATIVE STAI MODULATION ---
# 22 participants

# --- INVERSE_LINEAR STAI MODULATION ---
# 3 participants

# --- CUSTOM STAI MODULATION ---
# 1 participants
