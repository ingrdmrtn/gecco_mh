"""
Cognitive Library Learner v4.0
- Deep extraction of cognitive modeling primitives
- Multi-pass pattern recognition for maximum compression
- Semantic naming based on cognitive function
- Handles: STAI modulation, softmax, decay, MB/MF hybrid, TD updates, etc.
"""

import pandas as pd
import ast
import astunparse
import os
from collections import Counter, defaultdict
import re
import copy
import numpy as np

# Configuration
INPUT_FILE = '/home/aj9225/gecco-1/results/two_step_task_individual_stai/combined_models_bics.csv'
OUTPUT_DIR = '/home/aj9225/gecco-1/results/two_step_task_individual_stai/library_learned'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_best_models(csv_path):
    df = pd.read_csv(csv_path)
    df['participant'] = df['filename'].apply(lambda x: int(x.split('participant')[1].split('.')[0]))
    best_models = {}
    for p in df['participant'].unique():
        p_df = df[df['participant'] == p]
        best_row = p_df.loc[p_df['bic'].idxmin()]
        best_models[p] = best_row['code']
    return best_models

# =============================================================================
# COGNITIVE PATTERN TEMPLATES
# These are the core cognitive mechanisms we want to extract
# =============================================================================

COGNITIVE_TEMPLATES = {
    # Softmax choice probability
    'softmax_choice': {
        'pattern': r'(\w+)\s*=\s*\(\s*(\w+)\s*-\s*np\.max\(\2\)\s*\)\s*\n\s*(\w+)\s*=\s*np\.exp\(.*\*\s*\1\)\s*\n\s*\3\s*(?:=|/=).*np\.sum\(\3\)',
        'name': 'softmax_choice',
        'description': 'Convert Q-values to choice probabilities via softmax'
    },
    # Model-based Q1 update
    'mb_q1_update': {
        'pattern': r'max_q2\s*=\s*np\.max\(.*axis=1\).*\n.*q1_mb\s*=.*@.*max_q2',
        'name': 'compute_mb_values',
        'description': 'Compute model-based Q1 values from transition matrix and Q2'
    },
    # MF-MB hybrid
    'hybrid_q1': {
        'pattern': r'q1\s*=\s*\(\s*\(\s*\(\s*1\.0?\s*-\s*(\w+)\s*\)\s*\*\s*q1\s*\)\s*\+\s*\(\s*q1_mb\s*\*\s*\1\s*\)\s*\)',
        'name': 'hybrid_mf_mb',
        'description': 'Combine model-free and model-based Q1 values'
    },
    # TD update
    'td_update': {
        'pattern': r'(\w+)\s*=\s*\(\s*(\w+)\s*-\s*(\w+)\[.*\]\s*\)\s*\n.*\3\[.*\]\s*\+=\s*\(\s*\w+\s*\*\s*\1\s*\)',
        'name': 'td_update',
        'description': 'Temporal difference learning update'
    },
    # Decay/forgetting
    'decay_forget': {
        'pattern': r'(\w+)\s*=\s*\(\s*\(\s*\(\s*1\.0?\s*-\s*(\w+)\s*\)\s*\*\s*\1\s*\)\s*\+\s*\(\s*0\.5\s*\*\s*\2\s*\)\s*\)',
        'name': 'decay_to_prior',
        'description': 'Decay Q-values towards prior (0.5)'
    },
    # Perseveration bias
    'perseveration': {
        'pattern': r'if.*last_a1.*\n.*bias.*\[last_a1\].*\+=',
        'name': 'add_perseveration',
        'description': 'Add perseveration bonus for previous action'
    },
    # Asymmetric learning rates
    'asymmetric_lr': {
        'pattern': r'(\w+)\s*=\s*\(\s*(\w+)\s*if\s*\(\s*(\w+)\s*>=\s*0.*\)\s*else\s*(\w+)\s*\)',
        'name': 'select_learning_rate',
        'description': 'Select learning rate based on prediction error sign'
    },
    # NLL calculation
    'nll_calc': {
        'pattern': r'nll\s*=\s*\(\s*-\s*\(\s*np\.sum\(\s*np\.log\(.*p1.*\)\s*\)\s*\+\s*np\.sum\(\s*np\.log\(.*p2.*\)\s*\)\s*\)\s*\)',
        'name': 'compute_nll',
        'description': 'Compute negative log-likelihood from choice probabilities'
    }
}

# =============================================================================
# Variable Normalizer - Maps diverse variable names to canonical ones
# =============================================================================

class VariableNormalizer(ast.NodeTransformer):
    def __init__(self):
        # Comprehensive mapping from observed variable names to canonical names
        self.base_mapping = {
            # Input sequences
            'action_1': 'a1_seq', 'action_2': 'a2_seq', 'state': 's_seq', 
            'reward': 'r_seq', 'stai': 'stai_arr', 'model_parameters': 'params',
            
            # Trial count
            'n_trials': 'num_trials', 'N': 'num_trials',
            
            # STAI scalar
            'stai_score': 'stai', 'stai_val': 'stai', 'st': 'stai', 's_anx': 'stai',
            'stai0': 'stai',
            
            # Q-values stage 1
            'q_stage1_mf': 'q1', 'q1_mf': 'q1', 'Q1_mf': 'q1', 'Q1': 'q1',
            'q_stage1_mb': 'q1_mb', 'Q1_mb': 'q1_mb', 'Q1_MB': 'q1_mb',
            'q1_hybrid': 'q1_combined', 'q1_eff': 'q1_combined', 'q1_net': 'q1_combined',
            'q1_combined': 'q1_combined', 'q1_total': 'q1_combined', 'q1_comb': 'q1_combined',
            'q1_policy_vals': 'q1_combined',
            
            # Q-values stage 2 - row access patterns
            'q2_s': 'q2_row', 'q2_vec': 'q2_row', 'q2_state': 'q2_row',
            'q2_policy_vals': 'q2_row', 'pref2': 'q2_row',
            
            # Transition matrix
            'transition_matrix': 'T', 'T_known': 'T', 'T_fixed': 'T', 'T_est': 'T',
            'trans_counts': 'T_counts', 'T_counts': 'T_counts',
            
            # Probability sequences
            'p_choice_1': 'p1_seq', 'p_choice_2': 'p2_seq',
            # NOTE: Don't normalize 'probs', 'probs1', 'probs2' - they have different roles
            # in different models (trial sequences vs temporary softmax outputs)
            
            # Logits / centered values
            'logits1': 'logits', 'logits2': 'logits', 'l1': 'logits', 'l2': 'logits',
            'q2c': 'q_centered', 'q1c': 'q_centered', 'q_centered': 'q_centered',
            'centered1': 'q_centered', 'centered2': 'q_centered',
            'q1_shift': 'q_centered', 'q2_shift': 'q_centered',
            'q1_centered': 'q_centered', 'q2_centered': 'q_centered',
            'z1': 'q_centered', 'z2': 'q_centered',
            'c_q1': 'q_centered', 'c_q2': 'q_centered',
            
            # Probabilities
            'ps2': 'probs', 'ps1': 'probs', 'pr1': 'probs', 'pr2': 'probs',
            'probs_1': 'probs1', 'probs_2': 'probs2',
            # NOTE: Don't normalize 'p1', 'p2' - they have different roles in different models
            # (trial sequences vs temporary softmax outputs)
            'exp_q1': 'exp_q', 'exp_q2': 'exp_q', 'exp1': 'exp_q', 'exp2': 'exp_q',
            'soft1': 'soft_probs', 'soft2': 'soft_probs', 'probs2_soft': 'soft_probs',
            
            # Decay/forgetting
            'forget': 'decay', 'rho': 'decay', 'rho_forget': 'decay', 
            'tau_forget': 'decay', 'f_forget': 'decay', 'forget0': 'decay_base',
            'decay_eff': 'decay_eff', 'decay_eff_1': 'decay_eff', 'decay_eff_2': 'decay_eff',
            'keep': 'keep', 'f': 'decay',
            
            # Prediction errors
            'pe1': 'delta1', 'pe2': 'delta2', 'td1': 'delta1', 'td2': 'delta2',
            'delta1_boot': 'delta1', 'delta1_bootstrap': 'delta1',
            
            # Learning rates
            'eta': 'alpha', 'rho_v': 'alpha', 'alphaQ': 'alpha',
            'alpha_pos': 'alpha_pos', 'alpha_neg': 'alpha_neg',
            'alpha_plus': 'alpha_pos', 'alpha_minus': 'alpha_neg',
            'eff_alpha_pos': 'alpha_pos', 'eff_alpha_neg': 'alpha_neg',
            'lr1': 'lr', 'lr2': 'lr', 'a1_lr': 'lr', 'a2_lr': 'lr',
            'alpha2': 'alpha', 'alpha_q': 'alpha', 'alpha_r': 'alpha',
            'mu_win': 'alpha_pos', 'mu_loss': 'alpha_neg',
            
            # Inverse temperature
            'beta1': 'beta', 'beta2': 'beta', 'beta_eff': 'beta', 
            'b1': 'beta', 'b2': 'beta', 'beta_t': 'beta',
            'beta_base_eff': 'beta',
            
            # Perseveration / stickiness
            'kappa': 'pers', 'kappa_eff': 'pers_eff', 'kappa_stick': 'pers',
            'kappa0': 'pers', 'kappa1': 'pers',
            'pi': 'pers', 'pi_eff': 'pers_eff', 'stick': 'pers', 
            'stick_strength': 'pers_eff', 'stick_eff': 'pers_eff',
            'persev': 'pers', 'persev_eff': 'pers_eff', 'perseveration': 'pers',
            'psi_persist': 'pers', 'psi_perseverate': 'pers',
            'stickiness': 'pers_bias', 'persev_bias': 'pers_bias',
            'stick_vec': 'pers_vec', 'stick1': 'pers_vec', 'stick2': 'pers_vec',
            'pers_bias': 'pers_vec',
            
            # MB weight
            'w': 'w_mb', 'omega': 'w_mb', 'omega_eff': 'w_mb_eff', 'w_eff': 'w_mb_eff',
            'w_mb': 'w_mb', 'mb_weight': 'w_mb', 'mb_w_eff': 'w_mb_eff',
            'w_mf': 'w_mf', 'w0': 'w_mb_base', 'omega0': 'w_mb_base',
            'weight_mb': 'w_mb', 'phi': 'w_mb',
            
            # Bias vectors
            'bias1': 'bias', 'bias2': 'bias', 'bias': 'bias',
            'pers_bias': 'bias', 'perseveration_bias': 'bias',
            
            # Lambda (eligibility)
            'lam': 'lambda_', 'lambd': 'lambda_', 'lambda_e': 'lambda_',
            
            # Previous action tracking
            'prev_a1': 'last_a1', 'a1_prev': 'last_a1', 'prev_choice1': 'last_a1',
            
            # State indices
            's2': 's', 's2_prev': 's_prev', 'other_s': 'other_s',
            'other_a1': 'other_a1', 'other_a2': 'other_a2',
            
            # Max Q values
            'max_Q2': 'max_q2', 'max_q_stage2': 'max_q2', 'vmax': 'max_q2',
            'V_state': 'v_state',
            
            # Epsilon
            'eps': 'epsilon', 'eps_h': 'epsilon',
            
            # Target/backup values
            'target1': 'target1', 'boot': 'backup', 'backup': 'backup',
            'td_target1': 'target1',
        }
        self.q2_tables = set()
        self.t_tables = set()

    def analyze(self, tree):
        for node in ast.walk(tree):
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
                if isinstance(node.left, ast.Name):
                    self.t_tables.add(node.left.id)
            
            if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
                dims = []
                if isinstance(node.slice, ast.Tuple):
                    dims = node.slice.elts
                elif hasattr(node.slice, 'value') and isinstance(node.slice.value, ast.Tuple):
                    dims = node.slice.value.elts
                
                if len(dims) == 2:
                    row = dims[0]
                    col = dims[1]
                    row_name = row.id if isinstance(row, ast.Name) else None
                    col_name = col.id if isinstance(col, ast.Name) else None
                    
                    if row_name in ['state', 's', 's_seq', 's2'] and col_name in ['action_2', 'a2', 'a2_seq']:
                        self.q2_tables.add(node.value.id)
                    elif row_name in ['action_1', 'a1', 'a1_seq'] and col_name in ['state', 's', 's_seq']:
                        self.t_tables.add(node.value.id)

    def visit_Name(self, node):
        if node.id in self.t_tables:
            return ast.copy_location(ast.Name(id='T', ctx=node.ctx), node)
        if node.id in self.q2_tables:
            return ast.copy_location(ast.Name(id='Q2', ctx=node.ctx), node)
            
        if node.id in self.base_mapping:
            return ast.copy_location(ast.Name(id=self.base_mapping[node.id], ctx=node.ctx), node)
        return node

    def visit_arg(self, node):
        if node.arg in self.t_tables:
             return ast.copy_location(ast.arg(arg='T', annotation=node.annotation), node)
        if node.arg in self.q2_tables:
             return ast.copy_location(ast.arg(arg='Q2', annotation=node.annotation), node)
        if node.arg in self.base_mapping:
            return ast.copy_location(ast.arg(arg=self.base_mapping[node.arg], annotation=node.annotation), node)
        return node


class ExpressionCanonicalizer(ast.NodeTransformer):
    """Canonicalize commutative operations for better matching"""
    def visit_BinOp(self, node):
        self.generic_visit(node)
        if isinstance(node.op, (ast.Add, ast.Mult)):
            left_str = astunparse.unparse(node.left).strip()
            right_str = astunparse.unparse(node.right).strip()
            if left_str > right_str:
                node.left, node.right = node.right, node.left
        return node


class NumpyNormalizer(ast.NodeTransformer):
    """Normalize numpy calls to remove dtype=float (but keep dtype=int, dtype=bool)"""
    def visit_Call(self, node):
        self.generic_visit(node)
        # Check if this is a numpy call
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            if node.func.value.id == 'np':
                # Only remove dtype=float - keep dtype=int, dtype=bool, etc.
                new_keywords = []
                for kw in node.keywords:
                    if kw.arg == 'dtype':
                        # Check if it's dtype=float
                        if isinstance(kw.value, ast.Name) and kw.value.id == 'float':
                            continue  # Skip dtype=float
                        # Also skip dtype=np.float64, etc.
                        if isinstance(kw.value, ast.Attribute):
                            if kw.value.attr in ['float64', 'float32', 'float_']:
                                continue
                    new_keywords.append(kw)
                node.keywords = new_keywords
        return node


class StatementSorter(ast.NodeTransformer):
    """Sort independent statements for canonical ordering"""
    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        node.body = self.sort_block(node.body)
        return node
        
    def visit_For(self, node):
        self.generic_visit(node)
        node.body = self.sort_block(node.body)
        return node
        
    def sort_block(self, stmts):
        changed = True
        while changed:
            changed = False
            for i in range(len(stmts) - 1):
                s1 = stmts[i]
                s2 = stmts[i+1]
                
                if isinstance(s1, ast.Assign) and isinstance(s2, ast.Assign):
                    if self.are_independent(s1, s2):
                        str1 = astunparse.unparse(s1).strip()
                        str2 = astunparse.unparse(s2).strip()
                        if str1 > str2:
                            stmts[i], stmts[i+1] = stmts[i+1], stmts[i]
                            changed = True
        return stmts

    def are_independent(self, s1, s2):
        t1 = self.get_targets(s1)
        i1 = self.get_inputs(s1)
        t2 = self.get_targets(s2)
        i2 = self.get_inputs(s2)
        
        if not t1.isdisjoint(i2): return False
        if not t2.isdisjoint(i1): return False
        if not t1.isdisjoint(t2): return False
        return True
        
    def get_targets(self, node):
        targets = set()
        for t in node.targets:
            if isinstance(t, ast.Name):
                targets.add(t.id)
            elif isinstance(t, ast.Tuple):
                for elt in t.elts:
                    if isinstance(elt, ast.Name):
                        targets.add(elt.id)
        return targets
        
    def get_inputs(self, node):
        inputs = set()
        if hasattr(node, 'value'):
            for n in ast.walk(node.value):
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
                    inputs.add(n.id)
        targets = node.targets if isinstance(node, ast.Assign) else ([node.target] if hasattr(node, 'target') else [])
        for t in targets:
            if isinstance(t, ast.Subscript):
                for n in ast.walk(t.slice):
                    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
                        inputs.add(n.id)
        return inputs


class CodeCorpus:
    def __init__(self, models):
        self.models = models
        self.asts = {}
        self.normalized_asts = {}
        self.parse_and_normalize()
        
    def parse_and_normalize(self):
        for pid, code in self.models.items():
            try:
                tree = ast.parse(code)
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module)):
                        if ast.get_docstring(node):
                            node.body = node.body[1:] if len(node.body) > 1 and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) else node.body
                
                self.asts[pid] = tree
                
                norm_tree = copy.deepcopy(tree)
                normalizer = VariableNormalizer()
                normalizer.analyze(norm_tree)
                normalizer.visit(norm_tree)
                
                # Normalize numpy calls (remove dtype=float, etc.)
                numpy_norm = NumpyNormalizer()
                numpy_norm.visit(norm_tree)
                
                expr_canon = ExpressionCanonicalizer()
                expr_canon.visit(norm_tree)
                
                sorter = StatementSorter()
                sorter.visit(norm_tree)
                
                self.normalized_asts[pid] = norm_tree
                
            except SyntaxError:
                print(f"Syntax Error in participant {pid}")

    def extract_variables(self, stmt_nodes, known_functions=set()):
        class VariableVisitor(ast.NodeVisitor):
            def __init__(self):
                self.inputs = set()
                self.outputs = set()
                self.assigned = set()
                
            def visit_Assign(self, node):
                self.visit(node.value)
                for target in node.targets:
                    self.visit(target)
            
            def visit_AugAssign(self, node):
                self.visit(node.value)
                self.visit(node.target)

            def visit_Name(self, node):
                if isinstance(node.ctx, ast.Store):
                    self.assigned.add(node.id)
                    self.outputs.add(node.id)
                elif isinstance(node.ctx, ast.Load):
                    if node.id not in self.assigned:
                        self.inputs.add(node.id)
            
        visitor = VariableVisitor()
        if isinstance(stmt_nodes, list):
            block_code = "\n".join([astunparse.unparse(n).strip() for n in stmt_nodes])
        else:
            block_code = astunparse.unparse(stmt_nodes).strip()
            
        try:
            tree = ast.parse(block_code)
        except:
            return [], []
            
        visitor.visit(tree)
        
        inputs = visitor.inputs
        outputs = visitor.outputs
                        
        excludes = {'np', 'range', 'len', 'int', 'float', 'print', 'min', 'max', 'sum', 'abs', 'log', 'exp'}
        inputs = sorted(list(inputs - excludes - known_functions))
        outputs = sorted(list(outputs))
        return inputs, outputs

    def replace_sequence(self, seq_strs, func_name, inputs, outputs):
        count = 0
        for pid, tree in self.normalized_asts.items():
            for node in ast.walk(tree):
                if hasattr(node, 'body') and isinstance(node.body, list):
                    new_body = []
                    i = 0
                    stmts = node.body
                    while i < len(stmts):
                        match = False
                        if i <= len(stmts) - len(seq_strs):
                            current_seq = tuple(astunparse.unparse(s).strip() for s in stmts[i : i+len(seq_strs)])
                            if current_seq == tuple(seq_strs):
                                match = True
                        
                        if match:
                            call = ast.Call(
                                func=ast.Name(id=func_name, ctx=ast.Load()),
                                args=[ast.Name(id=v, ctx=ast.Load()) for v in inputs],
                                keywords=[]
                            )
                            if outputs:
                                if len(outputs) == 1:
                                    target = ast.Name(id=outputs[0], ctx=ast.Store())
                                else:
                                    target = ast.Tuple(elts=[ast.Name(id=v, ctx=ast.Store()) for v in outputs], ctx=ast.Store())
                                assign = ast.Assign(targets=[target], value=call)
                                new_body.append(assign)
                            else:
                                new_body.append(ast.Expr(value=call))
                            i += len(seq_strs)
                            count += 1
                        else:
                            new_body.append(stmts[i])
                            i += 1
                    node.body = new_body
        return count

    def find_common_sequence(self, min_len=2, max_len=10, min_freq=3):
        sequences = Counter()
        for pid, tree in self.normalized_asts.items():
            for node in ast.walk(tree):
                if hasattr(node, 'body') and isinstance(node.body, list):
                    stmts = node.body
                    norm_stmts = [astunparse.unparse(s).strip() for s in stmts]
                    for n in range(min_len, min(len(stmts), max_len) + 1):
                        for i in range(len(stmts) - n + 1):
                            has_return = False
                            for k in range(n):
                                if isinstance(stmts[i+k], ast.Return):
                                    has_return = True
                                    break
                            if has_return:
                                continue

                            seq = tuple(norm_stmts[i : i+n])
                            sequences[seq] += 1
        
        candidates = []
        for seq, freq in sequences.items():
            if freq < min_freq: continue
            n_stmts = len(seq)
            score = (n_stmts * freq) - n_stmts 
            candidates.append((score, seq, freq))
            
        if not candidates: return None
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0]

    def find_common_expression(self, library, min_freq=2):
        expressions = Counter()
        
        class ExprVisitor(ast.NodeVisitor):
            def visit_BinOp(self, node):
                self.generic_visit(node)
                s = astunparse.unparse(node).strip()
                # Focus on cognitively meaningful expressions
                keywords = ['stai', 'decay', 'w_mb', 'alpha', 'pers', '1.0 -', 
                           'np.clip', 'np.max', '0.5 *', '+ 0.5', 'lambda_',
                           '0.6 *', '0.7 *', '0.8 *', 'keep', 'np.sum']
                if len(s) > 8 and any(kw in s for kw in keywords): 
                    expressions[s] += 1
            
            def visit_Call(self, node):
                self.generic_visit(node)
                if isinstance(node.func, ast.Name) and node.func.id in library: return
                if isinstance(node.func, ast.Attribute):
                    # Capture numpy calls like np.max, np.exp, np.clip
                    s = astunparse.unparse(node).strip()
                    if len(s) > 10 and any(kw in s for kw in ['np.clip', 'np.max', 'np.exp']):
                        expressions[s] += 1
                else:
                    s = astunparse.unparse(node).strip()
                    if len(s) > 10 and any(kw in s for kw in ['stai', 'decay', 'np.clip', 'np.max']):
                        expressions[s] += 1

        for pid, tree in self.normalized_asts.items():
            ExprVisitor().visit(tree)
            
        candidates = []
        for expr, freq in expressions.items():
            if freq < min_freq: continue
            # Score by compression gain
            candidates.append((len(expr) * freq, expr, freq))
            
        if not candidates: return None
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0]

    def replace_expression(self, expr_str, func_name, inputs):
        count = 0
        
        class ExprReplacer(ast.NodeTransformer):
            def visit_BinOp(self, node):
                self.generic_visit(node)
                if astunparse.unparse(node).strip() == expr_str:
                    nonlocal count
                    count += 1
                    return ast.Call(
                        func=ast.Name(id=func_name, ctx=ast.Load()),
                        args=[ast.Name(id=v, ctx=ast.Load()) for v in inputs],
                        keywords=[]
                    )
                return node
            
            def visit_Call(self, node):
                self.generic_visit(node)
                if astunparse.unparse(node).strip() == expr_str:
                    nonlocal count
                    count += 1
                    return ast.Call(
                        func=ast.Name(id=func_name, ctx=ast.Load()),
                        args=[ast.Name(id=v, ctx=ast.Load()) for v in inputs],
                        keywords=[]
                    )
                return node

        for pid, tree in self.normalized_asts.items():
            ExprReplacer().visit(tree)
        return count


def name_function(block_code, iteration):
    """Assign meaningful names based on cognitive function"""
    
    # Initialization patterns
    if "np.zeros" in block_code and "p1_seq" in block_code and "p2_seq" in block_code:
        if "q1" in block_code:
            return "init_trial_vars"
        return "init_choice_probs"
    if "np.zeros" in block_code and "Q2" in block_code and "T" in block_code:
        return "init_q_values"
    if "len(a1_seq)" in block_code and "stai_arr" in block_code:
        return "init_trial_vars"
        
    # Model-based computation
    if "T @ max_q2" in block_code or "T @" in block_code:
        if "w_mb" in block_code:
            return "compute_hybrid_q1"
        return "compute_mb_values"
    if "np.max(Q2, axis=1)" in block_code:
        return "compute_max_q2"
        
    # Hybrid MF-MB combination
    if "1.0 - w" in block_code and "q1_mb * w" in block_code:
        return "combine_mf_mb"
    if "(1.0 - w_mb)" in block_code and "q1_mb" in block_code:
        return "combine_mf_mb"
        
    # Softmax / choice probability
    if "np.exp" in block_code and "np.sum" in block_code and "/=" in block_code:
        return "softmax_choice"
    if "np.exp" in block_code and "/ np.sum" in block_code:
        if "probs1" in block_code:
            return "softmax_stage1"
        if "probs2" in block_code:
            return "softmax_stage2"
        return "softmax_choice"
    if "- np.max" in block_code and "np.exp" in block_code:
        return "softmax_choice"
        
    # TD updates
    if "delta2" in block_code and "Q2" in block_code and "+=" in block_code:
        if "delta1" in block_code and "q1" in block_code:
            return "td_update_both"
        return "td_update_q2"
    if "delta1" in block_code and "q1" in block_code and "+=" in block_code:
        if "lambda_" in block_code:
            return "td_update_q1_eligibility"
        return "td_update_q1"
        
    # Decay / forgetting
    if "(1.0 - decay)" in block_code and "0.5 * decay" in block_code:
        return "decay_to_prior"
    if "*= decay" in block_code or "* decay" in block_code:
        if "Q2" in block_code and "q1" in block_code:
            return "decay_all_values"
        return "apply_decay"
    if "*= keep" in block_code or "* keep" in block_code:
        return "apply_keep_factor"
        
    # Perseveration
    if "last_a1" in block_code and "bias" in block_code and "+=" in block_code:
        return "add_perseveration"
    if "pers" in block_code and "bias" in block_code and "stick" in block_code:
        return "apply_stickiness"
        
    # NLL calculation
    if "np.log" in block_code and "nll" in block_code:
        return "compute_nll"
    if "np.sum" in block_code and "np.log" in block_code and "p1_seq" in block_code:
        return "compute_nll"
        
    # Recording choices
    if "p2_seq[t]" in block_code and "r_seq" in block_code:
        return "record_stage2"
    if "p1_seq[t]" in block_code and "s_seq" in block_code:
        return "record_stage1"
        
    # Get actions
    if "a1_seq[t]" in block_code and "a2_seq[t]" in block_code:
        return "get_actions"
    if "r_seq[t]" in block_code and "s_seq[t]" in block_code:
        return "get_outcomes"
        
    # Transition learning
    if "T[a1]" in block_code and "target" in block_code:
        return "update_transition"
    if "T_counts" in block_code:
        return "update_transition_counts"
        
    # Asymmetric learning
    if "alpha_pos" in block_code and "alpha_neg" in block_code:
        return "select_asymmetric_lr"
        
    # Uncertainty/entropy
    if "entropy" in block_code or "ent" in block_code:
        return "compute_entropy"
    if "uncertainty" in block_code or "unc" in block_code:
        return "compute_uncertainty"
        
    # STAI extraction
    if "stai_arr[0]" in block_code or "extract_stai" in block_code:
        return "extract_stai"
        
    # Default
    return f"mechanism_{iteration}"


def name_expression(expr_str, iteration):
    """Name expressions based on cognitive meaning"""
    
    # STAI modulation patterns
    if "1.0 - stai" in expr_str or "(1 - stai" in expr_str:
        if "np.clip" in expr_str:
            return "stai_modulate_clipped"
        return "stai_complement"
    if "0.5 * stai" in expr_str:
        if "+ 0.5" in expr_str:
            return "stai_scale_half_plus"
        return "stai_scale_half"
    if "1.0 + stai" in expr_str or "(1 + stai" in expr_str:
        return "stai_amplify"
    if "stai_arr[0]" in expr_str:
        return "extract_stai_scalar"
    if "0.6 * stai" in expr_str:
        return "stai_reduce_60pct"
    if "0.7 * stai" in expr_str:
        return "stai_reduce_70pct"
    if "0.8 * stai" in expr_str:
        return "stai_reduce_80pct"
    if "stai - 0.5" in expr_str or "(stai - 0.31" in expr_str:
        return "stai_center"
        
    # Decay patterns
    if "1.0 - decay" in expr_str or "(1 - decay" in expr_str:
        if "0.5 * decay" in expr_str:
            return "decay_to_prior"
        return "decay_complement"
    if "1.0 - decay_eff" in expr_str:
        return "keep_factor"
        
    # Weight modulation
    if "w_mb" in expr_str and "stai" in expr_str:
        if "np.clip" in expr_str:
            return "modulate_mb_weight_clipped"
        return "modulate_mb_weight"
        
    # Learning rate modulation
    if "alpha" in expr_str and "stai" in expr_str:
        if "np.clip" in expr_str:
            return "modulate_lr_clipped"
        return "modulate_lr"
        
    # Perseveration modulation
    if "pers" in expr_str and "stai" in expr_str:
        return "modulate_perseveration"
        
    # Beta/temperature modulation
    if "beta" in expr_str and "stai" in expr_str:
        return "modulate_temperature"
        
    # Max operations
    if "np.max(" in expr_str:
        if "axis=1" in expr_str:
            return "max_q2_per_state"
        if "logits" in expr_str or "q1" in expr_str or "q2" in expr_str:
            return "get_max_value"
            
    # Clip operations
    if "np.clip" in expr_str:
        return "clip_value"
        
    return f"calc_expr_{iteration}"


def main():
    print("=" * 60)
    print("Cognitive Library Learner v4.0")
    print("=" * 60)
    
    print("\nLoading models...")
    best_models = get_best_models(INPUT_FILE)
    print(f"Loaded {len(best_models)} models")
    
    corpus = CodeCorpus(best_models)
    library = {}
    
    # Multi-pass extraction for maximum compression
    total_passes = 3
    
    for pass_num in range(total_passes):
        print(f"\n{'='*60}")
        print(f"PASS {pass_num + 1}/{total_passes}")
        print(f"{'='*60}")
        
        # Phase 1: Extract common expressions (STAI modulation, decay patterns)
        print(f"\n--- Pass {pass_num+1} Phase 1: Extracting Expression Patterns ---")
        iter_expr = len(library)
        expr_count = 0
        while expr_count < 15:  # Extract up to 15 expressions per pass
            best = corpus.find_common_expression(library, min_freq=4)  
            if not best: break
            score, expr_str, freq = best
            
            if score < 50: break  # Not worth extracting
            
            # Skip expressions that could cause issues
            if "Q2[s]" in expr_str or "Q2[(s" in expr_str:
                break
            if "q1[a1]" in expr_str:
                break
            
            try:
                expr_node = ast.parse(expr_str).body[0].value
            except:
                break
                
            inputs, _ = corpus.extract_variables(expr_node, known_functions=set(library.keys()))
            
            func_name = name_expression(expr_str, iter_expr)
            
            # Avoid duplicate names
            base_name = func_name
            suffix = 0
            while func_name in library:
                suffix += 1
                func_name = f"{base_name}_{suffix}"
                
            print(f"  {func_name:<35} (freq={freq:2d}): {expr_str[:45]}...")
            
            library[func_name] = {
                'code': [f"return {expr_str}"],
                'inputs': inputs,
                'outputs': []
            }
            
            corpus.replace_expression(expr_str, func_name, inputs)
            iter_expr += 1
            expr_count += 1
        
        # Phase 2: Extract statement sequences (cognitive mechanisms)
        print(f"\n--- Pass {pass_num+1} Phase 2: Extracting Statement Sequences ---")
        iteration = len(library)
        seq_count = 0
        skipped = set()
        while seq_count < 25:  # Extract up to 25 sequences per pass
            best = corpus.find_common_sequence(min_len=2, max_len=8, min_freq=5)
            if not best: break
            score, seq_strs, freq = best
            if score <= 0: break
            
            # Skip if we've seen this pattern before
            if seq_strs in skipped:
                break
            
            # Skip sequences that could cause variable aliasing issues
            block_code = "\n".join(seq_strs)
            skip_this = False
            
            # Pattern 1: Reassigning a trial-indexed array to a different shape
            if "= Q2[s]" in block_code and "Q2[(" not in block_code:
                skip_this = True
            
            # Pattern 2: Reassigning probs arrays that are used both for trial storage and softmax
            if "probs1 = " in block_code or "probs2 = " in block_code:
                skip_this = True
                
            if skip_this:
                skipped.add(seq_strs)
                continue
                
            try:
                dummy_code = f"def dummy():\n" + "\n".join(["    " + s for s in seq_strs])
                dummy_tree = ast.parse(dummy_code)
                stmt_nodes = dummy_tree.body[0].body
            except:
                break

            inputs, outputs = corpus.extract_variables(stmt_nodes, known_functions=set(library.keys()))
            
            func_name = name_function(block_code, iteration)
            
            # Avoid duplicate names
            base_name = func_name
            suffix = 0
            while func_name in library:
                suffix += 1
                func_name = f"{base_name}_{suffix}"
                
            print(f"  {func_name:<35} (freq={freq:2d}, lines={len(seq_strs)})")
            
            library[func_name] = {
                'code': seq_strs,
                'inputs': inputs,
                'outputs': outputs
            }
            
            corpus.replace_sequence(seq_strs, func_name, inputs, outputs)
            iteration += 1
            seq_count += 1

    # Save Library
    print("\n" + "=" * 60)
    print("SAVING LIBRARY")
    print("=" * 60)
    
    lib_path = os.path.join(OUTPUT_DIR, "cognitive_library.py")
    with open(lib_path, "w") as f:
        f.write('"""Cognitive Modeling Library v4.0\n\n')
        f.write('Automatically extracted primitives for two-step task RL models.\n')
        f.write('Patterns extracted:\n')
        f.write('- STAI modulation (anxiety scaling)\n')
        f.write('- Decay/forgetting mechanisms\n')
        f.write('- Model-based/model-free hybrid computations\n')
        f.write('- Softmax choice probabilities\n')
        f.write('- TD learning updates\n')
        f.write('- Perseveration/stickiness biases\n')
        f.write('"""\n\n')
        f.write("import numpy as np\n\n")
        
        for name, data in library.items():
            args = ", ".join(data['inputs'])
            f.write(f"def {name}({args}):\n")
            for line in data['code']:
                f.write(f"    {line}\n")
            if data['outputs']:
                rets = ", ".join(data['outputs'])
                f.write(f"    return {rets}\n")
            f.write("\n")
    
    print(f"  Library saved to: {lib_path}")
    print(f"  Total functions: {len(library)}")
            
    # Save Models
    models_path = os.path.join(OUTPUT_DIR, "reconstructed_models.py")
    with open(models_path, "w") as f:
        f.write("import numpy as np\n")
        f.write("from .cognitive_library import *\n\n")
        
        for pid, tree in corpus.normalized_asts.items():
            func_def = tree.body[0]
            code = astunparse.unparse(func_def)
            f.write(f"# Participant {pid}\n")
            f.write(code)
            f.write("\n")

    print(f"  Models saved to: {models_path}")
    
    # Print compression statistics
    print("\n" + "=" * 60)
    print("COMPRESSION STATISTICS")
    print("=" * 60)
    
    # Count lines in original vs reconstructed
    orig_lines = sum(len(code.split('\n')) for code in best_models.values())
    with open(models_path, 'r') as f:
        recon_lines = len(f.readlines())
    with open(lib_path, 'r') as f:
        lib_lines = len(f.readlines())
    
    print(f"  Original total lines:     {orig_lines}")
    print(f"  Reconstructed lines:      {recon_lines}")
    print(f"  Library lines:            {lib_lines}")
    print(f"  Total compressed lines:   {recon_lines + lib_lines}")
    print(f"  Compression ratio:        {orig_lines / (recon_lines + lib_lines):.2f}x")
    
    print("\n" + "=" * 60)
    print("Done!")

if __name__ == "__main__":
    main()
