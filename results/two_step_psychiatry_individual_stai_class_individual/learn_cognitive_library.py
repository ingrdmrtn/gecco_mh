"""
Library Learning for Class-Based Cognitive Models (v2)
=======================================================

This version focuses on semantic clustering based on:
1. Cognitive mechanisms used (not just code matching)
2. STAI modulation strategies
3. Generating usable library functions

Key insight: The class-based structure naturally separates concerns:
- unpack_parameters: Parameter configuration
- init_model: STAI modulation setup  
- policy_stage1/2: Action selection mechanisms
- value_update: Learning rules
- post_trial: Additional effects (decay, etc.)
"""

import ast
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
import textwrap

# ============================================================
# Configuration
# ============================================================

MODELS_DIR = "/home/aj9225/gecco-1/results/two_step_psychiatry_individual_stai_class_individual/models"
OUTPUT_DIR = "/home/aj9225/gecco-1/results/two_step_psychiatry_individual_stai_class_individual/library_learned"

OVERRIDABLE_METHODS = [
    "unpack_parameters",
    "init_model", 
    "policy_stage1",
    "policy_stage2",
    "value_update",
    "pre_trial",
    "post_trial",
]

# ============================================================
# Semantic Analysis Patterns
# ============================================================

# Patterns to detect cognitive mechanisms
MECHANISM_PATTERNS = {
    # Model-based planning
    "model_based": [
        r"self\.T\s*@",
        r"self\.T\[",
        r"transition",
        r"q_mb",
        r"max\(self\.q_stage2",
        r"np\.max\(self\.q_stage2",
    ],
    
    # Perseveration / stickiness
    "perseveration": [
        r"last_action",
        r"stickiness",
        r"sticky",
        r"persev",
        r"habit",
        r"repeat",
    ],
    
    # Win-stay / reward-dependent
    "win_stay": [
        r"last_reward\s*==\s*1",
        r"last_reward\s*>\s*0",
        r"win",
        r"cling",
    ],
    
    # Asymmetric learning
    "asymmetric_learning": [
        r"alpha_pos",
        r"alpha_neg",
        r"delta.*[<>].*0",
        r"if.*delta",
    ],
    
    # Memory decay
    "memory_decay": [
        r"decay",
        r"\*=\s*\(1",
        r"forget",
        r"unchosen",
    ],
    
    # Eligibility traces
    "eligibility_trace": [
        r"lambda",
        r"trace",
        r"eligib",
    ],
}

# STAI modulation patterns
STAI_PATTERNS = {
    "multiplicative": r"[\*]\s*self\.stai(?!\s*[\*])",
    "inverse_linear": r"\(1\.?\d*\s*-\s*self\.stai\)",
    "additive_boost": r"\(1\.?\d*\s*\+\s*self\.stai\)",
    "inverse_division": r"/\s*\(1\.?\d*\s*\+\s*self\.stai\)",
    "stai_first": r"self\.stai\s*\*",
    "clipped": r"clip.*stai|stai.*clip",
}


# ============================================================
# Data Structures
# ============================================================

@dataclass
class MethodInfo:
    """Information about a method implementation."""
    method_name: str
    source_code: str
    participant_id: str
    mechanisms: Set[str] = field(default_factory=set)
    stai_pattern: Optional[str] = None
    
    def signature(self) -> str:
        """Create a semantic signature for clustering."""
        mechs = sorted(self.mechanisms) if self.mechanisms else ["standard"]
        stai = f"_stai_{self.stai_pattern}" if self.stai_pattern else ""
        return f"{self.method_name}_{'_'.join(mechs)}{stai}"


@dataclass
class ExtractedModel:
    """A parsed cognitive model."""
    class_name: str
    participant_id: str
    file_path: str
    docstring: str
    parameters: List[str]
    methods: Dict[str, MethodInfo]
    
    def summary(self) -> str:
        mechs = set()
        stai_patterns = set()
        for m in self.methods.values():
            mechs.update(m.mechanisms)
            if m.stai_pattern:
                stai_patterns.add(m.stai_pattern)
        return f"Mechanisms: {mechs}, STAI: {stai_patterns}"


@dataclass
class SemanticCluster:
    """A cluster of semantically similar method implementations."""
    signature: str
    method_name: str
    mechanisms: Set[str]
    stai_pattern: Optional[str]
    instances: List[MethodInfo] = field(default_factory=list)
    
    @property
    def frequency(self) -> int:
        return len(self.instances)
    
    def canonical_code(self) -> str:
        """Return the most common/representative code."""
        if not self.instances:
            return ""
        # Use the first instance as canonical
        return self.instances[0].source_code


# ============================================================
# Parsing Functions
# ============================================================

def extract_code_from_file(filepath: str) -> str:
    """Extract Python code from a model file."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    match = re.search(r"```(?:python|plaintext)?\s*(.*?)```", content, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    if "class " in content:
        return content.strip()
    
    return content.strip()


def parse_class(code: str) -> Optional[ast.ClassDef]:
    """Parse the first class definition from code."""
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                return node
    except SyntaxError as e:
        print(f"  [!] Syntax error: {e}")
    return None


def get_method_source(class_node: ast.ClassDef, method_name: str, full_code: str) -> Optional[str]:
    """Extract method source code."""
    for node in class_node.body:
        if isinstance(node, ast.FunctionDef) and node.name == method_name:
            lines = full_code.split('\n')
            return '\n'.join(lines[node.lineno - 1:node.end_lineno])
    return None


def extract_docstring(class_node: ast.ClassDef) -> str:
    """Extract class docstring."""
    if (class_node.body and 
        isinstance(class_node.body[0], ast.Expr) and
        isinstance(class_node.body[0].value, ast.Constant)):
        return class_node.body[0].value.value
    return ""


def extract_parameters(code: str) -> List[str]:
    """Extract parameter names from unpack_parameters."""
    pattern = re.search(r"self\.(\w+(?:\s*,\s*self\.\w+)*)\s*=\s*model_parameters", code)
    if pattern:
        return re.findall(r"self\.(\w+)", pattern.group(0))
    return []


def detect_mechanisms(code: str) -> Set[str]:
    """Detect cognitive mechanisms used in the code."""
    mechanisms = set()
    code_lower = code.lower()
    
    for mechanism, patterns in MECHANISM_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, code_lower, re.IGNORECASE):
                mechanisms.add(mechanism)
                break
    
    return mechanisms


def detect_stai_pattern(code: str) -> Optional[str]:
    """Detect STAI modulation pattern."""
    if "self.stai" not in code:
        return None
    
    for pattern_name, pattern in STAI_PATTERNS.items():
        if re.search(pattern, code, re.IGNORECASE):
            return pattern_name
    
    return "custom"


# ============================================================
# Model Extraction
# ============================================================

def extract_model(filepath: str) -> Optional[ExtractedModel]:
    """Extract a complete model from a file."""
    filename = os.path.basename(filepath)
    match = re.search(r"participant(\d+)", filename)
    participant_id = f"p{match.group(1)}" if match else "unknown"
    
    code = extract_code_from_file(filepath)
    class_node = parse_class(code)
    
    if class_node is None:
        return None
    
    docstring = extract_docstring(class_node)
    
    # Extract methods with semantic analysis
    methods = {}
    for method_name in OVERRIDABLE_METHODS:
        method_source = get_method_source(class_node, method_name, code)
        if method_source:
            methods[method_name] = MethodInfo(
                method_name=method_name,
                source_code=method_source,
                participant_id=participant_id,
                mechanisms=detect_mechanisms(method_source),
                stai_pattern=detect_stai_pattern(method_source),
            )
    
    parameters = []
    if "unpack_parameters" in methods:
        parameters = extract_parameters(methods["unpack_parameters"].source_code)
    
    return ExtractedModel(
        class_name=class_node.name,
        participant_id=participant_id,
        file_path=filepath,
        docstring=docstring,
        parameters=parameters,
        methods=methods,
    )


def load_all_models(models_dir: str) -> List[ExtractedModel]:
    """Load all best models."""
    models = []
    
    for filename in sorted(os.listdir(models_dir)):
        if filename.startswith("best_model_") and filename.endswith(".txt"):
            filepath = os.path.join(models_dir, filename)
            print(f"  Loading {filename}...")
            model = extract_model(filepath)
            if model:
                models.append(model)
                print(f"    ✓ {model.participant_id}: {model.summary()}")
            else:
                print(f"    ✗ Failed to parse")
    
    return models


# ============================================================
# Semantic Clustering
# ============================================================

def cluster_methods(models: List[ExtractedModel]) -> Dict[str, List[SemanticCluster]]:
    """Cluster methods by semantic signature."""
    
    # Group by method name and signature
    signature_groups: Dict[str, Dict[str, List[MethodInfo]]] = defaultdict(lambda: defaultdict(list))
    
    for model in models:
        for method_name, method_info in model.methods.items():
            sig = method_info.signature()
            signature_groups[method_name][sig].append(method_info)
    
    # Create clusters
    clusters: Dict[str, List[SemanticCluster]] = {}
    
    for method_name, sig_groups in signature_groups.items():
        method_clusters = []
        for sig, instances in sig_groups.items():
            # Extract mechanisms and STAI from first instance
            first = instances[0]
            cluster = SemanticCluster(
                signature=sig,
                method_name=method_name,
                mechanisms=first.mechanisms,
                stai_pattern=first.stai_pattern,
                instances=instances,
            )
            method_clusters.append(cluster)
        
        # Sort by frequency
        method_clusters.sort(key=lambda c: c.frequency, reverse=True)
        clusters[method_name] = method_clusters
    
    return clusters


# ============================================================
# Library Generation
# ============================================================

def generate_library(clusters: Dict[str, List[SemanticCluster]], models: List[ExtractedModel]) -> str:
    """Generate the cognitive library module."""
    
    # Track generated functions to avoid duplicates
    generated_functions = set()
    
    lines = [
        '"""',
        'Cognitive Library - Class-Based Model Primitives',
        '=' * 50,
        '',
        'Extracted from 27 participant-specific cognitive models.',
        'Each primitive represents a reusable cognitive mechanism.',
        '',
        'MECHANISM CATEGORIES:',
        '  - model_based: Forward planning using transition matrix',
        '  - perseveration: Tendency to repeat previous actions',
        '  - win_stay: Reward-dependent action repetition',
        '  - asymmetric_learning: Different learning rates for pos/neg outcomes',
        '  - memory_decay: Forgetting of unchosen options',
        '',
        'STAI MODULATION PATTERNS:',
        '  - multiplicative: param * stai',
        '  - inverse_linear: param * (1 - stai)',
        '  - additive_boost: param * (1 + stai)',
        '  - inverse_division: param / (1 + stai)',
        '"""',
        '',
        'import numpy as np',
        'from typing import Tuple, Optional, Callable',
        '',
        '',
        '# ============================================================',
        '# Core Helper Functions',
        '# ============================================================',
        '',
        'def softmax(values: np.ndarray, beta: float) -> np.ndarray:',
        '    """Numerically stable softmax."""',
        '    centered = values - np.max(values)',
        '    exp_vals = np.exp(beta * centered)',
        '    return exp_vals / np.sum(exp_vals)',
        '',
        '',
        'def compute_mb_values(q_stage2: np.ndarray, T: np.ndarray) -> np.ndarray:',
        '    """Compute model-based values: Q_MB = T @ max(Q2)."""',
        '    max_q2 = np.max(q_stage2, axis=1)',
        '    return T @ max_q2',
        '',
        '',
        '# ============================================================',
        '# STAI Modulation Functions',
        '# ============================================================',
        '',
        'def stai_multiplicative(param: float, stai: float) -> float:',
        '    """Scale parameter by STAI: higher anxiety = larger effect."""',
        '    return param * stai',
        '',
        '',
        'def stai_inverse_linear(param: float, stai: float) -> float:',
        '    """Scale parameter inversely: higher anxiety = smaller effect."""',
        '    return param * (1.0 - stai)',
        '',
        '',
        'def stai_additive_boost(param: float, stai: float) -> float:',
        '    """Boost parameter by STAI: higher anxiety = larger effect."""',
        '    return param * (1.0 + stai)',
        '',
        '',
        'def stai_inverse_division(param: float, stai: float) -> float:',
        '    """Divide by (1 + STAI): higher anxiety = smaller effect."""',
        '    return param / (1.0 + stai)',
        '',
        '',
        '# ============================================================',
        '# Policy Stage 1 Primitives',
        '# ============================================================',
        '',
    ]
    
    # Generate policy_stage1 functions
    if "policy_stage1" in clusters:
        for cluster in clusters["policy_stage1"]:
            func_name = make_function_name(cluster)
            if func_name not in generated_functions:
                generated_functions.add(func_name)
                func_code = generate_policy_function(cluster)
                lines.extend(func_code.split('\n'))
                lines.append('')
    
    lines.extend([
        '',
        '# ============================================================',
        '# Value Update Primitives',
        '# ============================================================',
        '',
    ])
    
    # Generate value_update functions
    if "value_update" in clusters:
        for cluster in clusters["value_update"][:5]:  # Top 5 patterns
            func_name = make_function_name(cluster)
            if func_name not in generated_functions:
                generated_functions.add(func_name)
                func_code = generate_value_update_function(cluster)
                lines.extend(func_code.split('\n'))
                lines.append('')
    
    lines.extend([
        '',
        '# ============================================================',
        '# Init Model Primitives',
        '# ============================================================',
        '',
    ])
    
    # Generate init_model functions
    if "init_model" in clusters:
        for cluster in clusters["init_model"]:
            func_name = make_function_name(cluster)
            if func_name not in generated_functions:
                generated_functions.add(func_name)
                func_code = generate_init_function(cluster)
                lines.extend(func_code.split('\n'))
                lines.append('')
    
    lines.extend([
        '',
        '# ============================================================',
        '# Post-Trial Primitives',
        '# ============================================================',
        '',
    ])
    
    # Generate post_trial functions
    if "post_trial" in clusters:
        for cluster in clusters["post_trial"]:
            func_name = make_function_name(cluster)
            if func_name not in generated_functions:
                generated_functions.add(func_name)
                func_code = generate_post_trial_function(cluster)
                lines.extend(func_code.split('\n'))
                lines.append('')
    
    return '\n'.join(lines)


def make_function_name(cluster: SemanticCluster) -> str:
    """Generate a clean function name from cluster signature."""
    # Build name from mechanisms
    mechs = sorted(cluster.mechanisms) if cluster.mechanisms else ["standard"]
    mechs_str = "_".join(mechs)
    
    # Add STAI suffix if present
    stai_suffix = f"_{cluster.stai_pattern}" if cluster.stai_pattern else ""
    
    return f"{cluster.method_name}_{mechs_str}{stai_suffix}"


def generate_policy_function(cluster: SemanticCluster) -> str:
    """Generate a policy_stage1 function from a cluster."""
    mechs = cluster.mechanisms
    stai = cluster.stai_pattern
    freq = cluster.frequency
    participants = [inst.participant_id for inst in cluster.instances]
    
    # Get unique function name
    func_name = make_function_name(cluster)
    
    # Create docstring
    doc = f'"""\n    Policy Stage 1: {", ".join(sorted(mechs)) if mechs else "standard"}\n'
    if stai:
        doc += f'    STAI modulation: {stai}\n'
    doc += f'    Used by: {", ".join(participants[:5])}{"..." if len(participants) > 5 else ""} ({freq} total)\n'
    doc += '    """'
    
    # Generate function based on mechanisms
    if "model_based" in mechs and "perseveration" in mechs:
        return generate_mb_perseveration_policy(func_name, cluster, doc)
    elif "model_based" in mechs:
        return generate_mb_policy(func_name, cluster, doc)
    elif "win_stay" in mechs:
        return generate_win_stay_policy(func_name, cluster, doc)
    elif "perseveration" in mechs:
        return generate_perseveration_policy(func_name, cluster, doc)
    else:
        return generate_standard_policy(func_name, cluster, doc)


def generate_perseveration_policy(func_name: str, cluster: SemanticCluster, doc: str) -> str:
    """Generate a perseveration-based policy function."""
    stai = cluster.stai_pattern
    
    stai_modulation = ""
    if stai == "multiplicative":
        stai_modulation = "stai_multiplicative(bonus, stai)"
    elif stai == "inverse_linear":
        stai_modulation = "stai_inverse_linear(bonus, stai)"
    elif stai == "inverse_division":
        stai_modulation = "stai_inverse_division(bonus, stai)"
    elif stai == "additive_boost":
        stai_modulation = "stai_additive_boost(bonus, stai)"
    else:
        stai_modulation = "bonus"
    
    return f'''def {func_name}(
    q_stage1: np.ndarray,
    beta: float,
    bonus: float,
    stai: float,
    last_action: Optional[int]
) -> np.ndarray:
    {doc}
    q_modified = q_stage1.copy()
    
    if last_action is not None:
        effective_bonus = {stai_modulation}
        q_modified[last_action] += effective_bonus
    
    return softmax(q_modified, beta)
'''


def generate_win_stay_policy(func_name: str, cluster: SemanticCluster, doc: str) -> str:
    """Generate a win-stay policy function."""
    stai = cluster.stai_pattern
    
    return f'''def {func_name}(
    q_stage1: np.ndarray,
    beta: float,
    bonus: float,
    stai: float,
    last_action: Optional[int],
    last_reward: Optional[float]
) -> np.ndarray:
    {doc}
    q_modified = q_stage1.copy()
    
    if last_reward == 1.0 and last_action is not None:
        effective_bonus = stai * bonus  # Anxiety amplifies win-stay
        q_modified[last_action] += effective_bonus
    
    return softmax(q_modified, beta)
'''


def generate_mb_policy(func_name: str, cluster: SemanticCluster, doc: str) -> str:
    """Generate a model-based policy function."""
    return f'''def {func_name}(
    q_stage1: np.ndarray,
    q_stage2: np.ndarray,
    T: np.ndarray,
    beta: float,
    w: float,
    stai: float
) -> np.ndarray:
    {doc}
    # Compute model-based values
    q_mb = compute_mb_values(q_stage2, T)
    
    # Anxiety reduces model-based control
    w_eff = w * (1.0 - stai)
    
    # Mix MF and MB
    q_net = (1.0 - w_eff) * q_stage1 + w_eff * q_mb
    
    return softmax(q_net, beta)
'''


def generate_mb_perseveration_policy(func_name: str, cluster: SemanticCluster, doc: str) -> str:
    """Generate a model-based + perseveration policy function."""
    return f'''def {func_name}(
    q_stage1: np.ndarray,
    q_stage2: np.ndarray,
    T: np.ndarray,
    beta: float,
    w: float,
    pers: float,
    stai: float,
    last_action: Optional[int]
) -> np.ndarray:
    {doc}
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
'''


def generate_standard_policy(func_name: str, cluster: SemanticCluster, doc: str) -> str:
    """Generate a standard softmax policy function."""
    return f'''def {func_name}(
    q_stage1: np.ndarray,
    beta: float
) -> np.ndarray:
    {doc}
    return softmax(q_stage1, beta)
'''


def generate_value_update_function(cluster: SemanticCluster) -> str:
    """Generate a value update function from a cluster."""
    mechs = cluster.mechanisms
    freq = cluster.frequency
    participants = [inst.participant_id for inst in cluster.instances]
    func_name = make_function_name(cluster)
    
    doc = f'"""\n    Value Update: {", ".join(sorted(mechs)) if mechs else "standard TD"}\n'
    doc += f'    Used by: {", ".join(participants[:5])}{"..." if len(participants) > 5 else ""} ({freq} total)\n'
    doc += '    """'
    
    if "asymmetric_learning" in mechs:
        return generate_asymmetric_value_update(func_name, doc)
    else:
        return generate_standard_value_update(func_name, doc)


def generate_standard_value_update(func_name: str, doc: str) -> str:
    """Generate standard TD value update."""
    return f'''def {func_name}(
    q_stage1: np.ndarray,
    q_stage2: np.ndarray,
    alpha: float,
    action_1: int,
    state: int,
    action_2: int,
    reward: float
) -> Tuple[np.ndarray, np.ndarray]:
    {doc}
    # Stage 2 TD update
    delta_2 = reward - q_stage2[state, action_2]
    q_stage2 = q_stage2.copy()
    q_stage2[state, action_2] += alpha * delta_2
    
    # Stage 1 TD update
    delta_1 = q_stage2[state, action_2] - q_stage1[action_1]
    q_stage1 = q_stage1.copy()
    q_stage1[action_1] += alpha * delta_1
    
    return q_stage1, q_stage2
'''


def generate_asymmetric_value_update(func_name: str, doc: str) -> str:
    """Generate asymmetric learning value update."""
    return f'''def {func_name}(
    q_stage1: np.ndarray,
    q_stage2: np.ndarray,
    alpha_pos: float,
    alpha_neg: float,
    action_1: int,
    state: int,
    action_2: int,
    reward: float
) -> Tuple[np.ndarray, np.ndarray]:
    {doc}
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
'''


def generate_init_function(cluster: SemanticCluster) -> str:
    """Generate an init_model function."""
    stai = cluster.stai_pattern
    freq = cluster.frequency
    participants = [inst.participant_id for inst in cluster.instances]
    func_name = make_function_name(cluster)
    
    doc = f'    """Initialize model with STAI modulation: {stai or "none"}\n'
    doc += f'    Used by: {", ".join(participants)} ({freq} total)\n'
    doc += '    """'
    
    if stai == "inverse_division":
        return f'''def {func_name}(param: float, stai: float) -> float:
{doc}
    return param / (1.0 + stai)
'''
    elif stai == "multiplicative":
        return f'''def {func_name}(param: float, stai: float) -> float:
{doc}
    return param * stai
'''
    elif stai == "inverse_linear":
        return f'''def {func_name}(param: float, stai: float) -> float:
{doc}
    return param * (1.0 - stai)
'''
    else:
        return f'''def {func_name}(param: float, stai: float) -> float:
{doc}
    return param
'''


def generate_post_trial_function(cluster: SemanticCluster) -> str:
    """Generate a post_trial function."""
    mechs = cluster.mechanisms
    stai = cluster.stai_pattern
    freq = cluster.frequency
    participants = [inst.participant_id for inst in cluster.instances]
    func_name = make_function_name(cluster)
    
    doc = f'    """Post-trial processing: {", ".join(sorted(mechs)) if mechs else "standard"}\n'
    doc += f'    Used by: {", ".join(participants)} ({freq} total)\n'
    doc += '    """'
    
    if "memory_decay" in mechs:
        return f'''def {func_name}(
    q_stage1: np.ndarray,
    q_stage2: np.ndarray,
    decay_rate: float,
    action_1: int,
    state: int,
    action_2: int
) -> Tuple[np.ndarray, np.ndarray]:
{doc}
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
'''
    else:
        return f'''def {func_name}(
    last_action1: int,
    last_action2: int,
    last_state: int,
    last_reward: float,
    action_1: int,
    state: int,
    action_2: int,
    reward: float
) -> Tuple[int, int, int, float]:
{doc}
    return action_1, action_2, state, reward
'''


# ============================================================
# Analysis & Reporting
# ============================================================

def print_report(models: List[ExtractedModel], clusters: Dict[str, List[SemanticCluster]]):
    """Print analysis report."""
    print("\n" + "=" * 70)
    print("SEMANTIC LIBRARY LEARNING REPORT")
    print("=" * 70)
    
    print(f"\n📊 Models Analyzed: {len(models)}")
    
    # Mechanism frequency
    print("\n🧠 Cognitive Mechanisms Detected:")
    print("-" * 40)
    
    mech_counts = defaultdict(int)
    for model in models:
        for method in model.methods.values():
            for mech in method.mechanisms:
                mech_counts[mech] += 1
    
    for mech, count in sorted(mech_counts.items(), key=lambda x: -x[1]):
        bar = "█" * (count * 2) + "░" * (54 - count * 2)
        print(f"  {mech:20s} [{bar}] {count}")
    
    # STAI patterns
    print("\n📈 STAI Modulation Patterns:")
    print("-" * 40)
    
    stai_counts = defaultdict(int)
    for model in models:
        for method in model.methods.values():
            if method.stai_pattern:
                stai_counts[method.stai_pattern] += 1
    
    for pattern, count in sorted(stai_counts.items(), key=lambda x: -x[1]):
        print(f"  {pattern:20s}: {count}")
    
    # Semantic clusters
    print("\n🔍 Semantic Clusters:")
    print("-" * 40)
    
    for method_name in OVERRIDABLE_METHODS:
        if method_name not in clusters:
            continue
        
        print(f"\n  {method_name}:")
        for cluster in clusters[method_name][:3]:
            mechs = ", ".join(sorted(cluster.mechanisms)) if cluster.mechanisms else "standard"
            stai = f" [STAI: {cluster.stai_pattern}]" if cluster.stai_pattern else ""
            print(f"    • {mechs}{stai}: {cluster.frequency} participants")


def save_results(models: List[ExtractedModel], clusters: Dict[str, List[SemanticCluster]], output_dir: str):
    """Save results to files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model summary
    with open(os.path.join(output_dir, "model_summary.csv"), 'w') as f:
        f.write("participant,mechanisms,stai_patterns,parameters\n")
        for model in models:
            mechs = set()
            stai_pats = set()
            for m in model.methods.values():
                mechs.update(m.mechanisms)
                if m.stai_pattern:
                    stai_pats.add(m.stai_pattern)
            
            f.write(f"{model.participant_id},"
                    f"{'+'.join(sorted(mechs)) if mechs else 'standard'},"
                    f"{'+'.join(sorted(stai_pats)) if stai_pats else 'none'},"
                    f"{'+'.join(model.parameters)}\n")
    
    # Save cluster details
    with open(os.path.join(output_dir, "cluster_details.txt"), 'w') as f:
        f.write("Semantic Clusters\n")
        f.write("=" * 50 + "\n\n")
        
        for method_name in OVERRIDABLE_METHODS:
            if method_name not in clusters:
                continue
            
            f.write(f"\n## {method_name.upper()}\n")
            f.write("-" * 40 + "\n")
            
            for cluster in clusters[method_name]:
                f.write(f"\n### {cluster.signature}\n")
                f.write(f"Mechanisms: {cluster.mechanisms}\n")
                f.write(f"STAI: {cluster.stai_pattern or 'none'}\n")
                f.write(f"Frequency: {cluster.frequency}\n")
                f.write(f"Participants: {[i.participant_id for i in cluster.instances]}\n")
                f.write(f"\nCanonical code:\n```python\n{cluster.canonical_code()}\n```\n")
    
    print(f"\n📁 Results saved to {output_dir}/")


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("Semantic Library Learning for Class-Based Cognitive Models")
    print("=" * 70)
    
    print(f"\n📂 Loading models from {MODELS_DIR}")
    models = load_all_models(MODELS_DIR)
    
    if not models:
        print("No models found!")
        return
    
    print(f"\n✓ Loaded {len(models)} models")
    
    print("\n🔬 Clustering by semantic signatures...")
    clusters = cluster_methods(models)
    
    print_report(models, clusters)
    
    save_results(models, clusters, OUTPUT_DIR)
    
    print("\n📝 Generating cognitive library...")
    library_code = generate_library(clusters, models)
    
    library_path = os.path.join(OUTPUT_DIR, "cognitive_library.py")
    with open(library_path, 'w') as f:
        f.write(library_code)
    print(f"✓ Library saved to {library_path}")
    
    print("\n" + "=" * 70)
    print("Library learning complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
