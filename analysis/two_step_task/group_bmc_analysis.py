"""
Group Bayesian Model Comparison: Baseline vs GECCO Participant-Specific Models

This script performs group-level Bayesian model comparison between:
1. Baseline model: Hybrid Model-Based/Model-Free (MB/MF) model
2. GECCO models: Participant-specific cognitive models discovered by GECCO

References:
[1] Rigoux, L., Stephan, K. E., Friston, K. J., & Daunizeau, J. (2014).
    Bayesian model selection for group studies—revisited. NeuroImage, 84, 971-985.
[2] Stephan, K. E., Penny, W. D., Daunizeau, J., Moran, R. J., & Friston, K. J. (2009).
    Bayesian model selection for group studies. NeuroImage, 46(4), 1004-1017.
"""

import pandas as pd
import json
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from math import exp, log
from typing import List, Optional
from scipy import integrate
from scipy.stats import rv_continuous, dirichlet, multivariate_normal as mvn, pearsonr
from scipy.special import digamma as psi, gammainc, gammaln, softmax, expit

epsilon = np.finfo(float).eps


# ============================================================
# GROUP BMC IMPLEMENTATION
# ============================================================

def exceedance_probability(distribution, n_samples=None):
    """
    Calculates the exceedance probability of a random variable following 
    a continuous multivariate distribution.
    
    Exceedance probability: φ_i = p(∀j != i: x_i > x_j | x ~ distribution).
    
    Parameters
    ----------
    distribution : scipy.stats distribution
        The continuous multivariate distribution (e.g., Dirichlet).
    n_samples : int, optional
        Number of samples for Monte Carlo approximation. 
        If None, uses numerical integration.
    
    Returns
    -------
    np.ndarray
        Exceedance probabilities for each dimension.
    """
    if n_samples is None:
        from scipy.stats._multivariate import dirichlet_frozen, multivariate_normal_frozen
        if type(distribution) is multivariate_normal_frozen:
            distribution: multivariate_normal_frozen
            mu, Sigma = distribution.mean, distribution.cov
            n = len(mu)
            phi = np.zeros(n)
            I = -np.eye(n - 1)
            for i in range(n):
                A = np.insert(I, i, 1, axis=1)
                phi[i] = mvn.cdf(A @ mu, cov=A @ Sigma @ A.T)
        elif type(distribution) is dirichlet_frozen:
            alpha = distribution.alpha
            n = len(alpha)
            gamma_vals = [gammaln(alpha[i]) for i in range(n)]

            def f(x, i):
                phi_i = 1
                for j in range(n):
                    if i != j:
                        phi_i *= gammainc(alpha[j], x)
                return phi_i * exp((alpha[i] - 1) * log(x) - x - gamma_vals[i])
            
            phi = [integrate.quad(lambda x: f(x, i), 0, np.inf)[0] for i in range(n)]
        else:
            raise NotImplementedError('Numerical integration not implemented for this distribution!')
        phi = np.array(phi)
    else:
        samples = distribution.rvs(size=n_samples)
        phi = (samples == np.amax(samples, axis=1, keepdims=True)).sum(axis=0)
    return phi / phi.sum()


class GroupBMCResult:
    """Results of Bayesian model selection for group studies."""
    
    def __init__(self, alpha: np.ndarray, z: np.ndarray, bor: float):
        """
        Parameters
        ----------
        alpha : np.ndarray
            Sufficient statistics of the posterior Dirichlet density on model frequencies.
        z : np.ndarray
            Posterior probabilities for each subject to belong to each model.
        bor : float
            Bayesian omnibus risk p(y|H0)/(p(y|H0)+p(y|H1)).
        """
        self.attribution = z.copy()
        self.frequency_mean = dirichlet.mean(alpha)
        self.frequency_var = dirichlet.var(alpha)
        self.exceedance_probability = exceedance_probability(dirichlet(alpha))
        self.protected_exceedance_probability = self.exceedance_probability * (1 - bor) + bor / len(alpha)
        self.bor = bor
        self.alpha = alpha


class GroupBMC:
    """
    Variational Bayesian algorithm for group-level Bayesian Model Comparison.
    
    Based on Rigoux et al. (2014) NeuroImage.
    """
    
    def __init__(self,
                 L: np.ndarray,
                 alpha_0: Optional[np.ndarray] = None,
                 partitions: Optional[List[List[int]]] = None,
                 max_iter: int = 32,
                 min_iter: int = 1,
                 tolerance: float = 1e-4):
        """
        Uses variational Bayesian analysis to fit a Dirichlet distribution 
        on model frequencies to the data.
        
        Parameters
        ----------
        L : np.ndarray
            KxN array of the log-evidence of each of the K models given each of the N subjects.
        alpha_0 : np.ndarray, optional
            Kx1 array of sufficient statistics of the prior Dirichlet density.
        partitions : list of lists, optional
            Model family partitions.
        max_iter : int
            Maximum number of VB iterations.
        min_iter : int
            Minimum number of VB iterations.
        tolerance : float
            Convergence tolerance for free energy.
        """
        self.L = L
        K, N = L.shape
        partitions = [np.array([i]) for i in range(K)] if partitions is None else [np.array(p) - 1 for p in partitions]
        assert np.all(np.sort(np.concatenate(partitions)) == np.arange(K)), 'Invalid partition!'
        Nf = len(partitions)
        self.families = np.zeros((K, Nf), dtype=bool)
        for j in range(Nf):
            self.families[partitions[j], j] = True
        self.alpha_0 = (self.families / self.families.sum(axis=0) @ (np.ones(Nf) / Nf) if alpha_0 is None else alpha_0)[:, None]
        assert len(self.alpha_0) == K, 'Model evidence and priors size mismatch!'
        self.alpha, self.z = self.alpha_0.copy(), np.tile(self.alpha_0, (1, N))

        self.F = []
        for i in range(1, max_iter + 1):
            self.z = softmax(self.L + psi(self.alpha), axis=0)
            self.alpha = self.alpha_0 + self.z.sum(axis=1, keepdims=True)
            self.F.append(self.F1())
            if i > max(min_iter, 1) and abs(self.F[-1] - self.F[-2]) < tolerance:
                break

    def get_result(self) -> GroupBMCResult:
        """Get various statistics of the posterior Dirichlet distribution."""
        bor = 1 / (1 + exp(self.F1() - self.F0()))
        if self.families.size == 0:
            return GroupBMCResult(self.alpha.flatten(), self.z, bor)
        return GroupBMCResult(self.families.T @ self.alpha.flatten(), self.families.T @ self.z, bor)

    def F0(self) -> float:
        """Derives the free energy of the null hypothesis (H0: uniform priors)."""
        w = softmax(self.L, axis=0)
        return (w * (self.L + np.log(self.alpha_0) - np.log(w + epsilon))).sum()

    def F1(self) -> float:
        """Derives the free energy for the current approximate posteriors (H1)."""
        E_log_r = psi(self.alpha) - psi(self.alpha.sum())
        E_log_joint = (self.z * (self.L + E_log_r)).sum() + ((self.alpha_0 - 1) * E_log_r).sum()
        E_log_joint += gammaln(self.alpha_0.sum()) - gammaln(self.alpha_0).sum()
        entropy_z = -(self.z * np.log(self.z + epsilon)).sum()
        entropy_alpha = gammaln(self.alpha).sum() - gammaln(self.alpha.sum()) - ((self.alpha - 1) * E_log_r).sum()
        return E_log_joint + entropy_z + entropy_alpha


# ============================================================
# DATA LOADING FUNCTIONS
# ============================================================

def load_baseline_bics(data_path: str, participants: List[int]) -> dict:
    """Load baseline model BICs from the data file."""
    df = pd.read_csv(data_path)
    participant_baseline = df.groupby('participant')['baseline_bic'].first().reset_index()
    return {pid: participant_baseline[participant_baseline['participant'] == pid]['baseline_bic'].values[0] 
            for pid in participants}


def load_gecco_bics(bics_dir: str, participants: List[int]) -> dict:
    """Load best GECCO model BICs for each participant."""
    gecco_bics = {}
    for idx, pid in enumerate(participants):
        ## if you want plug in best bics directly
        # bics = [407.93073607039423, 558.1486246570571, 473.35230379814095, 462.8890419919733, 306.0607714913982, 225.60935154996739, 390.83602078476395, 364.287526315459, 305.42694661840176, 407.47321750089316, 490.3438439696416, 189.38688955636044, 240.03628882635164, 341.10382179045155, 395.49676864099183, 264.0658778772686, 495.6025468361587, 395.6375151193356, 337.65639784102973, 382.3917675248307, 372.41997516089623, 380.8885832003291, 450.3910162711118, 413.4644631000158, 378.40803666809813, 307.7907494370812, 409.7152809680055, 280.1907363756094, 254.8524137429026, 550.7165567441883, 417.32460251025975]
        # gecco_bics[pid] = bics[idx]
        pattern = os.path.join(bics_dir, f'iter*_participant{pid}.json')
        files = glob.glob(pattern)
        if files:
            best_bic = float('inf')
            for f in files:
                with open(f) as fp:
                    data = json.load(fp)
                    for model_result in data:
                        bic = model_result.get('metric_value', float('inf'))
                        if bic < best_bic:
                            best_bic = bic
            gecco_bics[pid] = best_bic
    return gecco_bics


def load_stai_scores(data_path: str, participants: List[int]) -> dict:
    """Load STAI scores for each participant."""
    df = pd.read_csv(data_path)
    participant_stai = df.groupby('participant')['stai'].first().reset_index()
    return {pid: participant_stai[participant_stai['participant'] == pid]['stai'].values[0] 
            for pid in participants}


# ============================================================
# ANALYSIS AND VISUALIZATION
# ============================================================

def run_group_bmc(baseline_bics: dict, gecco_bics: dict, participants: List[int]) -> GroupBMCResult:
    """
    Run Group Bayesian Model Comparison.
    
    Parameters
    ----------
    baseline_bics : dict
        Dictionary mapping participant ID to baseline BIC.
    gecco_bics : dict
        Dictionary mapping participant ID to GECCO BIC.
    participants : list
        List of participant IDs.
    
    Returns
    -------
    GroupBMCResult
        Results of the group BMC analysis.
    """
    n_participants = len(participants)
    
    # Model 0: Baseline, Model 1: GECCO
    # Convert BIC to log-evidence: log(p(y|m)) ≈ -BIC/2
    log_evidence = np.zeros((2, n_participants))
    
    for i, pid in enumerate(participants):
        log_evidence[0, i] = -baseline_bics[pid] / 2  # Baseline
        log_evidence[1, i] = -gecco_bics[pid] / 2     # GECCO
    
    bmc = GroupBMC(log_evidence)
    return bmc.get_result()


def print_participant_comparison(baseline_bics: dict, gecco_bics: dict, 
                                  participants: List[int]) -> tuple:
    """Print participant-level BIC comparison table."""
    print("=" * 70)
    print("PARTICIPANT-LEVEL MODEL COMPARISON: BASELINE vs GECCO")
    print("=" * 70)
    print('{:<12} {:<15} {:<15} {:<15} {:<10}'.format(
        'Participant', 'Baseline BIC', 'GECCO BIC', 'Difference', 'Better'))
    print("-" * 70)
    
    baseline_wins = 0
    gecco_wins = 0
    bic_diffs = []
    
    for pid in participants:
        baseline_bic = baseline_bics[pid]
        gecco_bic = gecco_bics.get(pid, float('inf'))
        diff = gecco_bic - baseline_bic
        better = 'GECCO' if gecco_bic < baseline_bic else 'Baseline'
        
        if gecco_bic < baseline_bic:
            gecco_wins += 1
        else:
            baseline_wins += 1
        
        bic_diffs.append(diff)
        print('{:<12} {:<15.2f} {:<15.2f} {:<15.2f} {:<10}'.format(
            f'p{pid}', baseline_bic, gecco_bic, diff, better))
    
    print("-" * 70)
    print(f'Summary: GECCO wins: {gecco_wins}, Baseline wins: {baseline_wins}')
    print(f'Total BIC improvement: {-sum(bic_diffs):.2f}')
    
    return bic_diffs, gecco_wins, baseline_wins


def print_bmc_results(result: GroupBMCResult, participants: List[int]):
    """Print Group BMC results."""
    print("\n" + "=" * 70)
    print("GROUP BAYESIAN MODEL COMPARISON RESULTS")
    print("=" * 70)
    
    print("\n1. POSTERIOR MODEL FREQUENCIES (Dirichlet distribution)")
    print("-" * 50)
    print(f"   Baseline model: Mean = {result.frequency_mean[0]:.4f}, Var = {result.frequency_var[0]:.6f}")
    print(f"   GECCO model:    Mean = {result.frequency_mean[1]:.4f}, Var = {result.frequency_var[1]:.6f}")
    
    print("\n2. EXCEEDANCE PROBABILITY")
    print("-" * 50)
    print(f"   P(Baseline is most frequent) = {result.exceedance_probability[0]:.4f}")
    print(f"   P(GECCO is most frequent)    = {result.exceedance_probability[1]:.4f}")
    
    print("\n3. PROTECTED EXCEEDANCE PROBABILITY")
    print("-" * 50)
    print(f"   Protected P(Baseline) = {result.protected_exceedance_probability[0]:.4f}")
    print(f"   Protected P(GECCO)    = {result.protected_exceedance_probability[1]:.4f}")
    
    print(f"\n4. BAYESIAN OMNIBUS RISK (BOR)")
    print("-" * 50)
    print(f"   BOR = {result.bor:.4f}")
    print(f"   (BOR < 0.05 indicates strong evidence for model differences)")
    
    print("\n5. POSTERIOR ATTRIBUTION (per participant)")
    print("-" * 50)
    print(f"   {'Participant':<12} {'P(Baseline)':<15} {'P(GECCO)':<15} {'Assigned':<10}")
    print("-" * 50)
    
    for i, pid in enumerate(participants):
        p_baseline = result.attribution[0, i]
        p_gecco = result.attribution[1, i]
        assigned = 'Baseline' if p_baseline > p_gecco else 'GECCO'
        print(f"   p{pid:<11} {p_baseline:<15.4f} {p_gecco:<15.4f} {assigned:<10}")
    
    n_gecco = np.sum(result.attribution[1, :] > 0.5)
    n_baseline = len(participants) - n_gecco
    print("-" * 50)
    print(f"   Total assigned to GECCO: {int(n_gecco)}/{len(participants)}")
    print(f"   Total assigned to Baseline: {int(n_baseline)}/{len(participants)}")


def create_visualization(baseline_bics: dict, gecco_bics: dict, 
                         stai_scores: dict, result: GroupBMCResult,
                         participants: List[int], output_path: str):
    """Create visualization of Group BMC results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    x = np.arange(len(participants))
    width = 0.35
    
    baseline_list = [baseline_bics[pid] for pid in participants]
    gecco_list = [gecco_bics.get(pid, np.nan) for pid in participants]
    #gecco_list  = plug in values
    stai_list = [stai_scores[pid] for pid in participants]
    bic_diff = [gecco_list[i] - baseline_list[i] for i in range(len(participants))]
    
    # 1. BIC Comparison Bar Plot
    ax1 = axes[0, 0]
    ax1.bar(x - width/2, baseline_list, width, label='Baseline (Hybrid MB/MF)', color='steelblue', alpha=0.8)
    ax1.bar(x + width/2, gecco_list, width, label='GECCO (Participant-Specific)', color='coral', alpha=0.8)
    ax1.set_xlabel('Participant')
    ax1.set_ylabel('BIC (lower is better)')
    ax1.set_title('Model Comparison: Baseline vs GECCO')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'p{p}' for p in participants], rotation=45, ha='right', fontsize=8)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. BIC Difference Plot
    ax2 = axes[0, 1]
    colors = ['green' if d < 0 else 'red' for d in bic_diff]
    ax2.bar(x, bic_diff, color=colors, alpha=0.8)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.axhline(y=-2, color='orange', linestyle='--', alpha=0.5, label='Positive evidence (ΔBIC=-2)')
    ax2.axhline(y=-6, color='red', linestyle='--', alpha=0.5, label='Strong evidence (ΔBIC=-6)')
    ax2.set_xlabel('Participant')
    ax2.set_ylabel('ΔBIC (GECCO - Baseline)')
    ax2.set_title('BIC Improvement (negative = GECCO better)')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'p{p}' for p in participants], rotation=45, ha='right', fontsize=8)
    ax2.legend(fontsize=8)
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Posterior Model Frequency Distribution
    ax3 = axes[1, 0]
    samples = dirichlet.rvs(result.alpha, size=10000)
    ax3.hist(samples[:, 1], bins=50, density=True, alpha=0.7, color='coral', edgecolor='black')
    ax3.axvline(x=result.frequency_mean[1], color='red', linestyle='--', 
                label=f'Mean = {result.frequency_mean[1]:.3f}')
    ax3.set_xlabel('GECCO Model Frequency')
    ax3.set_ylabel('Posterior Density')
    ax3.set_title('Posterior Distribution of GECCO Model Frequency')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # 4. BIC Improvement vs STAI
    ax4 = axes[1, 1]
    bic_improvement = [-d for d in bic_diff]
    ax4.scatter(stai_list, bic_improvement, c='coral', s=80, alpha=0.7, edgecolors='black')
    
    r, p = pearsonr(stai_list, bic_improvement)
    ax4.set_xlabel('STAI Score (Anxiety)')
    ax4.set_ylabel('BIC Improvement (Baseline - GECCO)')
    ax4.set_title(f'BIC Improvement vs Anxiety (r={r:.3f}, p={p:.3f})')
    ax4.grid(alpha=0.3)
    
    # Add trend line
    z = np.polyfit(stai_list, bic_improvement, 1)
    p_line = np.poly1d(z)
    x_line = np.linspace(min(stai_list), max(stai_list), 100)
    ax4.plot(x_line, p_line(x_line), 'r--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to: {output_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    """Run the complete Group BMC analysis."""
    # Configuration
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(base_dir, 'data', 'two_step_gillan_2016.csv')
    bics_dir = os.path.join(base_dir, 'results', 'two_step_psychiatry_individual_stai_class_individual', 'bics')
    output_dir = os.path.join(base_dir, 'results', 'two_step_psychiatry_individual_stai_class_individual')
    
    participants = list(range(14, 45))  # Participants 14-44
    
    # Load data
    print("Loading data...")
    baseline_bics = load_baseline_bics(data_path, participants)
    gecco_bics = load_gecco_bics(bics_dir, participants)
    stai_scores = load_stai_scores(data_path, participants)
    
    # Participant-level comparison
    bic_diffs, gecco_wins, baseline_wins = print_participant_comparison(
        baseline_bics, gecco_bics, participants)
    
    # Run Group BMC
    print("\nRunning Group Bayesian Model Comparison...")
    result = run_group_bmc(baseline_bics, gecco_bics, participants)
    
    # Print results
    print_bmc_results(result, participants)
    
    # Create visualization
    output_path = os.path.join(output_dir, 'group_bmc_results.png')
    create_visualization(baseline_bics, gecco_bics, stai_scores, result, 
                         participants, output_path)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total participants: {len(participants)}")
    print(f"GECCO wins at participant level: {gecco_wins}/{len(participants)}")
    print(f"Mean BIC improvement: {-np.mean(bic_diffs):.2f} (±{np.std(bic_diffs):.2f})")
    print(f"Posterior model frequency (GECCO): {result.frequency_mean[1]:.4f}")
    print(f"Exceedance probability (GECCO): {result.exceedance_probability[1]:.4f}")
    print(f"Protected exceedance probability (GECCO): {result.protected_exceedance_probability[1]:.4f}")
    print(f"Bayesian Omnibus Risk: {result.bor:.4f}")
    print(f'Mean BIC GECCO: {np.mean(list(gecco_bics.values()))} ± {np.std(list(gecco_bics.values()))/np.sqrt(len(gecco_bics))}')
    print(f'Mean BIC Baseline: {np.mean(list(baseline_bics.values()))} ± {np.std(list(baseline_bics.values()))/np.sqrt(len(baseline_bics))}')
    print("\nInterpretation:")
    if result.exceedance_probability[1] > 0.95:
        print("  → Decisive evidence that GECCO models outperform baseline")
    elif result.exceedance_probability[1] > 0.75:
        print("  → Strong evidence that GECCO models outperform baseline")
    else:
        print("  → Evidence is inconclusive")
    
    if result.bor < 0.05:
        print("  → BOR confirms models differ significantly at group level")


if __name__ == "__main__":
    main()
