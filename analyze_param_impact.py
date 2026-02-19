#!/usr/bin/env python3
"""
Analyze which hyperparameters significantly impact model performance.
Uses multiple statistical methods for robust inference:
  1. One-way ANOVA with multiple comparison correction (Bonferroni/Holm)
  2. N-way ANOVA with interaction terms
  3. Kruskal-Wallis non-parametric test
  4. Tukey HSD post-hoc pairwise comparisons
  5. Linear regression with standardized coefficients + VIF
  6. Permutation tests for distribution-free significance
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import re
from scipy import stats
from itertools import combinations
import warnings

# Optional imports with graceful degradation
try:
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    from statsmodels.stats.anova import anova_lm
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    warnings.warn("statsmodels not installed. N-way ANOVA, Tukey HSD, and VIF unavailable.")

try:
    from statsmodels.stats.multitest import multipletests
    HAS_MULTITEST = True
except ImportError:
    HAS_MULTITEST = False


# =============================================================================
# CONSTANTS
# =============================================================================
PARAMS = ['tau', 'gamma', 'alpha', 'beta']
SIGNIFICANCE_LEVELS = {'*': 0.05, '**': 0.01, '***': 0.001}
N_PERMUTATIONS = 10000


# =============================================================================
# DATA LOADING & PARSING
# =============================================================================
def parse_params(exp_name: str) -> dict:
    """Extract tau, gamma, alpha, beta from experiment name."""
    pattern = r'tau([\d.]+)_gamma([\d.]+)_alpha([\d.]+)_beta([\d.]+)'
    match = re.search(pattern, exp_name)
    if match:
        return {
            'tau': float(match.group(1)),
            'gamma': float(match.group(2)),
            'alpha': float(match.group(3)),
            'beta': float(match.group(4))
        }
    return {'tau': np.nan, 'gamma': np.nan, 'alpha': np.nan, 'beta': np.nan}


def load_and_prepare_data(csv_path: Path) -> pd.DataFrame:
    """Load CSV, parse parameters, extract converged (last epoch) performance."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Data file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    required_cols = ['experiment', 'epoch', 'mean_reward']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Parse parameters
    params_df = df['experiment'].apply(parse_params).apply(pd.Series)
    df = pd.concat([df, params_df], axis=1)
    
    # Check for failed parses
    nan_count = df[PARAMS].isna().any(axis=1).sum()
    if nan_count > 0:
        warnings.warn(f"{nan_count} experiments failed parameter parsing. Dropping them.")
        df = df.dropna(subset=PARAMS)
    
    if len(df) == 0:
        raise ValueError("No valid experiments after parsing parameters.")
    
    # Get converged performance: last epoch per experiment
    converged = df.loc[df.groupby('experiment')['epoch'].idxmax()].copy()
    
    return converged


def get_significance_stars(p_value: float) -> str:
    """Return significance stars based on p-value."""
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    return ""


# =============================================================================
# 1. ONE-WAY ANOVA WITH MULTIPLE COMPARISON CORRECTION
# =============================================================================
def run_oneway_anova(df: pd.DataFrame, target: str = 'mean_reward') -> pd.DataFrame:
    """One-way ANOVA per parameter with Bonferroni and Holm correction."""
    print("\n" + "=" * 70)
    print("1. ONE-WAY ANOVA WITH MULTIPLE COMPARISON CORRECTION")
    print("=" * 70)
    
    results = []
    raw_p_values = []
    
    for param in PARAMS:
        unique_vals = sorted(df[param].unique())
        if len(unique_vals) < 2:
            print(f"\n{param}: Only one value ({unique_vals}), skipping")
            continue
        
        groups = [df[df[param] == v][target].values for v in unique_vals]
        
        # Check minimum sample size
        min_size = min(len(g) for g in groups)
        if min_size < 2:
            print(f"\n{param}: Group with <2 samples, skipping")
            continue
        
        f_stat, p_value = stats.f_oneway(*groups)
        
        # Effect size: eta-squared
        grand_mean = df[target].mean()
        ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
        ss_total = ((df[target] - grand_mean) ** 2).sum()
        eta_squared = ss_between / ss_total if ss_total > 0 else 0
        
        # Omega-squared (less biased)
        k = len(groups)
        n = len(df)
        ms_within = (ss_total - ss_between) / (n - k)
        omega_squared = (ss_between - (k - 1) * ms_within) / (ss_total + ms_within)
        omega_squared = max(0, omega_squared)  # Can be negative, floor at 0
        
        results.append({
            'param': param,
            'f_stat': f_stat,
            'p_value_raw': p_value,
            'eta_squared': eta_squared,
            'omega_squared': omega_squared,
            'n_groups': len(unique_vals),
            'group_sizes': [len(g) for g in groups]
        })
        raw_p_values.append(p_value)
    
    if not results:
        print("No parameters with sufficient variation for ANOVA.")
        return pd.DataFrame()
    
    # Apply multiple comparison corrections
    if HAS_MULTITEST and len(raw_p_values) > 1:
        _, p_bonferroni, _, _ = multipletests(raw_p_values, method='bonferroni')
        _, p_holm, _, _ = multipletests(raw_p_values, method='holm')
        for i, r in enumerate(results):
            r['p_bonferroni'] = p_bonferroni[i]
            r['p_holm'] = p_holm[i]
    else:
        # Manual Bonferroni
        n_tests = len(raw_p_values)
        for r in results:
            r['p_bonferroni'] = min(1.0, r['p_value_raw'] * n_tests)
            r['p_holm'] = r['p_bonferroni']  # Simplified fallback
    
    # Print results
    results_df = pd.DataFrame(results)
    for _, row in results_df.iterrows():
        sig_raw = get_significance_stars(row['p_value_raw'])
        sig_corr = get_significance_stars(row['p_holm'])
        print(f"\n{row['param'].upper()}")
        print(f"  F-statistic:     {row['f_stat']:.4f}")
        print(f"  p-value (raw):   {row['p_value_raw']:.6f} {sig_raw}")
        print(f"  p-value (Holm):  {row['p_holm']:.6f} {sig_corr}")
        print(f"  η² (eta-sq):     {row['eta_squared']:.4f}")
        print(f"  ω² (omega-sq):   {row['omega_squared']:.4f}")
        print(f"  Groups: {row['n_groups']}, sizes: {row['group_sizes']}")
    
    return results_df


# =============================================================================
# 2. N-WAY ANOVA WITH INTERACTIONS
# =============================================================================
def run_nway_anova(df: pd.DataFrame, target: str = 'mean_reward') -> Optional[pd.DataFrame]:
    """N-way ANOVA with all main effects and 2-way interactions."""
    print("\n" + "=" * 70)
    print("2. N-WAY ANOVA WITH INTERACTION TERMS")
    print("=" * 70)
    
    if not HAS_STATSMODELS:
        print("  [SKIPPED] statsmodels not installed.")
        return None
    
    # Build formula: main effects + 2-way interactions
    # Treat params as categorical for proper ANOVA
    for param in PARAMS:
        df[f'{param}_cat'] = df[param].astype(str)
    
    cat_params = [f'C({p}_cat)' for p in PARAMS]
    main_effects = ' + '.join(cat_params)
    
    # 2-way interactions
    interactions = ' + '.join([f'{cat_params[i]}:{cat_params[j]}' 
                               for i, j in combinations(range(len(cat_params)), 2)])
    
    formula = f'{target} ~ {main_effects} + {interactions}'
    
    try:
        model = ols(formula, data=df).fit()
        anova_table = anova_lm(model, typ=2)
        
        print(f"\nFormula: {target} ~ main effects + 2-way interactions")
        print(f"R² = {model.rsquared:.4f}, Adj R² = {model.rsquared_adj:.4f}")
        print(f"\nANOVA Table (Type II SS):")
        print("-" * 70)
        
        # Format output
        for idx, row in anova_table.iterrows():
            if idx == 'Residual':
                continue
            # Clean up index name
            name = idx.replace('C(', '').replace('_cat)', '').replace(')', '')
            p_val = row['PR(>F)']
            sig = get_significance_stars(p_val) if not np.isnan(p_val) else ''
            print(f"  {name:30} F={row['F']:8.3f}  p={p_val:.6f} {sig}")
        
        return anova_table
    
    except Exception as e:
        print(f"  [ERROR] N-way ANOVA failed: {e}")
        return None


# =============================================================================
# 3. KRUSKAL-WALLIS NON-PARAMETRIC TEST
# =============================================================================
def run_kruskal_wallis(df: pd.DataFrame, target: str = 'mean_reward') -> pd.DataFrame:
    """Kruskal-Wallis H-test (non-parametric alternative to ANOVA)."""
    print("\n" + "=" * 70)
    print("3. KRUSKAL-WALLIS H-TEST (Non-parametric)")
    print("=" * 70)
    print("  Use when ANOVA assumptions (normality, homoscedasticity) are violated.\n")
    
    results = []
    
    for param in PARAMS:
        unique_vals = sorted(df[param].unique())
        if len(unique_vals) < 2:
            continue
        
        groups = [df[df[param] == v][target].values for v in unique_vals]
        if any(len(g) < 2 for g in groups):
            continue
        
        h_stat, p_value = stats.kruskal(*groups)
        
        # Effect size: epsilon-squared (η²H)
        n = len(df)
        epsilon_sq = (h_stat - len(groups) + 1) / (n - len(groups))
        epsilon_sq = max(0, epsilon_sq)
        
        results.append({
            'param': param,
            'h_stat': h_stat,
            'p_value': p_value,
            'epsilon_squared': epsilon_sq,
            'n_groups': len(unique_vals)
        })
        
        sig = get_significance_stars(p_value)
        print(f"  {param:10} H={h_stat:8.3f}  p={p_value:.6f} {sig}  ε²={epsilon_sq:.4f}")
    
    return pd.DataFrame(results)


# =============================================================================
# 4. TUKEY HSD POST-HOC COMPARISONS
# =============================================================================
def run_tukey_hsd(df: pd.DataFrame, target: str = 'mean_reward') -> None:
    """Tukey HSD for pairwise comparisons within significant parameters."""
    print("\n" + "=" * 70)
    print("4. TUKEY HSD POST-HOC PAIRWISE COMPARISONS")
    print("=" * 70)
    
    if not HAS_STATSMODELS:
        print("  [SKIPPED] statsmodels not installed.")
        return
    
    for param in PARAMS:
        unique_vals = sorted(df[param].unique())
        if len(unique_vals) < 3:
            print(f"\n  {param}: <3 levels, Tukey HSD not meaningful.")
            continue
        
        # First check if ANOVA is significant
        groups = [df[df[param] == v][target].values for v in unique_vals]
        _, p_anova = stats.f_oneway(*groups)
        
        print(f"\n  {param.upper()} (ANOVA p={p_anova:.6f})")
        
        if p_anova >= 0.05:
            print("    ANOVA not significant, post-hoc not warranted.")
            continue
        
        tukey = pairwise_tukeyhsd(df[target], df[param].astype(str), alpha=0.05)
        
        print("    Pairwise comparisons:")
        for i in range(len(tukey.summary().data) - 1):
            row = tukey.summary().data[i + 1]
            g1, g2, meandiff, p_adj, lower, upper, reject = row
            sig = "SIGNIFICANT" if reject else ""
            print(f"      {g1} vs {g2}: diff={float(meandiff):+.3f}, p={float(p_adj):.4f} {sig}")


# =============================================================================
# 5. LINEAR REGRESSION WITH STANDARDIZED COEFFICIENTS + VIF
# =============================================================================
def run_linear_regression(df: pd.DataFrame, target: str = 'mean_reward') -> Optional[pd.DataFrame]:
    """OLS regression with standardized coefficients and VIF for multicollinearity."""
    print("\n" + "=" * 70)
    print("5. LINEAR REGRESSION (Standardized Coefficients + VIF)")
    print("=" * 70)
    
    if not HAS_STATSMODELS:
        print("  [SKIPPED] statsmodels not installed.")
        return None
    
    # Standardize predictors for comparable coefficients
    X = df[PARAMS].copy()
    X_standardized = (X - X.mean()) / X.std()
    X_standardized = sm.add_constant(X_standardized)
    
    y = df[target]
    
    model = sm.OLS(y, X_standardized).fit()
    
    print(f"\n  R² = {model.rsquared:.4f}, Adj R² = {model.rsquared_adj:.4f}")
    print(f"  F-statistic = {model.fvalue:.4f}, p = {model.f_pvalue:.6f}")
    
    print("\n  Standardized Coefficients (β):")
    print("  " + "-" * 50)
    
    coef_results = []
    for param in PARAMS:
        coef = model.params[param]
        pval = model.pvalues[param]
        sig = get_significance_stars(pval)
        
        # VIF
        X_vif = X[PARAMS]
        vif = variance_inflation_factor(X_vif.values, PARAMS.index(param))
        
        coef_results.append({
            'param': param,
            'beta': coef,
            'p_value': pval,
            'vif': vif
        })
        
        vif_warn = " [HIGH VIF!]" if vif > 5 else ""
        print(f"    {param:10} β={coef:+.4f}  p={pval:.6f} {sig:3}  VIF={vif:.2f}{vif_warn}")
    
    print("\n  Interpretation:")
    print("    - |β| indicates relative importance (standardized scale)")
    print("    - VIF > 5 suggests multicollinearity concern")
    print("    - VIF > 10 is severe multicollinearity")
    
    return pd.DataFrame(coef_results)


# =============================================================================
# 6. PERMUTATION TESTS
# =============================================================================
def run_permutation_tests(df: pd.DataFrame, target: str = 'mean_reward', 
                          n_permutations: int = N_PERMUTATIONS) -> pd.DataFrame:
    """Permutation tests for distribution-free significance."""
    print("\n" + "=" * 70)
    print(f"6. PERMUTATION TESTS ({n_permutations:,} permutations)")
    print("=" * 70)
    print("  Distribution-free test. No assumptions about data distribution.\n")
    
    rng = np.random.default_rng(42)  # Reproducibility
    results = []
    
    for param in PARAMS:
        unique_vals = sorted(df[param].unique())
        if len(unique_vals) < 2:
            continue
        
        groups = [df[df[param] == v][target].values for v in unique_vals]
        
        # Observed F-statistic
        f_observed, _ = stats.f_oneway(*groups)
        
        # Pooled data for permutation
        pooled = df[target].values.copy()
        group_sizes = [len(g) for g in groups]
        
        # Permutation distribution
        f_permuted = np.zeros(n_permutations)
        for i in range(n_permutations):
            rng.shuffle(pooled)
            perm_groups = []
            start = 0
            for size in group_sizes:
                perm_groups.append(pooled[start:start + size])
                start += size
            f_permuted[i], _ = stats.f_oneway(*perm_groups)
        
        # p-value: proportion of permuted F >= observed F
        p_value = (np.sum(f_permuted >= f_observed) + 1) / (n_permutations + 1)
        
        results.append({
            'param': param,
            'f_observed': f_observed,
            'p_value_perm': p_value,
            'f_perm_mean': np.mean(f_permuted),
            'f_perm_95': np.percentile(f_permuted, 95)
        })
        
        sig = get_significance_stars(p_value)
        print(f"  {param:10} F_obs={f_observed:8.3f}  p_perm={p_value:.6f} {sig}")
        print(f"             F_null_mean={np.mean(f_permuted):.3f}, F_null_95%={np.percentile(f_permuted, 95):.3f}")
    
    return pd.DataFrame(results)


# =============================================================================
# SUMMARY & CONSENSUS
# =============================================================================
def print_consensus_summary(
    anova_results: pd.DataFrame,
    kruskal_results: pd.DataFrame,
    regression_results: Optional[pd.DataFrame],
    permutation_results: pd.DataFrame
) -> None:
    """Synthesize findings across all methods."""
    print("\n" + "=" * 70)
    print("CONSENSUS SUMMARY: Parameter Impact Ranking")
    print("=" * 70)
    
    summary = {}
    for param in PARAMS:
        summary[param] = {
            'significant_count': 0,
            'total_tests': 0,
            'avg_effect': 0,
            'methods': []
        }
    
    # Count significant findings per method
    if not anova_results.empty:
        for _, row in anova_results.iterrows():
            param = row['param']
            summary[param]['total_tests'] += 1
            if row['p_holm'] < 0.05:
                summary[param]['significant_count'] += 1
                summary[param]['methods'].append('ANOVA')
            summary[param]['avg_effect'] += row['omega_squared']
    
    if not kruskal_results.empty:
        for _, row in kruskal_results.iterrows():
            param = row['param']
            summary[param]['total_tests'] += 1
            if row['p_value'] < 0.05:
                summary[param]['significant_count'] += 1
                summary[param]['methods'].append('Kruskal')
    
    if regression_results is not None and not regression_results.empty:
        for _, row in regression_results.iterrows():
            param = row['param']
            summary[param]['total_tests'] += 1
            if row['p_value'] < 0.05:
                summary[param]['significant_count'] += 1
                summary[param]['methods'].append('Regression')
            summary[param]['avg_effect'] += abs(row['beta'])
    
    if not permutation_results.empty:
        for _, row in permutation_results.iterrows():
            param = row['param']
            summary[param]['total_tests'] += 1
            if row['p_value_perm'] < 0.05:
                summary[param]['significant_count'] += 1
                summary[param]['methods'].append('Permutation')
    
    # Rank by consensus
    ranking = sorted(
        summary.items(),
        key=lambda x: (x[1]['significant_count'], x[1]['avg_effect']),
        reverse=True
    )
    
    print("\n  Param       Sig/Total   Methods Agreeing")
    print("  " + "-" * 55)
    for param, data in ranking:
        methods_str = ', '.join(data['methods']) if data['methods'] else 'None'
        rate = data['significant_count'] / max(data['total_tests'], 1)
        bar = '█' * int(rate * 10) + '░' * (10 - int(rate * 10))
        print(f"  {param:10} {data['significant_count']}/{data['total_tests']}  {bar}  {methods_str}")
    
    print("\n  Interpretation:")
    print("    - Parameters with high Sig/Total across methods are robust findings")
    print("    - Single-method significance may be spurious")
    print("    - Consensus across ANOVA, non-parametric, and regression is strong evidence")


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive hyperparameter impact analysis"
    )
    parser.add_argument(
        'csv_path',
        nargs='?',
        default='/home/hoang/python/inventory_control/all_experiments_analysis.csv',
        help='Path to experiments CSV file'
    )
    parser.add_argument(
        '--target',
        default='mean_reward',
        help='Target variable to analyze (default: mean_reward)'
    )
    parser.add_argument(
        '--permutations',
        type=int,
        default=N_PERMUTATIONS,
        help=f'Number of permutations for permutation test (default: {N_PERMUTATIONS})'
    )
    args = parser.parse_args()
    
    csv_path = Path(args.csv_path)
    
    print("=" * 70)
    print("COMPREHENSIVE HYPERPARAMETER IMPACT ANALYSIS")
    print("=" * 70)
    print(f"Data: {csv_path}")
    print(f"Target: {args.target}")
    
    # Load data
    try:
        df = load_and_prepare_data(csv_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Experiments: {len(df)}")
    print(f"\nParameter ranges:")
    for param in PARAMS:
        unique_vals = sorted(df[param].unique())
        print(f"  {param}: {unique_vals}")
    
    # Run all analyses
    anova_results = run_oneway_anova(df, args.target)
    nway_results = run_nway_anova(df, args.target)
    kruskal_results = run_kruskal_wallis(df, args.target)
    run_tukey_hsd(df, args.target)
    regression_results = run_linear_regression(df, args.target)
    permutation_results = run_permutation_tests(df, args.target, args.permutations)
    
    # Consensus summary
    print_consensus_summary(anova_results, kruskal_results, regression_results, permutation_results)
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
