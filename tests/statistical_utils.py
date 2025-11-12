"""
Statistical utility functions for TR 38.901 channel model validation.

This module provides reusable statistical testing and analysis functions
for validating Large-Scale Parameters (LSPs) and cluster-level statistics.
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional
import math


def calculate_lsp_statistics(lsp_samples: List, param_name: str) -> Dict[str, float]:
    """
    Calculate basic statistics for a given LSP parameter in log10 domain.

    Args:
        lsp_samples: List of LargeScaleParameters objects
        param_name: Name of the parameter ('DS', 'ASD', 'ASA', 'ZSA', 'SF', 'K')

    Returns:
        Dictionary with keys: 'mean', 'std', 'min', 'max', 'median'
    """
    # Extract parameter values
    if param_name == 'DS':
        values = [lsp.delay_spread_s for lsp in lsp_samples]
    elif param_name == 'ASD':
        values = [lsp.departure_azimuth_spread_deg for lsp in lsp_samples]
    elif param_name == 'ASA':
        values = [lsp.arrival_azimuth_spread_deg for lsp in lsp_samples]
    elif param_name == 'ZSA':
        values = [lsp.arrival_zenith_spread_deg for lsp in lsp_samples]
    elif param_name == 'ZSD':
        values = [lsp.departure_zenith_spread_deg for lsp in lsp_samples]
    elif param_name == 'SF':
        values = [lsp.shadow_fade_db for lsp in lsp_samples]
    elif param_name == 'K':
        values = [lsp.k_factor_linear for lsp in lsp_samples]
    else:
        raise ValueError(f"Unknown parameter: {param_name}")

    # Convert to log10 domain (except SF which is already in dB)
    if param_name in ['DS', 'ASD', 'ASA', 'ZSA', 'ZSD', 'K']:
        log_values = np.log10(values)
    else:  # SF is already in dB
        log_values = np.array(values)

    return {
        'mean': float(np.mean(log_values)),
        'std': float(np.std(log_values, ddof=1)),  # Sample std with N-1
        'min': float(np.min(log_values)),
        'max': float(np.max(log_values)),
        'median': float(np.median(log_values))
    }


def get_expected_lsp_params(scenario: str, condition: str, fc_ghz: float = 3.5) -> Dict[str, Dict[str, float]]:
    """
    Get expected mean and std for LSPs based on TR 38.901 parameters.

    Args:
        scenario: Scenario name (e.g., 'UMa', 'UMi-StreetCanyon', 'RMa')
        condition: 'LOS' or 'NLOS'
        fc_ghz: Center frequency in GHz (default: 3.5)

    Returns:
        Dictionary mapping parameter names to {'mu': value, 'sigma': value}
    """
    log10_fc = math.log10(fc_ghz)

    # Parameter formulas from TR 38.901 Table 7.5-6
    params = {}

    if scenario == 'UMa':
        if condition == 'LOS':
            params['DS'] = {
                'mu': -7.067 - 0.0794 * log10_fc,
                'sigma': 0.57 + 0.026 * log10_fc
            }
            params['ASD'] = {'mu': 0.92, 'sigma': 0.31}
            params['ASA'] = {'mu': 1.76, 'sigma': 0.19}
            params['ZSA'] = {'mu': 0.96, 'sigma': 0.15}
            params['SF'] = {'mu': 0.0, 'sigma': 4.0}
            # K-factor: mu_log10_K=9.0dB means K_dB=9.0, so log10(K_linear)=K_dB/10=0.9
            params['K'] = {'mu': 9.0 / 10, 'sigma': 3.5 / 10}
        else:  # NLOS
            params['DS'] = {
                'mu': -6.47 - 0.134 * log10_fc,
                'sigma': 0.39
            }
            params['ASD'] = {'mu': 1.09, 'sigma': 0.44}
            params['ASA'] = {
                'mu': 2.04 - 0.25 * log10_fc,
                'sigma': 0.17 - 0.03 * log10_fc
            }
            params['ZSA'] = {
                'mu': -0.2856 * log10_fc + 1.445,
                'sigma': 0.17
            }
            params['SF'] = {'mu': 0.0, 'sigma': 6.0}
            # K-factor not applicable for NLOS

    elif scenario == 'UMi-StreetCanyon':
        if condition == 'LOS':
            params['DS'] = {
                'mu': -7.19 - 0.011 * log10_fc,
                'sigma': 0.14
            }
            params['ASD'] = {'mu': 1.20, 'sigma': 0.43}
            params['ASA'] = {'mu': 1.75, 'sigma': 0.19}
            params['ZSA'] = {'mu': 0.95, 'sigma': 0.14}
            params['SF'] = {'mu': 0.0, 'sigma': 4.0}
            # K-factor: mu_log10_K=9.0dB means K_dB=9.0, so log10(K_linear)=K_dB/10=0.9
            params['K'] = {'mu': 9.0 / 10, 'sigma': 5.0 / 10}
        else:  # NLOS
            params['DS'] = {
                'mu': -6.62 - 0.024 * log10_fc,
                'sigma': 0.32
            }
            params['ASD'] = {'mu': 1.41, 'sigma': 0.62}
            params['ASA'] = {
                'mu': 1.84 - 0.05 * log10_fc,
                'sigma': 0.15
            }
            params['ZSA'] = {
                'mu': -0.13 * log10_fc + 1.3,
                'sigma': 0.22
            }
            params['SF'] = {'mu': 0.0, 'sigma': 7.0}

    elif scenario == 'RMa':
        if condition == 'LOS':
            params['DS'] = {
                'mu': -7.49 + 0.055 * log10_fc,
                'sigma': 0.13
            }
            params['ASD'] = {'mu': 0.90, 'sigma': 0.38}
            params['ASA'] = {'mu': 1.52, 'sigma': 0.24}
            params['ZSA'] = {'mu': 0.47, 'sigma': 0.40}
            params['SF'] = {'mu': 0.0, 'sigma': 4.0}
            # K-factor: mu_log10_K=7.0dB means K_dB=7.0, so log10(K_linear)=K_dB/10=0.7
            params['K'] = {'mu': 7.0 / 10, 'sigma': 4.0 / 10}
        else:  # NLOS
            params['DS'] = {
                'mu': -7.43 + 0.0 * log10_fc,
                'sigma': 0.48
            }
            params['ASD'] = {'mu': 0.95, 'sigma': 0.45}
            params['ASA'] = {
                'mu': 1.52 - 0.08 * log10_fc,
                'sigma': 0.13
            }
            params['ZSA'] = {
                'mu': -0.19 * log10_fc + 1.363,
                'sigma': 0.43
            }
            params['SF'] = {'mu': 0.0, 'sigma': 8.0}

    return params


def test_mean_and_std(observed_stats: Dict[str, float],
                     expected_params: Dict[str, float],
                     param_name: str,
                     n_samples: int,
                     tolerance_mean: float = 0.3,
                     tolerance_std: float = 0.2) -> Dict[str, any]:
    """
    Test if observed mean and std are within tolerance of expected values.

    Args:
        observed_stats: Dictionary from calculate_lsp_statistics()
        expected_params: Dictionary with 'mu' and 'sigma' keys
        param_name: Parameter name for reporting
        n_samples: Number of samples used
        tolerance_mean: Absolute tolerance for mean error
        tolerance_std: Absolute tolerance for std error

    Returns:
        Dictionary with test results
    """
    obs_mean = observed_stats['mean']
    obs_std = observed_stats['std']
    exp_mean = expected_params['mu']
    exp_std = expected_params['sigma']

    mean_error = abs(obs_mean - exp_mean)
    std_error = abs(obs_std - exp_std)

    mean_pass = mean_error < tolerance_mean
    std_pass = std_error < tolerance_std

    # Calculate z-score for mean (more rigorous test)
    # Standard error of mean: σ/√n
    se_mean = exp_std / math.sqrt(n_samples)
    z_score_mean = (obs_mean - exp_mean) / se_mean if se_mean > 0 else 0

    return {
        'param_name': param_name,
        'n_samples': n_samples,
        'observed_mean': obs_mean,
        'expected_mean': exp_mean,
        'mean_error': mean_error,
        'mean_pass': mean_pass,
        'observed_std': obs_std,
        'expected_std': exp_std,
        'std_error': std_error,
        'std_pass': std_pass,
        'z_score_mean': z_score_mean,
        'overall_pass': mean_pass and std_pass
    }


def kolmogorov_smirnov_test(lsp_samples: List,
                           param_name: str,
                           expected_params: Dict[str, float],
                           alpha: float = 0.05) -> Dict[str, any]:
    """
    Perform Kolmogorov-Smirnov test for normality in log10 domain.

    Args:
        lsp_samples: List of LargeScaleParameters objects
        param_name: Parameter name
        expected_params: Dictionary with 'mu' and 'sigma' keys
        alpha: Significance level (default 0.05)

    Returns:
        Dictionary with test results
    """
    # Calculate statistics to get log values
    stats_dict = calculate_lsp_statistics(lsp_samples, param_name)

    # Extract values in log10 domain
    if param_name == 'DS':
        values = np.log10([lsp.delay_spread_s for lsp in lsp_samples])
    elif param_name == 'ASD':
        values = np.log10([lsp.departure_azimuth_spread_deg for lsp in lsp_samples])
    elif param_name == 'ASA':
        values = np.log10([lsp.arrival_azimuth_spread_deg for lsp in lsp_samples])
    elif param_name == 'ZSA':
        values = np.log10([lsp.arrival_zenith_spread_deg for lsp in lsp_samples])
    elif param_name == 'SF':
        values = np.array([lsp.shadow_fade_db for lsp in lsp_samples])
    elif param_name == 'K':
        values = np.log10([lsp.k_factor_linear for lsp in lsp_samples])

    # Perform K-S test
    mu = expected_params['mu']
    sigma = expected_params['sigma']
    statistic, p_value = stats.kstest(values, 'norm', args=(mu, sigma))

    return {
        'param_name': param_name,
        'test': 'Kolmogorov-Smirnov',
        'statistic': statistic,
        'p_value': p_value,
        'alpha': alpha,
        'pass': p_value > alpha,
        'n_samples': len(lsp_samples)
    }


def calculate_correlation_matrix(lsp_samples: List,
                                 param_names: List[str]) -> np.ndarray:
    """
    Calculate empirical correlation matrix for LSPs.

    Args:
        lsp_samples: List of LargeScaleParameters objects
        param_names: List of parameter names (e.g., ['DS', 'ASD', 'ASA', 'ZSA', 'SF', 'K'])

    Returns:
        Correlation matrix as numpy array
    """
    n_params = len(param_names)
    n_samples = len(lsp_samples)

    # Build data matrix: each row is a sample, each column is a parameter
    data_matrix = np.zeros((n_samples, n_params))

    for i, param_name in enumerate(param_names):
        stats_dict = calculate_lsp_statistics(lsp_samples, param_name)
        # Re-extract log values (inefficient but clear)
        if param_name == 'DS':
            values = np.log10([lsp.delay_spread_s for lsp in lsp_samples])
        elif param_name == 'ASD':
            values = np.log10([lsp.departure_azimuth_spread_deg for lsp in lsp_samples])
        elif param_name == 'ASA':
            values = np.log10([lsp.arrival_azimuth_spread_deg for lsp in lsp_samples])
        elif param_name == 'ZSA':
            values = np.log10([lsp.arrival_zenith_spread_deg for lsp in lsp_samples])
        elif param_name == 'SF':
            values = np.array([lsp.shadow_fade_db for lsp in lsp_samples])
        elif param_name == 'K':
            values = np.log10([lsp.k_factor_linear for lsp in lsp_samples])

        data_matrix[:, i] = values

    # Calculate correlation matrix
    corr_matrix = np.corrcoef(data_matrix, rowvar=False)

    return corr_matrix


def format_test_results(results: List[Dict]) -> str:
    """
    Format test results as a readable string.

    Args:
        results: List of test result dictionaries

    Returns:
        Formatted string
    """
    lines = []
    lines.append("=" * 80)
    lines.append("Statistical Validation Test Results")
    lines.append("=" * 80)

    for result in results:
        if 'mean_pass' in result:  # Mean/Std test
            status = "✓ PASS" if result['overall_pass'] else "✗ FAIL"
            lines.append(f"\n{result['param_name']} ({result['n_samples']} samples): {status}")
            lines.append(f"  Mean:   {result['observed_mean']:.4f} (expected: {result['expected_mean']:.4f}, error: {result['mean_error']:.4f})")
            lines.append(f"  Std:    {result['observed_std']:.4f} (expected: {result['expected_std']:.4f}, error: {result['std_error']:.4f})")
            lines.append(f"  Z-score: {result['z_score_mean']:.2f}")
        elif 'p_value' in result:  # Distribution test
            status = "✓ PASS" if result['pass'] else "✗ FAIL"
            lines.append(f"\n{result['param_name']} - {result['test']}: {status}")
            lines.append(f"  Statistic: {result['statistic']:.4f}")
            lines.append(f"  P-value:   {result['p_value']:.4f} (alpha: {result['alpha']})")

    lines.append("\n" + "=" * 80)

    return "\n".join(lines)
