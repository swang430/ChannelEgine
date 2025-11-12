"""
TR 38.901 Large-Scale Parameters (LSP) Statistical Validation Tests.

This module validates that generated LSPs conform to the statistical
distributions defined in 3GPP TR 38.901 specification.
"""

import pytest
import numpy as np
import os
import pickle
from typing import List, Dict
from channel_model_38901.simulator import ChannelSimulator
from channel_model_38901.common.data_models import LargeScaleParameters
from .statistical_utils import (
    calculate_lsp_statistics,
    get_expected_lsp_params,
    test_mean_and_std,
    kolmogorov_smirnov_test,
    format_test_results
)


# Test configuration
CACHE_DIR = os.path.join(os.path.dirname(__file__), 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)

# Sample sizes for different test tiers
QUICK_SAMPLES = 50      # Fast smoke test (~10 seconds)
STANDARD_SAMPLES = 200  # Standard validation (~1 minute)
COMPREHENSIVE_SAMPLES = 1000  # High confidence (~5 minutes)


def generate_lsp_samples(scenario: str,
                        condition: str,
                        num_samples: int,
                        fc_ghz: float = 3.5,
                        seed: int = 42) -> List[LargeScaleParameters]:
    """
    Generate LSP samples for a given scenario and condition.

    Args:
        scenario: Scenario name (e.g., 'UMa', 'UMi-StreetCanyon', 'RMa')
        condition: 'LOS' or 'NLOS'
        num_samples: Number of samples to generate
        fc_ghz: Center frequency in GHz
        seed: Random seed for reproducibility

    Returns:
        List of LargeScaleParameters objects
    """
    np.random.seed(seed)
    simulator = ChannelSimulator(
        scenario_name=scenario,
        center_frequency_hz=fc_ghz * 1e9,
        use_median_lsps=False
    )

    lsp_samples = []
    for i in range(num_samples):
        # Run simulation with logger disabled for speed
        results = simulator.run(force_condition=condition, logger=None)
        lsp_samples.append(results['lsps'])

    return lsp_samples


def get_cached_samples(scenario: str,
                      condition: str,
                      num_samples: int,
                      fc_ghz: float = 3.5,
                      seed: int = 42,
                      force_regenerate: bool = False) -> List[LargeScaleParameters]:
    """
    Get LSP samples from cache or generate if not cached.

    Args:
        scenario: Scenario name
        condition: 'LOS' or 'NLOS'
        num_samples: Number of samples
        fc_ghz: Center frequency in GHz
        seed: Random seed
        force_regenerate: Force regeneration even if cached

    Returns:
        List of LargeScaleParameters objects
    """
    cache_file = os.path.join(
        CACHE_DIR,
        f"{scenario}_{condition}_{num_samples}_fc{fc_ghz}_seed{seed}.pkl"
    )

    if os.path.exists(cache_file) and not force_regenerate:
        print(f"Loading cached samples from {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    print(f"Generating {num_samples} samples for {scenario} {condition}...")
    lsp_samples = generate_lsp_samples(scenario, condition, num_samples, fc_ghz, seed)

    # Cache the results
    with open(cache_file, 'wb') as f:
        pickle.dump(lsp_samples, f)

    return lsp_samples


# ============================================================================
# Pytest Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def uma_los_quick():
    """UMa LOS samples for quick testing (50 samples)."""
    return get_cached_samples('UMa', 'LOS', QUICK_SAMPLES)


@pytest.fixture(scope="module")
def uma_los_standard():
    """UMa LOS samples for standard testing (200 samples)."""
    return get_cached_samples('UMa', 'LOS', STANDARD_SAMPLES)


@pytest.fixture(scope="module")
def uma_los_comprehensive():
    """UMa LOS samples for comprehensive testing (1000 samples)."""
    return get_cached_samples('UMa', 'LOS', COMPREHENSIVE_SAMPLES)


@pytest.fixture(scope="module")
def uma_nlos_standard():
    """UMa NLOS samples for standard testing (200 samples)."""
    return get_cached_samples('UMa', 'NLOS', STANDARD_SAMPLES)


# ============================================================================
# Basic Mean and Std Tests
# ============================================================================

@pytest.mark.quick
@pytest.mark.parametrize("param_name", ['DS', 'ASD', 'ASA', 'ZSA', 'SF', 'K'])
def test_uma_los_mean_std_quick(uma_los_quick, param_name):
    """Quick test: UMa LOS mean and std validation (50 samples)."""
    lsp_samples = uma_los_quick
    n_samples = len(lsp_samples)

    # Get observed statistics
    obs_stats = calculate_lsp_statistics(lsp_samples, param_name)

    # Get expected parameters
    expected_params_dict = get_expected_lsp_params('UMa', 'LOS', fc_ghz=3.5)
    expected_params = expected_params_dict[param_name]

    # Run test with relaxed tolerances for quick test
    result = test_mean_and_std(
        obs_stats,
        expected_params,
        param_name,
        n_samples,
        tolerance_mean=0.5,  # Relaxed for small sample
        tolerance_std=0.3
    )

    # Print results
    print(f"\n{param_name}: mean={result['observed_mean']:.3f} (exp: {result['expected_mean']:.3f}), "
          f"std={result['observed_std']:.3f} (exp: {result['expected_std']:.3f})")

    # Assert pass
    assert result['overall_pass'], (
        f"{param_name} failed: mean_error={result['mean_error']:.3f}, "
        f"std_error={result['std_error']:.3f}"
    )


@pytest.mark.standard
@pytest.mark.parametrize("param_name", ['DS', 'ASD', 'ASA', 'ZSA', 'SF', 'K'])
def test_uma_los_mean_std_standard(uma_los_standard, param_name):
    """Standard test: UMa LOS mean and std validation (200 samples)."""
    lsp_samples = uma_los_standard
    n_samples = len(lsp_samples)

    # Get observed statistics
    obs_stats = calculate_lsp_statistics(lsp_samples, param_name)

    # Get expected parameters
    expected_params_dict = get_expected_lsp_params('UMa', 'LOS', fc_ghz=3.5)
    expected_params = expected_params_dict[param_name]

    # Run test with standard tolerances
    result = test_mean_and_std(
        obs_stats,
        expected_params,
        param_name,
        n_samples,
        tolerance_mean=0.3,
        tolerance_std=0.2
    )

    # Print results
    print(f"\n{param_name}: mean={result['observed_mean']:.3f} (exp: {result['expected_mean']:.3f}), "
          f"std={result['observed_std']:.3f} (exp: {result['expected_std']:.3f}), "
          f"z-score={result['z_score_mean']:.2f}")

    # Assert pass
    assert result['overall_pass'], (
        f"{param_name} failed: mean_error={result['mean_error']:.3f}, "
        f"std_error={result['std_error']:.3f}, z-score={result['z_score_mean']:.2f}"
    )


@pytest.mark.standard
@pytest.mark.parametrize("param_name", ['DS', 'ASD', 'ASA', 'ZSA', 'SF'])
def test_uma_nlos_mean_std_standard(uma_nlos_standard, param_name):
    """Standard test: UMa NLOS mean and std validation (200 samples)."""
    lsp_samples = uma_nlos_standard
    n_samples = len(lsp_samples)

    # Get observed statistics
    obs_stats = calculate_lsp_statistics(lsp_samples, param_name)

    # Get expected parameters
    expected_params_dict = get_expected_lsp_params('UMa', 'NLOS', fc_ghz=3.5)
    expected_params = expected_params_dict[param_name]

    # Run test
    result = test_mean_and_std(
        obs_stats,
        expected_params,
        param_name,
        n_samples,
        tolerance_mean=0.3,
        tolerance_std=0.2
    )

    # Print results
    print(f"\n{param_name}: mean={result['observed_mean']:.3f} (exp: {result['expected_mean']:.3f}), "
          f"std={result['observed_std']:.3f} (exp: {result['expected_std']:.3f})")

    # Assert pass
    assert result['overall_pass'], (
        f"{param_name} failed: mean_error={result['mean_error']:.3f}, "
        f"std_error={result['std_error']:.3f}"
    )


# ============================================================================
# Distribution Shape Tests (K-S Test)
# ============================================================================

@pytest.mark.standard
@pytest.mark.parametrize("param_name", ['DS', 'ASA', 'SF'])
def test_uma_los_distribution_ks(uma_los_standard, param_name):
    """K-S test: UMa LOS distribution validation (200 samples)."""
    lsp_samples = uma_los_standard

    # Get expected parameters
    expected_params_dict = get_expected_lsp_params('UMa', 'LOS', fc_ghz=3.5)
    expected_params = expected_params_dict[param_name]

    # Run K-S test
    result = kolmogorov_smirnov_test(lsp_samples, param_name, expected_params, alpha=0.05)

    # Print results
    print(f"\n{param_name} K-S Test: statistic={result['statistic']:.4f}, "
          f"p-value={result['p_value']:.4f}")

    # Assert pass (p-value > alpha means we cannot reject normality)
    assert result['pass'], (
        f"{param_name} K-S test failed: p-value={result['p_value']:.4f} < {result['alpha']}"
    )


@pytest.mark.comprehensive
@pytest.mark.parametrize("param_name", ['DS', 'ASD', 'ASA', 'ZSA', 'SF', 'K'])
def test_uma_los_mean_std_comprehensive(uma_los_comprehensive, param_name):
    """Comprehensive test: UMa LOS with 1000 samples."""
    lsp_samples = uma_los_comprehensive
    n_samples = len(lsp_samples)

    # Get observed statistics
    obs_stats = calculate_lsp_statistics(lsp_samples, param_name)

    # Get expected parameters
    expected_params_dict = get_expected_lsp_params('UMa', 'LOS', fc_ghz=3.5)
    expected_params = expected_params_dict[param_name]

    # Run test with strict tolerances for large sample
    result = test_mean_and_std(
        obs_stats,
        expected_params,
        param_name,
        n_samples,
        tolerance_mean=0.2,
        tolerance_std=0.15
    )

    # Print results
    print(f"\n{param_name}: mean={result['observed_mean']:.3f} (exp: {result['expected_mean']:.3f}), "
          f"std={result['observed_std']:.3f} (exp: {result['expected_std']:.3f}), "
          f"z-score={result['z_score_mean']:.2f}")

    # Assert pass
    assert result['overall_pass'], (
        f"{param_name} failed: mean_error={result['mean_error']:.3f}, "
        f"std_error={result['std_error']:.3f}"
    )


# ============================================================================
# Test Report Generation
# ============================================================================

@pytest.mark.standard
def test_generate_summary_report_uma_los(uma_los_standard):
    """Generate summary report for UMa LOS validation."""
    lsp_samples = uma_los_standard
    n_samples = len(lsp_samples)

    param_names = ['DS', 'ASD', 'ASA', 'ZSA', 'SF', 'K']
    expected_params_dict = get_expected_lsp_params('UMa', 'LOS', fc_ghz=3.5)

    results = []
    for param_name in param_names:
        obs_stats = calculate_lsp_statistics(lsp_samples, param_name)
        result = test_mean_and_std(
            obs_stats,
            expected_params_dict[param_name],
            param_name,
            n_samples,
            tolerance_mean=0.3,
            tolerance_std=0.2
        )
        results.append(result)

    # Format and print report
    report = format_test_results(results)
    print("\n" + report)

    # Check overall pass rate
    passed = sum(1 for r in results if r['overall_pass'])
    pass_rate = passed / len(results)

    print(f"\nOverall Pass Rate: {passed}/{len(results)} ({pass_rate*100:.1f}%)")

    # We expect at least 80% to pass
    assert pass_rate >= 0.80, f"Pass rate {pass_rate*100:.1f}% below 80% threshold"
