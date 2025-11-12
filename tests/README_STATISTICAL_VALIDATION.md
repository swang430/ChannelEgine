# TR 38.901 Statistical Validation Tests

## Overview

This test suite validates that the channel model implementation conforms to the statistical distributions defined in 3GPP TR 38.901 specification.

## Test Tiers

### Quick Tests (50 samples, ~10 seconds)
```bash
pytest tests/test_lsp_statistics.py -m quick
```
- **Purpose**: Fast smoke test to catch gross errors
- **Tolerance**: Relaxed (mean ±0.5, std ±0.3)
- **Use case**: After code changes, before commit

### Standard Tests (200 samples, ~1 minute)
```bash
pytest tests/test_lsp_statistics.py -m standard
```
- **Purpose**: Reasonable confidence validation
- **Tolerance**: Standard (mean ±0.3, std ±0.2)
- **Use case**: Before merging PR

### Comprehensive Tests (1000 samples, ~5 minutes)
```bash
pytest tests/test_lsp_statistics.py -m comprehensive
```
- **Purpose**: High confidence validation
- **Tolerance**: Strict (mean ±0.2, std ±0.15)
- **Use case**: Release validation, nightly CI

## What is Tested

### Large-Scale Parameters (LSPs)
- **DS** (Delay Spread): log10 domain mean and std
- **ASD** (Departure Azimuth Spread): log10 domain mean and std
- **ASA** (Arrival Azimuth Spread): log10 domain mean and std
- **ZSA** (Arrival Zenith Spread): log10 domain mean and std
- **SF** (Shadow Fading): dB domain mean and std
- **K** (K-factor): log10(K_linear) domain mean and std

### Statistical Tests
1. **Mean validation**: t-test against expected mean
2. **Std validation**: Levene's test against expected std
3. **Distribution shape**: Kolmogorov-Smirnov test for normality
4. **Correlation matrix**: Pearson correlation validation (future)

## Scenarios Tested

### Currently Implemented
- UMa LOS (Urban Macro, Line-of-Sight)
- UMa NLOS (Urban Macro, Non-Line-of-Sight)

### Planned
- UMi-StreetCanyon LOS/NLOS
- RMa LOS/NLOS (Rural Macro)
- InH-Office LOS/NLOS (Indoor Hotspot)
- SMa LOS/NLOS (Suburban Macro)
- InF LOS/NLOS (Indoor Factory)

## Caching

Samples are cached in `tests/cache/` to avoid regeneration:
```
tests/cache/UMa_LOS_50_fc3.5_seed42.pkl
tests/cache/UMa_LOS_200_fc3.5_seed42.pkl
tests/cache/UMa_LOS_1000_fc3.5_seed42.pkl
```

To force regeneration, delete the cache files.

## Expected Results

### Pass Criteria (200 samples)
- Mean error < 0.3 (in log10 domain)
- Std error < 0.2 (in log10 domain)
- K-S test p-value > 0.05

### Known Issues
- SF (Shadow Fading) may fail quick tests (50 samples) due to large variance (σ=4.0 dB)
- This is expected statistical variation; use standard tests (200+ samples) for reliable validation

## Implementation Details

### K-factor Conversion
TR 38.901 defines `mu_log10_K` in dB units:
- `mu_log10_K = 9.0` means K_dB = 9.0 dB
- K_linear = 10^(K_dB/10) = 10^0.9 ≈ 7.94
- log10(K_linear) = 0.9 (this is what we test)

### File Structure
```
tests/
├── statistical_utils.py       # Reusable statistical functions
├── test_lsp_statistics.py     # Main LSP validation tests
├── cache/                     # Sample cache directory
└── README_STATISTICAL_VALIDATION.md  # This file
```

## Adding New Scenarios

1. Add expected parameters to `statistical_utils.get_expected_lsp_params()`
2. Create pytest fixtures in `test_lsp_statistics.py`
3. Add parametrized tests for the new scenario

## References

- 3GPP TR 38.901 V19.0.0: "Study on channel model for frequencies from 0.5 to 100 GHz"
- Table 7.5-6: Large-scale parameters for various scenarios
