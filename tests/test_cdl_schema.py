"""Tests for mimo_ota_simulator.cdl_schema (Phase 5 T5.1 / T5.2)."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from pydantic import ValidationError

from mimo_ota_simulator.cdl_schema import CDLClusterSpec, CustomCDLProfile


VALID_CLUSTER = {
    'delay_s': 0.0,
    'power_linear': 0.7,
    'aoa_deg': 10.0,
    'aod_deg': -15.0,
    'as_aoa_deg': 5.0,
    'as_aod_deg': 2.0,
}

VALID_PROFILE = {
    'center_frequency_hz': 2.6e9,
    'pathloss_db': 85.0,
    'ue_velocity_mps': [4.17, 0.0, 0.0],
    'is_los': True,
    'clusters': [
        VALID_CLUSTER,
        {**VALID_CLUSTER, 'delay_s': 2e-7, 'power_linear': 0.3, 'aoa_deg': -45.0},
    ],
}


class TestCDLClusterSpec(unittest.TestCase):
    """Field-level validation for one cluster."""

    def test_minimal_required_fields_pass(self):
        c = CDLClusterSpec(**VALID_CLUSTER)
        self.assertEqual(c.delay_s, 0.0)
        self.assertEqual(c.power_linear, 0.7)
        # Defaults
        self.assertEqual(c.zoa_deg, 90.0)
        self.assertEqual(c.zod_deg, 90.0)
        self.assertEqual(c.num_rays, 20)
        self.assertIsNone(c.xpr_db)
        self.assertIsNone(c.initial_phases_rad)

    def test_full_field_set_pass(self):
        full = {
            **VALID_CLUSTER,
            'zoa_deg': 95.0, 'zod_deg': 85.0,
            'as_zoa_deg': 1.5, 'as_zod_deg': 1.0,
            'xpr_db': 7.0,
            'initial_phases_rad': [0.3, -1.2, 0.5, 2.1],
            'num_rays': 30,
        }
        c = CDLClusterSpec(**full)
        self.assertEqual(c.xpr_db, 7.0)
        self.assertEqual(c.initial_phases_rad, [0.3, -1.2, 0.5, 2.1])
        self.assertEqual(c.num_rays, 30)

    def test_negative_delay_rejected(self):
        with self.assertRaises(ValidationError):
            CDLClusterSpec(**{**VALID_CLUSTER, 'delay_s': -1e-9})

    def test_zero_power_rejected(self):
        with self.assertRaises(ValidationError):
            CDLClusterSpec(**{**VALID_CLUSTER, 'power_linear': 0.0})

    def test_negative_as_rejected(self):
        with self.assertRaises(ValidationError):
            CDLClusterSpec(**{**VALID_CLUSTER, 'as_aoa_deg': -1.0})

    def test_initial_phases_wrong_length_rejected(self):
        with self.assertRaises(ValidationError):
            CDLClusterSpec(**{**VALID_CLUSTER, 'initial_phases_rad': [0.0, 0.0, 0.0]})
        with self.assertRaises(ValidationError):
            CDLClusterSpec(**{**VALID_CLUSTER, 'initial_phases_rad': [0.0] * 5})

    def test_num_rays_must_be_positive(self):
        with self.assertRaises(ValidationError):
            CDLClusterSpec(**{**VALID_CLUSTER, 'num_rays': 0})

    def test_extra_field_rejected(self):
        """extra='forbid' catches typos like 'pwr_linear'."""
        with self.assertRaises(ValidationError):
            CDLClusterSpec(**{**VALID_CLUSTER, 'pwr_linear': 0.5})


class TestCustomCDLProfile(unittest.TestCase):
    """Top-level profile schema."""

    def test_valid_profile(self):
        p = CustomCDLProfile.from_dict(VALID_PROFILE)
        self.assertEqual(p.center_frequency_hz, 2.6e9)
        self.assertEqual(p.ue_velocity_mps, (4.17, 0.0, 0.0))
        self.assertEqual(len(p.clusters), 2)
        self.assertIsNone(p.k_factor_db)

    def test_velocity_accepts_list(self):
        p = CustomCDLProfile.from_dict(VALID_PROFILE)
        self.assertIsInstance(p.ue_velocity_mps, tuple)

    def test_velocity_wrong_length_rejected(self):
        bad = {**VALID_PROFILE, 'ue_velocity_mps': [1.0, 2.0]}
        with self.assertRaises(ValidationError):
            CustomCDLProfile.from_dict(bad)

    def test_velocity_non_numeric_rejected(self):
        bad = {**VALID_PROFILE, 'ue_velocity_mps': ['a', 'b', 'c']}
        with self.assertRaises(ValidationError):
            CustomCDLProfile.from_dict(bad)

    def test_empty_clusters_rejected(self):
        bad = {**VALID_PROFILE, 'clusters': []}
        with self.assertRaises(ValidationError):
            CustomCDLProfile.from_dict(bad)

    def test_missing_center_frequency_rejected(self):
        bad = {k: v for k, v in VALID_PROFILE.items() if k != 'center_frequency_hz'}
        with self.assertRaises(ValidationError):
            CustomCDLProfile.from_dict(bad)

    def test_k_factor_db_optional(self):
        with_k = {**VALID_PROFILE, 'k_factor_db': 10.0}
        p = CustomCDLProfile.from_dict(with_k)
        self.assertEqual(p.k_factor_db, 10.0)

    def test_extra_top_level_field_rejected(self):
        bad = {**VALID_PROFILE, 'global_params': {'unused': 1}}
        with self.assertRaises(ValidationError):
            CustomCDLProfile.from_dict(bad)


class TestFromFile(unittest.TestCase):
    """from_file factory + error paths."""

    def test_roundtrip(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(VALID_PROFILE, f)
            path = f.name
        try:
            p = CustomCDLProfile.from_file(path)
            self.assertEqual(p.center_frequency_hz, 2.6e9)
        finally:
            os.remove(path)

    def test_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            CustomCDLProfile.from_file('/nonexistent/path/cdl.json')

    def test_invalid_json(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('{ not valid json')
            path = f.name
        try:
            with self.assertRaises(ValueError) as ctx:
                CustomCDLProfile.from_file(path)
            self.assertIn('not valid JSON', str(ctx.exception))
        finally:
            os.remove(path)


class TestCustomCDLBuilder(unittest.TestCase):
    """T5.3: CustomCDLBuilder + CustomJSONBuilder deprecation shim smoke tests."""

    def setUp(self):
        from mimo_ota_simulator.channel_builders import CustomCDLBuilder
        self.CustomCDLBuilder = CustomCDLBuilder

    def test_new_builder_basic(self):
        """CustomCDLBuilder takes a CustomCDLProfile and produces UniversalChannelProfile."""
        profile = CustomCDLProfile.from_dict(VALID_PROFILE)
        result = self.CustomCDLBuilder(profile).build()
        self.assertEqual(result.center_frequency_hz, 2.6e9)
        self.assertEqual(len(result.clusters), 2)
        # Power normalized to sum 1.0
        total = sum(c.power for c in result.clusters)
        self.assertAlmostEqual(total, 1.0, places=10)
        # Geometric fields plumbed
        self.assertEqual(result.clusters[0].aoa_phi, 10.0)
        self.assertEqual(result.clusters[0].aod_phi, -15.0)

    def test_new_builder_plumbs_xpr(self):
        """xpr_db → EngineCluster.xpr_linear (dB → linear conversion)."""
        d = {**VALID_PROFILE}
        d['clusters'] = [{**VALID_CLUSTER, 'xpr_db': 7.0}]
        profile = CustomCDLProfile.from_dict(d)
        result = self.CustomCDLBuilder(profile).build()
        # 7 dB = 10^(7/10) ≈ 5.012
        self.assertAlmostEqual(result.clusters[0].xpr_linear, 10 ** 0.7, places=6)

    def test_new_builder_plumbs_initial_phases(self):
        """initial_phases_rad list → np.ndarray on EngineCluster."""
        import numpy as np
        d = {**VALID_PROFILE}
        d['clusters'] = [{**VALID_CLUSTER, 'initial_phases_rad': [0.1, 0.2, 0.3, 0.4]}]
        profile = CustomCDLProfile.from_dict(d)
        result = self.CustomCDLBuilder(profile).build()
        self.assertIsInstance(result.clusters[0].initial_phases, np.ndarray)
        self.assertEqual(result.clusters[0].initial_phases.shape, (4,))
        np.testing.assert_allclose(
            result.clusters[0].initial_phases, [0.1, 0.2, 0.3, 0.4]
        )

    def test_new_builder_rejects_non_profile(self):
        """Caller must pass a CustomCDLProfile, not a dict directly."""
        with self.assertRaises(TypeError):
            self.CustomCDLBuilder(VALID_PROFILE)  # raw dict, should error

    def test_legacy_json_builder_still_works_with_warning(self):
        """CustomJSONBuilder (v1.0 file path) still works + emits DeprecationWarning."""
        import warnings as _warnings
        from mimo_ota_simulator.channel_builders import CustomJSONBuilder
        # Legacy v1.0 format
        legacy_data = {
            'global_params': {
                'center_frequency_hz': 2.6e9,
                'pathloss_db': 85.0,
                'ue_velocity_kph': [15.0, 0.0, 0.0],
            },
            'clusters': [
                {'delay_s': 0.0, 'power_relative_linear': 0.7, 'aoa_deg': 10, 'aod_deg': -15,
                 'as_aoa_deg': 5.0, 'as_aod_deg': 2.0},
            ],
            'is_los': True,
        }
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(legacy_data, f)
            path = f.name
        try:
            with _warnings.catch_warnings(record=True) as w:
                _warnings.simplefilter('always')
                builder = CustomJSONBuilder(path)
                result = builder.build()
            # Did the deprecation warning fire?
            deprecation_msgs = [
                str(wi.message) for wi in w
                if issubclass(wi.category, DeprecationWarning)
                and 'CustomJSONBuilder' in str(wi.message)
            ]
            self.assertGreater(len(deprecation_msgs), 0)
            # And output is sane
            self.assertEqual(result.center_frequency_hz, 2.6e9)
            self.assertEqual(len(result.clusters), 1)
            # ue_velocity converted kph→mps via legacy shim
            import numpy as np
            np.testing.assert_allclose(result.ue_velocity, [15 * 1000 / 3600, 0.0, 0.0])
        finally:
            os.remove(path)


class TestKFactorLosBoost(unittest.TestCase):
    """T5.4: K-factor LOS power boost per TR 38.901 §7.5.6.

    Contract:
      - is_los=True + k_factor_db is None  → power normalized to sum 1 (no boost)
      - is_los=True + k_factor_db = K (dB) → first cluster gets K_linear/(K+1),
                                            rest share 1/(K+1) proportionally
      - is_los=False                       → k_factor_db ignored (caller may
                                            speculatively set it; we skip silently)
    """

    def _build(self, is_los, k_factor_db, cluster_powers):
        from mimo_ota_simulator.channel_builders import CustomCDLBuilder
        d = {
            'center_frequency_hz': 2.6e9,
            'pathloss_db': 85.0,
            'ue_velocity_mps': [4.17, 0.0, 0.0],
            'is_los': is_los,
            'k_factor_db': k_factor_db,
            'clusters': [
                {'delay_s': i * 1e-7, 'power_linear': p,
                 'aoa_deg': 0, 'aod_deg': 0,
                 'as_aoa_deg': 1, 'as_aod_deg': 1}
                for i, p in enumerate(cluster_powers)
            ],
        }
        profile = CustomCDLProfile.from_dict(d)
        return CustomCDLBuilder(profile).build()

    def test_no_k_factor_just_normalizes(self):
        """k_factor_db=None: cluster powers sum to 1, no LOS boost."""
        result = self._build(is_los=True, k_factor_db=None, cluster_powers=[0.5, 0.3, 0.2])
        total = sum(c.power for c in result.clusters)
        self.assertAlmostEqual(total, 1.0, places=10)
        # Proportional: 0.5/1.0 = 0.5, 0.3/1.0 = 0.3, 0.2/1.0 = 0.2
        self.assertAlmostEqual(result.clusters[0].power, 0.5, places=10)
        self.assertAlmostEqual(result.clusters[1].power, 0.3, places=10)
        self.assertAlmostEqual(result.clusters[2].power, 0.2, places=10)

    def test_k_factor_10db_boost_first_cluster(self):
        """K = 10 dB → K_linear = 10 → first cluster gets 10/11, rest share 1/11."""
        result = self._build(is_los=True, k_factor_db=10.0,
                             cluster_powers=[0.4, 0.4, 0.2])
        # First cluster gets K/(K+1) = 10/11
        self.assertAlmostEqual(result.clusters[0].power, 10.0/11.0, places=10)
        # Rest share 1/11 in their original ratio (0.4 : 0.2 → 2:1 → 2/3, 1/3)
        nlos_total = sum(c.power for c in result.clusters[1:])
        self.assertAlmostEqual(nlos_total, 1.0/11.0, places=10)
        self.assertAlmostEqual(result.clusters[1].power, (1.0/11.0) * (2.0/3.0), places=10)
        self.assertAlmostEqual(result.clusters[2].power, (1.0/11.0) * (1.0/3.0), places=10)
        # Total = 10/11 + 1/11 = 1.0
        self.assertAlmostEqual(sum(c.power for c in result.clusters), 1.0, places=10)

    def test_k_factor_0db_equal_split(self):
        """K = 0 dB → K_linear = 1 → first gets 1/2, rest share 1/2."""
        result = self._build(is_los=True, k_factor_db=0.0,
                             cluster_powers=[0.5, 0.3, 0.2])
        self.assertAlmostEqual(result.clusters[0].power, 0.5, places=10)
        self.assertAlmostEqual(sum(c.power for c in result.clusters[1:]), 0.5, places=10)
        self.assertAlmostEqual(sum(c.power for c in result.clusters), 1.0, places=10)

    def test_k_factor_ignored_when_nlos(self):
        """is_los=False: k_factor_db is silently ignored."""
        result = self._build(is_los=False, k_factor_db=10.0,
                             cluster_powers=[0.5, 0.3, 0.2])
        # Should be normalized-only (no boost)
        self.assertAlmostEqual(result.clusters[0].power, 0.5, places=10)
        self.assertAlmostEqual(sum(c.power for c in result.clusters), 1.0, places=10)

    def test_k_factor_single_cluster(self):
        """Edge case: only 1 cluster. K-factor still puts K/(K+1) on it; no NLOS to scale."""
        result = self._build(is_los=True, k_factor_db=10.0, cluster_powers=[1.0])
        self.assertAlmostEqual(result.clusters[0].power, 10.0/11.0, places=10)


class TestPerRayInitialPhasesIntegration(unittest.TestCase):
    """T5.5: CDL initial_phases_rad[0] (Θθθ) consumed as per-cluster base phase
    in strict_pfs / ray synthesizer.

    Contract:
      - cluster.initial_phases = None  → base_phase=0 (legacy behavior)
      - cluster.initial_phases = [Θθθ, ...] → base_phase = Θθθ added to
        per-(probe, ray) random component
      - strict_pfs cross-corr contract (T2.1) STILL holds because adding a
        constant base to all probe phases doesn't affect E[exp(j(p_i - p_j))]
    """

    def _run_with_init_phase(self, base_phase, method='strict_pfs'):
        """Run sim with single-cluster CDL where cluster.initial_phases[0] = base_phase.
        Returns the resulting A matrix."""
        import warnings as _warnings
        from mimo_ota_simulator.simulator import MIMO_OTA_Simulator
        from mimo_ota_simulator.data_models import (
            AntennaArrayConfig, ChamberConfig, TargetChannelConfig,
        )
        import numpy as np

        chamber = ChamberConfig(
            num_probes=8, distribution='ring', radius_m=1.5,
            dual_polarized=False,
            probe_antenna={'gain_dBi': 0.0, 'pattern': 'Isotropic'},
        )
        cluster_dict = {
            'delay_s': 0.0, 'power_linear': 1.0,
            'aoa_deg': 30, 'aod_deg': 15,
            'as_aoa_deg': 5.0, 'as_aod_deg': 2.0,
        }
        if base_phase is not None:
            cluster_dict['initial_phases_rad'] = [base_phase, 0.0, 0.0, 0.0]

        profile = CustomCDLProfile.from_dict({
            'center_frequency_hz': 2.6e9, 'pathloss_db': 85.0,
            'ue_velocity_mps': [4.17, 0.0, 0.0], 'is_los': False,
            'clusters': [cluster_dict],
        })
        config = TargetChannelConfig(
            input_mode='custom', custom_profile=profile,
            tx_antenna=AntennaArrayConfig(num_rows=1, num_cols=2, spacing_h=0.5),
            rx_antenna=AntennaArrayConfig(num_rows=1, num_cols=2, spacing_h=0.5),
        )
        np.random.seed(42)
        with _warnings.catch_warnings():
            _warnings.simplefilter('ignore')
            res = MIMO_OTA_Simulator().run(chamber, config, synthesis_method=method)
        return res['transmit_mapping_matrix_A']

    def test_strict_pfs_base_phase_changes_A(self):
        """Different initial_phases_rad[0] → different A (proves base is consumed)."""
        A_zero = self._run_with_init_phase(0.0)
        A_pi_half = self._run_with_init_phase(1.5707963267948966)  # π/2
        import numpy as np
        diff = np.abs(A_zero - A_pi_half).max()
        self.assertGreater(diff, 1e-6,
            msg="A matrix unchanged when initial_phases_rad changes — base phase not consumed")

    def test_no_init_phases_matches_explicit_zero(self):
        """None initial_phases ≡ explicit [0,0,0,0] base (legacy behavior)."""
        A_none = self._run_with_init_phase(None)
        A_zero = self._run_with_init_phase(0.0)
        import numpy as np
        # Both seed=42, same setup; results should be bit-identical
        np.testing.assert_allclose(A_none, A_zero, rtol=1e-12, atol=1e-12)

    def test_strict_pfs_cross_corr_preserved_with_init_phases(self):
        """T2.1 contract: E[A_i·A_j*]→0 still holds with non-zero base phase.

        Adding a constant base_phase to all per-probe random draws shifts
        all probes by the same scalar; doesn't affect cross-correlation.
        """
        import numpy as np
        import warnings as _warnings
        from mimo_ota_simulator.simulator import MIMO_OTA_Simulator
        from mimo_ota_simulator.data_models import (
            AntennaArrayConfig, ChamberConfig, TargetChannelConfig,
        )

        chamber = ChamberConfig(
            num_probes=8, distribution='ring', radius_m=1.5,
            dual_polarized=False,
            probe_antenna={'gain_dBi': 0.0, 'pattern': 'Isotropic'},
        )
        N = 60
        P = chamber.num_probes
        R_sum = np.zeros((P, P), dtype=complex)
        sim = MIMO_OTA_Simulator()
        for k in range(N):
            profile = CustomCDLProfile.from_dict({
                'center_frequency_hz': 2.6e9, 'pathloss_db': 85.0,
                'ue_velocity_mps': [4.17, 0.0, 0.0], 'is_los': False,
                'clusters': [{
                    'delay_s': 0.0, 'power_linear': 1.0,
                    'aoa_deg': 30, 'aod_deg': 15,
                    'as_aoa_deg': 5.0, 'as_aod_deg': 2.0,
                    'initial_phases_rad': [0.7853981633974483, 0.0, 0.0, 0.0],  # π/4 base
                }],
            })
            config = TargetChannelConfig(
                input_mode='custom', custom_profile=profile,
                tx_antenna=AntennaArrayConfig(num_rows=1, num_cols=1),
                rx_antenna=AntennaArrayConfig(num_rows=1, num_cols=1),
            )
            np.random.seed(2000 + k)
            with _warnings.catch_warnings():
                _warnings.simplefilter('ignore')
                res = sim.run(chamber, config, synthesis_method='strict_pfs')
            a = res['transmit_mapping_matrix_A'][:, 0]
            R_sum += np.outer(a, a.conj())
        R = R_sum / N
        d = np.sqrt(np.abs(np.diag(R)))
        denom = np.outer(d, d)
        denom = np.where(denom < 1e-15, 1e-15, denom)
        rho = R / denom
        off_mask = ~np.eye(P, dtype=bool)
        mean_off = np.abs(rho[off_mask]).mean()
        # Same threshold as T2.1 (mean off-diag < 0.20 with N~60-80)
        self.assertLess(mean_off, 0.25,
            msg=f"Mean off-diag |rho|={mean_off:.4f} too high; base phase broke cross-corr contract")


if __name__ == '__main__':
    unittest.main()
