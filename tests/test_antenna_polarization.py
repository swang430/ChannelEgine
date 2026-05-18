"""Tests for AntennaArrayConfig.polarization field (Phase 6 T6.1).

Verifies the schema change is backward-compatible (default 'V') and that
explicit 'V' / 'H' is accepted while invalid values are rejected.
"""
from __future__ import annotations

import glob
import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pydantic import ValidationError

from mimo_ota_simulator.data_models import AntennaArrayConfig


class TestAntennaPolarizationField(unittest.TestCase):
    """Schema-level validation for the new polarization field."""

    def test_default_is_vertical(self):
        """Backward compat: no polarization specified → defaults to 'V'."""
        cfg = AntennaArrayConfig()
        self.assertEqual(cfg.polarization, 'V')

    def test_default_is_vertical_with_other_fields_set(self):
        """Default still V when other antenna fields are explicitly set."""
        cfg = AntennaArrayConfig(num_rows=1, num_cols=4, spacing_h=0.5)
        self.assertEqual(cfg.polarization, 'V')

    def test_explicit_vertical(self):
        cfg = AntennaArrayConfig(polarization='V')
        self.assertEqual(cfg.polarization, 'V')

    def test_explicit_horizontal(self):
        cfg = AntennaArrayConfig(polarization='H')
        self.assertEqual(cfg.polarization, 'H')

    def test_invalid_polarization_rejected(self):
        with self.assertRaises(ValidationError):
            AntennaArrayConfig(polarization='X')

    def test_slant_45_rejected_phase6(self):
        """+45 / -45 / slant are Phase 7 (deferred); rejected in Phase 6."""
        for invalid in ['+45', '-45', 'slant', 'circular', 'V/H']:
            with self.assertRaises(ValidationError, msg=f"polarization={invalid!r} should be rejected in Phase 6"):
                AntennaArrayConfig(polarization=invalid)


class TestBMatrixDualPol(unittest.TestCase):
    """T6.2: _synthesize_b_matrix dual-pol H/V port coupling.

    Convention (per design doc §6 / source comment):
      - Index 2p of the B-matrix column axis = V port (θ-polarized)
      - Index 2p+1 = H port (φ-polarized)
    DUT polarization picks which port couples; the orthogonal port gets 0.
    """

    def _build(self, dual_polarized, rx_pol):
        from mimo_ota_simulator.data_models import (
            AntennaArrayConfig, ChamberConfig, TargetChannelConfig,
        )
        from mimo_ota_simulator.simulator import MIMO_OTA_Simulator
        sim = MIMO_OTA_Simulator()
        chamber = ChamberConfig(
            num_probes=4, distribution='ring', radius_m=1.5,
            dual_polarized=dual_polarized,
            probe_antenna={'gain_dBi': 0.0, 'pattern': 'Isotropic'},
        )
        cfg = TargetChannelConfig(
            center_frequency_hz=2.6e9,
            tx_antenna=AntennaArrayConfig(num_cols=1),
            rx_antenna=AntennaArrayConfig(num_cols=1, polarization=rx_pol),
        )
        return sim._synthesize_b_matrix(chamber, cfg)

    def test_dual_pol_v_dut_h_ports_zero(self):
        """dual_polarized=True + DUT V-pol → H ports (2p+1) all zero."""
        import numpy as np
        B = self._build(dual_polarized=True, rx_pol='V')
        # Shape: (1 rx_antenna, 4 probes * 2 pol = 8 effective probes)
        self.assertEqual(B.shape, (1, 8))
        v_ports = B[:, 0::2]  # 2p = V ports
        h_ports = B[:, 1::2]  # 2p+1 = H ports
        self.assertTrue(np.all(np.abs(v_ports) > 0), "V ports should carry signal for V-pol DUT")
        self.assertTrue(np.all(np.abs(h_ports) == 0), "H ports should be zero for V-pol DUT")

    def test_dual_pol_h_dut_v_ports_zero(self):
        """dual_polarized=True + DUT H-pol → V ports (2p) all zero."""
        import numpy as np
        B = self._build(dual_polarized=True, rx_pol='H')
        self.assertEqual(B.shape, (1, 8))
        v_ports = B[:, 0::2]
        h_ports = B[:, 1::2]
        self.assertTrue(np.all(np.abs(v_ports) == 0), "V ports should be zero for H-pol DUT")
        self.assertTrue(np.all(np.abs(h_ports) > 0), "H ports should carry signal for H-pol DUT")

    def test_single_pol_v_dut_unchanged(self):
        """dual_polarized=False + DUT V (default) → behavior unchanged (regression baseline)."""
        import numpy as np
        B = self._build(dual_polarized=False, rx_pol='V')
        self.assertEqual(B.shape, (1, 4))  # single-pol: 4 effective probes
        self.assertTrue(np.all(np.abs(B) > 0), "Single-pol B-matrix should have all-nonzero values")

    def test_single_pol_h_dut_raises_oq4(self):
        """dual_polarized=False + DUT H → raise ValueError per OQ4 decision."""
        with self.assertRaises(ValueError) as ctx:
            self._build(dual_polarized=False, rx_pol='H')
        msg = str(ctx.exception)
        self.assertIn('dual_polarized=True', msg)  # error message suggests fix
        self.assertIn("'V'", msg)


class TestPolarizationMatrixHelper(unittest.TestCase):
    """T6.3: _build_pol_matrix(cluster) and _pol_vec(label) helpers.

    The 2×2 polarization matrix (TR 38.901 §7.3.2) is:
        [[exp(jΦ_θθ),             sqrt(1/κ)·exp(jΦ_θφ)],
         [sqrt(1/κ)·exp(jΦ_φθ),   exp(jΦ_φφ)         ]]
    """

    def _sim(self):
        from mimo_ota_simulator.simulator import MIMO_OTA_Simulator
        return MIMO_OTA_Simulator()

    def _cluster(self, xpr_linear=None, initial_phases=None):
        from channel_model_38901.common.data_models import EngineCluster
        import numpy as np
        kwargs = dict(delay=0.0, power=1.0)
        if xpr_linear is not None:
            kwargs['xpr_linear'] = xpr_linear
        if initial_phases is not None:
            kwargs['initial_phases'] = np.asarray(initial_phases, dtype=float)
        return EngineCluster(**kwargs)

    def test_diagonal_with_zero_phases(self):
        """xpr=7 dB, phases=[0,0,0,0]: diagonal entries are 1+0j, off-diag are sqrt(1/κ)."""
        import numpy as np
        m = self._sim()._build_pol_matrix(self._cluster(xpr_linear=10**0.7, initial_phases=[0,0,0,0]))
        self.assertEqual(m.shape, (2, 2))
        self.assertAlmostEqual(m[0, 0], 1.0 + 0j, places=10)
        self.assertAlmostEqual(m[1, 1], 1.0 + 0j, places=10)
        expected_off = 1.0 / np.sqrt(10**0.7)
        self.assertAlmostEqual(m[0, 1], expected_off + 0j, places=10)
        self.assertAlmostEqual(m[1, 0], expected_off + 0j, places=10)

    def test_phase_terms_correct(self):
        """Explicit Φ values appear in matrix entries as exp(jΦ)."""
        import numpy as np
        phi = [0.1, 0.2, 0.3, 0.4]
        m = self._sim()._build_pol_matrix(
            self._cluster(xpr_linear=4.0, initial_phases=phi)
        )
        # diag: exp(j Φ_θθ) and exp(j Φ_φφ)
        self.assertAlmostEqual(m[0, 0], np.exp(1j * phi[0]), places=10)
        self.assertAlmostEqual(m[1, 1], np.exp(1j * phi[3]), places=10)
        # off-diag: sqrt(1/κ) * exp(j Φ_*) with κ=4 → sqrt(1/4)=0.5
        self.assertAlmostEqual(m[0, 1], 0.5 * np.exp(1j * phi[1]), places=10)
        self.assertAlmostEqual(m[1, 0], 0.5 * np.exp(1j * phi[2]), places=10)

    def test_high_xpr_suppresses_cross_pol(self):
        """κ→∞ means off-diagonal sqrt(1/κ)→0 (no cross-pol leakage)."""
        import numpy as np
        m = self._sim()._build_pol_matrix(
            self._cluster(xpr_linear=1e10, initial_phases=[0,0,0,0])
        )
        self.assertAlmostEqual(np.abs(m[0, 0]), 1.0, places=10)
        self.assertAlmostEqual(np.abs(m[1, 1]), 1.0, places=10)
        # κ=1e10 → sqrt(1/κ)=1e-5, well below 1e-4 threshold
        self.assertLess(np.abs(m[0, 1]), 1e-4)
        self.assertLess(np.abs(m[1, 0]), 1e-4)

    def test_unity_xpr_equal_diag_offdiag(self):
        """κ=1 (0 dB): all four entries have unit magnitude."""
        import numpy as np
        m = self._sim()._build_pol_matrix(
            self._cluster(xpr_linear=1.0, initial_phases=[0,0,0,0])
        )
        for entry in m.flatten():
            self.assertAlmostEqual(np.abs(entry), 1.0, places=10)

    def test_default_xpr_with_warning(self):
        """Missing xpr_linear → DEFAULT_XPR_DB (7 dB) + warning per OQ2."""
        import numpy as np
        import warnings as _warnings
        with _warnings.catch_warnings(record=True) as w:
            _warnings.simplefilter('always')
            m = self._sim()._build_pol_matrix(
                self._cluster(xpr_linear=None, initial_phases=[0,0,0,0])
            )
            xpr_warns = [x for x in w if 'xpr_linear' in str(x.message)]
            self.assertEqual(len(xpr_warns), 1)
        # Off-diag should match default 7 dB
        expected_off = 1.0 / np.sqrt(10 ** 0.7)
        self.assertAlmostEqual(np.abs(m[0, 1]), expected_off, places=10)

    def test_negative_xpr_raises(self):
        with self.assertRaises(ValueError):
            self._sim()._build_pol_matrix(self._cluster(xpr_linear=-1.0))

    def test_default_phases_when_missing(self):
        """Missing initial_phases → zero phases (no warning)."""
        import numpy as np
        import warnings as _warnings
        with _warnings.catch_warnings():
            _warnings.simplefilter('ignore')
            m = self._sim()._build_pol_matrix(
                self._cluster(xpr_linear=1.0, initial_phases=None)
            )
        # All entries should be real positive (phases=0)
        for entry in m.flatten():
            self.assertAlmostEqual(entry.imag, 0.0, places=10)
            self.assertGreater(entry.real, 0.0)

    def test_pol_vec_v_and_h(self):
        import numpy as np
        sim = self._sim()
        np.testing.assert_array_equal(sim._pol_vec('V'), [1.0, 0.0])
        np.testing.assert_array_equal(sim._pol_vec('H'), [0.0, 1.0])

    def test_pol_vec_invalid_raises(self):
        with self.assertRaises(ValueError):
            self._sim()._pol_vec('+45')


class TestDualPolCrossCorrelationContract(unittest.TestCase):
    """T6.5: strict_pfs cross-corr contract (T2.1) preserved under dual-pol.

    Per design §3 / §6: applying a constant per-cluster pol_factor (m_col[0]
    or m_col[1]) doesn't break per-probe independence — random phase per
    effective probe still gives E[exp(j(phi_i - phi_j))] = 0 for i ≠ j,
    so probe-domain cross-correlation collapses regardless of pol_factor.

    This is a key safety net: dual-pol math must not silently degrade the
    strict_pfs property that motivates skipping output phase calibration.
    """

    N_REALIZATIONS = 60
    THRESHOLD_MEAN_OFF_DIAG = 0.25  # same as T2.1 floor (1/sqrt(60) ~ 0.13, 2x safety)

    def test_strict_pfs_dual_pol_off_diag_converges(self):
        """V DUT + V Tx + dual_polarized=True: probe-domain mean off-diag |rho| < 0.25.

        Uses 8 probes (= 16 effective in dual-pol). Threshold is mean
        off-diagonal of the normalized correlation matrix R̂ / sqrt(d_i d_j).
        """
        import numpy as np
        import warnings as _warnings
        from mimo_ota_simulator.cdl_schema import CustomCDLProfile
        from mimo_ota_simulator.data_models import (
            AntennaArrayConfig, ChamberConfig, TargetChannelConfig,
        )
        from mimo_ota_simulator.simulator import MIMO_OTA_Simulator

        chamber = ChamberConfig(
            num_probes=8, distribution='ring', radius_m=1.5,
            dual_polarized=True,  # ← Phase 6 dual-pol path
            probe_antenna={'gain_dBi': 0.0, 'pattern': 'Isotropic'},
        )
        P_eff = 16  # 8 * 2

        sim = MIMO_OTA_Simulator()
        R_sum = np.zeros((P_eff, P_eff), dtype=complex)
        for k in range(self.N_REALIZATIONS):
            profile = CustomCDLProfile.from_dict({
                'center_frequency_hz': 2.6e9, 'pathloss_db': 85.0,
                'ue_velocity_mps': [0.0, 0.0, 0.0], 'is_los': False,
                'clusters': [{
                    'delay_s': 0.0, 'power_linear': 1.0,
                    'aoa_deg': 30, 'aod_deg': 15,
                    'as_aoa_deg': 5.0, 'as_aod_deg': 2.0,
                    'xpr_db': 7.0,
                }],
            })
            config = TargetChannelConfig(
                input_mode='custom', custom_profile=profile,
                tx_antenna=AntennaArrayConfig(num_cols=1, polarization='V'),
                rx_antenna=AntennaArrayConfig(num_cols=1, polarization='V'),
            )
            np.random.seed(3000 + k)
            with _warnings.catch_warnings():
                _warnings.simplefilter('ignore')
                res = sim.run(chamber, config, synthesis_method='strict_pfs')
            a = res['transmit_mapping_matrix_A'][:, 0]  # (P_eff,)
            R_sum += np.outer(a, a.conj())
        R = R_sum / self.N_REALIZATIONS

        # Normalize to correlation coefficient
        d = np.sqrt(np.abs(np.diag(R)))
        denom = np.outer(d, d)
        denom = np.where(denom < 1e-15, 1e-15, denom)
        rho = R / denom
        off_mask = ~np.eye(P_eff, dtype=bool)
        mean_off = np.abs(rho[off_mask]).mean()
        self.assertLess(
            mean_off, self.THRESHOLD_MEAN_OFF_DIAG,
            msg=(f"dual-pol strict_pfs mean off-diag |rho|={mean_off:.4f} "
                 f">= threshold {self.THRESHOLD_MEAN_OFF_DIAG}. Dual-pol "
                 f"factorization broke the per-realization strict-PFS contract.")
        )


class TestDualPolE2E(unittest.TestCase):
    """T6.6: dual-pol DUT + XPR + initial_phases → strict_pfs → Propsim DL ASC."""

    def setUp(self):
        self.base = '_dual_pol_e2e.asc'

    def tearDown(self):
        for f in glob.glob(self.base.replace('.asc', '*.asc')):
            try:
                os.remove(f)
            except OSError:
                pass

    def _run(self, rx_pol, tx_pol):
        import numpy as np
        import warnings as _warnings
        from mimo_ota_simulator.cdl_schema import CustomCDLProfile
        from mimo_ota_simulator.data_models import (
            AntennaArrayConfig, ChamberConfig, TargetChannelConfig,
            PropsimExportConfig,
        )
        from mimo_ota_simulator.exporters import PropsimASCIIExporter
        from mimo_ota_simulator.simulator import MIMO_OTA_Simulator

        chamber = ChamberConfig(
            num_probes=8, distribution='ring', radius_m=1.5,
            dual_polarized=True,
            probe_antenna={'gain_dBi': 0.0, 'pattern': 'Isotropic'},
        )
        profile = CustomCDLProfile.from_dict({
            'center_frequency_hz': 2.6e9, 'pathloss_db': 88.5,
            'ue_velocity_mps': [4.17, 0.0, 0.0], 'is_los': False,
            'clusters': [{
                'delay_s': 0.0, 'power_linear': 1.0,
                'aoa_deg': 0, 'aod_deg': 0,
                'as_aoa_deg': 2.0, 'as_aod_deg': 1.0,
                'xpr_db': 7.0,
                'initial_phases_rad': [0.0, 0.7853981633974483, 1.5707963267948966, 3.141592653589793],
            }],
        })
        config = TargetChannelConfig(
            input_mode='custom', custom_profile=profile,
            tx_antenna=AntennaArrayConfig(num_cols=1, polarization=tx_pol),
            rx_antenna=AntennaArrayConfig(num_cols=1, polarization=rx_pol),
        )
        np.random.seed(42)
        sim = MIMO_OTA_Simulator()
        with _warnings.catch_warnings():
            _warnings.simplefilter('ignore')
            res = sim.run(chamber, config, synthesis_method='strict_pfs')
            cfg = PropsimExportConfig(
                filename=self.base, mode='B', duration_s=0.01, sample_rate_hz=1000,
            )
            PropsimASCIIExporter(cfg).export(res, direction='dl')
        return res

    def test_dual_pol_A_shape_and_asc_generated(self):
        import numpy as np
        result = self._run(rx_pol='V', tx_pol='V')
        A = result['transmit_mapping_matrix_A']
        # 8 probes * 2 pol = 16 effective probe ports; 1 Tx antenna
        self.assertEqual(A.shape, (16, 1))
        self.assertFalse(np.any(np.isnan(A)))
        files = glob.glob(self.base.replace('.asc', '*.asc'))
        self.assertGreater(len(files), 0)

    def test_v_tx_v_rx_v_port_dominates(self):
        """V Tx + V DUT: V probe ports get co-pol drive |exp(jΦθθ)|=1,
        H probe ports get cross-pol drive |sqrt(1/κ)·exp(jΦφθ)|=sqrt(1/κ).

        Per-port magnitude ratio |A_V| / |A_H| ≈ sqrt(κ) = sqrt(10^0.7) ≈ 2.24.
        Tolerance is loose because A is the sum of 20 randomly-phased rays per
        cluster; sample mean magnitude isn't exactly sqrt(κ).
        """
        import numpy as np
        result = self._run(rx_pol='V', tx_pol='V')
        A = result['transmit_mapping_matrix_A']
        v_ports = A[0::2, 0]   # (8,) V probe ports
        h_ports = A[1::2, 0]   # (8,) H probe ports
        expected_ratio = np.sqrt(10 ** 0.7)  # sqrt(κ) for 7 dB XPR
        actual_ratio = np.abs(v_ports).mean() / np.abs(h_ports).mean()
        # ±10% tolerance: V/H ratio is dominated by the |m_col| factor (exact
        # sqrt(κ)) but the random ray sum introduces ~few % sample variance.
        self.assertAlmostEqual(
            actual_ratio, expected_ratio, delta=expected_ratio * 0.10,
            msg=(f"V/H magnitude ratio {actual_ratio:.4f} not within ±10% of "
                 f"expected sqrt(κ)={expected_ratio:.4f} for XPR=7 dB"),
        )

    def test_effective_H_cross_pol_attenuation(self):
        """V DUT + V Tx vs H DUT + V Tx — both work, but effective_H magnitudes
        relate via XPR (orthogonal-pol case heavily attenuated)."""
        import numpy as np
        r_co = self._run(rx_pol='V', tx_pol='V')   # co-pol (channel scale 1)
        r_cross = self._run(rx_pol='H', tx_pol='V') # cross-pol (channel scale sqrt(1/κ))
        H_co_norm = np.linalg.norm(r_co['effective_H_matrix'])
        H_cross_norm = np.linalg.norm(r_cross['effective_H_matrix'])
        # cross-pol effective_H magnitude ratio ≈ sqrt(1/κ) for XPR 7 dB
        ratio = H_cross_norm / H_co_norm
        expected = 1.0 / np.sqrt(10 ** 0.7)  # sqrt(1/κ) ≈ 0.447
        # ±40% tolerance: effective_H aggregates many random-phase rays + the
        # cross-pol/co-pol channel difference. Looser than V/H per-port ratio.
        self.assertGreater(ratio, expected * 0.6)
        self.assertLess(ratio, expected * 1.4)


class TestStandard3GPPXPRPlumbed(unittest.TestCase):
    """PR #6 Codex P2 regression: standard 3GPP path must populate
    EngineCluster.xpr_linear (canonical Phase 6 field), not just the
    legacy `k_n` field.

    Without this fix, every 3GPP-scenario cluster fell through Phase 6
    _build_pol_matrix's fallback and used the 7 dB default XPR, silently
    ignoring the scenario-specific generated XPR.

    The fix mirrors channel_generator.py writes to both `k_n` (back-compat)
    and `xpr_linear` (canonical).
    """

    def _generate_3gpp_clusters(self, cluster_model_name):
        import numpy as np
        import warnings as _warnings
        from mimo_ota_simulator.data_models import (
            AntennaArrayConfig, ChamberConfig, TargetChannelConfig,
        )
        from mimo_ota_simulator.simulator import MIMO_OTA_Simulator
        chamber = ChamberConfig(
            num_probes=4, distribution='ring', radius_m=1.5,
            dual_polarized=True,  # exercises Phase 6 dual-pol path
            probe_antenna={'gain_dBi': 0.0, 'pattern': 'Isotropic'},
        )
        config = TargetChannelConfig(
            model_name='UMa', cluster_model_name=cluster_model_name,
            center_frequency_hz=3.5e9,
            tx_antenna=AntennaArrayConfig(num_cols=1, polarization='V'),
            rx_antenna=AntennaArrayConfig(num_cols=1, polarization='V'),
            force_condition='NLOS',
        )
        np.random.seed(42)
        with _warnings.catch_warnings(record=True) as w:
            _warnings.simplefilter('always')
            res = MIMO_OTA_Simulator().run(chamber, config, synthesis_method='strict_pfs')
            # Capture xpr_linear fallback warnings (should be NONE)
            xpr_fallback_warns = [
                wi for wi in w
                if 'EngineCluster.xpr_linear is None' in str(wi.message)
            ]
        return res, xpr_fallback_warns

    def test_deterministic_cdl_populates_xpr_linear(self):
        """CDL-A clusters carry scenario XPR in xpr_linear."""
        result, warns = self._generate_3gpp_clusters('CDL-A')
        clusters = result['clusters']
        self.assertGreater(len(clusters), 0)
        for i, c in enumerate(clusters):
            self.assertIsNotNone(c.xpr_linear,
                msg=f"CDL-A cluster {i} xpr_linear=None; Phase 6 will fall back to default")
            self.assertGreater(c.xpr_linear, 0)
            # k_n (legacy field) must mirror xpr_linear
            self.assertAlmostEqual(c.k_n, c.xpr_linear, places=10,
                msg="3GPP path: k_n and xpr_linear must mirror each other")
        # No fallback warning emitted (xpr_linear was populated)
        self.assertEqual(len(warns), 0,
            msg=f"Spurious xpr fallback warnings ({len(warns)}); xpr_linear should "
                f"have been populated by ChannelGenerator")

    def test_stochastic_uma_populates_xpr_linear(self):
        """Stochastic UMa (cluster_model_name=None / 'Stochastic') clusters
        also carry per-cluster generated XPR in xpr_linear."""
        result, warns = self._generate_3gpp_clusters('Stochastic')
        clusters = result['clusters']
        self.assertGreater(len(clusters), 0)
        for i, c in enumerate(clusters):
            self.assertIsNotNone(c.xpr_linear,
                msg=f"Stochastic UMa cluster {i} xpr_linear=None")
            self.assertGreater(c.xpr_linear, 0)
            self.assertAlmostEqual(c.k_n, c.xpr_linear, places=10)
        self.assertEqual(len(warns), 0)


if __name__ == '__main__':
    unittest.main()
