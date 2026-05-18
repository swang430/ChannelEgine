"""Tests for AntennaArrayConfig.polarization field (Phase 6 T6.1).

Verifies the schema change is backward-compatible (default 'V') and that
explicit 'V' / 'H' is accepted while invalid values are rejected.
"""
from __future__ import annotations

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


if __name__ == '__main__':
    unittest.main()
