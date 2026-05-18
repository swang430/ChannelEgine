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


if __name__ == '__main__':
    unittest.main()
