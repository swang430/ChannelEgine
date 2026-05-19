"""Audit batch 1 tests (2026-05-19) — silent-ignored config fields.

Covers three findings from the post-P2-7 audit:

  F1: ChamberConfig.quiet_zone_radius_m — docstring used to say "for
      validation" but no code path consumes it. We don't add behavior; we
      assert the field is still accepted (stored but unused) so docstring
      drift doesn't accidentally turn into a breaking change.

  F2: CustomCDLProfile.k_factor_db with is_los=False used to be silently
      skipped by CustomCDLBuilder. Now emits UserWarning.

  F4: TargetChannelConfig(standard='wifi', input_mode='custom') used to
      construct cleanly, then _run_wifi() would silently drop custom_profile.
      Now rejected at config-construction time with ValidationError.
"""
from __future__ import annotations

import os
import sys
import unittest
import warnings

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from pydantic import ValidationError

from mimo_ota_simulator.cdl_schema import CustomCDLProfile
from mimo_ota_simulator.channel_builders import CustomCDLBuilder
from mimo_ota_simulator.data_models import ChamberConfig, TargetChannelConfig


# --- F1 ----------------------------------------------------------------------

class TestF1QuietZoneRadiusMetadata(unittest.TestCase):
    """quiet_zone_radius_m is operator metadata; library does NOT validate."""

    def test_unset_default_is_none(self):
        ch = ChamberConfig(num_probes=8, radius_m=1.5, distribution='ring')
        self.assertIsNone(ch.quiet_zone_radius_m)

    def test_set_value_stored_and_readable(self):
        ch = ChamberConfig(num_probes=8, radius_m=1.5, distribution='ring', quiet_zone_radius_m=0.3)
        self.assertEqual(ch.quiet_zone_radius_m, 0.3)

    def test_qz_larger_than_radius_is_accepted_no_validation(self):
        """Library must NOT raise even when QZ > probe radius (operator's
        decision; possibly nonsensical but library is intentionally permissive).
        """
        ch = ChamberConfig(num_probes=8, radius_m=1.0, distribution='ring', quiet_zone_radius_m=5.0)
        self.assertEqual(ch.quiet_zone_radius_m, 5.0)


# --- F2 ----------------------------------------------------------------------

def _profile_with_k(is_los: bool, k_factor_db: float | None):
    return CustomCDLProfile.from_dict({
        'center_frequency_hz': 2.6e9,
        'pathloss_db': 85.0,
        'ue_velocity_mps': [4.17, 0.0, 0.0],
        'is_los': is_los,
        'k_factor_db': k_factor_db,
        'clusters': [
            {
                'delay_s': 0.0, 'power_linear': 0.6,
                'aoa_deg': 0.0, 'aod_deg': 0.0,
                'as_aoa_deg': 1.0, 'as_aod_deg': 1.0,
            },
            {
                'delay_s': 1e-7, 'power_linear': 0.4,
                'aoa_deg': 30.0, 'aod_deg': 30.0,
                'as_aoa_deg': 1.0, 'as_aod_deg': 1.0,
            },
        ],
    })


class TestF2KFactorNlosWarns(unittest.TestCase):
    """k_factor_db + is_los=False emits UserWarning (was silent skip)."""

    def test_nlos_with_k_factor_warns(self):
        profile = _profile_with_k(is_los=False, k_factor_db=10.0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            CustomCDLBuilder(profile).build()
        msgs = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
        assert any('k_factor_db' in m and 'is_los=False' in m for m in msgs), msgs

    def test_los_with_k_factor_does_not_warn(self):
        profile = _profile_with_k(is_los=True, k_factor_db=10.0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            CustomCDLBuilder(profile).build()
        for w in caught:
            self.assertNotIn('k_factor_db', str(w.message))

    def test_nlos_without_k_factor_does_not_warn(self):
        profile = _profile_with_k(is_los=False, k_factor_db=None)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            CustomCDLBuilder(profile).build()
        for w in caught:
            self.assertNotIn('k_factor_db', str(w.message))

    def test_nlos_with_k_factor_powers_remain_normalized(self):
        """Warning fires but build() still returns a valid profile (cluster
        powers sum to ~1.0); the K-factor boost block is skipped, not the rest.
        """
        profile = _profile_with_k(is_los=False, k_factor_db=10.0)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = CustomCDLBuilder(profile).build()
        total = sum(c.power for c in result.clusters)
        self.assertAlmostEqual(total, 1.0, places=6)


# --- F4 ----------------------------------------------------------------------

class TestF4WifiCustomRejected(unittest.TestCase):
    """standard='wifi' + input_mode='custom' is rejected at config time."""

    def _minimal_custom_profile(self):
        return CustomCDLProfile.from_dict({
            'center_frequency_hz': 5.5e9,
            'pathloss_db': 80.0,
            'ue_velocity_mps': [0.0, 0.0, 0.0],
            'is_los': True,
            'clusters': [{
                'delay_s': 0.0, 'power_linear': 1.0,
                'aoa_deg': 0.0, 'aod_deg': 0.0,
                'as_aoa_deg': 1.0, 'as_aod_deg': 1.0,
            }],
        })

    def test_wifi_plus_custom_with_profile_object_raises(self):
        with self.assertRaises(ValidationError) as ctx:
            TargetChannelConfig(
                standard='wifi',
                input_mode='custom',
                custom_profile=self._minimal_custom_profile(),
            )
        self.assertIn("standard='wifi'", str(ctx.exception))
        self.assertIn("input_mode='custom'", str(ctx.exception))

    def test_wifi_plus_custom_with_path_raises(self):
        with self.assertRaises(ValidationError):
            TargetChannelConfig(
                standard='wifi',
                input_mode='custom',
                custom_profile_path='ignored.json',
            )

    def test_wifi_plus_standard_ok(self):
        cfg = TargetChannelConfig(standard='wifi', input_mode='standard')
        self.assertEqual(cfg.standard, 'wifi')
        self.assertEqual(cfg.input_mode, 'standard')

    def test_threegpp_plus_custom_ok(self):
        cfg = TargetChannelConfig(
            standard='3gpp',
            input_mode='custom',
            custom_profile=self._minimal_custom_profile(),
        )
        self.assertEqual(cfg.standard, '3gpp')
        self.assertEqual(cfg.input_mode, 'custom')


if __name__ == '__main__':
    unittest.main()
