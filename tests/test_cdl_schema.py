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


if __name__ == '__main__':
    unittest.main()
