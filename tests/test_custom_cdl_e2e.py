"""End-to-end tests for the external CDL input pipeline (Phase 5 T5.8).

Exercises the full path: caller-provided dict (or file) →
CustomCDLProfile.from_dict / from_file → TargetChannelConfig(custom_profile=p)
→ MIMO_OTA_Simulator.run(synthesis_method='strict_pfs')
→ PropsimASCIIExporter.export(direction='dl')
→ .asc files on disk.

This is the path MIMO-First (or any external service) takes when generating
hardware drive files from its own ray-tracing CDL output.
"""
from __future__ import annotations

import glob
import json
import os
import sys
import tempfile
import unittest
import warnings

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mimo_ota_simulator.cdl_schema import CustomCDLProfile
from mimo_ota_simulator.data_models import (
    AntennaArrayConfig,
    ChamberConfig,
    PropsimExportConfig,
    TargetChannelConfig,
)
from mimo_ota_simulator.exporters import PropsimASCIIExporter
from mimo_ota_simulator.simulator import MIMO_OTA_Simulator


FULL_CDL_DICT = {
    'center_frequency_hz': 2.6e9,
    'pathloss_db': 88.5,
    'ue_velocity_mps': [4.17, 0.0, 0.0],
    'is_los': True,
    'k_factor_db': 10.0,
    'clusters': [
        # First cluster = LOS path (gets K-factor boost)
        {'delay_s': 0.0, 'power_linear': 1.0,
         'aoa_deg': 0.0, 'aod_deg': 0.0,
         'zoa_deg': 90.0, 'zod_deg': 90.0,
         'as_aoa_deg': 1.0, 'as_aod_deg': 0.5,
         'xpr_db': 12.0,
         'initial_phases_rad': [0.7853981633974483, 0.0, 0.0, 0.0]},  # π/4
        # NLOS clusters
        {'delay_s': 2.0e-7, 'power_linear': 0.5,
         'aoa_deg': 30.0, 'aod_deg': -15.0,
         'as_aoa_deg': 5.0, 'as_aod_deg': 2.0,
         'xpr_db': 7.0,
         'initial_phases_rad': [1.0, 0.5, -0.3, 0.8]},
        {'delay_s': 5.0e-7, 'power_linear': 0.3,
         'aoa_deg': -45.0, 'aod_deg': 60.0,
         'as_aoa_deg': 8.0, 'as_aod_deg': 4.0,
         'xpr_db': 7.0},
        {'delay_s': 1.0e-6, 'power_linear': 0.2,
         'aoa_deg': 120.0, 'aod_deg': -80.0,
         'as_aoa_deg': 15.0, 'as_aod_deg': 10.0},
    ],
}


class TestExternalCDLToASCEndToEnd(unittest.TestCase):
    """The complete external CDL → Propsim DL ASC pipeline."""

    def setUp(self):
        self.base = '_e2e_custom_cdl.asc'
        self.chamber = ChamberConfig(
            num_probes=8, distribution='ring', radius_m=1.5,
            dual_polarized=False,
            probe_antenna={'gain_dBi': 0.0, 'pattern': 'Isotropic'},
        )

    def tearDown(self):
        for f in glob.glob(self.base.replace('.asc', '*.asc')):
            try:
                os.remove(f)
            except OSError:
                pass

    def _run_e2e(self, profile):
        config = TargetChannelConfig(
            input_mode='custom',
            custom_profile=profile,
            tx_antenna=AntennaArrayConfig(num_rows=1, num_cols=2, spacing_h=0.5),
            rx_antenna=AntennaArrayConfig(num_rows=1, num_cols=2, spacing_h=0.5),
        )
        sim = MIMO_OTA_Simulator()
        np.random.seed(42)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = sim.run(self.chamber, config, synthesis_method='strict_pfs')
            cfg = PropsimExportConfig(
                filename=self.base, mode='B',
                duration_s=0.01, sample_rate_hz=1000,
            )
            PropsimASCIIExporter(cfg).export(result, direction='dl')
        return result

    def test_dict_input_strict_pfs_to_asc(self):
        """from_dict → strict_pfs → DL ASC files generated, sensible size."""
        profile = CustomCDLProfile.from_dict(FULL_CDL_DICT)
        result = self._run_e2e(profile)

        # Synthesizer output sanity
        A = result['transmit_mapping_matrix_A']
        self.assertEqual(A.shape, (8, 2))  # 8 probes × 2 Tx
        self.assertFalse(np.any(np.isnan(A)))
        self.assertFalse(np.any(np.isinf(A)))

        # ASC files generated
        files = sorted(glob.glob(self.base.replace('.asc', '*.asc')))
        self.assertGreater(len(files), 0, "No ASC files generated")
        total_bytes = sum(os.path.getsize(f) for f in files)
        self.assertGreater(total_bytes, 1000, f"ASC total too small: {total_bytes} bytes")

    def test_file_input_strict_pfs_to_asc(self):
        """from_file (JSON on disk) → identical e2e result as from_dict."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(FULL_CDL_DICT, f)
            path = f.name
        try:
            profile_file = CustomCDLProfile.from_file(path)
            profile_dict = CustomCDLProfile.from_dict(FULL_CDL_DICT)
            # Profiles equivalent (Pydantic == on validated models)
            self.assertEqual(profile_file, profile_dict)

            result = self._run_e2e(profile_file)
            files = sorted(glob.glob(self.base.replace('.asc', '*.asc')))
            self.assertGreater(len(files), 0)
        finally:
            os.remove(path)

    def test_k_factor_visible_in_cluster_powers(self):
        """e2e: K-factor 10 dB applied; first cluster carries 10/11 of power."""
        profile = CustomCDLProfile.from_dict(FULL_CDL_DICT)
        result = self._run_e2e(profile)
        # Inspect cluster powers in the synthesizer result
        clusters = result['clusters']
        self.assertAlmostEqual(clusters[0].power, 10.0 / 11.0, places=10)
        nlos_total = sum(c.power for c in clusters[1:])
        self.assertAlmostEqual(nlos_total, 1.0 / 11.0, places=10)

    def test_xpr_plumbed_to_engine_cluster(self):
        """e2e: xpr_db converted to xpr_linear and stored on EngineCluster."""
        profile = CustomCDLProfile.from_dict(FULL_CDL_DICT)
        result = self._run_e2e(profile)
        clusters = result['clusters']
        # First cluster: xpr_db=12 → linear = 10^1.2 ≈ 15.85
        self.assertAlmostEqual(clusters[0].xpr_linear, 10 ** 1.2, places=6)
        # Last cluster: no xpr_db → None
        self.assertIsNone(clusters[3].xpr_linear)

    def test_initial_phases_plumbed_to_engine_cluster(self):
        """e2e: initial_phases_rad list → numpy array on EngineCluster."""
        profile = CustomCDLProfile.from_dict(FULL_CDL_DICT)
        result = self._run_e2e(profile)
        clusters = result['clusters']
        # First cluster has [π/4, 0, 0, 0]
        self.assertIsNotNone(clusters[0].initial_phases)
        np.testing.assert_allclose(
            clusters[0].initial_phases, [np.pi / 4, 0.0, 0.0, 0.0]
        )
        # Last cluster: no initial_phases
        self.assertIsNone(clusters[3].initial_phases)

    def test_minimal_required_fields_only(self):
        """Schema's required-only fields (no XPR, K-factor, init phases) still works e2e."""
        minimal = {
            'center_frequency_hz': 2.6e9,
            'pathloss_db': 90.0,
            'ue_velocity_mps': [0.0, 0.0, 0.0],
            'is_los': False,
            'clusters': [
                {'delay_s': 0.0, 'power_linear': 1.0,
                 'aoa_deg': 0.0, 'aod_deg': 0.0,
                 'as_aoa_deg': 5.0, 'as_aod_deg': 2.0},
            ],
        }
        profile = CustomCDLProfile.from_dict(minimal)
        result = self._run_e2e(profile)
        files = glob.glob(self.base.replace('.asc', '*.asc'))
        self.assertGreater(len(files), 0)


if __name__ == '__main__':
    unittest.main()
