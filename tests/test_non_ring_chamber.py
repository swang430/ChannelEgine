"""Tests for P2-7 non-ring chamber probe geometry support.

Cross-repo trigger: MIMO-First docs/architecture/channel-engine-data-flow.md
(2026-05-19) flagged "ring-only silent constraint" as cross-repo work item.
"""
from __future__ import annotations

import os
import sys
import unittest
import warnings

import numpy as np
from pydantic import ValidationError

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mimo_ota_simulator.data_models import (
    AntennaArrayConfig,
    ChamberConfig,
    ProbePosition,
    TargetChannelConfig,
)
from mimo_ota_simulator.simulator import MIMO_OTA_Simulator


class TestProbePositionSchema(unittest.TestCase):
    """T8.1: ProbePosition validation."""

    def test_minimal_azimuth_only(self):
        """Elevation defaults to 0."""
        p = ProbePosition(azimuth_deg=45.0)
        self.assertEqual(p.azimuth_deg, 45.0)
        self.assertEqual(p.elevation_deg, 0.0)

    def test_full(self):
        p = ProbePosition(azimuth_deg=30.0, elevation_deg=15.0)
        self.assertEqual(p.azimuth_deg, 30.0)
        self.assertEqual(p.elevation_deg, 15.0)

    def test_azimuth_range(self):
        ProbePosition(azimuth_deg=-360.0)
        ProbePosition(azimuth_deg=360.0)
        with self.assertRaises(ValidationError):
            ProbePosition(azimuth_deg=361.0)
        with self.assertRaises(ValidationError):
            ProbePosition(azimuth_deg=-361.0)

    def test_elevation_range(self):
        ProbePosition(azimuth_deg=0.0, elevation_deg=-90.0)
        ProbePosition(azimuth_deg=0.0, elevation_deg=90.0)
        with self.assertRaises(ValidationError):
            ProbePosition(azimuth_deg=0.0, elevation_deg=91.0)


class TestChamberConfigDistributionConsistency(unittest.TestCase):
    """T8.1: ChamberConfig validator: distribution × probe_positions."""

    def _common_kwargs(self):
        return dict(
            num_probes=4, radius_m=1.5, dual_polarized=False,
            probe_antenna={'gain_dBi': 0.0, 'pattern': 'Isotropic'},
        )

    def test_ring_no_positions(self):
        """distribution='ring' + probe_positions=None is the legacy default."""
        c = ChamberConfig(distribution='ring', **self._common_kwargs())
        self.assertIsNone(c.probe_positions)

    def test_ring_with_positions_allowed(self):
        """distribution='ring' + probe_positions set is allowed (positions ignored)."""
        positions = [ProbePosition(azimuth_deg=a) for a in [0, 90, 180, 270]]
        c = ChamberConfig(distribution='ring', probe_positions=positions, **self._common_kwargs())
        self.assertIsNotNone(c.probe_positions)

    def test_custom_requires_positions(self):
        with self.assertRaises(ValidationError) as ctx:
            ChamberConfig(distribution='custom', **self._common_kwargs())
        self.assertIn('probe_positions', str(ctx.exception))

    def test_multi_ring_requires_positions(self):
        with self.assertRaises(ValidationError):
            ChamberConfig(distribution='multi-ring', **self._common_kwargs())

    def test_custom_positions_count_mismatch(self):
        """len(probe_positions) must equal num_probes."""
        positions = [ProbePosition(azimuth_deg=a) for a in [0, 120, 240]]  # 3, expecting 4
        with self.assertRaises(ValidationError) as ctx:
            ChamberConfig(distribution='custom', probe_positions=positions, **self._common_kwargs())
        msg = str(ctx.exception)
        self.assertIn('3', msg)
        self.assertIn('num_probes', msg)


class TestCalculateProbePositions(unittest.TestCase):
    """T8.2: _calculate_probe_positions for ring + custom."""

    def setUp(self):
        self.sim = MIMO_OTA_Simulator()

    def test_ring_unchanged_bit_identical(self):
        """Ring layout returns the same Cartesian as pre-P2-7."""
        c = ChamberConfig(
            num_probes=8, radius_m=1.5, distribution='ring', dual_polarized=False,
            probe_antenna={'gain_dBi': 0.0, 'pattern': 'Isotropic'},
        )
        pos = self.sim._calculate_probe_positions(c)
        self.assertEqual(pos.shape, (8, 3))
        # First probe at (radius, 0, 0)
        self.assertAlmostEqual(pos[0, 0], 1.5, places=12)
        self.assertAlmostEqual(pos[0, 1], 0.0, places=12)
        self.assertAlmostEqual(pos[0, 2], 0.0, places=12)
        # All z=0 (horizontal plane)
        np.testing.assert_array_equal(pos[:, 2], np.zeros(8))
        # All at radius (within float tolerance)
        radii = np.linalg.norm(pos, axis=1)
        np.testing.assert_allclose(radii, 1.5, rtol=1e-12)

    def test_custom_horizontal_plane_matches_ring(self):
        """Custom with same angles as ring uniform == ring output."""
        N = 8
        angles_deg = [i * 360.0 / N for i in range(N)]
        positions = [ProbePosition(azimuth_deg=a) for a in angles_deg]
        c_custom = ChamberConfig(
            num_probes=N, radius_m=1.5, distribution='custom',
            probe_positions=positions, dual_polarized=False,
            probe_antenna={'gain_dBi': 0.0, 'pattern': 'Isotropic'},
        )
        c_ring = ChamberConfig(
            num_probes=N, radius_m=1.5, distribution='ring', dual_polarized=False,
            probe_antenna={'gain_dBi': 0.0, 'pattern': 'Isotropic'},
        )
        pos_custom = self.sim._calculate_probe_positions(c_custom)
        pos_ring = self.sim._calculate_probe_positions(c_ring)
        np.testing.assert_allclose(pos_custom, pos_ring, rtol=1e-12, atol=1e-12)

    def test_custom_sparse_layout(self):
        """Sparse 4-probe at 0/30/60/90 deg (not uniform)."""
        positions = [ProbePosition(azimuth_deg=a) for a in [0.0, 30.0, 60.0, 90.0]]
        c = ChamberConfig(
            num_probes=4, radius_m=2.0, distribution='custom',
            probe_positions=positions, dual_polarized=False,
            probe_antenna={'gain_dBi': 0.0, 'pattern': 'Isotropic'},
        )
        pos = self.sim._calculate_probe_positions(c)
        self.assertEqual(pos.shape, (4, 3))
        # Probe 0 at (2, 0, 0)
        np.testing.assert_allclose(pos[0], [2.0, 0.0, 0.0], atol=1e-12)
        # Probe 1 at radius=2, az=30°: (2*cos30°, 2*sin30°, 0) = (~1.732, 1.0, 0)
        np.testing.assert_allclose(pos[1], [2.0*np.cos(np.pi/6), 1.0, 0.0], atol=1e-12)
        # All at radius=2
        np.testing.assert_allclose(np.linalg.norm(pos, axis=1), 2.0, rtol=1e-12)

    def test_custom_with_elevation(self):
        """Non-zero elevation → z component populated."""
        positions = [
            ProbePosition(azimuth_deg=0.0, elevation_deg=30.0),
            ProbePosition(azimuth_deg=180.0, elevation_deg=-30.0),
        ]
        c = ChamberConfig(
            num_probes=2, radius_m=1.0, distribution='custom',
            probe_positions=positions, dual_polarized=False,
            probe_antenna={'gain_dBi': 0.0, 'pattern': 'Isotropic'},
        )
        pos = self.sim._calculate_probe_positions(c)
        # |z| = radius * sin(30°) = 0.5
        np.testing.assert_allclose(np.abs(pos[:, 2]), 0.5, atol=1e-12)
        # Probe 0 elevation up (+z), probe 1 down (-z)
        self.assertGreater(pos[0, 2], 0)
        self.assertLess(pos[1, 2], 0)


class TestCalculateWeightsForCluster(unittest.TestCase):
    """T8.3: _calculate_weights_for_cluster uses real azimuths when custom."""

    def setUp(self):
        self.sim = MIMO_OTA_Simulator()

    def test_ring_unchanged(self):
        """Ring: weights match pre-P2-7 (linspace assumption)."""
        from channel_model_38901.common.data_models import EngineCluster
        c = ChamberConfig(
            num_probes=8, radius_m=1.5, distribution='ring', dual_polarized=False,
            probe_antenna={'gain_dBi': 0.0, 'pattern': 'Isotropic'},
        )
        # Cluster with aoa=45° matches probe[1] for N=8 ring (probe_angles = 0, 45, 90, ...)
        cluster = EngineCluster(aoa_phi=45.0, as_aoa=5.0, power=1.0, delay=0.0)
        w = self.sim._calculate_weights_for_cluster(cluster, c)
        # Probe 1 should be the max weight
        idx_max = int(np.argmax(w[:, 0]))
        self.assertEqual(idx_max, 1)

    def test_custom_uses_real_azimuth(self):
        """Custom: cluster pointing at a specific probe's actual azimuth → that probe wins."""
        from channel_model_38901.common.data_models import EngineCluster
        # Sparse 4-probe layout at 0/77/154/231 (irregular, NOT 0/90/180/270)
        positions = [ProbePosition(azimuth_deg=a) for a in [0.0, 77.0, 154.0, 231.0]]
        c = ChamberConfig(
            num_probes=4, radius_m=1.5, distribution='custom',
            probe_positions=positions, dual_polarized=False,
            probe_antenna={'gain_dBi': 0.0, 'pattern': 'Isotropic'},
        )
        # Cluster at 77° should peak at probe[1] (the one actually at 77°),
        # NOT probe[1] of ring assumption (which would be 90°).
        cluster = EngineCluster(aoa_phi=77.0, as_aoa=3.0, power=1.0, delay=0.0)
        w = self.sim._calculate_weights_for_cluster(cluster, c)
        idx_max = int(np.argmax(w[:, 0]))
        self.assertEqual(idx_max, 1,
            msg=f"Cluster at 77° did NOT pick probe at azimuth=77°; weights={w[:, 0]}")

    def test_custom_ring_equivalent(self):
        """Custom probe_positions same as ring → weights bit-identical to ring path."""
        from channel_model_38901.common.data_models import EngineCluster
        N = 8
        angles_deg = [i * 360.0 / N for i in range(N)]
        c_custom = ChamberConfig(
            num_probes=N, radius_m=1.5, distribution='custom',
            probe_positions=[ProbePosition(azimuth_deg=a) for a in angles_deg],
            dual_polarized=False,
            probe_antenna={'gain_dBi': 0.0, 'pattern': 'Isotropic'},
        )
        c_ring = ChamberConfig(
            num_probes=N, radius_m=1.5, distribution='ring', dual_polarized=False,
            probe_antenna={'gain_dBi': 0.0, 'pattern': 'Isotropic'},
        )
        cluster = EngineCluster(aoa_phi=42.0, as_aoa=8.0, power=1.0, delay=0.0)
        w_custom = self.sim._calculate_weights_for_cluster(cluster, c_custom)
        w_ring = self.sim._calculate_weights_for_cluster(cluster, c_ring)
        np.testing.assert_allclose(w_custom, w_ring, rtol=1e-12, atol=1e-12)


class TestE2ENonRingRun(unittest.TestCase):
    """T8.4: full sim.run() with non-ring chamber."""

    def test_sparse_chamber_runs_end_to_end(self):
        """Non-ring chamber + strict_pfs → no exception, sane A matrix shape."""
        from mimo_ota_simulator.cdl_schema import CustomCDLProfile
        # 4 probes irregularly placed
        positions = [ProbePosition(azimuth_deg=a) for a in [10.0, 95.0, 180.0, 270.0]]
        chamber = ChamberConfig(
            num_probes=4, radius_m=1.5, distribution='custom',
            probe_positions=positions, dual_polarized=False,
            probe_antenna={'gain_dBi': 0.0, 'pattern': 'Isotropic'},
        )
        profile = CustomCDLProfile.from_dict({
            'center_frequency_hz': 2.6e9, 'pathloss_db': 85.0,
            'ue_velocity_mps': [0.0, 0.0, 0.0], 'is_los': False,
            'clusters': [{'delay_s': 0.0, 'power_linear': 1.0,
                          'aoa_deg': 95.0, 'aod_deg': 0.0,
                          'as_aoa_deg': 5.0, 'as_aod_deg': 2.0}],
        })
        config = TargetChannelConfig(
            input_mode='custom', custom_profile=profile,
            tx_antenna=AntennaArrayConfig(num_cols=1),
            rx_antenna=AntennaArrayConfig(num_cols=1),
        )
        np.random.seed(42)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = MIMO_OTA_Simulator().run(chamber, config, synthesis_method='strict_pfs')
        A = result['transmit_mapping_matrix_A']
        self.assertEqual(A.shape, (4, 1))
        self.assertFalse(np.any(np.isnan(A)))
        self.assertFalse(np.any(np.isinf(A)))

    def test_ring_e2e_unchanged_vs_pre_p2_7(self):
        """Ring layout end-to-end matches pre-P2-7 ray output bit-identically.

        Uses existing golden_mimo_ota_ray_results.npz baseline via the existing
        TestRayRegression-style approach (same config, same seed, same A).
        """
        # Reuse the golden file generation config from create_golden_files.py
        # to confirm ring path is bit-identical post-P2-7.
        import glob
        golden_path = os.path.join(
            os.path.dirname(__file__), 'golden_mimo_ota_ray_results.npz',
        )
        self.assertTrue(os.path.exists(golden_path), f"Golden file missing: {golden_path}")
        golden_A = np.load(golden_path)['golden_A']

        chamber = ChamberConfig(
            num_probes=8, dual_polarized=False, distribution='ring', radius_m=1.0,
            probe_antenna={'gain_dBi': 0.0, 'pattern': 'Isotropic'},
        )
        channel = TargetChannelConfig(
            model_name='UMi-StreetCanyon', cluster_model_name='CDL-A',
            center_frequency_hz=3.5e9,
            tx_antenna=AntennaArrayConfig(num_cols=4, array_type='ULA'),
            rx_antenna=AntennaArrayConfig(num_cols=2, array_type='ULA'),
            random_seed=42,
        )
        np.random.seed(42)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = MIMO_OTA_Simulator().run(chamber, channel, synthesis_method='ray')
        np.testing.assert_allclose(
            result['transmit_mapping_matrix_A'], golden_A, rtol=1e-6, atol=1e-6,
            err_msg="Ring path regression: P2-7 changed ring-mode output",
        )


if __name__ == '__main__':
    unittest.main()
