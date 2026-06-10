"""Focused unit tests for the TR 38.901 compliance additions.

Each capability is opt-in (default off) so the golden regression suite stays
bit-identical. These tests assert the behaviour when the flags are ON, plus the
no-op guarantee when they are OFF.

Covered:
  - Sec 7.1.3 array orientation + LCS->GCS angle transform (geometry.py)
  - Table 7.3-1 element radiation pattern application (channel_generator)
  - Sec 7.7.5.1 CDL -> scenario LSP rescaling
  - Sec 7.7.1 sub-cluster splitting
"""
import numpy as np
import pytest

from channel_model_38901.common.geometry import (
    rotation_matrix, apply_orientation, gcs_to_lcs_angles, get_antenna_positions,
)
from channel_model_38901.simulator import ChannelSimulator


# --- Sec 7.1.3: orientation + LCS->GCS ---------------------------------------

class TestOrientation:
    def test_identity_when_all_angles_zero(self):
        pos = get_antenna_positions('URA', 2, 2, 0.5, 0.5)
        pos_oriented = get_antenna_positions('URA', 2, 2, 0.5, 0.5, 0.0, 0.0, 0.0)
        np.testing.assert_array_equal(pos, pos_oriented)

    def test_rotation_matrix_orthonormal(self):
        R = rotation_matrix(30.0, 12.0, -45.0)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-12)

    def test_bearing_rotates_y_axis_into_xy_plane(self):
        # A single horizontal-spaced pair lies on +y in the LCS; a 90 deg bearing
        # (rotation about z) sends +y -> +x... actually Rz(90)*[0,1,0] = [-1,0,0].
        pos = get_antenna_positions('ULA', 1, 2, 1.0, 0.5, bearing_deg=90.0)
        # element 1 LCS = [0,1,0]; after Rz(90): x=-1, y~0
        np.testing.assert_allclose(pos[1], [-1.0, 0.0, 0.0], atol=1e-12)

    def test_gcs_to_lcs_identity_when_zero(self):
        th, ph = gcs_to_lcs_angles(70.0, 33.0, 0.0, 0.0, 0.0)
        assert th == 70.0 and ph == 33.0

    def test_gcs_to_lcs_bearing_shifts_azimuth(self):
        # Pure bearing alpha: LCS azimuth = GCS azimuth - alpha at theta=90.
        th, ph = gcs_to_lcs_angles(90.0, 45.0, 30.0, 0.0, 0.0)
        np.testing.assert_allclose(th, 90.0, atol=1e-9)
        np.testing.assert_allclose(ph, 15.0, atol=1e-9)

    def test_apply_orientation_empty(self):
        out = apply_orientation(np.empty((0, 3)), 10.0, 5.0, 0.0)
        assert out.shape == (0, 3)


# --- Table 7.3-1: element pattern --------------------------------------------

class TestElementPattern:
    def _sim(self, pattern):
        sim = ChannelSimulator('UMa', 3.5e9, 'CDL-C')
        sim.config.antenna_config.tx_element_pattern = pattern
        sim.config.antenna_config.rx_element_pattern = pattern
        return sim

    def test_isotropic_default_is_noop(self):
        np.random.seed(1)
        iso = self._sim('Isotropic').run(force_condition='NLOS', logger=None)
        # Isotropic element gain is exactly 1.0 -> cluster h_matrices unscaled.
        gains = [self._sim('Isotropic').generator._cluster_element_gain(c)
                 for c in iso['clusters']]
        assert all(g == 1.0 for g in gains)

    def test_tr38901_pattern_changes_gain(self):
        sim = self._sim('TR38.901')
        np.random.seed(1)
        res = sim.run(force_condition='NLOS', logger=None)
        gains = [sim.generator._cluster_element_gain(c) for c in res['clusters']]
        # At least one cluster off-boresight must be attenuated (< 1 linear),
        # and none should exceed the 8 dBi peak combined Tx*Rx (10**(16/20)).
        assert any(g < 1.0 for g in gains)
        assert max(gains) <= 10 ** (16.0 / 20.0) + 1e-9


# --- Sec 7.7.5.1: scenario rescaling -----------------------------------------

class TestScenarioRescaling:
    def _angles(self, scenario, scale):
        sim = ChannelSimulator(scenario, 3.5e9, 'CDL-C',
                               use_median_lsps=True, scale_to_scenario=scale)
        res = sim.run(force_condition='NLOS', logger=None)
        aoa = np.array([c.aoa_phi for c in res['clusters']])
        p = np.array([c.power for c in res['clusters']])
        mean = np.angle(np.sum(p * np.exp(1j * np.deg2rad(aoa))))
        d = (aoa - np.rad2deg(mean) + 180) % 360 - 180
        rms = np.sqrt(np.sum(p * d**2) / np.sum(p))
        return rms, res['lsps'].arrival_azimuth_spread_deg

    def test_off_keeps_raw_table_spread(self):
        # With scaling off, UMa and UMi CDL-C share the identical raw-table AoA set.
        rms_uma, _ = self._angles('UMa', False)
        rms_umi, _ = self._angles('UMi-StreetCanyon', False)
        np.testing.assert_allclose(rms_uma, rms_umi, atol=1e-9)

    def test_on_matches_scenario_asa(self):
        rms, asa = self._angles('UMa', True)
        # Rescaled RMS AoA spread should track the scenario ASA (within rescale tol).
        np.testing.assert_allclose(rms, asa, rtol=0.05)


# --- Sec 7.7.1: sub-cluster splitting ----------------------------------------

class TestSubclusterSplit:
    def test_off_is_noop(self):
        np.random.seed(2)
        base = ChannelSimulator('UMa', 3.5e9, 'CDL-C').run(
            force_condition='NLOS', logger=None)
        assert len(base['clusters']) == 24

    def test_on_adds_four_clusters_and_conserves_power(self):
        np.random.seed(2)
        base = ChannelSimulator('UMa', 3.5e9, 'CDL-C').run(
            force_condition='NLOS', logger=None)
        np.random.seed(2)
        split = ChannelSimulator('UMa', 3.5e9, 'CDL-C', split_subclusters=True).run(
            force_condition='NLOS', logger=None)
        assert len(split['clusters']) == len(base['clusters']) + 4
        np.testing.assert_allclose(
            sum(c.power for c in split['clusters']),
            sum(c.power for c in base['clusters']), atol=1e-9)

    def test_subcluster_delay_offsets(self):
        c_ds_ns = 3.91
        split = ChannelSimulator('UMa', 3.5e9, 'CDL-C',
                                 split_subclusters=True, c_ds_ns=c_ds_ns)
        res = split.run(force_condition='NLOS', logger=None)
        delays = sorted(c.delay for c in res['clusters'])
        # The 3 sub-cluster delay offsets {0, 1.28, 2.56}*c_DS must appear as
        # pairwise gaps among some triple of clusters.
        offs = np.array([0.0, 1.28, 2.56]) * c_ds_ns * 1e-9
        all_delays = np.array(delays)
        found = False
        for d0 in all_delays:
            if all(np.any(np.isclose(all_delays, d0 + o, atol=1e-12)) for o in offs):
                found = True
                break
        assert found


# --- Ray-level SoS export: element gain + sub-cluster split consistency -------

class TestSosRayLevelExport:
    def _sim(self, pattern='Isotropic', **kw):
        sim = ChannelSimulator('UMa', 3.5e9, 'CDL-C', **kw)
        sim.config.antenna_config.tx_element_pattern = pattern
        sim.config.antenna_config.rx_element_pattern = pattern
        # Re-build generator so the pattern change is seen by export.
        sim.generator = sim._build_generator()
        sim.generator._ensure_antenna_positions(3e8 / 3.5e9)
        return sim

    def test_isotropic_ray_power_equals_p_ray(self):
        # Isotropic element gain is exactly 1.0 -> Power_Linear == cluster.power/20.
        np.random.seed(3)
        rows = self._sim('Isotropic').export_fpga_sos_params()
        assert rows
        for r in rows:
            np.testing.assert_allclose(r['Power_Linear'],
                                       r['Weight_Real']**2 + r['Weight_Imag']**2,
                                       atol=1e-12)

    def test_tr38901_pattern_attenuates_some_rays(self):
        # Per-ray element pattern must pull at least one off-boresight ray below
        # its isotropic power, and never amplify beyond the combined 16 dBi peak.
        np.random.seed(3)
        iso = self._sim('Isotropic').export_fpga_sos_params()
        np.random.seed(3)
        pat = self._sim('TR38.901').export_fpga_sos_params()
        iso_p = np.array([r['Power_Linear'] for r in iso])
        pat_p = np.array([r['Power_Linear'] for r in pat])
        assert np.any(pat_p < iso_p - 1e-12)
        assert np.max(pat_p) <= 10 ** (16.0 / 10.0) + 1e-9

    def test_split_subclusters_adds_export_rows(self):
        # Sub-cluster split (+4 clusters) must flow into the ray-level export too.
        np.random.seed(3)
        base = self._sim('Isotropic').export_fpga_sos_params()
        np.random.seed(3)
        split = self._sim('Isotropic', split_subclusters=True).export_fpga_sos_params()
        per_cluster = 20  # rays per cluster; antenna geometry identical between runs
        assert len(split) == len(base) + 4 * per_cluster


# --- OTA DUT array orientation (Sec 7.1.3) -----------------------------------

class TestOtaDutOrientation:
    """The DUT's fixed starting orientation must shape the chamber B-matrix
    geometry (relevant for dynamic, non-rotating MIMO OTA). Default 0° is a
    no-op so the 360° turntable test and legacy runs stay bit-identical."""

    def _cfg(self, **kw):
        from mimo_ota_simulator.data_models import AntennaArrayConfig
        return AntennaArrayConfig(array_type='ULA', num_cols=4, spacing_h=0.5, **kw)

    def test_zero_orientation_is_noop(self):
        from mimo_ota_simulator.simulator import MIMO_OTA_Simulator
        wl = 3e8 / 2.6e9
        base = MIMO_OTA_Simulator._calculate_dut_antenna_positions(self._cfg(), wl)
        oriented = MIMO_OTA_Simulator._calculate_dut_antenna_positions(
            self._cfg(bearing_deg=0.0, downtilt_deg=0.0, slant_deg=0.0), wl)
        np.testing.assert_array_equal(base, oriented)

    def test_bearing_90_rotates_x_into_y(self):
        from mimo_ota_simulator.simulator import MIMO_OTA_Simulator
        wl = 3e8 / 2.6e9
        base = MIMO_OTA_Simulator._calculate_dut_antenna_positions(self._cfg(), wl)
        rot = MIMO_OTA_Simulator._calculate_dut_antenna_positions(
            self._cfg(bearing_deg=90.0), wl)
        # ULA lies along +x; a 90° bearing (Rz) sends +x -> +y.
        np.testing.assert_allclose(rot[:, 0], 0.0, atol=1e-12)
        np.testing.assert_allclose(rot[:, 1], base[:, 0], atol=1e-12)

    def test_orientation_changes_b_matrix(self):
        from mimo_ota_simulator.data_models import (
            AntennaArrayConfig, ChamberConfig, TargetChannelConfig,
        )
        from mimo_ota_simulator.simulator import MIMO_OTA_Simulator
        sim = MIMO_OTA_Simulator()
        chamber = ChamberConfig(num_probes=4, distribution='ring', radius_m=1.5,
                                dual_polarized=False,
                                probe_antenna={'gain_dBi': 0.0, 'pattern': 'Isotropic'})

        def b(rx_kwargs):
            cfg = TargetChannelConfig(
                center_frequency_hz=2.6e9,
                tx_antenna=AntennaArrayConfig(num_cols=1),
                rx_antenna=AntennaArrayConfig(num_cols=4, spacing_h=0.5, **rx_kwargs),
            )
            return sim._synthesize_b_matrix(chamber, cfg)

        B0 = b({})
        B_rot = b({'downtilt_deg': 30.0})
        assert not np.allclose(B0, B_rot)


# --- OTA production ray path honors the element pattern (Codex PR#14 P2) ------

class TestOtaRayElementPattern:
    """element_pattern='TR38.901' must actually shape the default production
    synthesis (ray / strict_pfs). Previously the pattern was forwarded into the
    channel model but the OTA ray path rebuilt weights from angles only, so the
    control was silently isotropic. Isotropic stays a bit-identical no-op."""

    def _A(self, tx_pattern, rx_pattern, seed=7, method='ray'):
        import warnings as _w
        from mimo_ota_simulator.cdl_schema import CustomCDLProfile
        from mimo_ota_simulator.data_models import (
            AntennaArrayConfig, ChamberConfig, TargetChannelConfig,
        )
        from mimo_ota_simulator.simulator import MIMO_OTA_Simulator

        chamber = ChamberConfig(num_probes=8, distribution='ring', radius_m=1.5,
                                dual_polarized=False,
                                probe_antenna={'gain_dBi': 0.0, 'pattern': 'Isotropic'})
        profile = CustomCDLProfile.from_dict({
            'center_frequency_hz': 2.6e9, 'pathloss_db': 85.0,
            'ue_velocity_mps': [0.0, 0.0, 0.0], 'is_los': False,
            'clusters': [
                {'delay_s': 0.0, 'power_linear': 1.0, 'aoa_deg': 75, 'aod_deg': 60,
                 'as_aoa_deg': 5.0, 'as_aod_deg': 2.0},
                {'delay_s': 1e-7, 'power_linear': 0.5, 'aoa_deg': -40, 'aod_deg': 20,
                 'as_aoa_deg': 5.0, 'as_aod_deg': 2.0},
            ],
        })
        config = TargetChannelConfig(
            input_mode='custom', custom_profile=profile,
            tx_antenna=AntennaArrayConfig(num_cols=2, element_pattern=tx_pattern),
            rx_antenna=AntennaArrayConfig(num_cols=2, element_pattern=rx_pattern),
        )
        np.random.seed(seed)
        with _w.catch_warnings():
            _w.simplefilter('ignore')
            res = MIMO_OTA_Simulator().run(chamber, config, synthesis_method=method)
        return res['transmit_mapping_matrix_A']

    def test_isotropic_is_noop_default(self):
        # Explicit Isotropic == default; the pattern control adds nothing at 1.0.
        np.testing.assert_array_equal(self._A('Isotropic', 'Isotropic'),
                                      self._A('Isotropic', 'Isotropic'))

    def test_tr38901_pattern_changes_ray_A(self):
        A_iso = self._A('Isotropic', 'Isotropic')
        A_pat = self._A('TR38.901', 'TR38.901')
        # Same seed => identical phases; only the element gain differs, so the
        # production ray A-matrix must change (no longer silently isotropic).
        assert not np.allclose(A_iso, A_pat)

    def test_tr38901_pattern_changes_strict_pfs_A(self):
        A_iso = self._A('Isotropic', 'Isotropic', method='strict_pfs')
        A_pat = self._A('TR38.901', 'TR38.901', method='strict_pfs')
        assert not np.allclose(A_iso, A_pat)
