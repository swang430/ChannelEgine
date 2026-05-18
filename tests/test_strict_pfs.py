"""Tests for strict_pfs synthesis method (Phase 1+).

Currently covers:
- T1.2 phi_init data structure lock: exporter must handle both scalar (ray mode)
  and (N_probes,) vector (strict_pfs mode) ray_components.phi_init.

Future Phase 2 statistical tests (T2.1-T2.3) will land here too.
"""
import glob
import os
import sys
import warnings
import unittest

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mimo_ota_simulator.data_models import (
    AntennaArrayConfig,
    ChamberConfig,
    PropsimExportConfig,
    TargetChannelConfig,
)
from mimo_ota_simulator.exporters import PropsimASCIIExporter
from mimo_ota_simulator.simulator import MIMO_OTA_Simulator


def _make_baseline_ray_result():
    """Run ray mode and return its result; used as scaffolding to test exporter
    against both scalar phi_init (as-is) and vector phi_init (mutated)."""
    chamber = ChamberConfig(
        num_probes=8, distribution='ring', radius_m=1.5, dual_polarized=False,
        probe_antenna={'gain_dBi': 0.0, 'pattern': 'Isotropic'},
    )
    channel = TargetChannelConfig(
        model_name='UMa', cluster_model_name='CDL-C',
        center_frequency_hz=3.5e9, random_seed=42,
        tx_antenna=AntennaArrayConfig(num_rows=1, num_cols=1),
        rx_antenna=AntennaArrayConfig(num_rows=1, num_cols=1),
        force_condition='NLOS',
        ue_velocity=(3.0, 0.0, 0.0),
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return chamber, channel, MIMO_OTA_Simulator().run(
            chamber, channel, synthesis_method='ray'
        )


class TestPhiInitDataContract(unittest.TestCase):
    """Locks the data structure decision from design §3:
    - ray mode produces scalar phi_init
    - strict_pfs mode produces (N_probes,) vector phi_init
    - exporter handles both via np.ndim(phi)==0 dispatch (exporters.py L250)
    """

    def setUp(self):
        self.chamber, self.channel, self.result = _make_baseline_ray_result()
        self.base = '_strict_pfs_test.asc'

    def tearDown(self):
        for f in glob.glob(self.base.replace('.asc', '*.asc')):
            try:
                os.remove(f)
            except OSError:
                pass

    def test_ray_mode_produces_scalar_phi_init(self):
        """Lock: ray mode's phi_init is a Python float (np.ndim == 0)."""
        for rc in self.result['ray_components']:
            self.assertEqual(
                np.ndim(rc['phi_init']), 0,
                msg=f"ray mode phi_init should be scalar (ndim=0); got ndim={np.ndim(rc['phi_init'])}"
            )

    def _export_and_count_bytes(self, ray_components):
        result = dict(self.result)
        result['ray_components'] = ray_components
        cfg = PropsimExportConfig(
            filename=self.base, mode='B', duration_s=0.01, sample_rate_hz=1000,
        )
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            PropsimASCIIExporter(cfg).export(result)
        files = sorted(glob.glob(self.base.replace('.asc', '*.asc')))
        return len(files), sum(os.path.getsize(f) for f in files)

    def test_exporter_handles_scalar_phi_init(self):
        """Baseline: ray mode's scalar phi_init exports successfully."""
        nfiles, nbytes = self._export_and_count_bytes(self.result['ray_components'])
        self.assertGreater(nfiles, 0)
        self.assertGreater(nbytes, 0)

    def test_exporter_handles_vector_phi_init(self):
        """T1.2 lock: (N_probes,) vector phi_init exports successfully.

        This simulates the future strict_pfs output. If this test fails, the
        np.ndim dispatch in exporters.py _compute_dynamic_tensor was broken.
        """
        num_probes = self.chamber.num_probes
        rng = np.random.default_rng(123)
        mutated = []
        for rc in self.result['ray_components']:
            new_rc = dict(rc)
            new_rc['phi_init'] = rng.uniform(-np.pi, np.pi, size=num_probes)
            mutated.append(new_rc)

        # Sanity: confirm mutation produces vector
        self.assertEqual(np.ndim(mutated[0]['phi_init']), 1)
        self.assertEqual(mutated[0]['phi_init'].shape, (num_probes,))

        nfiles, nbytes = self._export_and_count_bytes(mutated)
        self.assertGreater(nfiles, 0)
        self.assertGreater(nbytes, 0)

    def test_exporter_handles_zero_d_numpy_phi_init(self):
        """Edge case: 0-d numpy array (np.array(0.5)) must also work like scalar.

        Locks the np.ndim==0 judgment (not np.isscalar, which is False for 0-d arrays).
        """
        mutated = []
        for rc in self.result['ray_components']:
            new_rc = dict(rc)
            new_rc['phi_init'] = np.array(float(rc['phi_init']))  # 0-d ndarray
            mutated.append(new_rc)

        self.assertEqual(np.ndim(mutated[0]['phi_init']), 0)
        self.assertFalse(np.isscalar(mutated[0]['phi_init']))  # 0-d is NOT scalar by isscalar
        # Must still export
        nfiles, _ = self._export_and_count_bytes(mutated)
        self.assertGreater(nfiles, 0)


class TestStrictPfsSmoke(unittest.TestCase):
    """End-to-end smoke tests for synthesis_method='strict_pfs' (T1.3, T1.5)."""

    def setUp(self):
        self.base = '_strict_pfs_smoke.asc'
        self.chamber = ChamberConfig(
            num_probes=8, distribution='ring', radius_m=1.5, dual_polarized=False,
            probe_antenna={'gain_dBi': 0.0, 'pattern': 'Isotropic'},
        )
        self.channel = TargetChannelConfig(
            model_name='UMa', cluster_model_name='CDL-C',
            center_frequency_hz=3.5e9, random_seed=42,
            tx_antenna=AntennaArrayConfig(num_rows=1, num_cols=1),
            rx_antenna=AntennaArrayConfig(num_rows=1, num_cols=1),
            force_condition='NLOS',
            ue_velocity=(3.0, 0.0, 0.0),
        )

    def tearDown(self):
        for f in glob.glob(self.base.replace('.asc', '*.asc')):
            try:
                os.remove(f)
            except OSError:
                pass

    def test_strict_pfs_runs_without_error(self):
        """strict_pfs path returns dict with expected keys, no NaN/Inf."""
        sim = MIMO_OTA_Simulator()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = sim.run(self.chamber, self.channel, synthesis_method='strict_pfs')
        self.assertIn('transmit_mapping_matrix_A', result)
        self.assertIn('effective_H_matrix', result)
        self.assertIn('ray_components', result)
        A = result['transmit_mapping_matrix_A']
        self.assertFalse(np.any(np.isnan(A)))
        self.assertFalse(np.any(np.isinf(A)))

    def test_strict_pfs_produces_vector_phi_init(self):
        """strict_pfs output ray_components must carry (N_probes,) vector phi_init."""
        sim = MIMO_OTA_Simulator()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = sim.run(self.chamber, self.channel, synthesis_method='strict_pfs')
        num_probes = self.chamber.num_probes
        for rc in result['ray_components']:
            phi = rc['phi_init']
            self.assertEqual(np.ndim(phi), 1)
            self.assertEqual(phi.shape, (num_probes,))

    def test_strict_pfs_ul_dl_reciprocity(self):
        """T1.5: enable_ul_dl=True with strict_pfs must work end-to-end.

        Option A reciprocity (per design §5): UL phi_init mirrors DL via negation
        (consistent with W_ul = W_dl.conj().T phase conjugation). Exporter must
        handle UL W_base shape (Tx, P) by broadcasting phi on the col axis.
        """
        channel_ul = TargetChannelConfig(
            model_name='UMa', cluster_model_name='CDL-C',
            center_frequency_hz=3.5e9, random_seed=42,
            tx_antenna=AntennaArrayConfig(num_rows=1, num_cols=2, spacing_h=0.5),
            rx_antenna=AntennaArrayConfig(num_rows=1, num_cols=2, spacing_h=0.5),
            force_condition='NLOS', ue_velocity=(3.0, 0.0, 0.0),
            enable_ul_dl=True, reciprocity_transform='conjugate_transpose',
        )
        sim = MIMO_OTA_Simulator()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = sim.run(self.chamber, channel_ul, synthesis_method='strict_pfs')

        rc_dl = result['ray_components'][0]
        rc_ul = result['ray_components_ul'][0]
        # Shape sanity
        self.assertEqual(rc_dl['W_base'].shape, (8, 2))
        self.assertEqual(rc_ul['W_base'].shape, (2, 8))
        self.assertEqual(rc_dl['phi_init'].shape, (8,))
        self.assertEqual(rc_ul['phi_init'].shape, (8,))
        # Option A: phi_ul = -phi_dl (verified per-ray for all clusters)
        for d, u in zip(result['ray_components'], result['ray_components_ul']):
            np.testing.assert_allclose(u['phi_init'], -d['phi_init'])

        # UL F64 export must succeed (axis-aware exporter dispatches col-broadcast)
        from mimo_ota_simulator.exporters import PropsimASCIIExporter
        cfg = PropsimExportConfig(
            filename=self.base, mode='B', duration_s=0.01, sample_rate_hz=1000,
        )
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            PropsimASCIIExporter(cfg).export(result, direction='ul')
        files = glob.glob(self.base.replace('.asc', '*.asc'))
        self.assertGreater(len(files), 0)

    def test_strict_pfs_differs_from_ray(self):
        """Sanity: strict_pfs and ray with same seed produce DIFFERENT A.

        The random draws differ in shape (scalar vs vector per ray), so the
        sequence of np.random.uniform calls consumes different RNG state, and
        the resulting A matrices must not match.
        """
        sim = MIMO_OTA_Simulator()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            r_ray = sim.run(self.chamber, self.channel, synthesis_method='ray')
            r_strict = sim.run(self.chamber, self.channel, synthesis_method='strict_pfs')
        diff = np.abs(
            r_ray['transmit_mapping_matrix_A'] - r_strict['transmit_mapping_matrix_A']
        ).mean()
        self.assertGreater(diff, 1e-6, msg="strict_pfs should produce different A than ray")


class TestRayRegression(unittest.TestCase):
    """T2.4: ray-mode bit-identical regression against golden baseline.

    The existing `mimo_ota_simulator/tests/test_mimo_ota_simulator.py::test_regression_against_golden_file`
    pins cluster mode (deprecated). This adds the equivalent pin for the non-deprecated 'ray'
    method — protects Phase 1 ray/strict_pfs branch merge against silent regression.
    """

    def test_ray_regression_against_golden(self):
        golden_path = os.path.join(os.path.dirname(__file__), 'golden_mimo_ota_ray_results.npz')
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
        # Explicit global seed: the 3GPP path does NOT propagate channel_config.random_seed
        # to the OTA synthesis stage (only WiFi _run_wifi() does). ray mode draws
        # np.random.uniform per ray, so without this seed the run isn't reproducible.
        # Golden file was generated with the same seed (see create_golden_file.py).
        np.random.seed(42)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = MIMO_OTA_Simulator().run(chamber, channel, synthesis_method='ray')
        np.testing.assert_allclose(
            result['transmit_mapping_matrix_A'], golden_A, rtol=1e-6, atol=1e-6,
            err_msg="ray mode regression: A matrix diverged from golden baseline",
        )


if __name__ == '__main__':
    unittest.main()
