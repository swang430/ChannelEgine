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


if __name__ == '__main__':
    unittest.main()
