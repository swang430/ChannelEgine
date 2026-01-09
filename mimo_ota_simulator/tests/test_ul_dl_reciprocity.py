import numpy as np

from mimo_ota_simulator.data_models import ChamberConfig, TargetChannelConfig, AntennaArrayConfig
from mimo_ota_simulator.simulator import MIMO_OTA_Simulator


def test_ul_dl_reciprocity_conjugate_transpose():
    chamber_config = ChamberConfig(
        num_probes=4,
        dual_polarized=False,
        distribution='ring',
        radius_m=1.0
    )
    channel_config = TargetChannelConfig(
        model_name='UMa',
        cluster_model_name='CDL-A',
        center_frequency_hz=3.5e9,
        random_seed=42,
        tx_antenna=AntennaArrayConfig(num_cols=2, array_type='ULA'),
        rx_antenna=AntennaArrayConfig(num_cols=1, array_type='ULA'),
        enable_ul_dl=True,
        reciprocity_transform='conjugate_transpose'
    )

    simulator = MIMO_OTA_Simulator()
    results = simulator.run(chamber_config, channel_config, synthesis_method='cluster')

    a_dl = results['transmit_mapping_matrix_A']
    a_ul = results['transmit_mapping_matrix_A_ul']
    assert a_ul is not None
    np.testing.assert_allclose(a_ul, a_dl.conj().T, rtol=1e-6, atol=1e-6)

    eff_ul = results['effective_H_matrix_ul']
    assert eff_ul is not None
    np.testing.assert_allclose(eff_ul, a_ul @ results['B_matrix'].T, rtol=1e-6, atol=1e-6)

    dl_taps = results['channel_impulse_response']
    ul_taps = results['channel_impulse_response_ul']
    assert dl_taps
    assert ul_taps
    for dl_tap, ul_tap in zip(dl_taps, ul_taps):
        np.testing.assert_allclose(ul_tap['coeff'], dl_tap['coeff'].conj().T, rtol=1e-6, atol=1e-6)


def test_ul_dl_disabled_returns_none():
    chamber_config = ChamberConfig(
        num_probes=4,
        dual_polarized=False,
        distribution='ring',
        radius_m=1.0
    )
    channel_config = TargetChannelConfig(
        model_name='UMa',
        cluster_model_name='CDL-A',
        center_frequency_hz=3.5e9,
        random_seed=7,
        tx_antenna=AntennaArrayConfig(num_cols=1, array_type='ULA'),
        rx_antenna=AntennaArrayConfig(num_cols=1, array_type='ULA'),
        enable_ul_dl=False
    )

    simulator = MIMO_OTA_Simulator()
    results = simulator.run(chamber_config, channel_config, synthesis_method='cluster')

    assert results['channel_impulse_response_ul'] is None
    assert results['transmit_mapping_matrix_A_ul'] is None
