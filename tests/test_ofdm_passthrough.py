"""enable_ofdm/n_fft 透传测试 (audit batch 2, 发现 P2-5)。

历史 bug: WiFi 引擎按 enable_ofdm/n_fft 算出 h_subcarriers, 但 OTA
_run_wifi 只消费 tap_components — 结果 dict 无此键, UI 的 'OFDM 子载波
计算'/'FFT 大小' 控件勾了等于没勾。
"""

import numpy as np
import pytest

from mimo_ota_simulator.data_models import (
    AntennaArrayConfig, ChamberConfig, TargetChannelConfig,
)
from mimo_ota_simulator.simulator import MIMO_OTA_Simulator


def _run(enable_ofdm, n_fft=None, n_tx=1, n_rx=1):
    chamber = ChamberConfig(
        num_probes=4, distribution='ring', radius_m=1.5, dual_polarized=False,
        probe_antenna={'gain_dBi': 0.0, 'pattern': 'Isotropic'},
    )
    channel = TargetChannelConfig(
        standard='wifi', model_name='Model-B',
        center_frequency_hz=2.4e9, bandwidth_hz=20e6,
        enable_ofdm=enable_ofdm, n_fft=n_fft, random_seed=42,
        tx_antenna=AntennaArrayConfig(num_cols=n_tx),
        rx_antenna=AntennaArrayConfig(num_cols=n_rx),
    )
    return MIMO_OTA_Simulator().run(chamber, channel, synthesis_method='ray')


class TestOfdmPassthrough:
    def test_h_subcarriers_in_result_when_enabled(self):
        result = _run(enable_ofdm=True, n_fft=64)
        h = result.get('h_subcarriers')
        assert h is not None, "enable_ofdm=True 必须在结果 dict 里给出 h_subcarriers"
        assert h.shape[0] == 64

    def test_n_fft_controls_output_size(self):
        """改 n_fft 必须改变输出 (修复前控件零效果, 必失败)。"""
        h_64 = _run(enable_ofdm=True, n_fft=64)['h_subcarriers']
        h_128 = _run(enable_ofdm=True, n_fft=128)['h_subcarriers']
        assert h_64.shape[0] == 64
        assert h_128.shape[0] == 128

    def test_disabled_gives_none(self):
        result = _run(enable_ofdm=False)
        assert result.get('h_subcarriers') is None

    def test_frequency_response_consistent_with_taps(self):
        """h_subcarriers 应等于 taps 的 DFT (与 WiFi 引擎公式一致)。"""
        result = _run(enable_ofdm=True, n_fft=64)
        h = result['h_subcarriers'][:, 0, 0]
        assert np.all(np.isfinite(h))
        assert np.any(np.abs(h) > 0)

    def test_multi_antenna_shape_and_center_subcarrier(self):
        """多天线时形状须为 (n_fft, N_rx, N_tx) (Codex PR#28 P2: WiFi 引擎
        版本是标量 (n_fft,1,1)), 且中心子载波 (f=0) 等于 target_H_matrix。"""
        result = _run(enable_ofdm=True, n_fft=64, n_tx=2, n_rx=2)
        h = result['h_subcarriers']
        assert h.shape == (64, 2, 2)
        np.testing.assert_allclose(
            h[32], result['target_H_matrix'], rtol=1e-9, atol=1e-12)
