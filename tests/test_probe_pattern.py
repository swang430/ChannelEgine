"""探头方向图 (ProbeAntennaConfig.pattern='TR38.901') 测试 (audit batch 2, A4)。

历史 bug: pattern 字段被 Pydantic 接受但全 repo 无消费点, 设 'TR38.901'
静默等同 Isotropic。修复后 B 矩阵按 Table 7.3-1 形状因子 (峰值 0 dB,
boresight 指向暗室中心) 衰减离轴方向; 峰值增益仍由 gain_dBi 单独给出。
"""

import numpy as np
import pytest

from mimo_ota_simulator.data_models import (
    AntennaArrayConfig, ChamberConfig, TargetChannelConfig,
)
from mimo_ota_simulator.simulator import (
    MIMO_OTA_Simulator, _probe_pattern_field_amplitude,
)


class TestProbePatternHelper:
    def test_boresight_factor_is_unity(self):
        # 探头在 (1.5, 0, 0), boresight 指向原点; 朝原点方向 → 峰值 → 因子 1
        probe = np.array([1.5, 0.0, 0.0])
        assert _probe_pattern_field_amplitude(probe, -probe) == pytest.approx(1.0)

    def test_elevated_probe_boresight_is_unity(self):
        # 立体布局探头 (P2-7 custom): boresight 仍指向中心, 因子必须为 1
        probe = np.array([1.0, 1.0, 1.0])
        assert _probe_pattern_field_amplitude(probe, -probe) == pytest.approx(1.0)

    def test_off_boresight_attenuates_per_table_7_3_1(self):
        # 探头 (1.5,0,0) boresight -x; 看向 +y 的 DUT 点 → 方位离轴 45°
        probe = np.array([1.5, 0.0, 0.0])
        target = np.array([0.0, 1.5, 0.0])
        amp = _probe_pattern_field_amplitude(probe, target - probe)
        expected_db = -12.0 * (45.0 / 65.0) ** 2  # A_H(45°), 0 dB 峰值
        assert 20 * np.log10(amp) == pytest.approx(expected_db, abs=1e-6)

    def test_never_amplifies(self):
        rng = np.random.default_rng(2)
        probe = np.array([2.0, 0.0, 0.5])
        for _ in range(50):
            d = rng.normal(size=3)
            assert _probe_pattern_field_amplitude(probe, d) <= 1.0 + 1e-12

    def test_degenerate_probe_at_origin_is_isotropic(self):
        assert _probe_pattern_field_amplitude(
            np.zeros(3), np.array([1.0, 0, 0])) == 1.0


class TestProbePatternInBMatrix:
    def _b_matrix(self, pattern):
        chamber = ChamberConfig(
            num_probes=8, dual_polarized=False, distribution='ring', radius_m=1.5,
            probe_antenna={'gain_dBi': 3.0, 'pattern': pattern},
        )
        channel = TargetChannelConfig(
            model_name='UMa', center_frequency_hz=3.5e9,
            tx_antenna=AntennaArrayConfig(num_cols=2),
            rx_antenna=AntennaArrayConfig(num_cols=4, spacing_h=2.0),  # 大间距拉开离轴角
        )
        sim = MIMO_OTA_Simulator()
        return sim._synthesize_b_matrix(chamber, channel)

    def test_tr38901_changes_b_matrix(self):
        b_iso = self._b_matrix('Isotropic')
        b_pat = self._b_matrix('TR38.901')
        assert not np.allclose(b_iso, b_pat)

    def test_tr38901_only_attenuates(self):
        b_iso = np.abs(self._b_matrix('Isotropic'))
        b_pat = np.abs(self._b_matrix('TR38.901'))
        assert np.all(b_pat <= b_iso + 1e-12)
        assert np.any(b_pat < b_iso - 1e-12)

    def test_isotropic_unchanged_default(self):
        # 默认 Isotropic: 与显式 Isotropic 完全一致 (回归保护)
        chamber_default = ChamberConfig(
            num_probes=8, dual_polarized=False, distribution='ring', radius_m=1.5,
            probe_antenna={'gain_dBi': 3.0},
        )
        channel = TargetChannelConfig(
            model_name='UMa', center_frequency_hz=3.5e9,
            tx_antenna=AntennaArrayConfig(num_cols=2),
            rx_antenna=AntennaArrayConfig(num_cols=4, spacing_h=2.0),
        )
        b_default = MIMO_OTA_Simulator()._synthesize_b_matrix(chamber_default, channel)
        np.testing.assert_allclose(b_default, self._b_matrix('Isotropic'))
