"""球坐标 zenith 约定一致性测试 (audit batch 2, 发现 A2)。

历史 bug: channel_generator 的 `_unit_vector` 与簇级 Doppler 用 elevation
约定 (z=sinθ), 而 SoS 导出 / LCS 变换 / 元件方向图 / mimo_ota_simulator
用 zenith 约定 (z=cosθ); stochastic 簇的 zenith 角又以 0° 为中心生成,
两错相抵掩盖了主路径问题, 但同一批角度进 SoS 导出和元件方向图时即错。

本文件锁定: (1) _unit_vector 是 TR 38.901 §7.1 zenith 约定;
(2) 跨包 (channel model vs OTA) 公式一致; (3) stochastic zenith 角
以 90° (水平) 为中心; (4) URA 垂直相关在水平传播时无相位差。
"""

import numpy as np
import pytest

from channel_model_38901.channel_generator import ChannelGenerator
from channel_model_38901.common.data_models import EngineCluster
from channel_model_38901.models.spatial_correlation import get_spatial_correlation_matrix
from mimo_ota_simulator.simulator import MIMO_OTA_Simulator


class TestUnitVectorZenithConvention:
    def test_horizontal_at_zenith_90(self):
        # θ=90° (zenith 约定) = 水平方向
        np.testing.assert_allclose(
            ChannelGenerator._unit_vector(90.0, 0.0), [1.0, 0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(
            ChannelGenerator._unit_vector(90.0, 90.0), [0.0, 1.0, 0.0], atol=1e-12)

    def test_vertical_at_zenith_0(self):
        # θ=0° = 正上方 (+z)
        np.testing.assert_allclose(
            ChannelGenerator._unit_vector(0.0, 37.0)[2], 1.0, atol=1e-12)

    def test_matches_tr38901_eq_7_1_6(self):
        rng = np.random.default_rng(3)
        for _ in range(20):
            theta = rng.uniform(0, 180)
            phi = rng.uniform(-180, 180)
            t, p = np.deg2rad(theta), np.deg2rad(phi)
            expected = [np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)]
            np.testing.assert_allclose(
                ChannelGenerator._unit_vector(theta, phi), expected, atol=1e-12)


class TestCrossPackageConsistency:
    def test_channel_model_matches_ota_unit_vector(self):
        """channel model 与 mimo_ota_simulator 的方向向量公式必须逐点一致。

        历史 bug 下 channel model 是 elevation 约定、OTA 是 zenith 约定,
        同一个 EngineCluster 在两包里指向不同方向。
        """
        rng = np.random.default_rng(7)
        for _ in range(20):
            theta = rng.uniform(0, 180)
            phi = rng.uniform(-180, 180)
            np.testing.assert_allclose(
                ChannelGenerator._unit_vector(theta, phi),
                MIMO_OTA_Simulator._unit_vector_from_angles(phi, theta),
                atol=1e-12)

    def test_engine_cluster_default_is_horizontal(self):
        # 默认簇必须是水平方向 (zenith 90°), 与 OTA 侧默认一致
        c = EngineCluster()
        assert c.zoa_theta == 90.0
        assert c.zod_theta == 90.0


class TestStochasticZenithCentering:
    def test_zenith_angles_centered_at_horizon(self):
        """stochastic 簇 zenith 角以 90° 为中心 (zenith 约定的水平方向)。

        历史 bug 下以 0° 为中心 — 在修复后的 zenith 约定中那意味着所有
        簇都从天顶方向到达。
        """
        from channel_model_38901.simulator import ChannelSimulator

        np.random.seed(11)
        sim = ChannelSimulator(scenario_name='UMa', center_frequency_hz=3.5e9)
        zoas, zods = [], []
        for _ in range(20):
            results = sim.run(logger=None)
            zoas.extend(c.zoa_theta for c in results['clusters'])
            zods.extend(c.zod_theta for c in results['clusters'])
        assert 60.0 < np.mean(zoas) < 120.0
        assert 60.0 < np.mean(zods) < 120.0
        # 修复前均值在 0 附近, 必然失败
        assert abs(np.mean(zoas)) > 30.0


class TestZenithFolding:
    def test_fold_keeps_valid_range(self):
        theta = np.array([-30.0, 0.0, 90.0, 180.0, 210.0, 365.0, -200.0])
        folded = ChannelGenerator._fold_zenith(theta)
        assert np.all(folded >= 0.0) and np.all(folded <= 180.0)
        # §7.5 step 7 镜像语义: 210° → 150°; -30° → 30°
        np.testing.assert_allclose(
            folded, [30.0, 0.0, 90.0, 180.0, 150.0, 5.0, 160.0])

    def test_weak_clusters_stay_in_range(self):
        """弱簇 -ZS·log(P/Pmax) 超 90° 时 zenith 不得越界 (Codex PR#17 P2)。"""
        from channel_model_38901.common.data_models import LargeScaleParameters

        lsps = LargeScaleParameters(
            delay_spread_s=1e-7,
            departure_azimuth_spread_deg=10.0, arrival_azimuth_spread_deg=20.0,
            departure_zenith_spread_deg=40.0, arrival_zenith_spread_deg=40.0,
            k_factor_linear=0.0, shadow_fade_db=0.0)
        P_n = np.array([1.0, 1e-6, 1e-12])  # -log 比值高达 27.6 → 偏移远超 90°
        np.random.seed(5)
        angles = ChannelGenerator._generate_angles(
            ChannelGenerator.__new__(ChannelGenerator), P_n, lsps, len(P_n))
        for key in ('zoa_theta', 'zod_theta'):
            assert np.all(angles[key] >= 0.0), angles[key]
            assert np.all(angles[key] <= 180.0), angles[key]


class TestUraVerticalCorrelation:
    def test_no_vertical_phase_at_horizontal_propagation(self):
        # zenith=90° (水平传播) 时垂直阵元间无相位差 → R 全实数非负
        R = get_spatial_correlation_matrix(
            'URA', num_rows=4, num_cols=1, spacing_v=0.5, spacing_h=0.5,
            zenith_deg=90.0, azimuth_deg=0.0,
            zenith_spread_deg=5.0, azimuth_spread_deg=10.0)
        np.testing.assert_allclose(np.imag(R), 0.0, atol=1e-12)

    def test_vertical_phase_appears_off_horizon(self):
        # 偏离水平 (zenith=60°) 时相邻垂直阵元相位差 ∝ cos(60°)=0.5
        R = get_spatial_correlation_matrix(
            'URA', num_rows=2, num_cols=1, spacing_v=0.5, spacing_h=0.5,
            zenith_deg=60.0, azimuth_deg=0.0,
            zenith_spread_deg=0.0, azimuth_spread_deg=0.0)
        # 相位符号取决于阵元索引方向 (R[0,1] 对应 p-q=-1), 只断言幅值
        expected_phase = 2 * np.pi * 0.5 * np.cos(np.deg2rad(60.0))
        assert abs(np.angle(R[0, 1])) == pytest.approx(expected_phase, abs=1e-9)
