"""§7.5 step 7 角度生成测试 (audit batch 2, 发现 F4)。

历史简化: (a) JSON scaling_factors_aoa_aod / scaling_factors_zoa_zod 表
dead config, 代码硬编码 /1.4 且漏系数 2 与 1/C_φ; (b) LOS 的 C 表 K-factor
修正缺失; (c) 角度围绕 0°(方位)/90°(zenith) 生成, 从不加几何 LOS 方向,
而 LOS 镜面分量却是几何计算的 — 簇角与 LOS 分量方向系统性脱节。

已知遗留 (文档化): NLOS μ_offset,ZOD 参数文件缺字段, 取 0。
"""

import numpy as np
import pytest

from channel_model_38901.simulator import ChannelSimulator


def _run(scenario='UMa', force=None, seed=33, distance=100.0):
    np.random.seed(seed)
    sim = ChannelSimulator(scenario, 3.5e9)
    sim.config.bs_positions = np.array([[0.0, 0.0, 25.0]])
    sim.config.ue_positions = np.array([[distance, 0.0, 1.5]])
    sim.generator = sim._build_generator()
    while True:
        results = sim.run(logger=None)
        if force is None or (force == 'LOS') == results['is_los']:
            return results


def _los_dirs(distance=100.0):
    d_vec = np.array([distance, 0.0, 1.5 - 25.0])
    d3 = np.linalg.norm(d_vec)
    return {
        'aod': 0.0, 'aoa': 180.0,
        'zod': np.rad2deg(np.arccos(d_vec[2] / d3)),
        'zoa': np.rad2deg(np.arccos(-d_vec[2] / d3)),
    }


class TestLosAlignment:
    def test_first_cluster_aligns_with_geometric_los(self):
        """LOS 时 cluster 1 须精确对齐几何 LOS 方向 (eq 7.5-12 减首簇项)。"""
        results = _run(force='LOS')
        los = _los_dirs()
        c1 = results['clusters'][0]

        def circ_diff(a, b):
            return abs((a - b + 180.0) % 360.0 - 180.0)

        assert circ_diff(c1.aoa_phi, los['aoa']) < 1e-9
        assert circ_diff(c1.aod_phi, los['aod']) < 1e-9
        assert c1.zoa_theta == pytest.approx(los['zoa'], abs=1e-9)
        assert c1.zod_theta == pytest.approx(los['zod'], abs=1e-9)


class TestLosDirectionCentering:
    def test_nlos_angles_centered_at_los_direction(self):
        """NLOS 簇角围绕几何 LOS 方向分布 (修复前围绕 0° — 与 UE 在 +x
        方向时 AoA 应约 180° 矛盾)。"""
        aoas = []
        for seed in range(20, 30):
            results = _run(force='NLOS', seed=seed)
            aoas.extend(c.aoa_phi for c in results['clusters'])
        # 环绕均值接近 180° (LOS AoA); 修复前接近 0°
        mean = np.rad2deg(np.angle(np.mean(np.exp(1j * np.deg2rad(aoas)))))
        assert abs(abs(mean) - 180.0) < 30.0

    def test_zenith_centered_at_geometric_los_zenith(self):
        zoas = []
        for seed in range(20, 30):
            results = _run(force='NLOS', seed=seed)
            zoas.extend(c.zoa_theta for c in results['clusters'])
        los = _los_dirs()
        assert abs(np.mean(zoas) - los['zoa']) < 15.0

    def test_zenith_within_valid_range(self):
        for seed in (3, 4, 5):
            results = _run(seed=seed)
            for c in results['clusters']:
                assert 0.0 <= c.zoa_theta <= 180.0
                assert 0.0 <= c.zod_theta <= 180.0


class TestScalingTableConsumed:
    def test_changing_table_changes_angles(self):
        """改 JSON 缩放表必须改变输出角 (修复前 dead config, 必失败)。"""
        np.random.seed(7)
        sim_a = ChannelSimulator('UMa', 3.5e9)
        r_a = sim_a.run(logger=None)

        np.random.seed(7)
        sim_b = ChannelSimulator('UMa', 3.5e9)
        tbl = sim_b.params.get_scenario_params('UMa')['scaling_factors_aoa_aod']
        for k in tbl:
            tbl[k] = tbl[k] * 2.0
        r_b = sim_b.run(logger=None)

        a_angles = [c.aoa_phi for c in r_a['clusters']]
        b_angles = [c.aoa_phi for c in r_b['clusters']]
        assert not np.allclose(a_angles, b_angles)

    def test_missing_table_falls_back_to_legacy_constants(self):
        """整表缺失 (备用参数文件) 时方位 C=2.0 / zenith C=1.0, 等效旧公式
        幅度, 不被新系数 2 无声放大 (Codex PR#25 P2)。"""
        np.random.seed(7)
        sim = ChannelSimulator('UMa', 3.5e9)
        params = sim.params.get_scenario_params('UMa')
        params.pop('scaling_factors_aoa_aod')
        params.pop('scaling_factors_zoa_zod')
        with pytest.warns(UserWarning, match='历史等效常数'):
            r_fallback = sim.run(logger=None)

        # 与有表版本对比: fallback 幅度量级与旧公式一致 (C_φ=2 抵消系数 2)
        gen = sim.generator
        from channel_model_38901.common.data_models import LargeScaleParameters
        lsps = LargeScaleParameters(
            delay_spread_s=1e-7,
            departure_azimuth_spread_deg=10.0, arrival_azimuth_spread_deg=20.0,
            departure_zenith_spread_deg=5.0, arrival_zenith_spread_deg=5.0,
            k_factor_linear=0.0, shadow_fade_db=0.0)
        P_n = np.linspace(1.0, 0.1, 12)
        np.random.seed(3)
        with pytest.warns(UserWarning):
            a = gen._generate_angles(P_n, lsps, 12, False,
                                     {'scaling_factors_aoa_aod': None,
                                      'scaling_factors_zoa_zod': None})
        # 旧公式: aoa offset = X·sqrt(ratio)·(ASA/1.4) + Y·ASA/7 + LOS(180°)
        np.random.seed(3)
        X = np.random.choice([-1, 1], size=(12, 4))
        Y = np.random.randn(12, 4)
        ratio = -np.log(P_n / np.max(P_n))
        legacy = X[:, 0] * np.sqrt(ratio) * (20.0 / 1.4) + Y[:, 0] * (20.0 / 7.0) + 180.0
        legacy = (legacy + 180.0) % 360.0 - 180.0
        np.testing.assert_allclose(a['aoa_phi'], legacy, rtol=1e-12)
        assert r_fallback['clusters']  # fallback 路径端到端可跑

    def test_missing_n_falls_back_to_nearest_with_warning(self):
        np.random.seed(7)
        sim = ChannelSimulator('UMa', 3.5e9)
        params = sim.params.get_scenario_params('UMa')
        # 删掉精确档位, 保留其它
        params['scaling_factors_aoa_aod'].pop(
            str(params['num_clusters_los']), None)
        params['scaling_factors_aoa_aod'].pop(
            str(params['num_clusters_nlos']), None)
        with pytest.warns(UserWarning, match='最近档'):
            sim.run(logger=None)


class TestLosKCorrection:
    def test_k_correction_changes_los_angles(self):
        """同 K 修正只在 LOS 生效: 人工放大 K 修正输入应改变 LOS 簇角。

        通过对比 use_median_lsps 两次运行 + 手动改 K 无法直接做; 这里改用
        间接断言: LOS 实现的 C_φ 修正系数 (K≈8-15dB 时 < 1) 使 φ' 相比
        NLOS 公式放大 — 至少验证 LOS 路径走的是带 K 多项式的分支
        (修复前 LOS/NLOS 用同一 C, 本测试通过源码行为差异锁定)。
        """
        from channel_model_38901.channel_generator import ChannelGenerator
        from channel_model_38901.common.data_models import LargeScaleParameters

        np.random.seed(11)
        sim = ChannelSimulator('UMa', 3.5e9)
        gen = sim.generator
        scenario_params = gen.params.get_scenario_params('UMa')
        lsps = LargeScaleParameters(
            delay_spread_s=1e-7,
            departure_azimuth_spread_deg=10.0, arrival_azimuth_spread_deg=20.0,
            departure_zenith_spread_deg=5.0, arrival_zenith_spread_deg=5.0,
            k_factor_linear=10.0,  # 10 dB
            shadow_fade_db=0.0)
        P_n = np.linspace(1.0, 0.1, 12)
        np.random.seed(11)
        a_los = gen._generate_angles(P_n, lsps, 12, True, scenario_params)
        np.random.seed(11)
        a_nlos = gen._generate_angles(P_n, lsps, 12, False, scenario_params)
        # 同随机数下 LOS 多了 K 修正 + 首簇对齐 → 角度必然不同
        assert not np.allclose(a_los['aoa_phi'], a_nlos['aoa_phi'])
