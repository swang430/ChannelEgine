"""§7.5 step 4/5 LOS 条件化测试 (audit batch 2, 发现 F6+F7)。

历史简化: (a) LOS 的 D(K) 延迟缩放缺失; (b) per-cluster shadowing ζ 硬编码
σ=1 dB (Table 7.5-6 按场景定义); (c) r_τ 不分 LOS/NLOS; (d) -25 dB 弱簇
删除未实现; (e) CLAUDE.md 声称 K 因子进簇功率, 与代码不符 (文档已修)。
"""

import warnings

import numpy as np
import pytest

from channel_model_38901.channel_generator import ChannelGenerator
from channel_model_38901.common.data_models import LargeScaleParameters
from channel_model_38901.simulator import ChannelSimulator


def _lsps(k_linear=0.0):
    return LargeScaleParameters(
        delay_spread_s=1e-7,
        departure_azimuth_spread_deg=10.0, arrival_azimuth_spread_deg=20.0,
        departure_zenith_spread_deg=5.0, arrival_zenith_spread_deg=5.0,
        k_factor_linear=k_linear, shadow_fade_db=0.0)


class TestPruneWeakClusters:
    def test_below_threshold_removed_and_renormalized(self):
        tau = np.array([0.0, 1e-8, 2e-8, 3e-8])
        P = np.array([0.9, 0.0998, 0.0001, 0.0001])  # 后两个 < max·10^-2.5
        tau2, P2 = ChannelGenerator._prune_weak_clusters(tau, P)
        assert len(P2) == 2
        np.testing.assert_allclose(P2.sum(), 1.0)
        np.testing.assert_allclose(tau2, tau[:2])

    def test_all_kept_when_within_25db(self):
        tau = np.array([0.0, 1e-8])
        P = np.array([0.6, 0.4])
        tau2, P2 = ChannelGenerator._prune_weak_clusters(tau, P)
        np.testing.assert_allclose(P2, P)

    def test_los_specular_first_cluster_survives_pruning(self):
        """LOS 首簇承载镜面 K/(K+1), 即便扩散功率 < -25 dB 也不得被删
        (Codex PR#26 P2)。"""
        tau = np.array([0.0, 1e-8, 2e-8])
        P = np.array([0.0001, 0.6, 0.3999])  # 首簇扩散功率远低于 -25 dB
        k_lin = 10.0
        basis = P / (k_lin + 1.0)
        basis[0] += k_lin / (k_lin + 1.0)
        tau2, P2 = ChannelGenerator._prune_weak_clusters(tau, P, prune_basis=basis)
        assert len(P2) == 3  # 全部存活 (基准下无簇低于 -25 dB)
        # 反向确认: 不带 basis 时首簇会被误删 (旧行为)
        tau3, P3 = ChannelGenerator._prune_weak_clusters(tau, P)
        assert len(P3) == 2

    def test_end_to_end_cluster_count_can_shrink(self):
        """大 ζ 下功率离散 > 25 dB, 端到端簇数应可小于场景 N。"""
        np.random.seed(2)
        sim = ChannelSimulator('UMa', 3.5e9)
        params = sim.params.get_scenario_params('UMa')
        for ls in ('lsp_parameters_los', 'lsp_parameters_nlos'):
            params[ls]['per_cluster_shadowing_std_db'] = 15.0
        shrunk = False
        for _ in range(10):
            results = sim.run(logger=None)
            n_expected = (params['num_clusters_los'] if results['is_los']
                          else params['num_clusters_nlos'])
            if len(results['clusters']) < n_expected:
                shrunk = True
                break
        assert shrunk


class TestLosDelayScaling:
    def test_los_delays_scaled_by_d_of_k(self):
        """LOS 延迟 = 未缩放延迟 / D(K) (step 4); 功率用未缩放延迟。"""
        np.random.seed(17)
        sim = ChannelSimulator('UMa', 3.5e9)
        gen = sim.generator
        scenario_params = gen.params.get_scenario_params('UMa')
        lsps = _lsps(k_linear=10.0)  # K = 10 dB

        np.random.seed(17)
        clusters = gen._generate_stochastic_clusters(lsps, True, None)

        # 复算: 同 seed 下未缩放延迟与功率 (LOS 修剪判据为 K 调整功率)
        np.random.seed(17)
        tau_raw = gen._generate_cluster_delays(
            lsps, scenario_params['num_clusters_los'], True, scenario_params)
        P_n = gen._generate_cluster_powers(tau_raw, lsps, True, scenario_params)
        basis = P_n / 11.0
        basis[0] += 10.0 / 11.0
        tau_kept, _ = gen._prune_weak_clusters(tau_raw, P_n, prune_basis=basis)
        k_db = 10.0
        d_scale = 0.7705 - 0.0433 * k_db + 0.0002 * k_db**2 + 0.000017 * k_db**3
        np.testing.assert_allclose(
            [c.delay for c in clusters], tau_kept / d_scale, rtol=1e-12)

    def test_nlos_delays_not_scaled(self):
        np.random.seed(17)
        sim = ChannelSimulator('UMa', 3.5e9)
        gen = sim.generator
        scenario_params = gen.params.get_scenario_params('UMa')
        lsps = _lsps()
        np.random.seed(17)
        clusters = gen._generate_stochastic_clusters(lsps, False, None)
        np.random.seed(17)
        tau_raw = gen._generate_cluster_delays(
            lsps, scenario_params['num_clusters_nlos'], False, scenario_params)
        P_n = gen._generate_cluster_powers(tau_raw, lsps, False, scenario_params)
        tau_kept, _ = gen._prune_weak_clusters(tau_raw, P_n)
        np.testing.assert_allclose([c.delay for c in clusters], tau_kept, rtol=1e-12)


class TestPerClusterShadowing:
    def test_zeta_from_json_changes_powers(self):
        """改 ζ 必须改变功率分布 (修复前硬编码 σ=1 dB, 必失败)。"""
        sim = ChannelSimulator('UMa', 3.5e9)
        gen = sim.generator
        params = gen.params.get_scenario_params('UMa')
        lsps = _lsps()
        tau = np.linspace(0, 3e-7, 12)

        np.random.seed(8)
        p_a = gen._generate_cluster_powers(tau, lsps, False, params)
        params['lsp_parameters_nlos']['per_cluster_shadowing_std_db'] = 9.0
        np.random.seed(8)
        p_b = gen._generate_cluster_powers(tau, lsps, False, params)
        assert not np.allclose(p_a, p_b)

    def test_missing_zeta_warns_and_uses_1db(self):
        sim = ChannelSimulator('UMa', 3.5e9)
        gen = sim.generator
        params = gen.params.get_scenario_params('UMa')
        params['lsp_parameters_nlos'].pop('per_cluster_shadowing_std_db')
        lsps = _lsps()
        tau = np.linspace(0, 3e-7, 12)
        with pytest.warns(UserWarning, match='per_cluster_shadowing_std_db'):
            np.random.seed(8)
            p_legacy = gen._generate_cluster_powers(tau, lsps, False, params)
        # 等价于 ζ=1 的显式值
        params['lsp_parameters_nlos']['per_cluster_shadowing_std_db'] = 1.0
        np.random.seed(8)
        p_1db = gen._generate_cluster_powers(tau, lsps, False, params)
        np.testing.assert_allclose(p_legacy, p_1db)


class TestRTauSplit:
    def test_split_keys_take_precedence(self):
        sim = ChannelSimulator('UMa', 3.5e9)
        gen = sim.generator
        params = dict(gen.params.get_scenario_params('UMa'))
        params['delay_scaling_factor_los'] = 5.0
        assert gen._r_tau(params, True) == 5.0
        assert gen._r_tau(params, False) == params['delay_scaling_factor']

    def test_changing_split_value_changes_delays(self):
        sim = ChannelSimulator('UMa', 3.5e9)
        gen = sim.generator
        params = dict(gen.params.get_scenario_params('UMa'))
        lsps = _lsps()
        np.random.seed(4)
        t_a = gen._generate_cluster_delays(lsps, 12, True, params)
        params['delay_scaling_factor_los'] = params['delay_scaling_factor'] * 2
        np.random.seed(4)
        t_b = gen._generate_cluster_delays(lsps, 12, True, params)
        assert not np.allclose(t_a, t_b)
