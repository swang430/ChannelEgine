"""簇内扩展常数测试 (audit batch 2, 发现 F5)。

历史 bug: _generate_stochastic_clusters 把 LSP 总扩展 (ASA 等, 量级几十度)
直接塞进 cluster.as_* 当簇内 ray spread 用 — spec 的射线偏移是
φ_n,m = φ_n + c_ASA·α_m (Table 7.5-6 per-cluster 常数, 量级几度), SoS/ray
导出的角度扇出被系统性放大; JSON 里现成的 c_asa_deg/c_asd_deg/c_zsa_deg
从未被读取 (dead config)。
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from channel_model_38901.simulator import ChannelSimulator

PARAM_FILE = (
    Path(__file__).resolve().parents[1]
    / 'channel_model_38901' / 'parameters' / 'parameters1013_tr38901_v19.0.0.json'
)
with open(PARAM_FILE, encoding='utf-8') as f:
    SCENARIOS = json.load(f)['scenarios']


def _run(scenario='UMa', seed=21):
    np.random.seed(seed)
    sim = ChannelSimulator(scenario, 3.5e9)
    results = sim.run(logger=None)
    return results


class TestClusterSpreadUsesTableConstants:
    @pytest.mark.parametrize('scenario', sorted(SCENARIOS.keys()))
    def test_as_fields_equal_c_constants(self, scenario):
        results = _run(scenario)
        lsp_set = 'lsp_parameters_los' if results['is_los'] else 'lsp_parameters_nlos'
        c = SCENARIOS[scenario][lsp_set]
        for cl in results['clusters']:
            assert cl.as_aoa == c['c_asa_deg']
            assert cl.as_aod == c['c_asd_deg']
            assert cl.as_zoa == c['c_zsa_deg']

    def test_as_zod_is_three_eighths_mean_zsd(self):
        """c_ZSD = (3/8)·10^μ_lgZSD — 表均值, 非采样值 (Codex PR#24 P2)。"""
        from channel_model_38901.common.expression_eval import safe_eval

        results = _run('UMa')
        lsp_set = 'lsp_parameters_los' if results['is_los'] else 'lsp_parameters_nlos'
        mu_expr = SCENARIOS['UMa'][lsp_set]['mu_log10_ZSD']
        # 与 ChannelSimulator 默认几何一致: BS (0,0,25), UE (100,0,1.5)
        mu = safe_eval(mu_expr, {'fc_ghz': 3.5, 'h_ut': 1.5, 'h_bs': 25.0,
                                 'd_2d': 100.0})
        expected = 0.375 * 10.0 ** mu
        for cl in results['clusters']:
            assert cl.as_zod == pytest.approx(expected)

    def test_as_zod_deterministic_across_realizations(self):
        # 采样 ZSD 每次实现都抖, 表均值不抖 — as_zod 必须跨实现恒定
        a = _run('UMa', seed=1)['clusters'][0].as_zod
        b = _run('UMa', seed=2)['clusters'][0].as_zod
        assert a == pytest.approx(b)

    def test_spread_no_longer_total_lsp(self):
        """簇内扩展不得再等于 LSP 总 ASA (修复前恒等, 必失败)。"""
        results = _run('UMa')
        total_asa = results['lsps'].arrival_azimuth_spread_deg
        c_asa = results['clusters'][0].as_aoa
        # 总 ASA 是随机采样, 与表常数撞值概率为 0
        assert c_asa != pytest.approx(total_asa)


class TestZsdFallbackNoShrink:
    def test_missing_mu_zsd_falls_back_to_sampled_without_multiplier(self):
        """mu_log10_ZSD 缺失时回退采样 ZSD 原值, 不乘 3/8 (Codex PR#24 P2 r2)。"""
        np.random.seed(5)
        sim = ChannelSimulator('UMa', 3.5e9)
        for ls in ('lsp_parameters_los', 'lsp_parameters_nlos'):
            sim.params.get_scenario_params('UMa')[ls].pop('mu_log10_ZSD', None)
        results = sim.run(logger=None)
        assert results['clusters'][0].as_zod == pytest.approx(
            results['lsps'].departure_zenith_spread_deg)


class TestMissingConstantFallback:
    def test_missing_constants_warn_and_fall_back(self):
        np.random.seed(5)
        sim = ChannelSimulator('UMa', 3.5e9)
        # 本实例私有的参数表, 删掉常数模拟旧参数文件
        for ls in ('lsp_parameters_los', 'lsp_parameters_nlos'):
            for k in ('c_asa_deg', 'c_asd_deg', 'c_zsa_deg'):
                sim.params.get_scenario_params('UMa')[ls].pop(k, None)
        with pytest.warns(UserWarning, match='c_asa_deg'):
            results = sim.run(logger=None)
        # 回退到 LSP 总扩展 (旧行为)
        assert results['clusters'][0].as_aoa == pytest.approx(
            results['lsps'].arrival_azimuth_spread_deg)
