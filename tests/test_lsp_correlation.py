"""LSP 互相关矩阵构建测试 (audit batch 2, 发现 A1)。

历史 bug: 7 个场景 model 复制的构建代码只按 ``f'{p1}_vs_{p2}'`` 单方向
查 correlations key, 而参数 JSON 存反方向 (如 ``ASA_vs_DS``), 导致 11 个
相关条目中 7 个被静默丢弃。本文件覆盖:

1. helper 单元行为 (双向 key、冲突、未知 key warning);
2. 全场景扫描 — JSON 每个 correlations key 必须真正落进矩阵;
3. 端到端 — 修复后采样的经验相关系数符合 Table 7.5-6。
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from channel_model_38901.common.lsp_correlation import build_lsp_correlation_matrix

PARAM_FILE = (
    Path(__file__).resolve().parents[1]
    / 'channel_model_38901' / 'parameters' / 'parameters1013_tr38901_v19.0.0.json'
)
FULL_ORDER = ['DS', 'ASD', 'ASA', 'ZSA', 'SF', 'K']

with open(PARAM_FILE, encoding='utf-8') as f:
    SCENARIOS = json.load(f)['scenarios']


class TestBuildMatrixUnit:
    def test_reverse_key_direction_is_consumed(self):
        # JSON 实际存的方向 — 历史 bug 下这个 key 被静默丢弃
        C = build_lsp_correlation_matrix(['DS', 'ASA'], {'ASA_vs_DS': 0.8})
        assert C[0, 1] == 0.8
        assert C[1, 0] == 0.8

    def test_forward_key_direction_is_consumed(self):
        C = build_lsp_correlation_matrix(['DS', 'ASA'], {'DS_vs_ASA': 0.8})
        assert C[0, 1] == 0.8

    def test_removed_param_keys_skipped_silently(self):
        # K='N/A' 场景下 param_order 不含 K, K_vs_* 应合法跳过且无 warning
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            C = build_lsp_correlation_matrix(
                ['DS', 'ASA'], {'K_vs_DS': -0.4, 'ASA_vs_DS': 0.8})
        assert C[0, 1] == 0.8

    def test_unknown_key_warns_and_is_ignored(self):
        with pytest.warns(UserWarning, match='无法识别'):
            C = build_lsp_correlation_matrix(['DS', 'ASA'], {'XX_vs_DS': 0.5})
        np.testing.assert_allclose(C, np.identity(2))

    def test_malformed_key_warns(self):
        with pytest.warns(UserWarning, match='无法识别'):
            build_lsp_correlation_matrix(['DS', 'ASA'], {'DS-ASA': 0.5})

    def test_conflicting_directions_raise(self):
        with pytest.raises(ValueError, match='不一致'):
            build_lsp_correlation_matrix(
                ['DS', 'ASA'], {'DS_vs_ASA': 0.8, 'ASA_vs_DS': 0.3})

    def test_consistent_duplicate_directions_ok(self):
        C = build_lsp_correlation_matrix(
            ['DS', 'ASA'], {'DS_vs_ASA': 0.8, 'ASA_vs_DS': 0.8})
        assert C[0, 1] == 0.8


def _all_lsp_sets():
    cases = []
    for scenario, params in SCENARIOS.items():
        for lsp_set in ('lsp_parameters_los', 'lsp_parameters_nlos'):
            if lsp_set in params and 'correlations' in params[lsp_set]:
                cases.append((scenario, lsp_set))
    return cases


class TestAllScenarioKeysConsumed:
    """全场景扫描: 参数 JSON 里每个 correlations 条目都必须进矩阵。

    历史 bug 下 UMa LOS 的 11 个条目只有 4 个 (*_vs_SF) 命中, 本测试在
    修复前会失败, 防止单方向查找回归。
    """

    @pytest.mark.parametrize('scenario,lsp_set', _all_lsp_sets())
    def test_every_json_key_lands_in_matrix(self, scenario, lsp_set):
        lsp_params = SCENARIOS[scenario][lsp_set]
        corr = lsp_params['correlations']
        order = list(FULL_ORDER)
        if lsp_params.get('mu_log10_K') == 'N/A':
            order.remove('K')
        idx = {p: i for i, p in enumerate(order)}

        with warnings.catch_warnings():
            warnings.simplefilter('error')  # 生产 JSON 不允许触发任何 warning
            C = build_lsp_correlation_matrix(order, corr)

        np.testing.assert_allclose(C, C.T)
        np.testing.assert_allclose(np.diag(C), 1.0)
        for key, value in corr.items():
            p1, _, p2 = key.partition('_vs_')
            if p1 in idx and p2 in idx:
                assert C[idx[p1], idx[p2]] == pytest.approx(value), \
                    f'{scenario}/{lsp_set}: key {key} 未进矩阵'


class TestEndToEndCorrelation:
    """端到端: 修复后实际采样的经验相关系数符合 JSON (Table 7.5-6)。"""

    @pytest.mark.quick
    def test_uma_los_asa_ds_empirical_correlation(self):
        from channel_model_38901.common.data_models import ScenarioConfig
        from channel_model_38901.models.uma_models import UMa_Fading
        from channel_model_38901.parameters.parameter_loader import ParameterLoader

        expected = SCENARIOS['UMa']['lsp_parameters_los']['correlations']['ASA_vs_DS']
        assert expected != 0.0  # 前提: 这是个强相关条目, 否则测试无意义

        params = ParameterLoader.load_parameters(str(PARAM_FILE))
        fading = UMa_Fading(params)
        config = ScenarioConfig(
            scenario_name='UMa',
            center_frequency_hz=3.5e9,
            bs_positions=np.array([[0, 0, 25]]),
            ue_positions=np.array([[100, 0, 1.5]]),
            ue_velocities=np.array([[0, 0, 0]]),
        )
        np.random.seed(42)
        log_ds, log_asa = [], []
        for _ in range(2000):
            lsps = fading.generate_lsps(config, is_los=True)
            log_ds.append(np.log10(lsps.delay_spread_s))
            log_asa.append(np.log10(lsps.arrival_azimuth_spread_deg))
        rho = np.corrcoef(log_ds, log_asa)[0, 1]
        # N=2000, rho=0.8 时经验相关系数标准差约 0.008, abs=0.06 极宽松;
        # 历史 bug 下 rho≈0 (仅经 SF 间接弱相关), 必然失败
        assert rho == pytest.approx(expected, abs=0.06)
