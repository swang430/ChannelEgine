"""CDL per-cluster ray spread 测试 (audit batch 2, 发现 A3)。

历史 bug: 参数 JSON 的 cdl_models 缺 TR 38.901 Table 7.7.1-1..5 的簇内
ray spread (c_ASD/c_ASA/c_ZSD/c_ZSA), 代码 ``.get('as_*_deg', 0.0)`` 静默
回退 0 → SoS/ray 级导出的每簇 20 条射线角度全部塌缩到簇均值, 角度扩展
为 0 且无任何告警。

本文件锁定: (1) 每个 CDL 模型的簇携带 JSON ray_spread_deg 的非零扩展;
(2) 簇级 as_*_deg 仍可覆盖模型级默认; (3) 缺 ray_spread_deg 的模型
(如 SCME) 必须告警; (4) 端到端 — SoS 导出的簇内射线 Doppler 真正散开。
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from channel_model_38901.parameters.parameter_loader import ParameterTables
from channel_model_38901.simulator import ChannelSimulator

PARAM_FILE = (
    Path(__file__).resolve().parents[1]
    / 'channel_model_38901' / 'parameters' / 'parameters1013_tr38901_v19.0.0.json'
)
with open(PARAM_FILE, encoding='utf-8') as f:
    _DATA = json.load(f)
CDL_MODELS = sorted(_DATA['cdl_models'].keys())


def _generator(cluster_model):
    np.random.seed(9)
    return ChannelSimulator('UMa', 3.5e9, cluster_model).generator


class TestRaySpreadLoaded:
    @pytest.mark.parametrize('model', CDL_MODELS)
    def test_json_defines_ray_spread(self, model):
        spread = _DATA['cdl_models'][model].get('ray_spread_deg')
        assert spread, f'{model} 缺 ray_spread_deg 块'
        for key in ('c_asd_deg', 'c_asa_deg', 'c_zsd_deg', 'c_zsa_deg'):
            assert spread.get(key, 0.0) > 0.0, f'{model}.{key} 必须为正'

    @pytest.mark.parametrize('model', CDL_MODELS)
    def test_clusters_carry_model_level_spread(self, model):
        spread = _DATA['cdl_models'][model]['ray_spread_deg']
        gen = _generator(model)
        clusters = gen._load_deterministic_clusters(model)
        assert clusters
        for i, c in enumerate(clusters):
            raw = _DATA['cdl_models'][model]['clusters'][i]
            if 'as_aoa_deg' in raw:
                continue  # 簇级显式覆盖 (如 LOS 镜面行), 单独测试
            assert c.as_aod == spread['c_asd_deg']
            assert c.as_aoa == spread['c_asa_deg']
            assert c.as_zod == spread['c_zsd_deg']
            assert c.as_zoa == spread['c_zsa_deg']

    @pytest.mark.parametrize('model', ['CDL-D', 'CDL-E'])
    def test_los_specular_row_stays_unspread(self, model):
        """LOS 模型首行是镜面分量 (Table 7.7.1-4/5 'Specular(LOS path)'),
        是单条确定性射线, 不得继承模型级 Laplacian 扩展 (Codex PR#18 P1)。"""
        gen = _generator(model)
        clusters = gen._load_deterministic_clusters(model)
        spec = clusters[0]
        assert spec.as_aod == 0.0
        assert spec.as_aoa == 0.0
        assert spec.as_zod == 0.0
        assert spec.as_zoa == 0.0
        # 第二行 (Laplacian 子簇) 正常携带模型级扩展
        assert clusters[1].as_aoa > 0.0

    def test_cluster_level_override_wins(self):
        gen = _generator('CDL-A')
        pt = ParameterTables('synthetic')
        pt.data = {'cdl_models': {'CDL-X': {
            'xpr_db': 10,
            'ray_spread_deg': {'c_asd_deg': 5.0, 'c_asa_deg': 11.0,
                               'c_zsd_deg': 3.0, 'c_zsa_deg': 3.0},
            'clusters': [
                {'id': 1, 'delay_ns': 0.0, 'power_db': 0.0,
                 'aod_deg': 0.0, 'aoa_deg': 0.0, 'zod_deg': 90.0, 'zoa_deg': 90.0,
                 'as_aoa_deg': 25.0},
            ],
        }}}
        gen.params = pt
        clusters = gen._load_deterministic_clusters('CDL-X')
        assert clusters[0].as_aoa == 25.0   # 簇级覆盖
        assert clusters[0].as_aod == 5.0    # 模型级默认


class TestScenarioScalingAppliesToRaySpread:
    def test_spread_scales_with_scenario_ratio(self):
        """scale_to_scenario=True 时簇内 ray spread 须按 §7.7.5.1 的
        AS_desired/AS_model 比例缩放, 而非保持表值 (Codex PR#18 P2)。"""
        from channel_model_38901.channel_generator import ChannelGenerator

        np.random.seed(13)
        sim = ChannelSimulator('UMa', 3.5e9, 'CDL-A',
                               use_median_lsps=True, scale_to_scenario=True)
        results = sim.run(logger=None)
        clusters, lsps = results['clusters'], results['lsps']

        table = _DATA['cdl_models']['CDL-A']
        powers = np.array([10 ** (c['power_db'] / 10) for c in table['clusters']])
        powers = powers / powers.sum()
        aoa_table = np.array([c['aoa_deg'] for c in table['clusters']])
        ratio = ChannelGenerator._azimuth_spread_ratio(
            aoa_table, powers, lsps.arrival_azimuth_spread_deg)
        expected = table['ray_spread_deg']['c_asa_deg'] * ratio

        assert ratio != pytest.approx(1.0, abs=1e-3)  # 前提: 缩放真发生了
        for c in clusters:
            assert c.as_aoa == pytest.approx(expected)

    def test_no_scaling_keeps_table_values(self):
        np.random.seed(13)
        sim = ChannelSimulator('UMa', 3.5e9, 'CDL-A', scale_to_scenario=False)
        clusters = sim.run(logger=None)['clusters']
        expected = _DATA['cdl_models']['CDL-A']['ray_spread_deg']['c_asa_deg']
        for c in clusters:
            assert c.as_aoa == expected


class TestMissingSpreadWarns:
    def test_scme_warns_on_missing_ray_spread(self):
        gen = _generator('SCME')
        with pytest.warns(UserWarning, match='ray_spread_deg'):
            gen._load_deterministic_clusters('SCME')

    def test_cdl_does_not_warn(self):
        gen = _generator('CDL-A')
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            gen._load_deterministic_clusters('CDL-A')

    def test_partial_ray_spread_block_warns_for_missing_dim(self):
        """ray_spread_deg 块存在但缺某个 key 时, 该维度仍须告警 (Codex PR#18 P2)。"""
        gen = _generator('CDL-A')
        pt = ParameterTables('synthetic')
        pt.data = {'cdl_models': {'CDL-Z': {
            'xpr_db': 10,
            'ray_spread_deg': {'c_asd_deg': 5.0, 'c_asa_deg': 11.0,
                               'c_zsd_deg': 3.0},  # 缺 c_zsa_deg
            'clusters': [
                {'id': 1, 'delay_ns': 0.0, 'power_db': 0.0,
                 'aod_deg': 0.0, 'aoa_deg': 0.0, 'zod_deg': 90.0, 'zoa_deg': 90.0},
            ],
        }}}
        gen.params = pt
        with pytest.warns(UserWarning, match='as_zoa_deg'):
            clusters = gen._load_deterministic_clusters('CDL-Z')
        assert clusters[0].as_zoa == 0.0
        assert clusters[0].as_aoa == 11.0

    def test_full_cluster_level_spread_does_not_warn(self):
        """模型级缺 ray_spread_deg 但簇级 as_* 齐全时不告警 (Codex PR#18 P3)。"""
        gen = _generator('CDL-A')
        pt = ParameterTables('synthetic')
        pt.data = {'cdl_models': {'CDL-Y': {
            'xpr_db': 10,
            'clusters': [
                {'id': 1, 'delay_ns': 0.0, 'power_db': 0.0,
                 'aod_deg': 0.0, 'aoa_deg': 0.0, 'zod_deg': 90.0, 'zoa_deg': 90.0,
                 'as_aod_deg': 5.0, 'as_aoa_deg': 11.0,
                 'as_zod_deg': 3.0, 'as_zoa_deg': 3.0},
            ],
        }}}
        gen.params = pt
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            clusters = gen._load_deterministic_clusters('CDL-Y')
        assert clusters[0].as_aoa == 11.0


class TestEndToEndSosExport:
    def test_rays_fan_out_in_doppler(self):
        """同簇 20 条射线必须有不同的 Doppler (角度真正散开)。

        历史 bug 下 as_*=0 → 射线角度全等 → 每簇 Doppler 唯一值。
        """
        np.random.seed(3)
        sim = ChannelSimulator('UMa', 3.5e9, 'CDL-A')
        sim.config.ue_velocities = np.array([[10.0, 0.0, 0.0]])
        sim.generator = sim._build_generator()
        sim.generator._ensure_antenna_positions(3e8 / 3.5e9)
        rows = sim.export_fpga_sos_params()
        assert rows
        by_cluster = {}
        for r in rows:
            by_cluster.setdefault(r['Cluster_ID'], set()).add(round(r['Doppler_Hz'], 9))
        spread_counts = [len(v) for v in by_cluster.values()]
        # 至少绝大多数簇的 20 条射线 Doppler 互不相同
        assert np.median(spread_counts) > 10
