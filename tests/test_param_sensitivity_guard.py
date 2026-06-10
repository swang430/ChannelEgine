"""参数敏感性守门测试 (audit batch 2, PR14 — 防 silent-ignore 复发机制)。

原则: 每个用户可配置字段, 取一个非默认值后输出必须变化 (或显式
raise/warn)。历史上的 element_pattern (PR#15/#19)、num_rays (PR#20)、
target_output_power_dbm (PR#21)、enable_ofdm (PR#28) 等 silent-ignore
都能被这一类测试当场抓住。

新增 config 字段时必须在 CASES 表里登记 — 要么有敏感性条目, 要么在
EXEMPT 里写明豁免原因 (如 metadata-only 字段)。
"""

import numpy as np
import pytest

from channel_model_38901.simulator import ChannelSimulator
from mimo_ota_simulator.data_models import (
    AntennaArrayConfig, ChamberConfig, ProbePosition, TargetChannelConfig,
)
from mimo_ota_simulator.simulator import MIMO_OTA_Simulator


# ---------- OTA 端到端 ----------

def _chamber(**over):
    base = dict(num_probes=8, distribution='ring', radius_m=1.5,
                dual_polarized=False,
                probe_antenna={'gain_dBi': 0.0, 'pattern': 'Isotropic'})
    base.update(over)
    return ChamberConfig(**base)


def _antenna(**over):
    base = dict(num_cols=2)
    base.update(over)
    return AntennaArrayConfig(**base)


def _channel(**over):
    base = dict(model_name='UMa', cluster_model_name='CDL-C',
                center_frequency_hz=3.5e9, force_condition='NLOS',
                random_seed=42, tx_antenna=_antenna(), rx_antenna=_antenna())
    base.update(over)
    return TargetChannelConfig(**base)


def _signature(chamber, channel, jitter=False):
    """可比较的输出签名: A + B + ray Doppler 序列。"""
    np.random.seed(42)
    r = MIMO_OTA_Simulator().run(chamber, channel, synthesis_method='ray',
                                 probe_phase_jitter=jitter)
    parts = [r['transmit_mapping_matrix_A'].ravel(), r['B_matrix'].ravel()]
    if r.get('ray_components'):
        parts.append(np.array([rc['doppler_hz'] for rc in r['ray_components']],
                              dtype=complex))
    return np.concatenate([p.astype(complex) for p in parts])


def _changed(sig_a, sig_b):
    return sig_a.shape != sig_b.shape or not np.allclose(sig_a, sig_b)


# (case 名, chamber 覆盖, channel 覆盖, run kwargs)
OTA_CASES = [
    ('chamber.num_probes', dict(num_probes=6), {}, {}),
    ('chamber.radius_m', dict(radius_m=2.5), {}, {}),
    ('chamber.dual_polarized', dict(dual_polarized=True), {}, {}),
    ('chamber.probe_antenna.gain_dBi',
     dict(probe_antenna={'gain_dBi': 6.0, 'pattern': 'Isotropic'}), {}, {}),
    ('chamber.probe_antenna.pattern',
     dict(probe_antenna={'gain_dBi': 0.0, 'pattern': 'TR38.901'}), {}, {}),
    ('chamber.probe_positions',
     dict(distribution='custom',
          probe_positions=[ProbePosition(azimuth_deg=a, elevation_deg=e)
                           for a, e in ((0, 30), (90, 30), (180, 30), (270, 30),
                                        (0, -30), (90, -30), (180, -30), (270, -30))]),
     {}, {}),
    ('channel.model_name', {}, dict(model_name='UMi-StreetCanyon'), {}),
    ('channel.cluster_model_name', {}, dict(cluster_model_name='CDL-B'), {}),
    ('channel.center_frequency_hz', {}, dict(center_frequency_hz=2.6e9), {}),
    ('channel.random_seed', {}, dict(random_seed=43), {}),
    ('channel.force_condition', {}, dict(force_condition='LOS'), {}),
    ('channel.use_median_lsps', {}, dict(use_median_lsps=True), {}),
    ('channel.scale_to_scenario', {}, dict(scale_to_scenario=True), {}),
    ('channel.split_subclusters', {}, dict(split_subclusters=True), {}),
    ('channel.c_ds_ns', {},
     dict(split_subclusters=True, c_ds_ns=10.0), {}),
    ('tx_antenna.num_cols', {}, dict(tx_antenna=_antenna(num_cols=3)), {}),
    ('rx_antenna.num_cols', {}, dict(rx_antenna=_antenna(num_cols=3)), {}),
    ('tx_antenna.spacing_h', {}, dict(tx_antenna=_antenna(spacing_h=0.8)), {}),
    ('rx_antenna.spacing_h', {}, dict(rx_antenna=_antenna(spacing_h=0.8)), {}),
    ('tx_antenna.element_pattern', {},
     dict(tx_antenna=_antenna(element_pattern='TR38.901')), {}),
    ('rx_antenna.element_pattern', {},
     dict(rx_antenna=_antenna(element_pattern='TR38.901')), {}),
    ('tx_antenna.bearing_deg', {}, dict(tx_antenna=_antenna(bearing_deg=30.0)), {}),
    ('rx_antenna.bearing_deg', {}, dict(rx_antenna=_antenna(bearing_deg=30.0)), {}),
    ('tx_antenna.downtilt_deg', {}, dict(tx_antenna=_antenna(downtilt_deg=10.0)), {}),
    ('rx_antenna.downtilt_deg', {}, dict(rx_antenna=_antenna(downtilt_deg=10.0)), {}),
    ('run.probe_phase_jitter', {}, {}, dict(jitter=True)),
    ('tx_antenna.array_type_URA', {},
     dict(tx_antenna=_antenna(array_type='URA', num_rows=2, num_cols=2)), {}),
]

# 已知 metadata-only / 间接生效字段的豁免清单 (必须写明原因)
EXEMPT = {
    'chamber.quiet_zone_radius_m': 'metadata-only (audit batch 1 F1 已注明)',
    'channel.bandwidth_hz': '仅 WiFi 标准生效 (test_ofdm_passthrough 覆盖)',
    'channel.enable_ofdm': 'test_ofdm_passthrough 覆盖',
    'channel.n_fft': 'test_ofdm_passthrough 覆盖',
    'channel.enable_ul_dl': '只增 *_ul 键, A/B 不变 (test_jitter_export 覆盖 UL)',
    'channel.reciprocity_transform': '仅 enable_ul_dl=True 生效 (test_jitter_export 覆盖)',
    'channel.input_mode/custom_profile': 'test_validation_guards/test_custom_cdl_e2e 覆盖',
    'export.*': 'test_propsim_export/test_mode_a_power/test_jitter_export 覆盖',
}


@pytest.fixture(scope='module')
def baseline_signature():
    return _signature(_chamber(), _channel())


class TestOtaParamSensitivity:
    @pytest.mark.parametrize(
        'name,chamber_over,channel_over,run_kw',
        OTA_CASES, ids=[c[0] for c in OTA_CASES])
    def test_non_default_value_changes_output(
            self, baseline_signature, name, chamber_over, channel_over, run_kw):
        sig = _signature(_chamber(**chamber_over), _channel(**channel_over),
                         **run_kw)
        assert _changed(baseline_signature, sig), \
            f'{name}: 改非默认值后输出未变化 — silent-ignore?'


# ---------- channel model 直连 ----------

def _h_signature(scenario='UMa', fc=3.5e9, cluster='CDL-A', seed=42,
                 bs=(0.0, 0.0, 25.0), ue=(100.0, 0.0, 1.5), **kw):
    np.random.seed(seed)
    sim = ChannelSimulator(scenario, fc, cluster, **kw)
    sim.config.bs_positions = np.array([bs])
    sim.config.ue_positions = np.array([ue])
    # 多天线阵: 1×1 时平面波相位恒为 1, H 对角度类参数天然不敏感,
    # 守门测试必须用能感知角度的配置
    ant = sim.config.antenna_config
    ant.tx_array_type = 'ULA'; ant.num_tx_rows = 1; ant.num_tx_cols = 4
    ant.rx_array_type = 'ULA'; ant.num_rx_rows = 1; ant.num_rx_cols = 2
    sim.generator = sim._build_generator()
    r = sim.run(logger=None)
    return np.concatenate([
        r['channel_matrix'].ravel(),
        np.array([r['pathloss_db']], dtype=complex),
    ])


CM_CASES = [
    ('scenario_name', dict(scenario='RMa')),
    ('center_frequency_hz', dict(fc=2.6e9)),
    ('cluster_model_name', dict(cluster='CDL-B')),
    ('cluster_stochastic', dict(cluster=None)),
    ('use_median_lsps', dict(use_median_lsps=True)),
    ('scale_to_scenario', dict(scale_to_scenario=True)),
    ('split_subclusters', dict(split_subclusters=True)),
    ('bs_position', dict(bs=(0.0, 0.0, 30.0))),
    ('ue_position', dict(ue=(200.0, 50.0, 1.5))),
    ('seed', dict(seed=43)),
]


class TestChannelModelParamSensitivity:
    @pytest.mark.parametrize('name,over', CM_CASES, ids=[c[0] for c in CM_CASES])
    def test_non_default_value_changes_output(self, name, over):
        base = _h_signature()
        sig = _h_signature(**over)
        assert base.shape != sig.shape or not np.allclose(base, sig), \
            f'{name}: 改非默认值后 H/pathloss 未变化 — silent-ignore?'
