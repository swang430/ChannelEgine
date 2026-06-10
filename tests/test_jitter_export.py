"""probe_phase_jitter 进 F64 导出测试 (audit batch 2, P2-1 + Codex PR#22 P2)。

历史 bug: jitter 只乘进 transmit_mapping_matrix_A 快照, exporter 从不消费
probe_phase_jitter_vector — 导出 .asc 是无 jitter 信道, 与 effective_H_matrix
静默不一致 (simulator.py 注释自己承认 "must apply jitter there too")。

修复后: export() 出口把 jitter 相量施加到 tensor 的物理探头轴 — DL 在
dim0; UL 因互易转置探头在 dim1, conjugate_transpose 下相量共轭 (simulator
预生成 probe_phase_jitter_vector_ul)。
"""

import warnings

import numpy as np
import pytest

from mimo_ota_simulator.data_models import (
    AntennaArrayConfig, ChamberConfig, PropsimExportConfig, TargetChannelConfig,
)
from mimo_ota_simulator.exporters import PropsimASCIIExporter
from mimo_ota_simulator.simulator import MIMO_OTA_Simulator


def _run(jitter, ul=False, reciprocity='conjugate_transpose'):
    chamber = ChamberConfig(
        num_probes=4, distribution='ring', radius_m=2.0, dual_polarized=False,
        probe_antenna={"gain_dBi": 0.0, "pattern": "Isotropic"},
    )
    channel = TargetChannelConfig(
        model_name='UMa', cluster_model_name='CDL-C',
        center_frequency_hz=3.5e9, force_condition='NLOS', random_seed=42,
        tx_antenna=AntennaArrayConfig(num_rows=1, num_cols=2),
        rx_antenna=AntennaArrayConfig(num_rows=1, num_cols=1),
        enable_ul_dl=ul, reciprocity_transform=reciprocity,
    )
    np.random.seed(42)
    return MIMO_OTA_Simulator().run(
        chamber, channel, synthesis_method='ray', probe_phase_jitter=jitter)


def _exporter():
    return PropsimASCIIExporter(PropsimExportConfig(
        filename='unused.asc', mode='A', duration_s=0.001, sample_rate_hz=1000))


def _tensor(sim_result, direction='dl', return_exporter=False):
    exporter = _exporter()
    key_rc = 'ray_components' if direction == 'dl' else 'ray_components_ul'
    key_ir = ('channel_impulse_response' if direction == 'dl'
              else 'channel_impulse_response_ul')
    tensor, _ = exporter._compute_dynamic_tensor(
        sim_result, sim_result[key_ir], sim_result[key_rc],
        apply_calibration=(direction == 'dl'))
    tensor = exporter._apply_export_jitter(tensor, sim_result, direction)
    return (tensor, exporter) if return_exporter else tensor


class TestJitterReachesExport:
    def test_export_tensor_carries_jitter_phasor(self):
        """导出 tensor(有 jitter)/tensor(无 jitter) 必须等于 jitter 相量。"""
        r_plain = _run(jitter=False)
        r_jit = _run(jitter=True)
        jv = r_jit['probe_phase_jitter_vector']
        assert jv is not None

        t_plain = _tensor(r_plain)
        t_jit = _tensor(r_jit)
        # 同 seed 下底层 W_base 相同; 修复前 t_jit == t_plain (jitter 丢失),
        # 修复后 DL 探头轴 (dim0) 相差对应相量
        mask = np.abs(t_plain) > 1e-12
        ratio = np.where(mask, t_jit / np.where(mask, t_plain, 1.0), np.nan)
        for p in range(t_plain.shape[1]):
            vals = ratio[:, p, :, :][~np.isnan(ratio[:, p, :, :])]
            if vals.size:
                np.testing.assert_allclose(vals, jv[p, 0], rtol=1e-9)

    def test_export_consistent_with_effective_h(self):
        """效验口径: t=0 tap 求和 (除以 Mode A scale) == jitter 后的 A。"""
        r_jit = _run(jitter=True)
        t_jit, exporter = _tensor(r_jit, return_exporter=True)
        scale = exporter.last_mode_a_info['scale']
        a_from_taps = t_jit[0].sum(axis=-1) / scale
        np.testing.assert_allclose(
            a_from_taps, r_jit['transmit_mapping_matrix_A'], rtol=1e-6, atol=1e-9)

    @pytest.mark.parametrize('reciprocity', ['transpose', 'conjugate_transpose'])
    def test_ul_export_consistent_with_a_ul(self, reciprocity):
        """UL: 探头在 dim1 且按互易模式取共轭 (Codex PR#22 P2)。

        t=0 tap 求和 (除以 scale) 必须等于 transmit_mapping_matrix_A_ul —
        修复前 UL jitter 因轴不匹配被丢弃或乘错轴。
        """
        r = _run(jitter=True, ul=True, reciprocity=reciprocity)
        assert r['probe_phase_jitter_vector_ul'] is not None
        t_ul, exporter = _tensor(r, direction='ul', return_exporter=True)
        scale = exporter.last_mode_a_info['scale']
        a_ul_from_taps = t_ul[0].sum(axis=-1) / scale
        np.testing.assert_allclose(
            a_ul_from_taps, r['transmit_mapping_matrix_A_ul'],
            rtol=1e-6, atol=1e-9)

    def test_no_jitter_export_unchanged(self):
        r_plain = _run(jitter=False)
        assert r_plain['probe_phase_jitter_vector'] is None
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            _tensor(r_plain)  # 无 jitter: 行为与修复前 bit-identical, 零告警

    def test_ul_vector_derived_when_filtered_out(self):
        """白名单过滤掉 *_ul 键时, 由 DL 相量 + 互易模式就地推导
        (Codex PR#22 P2: GUI adapter 场景)。"""
        r = _run(jitter=True, ul=True, reciprocity='conjugate_transpose')
        expected = r['probe_phase_jitter_vector_ul']
        filtered = dict(r)
        filtered['probe_phase_jitter_vector_ul'] = None  # 模拟 GUI 白名单丢键
        t_full, exporter = _tensor(r, direction='ul', return_exporter=True)
        t_filtered = _tensor(filtered, direction='ul')
        np.testing.assert_allclose(t_filtered, t_full, rtol=1e-12)
        np.testing.assert_allclose(
            expected, np.conj(r['probe_phase_jitter_vector']))

    def test_ul_vector_missing_and_underivable_warns(self):
        r = _run(jitter=True, ul=True)
        filtered = dict(r)
        filtered['probe_phase_jitter_vector_ul'] = None
        filtered['reciprocity_transform'] = None
        with pytest.warns(UserWarning, match='probe_phase_jitter_vector_ul'):
            _tensor(filtered, direction='ul')

    def test_mismatched_jitter_length_warns(self):
        r_jit = _run(jitter=True)
        r_jit['probe_phase_jitter_vector'] = np.ones((7, 1), dtype=complex)
        with pytest.warns(UserWarning, match='probe_phase_jitter_vector'):
            _tensor(r_jit)
