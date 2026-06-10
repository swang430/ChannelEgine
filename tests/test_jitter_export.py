"""probe_phase_jitter 进 F64 导出测试 (audit batch 2, A-jitter/P2-1)。

历史 bug: jitter 只乘进 transmit_mapping_matrix_A 快照, exporter 从不消费
probe_phase_jitter_vector — 导出 .asc 是无 jitter 信道, 与 effective_H_matrix
静默不一致 (simulator.py 注释自己承认 "must apply jitter there too")。

修复后: 导出 tensor 的 per-probe 因子折入 jitter 相量, 两条路径一致。
"""

import warnings

import numpy as np
import pytest

from mimo_ota_simulator.data_models import (
    AntennaArrayConfig, ChamberConfig, PropsimExportConfig, TargetChannelConfig,
)
from mimo_ota_simulator.exporters import PropsimASCIIExporter
from mimo_ota_simulator.simulator import MIMO_OTA_Simulator


def _run(jitter):
    chamber = ChamberConfig(
        num_probes=4, distribution='ring', radius_m=2.0, dual_polarized=False,
        probe_antenna={"gain_dBi": 0.0, "pattern": "Isotropic"},
    )
    channel = TargetChannelConfig(
        model_name='UMa', cluster_model_name='CDL-C',
        center_frequency_hz=3.5e9, force_condition='NLOS', random_seed=42,
        tx_antenna=AntennaArrayConfig(num_rows=1, num_cols=1),
        rx_antenna=AntennaArrayConfig(num_rows=1, num_cols=1),
    )
    np.random.seed(42)
    return MIMO_OTA_Simulator().run(
        chamber, channel, synthesis_method='ray', probe_phase_jitter=jitter)


def _tensor(sim_result, return_exporter=False):
    exporter = PropsimASCIIExporter(PropsimExportConfig(
        filename='unused.asc', mode='A', duration_s=0.001, sample_rate_hz=1000))
    tensor, _ = exporter._compute_dynamic_tensor(
        sim_result, sim_result['channel_impulse_response'],
        sim_result['ray_components'])
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
        # 修复后每个 probe 轴相差对应相量
        mask = np.abs(t_plain) > 1e-12
        ratio = np.where(mask, t_jit / np.where(mask, t_plain, 1.0), np.nan)
        for p in range(t_plain.shape[1]):
            vals = ratio[:, p, :, :][~np.isnan(ratio[:, p, :, :])]
            if vals.size:
                np.testing.assert_allclose(vals, jv[p, 0], rtol=1e-9)

    def test_export_consistent_with_effective_h(self):
        """效验口径: 导出含 jitter 后, A 快照与 tensor 同源 — 静态簇求和
        (t=0) 应等于 jitter 后的 A (ray 模式下两者同构)。"""
        r_jit = _run(jitter=True)
        t_jit, exporter = _tensor(r_jit, return_exporter=True)
        scale = exporter.last_mode_a_info['scale']  # tensor 含 Mode A 归一化
        a_from_taps = t_jit[0].sum(axis=-1) / scale  # (P, T): tap 在 t=0 求和
        np.testing.assert_allclose(
            a_from_taps, r_jit['transmit_mapping_matrix_A'], rtol=1e-6, atol=1e-9)

    def test_no_jitter_export_unchanged(self):
        r_plain = _run(jitter=False)
        assert r_plain['probe_phase_jitter_vector'] is None
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            _tensor(r_plain)  # 无 jitter: 行为与修复前 bit-identical, 零告警

    def test_mismatched_jitter_length_warns(self):
        r_jit = _run(jitter=True)
        r_jit['probe_phase_jitter_vector'] = np.ones((7, 1), dtype=complex)
        with pytest.warns(UserWarning, match='probe_phase_jitter_vector'):
            _tensor(r_jit)
