"""Mode A 链路预算测试 (audit batch 2, 发现 A6)。

历史 bug: PropsimExportConfig.target_output_power_dbm 被接受 (gui.py 还在
传) 但全 repo 零读取 — Mode A 打印的 "Req ATT" 与 F64 口目标输出功率无关。

修复后: req_att = target_output_power_dbm + pathloss_db + digital_boost
(0 dBm 等效发射参考; 旧公式即 target=0 的特例), 且链路预算存入
exporter.last_mode_a_info 供调用方读取。
"""

import glob
import os

import numpy as np
import pytest

from mimo_ota_simulator.data_models import (
    AntennaArrayConfig, ChamberConfig, PropsimExportConfig, TargetChannelConfig,
)
from mimo_ota_simulator.exporters import PropsimASCIIExporter
from mimo_ota_simulator.simulator import MIMO_OTA_Simulator

BASE = "test_mode_a_power_output.asc"


@pytest.fixture(scope='module')
def sim_result():
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
    return MIMO_OTA_Simulator().run(chamber, channel, synthesis_method='ray')


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    for f in glob.glob(f"{os.path.splitext(BASE)[0]}*.asc"):
        try:
            os.remove(f)
        except OSError:
            pass


def _export(sim_result, mode='A', target_dbm=-10.0):
    cfg = PropsimExportConfig(
        filename=BASE, mode=mode, duration_s=0.001, sample_rate_hz=1000,
        target_output_power_dbm=target_dbm,
    )
    exporter = PropsimASCIIExporter(cfg)
    exporter.export(sim_result)
    return exporter


class TestModeATargetPower:
    def test_link_budget_exposed(self, sim_result):
        exp = _export(sim_result, target_dbm=-10.0)
        info = exp.last_mode_a_info
        assert info is not None, "Mode A 必须填充 last_mode_a_info"
        assert info['f64_output_power_dbm'] == -10.0
        assert info['required_attenuation_db'] == pytest.approx(
            info['f64_output_power_dbm'] + info['pathloss_db']
            + info['digital_boost_db'])

    def test_target_power_shifts_req_att_one_to_one(self, sim_result):
        """改 target_output_power_dbm 必须 1:1 改 req_att (修复前零效果)。"""
        att_lo = _export(sim_result, target_dbm=-10.0).last_mode_a_info['required_attenuation_db']
        att_hi = _export(sim_result, target_dbm=0.0).last_mode_a_info['required_attenuation_db']
        assert att_hi - att_lo == pytest.approx(10.0)

    def test_zero_target_recovers_legacy_formula(self, sim_result):
        info = _export(sim_result, target_dbm=0.0).last_mode_a_info
        assert info['required_attenuation_db'] == pytest.approx(
            info['pathloss_db'] + info['digital_boost_db'])

    def test_scale_independent_of_target_power(self, sim_result):
        # 归一化 scale 只看波形峰值, 不随目标功率变 (功率差只进 req_att)
        s1 = _export(sim_result, target_dbm=-10.0).last_mode_a_info['scale']
        s2 = _export(sim_result, target_dbm=5.0).last_mode_a_info['scale']
        assert s1 == pytest.approx(s2)

    def test_mode_b_unaffected(self, sim_result):
        exp = _export(sim_result, mode='B', target_dbm=-10.0)
        assert exp.last_mode_a_info is None
