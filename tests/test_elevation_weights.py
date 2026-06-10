"""探头权重 elevation-aware 测试 (audit batch 2, P8 / P2-2 / P1-2)。

历史 bug: cluster→probe 功率路由是 azimuth-only 高斯 — 同方位不同仰角的
探头 (P2-7 dual-ring) 拿到完全相同的权重, 簇 zoa=30° 与 zoa=90° 的映射
一模一样; 而 Integration Spec 把 zoa "Consumed" 写得没有 caveat。

修复后: (az, el) 各向异性高斯 (σ_az=as_aoa, σ_el=as_zoa, 各带展宽下限)。
ring 布局 + 水平簇 (zoa=90°) 时仰角项恒 1 — 与旧公式 bit-identical。
"""

import numpy as np
import pytest

from channel_model_38901.common.data_models import EngineCluster
from mimo_ota_simulator.data_models import ChamberConfig, ProbePosition
from mimo_ota_simulator.simulator import MIMO_OTA_Simulator


def _dual_ring_chamber():
    # 同方位 0/90/180/270, 上环 +30° 下环 -30°
    positions = [ProbePosition(azimuth_deg=az, elevation_deg=el)
                 for el in (30.0, -30.0) for az in (0.0, 90.0, 180.0, 270.0)]
    return ChamberConfig(
        num_probes=8, distribution='custom', probe_positions=positions,
        radius_m=1.5, dual_polarized=False,
        probe_antenna={'gain_dBi': 0.0, 'pattern': 'Isotropic'},
    )


def _ring_chamber(n=8):
    return ChamberConfig(
        num_probes=n, distribution='ring', radius_m=1.5, dual_polarized=False,
        probe_antenna={'gain_dBi': 0.0, 'pattern': 'Isotropic'},
    )


def _weights(cluster, chamber):
    return MIMO_OTA_Simulator._calculate_weights_for_cluster(cluster, chamber).ravel()


class TestElevationSelectivity:
    def test_dual_ring_prefers_matching_elevation(self):
        """簇仰角 +30° (zoa=60°) 时上环探头权重必须大于下环 (修复前两环相同)。"""
        chamber = _dual_ring_chamber()
        cluster = EngineCluster(aoa_phi=0.0, zoa_theta=60.0,
                                as_aoa=10.0, as_zoa=10.0)
        w = _weights(cluster, chamber)
        upper = w[:4].sum()   # +30° 环
        lower = w[4:].sum()   # -30° 环
        assert upper > lower * 2.0

    def test_dual_ring_symmetric_for_horizontal_cluster(self):
        # 水平簇 (zoa=90°) 对上下环对称。aoa=45° 使候选 top-4 截断
        # 时上下环成对并列 (aoa 正对探头时截断的并列破缺与仰角无关,
        # legacy azimuth-only 同样存在)
        chamber = _dual_ring_chamber()
        cluster = EngineCluster(aoa_phi=45.0, zoa_theta=90.0,
                                as_aoa=10.0, as_zoa=10.0)
        w = _weights(cluster, chamber)
        assert w[:4].sum() == pytest.approx(w[4:].sum())

    def test_cluster_zenith_now_changes_weights(self):
        """zoa=60° 与 zoa=90° 在立体布局下权重必须不同 (修复前相同 = P1-2)。"""
        chamber = _dual_ring_chamber()
        w_60 = _weights(EngineCluster(aoa_phi=0.0, zoa_theta=60.0,
                                      as_aoa=10.0, as_zoa=10.0), chamber)
        w_90 = _weights(EngineCluster(aoa_phi=0.0, zoa_theta=90.0,
                                      as_aoa=10.0, as_zoa=10.0), chamber)
        assert not np.allclose(w_60, w_90)


class TestRingBackwardCompat:
    def test_ring_horizontal_cluster_matches_legacy_formula(self):
        """ring + zoa=90°: 权重必须等于历史 azimuth-only 高斯 (bit-compat)。"""
        chamber = _ring_chamber(8)
        cluster = EngineCluster(aoa_phi=37.0, zoa_theta=90.0,
                                as_aoa=20.0, as_zoa=15.0)
        w = _weights(cluster, chamber)

        # 重算 legacy 公式
        probe_az = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        target = np.deg2rad(37.0)
        sigma = max(np.deg2rad(20.0), (2 * np.pi / 8) / 1.5)
        d = np.abs(probe_az - target)
        d = np.minimum(d, 2 * np.pi - d)
        idx = np.argsort(d)[:4]
        legacy = np.zeros(8)
        legacy[idx] = np.exp(-d[idx] ** 2 / (2 * sigma ** 2))
        legacy /= legacy.sum()
        np.testing.assert_allclose(w, legacy, rtol=1e-12)

    def test_ring_off_horizon_cluster_attenuates_uniformly(self):
        # ring (全部 el=0) 对 zoa=60° 簇: 所有探头同样的仰角失配,
        # 归一化后权重分布与水平簇一致 (相对路由不变, 物理上限由 B 矩阵管)
        chamber = _ring_chamber(8)
        w_60 = _weights(EngineCluster(aoa_phi=37.0, zoa_theta=60.0,
                                      as_aoa=20.0, as_zoa=15.0), chamber)
        w_90 = _weights(EngineCluster(aoa_phi=37.0, zoa_theta=90.0,
                                      as_aoa=20.0, as_zoa=15.0), chamber)
        np.testing.assert_allclose(w_60, w_90, rtol=1e-12)
