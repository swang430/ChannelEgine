"""输入校验批测试 (audit batch 2, A5+A7+F12 — "no silent accept")。

历史 bug 共性: 非法/不生效的 config 被静默接受。本批锁定:
1. num_rays != 20 → UserWarning (合成器硬编码 20, 此前完全无视该字段);
2. 单极化 chamber + tx polarization 非 V → ValueError (rx 侧一直有, tx 漏);
3. 未知 cluster_model_name (如小写 typo) → ValueError (此前静默落 Stochastic);
4. custom profile 给了 xpr_db/initial_phases 但 chamber 单极化 → UserWarning。
"""

import warnings

import numpy as np
import pytest

from channel_model_38901.simulator import ChannelSimulator
from mimo_ota_simulator.cdl_schema import CustomCDLProfile
from mimo_ota_simulator.channel_builders import CustomCDLBuilder
from mimo_ota_simulator.data_models import (
    AntennaArrayConfig, ChamberConfig, TargetChannelConfig,
)
from mimo_ota_simulator.simulator import MIMO_OTA_Simulator


def _profile(**cluster_extra):
    base = {
        'delay_s': 0.0, 'power_linear': 1.0,
        'aoa_deg': 30.0, 'aod_deg': 10.0,
        'as_aoa_deg': 5.0, 'as_aod_deg': 2.0,
    }
    base.update(cluster_extra)
    return CustomCDLProfile.from_dict({
        'center_frequency_hz': 2.6e9, 'pathloss_db': 80.0,
        'ue_velocity_mps': [1.0, 0.0, 0.0], 'is_los': False,
        'clusters': [base],
    })


def _chamber(dual=False):
    return ChamberConfig(
        num_probes=8, dual_polarized=dual, distribution='ring', radius_m=1.5,
        probe_antenna={'gain_dBi': 0.0, 'pattern': 'Isotropic'},
    )


def _channel(profile, tx_pol='V', rx_pol='V'):
    return TargetChannelConfig(
        input_mode='custom', custom_profile=profile,
        center_frequency_hz=profile.center_frequency_hz,  # 避免触发频率不一致 warning
        tx_antenna=AntennaArrayConfig(num_cols=2, polarization=tx_pol),
        rx_antenna=AntennaArrayConfig(num_cols=2, polarization=rx_pol),
    )


class TestNumRaysWarn:
    def test_non_20_num_rays_warns(self):
        profile = _profile(num_rays=10)
        with pytest.warns(UserWarning, match='num_rays'):
            CustomCDLBuilder(profile).build()

    def test_default_20_does_not_warn(self):
        profile = _profile()
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            CustomCDLBuilder(profile).build()


class TestTxPolarizationGuard:
    @pytest.mark.parametrize('tx_pol', ['H', '+45', '-45'])
    def test_single_pol_chamber_rejects_non_v_tx(self, tx_pol):
        with pytest.raises(ValueError, match='tx_antenna.polarization'):
            MIMO_OTA_Simulator().run(
                _chamber(dual=False), _channel(_profile(), tx_pol=tx_pol),
                synthesis_method='ray')

    def test_dual_pol_chamber_accepts_h_tx(self):
        np.random.seed(7)
        result = MIMO_OTA_Simulator().run(
            _chamber(dual=True),
            _channel(_profile(xpr_db=7.0), tx_pol='H', rx_pol='V'),
            synthesis_method='ray')
        assert result['transmit_mapping_matrix_A'] is not None


class TestClusterModelNameWhitelist:
    @pytest.mark.parametrize('bad', ['cdl-a', 'CDLA', 'cdl_a', 'random', 'CDLB'])
    def test_unknown_name_raises(self, bad):
        with pytest.raises(ValueError, match='cluster_model_name'):
            ChannelSimulator('UMa', 3.5e9, bad)

    @pytest.mark.parametrize('good', [None, 'Stochastic', 'CDL-A', 'CDL-E', 'SCME'])
    def test_known_names_accepted(self, good):
        np.random.seed(3)
        ChannelSimulator('UMa', 3.5e9, good)


class TestSinglePolXprWarn:
    def test_xpr_under_single_pol_warns(self):
        np.random.seed(7)
        with pytest.warns(UserWarning, match='dual_polarized'):
            MIMO_OTA_Simulator().run(
                _chamber(dual=False), _channel(_profile(xpr_db=7.0)),
                synthesis_method='ray')

    def test_initial_phases_under_single_pol_warns(self):
        np.random.seed(7)
        with pytest.warns(UserWarning, match='dual_polarized'):
            MIMO_OTA_Simulator().run(
                _chamber(dual=False),
                _channel(_profile(initial_phases_rad=[0.0, 0.5, 1.0, 1.5])),
                synthesis_method='ray')

    def test_plain_profile_single_pol_does_not_warn(self):
        np.random.seed(7)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            MIMO_OTA_Simulator().run(
                _chamber(dual=False), _channel(_profile()),
                synthesis_method='ray')
