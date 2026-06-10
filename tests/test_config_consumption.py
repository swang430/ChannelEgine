"""配置消费校验测试 (audit batch 2, PR15 — F13/F14/F16/F17/F18 清理 + 机制)。

历史 dead config: los_probability 块、pathloss 内 shadow_fading_std_db、
scaling_factors 表 (PR10 前)、c_*_deg (PR9 前) — 共性是 JSON 提供了但
引擎从不读取, 且无任何告警。本机制让 ParameterLoader 在加载时核对
key 消费集合, 未知 key 直接告警。
"""

import json
import warnings
from pathlib import Path

import pytest

from channel_model_38901.parameters.parameter_loader import ParameterLoader

PARAM_FILE = str(
    Path(__file__).resolve().parents[1]
    / 'channel_model_38901' / 'parameters' / 'parameters1013_tr38901_v19.0.0.json'
)


class TestConsumptionValidator:
    def test_canonical_file_loads_clean(self):
        """清理后的生产参数文件不得触发任何消费告警。"""
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            ParameterLoader.load_parameters(PARAM_FILE)

    def test_unknown_scenario_key_warns(self, tmp_path):
        with open(PARAM_FILE, encoding='utf-8') as f:
            d = json.load(f)
        d['scenarios']['UMa']['los_probability'] = {'formula': 'x'}  # 历史 dead key
        p = tmp_path / 'params.json'
        p.write_text(json.dumps(d))
        with pytest.warns(UserWarning, match='los_probability'):
            ParameterLoader.load_parameters(str(p))

    def test_unknown_lsp_key_warns(self, tmp_path):
        with open(PARAM_FILE, encoding='utf-8') as f:
            d = json.load(f)
        d['scenarios']['UMa']['lsp_parameters_los']['c_asa_dge'] = 11  # 拼写错误
        p = tmp_path / 'params.json'
        p.write_text(json.dumps(d))
        with pytest.warns(UserWarning, match='c_asa_dge'):
            ParameterLoader.load_parameters(str(p))

    def test_unknown_pathloss_key_warns(self, tmp_path):
        with open(PARAM_FILE, encoding='utf-8') as f:
            d = json.load(f)
        d['scenarios']['UMa']['pathloss_los']['shadow_fading_std_db'] = 4  # 历史 dead key
        p = tmp_path / 'params.json'
        p.write_text(json.dumps(d))
        with pytest.warns(UserWarning, match='shadow_fading_std_db'):
            ParameterLoader.load_parameters(str(p))


class TestDeadCodeRemoved:
    def test_parsers_module_gone(self):
        with pytest.raises(ImportError):
            from mimo_ota_simulator import parsers  # noqa: F401

    def test_backup_param_files_gone(self):
        pdir = Path(PARAM_FILE).parent
        assert not (pdir / 'parameters_tr38901_v19.0.0.json').exists()
        assert not (pdir / 'parameters2_tr38901_v19.0.0.json').exists()

    def test_dead_channel_profile_gone(self):
        import channel_model_38901.common.data_models as dm
        assert not hasattr(dm, 'ChannelProfile')
