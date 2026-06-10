import json
from typing import Dict, Any

class ParameterTables:
    """A data class to hold all parameter tables loaded from the standard."""
    def __init__(self, file_path: str):
        self.param_file_path = file_path
        self.data: Dict[str, Any] = {}

    def get_scenario_params(self, scenario_name: str) -> Dict[str, Any]:
        return self.data.get("scenarios", {}).get(scenario_name, {})

    def get_cdl_model_params(self, model_name: str) -> Dict[str, Any]:
        return self.data.get("cdl_models", {}).get(model_name, {})

    def get_scme_model_params(self, model_name: str) -> Dict[str, Any]:
        return self.data.get("scme_models", {}).get(model_name, {})

class ParameterLoader:
    """
    Loads and parses the 3GPP TR 38.901 parameters from a JSON file.
    """
    @staticmethod
    def load_parameters(file_path: str) -> ParameterTables:
        """
        Parses the JSON parameter file and populates a ParameterTables object.

        Args:
            file_path: The absolute path to the JSON parameter file.

        Returns:
            A ParameterTables object filled with the standard's data.
        """
        tables = ParameterTables(file_path)
        with open(file_path, 'r') as f:
            tables.data = json.load(f)
        # Basic validation
        if "standard_version" not in tables.data or "scenarios" not in tables.data:
            raise ValueError("Invalid parameter file format")
        ParameterLoader._warn_unconsumed_keys(tables.data)
        return tables

    # 引擎实际消费的 key 集合 (audit batch 2 / PR15 配置消费校验)。
    # 新增 JSON 字段时必须同步登记, 否则加载时告警 — 防止 dead config
    # (历史案例: los_probability / shadow_fading_std_db / scaling_factors
    # 表在被消费前分别 dead 了数月)。
    _KNOWN_SCENARIO_KEYS = frozenset({
        'delay_scaling_factor', 'delay_scaling_factor_los', 'delay_scaling_factor_nlos',
        'lsp_parameters_los', 'lsp_parameters_nlos',
        'num_clusters_los', 'num_clusters_nlos',
        'pathloss_los', 'pathloss_nlos',
        'ray_angle_offsets_deg',
        'scaling_factors_aoa_aod', 'scaling_factors_zoa_zod',
        'xpr_mu_los', 'xpr_mu_nlos', 'xpr_sigma_los', 'xpr_sigma_nlos',
        'V', 'S',  # InF 专用 (inf_models.py)
    })
    # mu/sigma 按 LSP 名枚举而非前缀匹配 (Codex PR#30 P2: 前缀会放过
    # mu_log10_DSS 这类拼写错误)。注: 类体内推导式不可引用类属性,
    # LSP 名元组直接内联。
    _KNOWN_LSP_KEYS = frozenset(
        {'correlations',
         'c_asa_deg', 'c_asd_deg', 'c_zsa_deg',
         'per_cluster_shadowing_std_db'}
        | {f'{p}{n}' for p in ('mu_log10_', 'sigma_log10_')
           for n in ('DS', 'ASD', 'ASA', 'ZSA', 'SF', 'K', 'ZSD')}
    )
    _KNOWN_PATHLOSS_KEYS = frozenset({'formula', 'formula_1', 'formula_2'})

    @classmethod
    def _warn_unconsumed_keys(cls, data: dict) -> None:
        """对 scenario / lsp / pathloss 块做消费校验, 未知 key 发 UserWarning。"""
        import warnings
        for scenario, params in data.get('scenarios', {}).items():
            unknown = [k for k in params if k not in cls._KNOWN_SCENARIO_KEYS]
            if unknown:
                warnings.warn(
                    f"参数文件 scenario '{scenario}' 含引擎不消费的 key {unknown} "
                    f"— dead config 或拼写错误? (新字段须登记 _KNOWN_SCENARIO_KEYS)",
                    UserWarning,
                )
            for lsp_set in ('lsp_parameters_los', 'lsp_parameters_nlos'):
                lsp = params.get(lsp_set) or {}
                unknown = [k for k in lsp if k not in cls._KNOWN_LSP_KEYS]
                if unknown:
                    warnings.warn(
                        f"参数文件 '{scenario}/{lsp_set}' 含引擎不消费的 key "
                        f"{unknown} — dead config 或拼写错误?",
                        UserWarning,
                    )
            for pl_set in ('pathloss_los', 'pathloss_nlos'):
                pl = params.get(pl_set) or {}
                unknown = [k for k in pl if k not in cls._KNOWN_PATHLOSS_KEYS]
                if unknown:
                    warnings.warn(
                        f"参数文件 '{scenario}/{pl_set}' 含引擎不消费的 key "
                        f"{unknown} — dead config 或拼写错误?",
                        UserWarning,
                    )