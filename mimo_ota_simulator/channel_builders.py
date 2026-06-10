import json
import warnings
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, List
from mimo_ota_simulator.data_models import TargetChannelConfig, UniversalChannelProfile
from mimo_ota_simulator.cdl_schema import CustomCDLProfile
from channel_model_38901.common.data_models import EngineCluster
from channel_model_38901.simulator import ChannelSimulator

class ChannelProfileBuilder(ABC):
    """Abstract base class for channel profile builders."""
    @abstractmethod
    def build(self) -> UniversalChannelProfile:
        pass

class Standard3GPPBuilder(ChannelProfileBuilder):
    """Builds a channel profile using the internal 3GPP TR 38.901 Generation Engine."""
    def __init__(self, config: TargetChannelConfig):
        self.config = config

    def build(self) -> UniversalChannelProfile:
        # Initialize the underlying 3GPP channel simulator
        simulator = ChannelSimulator(
            scenario_name=self.config.model_name,
            center_frequency_hz=self.config.center_frequency_hz,
            cluster_model_name=self.config.cluster_model_name,
            use_median_lsps=self.config.use_median_lsps,
            scale_to_scenario=self.config.scale_to_scenario,
            split_subclusters=self.config.split_subclusters,
            c_ds_ns=self.config.c_ds_ns,
        )
        
        # Set positions and velocity
        simulator.set_bs_ue_positions(
            bs_positions=np.array([self.config.bs_position]),
            ue_positions=np.array([self.config.ue_position])
        )
        simulator.set_velocities(np.array([self.config.ue_velocity]))

        # Antenna config must flow through to the 3GPP generator so cluster.h_matrix
        # carries the correct (num_rx, num_tx) shape with per-antenna phase variation.
        # Historical bug: this section was originally skipped on the assumption that
        # MIMO_OTA_Simulator computes its own probe/antenna synthesis. That holds for
        # 'ray'/'strict_pfs'/'pinv' which derive Tx steering locally, but breaks the
        # 'cluster' method which reads g_tx = cluster.h_matrix[0:1, :]. Without this
        # call the default (1, 1) antenna config produced a rank-1 h_matrix and the
        # cluster mode emitted A with all Tx columns identical.
        tx = self.config.tx_antenna
        rx = self.config.rx_antenna
        simulator.set_antenna_config(
            num_tx_rows=tx.num_rows,
            num_tx_cols=tx.num_cols,
            num_rx_rows=rx.num_rows,
            num_rx_cols=rx.num_cols,
            tx_spacing_h=tx.spacing_h,
            tx_spacing_v=tx.spacing_v,
            rx_spacing_h=rx.spacing_h,
            rx_spacing_v=rx.spacing_v,
            tx_array_type=tx.array_type,
            rx_array_type=rx.array_type,
            # Previously dropped: element pattern + array orientation. Without these
            # the OTA standard path silently forced Isotropic / 0° orientation on the
            # target channel regardless of the AntennaArrayConfig. Default values keep
            # legacy behaviour bit-identical.
            tx_element_pattern=tx.element_pattern,
            rx_element_pattern=rx.element_pattern,
            tx_bearing_deg=tx.bearing_deg,
            tx_downtilt_deg=tx.downtilt_deg,
            tx_slant_deg=tx.slant_deg,
            rx_bearing_deg=rx.bearing_deg,
            rx_downtilt_deg=rx.downtilt_deg,
            rx_slant_deg=rx.slant_deg,
        )

        # Generate the channel realization
        results = simulator.run(
            force_condition=self.config.force_condition,
            use_median_lsps=self.config.use_median_lsps,
            logger=None
        )
        
        return UniversalChannelProfile(
            center_frequency_hz=self.config.center_frequency_hz,
            ue_velocity=np.array(self.config.ue_velocity),
            pathloss_db=results['pathloss_db'],
            clusters=results['clusters'],
            lsps=results['lsps'],
            is_los=results['is_los']
        )

class CustomCDLBuilder(ChannelProfileBuilder):
    """Builds a UniversalChannelProfile from a validated CustomCDLProfile (Phase 5 T5.3).

    Caller responsibility:
      profile = CustomCDLProfile.from_file(path)  # or from_dict(d)
      builder = CustomCDLBuilder(profile)
      channel_profile = builder.build()

    This builder is the preferred path for external CDL input (MIMO-First
    micro-service, GUI uploads, scripts). It supersedes ``CustomJSONBuilder``
    which is now a thin deprecation shim.
    """

    def __init__(self, profile: CustomCDLProfile):
        if not isinstance(profile, CustomCDLProfile):
            raise TypeError(
                f"CustomCDLBuilder expects a CustomCDLProfile object; got "
                f"{type(profile).__name__}. Use CustomCDLProfile.from_dict(d) "
                f"or CustomCDLProfile.from_file(path) to construct one."
            )
        self.profile = profile

    def build(self) -> UniversalChannelProfile:
        # Audit batch 2 / A5: num_rays 字段被 schema 校验但合成器硬编码
        # 20 rays/cluster (Table 7.5-3 偏移表只有 20 项) — 非 20 值会被
        # 静默忽略, 必须告警而非无声接受。
        ignored_rays = [i for i, c in enumerate(self.profile.clusters)
                        if c.num_rays != 20]
        if ignored_rays:
            warnings.warn(
                f"clusters {ignored_rays}: num_rays != 20 将被忽略 — 当前实现"
                f"固定 20 rays/cluster (TR 38.901 Table 7.5-3 偏移表为 20 项)。"
                f"可变 ray 数尚未支持。",
                UserWarning,
                stacklevel=2,
            )
        clusters: List[EngineCluster] = []
        for c in self.profile.clusters:
            # Convert xpr from dB to linear (Phase 5 plumb-only, see EngineCluster docstring)
            xpr_linear = (10 ** (c.xpr_db / 10.0)) if c.xpr_db is not None else None
            # initial_phases: stored as numpy (4,) array when provided. Phase 5
            # uses index 0 (Θθθ) in strict_pfs T5.5; indices 1-3 plumbed for Phase 6.
            init_phases = (
                np.asarray(c.initial_phases_rad, dtype=float)
                if c.initial_phases_rad is not None
                else None
            )
            clusters.append(EngineCluster(
                delay=c.delay_s,
                power=c.power_linear,
                aod_phi=c.aod_deg,
                aoa_phi=c.aoa_deg,
                zod_theta=c.zod_deg,
                zoa_theta=c.zoa_deg,
                as_aod=c.as_aod_deg,
                as_aoa=c.as_aoa_deg,
                as_zod=c.as_zod_deg,
                as_zoa=c.as_zoa_deg,
                # NOTE: despite the legacy name `k_n` (kept for back-compat), this
                # field stores per-cluster XPR (cross-polarization ratio), NOT the
                # global Ricean K-factor. The K-factor LOS boost from CustomCDLProfile.
                # k_factor_db is applied separately in T5.4 by redistributing cluster.power.
                # Per Phase 6: canonical XPR storage is xpr_linear (set below from c.xpr_db);
                # k_n is mirrored to xpr_linear (or left default 1.0 when xpr_db is None)
                # for back-compat with downstream code still reading k_n.
                k_n=xpr_linear if xpr_linear is not None else 1.0,
                initial_phases=init_phases,
                xpr_linear=xpr_linear,
            ))

        # Normalize cluster powers to sum to 1.0 (defensive — schema doesn't enforce)
        total_power = sum(c.power for c in clusters)
        if total_power > 0:
            for c in clusters:
                c.power /= total_power

        # T5.4: K-factor LOS boost (TR 38.901 §7.5.6).
        # If LOS scenario AND user provided global k_factor_db, redistribute
        # cluster powers so first cluster (LOS path) carries K/(K+1) and the
        # remaining clusters share 1/(K+1) of total power (preserving their
        # relative shape). K-factor only applies under LOS.
        # Audit F2 (2026-05-19): emit UserWarning when k_factor_db is set but
        # is_los=False — previously we skipped silently, which masked user
        # config errors (e.g. MIMO-First adapter forgetting to flip is_los).
        if self.profile.k_factor_db is not None and not self.profile.is_los:
            warnings.warn(
                f"CustomCDLProfile.k_factor_db={self.profile.k_factor_db} dB is set "
                f"but is_los=False; K-factor LOS boost (TR 38.901 §7.5.6) only applies "
                f"under LOS, so the value will be IGNORED. Either set is_los=True or "
                f"remove k_factor_db. Audit batch 1 / F2.",
                UserWarning,
                stacklevel=2,
            )
        if self.profile.is_los and self.profile.k_factor_db is not None and len(clusters) >= 1:
            k_linear = 10 ** (self.profile.k_factor_db / 10.0)
            p_los_share = k_linear / (k_linear + 1.0)
            p_nlos_share = 1.0 / (k_linear + 1.0)
            clusters[0].power = p_los_share
            if len(clusters) > 1:
                # Scale remaining clusters so they sum to p_nlos_share
                nlos_total = sum(c.power for c in clusters[1:])
                if nlos_total > 0:
                    scale = p_nlos_share / nlos_total
                    for c in clusters[1:]:
                        c.power *= scale

        return UniversalChannelProfile(
            center_frequency_hz=self.profile.center_frequency_hz,
            ue_velocity=np.array(self.profile.ue_velocity_mps),
            pathloss_db=self.profile.pathloss_db,
            clusters=clusters,
            lsps=None,
            is_los=self.profile.is_los,
        )


class CustomJSONBuilder(ChannelProfileBuilder):
    """**DEPRECATED**: use ``CustomCDLBuilder(CustomCDLProfile.from_file(path))``.

    Backward-compat shim that accepts a JSON file path. Schema is the legacy
    v1.0 format (flat ``global_params`` + ``clusters`` with field names like
    ``power_relative_linear``, ``ue_velocity_kph``). For new code use the
    Pydantic-validated path via ``CustomCDLBuilder``.
    """

    def __init__(self, json_filepath: str):
        warnings.warn(
            "CustomJSONBuilder is deprecated and will be removed in a future "
            "release. Use CustomCDLBuilder(CustomCDLProfile.from_file(path)) "
            "instead. See docs/plans/external-cdl-input-design.md §5.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.json_filepath = json_filepath

    def build(self) -> UniversalChannelProfile:
        try:
            with open(self.json_filepath, 'r') as f:
                data = json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to read custom channel profile {self.json_filepath}: {str(e)}")

        # Legacy v1.0 → bridge into the v2.0 schema where possible
        global_params = data.get('global_params', {})
        clusters_data = data.get('clusters', [])

        center_frequency_hz = global_params.get('center_frequency_hz', 2.4e9)
        pathloss_db = global_params.get('pathloss_db', 100.0)
        vel_kph = global_params.get('ue_velocity_kph', [0.0, 0.0, 0.0])
        if not isinstance(vel_kph, (list, tuple)):
            vel_kph = [float(vel_kph), 0.0, 0.0]
        vel_mps = [v * (1000.0 / 3600.0) for v in vel_kph]

        # Map legacy field names to v2.0 cluster spec
        v2_clusters = []
        for c in clusters_data:
            v2_clusters.append({
                'delay_s': c.get('delay_s', 0.0),
                'power_linear': c.get('power_relative_linear', c.get('power_linear', 1.0)),
                'aoa_deg': c.get('aoa_deg', 0.0),
                'aod_deg': c.get('aod_deg', 0.0),
                'zoa_deg': c.get('zoa_deg', 90.0),
                'zod_deg': c.get('zod_deg', 90.0),
                'as_aoa_deg': c.get('as_aoa_deg', 0.0),
                'as_aod_deg': c.get('as_aod_deg', 0.0),
                'as_zoa_deg': c.get('as_zoa_deg', 0.0),
                'as_zod_deg': c.get('as_zod_deg', 0.0),
            })

        profile = CustomCDLProfile.from_dict({
            'center_frequency_hz': center_frequency_hz,
            'pathloss_db': pathloss_db,
            'ue_velocity_mps': vel_mps,
            'is_los': data.get('is_los', False),
            'clusters': v2_clusters,
        })
        return CustomCDLBuilder(profile).build()

def create_builder(config: TargetChannelConfig) -> ChannelProfileBuilder:
    """Factory: pick the right ChannelProfileBuilder for a TargetChannelConfig.

    Precedence (Phase 5+):
      1. input_mode='custom' + custom_profile is set  -> CustomCDLBuilder (preferred)
      2. input_mode='custom' + custom_profile_path is set -> CustomJSONBuilder (deprecated shim)
      3. otherwise -> Standard3GPPBuilder
    """
    if config.input_mode == 'custom':
        if config.custom_profile is not None:
            return CustomCDLBuilder(config.custom_profile)
        if config.custom_profile_path:
            return CustomJSONBuilder(config.custom_profile_path)
        raise ValueError(
            "input_mode='custom' but neither custom_profile nor "
            "custom_profile_path is set."
        )
    return Standard3GPPBuilder(config)
