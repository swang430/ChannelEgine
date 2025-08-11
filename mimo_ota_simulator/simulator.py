import numpy as np
from typing import Dict, List, Tuple

from mimo_ota_simulator.data_models import ChannelProfile, ChamberConfig, TargetChannelConfig
from channel_model_38901.common.data_models import EngineCluster
from channel_model_38901.simulator import ChannelSimulator

class MIMO_OTA_Simulator:
    def __init__(self):
        self.channel_engine = None

    def _get_target_channel_profile(self, channel_config: TargetChannelConfig) -> Tuple[ChannelProfile, np.ndarray]:
        self.channel_engine = ChannelSimulator(
            scenario_name=channel_config.model_name, 
            center_frequency_hz=3.5e9
        )
        
        num_tx, num_rx = [int(x) for x in channel_config.mimo_config.split('x')]
        self.channel_engine.config.antenna_config.num_tx_cols = num_tx
        self.channel_engine.config.antenna_config.num_rx_cols = num_rx
        self.channel_engine.config.ue_velocities = np.array([[channel_config.ue_speed_kph * 1000 / 3600, 0, 0]])

        _f, H_normalized, clusters_data, _lsps, _pl, _il = self.channel_engine.run(logger=None)
        
        profile = ChannelProfile(
            name=channel_config.model_name,
            clusters=clusters_data,
            num_tx_antennas=num_tx,
            num_rx_antennas=num_rx
        )
        
        return profile, H_normalized

    def _get_sorted_probes_by_angle(self, chamber_config: ChamberConfig) -> List[Tuple[int, float]]:
        probe_angles = []
        for probe_id, location in chamber_config.probe_locations.items():
            angle_rad = np.arctan2(location[1], location[0])
            probe_angles.append((probe_id, angle_rad))
        return sorted(probe_angles, key=lambda item: item[1])

    def _map_cluster_to_probes(self, cluster: EngineCluster, chamber_config: ChamberConfig) -> Dict[int, float]:
        cluster_aoa_rad = np.deg2rad(cluster.aoa_phi)
        if cluster_aoa_rad < 0:
            cluster_aoa_rad += 2 * np.pi

        sorted_probes = self._get_sorted_probes_by_angle(chamber_config)
        probe_ids = [p[0] for p in sorted_probes]
        angles_rad = np.array([p[1] for p in sorted_probes])

        cluster_aoa_norm = np.arctan2(np.sin(cluster_aoa_rad), np.cos(cluster_aoa_rad))

        idx_right = np.searchsorted(angles_rad, cluster_aoa_norm)

        if idx_right == 0:
            idx_left = len(angles_rad) - 1
        elif idx_right == len(angles_rad):
            idx_right = 0
            idx_left = len(angles_rad) - 1
        else:
            idx_left = idx_right - 1
        
        if np.isclose(angles_rad[idx_right], cluster_aoa_norm):
            return {probe_ids[idx_right]: 1.0}
        if np.isclose(angles_rad[idx_left], cluster_aoa_norm):
            return {probe_ids[idx_left]: 1.0}

        angle_left = angles_rad[idx_left]
        angle_right = angles_rad[idx_right]
        id_left = probe_ids[idx_left]
        id_right = probe_ids[idx_right]

        total_angular_dist = angle_right - angle_left
        if total_angular_dist < 0:
            total_angular_dist += 2 * np.pi

        if total_angular_dist < 1e-9:
            return {id_left: 1.0}

        dist_to_left = cluster_aoa_norm - angle_left
        if dist_to_left < 0:
            dist_to_left += 2 * np.pi

        weight_right = dist_to_left / total_angular_dist
        weight_left = 1.0 - weight_right

        return {id_left: weight_left, id_right: weight_right}

    def run(self, chamber_config: ChamberConfig, channel_config: TargetChannelConfig) -> Dict:
        channel_profile, target_H = self._get_target_channel_profile(channel_config)
        
        Nt = channel_profile.num_tx_antennas
        Nr = int(channel_config.mimo_config.split('x')[1])
        Np = chamber_config.num_probes

        A = np.zeros((Np, Nt), dtype=np.complex128)
        for cluster in channel_profile.clusters:
            probe_weights = self._map_cluster_to_probes(cluster, chamber_config)
            for probe_id, weight in probe_weights.items():
                A[probe_id - 1, 0] += np.sqrt(cluster.power * weight) * np.exp(1j * np.random.uniform(-np.pi, np.pi))

        B = chamber_config.calibration_matrix_B
        if B is None:
            B = np.eye(Np, dtype=np.complex128)[:Nr, :]

        H_effective = B @ A

        return {
            'target_H_matrix': target_H,
            'transmit_mapping_matrix': A,
            'effective_H_matrix': H_effective,
            'probe_signals': {}
        }