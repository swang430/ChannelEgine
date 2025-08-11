import json
import numpy as np
from ..common.data_models import ChannelProfile, EngineCluster

class CDL_Parser:
    """Parses a CDL model definition file into a ChannelProfile object."""

    @staticmethod
    def parse(file_path: str) -> ChannelProfile:
        """
        Parses a JSON file containing CDL parameters.
        """
        with open(file_path, 'r') as f:
            data = json.load(f)

        clusters = []
        for c_data in data['clusters']:
            power_linear = 10**(c_data['power_dB'] / 10)
            cluster = EngineCluster(
                delay=c_data['delay'],
                power=power_linear,
                aod_phi=c_data['aod_phi_deg'],
                aoa_phi=c_data['aoa_phi_deg'],
                zod_theta=c_data['zod_theta_deg'],
                zoa_theta=c_data['zoa_theta_deg']
                # h_matrix will be None by default
            )
            clusters.append(cluster)

        return ChannelProfile(
            name=data['name'],
            clusters=clusters,
            num_tx_antennas=data['num_tx_antennas'],
            num_rx_antennas=data['num_rx_antennas']
        )
