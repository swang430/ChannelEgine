import numpy as np
import pickle
import os

from channel_model_38901.simulator import ChannelSimulator
from channel_model_38901.channel_generator import ChannelGenerator
from channel_model_38901.models.spatial_correlation import get_spatial_correlation_matrix
from channel_model_38901.common.geometry import get_antenna_positions

def generate_h_matrix_data(scenario='UMa', condition='NLOS', num_runs=2000):
    """Generates H-matrix data for a URA configuration with Doppler."""
    
    file_id = f'{scenario}_{condition}_URA_4x2_Doppler'
    print(f"--- Generating H-Matrix Data for {file_id} ---")
    print(f"Number of simulation runs: {num_runs}")
    print("Starting simulation loop (this will be silent)...")

    simulator = ChannelSimulator(scenario_name=scenario, center_frequency_hz=3.5e9)
    ant_config = simulator.config.antenna_config
    
    # Configure a 4x2 URA setup
    ant_config.tx_array_type = 'URA'
    ant_config.num_tx_rows = 2
    ant_config.num_tx_cols = 2
    ant_config.rx_array_type = 'URA'
    ant_config.num_rx_rows = 2
    ant_config.num_rx_cols = 1
    
    # Set a non-zero velocity to activate Doppler
    simulator.config.ue_velocities = np.array([[10.0, 0, 0]]) # 10 m/s

    # Re-initialize generator with the new config
    simulator.generator = ChannelGenerator(simulator.config, simulator.params)

    collected_H_normalized = []
    collected_R_tx = []
    collected_R_rx = []

    for i in range(num_runs):
        while True:
            np.random.seed(None)
            # Run with a non-zero time to include Doppler effect
            _f, H_normalized, clusters, lsps, _pl, is_los = simulator.generator.generate_channel_realization(logger=None, time_s=0.1)
            
            if (condition == 'LOS' and is_los) or (condition == 'NLOS' and not is_los):
                collected_H_normalized.append(H_normalized)
                
                mean_aoa = clusters[0]['aoa_phi']; mean_zoa = clusters[0]['zoa_theta']
                mean_aod = clusters[0]['aod_phi']; mean_zod = clusters[0]['zod_theta']

                R_rx_th = get_spatial_correlation_matrix(ant_config.rx_array_type, ant_config.num_rx_rows, ant_config.num_rx_cols, ant_config.rx_spacing_v, ant_config.rx_spacing_h, mean_zoa, mean_aoa, lsps.arrival_zenith_spread_deg, lsps.arrival_azimuth_spread_deg)
                collected_R_rx.append(R_rx_th)

                R_tx_th = get_spatial_correlation_matrix(ant_config.tx_array_type, ant_config.num_tx_rows, ant_config.num_tx_cols, ant_config.tx_spacing_v, ant_config.tx_spacing_h, mean_zod, mean_aod, lsps.departure_zenith_spread_deg, lsps.departure_azimuth_spread_deg)
                collected_R_tx.append(R_tx_th)
                break

    output_file = os.path.join('tests', f'{file_id}_h_matrix_data.pkl')
    with open(output_file, 'wb') as f:
        pickle.dump({
            'h_matrices': collected_H_normalized,
            'r_tx_th_list': collected_R_tx,
            'r_rx_th_list': collected_R_rx
        }, f)
    
    print(f"\nSuccessfully generated and saved {len(collected_H_normalized)} H-matrix samples to {output_file}")

if __name__ == "__main__":
    generate_h_matrix_data()