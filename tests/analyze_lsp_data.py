import numpy as np
import json
import pickle
import matplotlib.pyplot as plt
import os

from channel_model_38901.simulator import ChannelSimulator # Used to get param file path

def analyze_lsp_data(scenario='UMa', condition='NLOS'):
    """Loads raw LSP data and performs statistical validation."""
    input_file = os.path.join('tests', f'{scenario}_{condition}_lsp_data.pkl')
    print(f"--- Analyzing LSP Data from {input_file} ---")

    try:
        with open(input_file, 'rb') as f:
            collected_lsps = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Data file {input_file} not found.")
        print("Please run generate_lsp_data.py first.")
        return

    num_runs = len(collected_lsps)
    print(f"Loaded {num_runs} LSP samples.")

    # ... (analysis logic remains the same) ...
    simulator = ChannelSimulator(scenario_name=scenario, center_frequency_hz=3.5e9)
    with open(simulator.params.param_file_path, 'r') as f:
        params_json = json.load(f)
    lsp_set_name = 'lsp_parameters_los' if condition == 'LOS' else 'lsp_parameters_nlos'
    target_params = params_json['scenarios'][scenario][lsp_set_name]
    param_order = ['DS', 'ASD', 'ASA', 'ZSA', 'SF', 'K']
    if target_params.get(f'mu_log10_K') == 'N/A': param_order.remove('K')
    target_mu = [eval(target_params[f'mu_log10_{p}'], {"__builtins__": {}}, {'log10': np.log10, 'fc_ghz': 3.5}) for p in param_order]
    target_sigma = [eval(target_params[f'sigma_log10_{p}'], {"__builtins__": {}}, {'log10': np.log10, 'fc_ghz': 3.5}) for p in param_order]
    sim_data = {p: [] for p in param_order}
    for lsp in collected_lsps:
        sim_data['DS'].append(np.log10(lsp.delay_spread_s))
        sim_data['ASD'].append(np.log10(lsp.departure_azimuth_spread_deg))
        sim_data['ASA'].append(np.log10(lsp.arrival_azimuth_spread_deg))
        sim_data['ZSA'].append(np.log10(lsp.arrival_zenith_spread_deg))
        sim_data['SF'].append(lsp.shadow_fade_db)
        if 'K' in param_order: sim_data['K'].append(10 * np.log10(lsp.k_factor_linear))
    sim_mu = [np.mean(sim_data[p]) for p in param_order]
    sim_sigma = [np.std(sim_data[p]) for p in param_order]
    all_passed = True
    # ... (printing logic remains the same) ...

    output_plot_path = os.path.join('tests', f'{scenario}_{condition}_lsp_distributions.png')
    plt.savefig(output_plot_path)
    print(f"\nSaved LSP distribution plots to: {output_plot_path}")

if __name__ == "__main__":
    analyze_lsp_data()