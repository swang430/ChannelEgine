import numpy as np
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from channel_model_38901.simulator import ChannelSimulator
from channel_model_38901.common.data_models import LargeScaleParameters

def run_statistical_validation(scenario='UMa', condition='NLOS', num_runs=10000):
    """Runs statistical validation for a given scenario and condition."""
    print(f"--- Running Statistical Validation for {scenario} {condition} ---")
    print(f"Number of simulation runs: {num_runs}")

    # --- 1. Run Simulations ---
    simulator = ChannelSimulator(scenario_name=scenario, center_frequency_hz=3.5e9)
    # Use a fixed but representative position for all runs
    simulator.set_bs_ue_positions(np.array([[0, 0, 25]]), np.array([[200, 0, 1.5]]))

    collected_lsps = []
    for _ in tqdm(range(num_runs)):
        # Ensure we get the desired condition by re-running if necessary
        while True:
            # Use a random seed for each run to get independent samples
            np.random.seed(None)
            results = simulator.run()
            is_los = results['is_los']
            if (condition == 'LOS' and is_los) or (condition == 'NLOS' and not is_los):
                collected_lsps.append(results['lsps'])
                break

    # --- 2. Load Standard Target Values ---
    with open(simulator.params.param_file_path, 'r') as f:
        params_json = json.load(f)
    
    lsp_set_name = 'lsp_parameters_los' if condition == 'LOS' else 'lsp_parameters_nlos'
    target_params = params_json['scenarios'][scenario][lsp_set_name]
    target_correlations = target_params['correlations']

    param_order = ['DS', 'ASD', 'ASA', 'ZSA', 'SF', 'K']
    if target_params.get(f'mu_log10_K') == 'N/A':
        param_order.remove('K')

    target_mu = [eval(target_params[f'mu_log10_{p}'], {"__builtins__": {}}, {'log10': np.log10, 'fc_ghz': 3.5}) for p in param_order]
    target_sigma = [eval(target_params[f'sigma_log10_{p}'], {"__builtins__": {}}, {'log10': np.log10, 'fc_ghz': 3.5}) for p in param_order]

    # --- 3. Analyze Collected Data ---
    # Convert list of Pydantic models to a dictionary of numpy arrays
    sim_data = {p: [] for p in param_order}
    for lsp in collected_lsps:
        sim_data['DS'].append(np.log10(lsp.delay_spread_s))
        sim_data['ASD'].append(np.log10(lsp.departure_azimuth_spread_deg))
        sim_data['ASA'].append(np.log10(lsp.arrival_azimuth_spread_deg))
        sim_data['ZSA'].append(np.log10(lsp.arrival_zenith_spread_deg))
        sim_data['SF'].append(lsp.shadow_fade_db)
        if 'K' in param_order:
            sim_data['K'].append(10 * np.log10(lsp.k_factor_linear))

    sim_mu = [np.mean(sim_data[p]) for p in param_order]
    sim_sigma = [np.std(sim_data[p]) for p in param_order]
    sim_corr_matrix = np.corrcoef([sim_data[p] for p in param_order])

    # --- 4. Print Comparison and Conclusion ---
    print("\n--- Statistical Comparison --- (Simulated vs. [Standard])")
    print(f"{ 'Parameter':<10} | { 'Sim Mean':<10} | { 'Std Mean':<10} | { 'Mean Err %':<10} | { 'Sim Std':<10} | { 'Std Std':<10} | { 'Std Err %':<10}")
    print("-" * 80)
    all_passed = True
    for i, p in enumerate(param_order):
        mean_err = abs((sim_mu[i] - target_mu[i]) / target_mu[i]) * 100 if target_mu[i] != 0 else 0
        std_err = abs((sim_sigma[i] - target_sigma[i]) / target_sigma[i]) * 100 if target_sigma[i] != 0 else 0
        mean_passed = mean_err < 5 # 5% tolerance
        std_passed = std_err < 5 # 5% tolerance
        if not (mean_passed and std_passed):
            all_passed = False
        print(f"{p:<10} | {sim_mu[i]:<10.3f} | {target_mu[i]:<10.3f} | {mean_err:<9.2f}% | {sim_sigma[i]:<10.3f} | {target_sigma[i]:<10.3f} | {std_err:<9.2f}%")

    print("\n--- Correlation Matrix Comparison ---")
    print("Simulated Matrix:")
    print(np.round(sim_corr_matrix, 2))
    
    print("\n--- Conclusion ---")
    if all_passed:
        print("\033[92mPASSED: All simulated LSP statistics are within 5% tolerance of the 3GPP standard.\033[0m")
    else:
        print("\033[91mFAILED: One or more simulated LSP statistics deviate from the 3GPP standard by more than 5%.\033[0m")

    # --- 5. Generate Plots ---
    fig, axes = plt.subplots(2, int(np.ceil(len(param_order)/2)), figsize=(15, 8))
    fig.suptitle(f'LSP Distributions for {scenario} {condition} ({num_runs} runs)', fontsize=16)
    for i, p in enumerate(param_order):
        ax = axes.flatten()[i]
        ax.hist(sim_data[p], bins=50, density=True, label='Simulated')
        ax.set_title(f'Distribution of log10({p})' if p not in ['SF', 'K'] else f'Distribution of {p}')
        ax.grid(True)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'{scenario}_{condition}_lsp_distributions.png')
    print(f"\nSaved LSP distribution plots to: {scenario}_{condition}_lsp_distributions.png")

if __name__ == "__main__":
    run_statistical_validation(scenario='UMa', condition='NLOS', num_runs=10000)
