import numpy as np
import pickle
import matplotlib.pyplot as plt
import os

def analyze_h_matrix_data(scenario='UMa', condition='NLOS'):
    """Loads raw H-matrix data and validates spatial correlation."""
    file_id = f'{scenario}_{condition}_URA_4x2_Doppler'
    input_file = os.path.join('tests', f'{file_id}_h_matrix_data.pkl')
    print(f"--- Analyzing H-Matrix Data from {input_file} ---")

    try:
        with open(input_file, 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Data file {input_file} not found.")
        print("Please run generate_h_matrix_data.py first.")
        return

    collected_H = data['h_matrices']
    R_tx_th_list = data['r_tx_th_list']
    R_rx_th_list = data['r_rx_th_list']
    num_runs = len(collected_H)
    print(f"Loaded {num_runs} H-matrix samples.")

    # --- Calculate Average Theoretical and Empirical Correlation ---
    R_tx_th_avg = np.mean(R_tx_th_list, axis=0)
    R_rx_th_avg = np.mean(R_rx_th_list, axis=0)

    num_rx, num_tx = R_rx_th_avg.shape[0], R_tx_th_avg.shape[0]
    R_tx_emp = np.zeros((num_tx, num_tx), dtype=np.complex128)
    R_rx_emp = np.zeros((num_rx, num_rx), dtype=np.complex128)

    for H in collected_H:
        R_tx_emp += (H.conj().T @ H)
        R_rx_emp += (H @ H.conj().T)

    R_tx_emp /= (num_runs * num_rx)
    R_rx_emp /= (num_runs * num_tx)

    # --- Compare and Conclude ---
    tx_diff = np.linalg.norm(R_tx_th_avg - R_tx_emp) / np.linalg.norm(R_tx_th_avg)
    rx_diff = np.linalg.norm(R_rx_th_avg - R_rx_emp) / np.linalg.norm(R_rx_th_avg)
    
    print("\n--- Spatial Correlation Validation Result ---")
    print(f"Tx Correlation Matrix Error (Frobenius Norm): {tx_diff:.4f}")
    print(f"Rx Correlation Matrix Error (Frobenius Norm): {rx_diff:.4f}")

    tolerance = 0.15 # 15% tolerance
    tx_passed = tx_diff < tolerance
    rx_passed = rx_diff < tolerance

    if tx_passed and rx_passed:
        print("\033[92mPASSED: Empirical spatial correlation matches theoretical model within tolerance.\033[0m")
    else:
        print("\033[91mFAILED: Empirical spatial correlation deviates from theoretical model.\033[0m")

    # --- Visualization ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Spatial Correlation Validation for {file_id}', fontsize=16)
    
    im_tx_th = axes[0, 0].imshow(np.abs(R_tx_th_avg), cmap='viridis'); fig.colorbar(im_tx_th, ax=axes[0, 0])
    axes[0, 0].set_title('Average Theoretical Tx Correlation')
    im_tx_emp = axes[0, 1].imshow(np.abs(R_tx_emp), cmap='viridis'); fig.colorbar(im_tx_emp, ax=axes[0, 1])
    axes[0, 1].set_title('Empirical Tx Correlation')

    im_rx_th = axes[1, 0].imshow(np.abs(R_rx_th_avg), cmap='viridis'); fig.colorbar(im_rx_th, ax=axes[1, 0])
    axes[1, 0].set_title('Average Theoretical Rx Correlation')
    im_rx_emp = axes[1, 1].imshow(np.abs(R_rx_emp), cmap='viridis'); fig.colorbar(im_rx_emp, ax=axes[1, 1])
    axes[1, 1].set_title('Empirical Rx Correlation')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_plot_path = os.path.join('tests', f'{file_id}_correlation.png')
    plt.savefig(output_plot_path)
    print(f"\nSaved correlation matrix plots to: {output_plot_path}")

if __name__ == "__main__":
    analyze_h_matrix_data()
