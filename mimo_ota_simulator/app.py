import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mimo_ota_simulator.simulator import MIMO_OTA_Simulator
from mimo_ota_simulator.data_models import ChamberConfig, TargetChannelConfig, AntennaArrayConfig, ProbeAntennaConfig

# --- Page Configuration ---
st.set_page_config(
    page_title="MIMO OTA Simulator V4.4",
    page_icon="🛰️",
    layout="wide"
)

# --- Backend Functions ---
def run_simulation(target_config_dict, chamber_config_dict):
    target_config = TargetChannelConfig.parse_obj(target_config_dict)
    chamber_config = ChamberConfig.parse_obj(chamber_config_dict)
    simulator = MIMO_OTA_Simulator()
    try:
        return simulator.run(chamber_config=chamber_config, channel_config=target_config, synthesis_method=st.session_state.synthesis_method)
    except Exception as e:
        st.error(f"Simulation Failed: {e}")
        return None

# --- UI: Helper function for antenna config ---
def antenna_panel(antenna_type: str):
    key_prefix = f"{antenna_type.lower()}"
    st.write(f"**{antenna_type} Antenna**")
    array_type = st.selectbox('Array Type', ('ULA', 'URA'), key=f'{key_prefix}_type')
    element_pattern = st.selectbox('Element Pattern', ('Isotropic', 'TR38.901'), key=f'{key_prefix}_pattern')
    if array_type == 'ULA':
        num_cols = st.number_input('Number of Elements', 1, 16, 2, key=f'{key_prefix}_ula_cols')
        spacing_h = st.number_input('Element Spacing (λ)', 0.1, 4.0, 0.5, 0.1, key=f'{key_prefix}_ula_spacing')
        return AntennaArrayConfig(array_type='ULA', element_pattern=element_pattern, num_rows=1, num_cols=num_cols, spacing_h=spacing_h)
    else: # URA
        num_rows = st.number_input('Rows', 1, 8, 2, key=f'{key_prefix}_ura_rows')
        num_cols = st.number_input('Columns', 1, 8, 2, key=f'{key_prefix}_ura_cols')
        spacing_h = st.number_input('Horizontal Spacing (λ)', 0.1, 4.0, 0.5, 0.1, key=f'{key_prefix}_ura_spacing_h')
        spacing_v = st.number_input('Vertical Spacing (λ)', 0.1, 4.0, 0.5, 0.1, key=f'{key_prefix}_ura_spacing_v')
        return AntennaArrayConfig(array_type='URA', element_pattern=element_pattern, num_rows=num_rows, num_cols=num_cols, spacing_h=spacing_h, spacing_v=spacing_v)

# --- UI Layout ---
st.title("MIMO OTA Simulator V4.4")

# --- Session State Initialization ---
if 'simulation_results' not in st.session_state:
    st.session_state.simulation_results = None
if 'synthesis_method' not in st.session_state:
    st.session_state.synthesis_method = 'cluster'

with st.sidebar:
    st.header("Simulation Control")
    with st.expander("1. Scenario & Geometry", expanded=True):
        model_name = st.selectbox('Large-Scale Environment', ('UMa', 'UMi-StreetCanyon', 'UMi-OpenArea', 'RMa', 'InH-Office', 'SMa', 'InF'), key='model_name')
        cluster_model_name = st.selectbox('Small-Scale Cluster Model', ('Stochastic', 'CDL-A', 'CDL-B', 'CDL-C', 'CDL-D', 'CDL-E', 'SCME'), key='cluster_model_name')
        center_frequency_hz = st.number_input('Frequency (GHz)', 0.5, 100.0, 3.5, 0.1, key='freq') * 1e9
        ue_velocity_kph = st.slider('UE Speed (km/h)', 0, 120, 3, key='speed')

    with st.expander("2. Chamber Configuration", expanded=True):
        num_probes = st.slider('Number of Probes', 1, 64, 16, 1, key="num_probes")
        radius_m = st.number_input('Probe Ring Radius (m)', 0.5, 5.0, 1.5, 0.1, key='radius')
        dual_polarized = st.checkbox('Dual Polarized Probes', value=True)

    with st.expander("3. Chamber Physical Parameters", expanded=False):
        probe_gain_dBi = st.number_input("Probe Antenna Gain (dBi)", -10.0, 20.0, 0.0, 0.1)
        qz_radius_m = st.number_input("Quiet Zone Radius (m)", 0.1, 2.0, 0.3, 0.05)
        cal_file = st.file_uploader("Upload Calibration Vector (real,imag.csv)", type=['csv'])

    with st.expander("4. Antenna Configuration", expanded=True):
        tx_antenna_config = antenna_panel("Tx")
        st.divider()
        rx_antenna_config = antenna_panel("Rx")

    with st.expander("5. Channel & Run Control", expanded=True):
        st.selectbox('Synthesis Method', ('pinv', 'cluster'), key='synthesis_method')
        force_condition = st.selectbox('LOS/NLOS', ('Default', 'Force LOS', 'Force NLOS'), key='los')
        use_median_lsps = st.checkbox('Use Median LSPs', value=False, help="Bypass random sampling and use the median value for all Large-Scale Parameters.")
        random_seed = st.number_input('Random Seed', value=None, placeholder="Optional", step=1, key='seed')

    if st.button("Run Simulation", type="primary"):
        cal_vector = None
        if cal_file is not None:
            try:
                df = pd.read_csv(cal_file, header=None)
                cal_vector = df[0].values + 1j * df[1].values
            except Exception as e:
                st.error(f"Failed to parse calibration file: {e}")

        chamber_config_params = {
            "num_probes": num_probes,
            "dual_polarized": dual_polarized,
            "distribution": 'ring',
            "radius_m": radius_m,
            "probe_antenna": {"gain_dBi": probe_gain_dBi, "pattern": "Isotropic"},
            "quiet_zone_radius_m": qz_radius_m,
            "calibration_vector": cal_vector
        }
        target_config_params = {
            "model_name": model_name,
            "cluster_model_name": cluster_model_name if cluster_model_name != 'Stochastic' else None,
            "center_frequency_hz": center_frequency_hz,
            "ue_velocity": (ue_velocity_kph * 1000 / 3600, 0, 0),
            "tx_antenna": tx_antenna_config.model_dump(),
            "rx_antenna": rx_antenna_config.model_dump(),
            "force_condition": (None if force_condition == 'Default' else force_condition.split(' ')[-1]),
            "random_seed": random_seed,
            "use_median_lsps": use_median_lsps
        }
        with st.spinner('Running simulation...'):
            st.session_state.simulation_results = run_simulation(target_config_params, chamber_config_params)

# --- Results Display Area ---
if st.session_state.simulation_results:
    results = st.session_state.simulation_results
    st.success('Simulation finished successfully!')
    
    # Unpack all results
    A_matrix = results['transmit_mapping_matrix_A']
    H_target = results['target_H_matrix']
    B_matrix = results['B_matrix']
    clusters = results['clusters']
    probe_tx_power_contributions = results['probe_tx_power_contributions']
    cluster_probe_mapping = results['cluster_probe_mapping']
    num_effective_probes, num_tx_antennas = A_matrix.shape

    st.header("Simulation Results")

    # --- Create the two main plots ---
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Power per Probe & Tx Contribution (Centered)")
        fig, ax = plt.subplots()
        
        if np.any(probe_tx_power_contributions):
            shift = num_effective_probes // 2
            x_coords = np.arange(num_effective_probes)
            original_indices = np.arange(num_effective_probes)
            shifted_indices = np.roll(original_indices, shift)
            
            bottoms = np.zeros(num_effective_probes)
            colors = plt.cm.get_cmap('tab10', num_tx_antennas)

            for tx_idx in range(num_tx_antennas):
                power_data = probe_tx_power_contributions[:, tx_idx]
                rolled_power_data = np.roll(power_data, shift)
                ax.bar(x_coords, rolled_power_data, bottom=bottoms, color=colors(tx_idx), label=f'Tx Ant {tx_idx}')
                bottoms += rolled_power_data

            ax.set_xlabel("Effective Probe ID (Centered)")
            ax.set_ylabel("Power (linear scale)")
            ax.set_title("Power per Probe with Tx Contributions")
            ax.set_xticks(x_coords)
            ax.set_xticklabels(shifted_indices)
            ax.tick_params(axis='x', rotation=90, labelsize='small')
            ax.grid(True, axis='y', linestyle='--', alpha=0.7)
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No power data available.', ha='center', va='center')
        st.pyplot(fig)

        with st.expander("Show Probe Power Values"):
            total_power_per_probe = probe_tx_power_contributions.sum(axis=1)
            power_df = pd.DataFrame({
                'Effective Probe ID': np.arange(num_effective_probes),
                'Total Power (linear)': total_power_per_probe
            })
            st.dataframe(power_df)

    with col2:
        st.subheader("Angular Distribution of Target Clusters")
        probe_angles_rad = np.linspace(0, 2 * np.pi, num_probes, endpoint=False)
        clusters_sorted = sorted(clusters, key=lambda c: c.power, reverse=True)
        cluster_aods_rad = np.deg2rad([c.aod_phi for c in clusters_sorted])
        cluster_powers_db = np.array([c.power for c in clusters_sorted])
        power_min, power_max = (cluster_powers_db.min(), cluster_powers_db.max()) if clusters_sorted else (0,0)
        norm_powers = (cluster_powers_db - power_min) / (power_max - power_min) if (power_max - power_min) > 0 else np.ones_like(cluster_powers_db)
        cmap = plt.get_cmap('plasma')
        cluster_colors = cmap(norm_powers)

        fig = plt.figure(figsize=(8, 6))
        gs = fig.add_gridspec(1, 2, width_ratios=[10, 1])
        ax_polar = fig.add_subplot(gs[0, 0], polar=True)
        ax_cbar = fig.add_subplot(gs[0, 1])

        if dual_polarized:
            v_angles, h_angles = probe_angles_rad - np.deg2rad(2), probe_angles_rad + np.deg2rad(2)
            ax_polar.scatter(v_angles, np.ones(num_probes) * 1.1, s=80, c='blue', marker='o', label='V-Probes')
            ax_polar.scatter(h_angles, np.ones(num_probes) * 1.1, s=80, c='red', marker='x', label='H-Probes')
            for i in range(num_probes):
                ax_polar.text(v_angles[i], 1.2, str(2*i), ha='center', va='center', fontsize=7, color='blue')
                ax_polar.text(h_angles[i], 1.2, str(2*i + 1), ha='center', va='center', fontsize=7, color='red')
        else:
            ax_polar.scatter(probe_angles_rad, np.ones(num_probes) * 1.1, s=100, c='blue', marker='X', label='Probes')
            for i, angle in enumerate(probe_angles_rad):
                ax_polar.text(angle, 1.2, str(i), ha='center', va='center', fontsize=8, color='blue')

        for i, (angle, color) in enumerate(zip(cluster_aods_rad, cluster_colors)):
            ax_polar.plot([angle, angle], [0, 1], color=color, linewidth=2)
        
        ax_polar.set_title("Probe and Target Cluster Angles")
        ax_polar.set_yticklabels([])
        ax_polar.set_theta_zero_location('N')
        ax_polar.set_theta_direction(-1)
        ax_polar.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1), fontsize='small')

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=power_min, vmax=power_max))
        cbar = fig.colorbar(sm, cax=ax_cbar)
        cbar.set_label('Target Cluster Power (dB)', rotation=270, labelpad=15)
        cbar.ax.yaxis.set_ticks(cluster_powers_db, labels=[f"{i+1}" for i in range(len(clusters_sorted))])
        cbar.ax.tick_params(axis='y', length=5, pad=2, labelsize='x-small')
        st.pyplot(fig)

    st.divider()

    # --- Cluster-to-Probe Power Contribution Matrix ---
    st.subheader("Cluster-to-Probe Power Contribution Matrix (dB, Centered)")
    if cluster_probe_mapping:
        # Create a DataFrame for cluster info (Delay, AoD)
        cluster_info_df = pd.DataFrame([
            {'Delay (s)': c.delay, 'AoD (deg)': c.aod_phi}
            for c in clusters
        ])
        cluster_info_df.index.name = "Cluster ID"

        # Create a DataFrame for the mapping
        mapping_df = pd.DataFrame.from_dict(cluster_probe_mapping, orient='index')
        
        # Join them
        full_mapping_df = cluster_info_df.join(mapping_df)

        # Center the probe columns
        probe_columns = [col for col in full_mapping_df.columns if isinstance(col, int)]
        non_probe_columns = [col for col in full_mapping_df.columns if not isinstance(col, int)]
        shift = len(probe_columns) // 2
        shifted_probe_columns = list(np.roll(probe_columns, shift))
        final_df = full_mapping_df[non_probe_columns + shifted_probe_columns]
        
        # Convert power contributions to dB
        with np.errstate(divide='ignore'):
            power_db_df = 10 * np.log10(final_df[shifted_probe_columns])
        
        # Combine info and power dataframes
        display_df = final_df[non_probe_columns].copy()
        for col in shifted_probe_columns:
            display_df[col] = power_db_df[col]

        st.dataframe(display_df.style.background_gradient(cmap='viridis', vmin=-80, vmax=0, axis=1, subset=shifted_probe_columns).format("{:.1f}", subset=shifted_probe_columns).format("{:.3e}", subset=['Delay (s)']).format("{:.2f}", subset=['AoD (deg)']))
    else:
        st.info("No cluster mapping data available for this synthesis method.")

    with st.expander("Show Physical Chamber Matrices"):
        st.write("Physical B-Matrix (Complex Values)")
        st.dataframe(B_matrix)
        st.divider()
        st.write("Transmit Mapping Matrix A (Complex Values)")
        st.dataframe(A_matrix)
        st.write("H_target Matrix (Complex Values)")
        st.dataframe(H_target)
else:
    st.info("Please configure parameters and click 'Run Simulation'.")