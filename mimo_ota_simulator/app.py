import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from mimo_ota_simulator.simulator import MIMO_OTA_Simulator
from mimo_ota_simulator.data_models import ChamberConfig, TargetChannelConfig, AntennaArrayConfig

# --- Page Configuration ---
st.set_page_config(
    page_title="MIMO OTA Simulator V4.0",
    page_icon="📡",
    layout="wide"
)

# --- Backend Functions ---
def run_simulation(target_config_dict, num_probes, radius_m, dual_polarized, synthesis_method):
    target_config = TargetChannelConfig.parse_obj(target_config_dict)
    simulator = MIMO_OTA_Simulator()
    num_rx_antennas = target_config.rx_antenna.num_rows * target_config.rx_antenna.num_cols
    num_effective_probes = num_probes * 2 if dual_polarized else num_probes
    
    B = np.zeros((num_rx_antennas, num_effective_probes))
    for i in range(min(num_rx_antennas, num_effective_probes)):
        B[i, i] = 1
        
    chamber_config = ChamberConfig(
        num_probes=num_probes,
        dual_polarized=dual_polarized,
        distribution='ring',
        radius_m=radius_m,
        calibration_matrix_B=B
    )
    try:
        return simulator.run(chamber_config=chamber_config, channel_config=target_config, synthesis_method=synthesis_method)
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
st.title("MIMO OTA Simulator V4.2")

with st.sidebar:
    st.header("Simulation Control")
    with st.expander("1. Scenario & Geometry", expanded=True):
        model_name = st.selectbox('Large-Scale Environment', ('UMa', 'UMi-StreetCanyon', 'UMi-OpenArea', 'RMa', 'InH-Office', 'SMa', 'InF'), key='model_name')
        cluster_model_name = st.selectbox('Small-Scale Cluster Model', ('Stochastic', 'CDL-A', 'CDL-B', 'CDL-C', 'CDL-D', 'CDL-E', 'SCME'), key='cluster_model_name')
        center_frequency_hz = st.number_input('Frequency (GHz)', 0.5, 100.0, 2.4, 0.1, key='freq') * 1e9
        ue_velocity_kph = st.slider('UE Speed (km/h)', 0, 120, 3, key='speed')

    with st.expander("2. Chamber Configuration", expanded=True):
        num_probes = st.slider('Number of Probes', 1, 64, 16, 1, key="num_probes")
        radius_m = st.number_input('Probe Ring Radius (m)', 0.5, 5.0, 1.5, 0.1, key='radius')
        dual_polarized = st.checkbox('Dual Polarized Probes', value=True)

    with st.expander("3. Antenna Configuration", expanded=True):
        tx_antenna_config = antenna_panel("Tx")
        st.divider()
        rx_antenna_config = antenna_panel("Rx")

    with st.expander("4. Channel & Run Control", expanded=True):
        synthesis_method = st.selectbox('Synthesis Method', ('pinv', 'cluster'), index=1, key='synthesis')
        max_hw_clusters = st.slider('Max Hardware Clusters', 1, 24, 24, 1, key='max_hw_clusters')
        force_condition = st.selectbox('LOS/NLOS', ('Default', 'Force LOS', 'Force NLOS'), key='los')
        use_median_lsps = st.checkbox('Use Median LSPs', value=False, help="Bypass random sampling and use the median value for all Large-Scale Parameters.")
        random_seed = st.number_input('Random Seed', value=None, placeholder="Optional", step=1, key='seed')

    run_button = st.button("Run Simulation", type="primary")

if run_button:
    target_config = TargetChannelConfig(
        model_name=model_name,
        cluster_model_name=cluster_model_name,
        center_frequency_hz=center_frequency_hz,
        ue_velocity=(ue_velocity_kph * 1000 / 3600, 0, 0),
        tx_antenna=tx_antenna_config,
        rx_antenna=rx_antenna_config,
        force_condition=(None if force_condition == 'Default' else force_condition.split(' ')[-1]),
        random_seed=random_seed,
        use_median_lsps=use_median_lsps
    )

    with st.spinner('Running simulation...'):
        results = run_simulation(target_config.model_dump(), num_probes, radius_m, dual_polarized, synthesis_method)
    
    if results:
        st.success('Simulation finished successfully!')
        A_matrix = results['transmit_mapping_matrix_A']
        H_target = results['target_H_matrix']
        clusters = results['clusters']
        
        num_effective_probes = A_matrix.shape[0]

        st.header("Simulation Results")
        st.metric("H_target Shape", f"{H_target.shape[0]} x {H_target.shape[1]} (Rx, Tx)")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("A-Matrix Visualization")
            
            # --- Y-axis Centering Logic ---
            num_rows = A_matrix.shape[0]
            shift = num_rows // 2
            
            # Roll the matrix data for centered view
            A_matrix_view = np.roll(A_matrix, shift=shift, axis=0)
            
            # Create new labels for the shifted y-axis
            original_indices = np.arange(1, num_rows + 1)
            shifted_indices = np.roll(original_indices, shift=shift)
            
            # Determine tick positions and labels
            tick_positions = np.linspace(0, num_rows - 1, num=min(num_rows, 16), dtype=int)
            tick_labels = shifted_indices[tick_positions]

            # --- Plotting ---
            fig, ax = plt.subplots()
            im = ax.imshow(np.abs(A_matrix_view), cmap='viridis', aspect='auto')
            ax.set_title("Transmit Mapping Matrix A (Magnitude)")
            ax.set_xlabel("Transmit Antenna Index (Nt)")
            ax.set_ylabel("Effective Probe Index (Np_eff) - Centered View")
            
            # Set the calculated ticks and labels
            ax.set_yticks(tick_positions)
            ax.set_yticklabels(tick_labels)

            # Force integer ticks on x-axis
            from matplotlib.ticker import MaxNLocator
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            
            fig.colorbar(im, ax=ax, label="Weight")
            st.pyplot(fig)

        with col2:
            st.subheader("Angular Distribution & Power")
            # --- Data Preparation ---
            probe_angles_rad = np.linspace(0, 2 * np.pi, num_probes, endpoint=False)
            clusters_sorted = sorted(clusters, key=lambda c: c.power, reverse=True)
            cluster_aods_rad = np.deg2rad([c.aod_phi for c in clusters_sorted])
            cluster_powers_db = np.array([c.power for c in clusters_sorted])
            
            # Normalize power for coloring, using a colormap
            power_min, power_max = cluster_powers_db.min(), cluster_powers_db.max()
            norm_powers = (cluster_powers_db - power_min) / (power_max - power_min) if (power_max - power_min) > 0 else np.ones_like(cluster_powers_db)
            cmap = plt.get_cmap('plasma')
            cluster_colors = cmap(norm_powers)

            # --- Plotting Setup ---
            fig = plt.figure(figsize=(8, 6))
            gs = fig.add_gridspec(1, 2, width_ratios=[10, 1])
            ax_polar = fig.add_subplot(gs[0, 0], polar=True)
            ax_cbar = fig.add_subplot(gs[0, 1])

            # --- Probe Plotting Logic ---
            if dual_polarized:
                v_angles, h_angles = probe_angles_rad - np.deg2rad(2), probe_angles_rad + np.deg2rad(2)
                ax_polar.scatter(v_angles, np.ones(num_probes) * 1.1, s=80, c='blue', marker='o', label='V-Probes')
                ax_polar.scatter(h_angles, np.ones(num_probes) * 1.1, s=80, c='red', marker='x', label='H-Probes')
                for i in range(num_probes):
                    ax_polar.text(v_angles[i], 1.2, str(2*i + 1), ha='center', va='center', fontsize=7, color='blue')
                    ax_polar.text(h_angles[i], 1.2, str(2*i + 2), ha='center', va='center', fontsize=7, color='red')
            else:
                ax_polar.scatter(probe_angles_rad, np.ones(num_probes) * 1.1, s=100, c='blue', marker='X', label='Probes')
                for i, angle in enumerate(probe_angles_rad):
                    ax_polar.text(angle, 1.2, str(i + 1), ha='center', va='center', fontsize=8, color='blue')

            # --- Cluster Plotting Logic ---
            for i, (angle, color) in enumerate(zip(cluster_aods_rad, cluster_colors)):
                ax_polar.plot([angle, angle], [0, 1], color=color, linewidth=2)
            
            ax_polar.set_title("Probe and Cluster Angles")
            ax_polar.set_yticklabels([])
            ax_polar.set_theta_zero_location('N')
            ax_polar.set_theta_direction(-1)
            ax_polar.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1), fontsize='small')

            # --- Colorbar with Cluster Markers ---
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=power_min, vmax=power_max))
            cbar = fig.colorbar(sm, cax=ax_cbar)
            cbar.set_label('Cluster Power (dB)', rotation=270, labelpad=15)
            
            # Add markers to the colorbar
            cbar.ax.yaxis.set_ticks(cluster_powers_db, labels=[f"{i+1}" for i in range(len(clusters_sorted))])
            cbar.ax.tick_params(axis='y', length=5, pad=2, labelsize='x-small')

            st.pyplot(fig)

        with st.expander("Show Raw Matrix Data"):
            st.write("A-Matrix (Magnitude)")
            st.dataframe(np.abs(A_matrix))
            st.write("A-Matrix (Phase)")
            st.dataframe(np.angle(A_matrix))
            st.write("H_target Matrix")
            st.dataframe(H_target)
else:
    st.info("Please configure parameters and click 'Run Simulation'.")