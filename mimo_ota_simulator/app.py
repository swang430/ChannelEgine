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
def run_simulation(target_config_dict, num_probes, radius_m):
    target_config = TargetChannelConfig.parse_obj(target_config_dict)
    simulator = MIMO_OTA_Simulator()
    num_rx_antennas = target_config.rx_antenna.num_rows * target_config.rx_antenna.num_cols
    B = np.zeros((num_rx_antennas, num_probes))
    for i in range(min(num_rx_antennas, num_probes)):
        B[i, i] = 1
    chamber_config = ChamberConfig(
        num_probes=num_probes,
        distribution='ring',
        radius_m=radius_m,
        calibration_matrix_B=B
    )
    try:
        return simulator.run(chamber_config=chamber_config, channel_config=target_config)
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
st.title("MIMO OTA Simulator V4.0 - Final")

with st.sidebar:
    st.header("Simulation Control")
    with st.expander("1. Scenario & Geometry", expanded=True):
        model_name = st.selectbox('Channel Model', ('UMa', 'UMi-StreetCanyon', 'UMi-OpenArea', 'RMa', 'InH-Office', 'SMa', 'InF'), key='model_name')
        center_frequency_hz = st.number_input('Frequency (GHz)', 0.5, 100.0, 2.4, 0.1, key='freq') * 1e9
        ue_velocity_kph = st.slider('UE Speed (km/h)', 0, 120, 3, key='speed')

    with st.expander("2. Chamber Configuration", expanded=True):
        num_probes = st.slider('Number of Probes', 1, 64, 16, 1, key="num_probes")
        radius_m = st.number_input('Probe Ring Radius (m)', 0.5, 5.0, 1.5, 0.1, key='radius')

    with st.expander("3. Antenna Configuration", expanded=True):
        tx_antenna_config = antenna_panel("Tx")
        st.divider()
        rx_antenna_config = antenna_panel("Rx")

    with st.expander("4. Channel & Run Control", expanded=True):
        force_condition = st.selectbox('LOS/NLOS', ('Default', 'Force LOS', 'Force NLOS'), key='los')
        random_seed = st.number_input('Random Seed', value=None, placeholder="Optional", step=1, key='seed')

    run_button = st.button("Run Simulation", type="primary")

if run_button:
    target_config = TargetChannelConfig(
        model_name=model_name,
        center_frequency_hz=center_frequency_hz,
        ue_velocity=(ue_velocity_kph * 1000 / 3600, 0, 0),
        tx_antenna=tx_antenna_config,
        rx_antenna=rx_antenna_config,
        force_condition=(None if force_condition == 'Default' else force_condition.split(' ')[-1]),
        random_seed=random_seed
    )

    with st.spinner('Running simulation...'):
        results = run_simulation(target_config.dict(), num_probes, radius_m)
    
    if results:
        st.success('Simulation finished successfully!')
        A_matrix = results['transmit_mapping_matrix_A']
        H_target = results['target_H_matrix']
        clusters = results['clusters']
        
        # CORRECT WAY to get the number of probes for plotting
        num_probes_actual = A_matrix.shape[0]

        st.header("Simulation Results")
        st.metric("H_target Shape", f"{H_target.shape[0]} x {H_target.shape[1]} (Rx, Tx)")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("A-Matrix Visualization")
            fig, ax = plt.subplots()
            im = ax.imshow(np.abs(A_matrix), cmap='viridis', aspect='auto')
            ax.set_title("Transmit Mapping Matrix A (Magnitude)")
            ax.set_xlabel("Transmit Antenna Index (Nt)")
            ax.set_ylabel("Probe Index (Np)")
            fig.colorbar(im, ax=ax, label="Weight")
            st.pyplot(fig)

        with col2:
            st.subheader("Angular Distribution")
            probe_angles_rad = np.linspace(0, 2 * np.pi, num_probes_actual, endpoint=False)
            cluster_aods_rad = np.deg2rad([c['aod_phi'] for c in clusters])
            cluster_powers_lin = [c['power'] for c in clusters]
            norm_powers = np.array(cluster_powers_lin) / np.max(cluster_powers_lin)

            fig_polar = plt.figure()
            ax_polar = fig_polar.add_subplot(111, polar=True)
            ax_polar.scatter(probe_angles_rad, np.ones(num_probes_actual) * 1.1, s=100, c='blue', marker='X', label='Probes')
            for i, (angle, power) in enumerate(zip(cluster_aods_rad, norm_powers)):
                ax_polar.plot([angle, angle], [0, 1], color=plt.cm.autumn(power), linewidth=2, label=f'Cluster {i+1}' if i < 5 else "")
            ax_polar.set_title("Probe and Cluster Angles")
            ax_polar.set_yticklabels([])
            ax_polar.set_theta_zero_location('N')
            ax_polar.set_theta_direction(-1)
            st.pyplot(fig_polar)

        with st.expander("Show Raw Matrix Data"):
            st.write("A-Matrix")
            st.dataframe(A_matrix)
            st.write("H_target Matrix")
            st.dataframe(H_target)
else:
    st.info("Please configure parameters and click 'Run Simulation'.")