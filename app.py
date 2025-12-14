
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mimo_ota_simulator.simulator import MIMO_OTA_Simulator
from mimo_ota_simulator.data_models import ChamberConfig, TargetChannelConfig, AntennaArrayConfig, ProbeAntennaConfig, PropsimExportConfig, CalibrationConfig, CalibrationEntry
from mimo_ota_simulator.exporters import PropsimASCIIExporter

# --- Page Configuration ---
st.set_page_config(
    page_title="MIMO OTA 仿真器 V4.4",
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
        st.error(f"仿真失败: {e}")
        return None

# --- UI: Helper function for antenna config ---
def antenna_panel(antenna_type: str):
    key_prefix = f"{antenna_type.lower()}"
    label_map = {'Tx': '发射端 (Tx)', 'Rx': '接收端 (Rx)'}
    st.write(f"**{label_map.get(antenna_type, antenna_type)} 天线配置**")
    
    array_type = st.selectbox('阵列类型', ('ULA', 'URA'), key=f'{key_prefix}_type')
    element_pattern = st.selectbox('单元方向图', ('Isotropic', 'TR38.901'), key=f'{key_prefix}_pattern')
    
    if array_type == 'ULA':
        num_cols = st.number_input('单元数量 (Elements)', 1, 16, 2, key=f'{key_prefix}_ula_cols')
        spacing_h = st.number_input('单元间距 (λ)', 0.1, 4.0, 0.5, 0.1, key=f'{key_prefix}_ula_spacing')
        return AntennaArrayConfig(array_type='ULA', element_pattern=element_pattern, num_rows=1, num_cols=num_cols, spacing_h=spacing_h)
    else: # URA
        num_rows = st.number_input('行数 (Rows)', 1, 8, 2, key=f'{key_prefix}_ura_rows')
        num_cols = st.number_input('列数 (Columns)', 1, 8, 2, key=f'{key_prefix}_ura_cols')
        spacing_h = st.number_input('水平间距 (λ)', 0.1, 4.0, 0.5, 0.1, key=f'{key_prefix}_ura_spacing_h')
        spacing_v = st.number_input('垂直间距 (λ)', 0.1, 4.0, 0.5, 0.1, key=f'{key_prefix}_ura_spacing_v')
        return AntennaArrayConfig(array_type='URA', element_pattern=element_pattern, num_rows=num_rows, num_cols=num_cols, spacing_h=spacing_h, spacing_v=spacing_v)

# --- UI Layout ---
st.title("MIMO OTA 仿真器 V4.4")

# --- Session State Initialization ---
if 'simulation_results' not in st.session_state:
    st.session_state.simulation_results = None
if 'synthesis_method' not in st.session_state:
    st.session_state.synthesis_method = 'cluster'

with st.sidebar:
    st.header("控制面板")
    with st.expander("1. 场景与几何 (Scenario)", expanded=False):
        model_name = st.selectbox('大尺度环境', ('UMa', 'UMi-StreetCanyon', 'UMi-OpenArea', 'RMa', 'InH-Office', 'SMa', 'InF'), key='model_name')
        cluster_model_name = st.selectbox('小尺度簇模型', ('Stochastic', 'CDL-A', 'CDL-B', 'CDL-C', 'CDL-D', 'CDL-E', 'SCME'), key='cluster_model_name')
        center_frequency_hz = st.number_input('中心频率 (GHz)', 0.5, 100.0, 3.5, 0.1, key='freq') * 1e9
        ue_velocity_kph = st.slider('UE 移动速度 (km/h)', 0, 120, 3, key='speed')

    with st.expander("2. 暗室配置 (Chamber)", expanded=False):
        num_probes = st.slider('探头数量', 1, 64, 16, 1, key="num_probes")
        radius_m = st.number_input('探头环半径 (m)', 0.5, 30.0, 1.5, 0.5, key='radius')
        dual_polarized = st.checkbox('双极化探头', value=True)

        # 远场条件提示
        wavelength = 3e8 / center_frequency_hz
        st.caption(f"💡 远场条件参考：R > 2D²/λ (当前波长: {wavelength*100:.2f} cm)")
        st.caption(f"   例如：DUT尺寸0.5m需要>5.8m，1.0m需要>23m，2.0m需要>92m")

    with st.expander("3. 物理参数与校准", expanded=False):
        probe_gain_dBi = st.number_input("探头天线增益 (dBi)", -10.0, 20.0, 0.0, 0.1)
        qz_radius_m = st.number_input("静区半径 (m)", 0.1, 2.0, 0.3, 0.05)
        cal_file = st.file_uploader("上传校准文件 (.csv)", type=['csv'], help="包含 Port, Loss, Phase, Gain 等列")

    with st.expander("4. 天线配置 (Antenna)", expanded=False):
        tx_antenna_config = antenna_panel("Tx")
        st.divider()
        rx_antenna_config = antenna_panel("Rx")

    with st.expander("5. 运行控制 (Run Control)", expanded=False):
        st.selectbox('合成方法 (Synthesis Method)', ('pinv', 'cluster', 'ray'), key='synthesis_method', help="Ray: 射线级(推荐); Cluster: 簇级; Pinv: 伪逆")
        
        run_mode = st.radio("运行模式", ('单次快照 (Single Snapshot)', '统计验证 (Statistical Validation)'))
        
        force_condition = st.selectbox('LOS/NLOS 条件', ('默认 (Default)', '强制 LOS', '强制 NLOS'), key='los')
        use_median_lsps = st.checkbox('使用 LSP 中值', value=False, help="跳过随机采样，使用大尺度参数的中位值。")
        probe_phase_jitter = st.checkbox('模拟探头相位抖动', value=False, help="为探头添加随机相位噪声 (+/- 10度) 以模拟硬件误差。")
        
        num_drops = 1
        if run_mode == '统计验证 (Statistical Validation)':
            num_drops = st.slider('Drop 次数', 10, 200, 50, 10)
            random_seed = None # Seed must vary
            st.info("统计验证模式下，随机种子将自动变化。")
        else:
            random_seed = st.number_input('随机种子 (Seed)', value=None, placeholder="留空则随机", step=1, key='seed')

    with st.expander("6. 硬件导出 (Keysight F64)", expanded=False):
        st.write("生成 Propsim F64 驱动文件 (.asc)")
        
        # 1. Export Config
        f64_mode = st.selectbox("幅度控制模式", ["A (固定输出+外部衰减)", "B (内部数字缩放)"], key="f64_mode", help="Mode A 动态范围更大，推荐使用。")
        f64_duration = st.number_input("时长 (s)", 0.001, 10.0, 0.005, 0.001, key="f64_dur", help="静态信道建议使用 5ms 以减小文件体积。")
        f64_sample_rate = st.number_input("采样率 (Hz)", 1e6, 1000e6, 100e6, 10e6, format="%.2e", key="f64_sr")
        
        # 2. Calibration Upload (Shown again here for convenience or linked?)
        # Since it's in Expander 3, maybe just remind user?
        # But user might look here. Let's rely on Expander 3 input.
        if cal_file is not None:
            st.success("已加载校准文件。")
        else:
            st.info("未加载校准文件，将使用理想参数。")
        
        # 3. Generate Button
        if st.button("生成 F64 文件"):
            if st.session_state.simulation_results and st.session_state.simulation_results.get('mode') == 'single':
                # We need raw weights. Did we save them?
                # Check app.py logic...
                # Current app.py saves: A_matrix = results['transmit_mapping_matrix_A']
                # But st.session_state.simulation_results is the raw return dict from simulator.run()
                # So 'transmit_mapping_matrix_A' IS available!
                
                try:
                    mode_char = f64_mode.split()[0]
                    p_config = PropsimExportConfig(
                        mode=mode_char,
                        duration_s=f64_duration,
                        sample_rate_hz=f64_sample_rate,
                        filename="propsim_channel.asc" # Temp name
                    )
                    
                    # Parse Cal File if present
                    cal_config = None
                    if cal_file is not None:
                        import csv
                        import io
                        stringio = io.StringIO(cal_file.getvalue().decode("utf-8"))
                        reader = csv.DictReader(stringio)
                        entries = []
                        for row in reader:
                            row_lower = {k.lower().strip(): v for k, v in row.items()}
                            def get_float(keys, default=0.0):
                                for k in keys:
                                    if k in row_lower: return float(row_lower[k])
                                return default
                            
                            p_id = int(get_float(['port_id', 'port', 'id']))
                            loss = get_float(['cable_loss_db', 'loss', 'loss_db'])
                            phase = get_float(['cable_phase_deg', 'phase', 'phase_deg'])
                            gain = get_float(['probe_gain_dbi', 'gain', 'gain_dbi'])
                            entries.append(CalibrationEntry(port_id=p_id, cable_loss_db=loss, cable_phase_deg=phase, probe_gain_dbi=gain))
                        if entries:
                            cal_config = CalibrationConfig(entries=entries)

                    exporter = PropsimASCIIExporter(p_config, calibration_config=cal_config)
                    
                    # Run Export (Generate file in temp)
                    out_path = exporter.export(st.session_state.simulation_results, output_path="propsim_output.asc")
                    
                    # Read back for download
                    with open(out_path, "r") as f:
                        file_content = f.read()
                    
                    st.download_button(
                        label="下载 .asc 文件",
                        data=file_content,
                        file_name=f"propsim_mode{mode_char}_{f64_duration}s.asc",
                        mime="text/plain"
                    )
                    st.success("文件生成成功！请点击上方按钮下载。")
                    
                except Exception as e:
                    st.error(f"导出失败: {e}")
            else:
                st.warning("请先运行 '单次快照 (Single Snapshot)' 模式以生成信道数据。")

    if st.button("开始仿真", type="primary"):
        cal_vector = None
        if cal_file is not None:
            try:
                # Reset pointer for re-reading if needed, though streamlit usually handles it.
                # Actually, parsing cal file for 'calibration_vector' (simulator input) vs 'CalibrationConfig' (exporter input)
                # Simulator input 'calibration_vector' is a simple complex array.
                # Exporter input 'CalibrationConfig' is rich struct.
                # We should parse once.
                cal_file.seek(0)
                df = pd.read_csv(cal_file, header=None)
                # Assuming simple format: real, imag for simulator vector?
                # Wait, the prompt implies "Port, Loss, Phase..." CSV format for the Exporter.
                # But 'calibration_vector' in ChamberConfig usually expects complex numbers for B-matrix synthesis.
                # If we use the rich CSV, we need to convert it to complex vector for ChamberConfig.
                # Let's assume user uploads the RICH CSV now.
                
                # Logic: We need to parse rich CSV into complex vector for ChamberConfig
                # H_sys = sqrt(gain_linear * loss_linear) * exp(j * phase)
                # ChamberConfig.calibration_vector expects Pre-distortion? Or System Response?
                # Usually Simulator applies B * W. If we want W to correct for Sys, 
                # Simulator B matrix generation might include probe gain.
                # Actually, let's keep it simple: ChamberConfig.calibration_vector is for B-matrix CORRECTION inside simulator?
                # Or we just pass None to simulator and handle everything in Exporter?
                # YES. Exporter handles the output link. Simulator handles the chamber.
                # Let's ignore cal_file for ChamberConfig (set to None) to avoid double correction or confusion.
                pass
            except Exception as e:
                st.error(f"校准文件解析失败: {e}")

        chamber_config_params = {
            "num_probes": num_probes,
            "dual_polarized": dual_polarized,
            "distribution": 'ring',
            "radius_m": radius_m,
            "probe_antenna": {"gain_dBi": probe_gain_dBi, "pattern": "Isotropic"},
            "quiet_zone_radius_m": qz_radius_m,
            "calibration_vector": None # Handle calibration in Export stage
        }
        target_config_params = {
            "model_name": model_name,
            "cluster_model_name": cluster_model_name if cluster_model_name != 'Stochastic' else None,
            "center_frequency_hz": center_frequency_hz,
            "ue_velocity": (ue_velocity_kph * 1000 / 3600, 0, 0),
            "tx_antenna": tx_antenna_config.model_dump(),
            "rx_antenna": rx_antenna_config.model_dump(),
            "force_condition": (None if force_condition == '默认 (Default)' else force_condition.split(' ')[-1]),
            "random_seed": random_seed,
            "use_median_lsps": use_median_lsps
        }
        
        target_config = TargetChannelConfig.parse_obj(target_config_params)
        chamber_config = ChamberConfig.parse_obj(chamber_config_params)
        simulator = MIMO_OTA_Simulator()

        with st.spinner(f'正在运行仿真 ({run_mode})...'):
            if run_mode == '单次快照 (Single Snapshot)':
                st.session_state.simulation_results = simulator.run(
                    chamber_config=chamber_config, 
                    channel_config=target_config, 
                    synthesis_method=st.session_state.synthesis_method,
                    probe_phase_jitter=probe_phase_jitter
                )
                st.session_state.simulation_results['mode'] = 'single'
            else:
                # Statistical Validation Loop
                num_rx = rx_antenna_config.num_rows * rx_antenna_config.num_cols
                R_sum = np.zeros((num_rx, num_rx), dtype=complex)
                progress_bar = st.progress(0)
                
                base_seed = np.random.randint(0, 10000)
                
                for i in range(num_drops):
                    # Vary seed
                    current_target = target_config.model_copy()
                    current_target.random_seed = base_seed + i
                    
                    res = simulator.run(chamber_config, current_target, st.session_state.synthesis_method, probe_phase_jitter=probe_phase_jitter)
                    H = res['effective_H_matrix']
                    R_sum += H @ H.conj().T
                    progress_bar.progress((i + 1) / num_drops)
                
                R_mean = R_sum / num_drops
                
                # Calculate Correlation Coeffs (Ref Ant 0)
                # rho_0i = |R_0i| / sqrt(R_00 * R_ii)
                correlations = []
                r00 = np.abs(R_mean[0,0])
                for i in range(num_rx):
                    rii = np.abs(R_mean[i,i])
                    if r00 > 0 and rii > 0:
                        rho = np.abs(R_mean[0,i]) / np.sqrt(r00 * rii)
                    else:
                        rho = 0.0
                    correlations.append(rho)
                
                st.session_state.simulation_results = {
                    'mode': 'validation',
                    'spatial_correlation': correlations,
                    'num_drops': num_drops,
                    'rx_config': target_config.rx_antenna.model_dump(),
                    'R_mean': R_mean
                }

# --- Results Display Area ---
if st.session_state.simulation_results:
    results = st.session_state.simulation_results
    st.success('仿真完成!')
    
    if results.get('mode') == 'validation':
        st.header("统计验证结果 (Statistical Validation)")
        corr = results['spatial_correlation']
        drops = results['num_drops']
        rx_config_debug = results.get('rx_config', {})
        R_mean_debug = results.get('R_mean')
        
        st.subheader(f"接收端空间相关性 (基准: 天线0, N={drops} Drops)")
        
        # --- DEBUG INFO ---
        with st.expander("调试信息: 验证详情", expanded=False):
            st.write("**Rx 天线配置:**", rx_config_debug)
            if R_mean_debug is not None:
                st.write("**平均协方差矩阵幅度 (|R_mean|):**")
                st.dataframe(pd.DataFrame(np.abs(R_mean_debug)))
        # ------------------

        fig_corr, ax_corr = plt.subplots()
        ax_corr.plot(corr, marker='o', linestyle='-', linewidth=2, color='green')
        ax_corr.set_title(f"Rx Spatial Correlation (Ref: Ant 0, N={drops})")
        ax_corr.set_xlabel("Rx Antenna Index")
        ax_corr.set_ylabel("Correlation Coefficient")
        ax_corr.set_ylim(0, 1.1)
        ax_corr.grid(True)
        # Add theoretical target line (0 for ideal uncorrelated)
        ax_corr.axhline(0, color='gray', linestyle='--', alpha=0.5, label='Ideal Uncorrelated')
        ax_corr.legend()
        
        st.pyplot(fig_corr)
        
        avg_cross_corr = np.mean(corr[1:]) if len(corr) > 1 else 0.0
        st.metric("平均互相关性 (Avg Cross-Correlation)", f"{avg_cross_corr:.4f}")
        
        if avg_cross_corr < 0.2:
            st.success("✅ 极佳的去相关性！暗室性能良好。")
        elif avg_cross_corr < 0.5:
            st.warning("⚠️ 相关性中等。考虑增加探头数量或天线间距。")
        else:
            st.error("❌ 相关性过高。请检查探头配置或模型参数。")
            
    else:
        # --- Single Snapshot Mode Display ---
        # Unpack all results
        A_matrix = results['transmit_mapping_matrix_A']
        H_target = results['target_H_matrix']
        B_matrix = results['B_matrix']
        clusters = results['clusters']
        probe_tx_power_contributions = results['probe_tx_power_contributions']
        cluster_probe_mapping = results['cluster_probe_mapping']
        lsps = results.get('lsps') # Safely get LSPs
        num_effective_probes, num_tx_antennas = A_matrix.shape

        st.header("仿真结果 (单次快照)")
        
        # --- Show LSPs ---
        if lsps:
            with st.expander("大尺度参数 (LSP)", expanded=False):
                # Convert Pydantic model to dict for display
                lsp_dict = lsps.model_dump()
                
                # Format numbers for better readability
                formatted_lsps = {}
                units = {
                    'delay_spread_s': 's',
                    'departure_azimuth_spread_deg': 'deg',
                    'arrival_azimuth_spread_deg': 'deg',
                    'departure_zenith_spread_deg': 'deg',
                    'arrival_zenith_spread_deg': 'deg',
                    'shadow_fade_db': 'dB',
                    'k_factor_linear': '(linear)'
                }
                
                for k, v in lsp_dict.items():
                    unit = units.get(k, '')
                    if isinstance(v, float):
                        if 'delay' in k:
                            formatted_lsps[k] = f"{v:.2e} {unit}"
                        else:
                            formatted_lsps[k] = f"{v:.2f} {unit}"
                    else:
                        formatted_lsps[k] = f"{v} {unit}"
                
                st.json(formatted_lsps)

        # --- Show SSPs (Clusters) ---
        if clusters:
            with st.expander("小尺度参数 (簇/Clusters)", expanded=False):
                cluster_data = []
                for i, c in enumerate(clusters):
                    cluster_data.append({
                        "Cluster ID": i,
                        "Delay (ns)": c.delay * 1e9,
                        "Power (Linear)": c.power,
                        "Power (dB)": 10 * np.log10(c.power) if c.power > 0 else -100,
                        "AoD (deg)": c.aod_phi,
                        "AoA (deg)": c.aoa_phi,
                        "ZoD (deg)": c.zod_theta,
                        "ZoA (deg)": c.zoa_theta
                    })
                st.dataframe(pd.DataFrame(cluster_data))

        # --- Spatial Correlation Validation (Single Snapshot) ---
        st.subheader("空间相关性验证")
        with st.expander("瞬时 Rx 空间相关性 (仅快照)", expanded=False):
            # Calculate R = H * H'
            # H is (N_rx, N_tx)
            # We want Rx correlation, so we average over Tx streams or just take H*H' which sums over Tx.
            # R_rx = H @ H.H
            
            effective_H = results['effective_H_matrix']
            R_inst = effective_H @ effective_H.conj().T
            
            # Normalize so diagonal is 1 (or max is 1)
            diag = np.diag(R_inst)
            # Avoid div by zero
            norm_factor = np.sqrt(np.outer(diag, diag))
            R_norm = np.abs(R_inst / (norm_factor + 1e-12))
            
            # Plot correlation of Ant 0 with others
            if R_norm.shape[0] > 0:
                corr_row = R_norm[0, :]
                
                fig_corr, ax_corr = plt.subplots()
                ax_corr.plot(corr_row, marker='o', linestyle='-')
                ax_corr.set_title("Spatial Correlation (Ref: Ant 0)")
                ax_corr.set_xlabel("Rx Antenna Index")
                ax_corr.set_ylabel("Correlation Magnitude")
                ax_corr.set_ylim(0, 1.1)
                ax_corr.grid(True)
                st.pyplot(fig_corr)
                
                st.caption("注意: 这是单次生成快照的瞬时相关性。要获得具有物理意义的统计结果，请使用'统计验证'模式。")

        # --- Create the two main plots ---
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("探头功率分布与 Tx 贡献")
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

            with st.expander("显示详细探头功率数值", expanded=False):
                total_power_per_probe = probe_tx_power_contributions.sum(axis=1)
                power_df = pd.DataFrame({
                    'Effective Probe ID': np.arange(num_effective_probes),
                    'Total Power (linear)': total_power_per_probe
                })
                st.dataframe(power_df)

        with col2:
            st.subheader("目标簇角度分布 (极坐标)")
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

        # --- Cluster-to-Probe Power Contribution Matrix ---
        with st.expander("簇-探头功率映射矩阵 (dB, Centered)", expanded=False):
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
                st.info("该合成方法暂无簇映射数据。")

        with st.expander("显示物理暗室矩阵 (B & A Matrix)", expanded=False):
            st.write("Physical B-Matrix (Complex Values)")
            st.dataframe(B_matrix)
            st.divider()
            st.write("Transmit Mapping Matrix A (Complex Values)")
            st.dataframe(A_matrix)
            st.write("H_target Matrix (Complex Values)")
            st.dataframe(H_target)
