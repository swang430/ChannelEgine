
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import csv
import io

from mimo_ota_simulator.simulator import MIMO_OTA_Simulator
from mimo_ota_simulator.data_models import ChamberConfig, TargetChannelConfig, AntennaArrayConfig, ProbeAntennaConfig, PropsimExportConfig, CalibrationConfig, CalibrationEntry
from mimo_ota_simulator.cdl_schema import CustomCDLProfile
from mimo_ota_simulator.exporters import PropsimASCIIExporter
import json
from pydantic import ValidationError

# --- Page Configuration ---
st.set_page_config(
    page_title="MIMO OTA 仿真器 V4.4",
    page_icon="🛰️",
    layout="wide"
)

# --- Backend Functions ---
def parse_calibration_entries(cal_file):
    if cal_file is None:
        return []
    try:
        stringio = io.StringIO(cal_file.getvalue().decode("utf-8"))
        reader = csv.DictReader(stringio)
    except Exception as e:
        st.error(f"校准文件解析失败: {e}")
        return []

    entries = []
    for row in reader:
        row_lower = {k.lower().strip(): v for k, v in row.items() if k}
        if not row_lower:
            continue

        def get_float(keys, default=0.0):
            for key in keys:
                if key in row_lower and str(row_lower[key]).strip() != "":
                    return float(row_lower[key])
            return default

        try:
            port_id = int(get_float(['port_id', 'port', 'id'], default=-1))
            if port_id < 1:
                continue
            loss = get_float(['cable_loss_db', 'loss', 'loss_db'])
            phase = get_float(['cable_phase_deg', 'phase', 'phase_deg'])
            gain = get_float(['probe_gain_dbi', 'gain', 'gain_dbi'])
            entries.append(CalibrationEntry(port_id=port_id, cable_loss_db=loss, cable_phase_deg=phase, probe_gain_dbi=gain))
        except (ValueError, TypeError):
            continue

    return entries

def build_calibration_vector(entries, num_ports):
    if not entries:
        return None
    cal_config = CalibrationConfig(entries=entries)
    cal_vector = np.ones((num_ports, 1), dtype=complex)
    for port_id in range(1, num_ports + 1):
        cal_vector[port_id - 1, 0] = cal_config.get_system_response(port_id)
    return cal_vector

# --- UI: Helper function for antenna config ---
def antenna_panel(antenna_type: str):
    key_prefix = f"{antenna_type.lower()}"
    label_map = {'Tx': '发射端 (Tx)', 'Rx': '接收端 (Rx)'}
    st.write(f"**{label_map.get(antenna_type, antenna_type)} 天线配置**")

    array_type = st.selectbox('阵列类型', ('ULA', 'URA'), key=f'{key_prefix}_type')
    element_pattern = st.selectbox('单元方向图', ('Isotropic', 'TR38.901'), key=f'{key_prefix}_pattern')
    # Polarization per antenna array. Default 'V' preserves single-pol callers;
    # any non-V value (H / ±45° slant) requires chamber.dual_polarized=True.
    polarization = st.selectbox(
        '极化 (Polarization)', ('V', 'H', '+45', '-45'),
        key=f'{key_prefix}_pol',
        help=(
            "V (vertical, θ-pol)、H (horizontal, φ-pol) 或 ±45° 交叉极化. 默认 V. "
            "非 V 极化 (H / ±45°) 需要勾选暗室的「双极化探头」(dual_polarized=True)."
        ),
    )

    # Array orientation in the GCS (TR 38.901 Sec 7.1.3). All 0 => identity. Only
    # meaningful for dynamic (non-rotating) OTA where the DUT's fixed starting
    # orientation shapes chamber geometry; the 360° turntable test averages it out.
    with st.expander('阵列朝向 (Sec 7.1.3, 默认 0)', expanded=False):
        bearing_deg = st.number_input('Bearing α (deg)', -180.0, 180.0, 0.0, 1.0, key=f'{key_prefix}_bearing')
        downtilt_deg = st.number_input('Downtilt β (deg)', -180.0, 180.0, 0.0, 1.0, key=f'{key_prefix}_downtilt')
        slant_deg = st.number_input('Slant γ (deg)', -180.0, 180.0, 0.0, 1.0, key=f'{key_prefix}_slant')

    if array_type == 'ULA':
        num_cols = st.number_input('单元数量 (Elements)', 1, 16, 2, key=f'{key_prefix}_ula_cols')
        spacing_h = st.number_input('单元间距 (λ)', 0.1, 4.0, 0.5, 0.1, key=f'{key_prefix}_ula_spacing')
        return AntennaArrayConfig(array_type='ULA', element_pattern=element_pattern, num_rows=1, num_cols=num_cols, spacing_h=spacing_h, polarization=polarization,
                                  bearing_deg=bearing_deg, downtilt_deg=downtilt_deg, slant_deg=slant_deg)
    else: # URA
        num_rows = st.number_input('行数 (Rows)', 1, 8, 2, key=f'{key_prefix}_ura_rows')
        num_cols = st.number_input('列数 (Columns)', 1, 8, 2, key=f'{key_prefix}_ura_cols')
        spacing_h = st.number_input('水平间距 (λ)', 0.1, 4.0, 0.5, 0.1, key=f'{key_prefix}_ura_spacing_h')
        spacing_v = st.number_input('垂直间距 (λ)', 0.1, 4.0, 0.5, 0.1, key=f'{key_prefix}_ura_spacing_v')
        return AntennaArrayConfig(array_type='URA', element_pattern=element_pattern, num_rows=num_rows, num_cols=num_cols, spacing_h=spacing_h, spacing_v=spacing_v, polarization=polarization,
                                  bearing_deg=bearing_deg, downtilt_deg=downtilt_deg, slant_deg=slant_deg)

# --- UI Layout ---
st.title("MIMO OTA 仿真器 V4.4")

# --- Session State Initialization ---
if 'simulation_results' not in st.session_state:
    st.session_state.simulation_results = None
if 'synthesis_method' not in st.session_state:
    st.session_state.synthesis_method = 'ray'

with st.sidebar:
    st.header("控制面板")
    with st.expander("1. 场景与几何 (Scenario)", expanded=False):
        input_mode_label = st.radio("信道定义方式", ('标准 3GPP/WiFi 模型', '自定义 JSON (Custom Profile)'))
        input_mode = 'standard' if input_mode_label == '标准 3GPP/WiFi 模型' else 'custom'

        # TR 38.901 CDL 合规选项默认值（仅 3GPP CDL 模型下渲染控件覆盖）。
        scale_to_scenario = False
        split_subclusters = False
        c_ds_ns = 3.91

        custom_profile_path = None
        custom_profile_obj = None
        if input_mode == 'custom':
            st.info("已选择自定义信道。请上传您的自定义 JSON 配置文件：")
            uploaded_json = st.file_uploader("上传 JSON 文件", type=['json'], key="custom_json_upload")
            if uploaded_json is not None:
                # Phase 5 P3: try v2.0 Pydantic schema first; fall back to legacy
                # v1.0 file-path route (CustomJSONBuilder shim, emits DeprecationWarning).
                raw_bytes = uploaded_json.getbuffer()
                try:
                    cdl_dict = json.loads(bytes(raw_bytes).decode('utf-8'))
                    custom_profile_obj = CustomCDLProfile.from_dict(cdl_dict)
                    st.success("自定义配置 (v2.0 Pydantic) 已加载")
                except (ValidationError, json.JSONDecodeError, UnicodeDecodeError) as e:
                    # Fallback: persist to temp file so legacy CustomJSONBuilder can read it.
                    temp_path = os.path.abspath("temp_uploaded_custom_profile.json")
                    with open(temp_path, "wb") as f:
                        f.write(raw_bytes)
                    custom_profile_path = temp_path
                    st.warning(
                        f"v2.0 Pydantic 校验失败 ({type(e).__name__})，使用 v1.0 legacy 路径"
                        f"（DeprecationWarning）。建议升级到 v2.0 schema：见 "
                        f"doc/MIMO_First_Integration_Spec.md §3"
                    )
                
            standard = '3GPP'
            model_name = 'Custom'
            cluster_model_name = None
            center_frequency_hz = 3.5
            ue_velocity_kph = 0
            bandwidth_mhz = 20.0
            enable_ofdm = False
            n_fft_text = ""
            phase_policy_label = '固定'
            normalize_power = True
            delay_scale = None
        else:
            standard = st.selectbox('标准', ('3GPP', 'WiFi'), key='standard')
            if standard == '3GPP':
                model_name = st.selectbox('大尺度环境', ('UMa', 'UMi-StreetCanyon', 'UMi-OpenArea', 'RMa', 'InH-Office', 'SMa', 'InF'), key='model_name')
                cluster_model_name = st.selectbox('小尺度簇模型', ('Stochastic', 'CDL-A', 'CDL-B', 'CDL-C', 'CDL-D', 'CDL-E', 'SCME'), key='cluster_model_name')
                if cluster_model_name and cluster_model_name.startswith('CDL-'):
                    st.caption("TR 38.901 CDL 合规选项 (默认关，开启会改变信道统计)")
                    scale_to_scenario = st.checkbox(
                        '场景 LSP 重缩放 (Sec 7.7.5.1)', value=False, key='scale_to_scenario',
                        help="把 CDL 表格的时延/角度重缩放到所选场景的大尺度参数 (DS / ASA / ASD / ZSA / ZSD)。关闭时使用 CDL 表面原始值。"
                    )
                    split_subclusters = st.checkbox(
                        '子簇分裂 (Sec 7.7.1)', value=False, key='split_subclusters',
                        help="把两个最强簇各分裂为 3 个子簇 (延迟偏移 {0,1.28,2.56}·c_DS，功率比 {0.5,0.3,0.2})。"
                    )
                    if split_subclusters:
                        c_ds_ns = st.number_input(
                            'c_DS (ns)', 0.1, 50.0, 3.91, 0.01, key='c_ds_ns',
                            help="子簇延迟扩展常数，默认 3.91 ns。"
                        )
            else:
                model_name = st.selectbox('WiFi 模型', ('TGax-A', 'TGax-B', 'TGax-C', 'TGax-D', 'TGax-E', 'TGax-F'), key='wifi_model_name')
                cluster_model_name = None
            center_frequency_hz = st.number_input('中心频率 (GHz)', 0.5, 100.0, 3.5, 0.1, key='freq')
            ue_velocity_kph = st.slider('UE 移动速度 (km/h)', 0, 120, 3, key='speed')
            if standard == 'WiFi':
                bandwidth_mhz = st.number_input('WiFi 带宽 (MHz)', 5.0, 320.0, 80.0, 5.0, key='wifi_bw')
                enable_ofdm = st.checkbox('OFDM 子载波计算', value=False, key='wifi_ofdm')
                n_fft_text = st.text_input('FFT 大小 (可选)', value="", key='wifi_nfft')
                phase_policy_label = st.selectbox('Tap 相位', ('固定', '随机'), key='wifi_phase')
                normalize_power = st.checkbox('Tap 功率归一', value=True, key='wifi_norm')
                manual_delay_scale = st.checkbox('手动延迟缩放', value=False, key='wifi_delay_manual')
                delay_scale = None
                if manual_delay_scale:
                    delay_scale = st.number_input('延迟缩放系数', 0.01, 10.0, 1.0, 0.01, key='wifi_delay_scale')
            else:
                bandwidth_mhz = 20.0
                enable_ofdm = False
                n_fft_text = ""
                phase_policy_label = '固定'
                normalize_power = True
                delay_scale = None
                
        center_frequency_hz *= 1e9

    with st.expander("2. 暗室配置 (Chamber)", expanded=False):
        num_probes = st.slider('探头数量', 1, 64, 16, 1, key="num_probes")
        radius_m = st.number_input('探头环半径 (m)', 0.5, 30.0, 1.5, 0.5, key='radius')
        dual_polarized = st.checkbox('双极化探头', value=True)

        # 远场条件提示
        wavelength = 3e8 / center_frequency_hz
        st.caption(f"💡 远场条件参考：R > 2D²/λ (当前波长: {wavelength*100:.2f} cm)")
        st.caption(f"   例如：DUT尺寸0.5m需要>5.8m，1.0m需要>23m，2.0m需要>92m")
        st.caption(
            "📐 非标暗室 (sparse / sector / dual-ring) 通过 Python API 或 "
            "MIMO-First 微服务配置 `ChamberConfig.distribution='custom' + "
            "probe_positions=[ProbePosition(az, el), ...]`；详见 "
            "[docs/api/mimo_ota_simulator.md §9](../docs/api/mimo_ota_simulator.md)。"
            "GUI 暂不支持手工配 probe positions（场景需求暂未触发，避免 UI 噪声）。"
        )

    with st.expander("3. 物理参数与校准", expanded=False):
        probe_gain_dBi = st.number_input("探头天线增益 (dBi)", -10.0, 20.0, 0.0, 0.1)
        qz_radius_m = st.number_input(
            "静区半径 (m)", 0.1, 2.0, 0.3, 0.05,
            help="仅作记录 (metadata-only), 不参与任何计算 — 见 audit batch 1 F1")
        calibration_usage = st.selectbox(
            "校准应用范围",
            ("仅导出预失真（不影响仿真）", "仿真中引入硬件响应（导出不预失真）"),
            key="cal_usage"
        )
        cal_file = st.file_uploader("上传校准文件 (.csv)", type=['csv'], help="包含 Port, Loss, Phase, Gain 等列")

    with st.expander("4. 天线配置 (Antenna)", expanded=False):
        tx_antenna_config = antenna_panel("Tx")
        st.divider()
        rx_antenna_config = antenna_panel("Rx")

    with st.expander("5. 运行控制 (Run Control)", expanded=False):
        st.selectbox(
            '合成方法 (Synthesis Method)',
            ('pinv', 'cluster', 'ray', 'strict_pfs'),
            index=2,
            key='synthesis_method',
            help=(
                "**strict_pfs**: per-probe 独立 fading，per-realization 即 strict PFS，"
                "**推荐用于规避输出端相位校准**；"
                "ray: 射线级 GSCM (默认/传统/平均场景)；"
                "cluster: 簇级 (deprecated)；"
                "pinv: 伪逆 (deprecated, F64 export broken)。"
            )
        )
        
        run_mode = st.radio("运行模式", ('单次快照 (Single Snapshot)', '统计验证 (Statistical Validation)'))
        
        # custom 模式下这两个控件不生效 (profile 是 source of truth), 禁用
        # 而非静默忽略 (audit batch 2, P3-2)
        _custom_mode = input_mode == 'custom'
        force_condition = st.selectbox(
            'LOS/NLOS 条件', ('默认 (Default)', '强制 LOS', '强制 NLOS'),
            key='los', disabled=_custom_mode,
            help="自定义 JSON 模式下由 profile.is_los 决定, 此控件禁用" if _custom_mode else None)
        use_median_lsps = st.checkbox(
            '使用 LSP 中值', value=False, disabled=_custom_mode,
            help="自定义 JSON 模式下无 LSP 采样, 此控件禁用" if _custom_mode
            else "跳过随机采样，使用大尺度参数的中位值。")
        _jitter_disabled = (st.session_state.synthesis_method == 'strict_pfs')
        probe_phase_jitter = st.checkbox(
            '模拟探头相位抖动',
            value=False,
            disabled=_jitter_disabled,
            help=(
                "为每个探头添加 ±180度（完全随机相位）的相位噪声。"
                "用途：打破 Cluster/Ray 模型的人工相干性，**仅在大规模平均（多 drop 统计验证）场景下统计意义有效**；"
                "单次 realization 不能视为 strict PFS。**与上传的相位校准互斥**：jitter 会覆盖 calibration_vector 提供的 per-probe 静态相位补偿。"
                "**strict_pfs 模式下自动禁用**：strict_pfs 已数学保证 per-probe 独立 fading，jitter 多余且会污染。"
            )
        )
        if _jitter_disabled:
            st.info("ℹ️ strict_pfs 已内建 per-probe 独立 fading，无需 jitter（已自动禁用）。")
        # Only warn when cal will actually flow into the simulator. The "仅导出预失真"
        # mode keeps cal data for the F64 exporter only and never builds a
        # calibration_vector for ChamberConfig — jitter cannot swamp anything in
        # that case (per PR #8 Codex P2).
        _cal_goes_to_sim = (
            cal_file is not None
            and calibration_usage.startswith("仿真中引入硬件响应")
        )
        if _cal_goes_to_sim and probe_phase_jitter:
            st.warning(
                "⚠️ 已同时启用 probe_phase_jitter 和 calibration_vector："
                "jitter 注入 ±180° 随机相位会**完全覆盖**校准文件提供的 per-probe 相位补偿信息，"
                "导致 cal 数据被静默丢弃。请二选一：要校准就取消 jitter；要 jitter 就移除 cal 文件。"
            )
        enable_ul_dl = st.checkbox('UL/DL 双向生成', value=False, help="生成 UL/DL 两套结果（互易镜像，不包含独立 UL 信道建模）。")
        reciprocity_transform_label = st.selectbox('互易变换', ('共轭转置', '转置'), key='reciprocity', help="仅互易镜像：对 DL 结果做转置/共轭转置得到 UL。")
        
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
        f64_mode = st.selectbox("Amplitude Mode", ["A (Ext ATT)", "B (Internal Scaling)"], key="f64_mode")
        f64_duration = st.number_input("时长 (s)", 0.001, 10.0, 0.1, 0.001, key="f64_dur")
        f64_sample_rate = st.number_input("采样率 (Hz)", 1.0, 100e6, 1000.0, 100.0, format="%.2f", key="f64_sr")
        
        # 2. Calibration Upload (Shown again here for convenience or linked?)
        # Since it's in Expander 3, maybe just remind user?
        # But user might look here. Let's rely on Expander 3 input.
        if cal_file is not None:
            if calibration_usage.startswith("仅导出预失真"):
                st.success("已加载校准文件（仅用于导出预失真）。")
            else:
                st.success("已加载校准文件（用于仿真硬件响应，导出不预失真）。")
        else:
            st.info("未加载校准文件，将使用理想参数。")
        
        # 3. Generate Button
        export_ul_files = st.checkbox('导出 UL 文件', value=False, key='export_ul')
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
                    
                    cal_config = None
                    if cal_file is not None and calibration_usage.startswith("仅导出预失真"):
                        entries = parse_calibration_entries(cal_file)
                        if entries:
                            cal_config = CalibrationConfig(entries=entries)

                    exporter = PropsimASCIIExporter(p_config, calibration_config=cal_config)
                    
                    # Run Export (Generate files in temp)
                    out_files = exporter.export(st.session_state.simulation_results, output_path="propsim_output_dl.asc", direction="dl")
                    if export_ul_files:
                        if st.session_state.simulation_results.get('channel_impulse_response_ul') is None:
                            st.warning("当前结果不包含 UL 数据，请启用 UL/DL 双向生成后重新仿真。")
                        else:
                            out_files += exporter.export(st.session_state.simulation_results, output_path="propsim_output_ul.asc", direction="ul")
                    
                    # Create ZIP in memory
                    import zipfile
                    import io
                    
                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                        for fpath in out_files:
                            if os.path.exists(fpath):
                                zip_file.write(fpath, os.path.basename(fpath))
                    
                    zip_buffer.seek(0)
                    
                    st.download_button(
                        label="下载 .zip 文件 (含多个通道)",
                        data=zip_buffer,
                        file_name=f"propsim_mode{mode_char}_{f64_duration}s.zip",
                        mime="application/zip"
                    )
                    st.success(f"文件生成成功！共生成 {len(out_files)} 个通道文件。请点击上方按钮下载 ZIP 包。")
                    
                except Exception as e:
                    st.error(f"导出失败: {e}")
            else:
                st.warning("请先运行 '单次快照 (Single Snapshot)' 模式以生成信道数据。")

    if st.button("开始仿真", type="primary"):
        calibration_vector = None
        if cal_file is not None and calibration_usage.startswith("仿真中引入硬件响应"):
            entries = parse_calibration_entries(cal_file)
            num_effective_probes = num_probes * (2 if dual_polarized else 1)
            if entries:
                if any(entry.port_id > num_effective_probes for entry in entries):
                    st.warning("校准文件端口数量超过当前探头数量，超出部分将被忽略。")
                calibration_vector = build_calibration_vector(entries, num_effective_probes)

        chamber_config_params = {
            "num_probes": num_probes,
            "dual_polarized": dual_polarized,
            "distribution": 'ring',
            "radius_m": radius_m,
            "probe_antenna": {"gain_dBi": probe_gain_dBi, "pattern": "Isotropic"},
            "quiet_zone_radius_m": qz_radius_m,
            "calibration_vector": calibration_vector
        }
        n_fft = None
        if n_fft_text.strip():
            try:
                n_fft = int(n_fft_text)
            except ValueError:
                st.warning("FFT 大小无效，将使用自动推导。")
                n_fft = None
        phase_policy = 'random' if phase_policy_label == '随机' else 'zero'

        target_config_params = {
            "input_mode": input_mode,
            "custom_profile_path": custom_profile_path,
            "custom_profile": custom_profile_obj,  # Phase 5 Pydantic-validated object (preferred)
            "standard": "wifi" if standard == "WiFi" else "3gpp",
            "model_name": model_name,
            "cluster_model_name": cluster_model_name if cluster_model_name and cluster_model_name != 'Stochastic' else None,
            "center_frequency_hz": center_frequency_hz,
            "bandwidth_hz": bandwidth_mhz * 1e6,
            "enable_ofdm": enable_ofdm,
            "n_fft": n_fft,
            "phase_policy": phase_policy,
            "normalize_power": normalize_power,
            "delay_scale": delay_scale,
            "enable_ul_dl": enable_ul_dl,
            "reciprocity_transform": "conjugate_transpose" if reciprocity_transform_label == '共轭转置' else "transpose",
            "ue_velocity": (ue_velocity_kph * 1000 / 3600, 0, 0),
            "tx_antenna": tx_antenna_config.model_dump(),
            "rx_antenna": rx_antenna_config.model_dump(),
            "force_condition": (None if force_condition == '默认 (Default)' else force_condition.split(' ')[-1]),
            "random_seed": random_seed,
            "use_median_lsps": use_median_lsps,
            "scale_to_scenario": scale_to_scenario,
            "split_subclusters": split_subclusters,
            "c_ds_ns": c_ds_ns
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
        # Ensure Ray components are available for Exporter if present
        # Note: They are inside 'results' because session_state.simulation_results IS the raw return dict from simulator.run()
        # Wait, let's check run_simulation logic in app.py
        # st.session_state.simulation_results = simulator.run(...)
        # So 'results' IS the full dictionary returned by simulator.py.
        # Simulator.py returns: { ..., 'ray_components': ..., 'probe_phase_jitter_vector': ... }
        # So ray_components SHOULD be there if simulator.py was updated.
        
        # Let's verify simulator.py content.
        # I updated simulator.py in the previous turn to return 'ray_components'.
        # So app.py receives it automatically because it stores the whole dict.
        
        # BUT, let's check GUI.py. GUI.py constructs a NEW dict 'results = {...}'.
        # App.py assigns the return value DIRECTLY: 
        # st.session_state.simulation_results = simulator.run(...)
        
        # So App.py SHOULD be fine?
        # Let's re-read the App.py logic carefully.
        pass # Placeholder to allow thinking.
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

        # OFDM 频响 (audit batch 2 / P2-5: enable_ofdm 此前算了不显示)
        if results.get('h_subcarriers') is not None:
            with st.expander("OFDM 子载波频响 |H(f)|", expanded=False):
                h_sc = np.asarray(results['h_subcarriers'])
                mag_db = 20 * np.log10(np.maximum(np.abs(h_sc[:, 0, 0]), 1e-12))
                st.line_chart(mag_db)
                st.caption(f"N_FFT = {h_sc.shape[0]}; 纵轴 dB, 横轴子载波索引")

        # --- Create the two main plots ---
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("探头功率分布与 Tx 贡献")
            fig, ax = plt.subplots()
            
            if np.any(probe_tx_power_contributions):
                # Remove shift/roll to match physical port indices directly
                x_coords = np.arange(num_effective_probes)
                
                bottoms = np.zeros(num_effective_probes)
                colors = plt.cm.get_cmap('tab10', num_tx_antennas)

                for tx_idx in range(num_tx_antennas):
                    power_data = probe_tx_power_contributions[:, tx_idx]
                    ax.bar(x_coords, power_data, bottom=bottoms, color=colors(tx_idx), label=f'Tx Ant {tx_idx}')
                    bottoms += power_data

                ax.set_xlabel("Effective Probe ID (Physical Index)")
                ax.set_ylabel("Power (linear scale)")
                ax.set_title("Power per Probe (Direct Mapping)")
                # Show some ticks but not too crowded
                if num_effective_probes <= 32:
                    ax.set_xticks(x_coords)
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
            st.subheader("目标簇角度分布 (极坐标 - AoA)")
            probe_angles_rad = np.linspace(0, 2 * np.pi, num_probes, endpoint=False)
            clusters_sorted = sorted(clusters, key=lambda c: c.power, reverse=True)
            
            # FIXED: Use AoA (Arrival) for Chamber Probes, NOT AoD (Departure)
            cluster_aoas_rad = np.deg2rad([c.aoa_phi for c in clusters_sorted])
            
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
                    # Label logical ports: V=2*i, H=2*i+1
                    ax_polar.text(v_angles[i], 1.25, str(2*i), ha='center', va='center', fontsize=7, color='blue')
                    ax_polar.text(h_angles[i], 1.25, str(2*i + 1), ha='center', va='center', fontsize=7, color='red')
            else:
                ax_polar.scatter(probe_angles_rad, np.ones(num_probes) * 1.1, s=100, c='blue', marker='X', label='Probes')
                for i, angle in enumerate(probe_angles_rad):
                    ax_polar.text(angle, 1.2, str(i), ha='center', va='center', fontsize=8, color='blue')

            for i, (angle, color) in enumerate(zip(cluster_aoas_rad, cluster_colors)):
                ax_polar.plot([angle, angle], [0, 1], color=color, linewidth=2)
            
            ax_polar.set_title("Probe Positions vs Target Cluster AoA (0°=East)")
            ax_polar.set_yticklabels([])
            # Removed custom theta location/direction to match Simulator's math (0=East, CCW)
            # ax_polar.set_theta_zero_location('N')
            # ax_polar.set_theta_direction(-1)
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
