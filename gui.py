import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import json
import threading
import queue
import os
import zipfile

from channel_model_38901.simulator import ChannelSimulator
from channel_model_38901.channel_generator import ChannelGenerator
from channel_model_38901.common.data_models import AntennaArray, EngineCluster
from channel_model_38901.common.geometry import get_antenna_positions

# Try importing MIMO OTA Simulator (Optional extension)
try:
    from mimo_ota_simulator.simulator import MIMO_OTA_Simulator
    from mimo_ota_simulator.data_models import ChamberConfig, TargetChannelConfig, AntennaArrayConfig, PropsimExportConfig, CalibrationConfig, CalibrationEntry
    from mimo_ota_simulator.cdl_schema import CustomCDLProfile
    from mimo_ota_simulator.exporters import PropsimASCIIExporter
    MIMO_OTA_AVAILABLE = True
except ImportError:
    MIMO_OTA_AVAILABLE = False
    print("Warning: mimo_ota_simulator not found. OTA features disabled.")

try:
    from wifi_channel_model.simulator import WiFiChannelSimulator
    from wifi_channel_model.common.data_models import WiFiScenarioConfig
    WIFI_AVAILABLE = True
except ImportError:
    WIFI_AVAILABLE = False
    print("Warning: wifi_channel_model not found. WiFi features disabled.")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class ChannelModelApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("3GPP TR 38.901 Hybrid Channel Simulator")
        self.geometry("1200x900")

        self.simulation_queue = queue.Queue()
        self.controls = {}
        self.last_results = None
        self.after(100, self.process_queue)

        self._create_widgets()

    def _create_widgets(self):
        # Main Container
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Left Column (Scrollable) ---
        left_container = ttk.Frame(main_frame)
        left_container.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10), anchor='n')

        left_canvas = tk.Canvas(left_container, borderwidth=0, highlightthickness=0)
        left_scrollbar = ttk.Scrollbar(left_container, orient="vertical", command=left_canvas.yview)
        left_canvas.configure(yscrollcommand=left_scrollbar.set)
        left_canvas.pack(side=tk.LEFT, fill=tk.Y, expand=False)
        left_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        left_col = ttk.Frame(left_canvas)
        left_window = left_canvas.create_window((0, 0), window=left_col, anchor="nw")

        def _on_left_frame_configure(event):
            left_canvas.configure(scrollregion=left_canvas.bbox("all"))
            left_canvas.itemconfig(left_window, width=left_canvas.winfo_width())

        left_col.bind("<Configure>", _on_left_frame_configure)

        def _on_left_canvas_configure(event):
            left_canvas.itemconfig(left_window, width=event.width)

        left_canvas.bind("<Configure>", _on_left_canvas_configure)

        def _on_left_mousewheel(event):
            lx = left_container.winfo_rootx()
            ly = left_container.winfo_rooty()
            if not (lx <= event.x_root <= lx + left_container.winfo_width() and ly <= event.y_root <= ly + left_container.winfo_height()):
                return
            if event.delta:
                left_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            elif event.num == 4:
                left_canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                left_canvas.yview_scroll(1, "units")

        left_canvas.bind_all("<MouseWheel>", _on_left_mousewheel)
        left_canvas.bind_all("<Button-4>", _on_left_mousewheel)
        left_canvas.bind_all("<Button-5>", _on_left_mousewheel)
        
        general_param_frame = ttk.LabelFrame(left_col, text="通用参数", padding="10")
        general_param_frame.pack(fill=tk.X, pady=(0, 10))
        
        ota_param_frame = ttk.LabelFrame(left_col, text="MIMO OTA 参数 (可选)", padding="10")
        ota_param_frame.pack(fill=tk.X) # Use fill=X so it doesn't stretch weirdly vertically if empty

        # --- Right Column (Expandable) ---
        right_col = ttk.Frame(main_frame)
        right_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # 1. Antenna Configs (Tx | Rx)
        ant_frame = ttk.Frame(right_col)
        ant_frame.pack(fill=tk.X, pady=(0, 10))
        
        tx_param_frame = ttk.LabelFrame(ant_frame, text="发射端 (Tx) 参数", padding="10")
        tx_param_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        rx_param_frame = ttk.LabelFrame(ant_frame, text="接收端 (Rx) 参数", padding="10")
        rx_param_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))

        # 2. Action Panel
        action_frame = ttk.LabelFrame(right_col, text="操作", padding="10")
        action_frame.pack(fill=tk.X, pady=(0, 10))

        # 3. Notebook (Results) - Fills remaining space
        self.notebook = ttk.Notebook(right_col)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Initialize Internal Widgets
        # Note: We need to configure grid columns for the frames that use create_entry (which uses grid)
        for frame in [general_param_frame, ota_param_frame, tx_param_frame, rx_param_frame]:
            frame.grid_columnconfigure(1, weight=1)

        self._create_parameter_widgets(general_param_frame, ota_param_frame, tx_param_frame, rx_param_frame)
        self._create_control_buttons(action_frame)
        
        self._create_log_display()
        self._create_results_tabs()
        
        self.update_antenna_ui('tx')
        self.update_antenna_ui('rx')

    def _create_parameter_widgets(self, general_frame, ota_frame, tx_frame, rx_frame):
        environments = ['UMa', 'UMi-StreetCanyon', 'UMi-OpenArea', 'RMa', 'InH-Office', 'SMa', 'InF']
        cluster_models = ['Stochastic', 'CDL-A', 'CDL-B', 'CDL-C', 'CDL-D', 'CDL-E', 'SCME']
        standards = ['3GPP', 'WiFi', 'Custom JSON']
        wifi_models = ['TGax-A', 'TGax-B', 'TGax-C', 'TGax-D', 'TGax-E', 'TGax-F']
        
        general_params = {
            "standard": ("标准:", "3GPP", standards),
            "scenario_name": ("大尺度环境:", "UMa", environments),
            "cluster_model_name": ("小尺度簇模型:", "Stochastic", cluster_models),
            "wifi_model_name": ("WiFi 模型:", "TGax-D", wifi_models),
            "wifi_bandwidth_mhz": ("WiFi 带宽 (MHz):", "80", None),
            "wifi_enable_ofdm": ("WiFi OFDM:", False, [True, False]),
            "wifi_n_fft": ("WiFi FFT 大小:", "", None),
            "wifi_phase_policy": ("WiFi Tap 相位:", "zero", ["zero", "random"]),
            "wifi_normalize_power": ("WiFi 功率归一:", True, [True, False]),
            "wifi_delay_scale": ("WiFi 延迟缩放:", "", None),
            "center_frequency_ghz": ("中心频率 (GHz):", "3.5", None),
            "los_condition": ("LOS 条件:", "Automatic", ["Automatic", "LOS", "NLOS"]),
            "use_median_lsps": ("使用LSP中值:", False, [True, False]),
            "random_seed": ("随机种子:", "", None),
            "simulation_time_s": ("仿真时间 (s):", "0.0", None)
        }
        for i, (key, (text, default, values)) in enumerate(general_params.items()):
            self._create_entry(general_frame, key, text, default, values, i)
        self.controls["standard"]["var"].trace_add('write', lambda *args: self.update_standard_ui())

        custom_frame = ttk.Frame(general_frame)
        custom_frame.grid(row=len(general_params), column=0, columnspan=2, sticky='ew', pady=2)
        self.btn_load_custom = ttk.Button(custom_frame, text="选择 JSON", command=self._load_custom_json)
        self.btn_load_custom.pack(side=tk.LEFT, padx=5)
        self.lbl_custom_json = ttk.Label(custom_frame, text="无文件", font=("Arial", 9, "italic"))
        self.lbl_custom_json.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.current_custom_json_path = None


        # OTA Parameters
        self.ota_vars = {}
        if MIMO_OTA_AVAILABLE:
            # Vertical Layout: OTA Gen -> Separator -> Propsim Export
            
            ota_enable = self._create_entry(ota_frame, "ota_enable", "启用 MIMO OTA:", False, [True, False], 0)
            ota_enable.trace_add('write', lambda *args: self.update_ota_ui())
            
            self._create_entry(ota_frame, "ota_num_probes", "探头数量:", "16", None, 1)
            self._create_entry(ota_frame, "ota_radius_m", "探头半径 (m):", "1.5", None, 2)
            self._create_entry(ota_frame, "ota_dual_pol", "双极化探头:", True, [True, False], 3)
            # Default 'ray' (PR #7). 'strict_pfs' = Phase 1 recommended (avoids output
            # phase calibration). cluster/pinv are deprecated (Phase 3). Phase 6 dual-pol
            # synthesis is active when ota_dual_pol=True.
            self._create_entry(ota_frame, "ota_method", "合成方法 (cluster/pinv deprecated):",
                               "ray", ["ray", "strict_pfs", "cluster", "pinv"], 4)
            # Phase 0 jitter UI: ±180° random phase per probe. Only meaningful for
            # multi-drop ensemble averaging; strict_pfs makes it redundant + cal makes
            # it mutually exclusive (validated in start_simulation_thread).
            self._create_entry(ota_frame, "ota_jitter", "相位抖动 ±180° (仅多平均):",
                               False, [True, False], 5)
            self._create_entry(ota_frame, "ota_ul_dl", "UL/DL 双向(互易镜像):", False, [True, False], 6)
            self._create_entry(ota_frame, "ota_reciprocity", "互易变换(镜像):", "conjugate_transpose", ["conjugate_transpose", "transpose"], 7)
            
            # Propsim Hardware Export
            ttk.Separator(ota_frame, orient='horizontal').grid(row=8, column=0, columnspan=2, sticky='ew', pady=10)
            ttk.Label(ota_frame, text="Propsim F64 导出").grid(row=9, column=0, columnspan=2, sticky='w')
            
            self._create_entry(ota_frame, "f64_mode", "幅度模式:", "A", ["A (Ext ATT)", "B (Internal)"], 10)
            self._create_entry(ota_frame, "f64_duration_s", "时长 (s):", "0.1", None, 11)
            self._create_entry(ota_frame, "f64_sample_rate_hz", "采样率 (Hz):", "1000.0", None, 12)
            self._create_entry(ota_frame, "f64_export_ul", "导出 UL 文件:", False, [True, False], 13)
            
            # Calibration Loader
            cal_frame = ttk.Frame(ota_frame)
            cal_frame.grid(row=14, column=0, columnspan=2, sticky='ew', pady=2)
            self.load_cal_button = ttk.Button(cal_frame, text="Load Cal CSV", width=12, command=self.load_calibration_file)
            self.load_cal_button.pack(side=tk.LEFT, padx=(5, 5))
            self.cal_file_label = ttk.Label(cal_frame, text="No File Selected", font=("Arial", 9, "italic"))
            self.cal_file_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self.current_cal_config = None 

            self.export_f64_button = ttk.Button(ota_frame, text="Export to F64", command=self.export_to_propsim)
            self.export_f64_button.grid(row=15, column=0, columnspan=2, pady=5)
            
        else:
            ttk.Label(ota_frame, text="MIMO OTA 库未安装").pack()

        self._create_antenna_widgets('tx', tx_frame)
        self._create_antenna_widgets('rx', rx_frame)
        
        if MIMO_OTA_AVAILABLE:
            self.update_ota_ui()
        self.update_standard_ui()

    def update_ota_ui(self):
        if not MIMO_OTA_AVAILABLE: return
        is_enabled = self.controls['ota_enable']['var'].get()
        state = tk.NORMAL if is_enabled else tk.DISABLED
        
        # OTA Gen Params
        for key in ["ota_num_probes", "ota_radius_m", "ota_dual_pol", "ota_method", "ota_jitter", "ota_ul_dl", "ota_reciprocity"]:
            self.controls[key]['label'].config(state=state)
            self.controls[key]['widget'].config(state=state)
            
        # Propsim Params
        for key in ["f64_mode", "f64_duration_s", "f64_sample_rate_hz", "f64_export_ul"]:
            self.controls[key]['label'].config(state=state)
            self.controls[key]['widget'].config(state=state)
        
        self.export_f64_button.config(state=state)
        self.load_cal_button.config(state=state)
        self.update_standard_ui()

    def _load_custom_json(self):
        from tkinter.filedialog import askopenfilename
        filepath = askopenfilename(filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")], title="Load Custom Channel JSON")
        if filepath:
            self.current_custom_json_path = filepath
            self.lbl_custom_json.config(text=filepath.split('/')[-1])
            self.controls["standard"]["var"].set("Custom JSON")

    def update_standard_ui(self):
        standard = self.controls['standard']['var'].get()
        is_wifi = (standard == 'WiFi')
        is_custom = (standard == 'Custom JSON')
        ota_enabled = False
        if "ota_enable" in self.controls:
            ota_enabled = self.controls["ota_enable"]["var"].get()
        wifi_keys = [
            "wifi_model_name",
            "wifi_bandwidth_mhz",
            "wifi_enable_ofdm",
            "wifi_n_fft",
            "wifi_phase_policy",
            "wifi_normalize_power",
            "wifi_delay_scale",
        ]
        for key in wifi_keys:
            self.controls[key]['label'].config(state=tk.NORMAL if is_wifi else tk.DISABLED)
            self.controls[key]['widget'].config(state=tk.NORMAL if is_wifi else tk.DISABLED)
        for key in ["scenario_name", "cluster_model_name", "los_condition", "use_median_lsps"]:
            self.controls[key]['label'].config(state=tk.DISABLED if (is_wifi or is_custom) else tk.NORMAL)
            self.controls[key]['widget'].config(state=tk.DISABLED if (is_wifi or is_custom) else tk.NORMAL)
        if is_wifi:
            self.controls["cluster_model_name"]["var"].set("Stochastic")
        show_sim_time = (not is_wifi) and (not ota_enabled)
        if show_sim_time:
            self.controls["simulation_time_s"]["label"].grid()
            self.controls["simulation_time_s"]["widget"].grid()
        else:
            self.controls["simulation_time_s"]["label"].grid_remove()
            self.controls["simulation_time_s"]["widget"].grid_remove()

    def load_calibration_file(self):
        """Loads a calibration CSV file."""
        from tkinter.filedialog import askopenfilename
        import csv
        
        filepath = askopenfilename(filetypes=[("CSV Files", "*.csv"), ("All files", "*.*")], title="Select Calibration File")
        if not filepath: return
        
        try:
            entries = []
            with open(filepath, 'r') as f:
                reader = csv.DictReader(f)
                # Expected columns: Port_ID, Cable_Loss_dB, Cable_Phase_deg, Probe_Gain_dBi
                # We'll allow case-insensitive matching or standard names
                for row in reader:
                    # Normalize keys
                    row_lower = {k.lower().strip(): v for k, v in row.items()}
                    
                    # Safe parse helper
                    def get_float(keys, default=0.0):
                        for k in keys:
                            if k in row_lower: return float(row_lower[k])
                        return default
                    
                    p_id = int(get_float(['port_id', 'port', 'id']))
                    loss = get_float(['cable_loss_db', 'loss', 'loss_db'])
                    phase = get_float(['cable_phase_deg', 'phase', 'phase_deg'])
                    gain = get_float(['probe_gain_dbi', 'gain', 'gain_dbi'])
                    
                    entries.append(CalibrationEntry(
                        port_id=p_id, cable_loss_db=loss, cable_phase_deg=phase, probe_gain_dbi=gain
                    ))
            
            if not entries:
                raise ValueError("No valid calibration entries found in CSV.")
                
            self.current_cal_config = CalibrationConfig(entries=entries)
            filename = filepath.split('/')[-1]
            self.cal_file_label.config(text=f"Loaded: {filename}")
            self.log(f"Loaded calibration for {len(entries)} ports from {filename}")
            
        except Exception as e:
            import traceback
            self.log(f"Calibration Load Failed: {e}\n{traceback.format_exc()}")
            messagebox.showerror("Load Error", f"Failed to load calibration:\n{e}")
            self.current_cal_config = None
            self.cal_file_label.config(text="Load Failed")

    def _create_antenna_widgets(self, prefix, parent_frame):
        self._create_entry(parent_frame, f'{prefix}_position', "Position (x,y,z):", "0,0,25" if prefix == 'tx' else "100,0,1.5", None, 0)
        array_type_var = self._create_entry(parent_frame, f'{prefix}_array_type', "Array Type:", 'ULA', ['ULA', 'URA'], 1)
        array_type_var.trace_add('write', lambda *args, p=prefix: self.update_antenna_ui(p))
        self._create_entry(parent_frame, f'{prefix}_num_rows', "Num Rows:", "1", None, 2)
        self._create_entry(parent_frame, f'{prefix}_num_cols', "Num Cols (ULA: Num Ant):", "2", None, 3)
        self._create_entry(parent_frame, f'{prefix}_spacing_h', "Spacing H (lambda):", "0.5", None, 4)
        self._create_entry(parent_frame, f'{prefix}_spacing_v', "Spacing V (lambda):", "0.5", None, 5)
        self._create_entry(parent_frame, f'{prefix}_element_pattern', "Element Pattern:", 'Isotropic', ['Isotropic', 'TR38.901'], 6)
        # Phase 6 T6.1: polarization V/H. Default 'V' = backward compat.
        # Only matters when chamber.dual_polarized=True. Slant ±45° = Phase 7 (deferred).
        self._create_entry(parent_frame, f'{prefix}_polarization', "Polarization (V/H):", 'V', ['V', 'H'], 7)
        if prefix == 'rx':
            self._create_entry(parent_frame, 'ue_velocity_kph', "UE Velocity (km/h):", "3", None, 8)

    def _create_entry(self, parent, key, text, default, values, row):
        label = ttk.Label(parent, text=text)
        label.grid(row=row, column=0, sticky=tk.W, pady=2)
        if isinstance(default, bool):
            var = tk.BooleanVar(value=default)
            widget = ttk.Checkbutton(parent, variable=var)
        else:
            var = tk.StringVar(value=default)
            if values:
                widget = ttk.Combobox(parent, textvariable=var, values=values, width=18)
            else:
                widget = ttk.Entry(parent, textvariable=var, width=20)
        widget.grid(row=row, column=1, sticky=tk.EW, pady=2, padx=5)
        parent.grid_columnconfigure(1, weight=1)
        self.controls[key] = {'var': var, 'widget': widget, 'label': label}
        return var

    def update_antenna_ui(self, prefix):
        array_type = self.controls[f'{prefix}_array_type']['var'].get()
        is_ura = (array_type == 'URA')
        state = tk.NORMAL if is_ura else tk.DISABLED
        self.controls[f'{prefix}_num_rows']['label'].config(state=state)
        self.controls[f'{prefix}_num_rows']['widget'].config(state=state)
        self.controls[f'{prefix}_spacing_v']['label'].config(state=state)
        self.controls[f'{prefix}_spacing_v']['widget'].config(state=state)
        if not is_ura:
            self.controls[f'{prefix}_num_rows']['var'].set("1")

    def _create_control_buttons(self, parent_frame):
        # Using grid layout for better organization in the bottom-right action frame
        parent_frame.columnconfigure(0, weight=1)
        parent_frame.columnconfigure(1, weight=1)
        parent_frame.columnconfigure(2, weight=1)
        
        self.run_button = ttk.Button(parent_frame, text="Run Simulation", command=self.start_simulation_thread)
        self.run_button.grid(row=0, column=0, padx=5, pady=5, sticky='ew')
        
        self.save_results_button = ttk.Button(parent_frame, text="Save Results", command=self.save_results)
        self.save_results_button.grid(row=0, column=1, padx=5, pady=5, sticky='ew')
        
        self.export_fpga_button = ttk.Button(parent_frame, text="Export FPGA Params", command=self.export_fpga_sos_csv)
        self.export_fpga_button.grid(row=0, column=2, padx=5, pady=5, sticky='ew')

        self.save_button = ttk.Button(parent_frame, text="Save Config", command=self.save_config)
        self.save_button.grid(row=1, column=0, padx=5, pady=5, sticky='ew')
        
        self.load_button = ttk.Button(parent_frame, text="Load Config", command=self.load_config)
        self.load_button.grid(row=1, column=1, padx=5, pady=5, sticky='ew')

    def _create_log_display(self):
        log_frame = ttk.Frame(self.notebook)
        self.log_text = tk.Text(log_frame, height=10, wrap=tk.WORD, state=tk.DISABLED, font=("Courier New", 10))
        scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.config(yscrollcommand=scrollbar.set)
        scrollbar.pack(fill=tk.Y, side=tk.RIGHT)
        self.log_text.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        self.notebook.add(log_frame, text='Log')

    def _create_results_tabs(self):
        pdp_frame = ttk.Frame(self.notebook); self.notebook.add(pdp_frame, text='PDP')
        self.fig_pdp, self.ax_pdp, self.pdp_tree = self._create_plot_and_table(pdp_frame, ["Cluster #", "Delay (s)", "Power (dB)"])
        self.canvas_pdp = FigureCanvasTkAgg(self.fig_pdp, master=pdp_frame)
        self.canvas_pdp.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        aoa_frame = ttk.Frame(self.notebook); self.notebook.add(aoa_frame, text='AoA')
        self.fig_aoa, self.ax_aoa, self.aoa_tree = self._create_plot_and_table(aoa_frame, ["Cluster #", "AoA (deg)", "Power (dB)"])
        self.canvas_aoa = FigureCanvasTkAgg(self.fig_aoa, master=aoa_frame)
        self.canvas_aoa.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        zoa_frame = ttk.Frame(self.notebook); self.notebook.add(zoa_frame, text='ZoA')
        self.fig_zoa, self.ax_zoa, self.zoa_tree = self._create_plot_and_table(zoa_frame, ["Cluster #", "ZoA (deg)", "Power (dB)"])
        self.canvas_zoa = FigureCanvasTkAgg(self.fig_zoa, master=zoa_frame)
        self.canvas_zoa.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        
        cdl_frame = ttk.Frame(self.notebook); self.notebook.add(cdl_frame, text='CDL/SCME')
        cdl_cols = ["#", "Delay(s)", "Pwr(dB)", "AoA(d)", "ZoA(d)", "AoD(d)", "ZoD(d)", "XPR(dB)"]
        self.cdl_tree, cdl_copy_btn = self._create_table_with_copy(cdl_frame, cdl_cols, height=15)
        self.cdl_tree.grid(row=0, column=0, sticky="nsew"); cdl_copy_btn.grid(row=1, column=0, sticky="e", pady=5, padx=5)
        cdl_frame.grid_rowconfigure(0, weight=1); cdl_frame.grid_columnconfigure(0, weight=1)

        self.clear_results()

    def _create_plot_and_table(self, parent, columns):
        parent.grid_rowconfigure(0, weight=1); parent.grid_columnconfigure(0, weight=1)
        fig, ax = plt.subplots(1, 1)
        tree, copy_btn = self._create_table_with_copy(parent, columns)
        tree.grid(row=1, column=0, sticky="ew", pady=(5,0))
        copy_btn.grid(row=2, column=0, sticky="e", pady=5, padx=5)
        return fig, ax, tree

    def _create_table_with_copy(self, parent, columns, height=4):
        tree = ttk.Treeview(parent, columns=columns, show='headings', height=height)
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, anchor=tk.CENTER, width=100)
        copy_btn = ttk.Button(parent, text="Copy Data", command=lambda t=tree: self.copy_tree_data(t))
        return tree, copy_btn

    def clear_results(self):
        for ax, title, xlabel in [(self.ax_pdp, "Power Delay Profile (PDP)", "Delay (s)"),
                                  (self.ax_aoa, "Azimuth of Arrival (AoA) vs. Power", "Angle (degrees)"),
                                  (self.ax_zoa, "Zenith of Arrival (ZoA) vs. Power", "Angle (degrees)")]:
            ax.clear(); ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel("Relative Power (dB)"); ax.grid(True)
        
        for fig in [self.fig_pdp, self.fig_aoa, self.fig_zoa]: fig.tight_layout()
        for canvas in [self.canvas_pdp, self.canvas_aoa, self.canvas_zoa]: canvas.draw()
        
        for tree in [self.pdp_tree, self.aoa_tree, self.zoa_tree, self.cdl_tree]:
            for item in tree.get_children(): tree.delete(item)

    def log(self, message):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def start_simulation_thread(self):
        # Phase 0+1 P0: pre-run validation for jitter conflicts.
        # Surface the same mutex semantics that simulator.py enforces at runtime,
        # via dialog box (TKinter has no native tooltip/inline-warning UX like
        # Streamlit). Block run if config is internally inconsistent.
        try:
            ota_enabled = bool(self.controls.get('ota_enable', {}).get('var',
                               tk.BooleanVar(value=False)).get())
        except Exception:
            ota_enabled = False
        if ota_enabled:
            method = self.controls.get('ota_method', {}).get('var', tk.StringVar()).get()
            jitter = bool(self.controls.get('ota_jitter', {}).get('var',
                          tk.BooleanVar(value=False)).get())
            if method == 'strict_pfs' and jitter:
                messagebox.showwarning(
                    "Conflict: strict_pfs + jitter",
                    "synthesis_method='strict_pfs' already provides per-(probe, cluster) "
                    "independent fading; adding probe phase jitter is redundant and "
                    "pollutes the intentional phase structure.\n\n"
                    "Fix: uncheck '相位抖动 ±180°' OR switch '合成方法' to 'ray'."
                )
                return
            # NOTE (PR #8 Codex P2): no cal+jitter check here. In this TKinter GUI
            # `current_cal_config` is only consumed by export_to_propsim (F64 file
            # writer); run_simulation builds ChamberConfig WITHOUT calibration_vector,
            # so jitter cannot swamp anything in the simulation path. If a future
            # change wires cal into the simulator (à la app.py's "仿真中引入硬件响应"
            # mode), re-add the guard gated on that flag.

        self.run_button.config(state=tk.DISABLED)
        self.log_text.config(state=tk.NORMAL); self.log_text.delete(1.0, tk.END); self.log_text.config(state=tk.DISABLED)
        self.clear_results()
        thread = threading.Thread(target=self.run_simulation)
        thread.daemon = True
        thread.start()

    def run_simulation(self):
        try:
            params = {key: control['var'].get() for key, control in self.controls.items()}
            standard = params["standard"]
            is_wifi = (standard == 'WiFi')
            is_custom = (standard == 'Custom JSON')
            scenario_name = params["scenario_name"]
            wifi_model_name = params["wifi_model_name"]
            cluster_model_name = params["cluster_model_name"]
            if cluster_model_name == 'Stochastic': cluster_model_name = None
            if is_wifi:
                cluster_model_name = None

            center_frequency_hz = float(params["center_frequency_ghz"]) * 1e9
            bs_pos = np.array([[float(p) for p in params["tx_position"].split(',')]])
            ue_pos = np.array([[float(p) for p in params["rx_position"].split(',')]])
            ue_velocity_kph = float(params["ue_velocity_kph"])
            ue_velocity_mps = ue_velocity_kph * 1000 / 3600
            ue_vel_vector = np.array([[ue_velocity_mps, 0, 0]])
            seed_str = params["random_seed"]
            seed = int(seed_str) if seed_str else None
            los_condition_str = params["los_condition"]
            force_los_condition = None if los_condition_str == "Automatic" else los_condition_str
            if is_wifi or is_custom:
                force_los_condition = None
            time_s = float(params["simulation_time_s"])
            use_median_lsps = params["use_median_lsps"]

            # Check if OTA is enabled
            ota_enabled = False
            if MIMO_OTA_AVAILABLE and "ota_enable" in params:
                 ota_enabled = params["ota_enable"]

            c = 3e8
            lambda_meters = c / center_frequency_hz
            
            tx_positions = get_antenna_positions(
                array_type=params['tx_array_type'], num_rows=int(params['tx_num_rows']),
                num_cols=int(params['tx_num_cols']), spacing_h_m=float(params['tx_spacing_h']) * lambda_meters,
                spacing_v_m=float(params['tx_spacing_v']) * lambda_meters
            )
            rx_positions = get_antenna_positions(
                array_type=params['rx_array_type'], num_rows=int(params['rx_num_rows']),
                num_cols=int(params['rx_num_cols']), spacing_h_m=float(params['rx_spacing_h']) * lambda_meters,
                spacing_v_m=float(params['rx_spacing_v']) * lambda_meters
            )

            ant_config = AntennaArray(tx_array_type=params['tx_array_type'], num_tx_rows=int(params['tx_num_rows']), num_tx_cols=int(params['tx_num_cols']), tx_spacing_h=float(params['tx_spacing_h']), tx_spacing_v=float(params['tx_spacing_v']), tx_element_pattern=params['tx_element_pattern'], tx_positions=tx_positions, rx_array_type=params['rx_array_type'], num_rx_rows=int(params['rx_num_rows']), num_rx_cols=int(params['rx_num_cols']), rx_spacing_h=float(params['rx_spacing_h']), rx_spacing_v=float(params['rx_spacing_v']), rx_element_pattern=params['rx_element_pattern'], rx_positions=rx_positions)
            
            model_name = wifi_model_name if is_wifi else scenario_name
            log_msg = f"Starting simulation for Standard: {standard}, Model: {model_name}"
            if not is_wifi:
                log_msg += f", Cluster Model: {cluster_model_name if cluster_model_name else 'Stochastic'}"
            if ota_enabled:
                log_msg += f" [MIMO OTA Mode: {params['ota_method']}, Probes: {params['ota_num_probes']}]"
            self.simulation_queue.put(("log", log_msg))

            # Defensive: seed before sim.run(). MIMO_OTA_Simulator.run() now also
            # seeds when channel_config.random_seed is set (both WiFi + 3GPP paths,
            # fixed 2026-05-18 after GUI parity bug discovery), so this explicit
            # call is technically redundant for the OTA flow but kept for the
            # non-OTA ChannelSimulator path further below (which doesn't auto-seed).
            if seed is not None:
                np.random.seed(seed)

            if ota_enabled:
                if is_wifi and not WIFI_AVAILABLE:
                    self.simulation_queue.put(("error", "WiFi 模块未安装，无法运行 WiFi OTA 仿真。"))
                    self.simulation_queue.put(("done", None))
                    return
                # --- Run MIMO OTA Simulation ---
                simulator = MIMO_OTA_Simulator() # This is the OTA simulator facade
                
                # Build OTA Configs
                chamber_config = ChamberConfig(
                    num_probes=int(params["ota_num_probes"]),
                    radius_m=float(params["ota_radius_m"]),
                    distribution='ring',
                    dual_polarized=params.get("ota_dual_pol", False) # Use GUI value, default False if missing
                )
                
                # Map antenna config to OTA format
                # Note: MIMO_OTA_Simulator expects specific AntennaArrayConfig objects or dicts
                # We'll use the dict/object mapping as defined in mimo_ota_simulator/data_models.py
                
                tx_ant_ota = AntennaArrayConfig(
                    num_rows=int(params['tx_num_rows']), num_cols=int(params['tx_num_cols']),
                    spacing_h=float(params['tx_spacing_h']), spacing_v=float(params['tx_spacing_v']),
                    polarization=params.get('tx_polarization', 'V'),  # Phase 6 T6.1
                )
                rx_ant_ota = AntennaArrayConfig(
                    num_rows=int(params['rx_num_rows']), num_cols=int(params['rx_num_cols']),
                    spacing_h=float(params['rx_spacing_h']), spacing_v=float(params['rx_spacing_v']),
                    polarization=params.get('rx_polarization', 'V'),  # Phase 6 T6.1
                )

                wifi_bandwidth_hz = float(params["wifi_bandwidth_mhz"]) * 1e6 if is_wifi else 20e6
                wifi_n_fft = None
                if params["wifi_n_fft"]:
                    try:
                        wifi_n_fft = int(params["wifi_n_fft"])
                    except ValueError:
                        wifi_n_fft = None
                wifi_delay_scale = None
                if params["wifi_delay_scale"]:
                    try:
                        wifi_delay_scale = float(params["wifi_delay_scale"])
                    except ValueError:
                        wifi_delay_scale = None
                # Phase 5 P3: try v2.0 Pydantic CustomCDLProfile first; fall back to
                # v1.0 legacy file path (CustomJSONBuilder shim, emits DeprecationWarning).
                custom_profile_obj = None
                custom_profile_path = None
                if is_custom:
                    path = getattr(self, "current_custom_json_path", None)
                    if path:
                        try:
                            custom_profile_obj = CustomCDLProfile.from_file(path)
                        except Exception as e:
                            # Fall back to legacy file-path path
                            custom_profile_path = path
                            self.log(
                                f"[Warning] Custom JSON v2.0 Pydantic 校验失败 "
                                f"({type(e).__name__}); 使用 v1.0 legacy 路径 + "
                                f"DeprecationWarning. 建议升级 schema."
                            )

                channel_config = TargetChannelConfig(
                    input_mode="custom" if is_custom else "standard",
                    custom_profile=custom_profile_obj,
                    custom_profile_path=custom_profile_path,
                    standard="wifi" if is_wifi else "3gpp",
                    model_name="Custom" if is_custom else model_name,
                    cluster_model_name=cluster_model_name,
                    center_frequency_hz=center_frequency_hz,
                    bandwidth_hz=wifi_bandwidth_hz,
                    enable_ofdm=params["wifi_enable_ofdm"] if is_wifi else False,
                    n_fft=wifi_n_fft,
                    phase_policy=params["wifi_phase_policy"],
                    normalize_power=params["wifi_normalize_power"],
                    delay_scale=wifi_delay_scale,
                    enable_ul_dl=params["ota_ul_dl"],
                    reciprocity_transform=params["ota_reciprocity"],
                    random_seed=seed,
                    tx_antenna=tx_ant_ota,
                    rx_antenna=rx_ant_ota,
                    force_condition=force_los_condition,
                    ue_velocity=(ue_velocity_mps, 0, 0),
                    use_median_lsps=use_median_lsps
                )

                # Run OTA Sim
                ota_results = simulator.run(
                    chamber_config=chamber_config,
                    channel_config=channel_config,
                    synthesis_method=params["ota_method"],
                    probe_phase_jitter=params["ota_jitter"]
                )
                
                # Adapt OTA results to GUI format
                # OTA returns 'effective_H_matrix', 'lsps', 'clusters', 'pathloss_db', 'is_los', 'transmit_mapping_matrix_A'
                results = {
                    'channel_matrix': ota_results['effective_H_matrix'],
                    'lsps': ota_results['lsps'],
                    'pathloss_db': ota_results['pathloss_db'],
                    'is_los': ota_results['is_los'],
                    'clusters': ota_results['clusters'],
                    'transmit_mapping_matrix_A': ota_results['transmit_mapping_matrix_A'], # Required for F64 Export
                    'ray_components': ota_results.get('ray_components'), # Required for Doppler Export
                    'probe_phase_jitter_vector': ota_results.get('probe_phase_jitter_vector'),
                    'channel_impulse_response': ota_results.get('channel_impulse_response'), # Required for Exporter
                    'channel_impulse_response_ul': ota_results.get('channel_impulse_response_ul'),
                    'ray_components_ul': ota_results.get('ray_components_ul'),
                    'ul_enabled': ota_results.get('ul_enabled'),
                }

            else:
                # --- Run Standard Core Simulation ---
                if is_wifi:
                    if not WIFI_AVAILABLE:
                        self.simulation_queue.put(("error", "WiFi 模块未安装，无法运行 WiFi 仿真。"))
                        self.simulation_queue.put(("done", None))
                        return
                    wifi_bandwidth_hz = float(params["wifi_bandwidth_mhz"]) * 1e6
                    wifi_n_fft = None
                    if params["wifi_n_fft"]:
                        try:
                            wifi_n_fft = int(params["wifi_n_fft"])
                        except ValueError:
                            wifi_n_fft = None
                    wifi_delay_scale = None
                    if params["wifi_delay_scale"]:
                        try:
                            wifi_delay_scale = float(params["wifi_delay_scale"])
                        except ValueError:
                            wifi_delay_scale = None

                    wifi_config = WiFiScenarioConfig(
                        model_name=wifi_model_name,
                        center_frequency_hz=center_frequency_hz,
                        bandwidth_hz=wifi_bandwidth_hz,
                        enable_ofdm=params["wifi_enable_ofdm"],
                        n_fft=wifi_n_fft,
                        sample_rate_hz=None,
                        random_seed=seed,
                        phase_policy=params["wifi_phase_policy"],
                        normalize_power=params["wifi_normalize_power"],
                        delay_scale=wifi_delay_scale,
                        bs_positions=bs_pos,
                        ue_positions=ue_pos,
                        ue_velocities=ue_vel_vector,
                    )
                    wifi_sim = WiFiChannelSimulator(wifi_config)
                    wifi_result = wifi_sim.run()

                    clusters = []
                    impulse_response = wifi_result.impulse_response or []
                    tap_components = wifi_result.tap_components or []
                    wavelength = c / center_frequency_hz
                    k_factor = 2 * np.pi / wavelength

                    channel_matrix = np.zeros((rx_positions.shape[0], tx_positions.shape[0]), dtype=complex)
                    for idx, tap in enumerate(tap_components):
                        coeff = impulse_response[idx]["coeff"] if idx < len(impulse_response) else np.sqrt(tap.power_linear)
                        k_hat_rx = np.array([
                            np.cos(np.deg2rad(tap.aoa_deg)),
                            np.sin(np.deg2rad(tap.aoa_deg)),
                            0.0
                        ])
                        k_hat_tx = np.array([
                            np.cos(np.deg2rad(tap.aod_deg)),
                            np.sin(np.deg2rad(tap.aod_deg)),
                            0.0
                        ])
                        a_rx = np.exp(1j * k_factor * (rx_positions @ k_hat_rx))
                        a_tx = np.exp(1j * k_factor * (tx_positions @ k_hat_tx))
                        h_tap = coeff * np.outer(a_rx, a_tx)
                        channel_matrix += h_tap

                        clusters.append(EngineCluster(
                            delay=tap.delay_s,
                            power=tap.power_linear,
                            aod_phi=tap.aod_deg,
                            aoa_phi=tap.aoa_deg,
                            zod_theta=90.0,
                            zoa_theta=90.0,
                            as_aod=tap.as_tx_deg,
                            as_aoa=tap.as_rx_deg,
                            as_zod=0.0,
                            as_zoa=0.0,
                            k_n=1.0
                        ))

                    distance = np.linalg.norm(bs_pos[0] - ue_pos[0])
                    pathloss_db = 0.0
                    if distance > 1e-9:
                        pathloss_db = 20 * np.log10(distance) + 20 * np.log10(center_frequency_hz) - 147.55

                    results = {
                        'channel_matrix': channel_matrix,
                        'lsps': None,
                        'pathloss_db': pathloss_db,
                        'is_los': None,
                        'clusters': clusters,
                    }
                elif is_custom:
                    from mimo_ota_simulator.channel_builders import create_builder
                    cc = TargetChannelConfig(
                        input_mode="custom",
                        custom_profile_path=getattr(self, "current_custom_json_path", None),
                        standard="3gpp",
                        model_name="Custom",
                    )
                    profile = create_builder(cc).build()
                    results = {
                        'channel_matrix': np.zeros((1, 1), dtype=complex), 
                        'lsps': profile.lsps,
                        'pathloss_db': profile.pathloss_db,
                        'is_los': profile.is_los,
                        'clusters': profile.clusters,
                    }
                else:
                    simulator = ChannelSimulator(scenario_name, center_frequency_hz, cluster_model_name, use_median_lsps=use_median_lsps)
                    simulator.config.bs_positions = bs_pos
                    simulator.config.ue_positions = ue_pos
                    simulator.config.ue_velocities = ue_vel_vector
                    simulator.config.antenna_config = ant_config
                
                    simulator.generator = ChannelGenerator(simulator.config, simulator.params, simulator.config.cluster_model_name, use_median_lsps=use_median_lsps)

                    results = simulator.run(force_condition=force_los_condition, time_s=time_s, logger=lambda msg: self.simulation_queue.put(("log", msg)))

            self.simulation_queue.put(("results", results))

        except Exception as e:
            import traceback
            self.simulation_queue.put(("error", f"Simulation failed: {e}\n{traceback.format_exc()}"))
        finally:
            self.simulation_queue.put(("done", None))

    def process_queue(self):
        try:
            while True:
                msg_type, data = self.simulation_queue.get_nowait()
                try:
                    if msg_type == "log":
                        self.log(data)
                    elif msg_type == "results":
                        self.last_results = data
                        self.log("\n--- Simulation Successful ---")
                        if data['is_los'] is None:
                            self.log("Final Channel Condition: N/A")
                        else:
                            self.log(f"Final Channel Condition: {'LOS' if data['is_los'] else 'NLOS'}")
                        self.log(f"Pathloss: {data['pathloss_db']:.2f} dB")
                        self.log(f"Channel Matrix H (shape: {data['channel_matrix'].shape}):\n{data['channel_matrix']}")
                        if data['lsps'] is not None:
                            self.log("\n--- Large Scale Parameters (LSPs) ---")
                            lsp_text = json.dumps(data['lsps'].model_dump(), indent=2)
                            self.log(lsp_text)
                        self._update_results_display(data['clusters'])
                        self.log("\nPlots and data tables updated.")
                    elif msg_type == "error":
                        self.log(data)
                        messagebox.showerror("Simulation Error", data)
                    elif msg_type == "done":
                        self.run_button.config(state=tk.NORMAL)
                except Exception as e:
                    self.log(f"GUI 结果处理异常: {e}")
                    self.run_button.config(state=tk.NORMAL)
        except queue.Empty:
            self.after(100, self.process_queue)
        except Exception as e:
            self.log(f"队列处理异常: {e}")
            self.run_button.config(state=tk.NORMAL)
            self.after(100, self.process_queue)

    def _update_results_display(self, clusters):
        self.clear_results()
        if not clusters: self.log("No clusters generated, cannot plot."); return
        delays = [c.delay for c in clusters]; powers_linear = np.array([c.power for c in clusters])
        if not np.any(powers_linear): self.log("All cluster powers are zero, cannot plot relative power."); return

        max_power = np.max(powers_linear)
        powers_db_relative = 10 * np.log10(powers_linear / max_power) if max_power > 0 else np.full_like(powers_linear, -np.inf)
        
        min_finite_power = powers_db_relative[np.isfinite(powers_db_relative)]
        y_min = -50.0
        if min_finite_power.size > 0: y_min = max(-80.0, np.floor(np.min(min_finite_power) / 10) * 10 - 10)
        y_max = 5.0

        self.ax_pdp.stem(delays, powers_db_relative, bottom=y_min); self.ax_pdp.set_ylim(bottom=y_min, top=y_max)
        self.ax_aoa.stem([c.aoa_phi for c in clusters], powers_db_relative, bottom=y_min); self.ax_aoa.set_ylim(bottom=y_min, top=y_max)
        self.ax_zoa.stem([c.zoa_theta for c in clusters], powers_db_relative, bottom=y_min); self.ax_zoa.set_ylim(bottom=y_min, top=y_max)
        for fig in [self.fig_pdp, self.fig_aoa, self.fig_zoa]: fig.tight_layout()
        for canvas in [self.canvas_pdp, self.canvas_aoa, self.canvas_zoa]: canvas.draw()

        for i, cluster in enumerate(clusters):
            self.pdp_tree.insert("", "end", values=(i+1, f"{cluster.delay:.3e}", f"{powers_db_relative[i]:.2f}"))
            self.aoa_tree.insert("", "end", values=(i+1, f"{cluster.aoa_phi:.2f}", f"{powers_db_relative[i]:.2f}"))
            self.zoa_tree.insert("", "end", values=(i+1, f"{cluster.zoa_theta:.2f}", f"{powers_db_relative[i]:.2f}"))
            self.cdl_tree.insert("", "end", values=(i+1, f"{cluster.delay:.3e}", f"{powers_db_relative[i]:.2f}", f"{cluster.aoa_phi:.2f}", f"{cluster.zoa_theta:.2f}", f"{cluster.aod_phi:.2f}", f"{cluster.zod_theta:.2f}", f"{10*np.log10(cluster.k_n):.2f}"))

    def copy_tree_data(self, tree):
        headings = tree["columns"]
        text_to_copy = "\t".join(headings) + "\n"
        for item_id in tree.get_children():
            values = tree.item(item_id, "values")
            text_to_copy += "\t".join(map(str, values)) + "\n"
        self.clipboard_clear(); self.clipboard_append(text_to_copy)
        self.log(f"Table data for '{tree.heading(headings[0])['text']}' copied to clipboard.")

    def save_config(self):
        config_data = {key: control['var'].get() for key, control in self.controls.items()}
        from tkinter.filedialog import asksaveasfilename
        filepath = asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json"), ("All files", "*.*")], title="Save Simulation Configuration")
        if not filepath: return
        with open(filepath, 'w') as f: json.dump(config_data, f, indent=4)
        self.log(f"Configuration saved to {filepath}")

    def load_config(self):
        from tkinter.filedialog import askopenfilename
        filepath = askopenfilename(filetypes=[("JSON files", "*.json"), ("All files", "*.*")], title="Load Simulation Configuration")
        if not filepath: return
        with open(filepath, 'r') as f: config_data = json.load(f)
        for key, value in config_data.items():
            if key in self.controls:
                self.controls[key]['var'].set(value)
        self.log(f"Configuration loaded from {filepath}")

    def save_results(self):
        if not self.last_results:
            messagebox.showinfo("Info", "No simulation results to save. Please run a simulation first.")
            return

        from tkinter.filedialog import asksaveasfilename
        filepath = asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json"), ("All files", "*.*")], title="Save Simulation Results")
        if not filepath: return

        try:
            # Prepare data for export
            lsps = self.last_results.get('lsps')
            export_data = {
                "timestamp": str(np.datetime64('now')),
                "lsps": lsps.model_dump() if lsps is not None else None,
                "pathloss_db": self.last_results['pathloss_db'],
                "is_los": self.last_results['is_los'],
                "clusters": [
                    {
                        "delay": c.delay,
                        "power": c.power,
                        "aoa_phi": c.aoa_phi,
                        "zoa_theta": c.zoa_theta,
                        "aod_phi": c.aod_phi,
                        "zod_theta": c.zod_theta,
                        "xpr_db": 10 * np.log10(c.k_n) if c.k_n else 0.0,
                        "k_n": c.k_n
                    }
                    for c in self.last_results['clusters']
                ]
            }
            
            # Helper to handle numpy types
            def np_encoder(object):
                if isinstance(object, np.generic):
                    return object.item()
                raise TypeError

            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=4, default=np_encoder)
            self.log(f"Results saved to {filepath}")
        except Exception as e:
            self.log(f"Error saving results: {e}")
            messagebox.showerror("Error", f"Failed to save results: {e}")

    def export_fpga_sos_csv(self):
        """Re-instantiates the simulator with current config and exports FPGA SoS parameters to CSV."""
        from tkinter.filedialog import asksaveasfilename
        import csv

        filepath = asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")], title="Export FPGA SoS Parameters")
        if not filepath: return

        try:
            self.log("Generating FPGA parameters...")
            # Re-create simulator logic (copied from run_simulation)
            params = {key: control['var'].get() for key, control in self.controls.items()}
            if params["standard"] == "WiFi":
                self.log("WiFi 模式不支持 FPGA SoS 参数导出。")
                return
            scenario_name = params["scenario_name"]
            cluster_model_name = params["cluster_model_name"]
            if cluster_model_name == 'Stochastic': cluster_model_name = None

            center_frequency_hz = float(params["center_frequency_ghz"]) * 1e9
            bs_pos = np.array([[float(p) for p in params["tx_position"].split(',')]])
            ue_pos = np.array([[float(p) for p in params["rx_position"].split(',')]])
            ue_velocity_kph = float(params["ue_velocity_kph"])
            ue_velocity_mps = ue_velocity_kph * 1000 / 3600
            ue_vel_vector = np.array([[ue_velocity_mps, 0, 0]])
            seed_str = params["random_seed"]
            seed = int(seed_str) if seed_str else None
            
            # Setup Antennas
            c = 3e8
            lambda_meters = c / center_frequency_hz
            tx_positions = get_antenna_positions(
                array_type=params['tx_array_type'], num_rows=int(params['tx_num_rows']),
                num_cols=int(params['tx_num_cols']), spacing_h_m=float(params['tx_spacing_h']) * lambda_meters,
                spacing_v_m=float(params['tx_spacing_v']) * lambda_meters
            )
            rx_positions = get_antenna_positions(
                array_type=params['rx_array_type'], num_rows=int(params['rx_num_rows']),
                num_cols=int(params['rx_num_cols']), spacing_h_m=float(params['rx_spacing_h']) * lambda_meters,
                spacing_v_m=float(params['rx_spacing_v']) * lambda_meters
            )
            ant_config = AntennaArray(tx_array_type=params['tx_array_type'], num_tx_rows=int(params['tx_num_rows']), num_tx_cols=int(params['tx_num_cols']), tx_spacing_h=float(params['tx_spacing_h']), tx_spacing_v=float(params['tx_spacing_v']), tx_element_pattern=params['tx_element_pattern'], tx_positions=tx_positions, rx_array_type=params['rx_array_type'], num_rx_rows=int(params['rx_num_rows']), num_rx_cols=int(params['rx_num_cols']), rx_spacing_h=float(params['rx_spacing_h']), rx_spacing_v=float(params['rx_spacing_v']), rx_element_pattern=params['rx_element_pattern'], rx_positions=rx_positions)

            # Initialize Simulator
            np.random.seed(seed)
            simulator = ChannelSimulator(scenario_name, center_frequency_hz, cluster_model_name, use_median_lsps=params["use_median_lsps"])
            simulator.config.bs_positions = bs_pos
            simulator.config.ue_positions = ue_pos
            simulator.config.ue_velocities = ue_vel_vector
            simulator.config.antenna_config = ant_config
            simulator.generator = ChannelGenerator(simulator.config, simulator.params, simulator.config.cluster_model_name, use_median_lsps=params["use_median_lsps"])

            # Export
            sos_params = simulator.export_fpga_sos_params()
            
            # Write to CSV
            if not sos_params:
                self.log("No parameters generated.")
                return

            keys = sos_params[0].keys()
            with open(filepath, 'w', newline='') as output_file:
                dict_writer = csv.DictWriter(output_file, fieldnames=keys)
                dict_writer.writeheader()
                dict_writer.writerows(sos_params)
            
            self.log(f"FPGA parameters exported to {filepath} ({len(sos_params)} rows)")
            messagebox.showinfo("Success", f"Successfully exported {len(sos_params)} rows to:\n{filepath}")

        except Exception as e:
            import traceback
            self.log(f"Error exporting FPGA params: {e}\n{traceback.format_exc()}")
            messagebox.showerror("Export Error", f"Failed to export: {e}")

    def export_to_propsim(self):
        """Exports the current simulation result to Keysight Propsim F64 format."""
        if not self.last_results:
            messagebox.showinfo("Info", "Please Run Simulation first to generate channel data.")
            return
            
        # Check if we are in OTA mode (Propsim only makes sense for OTA probe weights)
        # However, self.last_results doesn't explicitly store 'mode'. 
        # But 'transmit_mapping_matrix_A' key presence is a good indicator of OTA results.
        if 'transmit_mapping_matrix_A' not in self.last_results:
             messagebox.showwarning("Warning", "Current results are not from MIMO OTA mode.\n\nPlease enable 'MIMO OTA' and run simulation again.")
             return

        from tkinter.filedialog import asksaveasfilename
        filepath = asksaveasfilename(defaultextension=".asc", filetypes=[("ASCII Channel", "*.asc"), ("All files", "*.*")], title="Export to Propsim F64")
        if not filepath: return

        try:
            params = {key: control['var'].get() for key, control in self.controls.items()}
            
            # 1. Prepare Export Config
            mode_str = params['f64_mode'].split()[0] # Extract 'A' or 'B'
            
            p_config = PropsimExportConfig(
                filename=filepath,
                mode=mode_str,
                duration_s=float(params['f64_duration_s']),
                sample_rate_hz=float(params['f64_sample_rate_hz']),
                target_output_power_dbm=-10.0 # Fixed for now or add GUI field
            )
            
            # 2. Calibration
            cal_config = self.current_cal_config
            if cal_config:
                self.log(f"Using Loaded Calibration ({len(cal_config.entries)} ports).")
            else:
                self.log("Using Ideal Calibration (No Cable/Probe compensation).")
            
            # 3. Export
            exporter = PropsimASCIIExporter(p_config, calibration_config=cal_config)
            
            # We need to pass the FULL simulator result dictionary
            # But wait, self.last_results in 'run_simulation' (OTA branch) was filtered!
            # Look at gui.py run_simulation:
            # results = {'channel_matrix':..., 'lsps':..., ...}
            # IT DOES NOT INCLUDE 'transmit_mapping_matrix_A'!
            # THIS IS A BUG/GAP. We need to pass the raw OTA weights to the GUI result.
            
            # Let's fix run_simulation first, but I can't interrupt this replace block.
            # I will assume I fix run_simulation in the next step.
            # For now, I'll write the logic assuming 'transmit_mapping_matrix_A' IS there.
            
            out_files = exporter.export(self.last_results, output_path=filepath, direction="dl")
            export_ul = params.get("f64_export_ul", False)
            if export_ul:
                if self.last_results.get("channel_impulse_response_ul") is None:
                    self.log("UL 数据不存在，请启用 UL/DL 双向并重新仿真。")
                else:
                    ul_path = os.path.splitext(filepath)[0] + "_ul.asc"
                    out_files += exporter.export(self.last_results, output_path=ul_path, direction="ul")
            
            # Zip the generated files
            zip_filename = os.path.splitext(filepath)[0] + ".zip"
            with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zf:
                for fpath in out_files:
                    if os.path.exists(fpath):
                        zf.write(fpath, os.path.basename(fpath))
                        # Remove the original .asc file to keep directory clean
                        os.remove(fpath)
            
            self.log(f"Successfully exported {len(out_files)} files and compressed to {zip_filename}")
            messagebox.showinfo("Success", f"Export Complete.\nMode: {mode_str}\n\nGenerated {len(out_files)} channel files.\nCompressed into: {zip_filename}")
            
        except Exception as e:
            import traceback
            self.log(f"Propsim Export Failed: {e}\n{traceback.format_exc()}")
            messagebox.showerror("Export Error", f"Failed: {e}")

if __name__ == '__main__':
    app = ChannelModelApp()
    app.mainloop()
