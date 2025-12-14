import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import json
import threading
import queue

from channel_model_38901.simulator import ChannelSimulator
from channel_model_38901.channel_generator import ChannelGenerator
from channel_model_38901.common.data_models import AntennaArray, EngineCluster
from channel_model_38901.common.geometry import get_antenna_positions

# Try importing MIMO OTA Simulator (Optional extension)
try:
    from mimo_ota_simulator.simulator import MIMO_OTA_Simulator
    from mimo_ota_simulator.data_models import ChamberConfig, TargetChannelConfig, AntennaArrayConfig
    MIMO_OTA_AVAILABLE = True
except ImportError:
    MIMO_OTA_AVAILABLE = False
    print("Warning: mimo_ota_simulator not found. OTA features disabled.")

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
        top_frame = ttk.Frame(self, padding="10")
        top_frame.pack(fill=tk.X, side=tk.TOP)

        param_frame = ttk.Frame(top_frame)
        param_frame.pack(fill=tk.X)
        param_frame.grid_columnconfigure(0, weight=1)
        param_frame.grid_columnconfigure(1, weight=1)
        param_frame.grid_columnconfigure(2, weight=1)

        general_param_frame = ttk.LabelFrame(param_frame, text="通用参数", padding="10")
        general_param_frame.grid(row=0, column=0, padx=5, pady=5, sticky=tk.NSEW)

        ota_param_frame = ttk.LabelFrame(param_frame, text="MIMO OTA 参数 (可选)", padding="10")
        ota_param_frame.grid(row=1, column=0, padx=5, pady=5, sticky=tk.NSEW)

        tx_param_frame = ttk.LabelFrame(param_frame, text="发射端 (Tx) 参数", padding="10")
        tx_param_frame.grid(row=0, column=1, rowspan=2, padx=5, pady=5, sticky=tk.NSEW)

        rx_param_frame = ttk.LabelFrame(param_frame, text="接收端 (Rx) 参数", padding="10")
        rx_param_frame.grid(row=0, column=2, rowspan=2, padx=5, pady=5, sticky=tk.NSEW)

        self._create_parameter_widgets(general_param_frame, ota_param_frame, tx_param_frame, rx_param_frame)
        self._create_control_buttons(top_frame)
        
        bottom_frame = ttk.Frame(self, padding="10")
        bottom_frame.pack(fill=tk.BOTH, expand=True, side=tk.BOTTOM)

        self.notebook = ttk.Notebook(bottom_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self._create_log_display()
        self._create_results_tabs()
        
        self.update_antenna_ui('tx')
        self.update_antenna_ui('rx')

    def _create_parameter_widgets(self, general_frame, ota_frame, tx_frame, rx_frame):
        environments = ['UMa', 'UMi-StreetCanyon', 'UMi-OpenArea', 'RMa', 'InH-Office', 'SMa', 'InF']
        cluster_models = ['Stochastic', 'CDL-A', 'CDL-B', 'CDL-C', 'CDL-D', 'CDL-E', 'SCME']
        
        general_params = {
            "scenario_name": ("大尺度环境:", "UMa", environments),
            "cluster_model_name": ("小尺度簇模型:", "Stochastic", cluster_models),
            "center_frequency_ghz": ("中心频率 (GHz):", "3.5", None),
            "los_condition": ("LOS 条件:", "Automatic", ["Automatic", "LOS", "NLOS"]),
            "use_median_lsps": ("使用LSP中值:", False, [True, False]),
            "random_seed": ("随机种子:", "", None),
            "simulation_time_s": ("仿真时间 (s):", "0.0", None)
        }
        for i, (key, (text, default, values)) in enumerate(general_params.items()):
            self._create_entry(general_frame, key, text, default, values, i)

        # OTA Parameters
        self.ota_vars = {}
        if MIMO_OTA_AVAILABLE:
            ota_enable = self._create_entry(ota_frame, "ota_enable", "启用 MIMO OTA:", False, [True, False], 0)
            ota_enable.trace_add('write', lambda *args: self.update_ota_ui())
            
            self._create_entry(ota_frame, "ota_num_probes", "探头数量:", "16", None, 1)
            self._create_entry(ota_frame, "ota_radius_m", "探头半径 (m):", "1.5", None, 2)
            self._create_entry(ota_frame, "ota_method", "合成方法:", "ray", ["ray", "cluster", "pinv"], 3)
            self._create_entry(ota_frame, "ota_jitter", "相位抖动 (Jitter):", False, [True, False], 4)
        else:
            ttk.Label(ota_frame, text="MIMO OTA 库未安装").pack()

        self._create_antenna_widgets('tx', tx_frame)
        self._create_antenna_widgets('rx', rx_frame)
        
        if MIMO_OTA_AVAILABLE:
            self.update_ota_ui()

    def update_ota_ui(self):
        if not MIMO_OTA_AVAILABLE: return
        is_enabled = self.controls['ota_enable']['var'].get()
        state = tk.NORMAL if is_enabled else tk.DISABLED
        for key in ["ota_num_probes", "ota_radius_m", "ota_method", "ota_jitter"]:
            self.controls[key]['label'].config(state=state)
            self.controls[key]['widget'].config(state=state)

    def _create_antenna_widgets(self, prefix, parent_frame):
        self._create_entry(parent_frame, f'{prefix}_position', "Position (x,y,z):", "0,0,25" if prefix == 'tx' else "100,0,1.5", None, 0)
        array_type_var = self._create_entry(parent_frame, f'{prefix}_array_type', "Array Type:", 'ULA', ['ULA', 'URA'], 1)
        array_type_var.trace_add('write', lambda *args, p=prefix: self.update_antenna_ui(p))
        self._create_entry(parent_frame, f'{prefix}_num_rows', "Num Rows:", "1", None, 2)
        self._create_entry(parent_frame, f'{prefix}_num_cols', "Num Cols (ULA: Num Ant):", "2", None, 3)
        self._create_entry(parent_frame, f'{prefix}_spacing_h', "Spacing H (lambda):", "0.5", None, 4)
        self._create_entry(parent_frame, f'{prefix}_spacing_v', "Spacing V (lambda):", "0.5", None, 5)
        self._create_entry(parent_frame, f'{prefix}_element_pattern', "Element Pattern:", 'Isotropic', ['Isotropic', 'TR38.901'], 6)
        if prefix == 'rx':
            self._create_entry(parent_frame, 'ue_velocity_kph', "UE Velocity (km/h):", "3", None, 7)

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
        control_frame = ttk.Frame(parent_frame)
        control_frame.pack(fill=tk.X, pady=5)
        self.run_button = ttk.Button(control_frame, text="Run Simulation", command=self.start_simulation_thread)
        self.run_button.pack(side=tk.LEFT, padx=5)
        self.save_button = ttk.Button(control_frame, text="Save Config", command=self.save_config)
        self.save_button.pack(side=tk.LEFT, padx=5)
        self.load_button = ttk.Button(control_frame, text="Load Config", command=self.load_config)
        self.load_button.pack(side=tk.LEFT, padx=5)
        self.save_results_button = ttk.Button(control_frame, text="Save Results", command=self.save_results)
        self.save_results_button.pack(side=tk.LEFT, padx=5)
        self.export_fpga_button = ttk.Button(control_frame, text="Export FPGA Params", command=self.export_fpga_sos_csv)
        self.export_fpga_button.pack(side=tk.LEFT, padx=5)

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
        self.run_button.config(state=tk.DISABLED)
        self.log_text.config(state=tk.NORMAL); self.log_text.delete(1.0, tk.END); self.log_text.config(state=tk.DISABLED)
        self.clear_results()
        thread = threading.Thread(target=self.run_simulation)
        thread.daemon = True
        thread.start()

    def run_simulation(self):
        try:
            params = {key: control['var'].get() for key, control in self.controls.items()}
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
            los_condition_str = params["los_condition"]
            force_los_condition = None if los_condition_str == "Automatic" else los_condition_str
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
            
            log_msg = f"Starting simulation for Environment: {scenario_name}, Cluster Model: {cluster_model_name if cluster_model_name else 'Stochastic'}"
            if ota_enabled:
                log_msg += f" [MIMO OTA Mode: {params['ota_method']}, Probes: {params['ota_num_probes']}]"
            self.simulation_queue.put(("log", log_msg))

            np.random.seed(seed)

            if ota_enabled:
                # --- Run MIMO OTA Simulation ---
                simulator = MIMO_OTA_Simulator() # This is the OTA simulator facade
                
                # Build OTA Configs
                chamber_config = ChamberConfig(
                    num_probes=int(params["ota_num_probes"]),
                    radius_m=float(params["ota_radius_m"]),
                    distribution='ring',
                    dual_polarized=False # GUI doesn't support dual-pol probe selection yet
                )
                
                # Map antenna config to OTA format
                # Note: MIMO_OTA_Simulator expects specific AntennaArrayConfig objects or dicts
                # We'll use the dict/object mapping as defined in mimo_ota_simulator/data_models.py
                
                tx_ant_ota = AntennaArrayConfig(
                    num_rows=int(params['tx_num_rows']), num_cols=int(params['tx_num_cols']),
                    spacing_h=float(params['tx_spacing_h']), spacing_v=float(params['tx_spacing_v'])
                    # pattern handled internally or default
                )
                rx_ant_ota = AntennaArrayConfig(
                    num_rows=int(params['rx_num_rows']), num_cols=int(params['rx_num_cols']),
                    spacing_h=float(params['rx_spacing_h']), spacing_v=float(params['rx_spacing_v'])
                )

                channel_config = TargetChannelConfig(
                    model_name=scenario_name,
                    cluster_model_name=cluster_model_name,
                    center_frequency_hz=center_frequency_hz,
                    random_seed=seed,
                    tx_antenna=tx_ant_ota,
                    rx_antenna=rx_ant_ota,
                    force_condition=force_los_condition,
                    ue_velocity_kmh=ue_velocity_kph,
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
                # OTA returns 'effective_H_matrix', 'lsps', 'clusters', 'pathloss_db', 'is_los'
                results = {
                    'channel_matrix': ota_results['effective_H_matrix'],
                    'lsps': ota_results['lsps'],
                    'pathloss_db': ota_results['pathloss_db'],
                    'is_los': ota_results['is_los'],
                    'clusters': ota_results['clusters']
                }

            else:
                # --- Run Standard Core Simulation ---
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
                if msg_type == "log": self.log(data)
                elif msg_type == "results":
                    self.last_results = data
                    self.log("\n--- Simulation Successful ---")
                    self.log(f"Final Channel Condition: {'LOS' if data['is_los'] else 'NLOS'}")
                    self.log(f"Pathloss: {data['pathloss_db']:.2f} dB")
                    self.log(f"Channel Matrix H (shape: {data['channel_matrix'].shape}):\n{data['channel_matrix']}")
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
        except queue.Empty:
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
            export_data = {
                "timestamp": str(np.datetime64('now')),
                "lsps": self.last_results['lsps'].model_dump(),
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

    def process_queue(self):
        try:
            while True:
                msg_type, data = self.simulation_queue.get_nowait()
                if msg_type == "log": self.log(data)
                elif msg_type == "results":
                    self.last_results = data
                    self.log("\n--- Simulation Successful ---")
                    self.log(f"Final Channel Condition: {'LOS' if data['is_los'] else 'NLOS'}")
                    self.log(f"Pathloss: {data['pathloss_db']:.2f} dB")
                    self.log(f"Channel Matrix H (shape: {data['channel_matrix'].shape}):\n{data['channel_matrix']}")
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
        except queue.Empty:
            self.after(100, self.process_queue)

if __name__ == '__main__':
    app = ChannelModelApp()
    app.mainloop()