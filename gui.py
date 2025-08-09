import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import json
import threading
import queue

from channel_model_38901.simulator import ChannelSimulator
from channel_model_38901.channel_generator import ChannelGenerator
from channel_model_38901.common.data_models import AntennaArray

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class ChannelModelApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("3GPP TR 38.901 Channel Simulator")
        self.geometry("1200x900")

        self.simulation_queue = queue.Queue()
        self.after(100, self.process_queue)

        self._create_widgets()

    def _create_widgets(self):
        top_frame = ttk.Frame(self, padding="10")
        top_frame.pack(fill=tk.X, side=tk.TOP)

        self.param_frame = ttk.Frame(top_frame)
        self.param_frame.pack(fill=tk.X)
        
        self.general_param_frame = ttk.LabelFrame(self.param_frame, text="通用参数", padding="10")
        self.general_param_frame.grid(row=0, column=0, padx=5, pady=5, sticky=tk.NSEW)

        self.tx_param_frame = ttk.LabelFrame(self.param_frame, text="发射端 (Tx) 参数", padding="10")
        self.tx_param_frame.grid(row=0, column=1, padx=5, pady=5, sticky=tk.NSEW)

        self.rx_param_frame = ttk.LabelFrame(self.param_frame, text="接收端 (Rx) 参数", padding="10")
        self.rx_param_frame.grid(row=0, column=2, padx=5, pady=5, sticky=tk.NSEW)
        
        self.param_frame.grid_columnconfigure(0, weight=1)
        self.param_frame.grid_columnconfigure(1, weight=1)
        self.param_frame.grid_columnconfigure(2, weight=1)

        self._create_parameter_widgets()
        self._create_control_buttons(top_frame)
        
        bottom_frame = ttk.Frame(self, padding="10")
        bottom_frame.pack(fill=tk.BOTH, expand=True, side=tk.BOTTOM)

        self.notebook = ttk.Notebook(bottom_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self._create_log_display()
        self._create_plots()

    def _create_parameter_widgets(self):
        self.entries = {}
        all_scenarios = ['UMa', 'UMi-StreetCanyon', 'UMi-OpenArea', 'RMa', 'InH-Office', 'SMa', 'InF']
        general_params = {
            "scenario_name": ("Scenario:", "UMa", all_scenarios),
            "center_frequency_ghz": ("Center Freq (GHz):", "3.5", None),
            "los_condition": ("LOS Condition:", "Automatic", ["Automatic", "LOS", "NLOS"]),
            "random_seed": ("Random Seed:", "", None)
        }
        for i, (key, (text, default, values)) in enumerate(general_params.items()):
            label = ttk.Label(self.general_param_frame, text=text)
            label.grid(row=i, column=0, sticky=tk.W, pady=2)
            var = tk.StringVar(value=default)
            if values:
                widget = ttk.Combobox(self.general_param_frame, textvariable=var, values=values, width=18)
            else:
                widget = ttk.Entry(self.general_param_frame, textvariable=var, width=20)
            widget.grid(row=i, column=1, sticky=tk.EW, pady=2, padx=5)
            self.entries[key] = var
        self.general_param_frame.grid_columnconfigure(1, weight=1)

        tx_params = {
            "bs_position": ("BS Position (x,y,z):", "0,0,25", None),
            "num_tx_antennas": ("Num Antennas:", "2", None),
            "tx_element_spacing": ("Spacing (lambda):", "0.5", None),
            "tx_element_pattern": ("Element Pattern:", "Isotropic", ["Isotropic", "TR38.901"])
        }
        for i, (key, (text, default, values)) in enumerate(tx_params.items()):
            label = ttk.Label(self.tx_param_frame, text=text)
            label.grid(row=i, column=0, sticky=tk.W, pady=2)
            var = tk.StringVar(value=default)
            if values:
                widget = ttk.Combobox(self.tx_param_frame, textvariable=var, values=values, width=18)
            else:
                widget = ttk.Entry(self.tx_param_frame, textvariable=var, width=20)
            widget.grid(row=i, column=1, sticky=tk.EW, pady=2, padx=5)
            self.entries[key] = var
        self.tx_param_frame.grid_columnconfigure(1, weight=1)

        rx_params = {
            "ue_position": ("UE Position (x,y,z):", "100,0,1.5", None),
            "num_rx_antennas": ("Num Antennas:", "2", None),
            "rx_element_spacing": ("Spacing (lambda):", "0.5", None),
            "rx_element_pattern": ("Element Pattern:", "Isotropic", ["Isotropic", "TR38.901"]),
            "ue_velocity_kph": ("UE Velocity (km/h):", "3", None)
        }
        for i, (key, (text, default, values)) in enumerate(rx_params.items()):
            label = ttk.Label(self.rx_param_frame, text=text)
            label.grid(row=i, column=0, sticky=tk.W, pady=2)
            var = tk.StringVar(value=default)
            if values:
                widget = ttk.Combobox(self.rx_param_frame, textvariable=var, values=values, width=18)
            else:
                widget = ttk.Entry(self.rx_param_frame, textvariable=var, width=20)
            widget.grid(row=i, column=1, sticky=tk.EW, pady=2, padx=5)
            self.entries[key] = var
        self.rx_param_frame.grid_columnconfigure(1, weight=1)

    def _create_control_buttons(self, parent_frame):
        control_frame = ttk.Frame(parent_frame)
        control_frame.pack(fill=tk.X, pady=5)
        self.run_button = ttk.Button(control_frame, text="Run Simulation", command=self.start_simulation_thread)
        self.run_button.pack(side=tk.LEFT, padx=5)
        self.save_button = ttk.Button(control_frame, text="Save Config", command=self.save_config)
        self.save_button.pack(side=tk.LEFT, padx=5)
        self.load_button = ttk.Button(control_frame, text="Load Config", command=self.load_config)
        self.load_button.pack(side=tk.LEFT, padx=5)

    def _create_log_display(self):
        log_frame = ttk.Frame(self.notebook)
        self.log_text = tk.Text(log_frame, height=10, wrap=tk.WORD, state=tk.DISABLED, font=("Courier New", 10))
        scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.config(yscrollcommand=scrollbar.set)
        scrollbar.pack(fill=tk.Y, side=tk.RIGHT)
        self.log_text.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        self.notebook.add(log_frame, text='Log')

    def _create_plots(self):
        # PDP Tab
        pdp_frame = ttk.Frame(self.notebook)
        self.notebook.add(pdp_frame, text='PDP')
        pdp_frame.grid_rowconfigure(0, weight=1)
        pdp_frame.grid_columnconfigure(0, weight=1)
        self.fig_pdp, self.ax_pdp = plt.subplots(1, 1)
        self.canvas_pdp = FigureCanvasTkAgg(self.fig_pdp, master=pdp_frame)
        self.canvas_pdp.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.pdp_tree, pdp_copy_btn = self._create_plot_table(pdp_frame, ["Cluster #", "Delay (s)", "Power (dB)"])
        self.pdp_tree.grid(row=1, column=0, sticky="ew", pady=(5, 0))
        pdp_copy_btn.grid(row=2, column=0, sticky="e", pady=5, padx=5)

        # AoA Tab
        aoa_frame = ttk.Frame(self.notebook)
        self.notebook.add(aoa_frame, text='AoA')
        aoa_frame.grid_rowconfigure(0, weight=1)
        aoa_frame.grid_columnconfigure(0, weight=1)
        self.fig_aoa, self.ax_aoa = plt.subplots(1, 1)
        self.canvas_aoa = FigureCanvasTkAgg(self.fig_aoa, master=aoa_frame)
        self.canvas_aoa.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.aoa_tree, aoa_copy_btn = self._create_plot_table(aoa_frame, ["Cluster #", "AoA (deg)", "Power (dB)"])
        self.aoa_tree.grid(row=1, column=0, sticky="ew", pady=(5, 0))
        aoa_copy_btn.grid(row=2, column=0, sticky="e", pady=5, padx=5)

        # ZoA Tab
        zoa_frame = ttk.Frame(self.notebook)
        self.notebook.add(zoa_frame, text='ZoA')
        zoa_frame.grid_rowconfigure(0, weight=1)
        zoa_frame.grid_columnconfigure(0, weight=1)
        self.fig_zoa, self.ax_zoa = plt.subplots(1, 1)
        self.canvas_zoa = FigureCanvasTkAgg(self.fig_zoa, master=zoa_frame)
        self.canvas_zoa.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.zoa_tree, zoa_copy_btn = self._create_plot_table(zoa_frame, ["Cluster #", "ZoA (deg)", "Power (dB)"])
        self.zoa_tree.grid(row=1, column=0, sticky="ew", pady=(5, 0))
        zoa_copy_btn.grid(row=2, column=0, sticky="e", pady=5, padx=5)

        self.clear_plots()

    def _create_plot_table(self, parent, columns):
        tree = ttk.Treeview(parent, columns=columns, show='headings', height=4)
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, anchor=tk.CENTER, width=150)
        copy_btn = ttk.Button(parent, text="Copy Data", command=lambda t=tree: self.copy_tree_data(t))
        return tree, copy_btn

    def clear_plots(self):
        for ax, title, xlabel in [(self.ax_pdp, "Power Delay Profile (PDP)", "Delay (s)"),
                                  (self.ax_aoa, "Azimuth of Arrival (AoA) vs. Power", "Angle (degrees)"),
                                  (self.ax_zoa, "Zenith of Arrival (ZoA) vs. Power", "Angle (degrees)")]:
            ax.clear()
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Relative Power (dB)")
            ax.grid(True)

        for fig in [self.fig_pdp, self.fig_aoa, self.fig_zoa]:
            fig.tight_layout()
        for canvas in [self.canvas_pdp, self.canvas_aoa, self.canvas_zoa]:
            canvas.draw()

        for tree in [self.pdp_tree, self.aoa_tree, self.zoa_tree]:
            for item in tree.get_children():
                tree.delete(item)

    def log(self, message):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def start_simulation_thread(self):
        self.run_button.config(state=tk.DISABLED)
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state=tk.DISABLED)
        self.clear_plots()
        thread = threading.Thread(target=self.run_simulation)
        thread.daemon = True
        thread.start()

    def run_simulation(self):
        try:
            params = {key: var.get() for key, var in self.entries.items()}
            scenario_name = params["scenario_name"]
            center_frequency_hz = float(params["center_frequency_ghz"]) * 1e9
            bs_pos = np.array([[float(p) for p in params["bs_position"].split(',')]])
            ue_pos = np.array([[float(p) for p in params["ue_position"].split(',')]])
            ue_velocity_kph = float(params["ue_velocity_kph"])
            ue_velocity_mps = ue_velocity_kph * 1000 / 3600
            ue_vel_vector = np.array([[ue_velocity_mps, 0, 0]])
            seed_str = params["random_seed"]
            seed = int(seed_str) if seed_str else None
            los_condition_str = params["los_condition"]
            force_los_condition = None if los_condition_str == "Automatic" else los_condition_str

            c = 3e8
            lambda_meters = c / center_frequency_hz
            num_tx = int(params["num_tx_antennas"])
            tx_spacing_m = float(params["tx_element_spacing"]) * lambda_meters
            tx_positions = np.array([[0, i * tx_spacing_m, 0] for i in range(num_tx)])
            num_rx = int(params["num_rx_antennas"])
            rx_spacing_m = float(params["rx_element_spacing"]) * lambda_meters
            rx_positions = np.array([[0, i * rx_spacing_m, 0] for i in range(num_rx)])

            ant_config = AntennaArray(
                num_tx_antennas=num_tx,
                tx_element_spacing=float(params["tx_element_spacing"]),
                tx_element_pattern=params["tx_element_pattern"],
                tx_positions=tx_positions,
                num_rx_antennas=num_rx,
                rx_element_spacing=float(params["rx_element_spacing"]),
                rx_element_pattern=params["rx_element_pattern"],
                rx_positions=rx_positions
            )

            log_msg = f"Starting simulation for {scenario_name}..."
            if force_los_condition:
                log_msg += f" (Condition Forced to: {force_los_condition})"
            elif seed is not None:
                log_msg += f" (Condition Determined by Seed: {seed})"
            else:
                log_msg += f" (Condition is Probabilistic)"
            self.simulation_queue.put(("log", log_msg))

            np.random.seed(seed)
            simulator = ChannelSimulator(scenario_name, center_frequency_hz)
            simulator.config.bs_positions = bs_pos
            simulator.config.ue_positions = ue_pos
            simulator.config.ue_velocities = ue_vel_vector
            simulator.config.antenna_config = ant_config
            simulator.generator = ChannelGenerator(simulator.config, simulator.params)
            results = simulator.run(force_condition=force_los_condition)
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
                if msg_type == "log":
                    self.log(data)
                elif msg_type == "results":
                    self.log("\n--- Simulation Successful ---")
                    self.log(f"Final Channel Condition: {'LOS' if data['is_los'] else 'NLOS'}")
                    self.log(f"Pathloss: {data['pathloss_db']:.2f} dB")
                    self.log(f"Channel Matrix H (shape: {data['channel_matrix'].shape}):\n{data['channel_matrix']}")
                    self.log("\n--- Large Scale Parameters (LSPs) ---")
                    lsp_text = json.dumps(data['lsps'].model_dump(), indent=2)
                    self.log(lsp_text)
                    self._update_plots(data['clusters'])
                    self.log("\nPlots and data tables updated.")
                elif msg_type == "error":
                    self.log(data)
                    messagebox.showerror("Simulation Error", data)
                elif msg_type == "done":
                    self.run_button.config(state=tk.NORMAL)
        except queue.Empty:
            self.after(100, self.process_queue)

    def _update_plots(self, clusters):
        self.clear_plots()
        if not clusters:
            self.log("No clusters generated, cannot plot.")
            return

        delays = [c['delay'] for c in clusters]
        powers_linear = np.array([c['power'] for c in clusters])

        if not np.any(powers_linear):
            self.log("All cluster powers are zero, cannot plot relative power.")
            return

        max_power = np.max(powers_linear)
        powers_db_relative = 10 * np.log10(powers_linear / max_power) if max_power > 0 else np.full_like(powers_linear, -np.inf)

        min_finite_power = powers_db_relative[np.isfinite(powers_db_relative)]
        y_min = -50.0
        if min_finite_power.size > 0:
            y_min = max(-80.0, np.floor(np.min(min_finite_power) / 10) * 10 - 10)
        y_max = 5.0

        aoa_azimuth = [c['aoa_phi'] for c in clusters]
        aoa_zenith = [c['zoa_theta'] for c in clusters]

        # PDP Plot and Table
        self.ax_pdp.stem(delays, powers_db_relative, bottom=y_min)
        self.ax_pdp.set_ylim(bottom=y_min, top=y_max)
        for i, cluster in enumerate(clusters):
            self.pdp_tree.insert("", "end", values=(i + 1, f"{cluster['delay']:.3e}", f"{powers_db_relative[i]:.2f}"))
        self.canvas_pdp.draw()

        # AoA Plot and Table
        self.ax_aoa.stem(aoa_azimuth, powers_db_relative, bottom=y_min)
        self.ax_aoa.set_ylim(bottom=y_min, top=y_max)
        for i, cluster in enumerate(clusters):
            self.aoa_tree.insert("", "end", values=(i + 1, f"{cluster['aoa_phi']:.2f}", f"{powers_db_relative[i]:.2f}"))
        self.canvas_aoa.draw()

        # ZoA Plot and Table
        self.ax_zoa.stem(aoa_zenith, powers_db_relative, bottom=y_min)
        self.ax_zoa.set_ylim(bottom=y_min, top=y_max)
        for i, cluster in enumerate(clusters):
            self.zoa_tree.insert("", "end", values=(i + 1, f"{cluster['zoa_theta']:.2f}", f"{powers_db_relative[i]:.2f}"))
        self.canvas_zoa.draw()

    def copy_tree_data(self, tree):
        headings = tree["columns"]
        text_to_copy = "\t".join(headings) + "\n"
        for item_id in tree.get_children():
            values = tree.item(item_id, "values")
            text_to_copy += "\t".join(map(str, values)) + "\n"
        self.clipboard_clear()
        self.clipboard_append(text_to_copy)
        self.log(f"Table data for '{tree.heading(headings[0])['text']}' copied to clipboard.")

    def save_config(self):
        config_data = {key: var.get() for key, var in self.entries.items()}
        from tkinter.filedialog import asksaveasfilename
        filepath = asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json"), ("All files", "*.*")], title="Save Simulation Configuration")
        if not filepath:
            return
        with open(filepath, 'w') as f:
            json.dump(config_data, f, indent=4)
        self.log(f"Configuration saved to {filepath}")

    def load_config(self):
        from tkinter.filedialog import askopenfilename
        filepath = askopenfilename(filetypes=[("JSON files", "*.json"), ("All files", "*.*")], title="Load Simulation Configuration")
        if not filepath:
            return
        with open(filepath, 'r') as f:
            config_data = json.load(f)
        for key, value in config_data.items():
            if key in self.entries:
                self.entries[key].set(value)
        self.log(f"Configuration loaded from {filepath}")

if __name__ == '__main__':
    app = ChannelModelApp()
    app.mainloop()