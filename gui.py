import tkinter as tk
from tkinter import ttk
import numpy as np
from channel_model_38901.simulator import ChannelSimulator
from channel_model_38901.models.uma_models import UMa_ChannelCondition # For deterministic check
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class ChannelSimGUI:
    def __init__(self, master):
        self.master = master
        master.title("3GPP TR 38.901 Channel Simulator")

        # --- Model for deterministic check ---
        self.condition_checker = UMa_ChannelCondition() # A bit of a hack, but works for now

        # --- Widgets ---
        # Frame for input parameters
        input_frame = ttk.LabelFrame(master, text="Simulation Parameters", padding="10")
        input_frame.grid(row=0, column=0, padx=10, pady=10, sticky=tk.EW)

        self.label_scenario = ttk.Label(input_frame, text="Scenario:")
        self.scenario_var = tk.StringVar(value='UMa')
        self.scenario_menu = ttk.Combobox(input_frame, textvariable=self.scenario_var, values=['UMa', 'UMi-StreetCanyon', 'UMi-OpenArea', 'RMa', 'InH-Office', 'SMa', 'InF'])
        
        self.label_freq = ttk.Label(input_frame, text="Center Freq (GHz):")
        self.freq_var = tk.StringVar(value='3.5')
        self.freq_entry = ttk.Entry(input_frame, textvariable=self.freq_var)

        self.label_dist = ttk.Label(input_frame, text="2D Distance (m):")
        self.dist_var = tk.StringVar(value='100')
        self.dist_entry = ttk.Entry(input_frame, textvariable=self.dist_var)

        self.label_condition = ttk.Label(input_frame, text="Channel Condition:")
        self.condition_var = tk.StringVar(value='Automatic')
        self.condition_menu = ttk.Combobox(input_frame, textvariable=self.condition_var, values=['Automatic', 'LOS', 'NLOS'])

        self.label_seed = ttk.Label(input_frame, text="Random Seed:")
        self.seed_var = tk.StringVar(value='')
        self.seed_entry = ttk.Entry(input_frame, textvariable=self.seed_var)
        self.deterministic_label = ttk.Label(input_frame, text="", foreground="blue")

        self.label_num_tx = ttk.Label(input_frame, text="Num TX Antennas:")
        self.num_tx_var = tk.StringVar(value='1')
        self.num_tx_entry = ttk.Entry(input_frame, textvariable=self.num_tx_var)

        self.label_tx_spacing = ttk.Label(input_frame, text="TX Spacing (λ):")
        self.tx_spacing_var = tk.StringVar(value='0.5')
        self.tx_spacing_entry = ttk.Entry(input_frame, textvariable=self.tx_spacing_var)

        self.label_num_rx = ttk.Label(input_frame, text="Num RX Antennas:")
        self.num_rx_var = tk.StringVar(value='1')
        self.num_rx_entry = ttk.Entry(input_frame, textvariable=self.num_rx_var)

        self.label_rx_spacing = ttk.Label(input_frame, text="RX Spacing (λ):")
        self.rx_spacing_var = tk.StringVar(value='0.5')
        self.rx_spacing_entry = ttk.Entry(input_frame, textvariable=self.rx_spacing_var)

        self.run_button = ttk.Button(master, text="Run Simulation", command=self.run_simulation)
        
        # --- Output Notebook ---
        self.notebook = ttk.Notebook(master)
        self.results_text = tk.Text(self.notebook, height=10, width=80, wrap=tk.WORD)
        self.results_text.config(state=tk.DISABLED)
        self.notebook.add(self.results_text, text='Log')

        self.params_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.params_frame, text='Channel Parameters')
        self.params_tree = ttk.Treeview(self.params_frame, columns=('Parameter', 'Value', 'Unit'), show='headings')
        self.params_tree.heading('Parameter', text='Parameter'); self.params_tree.heading('Value', text='Value'); self.params_tree.heading('Unit', text='Unit')
        self.params_tree.column('Parameter', anchor=tk.W, width=200); self.params_tree.column('Value', anchor=tk.W, width=150); self.params_tree.column('Unit', anchor=tk.W, width=100)
        self.params_tree.pack(fill=tk.BOTH, expand=True)

        self.fig, self.axs = plt.subplots(1, 2, figsize=(12, 4)); self.fig.tight_layout(pad=3.0)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.notebook)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.notebook.add(self.canvas_widget, text='Plots')

        # --- Layout ---
        self.label_scenario.grid(row=0, column=0, sticky=tk.W); self.scenario_menu.grid(row=0, column=1, sticky=tk.EW)
        self.label_freq.grid(row=1, column=0, sticky=tk.W); self.freq_entry.grid(row=1, column=1, sticky=tk.EW)
        self.label_dist.grid(row=2, column=0, sticky=tk.W); self.dist_entry.grid(row=2, column=1, sticky=tk.EW)
        self.label_condition.grid(row=3, column=0, sticky=tk.W); self.condition_menu.grid(row=3, column=1, sticky=tk.EW)
        self.label_seed.grid(row=4, column=0, sticky=tk.W); self.seed_entry.grid(row=4, column=1, sticky=tk.EW); self.deterministic_label.grid(row=4, column=2, sticky=tk.W, padx=5)
        self.label_num_tx.grid(row=5, column=0, sticky=tk.W); self.num_tx_entry.grid(row=5, column=1, sticky=tk.EW)
        self.label_tx_spacing.grid(row=6, column=0, sticky=tk.W); self.tx_spacing_entry.grid(row=6, column=1, sticky=tk.EW)
        self.label_num_rx.grid(row=7, column=0, sticky=tk.W); self.num_rx_entry.grid(row=7, column=1, sticky=tk.EW)
        self.label_rx_spacing.grid(row=8, column=0, sticky=tk.W); self.rx_spacing_entry.grid(row=8, column=1, sticky=tk.EW)
        for widget in input_frame.winfo_children(): widget.grid_configure(pady=3)

        self.run_button.grid(row=1, column=0, padx=10, pady=10, sticky=tk.EW)
        self.notebook.grid(row=2, column=0, padx=10, pady=10, sticky=tk.NSEW)
        master.grid_rowconfigure(2, weight=1); master.grid_columnconfigure(0, weight=1)

        # --- Bindings for dynamic updates ---
        self.condition_var.trace_add("write", self.update_ui_state)
        self.seed_var.trace_add("write", self.update_ui_state)
        self.dist_var.trace_add("write", self.update_ui_state)
        self.update_ui_state() # Initial call

    def update_ui_state(self, *args):
        condition = self.condition_var.get()
        seed_str = self.seed_var.get()
        dist_str = self.dist_var.get()

        if condition != 'Automatic':
            self.dist_entry.config(state='disabled')
            self.deterministic_label.config(text=f"(Forced to {condition})")
            return

        self.dist_entry.config(state='normal')
        if seed_str and dist_str:
            try:
                seed = int(seed_str)
                dist = float(dist_str)
                np.random.seed(seed)
                is_los = self.condition_checker.get_condition(distance_2d=dist, h_ut=1.5)
                result_text = "LOS" if is_los else "NLOS"
                self.deterministic_label.config(text=f"(Determined by seed: {result_text})")
            except (ValueError, tk.TclError):
                self.deterministic_label.config(text="(Invalid seed/dist)")
        else:
            self.deterministic_label.config(text="(Probabilistic)")

    def run_simulation(self):
        try:
            scenario = self.scenario_var.get()
            freq_ghz = float(self.freq_var.get())
            dist_m = float(self.dist_var.get())
            condition = self.condition_var.get()
            num_tx = int(self.num_tx_var.get())
            tx_spacing = float(self.tx_spacing_var.get())
            num_rx = int(self.num_rx_var.get())
            rx_spacing = float(self.rx_spacing_var.get())
            seed_str = self.seed_var.get()
        except ValueError:
            self.results_text.config(state=tk.NORMAL); self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, "Error: Invalid input. Please check that all fields are numbers.")
            self.results_text.config(state=tk.DISABLED)
            return

        self.results_text.config(state=tk.NORMAL); self.results_text.delete(1.0, tk.END)

        if seed_str:
            np.random.seed(int(seed_str))
            self.results_text.insert(tk.END, f"Using random seed: {seed_str}\n")
        else:
            np.random.seed(None)
            self.results_text.insert(tk.END, "Using random seed from system time.\n")

        self.results_text.insert(tk.END, f"Starting simulation for {scenario} at {freq_ghz} GHz...\n")
        self.master.update_idletasks()

        try:
            simulator = ChannelSimulator(scenario_name=scenario, center_frequency_hz=freq_ghz * 1e9)
            simulator.set_bs_ue_positions(np.array([[0, 0, 25]]), np.array([[dist_m, 0, 1.5]]))
            
            # Update antenna config from GUI
            ant_config = simulator.config.antenna_config
            ant_config.num_tx_antennas = num_tx
            ant_config.tx_element_spacing = tx_spacing
            ant_config.num_rx_antennas = num_rx
            ant_config.rx_element_spacing = rx_spacing
            # For now, we assume ULA along y-axis, so positions are simple
            ant_config.tx_positions = np.array([[0, i * tx_spacing, 0] for i in range(num_tx)])
            ant_config.rx_positions = np.array([[0, i * rx_spacing, 0] for i in range(num_rx)])

            force_condition = None if condition == 'Automatic' else condition
            results = simulator.run(force_condition=force_condition)
            self.update_results(results)

        except Exception as e:
            self.results_text.insert(tk.END, f"\n--- ERROR ---\n{e}")
        finally:
            self.results_text.config(state=tk.DISABLED)

    def update_results(self, results):
        self.results_text.insert(tk.END, "\n--- Simulation Complete ---")
        self.results_text.insert(tk.END, f"Generated Channel Matrix H:\n{results['channel_matrix']}\n")
        self.plot_pdp(results['clusters'])
        self.plot_aps(results['clusters'])
        self.update_params_tab(results)

    def update_params_tab(self, results):
        for item in self.params_tree.get_children(): self.params_tree.delete(item)
        lsps, is_los = results['lsps'], results['is_los']
        params = {
            "Channel Condition": (('LOS' if is_los else 'NLOS'), '-'),
            "Pathloss": (f"{results['pathloss_db']:.2f}", 'dB'),
            "Shadow Fading (SF)": (f"{lsps.shadow_fade_db:.2f}", 'dB'),
            "Delay Spread (DS)": (f"{lsps.delay_spread_s * 1e9:.2f}", 'ns'),
            "Arrival Azimuth Spread (ASA)": (f"{lsps.arrival_azimuth_spread_deg:.2f}", 'deg'),
            "Departure Azimuth Spread (ASD)": (f"{lsps.departure_azimuth_spread_deg:.2f}", 'deg'),
        }
        if is_los and lsps.k_factor_linear > 0:
            params["K-factor"] = (f"{10 * np.log10(lsps.k_factor_linear):.2f}", 'dB')
        for name, (value, unit) in params.items():
            self.params_tree.insert('', tk.END, values=(name, value, unit))

    def plot_pdp(self, clusters):
        self.axs[0].clear()
        delays = [c['delay'] for c in clusters]; powers = [c['power'] for c in clusters]
        self.axs[0].stem(delays, powers); self.axs[0].set_xlabel("Delay (s)"); self.axs[0].set_ylabel("Normalized Power"); self.axs[0].set_title("Power Delay Profile (PDP)"); self.axs[0].grid(True)
        self.canvas.draw()

    def plot_aps(self, clusters):
        self.axs[1].clear()
        aoa = [c['aoa_phi'] for c in clusters]; powers = [c['power'] for c in clusters]
        self.axs[1].stem(aoa, powers); self.axs[1].set_xlabel("Angle of Arrival (deg)"); self.axs[1].set_ylabel("Normalized Power"); self.axs[1].set_title("Angular Power Spectrum (APS)"); self.axs[1].grid(True)
        self.canvas.draw()

if __name__ == '__main__':
    root = tk.Tk()
    gui = ChannelSimGUI(root)
    root.mainloop()
