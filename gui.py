import tkinter as tk
from tkinter import ttk
import numpy as np
from channel_model_38901.simulator import ChannelSimulator
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class ChannelSimGUI:
    def __init__(self, master):
        self.master = master
        master.title("3GPP TR 38.901 Channel Simulator")

        # --- Widgets ---
        self.label_scenario = ttk.Label(master, text="Scenario:")
        self.scenario_var = tk.StringVar(value='UMa')
        self.scenario_menu = ttk.Combobox(master, textvariable=self.scenario_var, values=['UMa', 'UMi-StreetCanyon', 'UMi-OpenArea', 'RMa', 'InH-Office', 'SMa', 'InF'])
        
        self.label_freq = ttk.Label(master, text="Center Frequency (GHz):")
        self.freq_var = tk.StringVar(value='3.5')
        self.freq_entry = ttk.Entry(master, textvariable=self.freq_var)

        self.label_condition = ttk.Label(master, text="Channel Condition:")
        self.condition_var = tk.StringVar(value='Automatic')
        self.condition_menu = ttk.Combobox(master, textvariable=self.condition_var, values=['Automatic', 'LOS', 'NLOS'])

        self.label_num_tx = ttk.Label(master, text="Num TX Antennas:")
        self.num_tx_var = tk.StringVar(value='1')
        self.num_tx_entry = ttk.Entry(master, textvariable=self.num_tx_var)

        self.label_num_rx = ttk.Label(master, text="Num RX Antennas:")
        self.num_rx_var = tk.StringVar(value='1')
        self.num_rx_entry = ttk.Entry(master, textvariable=self.num_rx_var)

        self.label_seed = ttk.Label(master, text="Random Seed (optional):")
        self.seed_var = tk.StringVar(value='')
        self.seed_entry = ttk.Entry(master, textvariable=self.seed_var)

        self.run_button = ttk.Button(master, text="Run Simulation", command=self.run_simulation)
        
        self.notebook = ttk.Notebook(master)
        self.results_text = tk.Text(self.notebook, height=10, width=80)
        self.results_text.config(state=tk.DISABLED)
        self.notebook.add(self.results_text, text='Log')

        # --- Parameters Tab ---
        self.params_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.params_frame, text='Channel Parameters')
        self.params_tree = ttk.Treeview(self.params_frame, columns=('Value', 'Unit'), show='headings')
        self.params_tree.heading('Value', text='Value')
        self.params_tree.heading('Unit', text='Unit')
        self.params_tree.pack(fill=tk.BOTH, expand=True)

        # --- Matplotlib Figure ---
        self.fig, self.axs = plt.subplots(1, 2, figsize=(12, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.notebook)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.notebook.add(self.canvas_widget, text='Plots')

        # --- Layout ---
        master.grid_rowconfigure(7, weight=1)
        master.grid_columnconfigure(1, weight=1)

        self.label_scenario.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.scenario_menu.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)
        self.label_freq.grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.freq_entry.grid(row=1, column=1, padx=5, pady=5, sticky=tk.EW)
        self.label_condition.grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        self.condition_menu.grid(row=2, column=1, padx=5, pady=5, sticky=tk.EW)
        self.label_num_tx.grid(row=3, column=0, padx=5, pady=5, sticky=tk.W)
        self.num_tx_entry.grid(row=3, column=1, padx=5, pady=5, sticky=tk.EW)
        self.label_num_rx.grid(row=4, column=0, padx=5, pady=5, sticky=tk.W)
        self.num_rx_entry.grid(row=4, column=1, padx=5, pady=5, sticky=tk.EW)
        self.label_seed.grid(row=5, column=0, padx=5, pady=5, sticky=tk.W)
        self.seed_entry.grid(row=5, column=1, padx=5, pady=5, sticky=tk.EW)
        self.run_button.grid(row=6, column=0, columnspan=2, pady=10)
        self.notebook.grid(row=7, column=0, columnspan=2, padx=5, pady=5, sticky=tk.NSEW)

    def run_simulation(self):
        scenario = self.scenario_var.get()
        freq_ghz = float(self.freq_var.get())
        condition = self.condition_var.get()
        num_tx = int(self.num_tx_var.get())
        num_rx = int(self.num_rx_var.get())
        seed_str = self.seed_var.get()

        if seed_str:
            try:
                seed = int(seed_str)
                np.random.seed(seed)
                self.results_text.config(state=tk.NORMAL)
                self.results_text.delete(1.0, tk.END)
                self.results_text.insert(tk.END, f"Using random seed: {seed}\n")
                self.results_text.config(state=tk.DISABLED)
            except ValueError:
                self.results_text.config(state=tk.NORMAL)
                self.results_text.delete(1.0, tk.END)
                self.results_text.insert(tk.END, "Invalid random seed. Must be an integer.")
                self.results_text.config(state=tk.DISABLED)
                return
        else:
            np.random.seed(None) # Use a random seed

        self.results_text.config(state=tk.NORMAL)
        self.results_text.insert(tk.END, f"Starting simulation for {scenario} at {freq_ghz} GHz...\n")
        self.master.update_idletasks()

        try:
            simulator = ChannelSimulator(scenario_name=scenario, center_frequency_hz=freq_ghz * 1e9)
            simulator.config.antenna_config.num_tx_antennas = num_tx
            simulator.config.antenna_config.num_rx_antennas = num_rx
            simulator.config.antenna_config.tx_positions = np.zeros((num_tx, 3))
            simulator.config.antenna_config.rx_positions = np.zeros((num_rx, 3))

            force_condition = None if condition == 'Automatic' else condition
            results = simulator.run(force_condition=force_condition)
            
            self.update_results(results)

        except Exception as e:
            self.results_text.insert(tk.END, f"\n--- ERROR ---\n{e}")
        
        self.results_text.config(state=tk.DISABLED)

    def update_results(self, results):
        # Log Tab
        self.results_text.insert(tk.END, "\n--- Simulation Complete ---")
        self.results_text.insert(tk.END, f"Generated Channel Matrix H:\n{results['channel_matrix']}\n")
        
        # Plots Tab
        self.plot_pdp(results['clusters'])
        self.plot_aps(results['clusters'])

        # Parameters Tab
        self.update_params_tab(results)

    def update_params_tab(self, results):
        # Clear previous results
        for item in self.params_tree.get_children():
            self.params_tree.delete(item)

        lsps = results['lsps']
        is_los = results['is_los']

        params_to_display = {
            "Channel Condition": (('LOS' if is_los else 'NLOS'), '-'),
            "Pathloss": (f"{results['pathloss_db']:.2f}", 'dB'),
            "Shadow Fading (SF)": (f"{lsps.shadow_fade_db:.2f}", 'dB'),
            "Delay Spread (DS)": (f"{lsps.delay_spread_s * 1e9:.2f}", 'ns'),
            "Arrival Azimuth Spread (ASA)": (f"{lsps.arrival_azimuth_spread_deg:.2f}", 'degrees'),
            "Departure Azimuth Spread (ASD)": (f"{lsps.departure_azimuth_spread_deg:.2f}", 'degrees'),
        }

        if is_los and lsps.k_factor_linear > 0:
            k_factor_db = 10 * np.log10(lsps.k_factor_linear)
            params_to_display["K-factor"] = (f"{k_factor_db:.2f}", 'dB')

        for name, (value, unit) in params_to_display.items():
            self.params_tree.insert('', tk.END, text=name, values=(value, unit))

    def plot_pdp(self, clusters):
        self.axs[0].clear()
        delays = [c['delay'] for c in clusters]
        powers = [c['power'] for c in clusters]
        self.axs[0].stem(delays, powers)
        self.axs[0].set_xlabel("Delay (s)")
        self.axs[0].set_ylabel("Normalized Power")
        self.axs[0].set_title("Power Delay Profile (PDP)")
        self.canvas.draw()

    def plot_aps(self, clusters):
        self.axs[1].clear()
        aoa = [c['aoa_phi'] for c in clusters]
        powers = [c['power'] for c in clusters]
        self.axs[1].stem(aoa, powers)
        self.axs[1].set_xlabel("Angle of Arrival (deg)")
        self.axs[1].set_ylabel("Normalized Power")
        self.axs[1].set_title("Angular Power Spectrum (APS)")
        self.canvas.draw()

if __name__ == '__main__':
    root = tk.Tk()
    gui = ChannelSimGUI(root)
    root.mainloop()
