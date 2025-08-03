import tkinter as tk
from tkinter import ttk
from channel_model_38901.simulator import ChannelSimulator

class ChannelSimGUI:
    def __init__(self, master):
        self.master = master
        master.title("3GPP TR 38.901 Channel Simulator")

        # --- Widgets ---
        self.label_scenario = ttk.Label(master, text="Scenario:")
        self.scenario_var = tk.StringVar(value='UMa')
        self.scenario_menu = ttk.Combobox(master, textvariable=self.scenario_var, values=['UMa', 'UMi-StreetCanyon', 'RMa', 'InH-Office', 'SMa'])
        
        self.label_freq = ttk.Label(master, text="Center Frequency (GHz):")
        self.freq_var = tk.StringVar(value='3.5')
        self.freq_entry = ttk.Entry(master, textvariable=self.freq_var)

        self.run_button = ttk.Button(master, text="Run Simulation", command=self.run_simulation)
        
        self.results_text = tk.Text(master, height=15, width=80)
        self.results_text.config(state=tk.DISABLED)

        # --- Layout ---
        self.label_scenario.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.scenario_menu.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)
        self.label_freq.grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.freq_entry.grid(row=1, column=1, padx=5, pady=5, sticky=tk.EW)
        self.run_button.grid(row=2, column=0, columnspan=2, pady=10)
        self.results_text.grid(row=3, column=0, columnspan=2, padx=5, pady=5)

    def run_simulation(self):
        scenario = self.scenario_var.get()
        freq_ghz = float(self.freq_var.get())
        
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, f"Starting simulation for {scenario} at {freq_ghz} GHz...\n")
        self.master.update_idletasks()

        try:
            simulator = ChannelSimulator(scenario_name=scenario, center_frequency_hz=freq_ghz * 1e9)
            channel_matrix = simulator.run()
            
            self.results_text.insert(tk.END, "\n--- Simulation Complete ---\n")
            self.results_text.insert(tk.END, f"Generated Channel Matrix H:\n{channel_matrix}\n")

        except Exception as e:
            self.results_text.insert(tk.END, f"\n--- ERROR ---\n{e}")
        
        self.results_text.config(state=tk.DISABLED)

if __name__ == '__main__':
    root = tk.Tk()
    gui = ChannelSimGUI(root)
    root.mainloop()
