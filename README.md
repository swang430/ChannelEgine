
# 3GPP TR 38.901 Channel Model

This repository contains a Python-based implementation of the 3GPP TR 38.901 channel model for 5G and beyond.

## Features

- Implements the 3GPP TR 38.901 V19.0.0 standard.
- Supports major scenarios: UMa, UMi, RMa, InH, etc.
- **Hybrid Model**: Combine large-scale parameters (e.g., from UMa) with deterministic small-scale models (e.g., CDL-A) for physically meaningful simulations.
- **Deterministic LSPs**: Option to use median values for large-scale parameters for repeatable testing.
- Flexible and easy-to-use API.
- Included GUI for quick simulations and visualization.

## Installation

1. Clone the repository:
   ```
   git clone <repository_url>
   ```
2. Create a virtual environment:
   ```
   python3 -m venv .venv
   ```
3. Activate the virtual environment:
   ```
   source .venv/bin/activate
   ```
4. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
5. Install the packages in editable mode:
   ```
   pip install -e ./channel_model_38901
   pip install -e ./mimo_ota_simulator
   ```

## Usage

### GUI

To run a simulation, you can use the provided GUI:

```
python gui.py
```

### Scripting

You can use the `ChannelSimulator` class in your own scripts. Here are a few examples:

**Basic Simulation**

```python
from channel_model_38901.simulator import ChannelSimulator

# Create a simulator instance for UMa scenario
simulator = ChannelSimulator(scenario_name='UMa', center_frequency_hz=3.5e9)

# Run the simulation
results = simulator.run()

# Print the results
print(f"Channel Matrix:\n{results['channel_matrix']}")
print(f"Pathloss: {results['pathloss_db']:.2f} dB")
```

**Hybrid Model Simulation**

Combine the UMa large-scale environment with a deterministic CDL-A cluster model.

```python
from channel_model_38901.simulator import ChannelSimulator

simulator = ChannelSimulator(
    scenario_name='UMa', 
    cluster_model_name='CDL-A',
    center_frequency_hz=3.5e9
)

results = simulator.run()

print(f"Channel Matrix (UMa + CDL-A):\n{results['channel_matrix']}")
```

**Simulation with Median LSPs**

Run a simulation with deterministic large-scale parameters.

```python
from channel_model_38901.simulator import ChannelSimulator

simulator = ChannelSimulator(
    scenario_name='RMa', 
    center_frequency_hz=0.7e9,
    use_median_lsps=True
)

results = simulator.run()

print(f"Channel Matrix (RMa with median LSPs):\n{results['channel_matrix']}")
```

