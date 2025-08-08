
# 3GPP TR 38.901 Channel Model

This repository contains a Python-based implementation of the 3GPP TR 38.901 channel model for 5G and beyond.

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

## Usage

To run a simulation, you can use the provided GUI:

```
python gui.py
```

Alternatively, you can use the `ChannelSimulator` class in your own scripts:

```python
from channel_model_38901.simulator import ChannelSimulator

# Create a simulator instance
simulator = ChannelSimulator(scenario_name='UMa', center_frequency_hz=3.5e9)

# Run the simulation
channel_matrix, clusters = simulator.run()

# Print the results
print(channel_matrix)
```
