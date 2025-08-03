

import numpy as np
from channel_model_38901.simulator import ChannelSimulator

if __name__ == "__main__":
    print("--- 3GPP TR 38.901 Channel Simulator ---")
    
    # 1. Create a simulator instance for the UMa scenario at 3.5 GHz
    simulator = ChannelSimulator(scenario_name='UMa', center_frequency_hz=3.5e9)
    
    # 2. (Optional) Update positions if needed
    # simulator.set_bs_ue_positions(
    #     bs_positions=np.array([[0, 0, 30]]),
    #     ue_positions=np.array([[200, 50, 1.5]])
    # )
    
    # 3. Run the simulation
    channel_matrix = simulator.run()
    
    # 4. Print the results
    print("\n--- Simulation Results ---")
    print("Generated Channel Matrix H:")
    print(channel_matrix)
    print(f"Matrix Shape: {channel_matrix.shape}")
    print("--------------------------")

