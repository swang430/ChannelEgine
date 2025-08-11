import pytest
import numpy as np

from mimo_ota_simulator.simulator import MIMO_OTA_Simulator
from mimo_ota_simulator.data_models import ChamberConfig, TargetChannelConfig, AntennaArrayConfig

class TestSimulatorV2:
    def test_mimo_2x2_simulation_shape(self):
        """
        Verifies that the simulator can run a 2x2 MIMO configuration and that
        the output H_target has the correct shape.
        """
        # 1. Instantiate the simulator
        simulator = MIMO_OTA_Simulator()

        # 2. Define the V2.0 simulation configuration for a 2x2 ULA setup
        tx_ant = AntennaArrayConfig(num_cols=2)
        rx_ant = AntennaArrayConfig(num_cols=2)
        
        channel_config = TargetChannelConfig(
            model_name='UMa',
            tx_antenna=tx_ant,
            rx_antenna=rx_ant
        )

        # 3. Create a compatible chamber config
        # Nr = num_rx_rows * num_rx_cols = 2
        # Let's use Np = 16 probes.
        chamber_config = ChamberConfig(
            num_probes=16,
            distribution='ring',
            radius_m=1.5,
            calibration_matrix_B=np.zeros((2, 16)) # Dummy matrix with correct shape
        )

        # 4. Run the simulation
        results = simulator.run(
            chamber_config=chamber_config,
            channel_config=channel_config
        )

        # 5. Verify the output shape
        assert results['target_H_matrix'].shape == (2, 2), \
            f"Expected H_target shape of (2, 2) for a 2x2 MIMO config, but got {results['target_H_matrix'].shape}"

