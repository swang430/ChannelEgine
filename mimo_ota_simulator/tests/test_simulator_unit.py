
import numpy as np
import pytest
from mimo_ota_simulator.simulator import MIMO_OTA_Simulator
from mimo_ota_simulator.data_models import ChamberConfig

def test_calculate_probe_weights_simple_case():
    """
    Tests the probe weight calculation for a simple, non-wraparound case.
    4 probes at 0, 90, 180, 270 degrees. AOA is 45 degrees.
    Expected: Weights are shared between probe 0 (0 deg) and probe 1 (90 deg).
    """
    # 1. Setup
    simulator = MIMO_OTA_Simulator()
    chamber_config = ChamberConfig(
        num_probes=4,
        dual_polarized=False,
        distribution='ring',
        radius_m=1.0,
        calibration_matrix_B=np.zeros((2, 4)) # Dummy matrix
    )
    
    # 2. Execution
    # AOA is 45 degrees, exactly between probe 0 (0 deg) and probe 1 (90 deg)
    weights = simulator._calculate_probe_weights_for_aoa(45.0, chamber_config)

    # 3. Assertion
    expected_weights = np.zeros((4, 1))
    # Probe 0 (at 0 deg) and Probe 1 (at 90 deg) should share the weight.
    # The distance is 90 degrees. 45 degrees is halfway.
    # w1 = dist2 / total_dist = 45 / 90 = 0.5
    # w2 = dist1 / total_dist = 45 / 90 = 0.5
    expected_weights[0] = 0.5
    expected_weights[1] = 0.5
    
    np.testing.assert_allclose(weights, expected_weights, atol=1e-6)

def test_calculate_probe_weights_wraparound_case():
    """
    Tests the probe weight calculation for a case that spans the 0/360 degree wraparound.
    4 probes at 0, 90, 180, 270 degrees. AOA is 350 degrees.
    Expected: Weights are shared between probe 0 (0 deg) and probe 3 (270 deg).
    """
    # 1. Setup
    simulator = MIMO_OTA_Simulator()
    chamber_config = ChamberConfig(
        num_probes=4,
        dual_polarized=False,
        distribution='ring',
        radius_m=1.0,
        calibration_matrix_B=np.zeros((2, 4)) # Dummy matrix
    )
    
    # 2. Execution
    # AOA is 350 degrees, between probe 3 (270 deg) and probe 0 (0/360 deg)
    weights = simulator._calculate_probe_weights_for_aoa(350.0, chamber_config)

    # 3. Assertion
    expected_weights = np.zeros((4, 1))
    # Total angular distance between probe 3 and 0 is 90 degrees (270 -> 360)
    # AOA at 350 is 80 deg from p3 (350-270) and 10 deg from p0 (360-350)
    # w_p0 = dist_p3 / total_dist = 80 / 90 (This is wrong, interpolation is linear)
    # The two closest probes are p3 (270) and p0 (0 or 360).
    # Angular distance is 90 degrees.
    # dist from 350 to 270 is 80. dist from 350 to 360 is 10.
    # w_p3 = 10 / 90
    # w_p0 = 80 / 90
    # Let's re-read the code. It's linear interpolation.
    # dist1 = 10 (to p0), dist2 = 80 (to p3). total_dist = 90.
    # w_p0 = dist2 / total_dist = 80/90
    # w_p3 = dist1 / total_dist = 10/90
    expected_weights[0] = 80.0 / 90.0
    expected_weights[3] = 10.0 / 90.0
    
    np.testing.assert_allclose(weights, expected_weights, atol=1e-6)

def test_calculate_probe_weights_dual_polarized():
    """
    Tests that dual-polarized probes get weights duplicated correctly.
    AOA is 45 degrees, same as simple case.
    """
    # 1. Setup
    simulator = MIMO_OTA_Simulator()
    chamber_config = ChamberConfig(
        num_probes=4,
        dual_polarized=True,
        distribution='ring',
        radius_m=1.0,
        calibration_matrix_B=np.zeros((2, 8)) # Dummy matrix, 8 effective probes
    )
    
    # 2. Execution
    weights = simulator._calculate_probe_weights_for_aoa(45.0, chamber_config)

    # 3. Assertion
    expected_weights = np.zeros((8, 1))
    # Probe 0 (indices 0,1) and Probe 1 (indices 2,3) share the weight.
    expected_weights[0] = 0.5
    expected_weights[1] = 0.5
    expected_weights[2] = 0.5
    expected_weights[3] = 0.5
    
    np.testing.assert_allclose(weights, expected_weights, atol=1e-6)
