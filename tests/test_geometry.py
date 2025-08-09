import numpy as np
import pytest
from channel_model_38901.models.antenna_pattern import spherical_to_cartesian

def test_spherical_to_cartesian_conversion():
    """Tests the spherical to Cartesian coordinate conversion for key angles."""
    # Test case 1: Along X-axis
    np.testing.assert_allclose(spherical_to_cartesian(phi_deg=0, theta_deg=90), [1, 0, 0], atol=1e-7)
    
    # Test case 2: Along Y-axis
    np.testing.assert_allclose(spherical_to_cartesian(phi_deg=90, theta_deg=90), [0, 1, 0], atol=1e-7)

    # Test case 3: Along Z-axis (North Pole)
    np.testing.assert_allclose(spherical_to_cartesian(phi_deg=0, theta_deg=0), [0, 0, 1], atol=1e-7)

    # Test case 4: A random angle
    vec = spherical_to_cartesian(phi_deg=45, theta_deg=60)
    # The resulting vector must be a unit vector, i.e., its norm must be 1.
    np.testing.assert_allclose(np.linalg.norm(vec), 1.0, atol=1e-7)

def test_mimo_phase_difference():
    """Verifies the spatial phase term calculation in a controlled MIMO scenario."""
    # --- Setup ---
    num_rx = 2
    num_tx = 1
    lambda_0 = 0.1 # Assume wavelength is 0.1 meters
    rx_spacing = 0.5 # Half-wavelength spacing
    
    # 2x1 ULA receiver array along the y-axis
    d_rx = np.array([[0, 0, 0], [0, rx_spacing * lambda_0, 0]])
    d_tx = np.array([[0, 0, 0]]) # Single transmitter at origin

    # --- Test Case 1: Signal arrives perpendicular to the array (from X-axis) ---
    # AOA: phi=0, theta=90
    r_rx_perp = spherical_to_cartesian(0, 90)
    
    # Calculate phase term for each antenna
    phase_1_perp = np.exp(1j * 2 * np.pi * np.dot(r_rx_perp, d_rx[0]) / lambda_0)
    phase_2_perp = np.exp(1j * 2 * np.pi * np.dot(r_rx_perp, d_rx[1]) / lambda_0)
    
    # Expected: Phase difference should be 0, as the wavefront hits both antennas simultaneously.
    np.testing.assert_allclose(phase_1_perp, phase_2_perp, atol=1e-7)

    # --- Test Case 2: Signal arrives along the array axis (from Y-axis) ---
    # AOA: phi=90, theta=90
    r_rx_along = spherical_to_cartesian(90, 90)

    # Calculate phase term for each antenna
    phase_1_along = np.exp(1j * 2 * np.pi * np.dot(r_rx_along, d_rx[0]) / lambda_0)
    phase_2_along = np.exp(1j * 2 * np.pi * np.dot(r_rx_along, d_rx[1]) / lambda_0)

    # Expected: Phase difference should be pi (180 degrees), as the second antenna is half a wavelength further.
    # phase_2 / phase_1 = exp(j * delta_phase)
    # So, delta_phase = angle(phase_2 / phase_1)
    phase_difference = np.angle(phase_2_along / phase_1_along)
    np.testing.assert_allclose(phase_difference, np.pi, atol=1e-7)
