import numpy as np

def calculate_correlation_matrix(num_antennas: int, angle_of_arrival_or_departure_deg: float, 
                                 angle_spread_deg: float, element_spacing: float) -> np.ndarray:
    """
    Calculates the spatial correlation matrix for a Uniform Linear Array (ULA).

    This implementation is based on the formula provided in 3GPP TR 38.901, Table 7.5-1.

    Args:
        num_antennas: The number of antenna elements in the array.
        angle_of_arrival_or_departure_deg: The mean angle of arrival (AOA) or departure (AOD) in degrees.
        angle_spread_deg: The azimuth angle spread (ASA or ASD) in degrees.
        element_spacing: The spacing between antenna elements in multiples of the wavelength.

    Returns:
        A (num_antennas x num_antennas) numpy array representing the spatial correlation matrix.
    """
    # Convert angles from degrees to radians
    phi = np.deg2rad(angle_of_arrival_or_departure_deg)
    sigma_as = np.deg2rad(angle_spread_deg)

    # Initialize the correlation matrix
    R = np.zeros((num_antennas, num_antennas), dtype=np.complex128)

    # Calculate the correlation for each pair of antenna elements (p, q)
    for p in range(num_antennas):
        for q in range(num_antennas):
            if p == q:
                R[p, q] = 1.0
            else:
                # Formula from TR 38.901, Table 7.5-1
                # Note: The standard provides a simplified integral. We use a common approximation.
                # A more precise implementation might use numerical integration.
                alpha = 2 * np.pi * element_spacing * (p - q) * np.sin(phi)
                beta = (2 * np.pi * element_spacing * (p - q) * np.cos(phi) * sigma_as)**2 / 2
                
                # Using the closed-form approximation for Gaussian angular spectrum
                R[p, q] = np.exp(-beta) * np.exp(1j * alpha)

    return R
