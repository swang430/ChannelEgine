import numpy as np

def _calculate_ula_correlation(num_elements: int, angle_deg: float, angle_spread_deg: float, 
                               element_spacing: float) -> np.ndarray:
    """
    Calculates the spatial correlation matrix for a Uniform Linear Array (ULA).
    This is the base function for both horizontal and vertical correlation calculations.
    """
    # Convert angles from degrees to radians
    phi = np.deg2rad(angle_deg)
    sigma_as = np.deg2rad(angle_spread_deg)

    R = np.zeros((num_elements, num_elements), dtype=np.complex128)

    for p in range(num_elements):
        for q in range(num_elements):
            if p == q:
                R[p, q] = 1.0
            else:
                alpha = 2 * np.pi * element_spacing * (p - q) * np.sin(phi)
                beta = (2 * np.pi * element_spacing * (p - q) * np.cos(phi) * sigma_as)**2 / 2
                R[p, q] = np.exp(-beta) * np.exp(1j * alpha)

    return R

def _calculate_ura_correlation(num_rows: int, num_cols: int, spacing_v: float, spacing_h: float,
                               zenith_deg: float, azimuth_deg: float, 
                               zenith_spread_deg: float, azimuth_spread_deg: float) -> np.ndarray:
    """
    Calculates the spatial correlation matrix for a Uniform Rectangular Array (URA).
    The URA correlation is modeled as the Kronecker product of two ULA correlations.
    """
    # Correlation for the vertical elements (rows)
    # Note: The 3GPP model relates vertical correlation to zenith angle (theta)
    R_vertical = _calculate_ula_correlation(num_rows, zenith_deg, zenith_spread_deg, spacing_v)

    # Correlation for the horizontal elements (columns)
    # Note: The 3GPP model relates horizontal correlation to azimuth angle (phi)
    R_horizontal = _calculate_ula_correlation(num_cols, azimuth_deg, azimuth_spread_deg, spacing_h)

    # The full URA correlation matrix is the Kronecker product of the two
    return np.kron(R_vertical, R_horizontal)


def get_spatial_correlation_matrix(array_type: str, num_rows: int, num_cols: int, 
                                   spacing_v: float, spacing_h: float,
                                   zenith_deg: float, azimuth_deg: float,
                                   zenith_spread_deg: float, azimuth_spread_deg: float) -> np.ndarray:
    """
    Public function to get the spatial correlation matrix for any supported array type.

    Args:
        array_type: 'ULA' or 'URA'.
        num_rows: Number of rows in the array.
        num_cols: Number of columns in the array.
        spacing_v: Vertical element spacing (in wavelengths).
        spacing_h: Horizontal element spacing (in wavelengths).
        zenith_deg: Mean zenith angle of arrival/departure.
        azimuth_deg: Mean azimuth angle of arrival/departure.
        zenith_spread_deg: Zenith angle spread.
        azimuth_spread_deg: Azimuth angle spread.

    Returns:
        The calculated spatial correlation matrix.
    """
    if array_type == 'ULA':
        # For a ULA, we assume it's horizontal. Rows = 1.
        return _calculate_ula_correlation(num_cols, azimuth_deg, azimuth_spread_deg, spacing_h)
    elif array_type == 'URA':
        return _calculate_ura_correlation(num_rows, num_cols, spacing_v, spacing_h, 
                                          zenith_deg, azimuth_deg, 
                                          zenith_spread_deg, azimuth_spread_deg)
    else:
        raise ValueError(f"Unsupported array type: {array_type}")