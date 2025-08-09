import numpy as np

def spherical_to_cartesian(phi_deg: float, theta_deg: float) -> np.ndarray:
    """Converts spherical coordinates (azimuth, zenith) to a Cartesian unit vector."""
    phi_rad = np.deg2rad(phi_deg)
    theta_rad = np.deg2rad(theta_deg)
    
    x = np.sin(theta_rad) * np.cos(phi_rad)
    y = np.sin(theta_rad) * np.sin(phi_rad)
    z = np.cos(theta_rad)
    
    return np.array([x, y, z])

def calculate_element_gain(phi_deg: float, theta_deg: float, g_h_max_db: float = 8.0, 
                           phi_3db_deg: float = 65.0, theta_3db_deg: float = 65.0, 
                           a_v_db: float = 30.0) -> float:
    """
    Calculates the antenna element gain based on the model in 3GPP TR 38.901, Table 7.3-1.

    Args:
        phi_deg: Azimuth angle in degrees (-180 to 180).
        theta_deg: Zenith angle in degrees (0 to 180).
        g_h_max_db: Maximum horizontal gain (dBi).
        phi_3db_deg: Horizontal 3dB beamwidth.
        theta_3db_deg: Vertical 3dB beamwidth.
        a_v_db: Front-to-back ratio.

    Returns:
        The element gain in dBi for the given angles.
    """
    # --- Horizontal Pattern A_H(phi) ---
    a_h = -min(12 * ((phi_deg / phi_3db_deg)**2), 30.0)

    # --- Vertical Pattern A_V(theta) ---
    a_v = -min(12 * (((theta_deg - 90) / theta_3db_deg)**2), a_v_db)

    # --- Combined Pattern ---
    gain_db = g_h_max_db - min(-(a_h + a_v), a_v_db)
    
    return gain_db