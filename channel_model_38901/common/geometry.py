import numpy as np

def get_antenna_positions(array_type: str, num_rows: int, num_cols: int,
                          spacing_h_m: float, spacing_v_m: float,
                          bearing_deg: float = 0.0, downtilt_deg: float = 0.0,
                          slant_deg: float = 0.0) -> np.ndarray:
    """
    Calculates the 3D positions of antenna elements in an array.
    The array is defined in the y-z plane (local coordinate system, LCS) and then
    rotated into the global coordinate system (GCS) by the array orientation angles
    per 3GPP TR 38.901 Sec 7.1.3. With all orientation angles 0 the rotation is the
    identity, so legacy callers see the original y-z-plane layout unchanged.

    Args:
        array_type: 'ULA' or 'URA'.
        num_rows: Number of rows in the array.
        num_cols: Number of columns in the array.
        spacing_h_m: Horizontal spacing between elements in meters.
        spacing_v_m: Vertical spacing between elements in meters.
        bearing_deg: Array bearing angle alpha (azimuth) in degrees.
        downtilt_deg: Array downtilt angle beta (elevation) in degrees.
        slant_deg: Array slant angle gamma (rotation about boresight) in degrees.

    Returns:
        A numpy array of shape (num_antennas, 3) with the 3D coordinates.
    """
    positions = []
    if array_type == 'ULA':
        num_rows = 1 # Force rows to 1 for ULA

    for r in range(num_rows):
        for c in range(num_cols):
            # URA/ULA in y-z plane (LCS), starting from origin
            pos = np.array([0.0, c * spacing_h_m, r * spacing_v_m])
            positions.append(pos)

    positions = np.array(positions)
    return apply_orientation(positions, bearing_deg, downtilt_deg, slant_deg)


def rotation_matrix(bearing_deg: float, downtilt_deg: float, slant_deg: float) -> np.ndarray:
    """
    3GPP TR 38.901 Sec 7.1.3 LCS->GCS rotation matrix R = Rz(alpha)*Ry(beta)*Rx(gamma),
    where alpha=bearing, beta=downtilt, gamma=slant.
    """
    a = np.deg2rad(bearing_deg)
    b = np.deg2rad(downtilt_deg)
    g = np.deg2rad(slant_deg)
    Rz = np.array([[np.cos(a), -np.sin(a), 0.0],
                   [np.sin(a),  np.cos(a), 0.0],
                   [0.0,        0.0,       1.0]])
    Ry = np.array([[np.cos(b), 0.0, np.sin(b)],
                   [0.0,       1.0, 0.0],
                   [-np.sin(b), 0.0, np.cos(b)]])
    Rx = np.array([[1.0, 0.0,        0.0],
                   [0.0, np.cos(g), -np.sin(g)],
                   [0.0, np.sin(g),  np.cos(g)]])
    return Rz @ Ry @ Rx


def apply_orientation(positions: np.ndarray, bearing_deg: float, downtilt_deg: float,
                      slant_deg: float) -> np.ndarray:
    """Rotate LCS element positions (N,3) into the GCS. Identity when all angles are 0."""
    if bearing_deg == 0.0 and downtilt_deg == 0.0 and slant_deg == 0.0:
        return positions
    if positions.size == 0:
        return positions
    R = rotation_matrix(bearing_deg, downtilt_deg, slant_deg)
    return positions @ R.T


def gcs_to_lcs_angles(theta_deg: float, phi_deg: float, bearing_deg: float,
                      downtilt_deg: float, slant_deg: float) -> tuple:
    """
    Convert a GCS direction (zenith theta, azimuth phi) to the array LCS direction
    (theta', phi') per 3GPP TR 38.901 eq. 7.1-7 / 7.1-8. Used to evaluate the element
    radiation pattern, which is defined in the array LCS. Identity when angles are 0.

    theta/phi and the returned values are in degrees. Zenith is measured from +z.
    """
    if bearing_deg == 0.0 and downtilt_deg == 0.0 and slant_deg == 0.0:
        return theta_deg, phi_deg
    a = np.deg2rad(bearing_deg)
    b = np.deg2rad(downtilt_deg)
    g = np.deg2rad(slant_deg)
    th = np.deg2rad(theta_deg)
    dphi = np.deg2rad(phi_deg) - a

    cos_tp = (np.cos(b) * np.cos(g) * np.cos(th)
              + (np.sin(b) * np.cos(g) * np.cos(dphi) - np.sin(g) * np.sin(dphi)) * np.sin(th))
    cos_tp = np.clip(cos_tp, -1.0, 1.0)
    theta_p = np.arccos(cos_tp)

    real = np.cos(b) * np.sin(th) * np.cos(dphi) - np.sin(b) * np.cos(th)
    imag = (np.cos(b) * np.sin(g) * np.cos(th)
            + (np.sin(b) * np.sin(g) * np.cos(dphi) + np.cos(g) * np.sin(dphi)) * np.sin(th))
    phi_p = np.angle(real + 1j * imag)

    return np.rad2deg(theta_p), np.rad2deg(phi_p)
