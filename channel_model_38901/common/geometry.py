import numpy as np

def get_antenna_positions(array_type: str, num_rows: int, num_cols: int, 
                          spacing_h_m: float, spacing_v_m: float) -> np.ndarray:
    """
    Calculates the 3D positions of antenna elements in an array.
    The array is assumed to be in the y-z plane.

    Args:
        array_type: 'ULA' or 'URA'.
        num_rows: Number of rows in the array.
        num_cols: Number of columns in the array.
        spacing_h_m: Horizontal spacing between elements in meters.
        spacing_v_m: Vertical spacing between elements in meters.

    Returns:
        A numpy array of shape (num_antennas, 3) with the 3D coordinates.
    """
    positions = []
    if array_type == 'ULA':
        num_rows = 1 # Force rows to 1 for ULA

    for r in range(num_rows):
        for c in range(num_cols):
            # URA/ULA in y-z plane, starting from origin
            pos = np.array([0.0, c * spacing_h_m, r * spacing_v_m])
            positions.append(pos)
            
    return np.array(positions)
