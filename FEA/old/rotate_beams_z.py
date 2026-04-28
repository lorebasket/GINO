import numpy as np

def get_rotation_matrix_z(angle_deg):
    """
    Returns the rotation matrix for a given angle around the Z-axis.
    """
    angle_rad = np.deg2rad(float(angle_deg))
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    R = np.array([
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=float)
    return R

def get_T6_z(angle_deg):
    """
    Returns the 6x6 transformation matrix for a Z-axis rotation.
    """
    R = get_rotation_matrix_z(angle_deg)
    T = np.zeros((6, 6))
    T[:3, :3] = R  # Translations
    T[3:, 3:] = R  # Rotations
    return T

def rotate_beam_model_z(beam_model, angle_deg, rotation_center=None):
    """
    Rotates the beam model around the Z-axis.
    If rotation_center is provided, it rotates around that point.
    Otherwise, it rotates around the origin.
    """
    # 1. Rotate node positions
    R = get_rotation_matrix_z(angle_deg)
    
    if rotation_center is None:
        rotation_center = np.array([0.0, 0.0, 0.0])
    else:
        rotation_center = np.array(rotation_center)

    for node in beam_model["nodes"]:
        pos = np.array(node["position"])
        node["position"] = ((pos - rotation_center) @ R.T + rotation_center).tolist()

    # 2. Rotate all per-element K and M matrices
    T = get_T6_z(angle_deg)
    for elem in beam_model["elements"]:
        K = np.array(elem["stiffness"], dtype=float)
        M = np.array(elem["mass"], dtype=float)
        
        if K.shape != (6, 6):
            raise ValueError("Stiffness matrix K must be of shape 6x6.")
        if M.shape != (6, 6):
            raise ValueError("Mass matrix M must be of shape 6x6.")
            
        elem["stiffness"] = (T @ K @ T.T).tolist()
        elem["mass"] = (T @ M @ T.T).tolist()

    return beam_model
