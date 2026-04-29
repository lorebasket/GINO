import numpy as np

def get_rotation_matrix_y(angle_deg):
    angle_rad = np.deg2rad(float(angle_deg[0]))  # Ensure angle_deg is converted to float
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    R = np.array([
        [c, 0.0, s],
        [0.0, 1.0, 0.0],
        [-s, 0.0, c]
    ], dtype=float)  # Explicitly set dtype to float
    return R

def get_T6_y(angle_deg):

    R = get_rotation_matrix_y(angle_deg)
    T = np.zeros((6, 6))
    T[:3, :3] = R      # translations
    T[3:, 3:] = R      # rotations
    return T

def rotate_beam_model_y(beam_model, angle_deg):

    # 1. Rotate node positions
    R = get_rotation_matrix_y(angle_deg)
    for node in beam_model["nodes"]:
        pos = np.array(node["position"])
        node["position"] = (R @ pos).tolist()

    # 2. Rotate all per-element K and M matrices
    T = get_T6_y(angle_deg)
    for elem in beam_model["elements"]:
        # Defensive: ensure arrays and correct shape
        K = np.array(elem["stiffness"], dtype=float)
        M = np.array(elem["mass"], dtype=float)
        assert K.shape == (6, 6), "K must be 6x6"
        assert M.shape == (6, 6), "M must be 6x6"
        elem["stiffness"] = T @ K @ T.T
        elem["mass"] = T @ M @ T.T

    return beam_model

## Example usage:
#if __name__ == "__main__":


#    # Assume beam_model already created via create_beam_model(...)
#    # import or define/create_beam_model, then:
#    # beam_model = create_beam_model(K, M, beam_length, n_elements, pitch, agard_theory, flutter_benchmark, sonata)
#    # Rotate by 2 degrees around y-axis:
#    angle_deg = 2.0
#    beam_model = rotate_beam_model_y(beam_model, angle_deg)
#    print("Beam model rotated by", angle_deg, "degrees about y-axis.")