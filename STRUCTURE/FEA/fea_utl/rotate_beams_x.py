"""
Rotate beam node positions and 6×6 sectional K/M about global +X (pitch).

Same DOF ordering as rotate_beams_y: translations then rotations, T = blkdiag(R, R).
"""

import numpy as np


def get_rotation_matrix_x(angle_deg):
    angle_rad = np.deg2rad(float(angle_deg[0]))
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ],
        dtype=float,
    )


def get_T6_x(angle_deg):
    R = get_rotation_matrix_x(angle_deg)
    T = np.zeros((6, 6), dtype=float)
    T[:3, :3] = R
    T[3:, 3:] = R
    return T


def rotate_beam_model_x(beam_model, angle_deg):
    R = get_rotation_matrix_x(angle_deg)
    for node in beam_model["nodes"]:
        pos = np.array(node["position"], dtype=float)
        node["position"] = (R @ pos).tolist()

    T = get_T6_x(angle_deg)
    for elem in beam_model["elements"]:
        K = np.array(elem["stiffness"], dtype=float)
        M = np.array(elem["mass"], dtype=float)
        assert K.shape == (6, 6) and M.shape == (6, 6)
        elem["stiffness"] = T @ K @ T.T
        elem["mass"] = T @ M @ T.T

    return beam_model
