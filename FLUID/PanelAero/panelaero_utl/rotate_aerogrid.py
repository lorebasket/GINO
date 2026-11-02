def _rotation(aerogrid, angle_rad, axis='y'):

    import numpy as np

    ## Convert to radians
    #angle_rad = np.radians(angle_deg)

    # Rotation matrices for x, y, z
    if axis.lower() == 'x':
        R = np.array([
            [1, 0, 0],
            [0, np.cos(angle_rad), -np.sin(angle_rad)],
            [0, np.sin(angle_rad),  np.cos(angle_rad)]
        ])
    elif axis.lower() == 'y':
        R = np.array([
            [ np.cos(angle_rad), 0, np.sin(angle_rad)],
            [0, 1, 0],
            [-np.sin(angle_rad), 0, np.cos(angle_rad)]
        ])
    elif axis.lower() == 'z':
        R = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad), 0],
            [np.sin(angle_rad),  np.cos(angle_rad), 0],
            [0, 0, 1]
        ])
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'.")

    # Keys containing 3D coordinates
    coord_keys = [
        'offset_l', 'offset_k', 'offset_j',
        'offset_P1', 'offset_P3', 'r', 'N'
    ]

    # Rotate each coordinate array
    for key in coord_keys:
        if key in aerogrid:
            aerogrid[key] = aerogrid[key] @ R.T

    # Rotate cornerpoint_grids (skip first column: IDs)
    if 'cornerpoint_grids' in aerogrid:
        aerogrid['cornerpoint_grids'][:, 1:4] = (
            aerogrid['cornerpoint_grids'][:, 1:4] @ R.T
        )

    # Rotate centroids (skip first column: IDs)
    if 'centroids' in aerogrid:
        aerogrid['centroids'][:, 1:4] = (
            aerogrid['centroids'][:, 1:4] @ R.T
        )

    return aerogrid