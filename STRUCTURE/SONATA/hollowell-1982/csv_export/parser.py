import numpy as np

def parse_sectional_matrix_csv(file_path):
    matrices = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Detect header: "section in r/R,0.0"
        if line.startswith("section in r/R"):
            section = line.split(',')[1].strip()
            i += 1  # move to next line

            # Read next 6 lines as a 6x6 matrix
            matrix_lines = lines[i:i+6]
            matrix = []
            for l in matrix_lines:
                row = [float(x) for x in l.strip().split(',') if x]
                matrix.append(row)
            matrices[section] = np.array(matrix)
            i += 6  # skip past this matrix
        else:
            i += 1  # skip unrelated lines

    return matrices

def parse_section_props_csv(file_path):
    section_props = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()

    header = lines[0].strip().split(',')
    data_lines = lines[1:]

    columns = {name: [] for name in header}

    for line in data_lines:
        values = line.strip().split(',')
        for name, value in zip(header, values):
            columns[name].append(float(value))
    
    return columns

# === TRANFORM MATRICES === #

def transform_matrices(file_name, K, M):
    """
    Transform stiffness and mass matrices from Beam reference Axis to Shear Center.
    """
    import numpy as np
    
    # Store input type to return the same type
    input_is_dict = isinstance(K, dict)
    
    # Convert to dictionary format for processing
    if input_is_dict:
        K_dict = K.copy()
        M_dict = M.copy()
    else:
        K_dict = {'0': K}
        M_dict = {'0': M}
    
    # Extract sections props from csv file
    sections_props = parse_section_props_csv(file_name)
    eta_keys = sections_props['eta']  # Get all section keys from CSV
    
    # Initialize position arrays
    SC_X = np.array(sections_props['X'])
    SC_Y = np.array(sections_props['SC_X'])
    SC_Z = np.array(sections_props['SC_Y'])
    
    # Process each section that exists in our matrices
    for i, section_key in enumerate(eta_keys):
        section_key = str(section_key)
        
        # Skip if this section isn't in our matrices
        if section_key not in K_dict:
            continue
            
        dx = SC_Y[0]
        dy = 0
        dz = SC_Z[0]
        
        # Skip if no translation needed
        if abs(dx) < 1e-10 and abs(dy) < 1e-10 and abs(dz) < 1e-10:
            continue
            
        T = translate_stiffness_matrix([dx, dy, dz])
        K_dict[section_key] = T @ K_dict[section_key] @ T.T
        M_dict[section_key] = T @ M_dict[section_key] @ T.T
    
    # Convert back to original format if needed
    if not input_is_dict:
        return K_dict['0'], M_dict['0']
    return K_dict, M_dict

def skew(v):
    """Returns the skew-symmetric matrix of a 3D vector."""
    return np.array([
        [ 0,   -v[2],  v[1]],
        [ v[2],  0,   -v[0]],
        [-v[1], v[0],  0]
    ])

def translate_stiffness_matrix(d):
    """
    Translates stiffness matrix from REF to SC.
    
    Parameters:
    - K_NA: 6x6 stiffness matrix about the Neutral Axis
    - d:    3-element vector (dx, dy, dz) from REF to SC

    Returns:
    - K_SC: 6x6 stiffness matrix about the Shear Center
    """
    S = skew(d)
    I = np.eye(3)
    Z = np.zeros((3, 3))

    T = np.block([
        [ I,     Z ],
        [ S,     I ]
    ])

    return T


def build_rotation_matrix_3x3(axis, angle):
    """
    Build a 3x3 rotation matrix for rotation about a specified axis.
    
    Parameters:
    - axis: string ('x', 'y', 'z') or 3D unit vector
    - angle: rotation angle in radians
    """
    c = np.cos(angle)
    s = np.sin(angle)
    
    if isinstance(axis, str):
        if axis.lower() == 'x':
            R = np.array([
                [1,  0,  0],
                [0,  c, -s],
                [0,  s,  c]
            ])
        elif axis.lower() == 'y':
            R = np.array([
                [ c,  0,  s],
                [ 0,  1,  0],
                [-s,  0,  c]
            ])
        elif axis.lower() == 'z':
            R = np.array([
                [ c, -s,  0],
                [ s,  c,  0],
                [ 0,  0,  1]
            ])
        else:
            raise ValueError("Axis must be 'x', 'y', 'z', or a 3D unit vector")
    else:
        # Rodrigues' rotation formula for arbitrary axis
        axis = np.array(axis)
        axis = axis / np.linalg.norm(axis)  # Normalize to unit vector
        
        K = skew(axis)
        R = np.eye(3) + s * K + (1 - c) * K @ K
    
    return R

def rotate_matrices_by_angle(K, M, axis, angle, section_keys=None):

    # Get 3x3 rotation matrix
    R3 = build_rotation_matrix_3x3(axis, angle)
    
    # Build 6x6 rotation matrix
    R6 = np.block([
        [R3,                np.zeros((3, 3))],
        [np.zeros((3, 3)),  R3               ]
    ])
    
    # Handle case where K and M are single matrices
    if isinstance(K, np.ndarray) and isinstance(M, np.ndarray):
        return R6 @ K @ R6.T, R6 @ M @ R6.T
    
    # Handle case where K and M are dictionaries
    elif isinstance(K, dict) and isinstance(M, dict):
        K_rotated = {}
        M_rotated = {}
        
        # Determine which sections to rotate
        keys_to_rotate = section_keys if section_keys is not None else K.keys()
        
        # Apply rotation to each specified section
        for key in keys_to_rotate:
            if key in K and key in M:
                K_rotated[key] = R6 @ K[key] @ R6.T
                M_rotated[key] = R6 @ M[key] @ R6.T
            else:
                print(f"Warning: Section key '{key}' not found in matrices")
        
        return K_rotated, M_rotated
    
    else:
        raise ValueError("K and M must both be either NumPy arrays or dictionaries")