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
"""
Transform stiffness and mass matrices from Natural Axis to Shear Center.
"""

def transform_matrices(file_name, K, M, beam_length):
    
    # Extract sections props from csv file
    sections_props = parse_section_props_csv(file_name)
    # Calculate arrays of positions for each section while maintaining string keys
    eta_keys = sections_props['eta']  # Keep as string keys
    NA_X = np.array([float(eta) * beam_length for eta in eta_keys])
    NA_Y = np.array(sections_props['NA_X'])
    NA_Z = np.array(sections_props['NA_Y'])
    
    SC_X = NA_X.copy()  # SC_X is same as NA_X for each section
    SC_Y = np.array(sections_props['SC_X'])
    SC_Z = np.array(sections_props['SC_Y'])
    
    # Translation from NA to SC
    for i in range(len(eta_keys)):
        # Use the original string key
        section_key = str(eta_keys[i])  # Ensure key is string
        print(f"\nProcessing section: {section_key}")
        print(f"K keys: {list(K.keys())}")
        print(f"M keys: {list(M.keys())}")
        
        dx = SC_X[i] - NA_X[i]
        dy = SC_Y[i] - NA_Y[i]
        dz = SC_Z[i] - NA_Z[i]
        print(f"Displacements: dx={dx}, dy={dy}, dz={dz}")

        T = translate_stiffness_matrix([dx, dy, dz])
        print(f"Transformation matrix T:\n{T}")

        # Ensure keys are strings
        k_key = str(section_key)
        m_key = str(section_key)
        print(f"Unpreocessed K[{section_key}]: {K[section_key]}")

        K[k_key] = T @ K[k_key] @ T.T
        M[m_key] = T @ M[m_key] @ T.T
        print(f"Transformed K[{section_key}]: {K[section_key]}")

    # Rotation from CBMlocal_frame to XYZglobal_frame
    for i in range(len(sections_props['eta'])):
        section_key = str(sections_props['eta'][i])  # Get the string key
        # Get the vectors as lists
        x1 = np.array([sections_props['x1_x'][i], sections_props['x1_y'][i], sections_props['x1_z'][i]])
        x2 = np.array([sections_props['x2_x'][i], sections_props['x2_y'][i], sections_props['x2_z'][i]])
        x3 = np.array([sections_props['x3_x'][i], sections_props['x3_y'][i], sections_props['x3_z'][i]])
        
        # Ensure vectors are 3D
        if len(x1) != 3 or len(x2) != 3 or len(x3) != 3:
            raise ValueError(f"Vectors must be 3D at section {i}")
        
        R = build_rotation_matrix_6x6(x1, x2, x3)

        K[section_key] = R @ K[section_key] @ R.T
        M[section_key] = R @ M[section_key] @ R.T

    return K, M

def skew(v):
    """Returns the skew-symmetric matrix of a 3D vector."""
    return np.array([
        [ 0,   -v[2],  v[1]],
        [ v[2],  0,   -v[0]],
        [-v[1], v[0],  0]
    ])

def translate_stiffness_matrix(d):
    """
    Translates stiffness matrix from NA to SC.
    
    Parameters:
    - K_NA: 6x6 stiffness matrix about the Neutral Axis
    - d:    3-element vector (dx, dy, dz) from NA to SC

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

def build_rotation_matrix_6x6(x1, x2, x3):
    """
    Constructs the 6x6 rotation matrix from local to global frame
    using direction cosine matrix (x1, x2, x3 are local basis vectors).
    """
    R = np.column_stack((x1, x2, x3))  # 3x3 rotation matrix
    R6 = np.block([
        [R,         np.zeros((3, 3))],
        [np.zeros((3, 3)), R]
    ])
    return R6