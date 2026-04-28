import numpy as np

def calculate_shape_functions_linear(x, L):
    """
    Calculate the shape function matrix N for a 2-node beam element with
    6 INDEPENDENT DOFs per node (no kinematic constraints).
    
    Uses LINEAR interpolation for ALL DOFs.
    
    DOF ordering (per node): [u, v, w, θx, θy, θz]
    Element local DOFs (12): [u1,v1,w1,θx1,θy1,θz1, u2,v2,w2,θx2,θy2,θz2]

    This formulation treats ALL rotations (θx, θy, θz) as independent DOFs,
    which is correct for 3D beam elements used in flutter analysis.

    Returns:
        N_matrix: 6 x 12 matrix where each DOF is linearly interpolated
    """
    N_matrix = np.zeros((6, 12), dtype=float)
    xi = x / L  # Normalized coordinate ∈ [0, 1]

    # Linear shape functions
    N1 = 1.0 - xi  # Shape function for node 1
    N2 = xi         # Shape function for node 2

    # Row 0: u (displacement in X direction)
    N_matrix[0, 0] = N1   # u1
    N_matrix[0, 6] = N2   # u2

    # Row 1: v (displacement in Y direction - axial along beam)
    N_matrix[1, 1] = N1   # v1
    N_matrix[1, 7] = N2   # v2

    # Row 2: w (displacement in Z direction - vertical)
    N_matrix[2, 2] = N1   # w1
    N_matrix[2, 8] = N2   # w2

    # Row 3: θx (rotation about X axis)
    N_matrix[3, 3] = N1   # θx1
    N_matrix[3, 9] = N2   # θx2

    # Row 4: θy (rotation about Y axis - torsion)
    N_matrix[4, 4] = N1   # θy1
    N_matrix[4, 10] = N2  # θy2

    # Row 5: θz (rotation about Z axis)
    N_matrix[5, 5] = N1   # θz1
    N_matrix[5, 11] = N2  # θz2

    return N_matrix


def calculate_element_mass_matrix(element_mass, L):
    """
    Compute a consistent 12x12 element mass matrix using LINEAR shape functions:
        M_e = ∫_0^L N^T * element_mass * N dx
    
    where element_mass is the 6x6 sectional mass matrix (per unit length)
    in the DOF ordering: [u,v,w,θx,θy,θz].

    This formulation uses linear shape functions for ALL DOFs (including rotations),
    which is appropriate for 3D beam elements with independent rotational DOFs.

    Args:
        element_mass: 6×6 numpy array (sectional mass matrix per unit length)
        L: Element length

    Returns:
        M_element: 12x12 numpy array (element mass matrix)
    """
    M_element = np.zeros((12, 12), dtype=float)

    # 2-point Gauss quadrature (exact for cubic polynomials)
    # With linear shape functions, N^T * M * N is quadratic, so 2-point is exact
    gauss_points = np.array([-1.0/np.sqrt(3), 1.0/np.sqrt(3)])
    gauss_weights = np.array([1.0, 1.0])

    for xi, w in zip(gauss_points, gauss_weights):
        # Map xi ∈ [-1,1] to x ∈ [0,L]: x = (L/2)*(xi + 1)
        x = 0.5 * L * (xi + 1.0)
        
        # Compute shape function matrix at this point
        N = calculate_shape_functions_linear(x, L)   # 6 x 12
        
        # Compute integrand: N^T * M_section * N
        integrand = N.T @ element_mass @ N    # 12 x 12
        
        # Add weighted contribution (Jacobian = L/2)
        M_element += integrand * w * (L/2.0)

    return M_element


def assemble_global_mass_matrix(beam_model):
    """
    Assemble the global consistent mass matrix for the beam model.
    
    Uses linear shape functions for all 6 DOFs per node, which treats
    all rotations as independent (no kinematic constraints).
    
    This is the standard formulation for 3D beam elements in structural
    dynamics and flutter analysis.

    Args:
        beam_model: Dictionary containing:
            - 'nodes': list of nodes (each with "position" and "index")
            - 'elements': list of elements where each element has:
                - 'nodes': [i, i+1]
                - 'mass': 6x6 sectional mass matrix (per unit length)
                - 'length': L_e

    Returns:
        M_global: (n_nodes*6 x n_nodes*6) global mass matrix
    """
    n_nodes = len(beam_model["nodes"])
    total_dof = n_nodes * 6
    M_global = np.zeros((total_dof, total_dof), dtype=float)

    dof_per_node = 6

    for element in beam_model["elements"]:
        node1_idx, node2_idx = element["nodes"]
        L = element["length"]
        element_mass = element["mass"]  # 6x6 sectional mass (per unit length)
        
        # Compute 12x12 element mass matrix
        M_e = calculate_element_mass_matrix(element_mass, L)

        # Build local->global index map for element DOFs
        local_to_global = np.array([
            node1_idx * dof_per_node + 0,  # u1
            node1_idx * dof_per_node + 1,  # v1
            node1_idx * dof_per_node + 2,  # w1
            node1_idx * dof_per_node + 3,  # θx1
            node1_idx * dof_per_node + 4,  # θy1
            node1_idx * dof_per_node + 5,  # θz1
            node2_idx * dof_per_node + 0,  # u2
            node2_idx * dof_per_node + 1,  # v2
            node2_idx * dof_per_node + 2,  # w2
            node2_idx * dof_per_node + 3,  # θx2
            node2_idx * dof_per_node + 4,  # θy2
            node2_idx * dof_per_node + 5,  # θz2
        ], dtype=int)

        # Assemble into global mass matrix
        M_global[np.ix_(local_to_global, local_to_global)] += M_e

    return M_global
