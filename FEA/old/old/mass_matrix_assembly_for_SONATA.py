import numpy as np

def _element_transform(nodes, n1, n2):
    """
    Build 3x3 R and 12x12 T for element (n1,n2) from node positions.
    Local ex is along the element; ey/ez built from a stable reference.
    """
    p1 = np.asarray(nodes[n1]["position"], float)
    p2 = np.asarray(nodes[n2]["position"], float)
    print(f"Element nodes {n1}-{n2} positions: {p1} to {p2}")

    breakpoint()  # Debug: check node positions and element orientation

    ey = p2 - p1 # Global y-axis along the element
    L = np.linalg.norm(ey)
    print(f"Element length L: {L:.4f} m")
    breakpoint()  # Debug: check element length and direction vector before normalization
    if L <= 0:
        raise ValueError("Zero-length element detected")
    ey = ey / L

    # stable reference for plane construction
    k = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(ey, k)) > 0.99:
        k = np.array([0.0, 1.0, 0.0])
    
    ex = np.cross(k, ey); ex /= np.linalg.norm(ex)
    ez = np.cross(ey, ex)

    # 3x3 rotation local→global (rows are basis vectors)
    R = np.vstack((ex, ey, ez))  # (3,3)

    # 12x12 block-diagonal T = diag(R,R,R,R)
    T = np.zeros((12, 12))
    for a in range(4):
        T[3*a:3*a+3, 3*a:3*a+3] = R
    return R, T


def calculate_shape_functions(y, L):
    """Calculate the shape function matrix N for a beam element.

    Args:
        y: position along the element (local coordinate, 0 <= y <= L)
        L: length of the element
    Returns:
        N_matrix: 6x12 shape function matrix
    """
    N_matrix = np.zeros((6, 12))
    yi = y / L

    # Linear shape functions (N1, N2) for axial and rotational DOFs
    N1_val = 1 - yi
    N2_val = yi
    # Cubic Hermite shape functions (H1, H2, H3, H4) for transverse displacements
    H1_val = 1 - 3 * yi**2 + 2 * yi**3       # for v1, w1
    H3_val = 3 * yi**2 - 2 * yi**3           # for v2, w2
    
    # Row 0: u (axial displacement)
    N_matrix[0, 0] = N1_val  # u1
    N_matrix[0, 6] = N2_val  # u2

    # Row 1: v (transverse displacement in y)
    N_matrix[1, 1] = H1_val  # v1
    N_matrix[1, 7] = H3_val  # v2

    # Row 2: w (transverse displacement in z)
    N_matrix[2, 2] = H1_val  # w1
    N_matrix[2, 8] = H3_val  # w2

    # Row 3: θx (torsional rotation)
    N_matrix[3, 3] = N1_val  # θx1
    N_matrix[3, 9] = N2_val  # θx2

    # Row 4: θy (bending rotation about y - independent field)
    N_matrix[4, 4] = N1_val  # θy1
    N_matrix[4, 10] = N2_val # θy2

    # Row 5: θz (bending rotation about z)
    N_matrix[5, 5] = N1_val  # θz1
    N_matrix[5, 11] = N2_val # θz2

    return N_matrix

def calculate_element_mass_matrix(element_mass, L):
    """
    Calculate the element mass matrix using Gaussian quadrature
    
    Parameters:
    element_mass: 6x6 element mass matrix
    L: Length of beam element
    num_gauss_points: Number of Gaussian quadrature points
    
    M = ∫ Nᵀ * ρA * N dx


    Returns:
    K_element: 12x12 element stiffness matrix
    """
        # Initialize stiffness matrix
    M_element = np.zeros((12, 12))

    # Gauss points and weights for numerical integration
    gauss_points = np.array([-np.sqrt(1/3), np.sqrt(1/3)])
    gauss_weights = np.array([1.0, 1.0])

    # Map Gauss points from [-1, 1] to [0, L]
    y_points = L/2 * (gauss_points + 1)
    
    # Numerical integration using Gaussian quadrature
    for i, (y,w) in enumerate(zip(y_points, gauss_weights)):
        # Calculate B matrix at Gauss point
        N_matrix = calculate_shape_functions(y, L)
        
        # Calculate integrand B^T * D * B
        integrand = np.matmul(np.matmul(N_matrix.T, element_mass), N_matrix)
        
        # Add contribution to K (including Jacobian L/2 and weight)
        M_element += integrand * w * (L/2)
    
    return M_element

def assemble_global_mass_matrix(beam_model, flutter_benchmark, agard_theory, M_sec):
    """
    Calculate the global stiffness matrix for the entire beam model
    """
    # Get beam model from properties
    total_dof = len(beam_model["nodes"])*6
    # Initialize global stiffness matrix
    M_global = np.zeros((total_dof, total_dof))
    
    # Loop through all elements
    for i, element in enumerate(beam_model["elements"]):
        # Get nodes of this element
        node1_idx = element["nodes"][0]
        node2_idx = element["nodes"][1]
        
        element_length = element["length"]
        
        
        # Calculate element stiffness matrix
        if agard_theory:
            M_element = element["mass"]

        elif flutter_benchmark:
            element_mass = M_sec
            element_length = element["length"]
            M_element = calculate_element_mass_matrix(
                element_mass, 
                element_length
            )
        else:
            element_mass = element["mass"]
            element_length = element["length"]
            M_element = calculate_element_mass_matrix(
                element_mass, 
                element_length
            )
        
        _, T = _element_transform(beam_model["nodes"], node1_idx, node2_idx)
        M_element = T.T @ M_element @ T
        #M_element = M_element

        # Assemble into global stiffness matrix
        dof_per_node = 6
        dof_per_element = 12

        for local_i in range(dof_per_element):
            # Map local DOF to global DOF
            if local_i < dof_per_node:
                global_i = node1_idx * dof_per_node + local_i
            else:
                global_i = node2_idx * dof_per_node + (local_i - dof_per_node)
            
            for local_j in range(dof_per_element):
                # Map local DOF to global DOF
                if local_j < dof_per_node:
                    global_j = node1_idx * dof_per_node + local_j
                else:
                    global_j = node2_idx * dof_per_node + (local_j - dof_per_node)
                
                # Add contribution to global stiffness matrix
                M_global[global_i, global_j] += M_element[local_i, local_j]
    
    return M_global

# Boundary conditions are applied in the analysis code, not in the stiffness matrix calculation

if __name__ == "__main__":
    # Print beam parameters
    print(f"Beam length: {bp.beam_length} m")
    print(f"Cross-section: {bp.b} x {bp.h} m")
    print(f"Young's modulus: {bp.E/1e9} GPa")
    print(f"Number of elements: {bp.n_el}")
    print(f"Element length: {bp.element_length} m")
    print(f"Total DOFs: {bp.total_dof}")
    
    # Calculate global stiffness matrix
    M_global = assemble_global_mass_matrix()
    
    print(f"\nGlobal mass matrix size: {M_global.shape}")
    
    # Check conditioning of the mass matrix
    try:
        cond_num = np.linalg.cond(M_global)
        print(f"Condition number of global mass matrix: {cond_num:.2e}")
    except:
        print("Could not calculate condition number - matrix may be singular")
    
    # Show a small portion of the mass matrix for verification
    print("\nSample of global mass matrix (10x10 corner):")
    np.set_printoptions(precision=2, suppress=True)
    print(M_global[:10, :10])
    
    # Save matrix to file for further analysis
    np.save("M_global.npy", M_global)
    
    print("\nMass matrix saved to M_global.npy")