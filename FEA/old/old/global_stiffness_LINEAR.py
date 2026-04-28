import numpy as np

def calculate_B_matrix_linear(x, L):
    """
    Calculate the strain-displacement matrix B for a 2-node beam element
    using LINEAR shape functions for ALL DOFs (6 independent DOFs per node).
    
    This formulation treats all rotations as independent (no kinematic constraints).
    
    For a beam element aligned along Y-axis:
    - Strains/curvatures are derivatives of displacements/rotations along Y
    - All derivatives use linear shape function derivatives
    
    DOF ordering: [u, v, w, θx, θy, θz]
    Element DOFs: [u1,v1,w1,θx1,θy1,θz1, u2,v2,w2,θx2,θy2,θz2]
    
    Returns:
        B: 6×12 matrix relating element DOFs to generalized strains
    """
    B = np.zeros((6, 12), dtype=float)
    
    # Linear shape function derivatives: dN/dx
    dN1_dx = -1.0 / L  # derivative of N1 = (1 - x/L)
    dN2_dx = 1.0 / L   # derivative of N2 = x/L
    
    # Row 0: γ_xy = ∂u/∂y (engineering shear strain in XY plane)
    # For beam along Y: this is the derivative of X-displacement
    B[0, 0] = dN1_dx   # from u1
    B[0, 6] = dN2_dx   # from u2
    
    # Row 1: ε_y = ∂v/∂y (axial strain along beam Y-axis)
    B[1, 1] = dN1_dx   # from v1
    B[1, 7] = dN2_dx   # from v2
    
    # Row 2: γ_yz = ∂w/∂y (engineering shear strain in YZ plane)
    # For beam along Y: this is the derivative of Z-displacement
    B[2, 2] = dN1_dx   # from w1
    B[2, 8] = dN2_dx   # from w2
    
    # Row 3: κ_x = ∂θx/∂y (curvature about X-axis, flapwise bending)
    B[3, 3] = dN1_dx   # from θx1
    B[3, 9] = dN2_dx   # from θx2
    
    # Row 4: κ_y = ∂θy/∂y (twist rate, torsion along Y-axis)
    B[4, 4] = dN1_dx   # from θy1
    B[4, 10] = dN2_dx  # from θy2
    
    # Row 5: κ_z = ∂θz/∂y (curvature about Z-axis, chordwise bending)
    B[5, 5] = dN1_dx   # from θz1
    B[5, 11] = dN2_dx  # from θz2
    
    return B


def calculate_element_stiffness_matrix(sectional_stiffness, L):
    """
    Compute the 12×12 element stiffness matrix using linear strain-displacement
    relationship for 3D beam with 6 independent DOFs per node.
    
    Ke = ∫_0^L B^T * C * B dx
    
    where:
    - B is the 6×12 strain-displacement matrix (linear)
    - C is the 6×6 sectional stiffness matrix
    - L is the element length
    
    Uses 2-point Gauss quadrature (exact for linear B matrix).
    
    Args:
        sectional_stiffness: 6×6 numpy array (sectional stiffness)
        L: Element length
        
    Returns:
        Ke: 12×12 element stiffness matrix
    """
    Ke = np.zeros((12, 12), dtype=float)

    # Two-point Gauss-Legendre quadrature in [-1, 1]
    gauss_pts = np.array([-np.sqrt(1.0/3.0), np.sqrt(1.0/3.0)])
    weights = np.array([1.0, 1.0])

    for xi_hat, w in zip(gauss_pts, weights):
        # Map xi_hat from [-1, 1] to physical coordinate x in [0, L]
        x = (L / 2.0) * (xi_hat + 1.0)
        
        # Compute B matrix at this integration point
        B = calculate_B_matrix_linear(x, L)  # 6×12
        
        # Compute integrand: B^T * C * B
        integrand = B.T @ sectional_stiffness @ B  # 12×12
        
        # Add weighted contribution (Jacobian = L/2)
        Ke += integrand * w * (L / 2.0)

    return Ke


def assemble_global_stiffness_matrix(beam_model):
    """
    Assemble the global stiffness matrix for the beam model.
    
    Uses linear strain-displacement relationships consistent with
    6 independent DOFs per node (no kinematic constraints).
    
    Args:
        beam_model: Dictionary containing:
            - 'nodes': list of nodes
            - 'elements': list of elements where each has:
                - 'nodes': [i, i+1]
                - 'stiffness': 6×6 sectional stiffness
                - 'length': L_e
                
    Returns:
        K_global: (n_nodes*6 × n_nodes*6) global stiffness matrix
    """
    n_nodes = len(beam_model["nodes"])
    total_dof = n_nodes * 6
    K_global = np.zeros((total_dof, total_dof), dtype=float)

    dof_per_node = 6

    for element in beam_model["elements"]:
        node1_idx, node2_idx = element["nodes"]
        L = element["length"]
        sectional_stiffness = element["stiffness"]  # 6×6
        
        # Compute 12×12 element stiffness matrix
        Ke = calculate_element_stiffness_matrix(sectional_stiffness, L)

        # Build local->global index map
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

        # Assemble into global stiffness matrix
        K_global[np.ix_(local_to_global, local_to_global)] += Ke

    return K_global
