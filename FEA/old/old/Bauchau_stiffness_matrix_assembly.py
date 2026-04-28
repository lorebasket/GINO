import numpy as np
#import beam_properties as bp

## new for pitch != 0
def _element_transform(nodes, n1, n2, sonata, flutter_benchmark):
    """
    Build 3x3 R and 12x12 T for element (n1,n2) from node positions.
    Local ex is along the element; ey/ez built from a stable reference.
    """
    p1 = np.asarray(nodes[n1]["position"], float)  # Node 1 position (x, y, z)
    p2 = np.asarray(nodes[n2]["position"], float)  # Node 2 position
    
    # Local first-axis: along the element
    e1 = (p2 - p1)                             
    
    L = np.linalg.norm(e1)
    if L <= 0:
        raise ValueError("Zero-length element detected")
    e1 = e1 / L

    # stable reference for plane construction
    k = np.array([0.0, 0.0, 1.0])                   # Reference vector (global Z)
    if abs(np.dot(e1, k)) > 0.99:
        k = np.array([0.0, 1.0, 0.0])
    
    # Local second-axis: perpendicular to both e1 and k
    e2 = np.cross(k, e1); e2 /= np.linalg.norm(e2)
    # Local third-axis: perpendicular to both e1 and e2
    e3 = np.cross(e1, e2)

    ## FOR SONATA
    if sonata == True:
        # 3x3 rotation from local→global (rows are basis vectors)
        R = np.vstack((e1, e2, e3))  # shape (3,3)

        # Lift to 12x12 block-diagonal T = diag(R,R,R,R)
        T = np.zeros((12, 12))
        for a in range(4):
            T[3*a:3*a+3, 3*a:3*a+3] = R

    ## FOR GOLAND BENCHMARK
    if flutter_benchmark == True:
        # columns are local basis expressed in global
        R = np.column_stack((e1, e2, e3))    # shape (3,3)
        
        # Lift to 12x12 block-diagonal T = diag(R.T,R.T,R.T,R.T)
        T = np.zeros((12,12))
        for a in range(4):
            T[3*a:3*a+3, 3*a:3*a+3] = R.T    # so that q_local = R.T q_global


    return R, T

def calculate_element_stiffness_matrix(sectional_stiffness, L):
    """
    Compute the 12x12 element stiffness matrix using Bauchau's linear formulation:
    Ke = ∫ B(x)^T * C * B(x) dx

    Parameters:
    - sectional_stiffness: 6x6 matrix (from SONATA)
    - L: length of the beam element

    Returns:
    - 12x12 element stiffness matrix
    """
    Ke = np.zeros((12, 12))

    # Two-point Gauss-Legendre quadrature
    gauss_pts = np.array([-np.sqrt(1/3), np.sqrt(1/3)])
    weights = np.array([1.0, 1.0])

    for xi_hat, w in zip(gauss_pts, weights):
        # Map xi_hat from [-1, 1] to physical x in [0, L]
        x = (L / 2) * (xi_hat + 1)
        B = calculate_B_matrix(x, L)
        Ke += B.T @ sectional_stiffness @ B * w * (L / 2)

    return Ke

def calculate_B_matrix(x, L):
    """
    Construct the 6x12 strain-displacement matrix B(x)
    consistent with Bauchau's linear theory (warping via beta)
    """
    B = np.zeros((6, 12))
    xi = x / L

    # Linear shape functions
    N1 = 1 - xi
    N2 = xi
    dN1 = -1 / L
    dN2 = 1 / L

    # Cubic Hermite shape functions (bending)
    H1 = 1 - 3*xi**2 + 2*xi**3
    H2 = L * (xi - 2*xi**2 + xi**3)
    H3 = 3*xi**2 - 2*xi**3
    H4 = L * (-xi**2 + xi**3)

    dH1 = (-6*xi + 6*xi**2) / L
    dH2 = (1 - 4*xi + 3*xi**2)
    dH3 = (6*xi - 6*xi**2) / L
    dH4 = (-2*xi + 3*xi**2)

    beta = 0 #calculate_warping_parameter(sectional_stiffness)

    # εx = du/dx - beta*θx
    B[0, 0] = dN1
    B[0, 6] = dN2
    B[0, 3] = -beta * N1
    B[0, 9] = -beta * N2

    # γxy = dv/dx - θz
    B[1, 1] = dH1
    B[1, 7] = dH3
    B[1, 5] = -N1
    B[1, 11] = -N2

    # γxz = dw/dx - θy
    B[2, 2] = dH1
    B[2, 8] = dH3
    B[2, 4] = -N1
    B[2, 10] = -N2

    # κx = dθx/dx - beta*du/dx
    B[3, 3] = dN1
    B[3, 9] = dN2
    B[3, 0] = -beta * N1
    B[3, 6] = -beta * N2

    # κy = dθy/dx
    B[4, 4] = dN1
    B[4, 10] = dN2

    # κz = dθz/dx
    B[5, 5] = dN1
    B[5, 11] = dN2

    return B


def calculate_element_stiffness_matrix_flutter(sectional_stiffness, L, beta=0.0, n_gauss=2):
    """
    Compute the 12x12 element stiffness matrix for flutter_benchmark using
    Timoshenko beam kinematics:
      εx  = du/dx - beta*θx
      γxy = dv/dx - θz
      γxz = dw/dx - θy
      κx  = dθx/dx - beta*du/dx
      κy  = dθy/dx
      κz  = dθz/dx

    DOF order per node: [u, v, w, θx, θy, θz]  -> total element DOFs:
      q_e = [u1 v1 w1 θx1 θy1 θz1  u2 v2 w2 θx2 θy2 θz2]

    Parameters
    ----------
    sectional_stiffness : (6,6) ndarray
        Sectional stiffness matrix C for [εx, γxy, γxz, κx, κy, κz].
    L : float
        Element length.
    beta : float, optional
        Warping coupling parameter (default 0.0).
    n_gauss : int, optional
        Number of Gauss points (2 or 3 recommended). Default 2.

    Returns
    -------
    Ke : (12,12) ndarray
        Element stiffness matrix.
    """
    C = sectional_stiffness
    Ke = np.zeros((12, 12))

    # Gauss-Legendre points and weights (2- or 3-point)
    if n_gauss == 2:
        gp = np.array([-np.sqrt(1/3),  np.sqrt(1/3)])
        gw = np.array([1.0, 1.0])
    elif n_gauss == 3:
        gp = np.array([-np.sqrt(3/5), 0.0, np.sqrt(3/5)])
        gw = np.array([5/9, 8/9, 5/9])
    else:
        raise ValueError("n_gauss must be 2 or 3.")

    for xi_hat, w in zip(gp, gw):
        # map [-1,1] -> x in [0,L]
        x = 0.5 * L * (xi_hat + 1.0)
        B = calculate_B_matrix_flutter(x, L, beta)

        # Ke += ∫ B^T C B dx  ≈ Σ B^T C B * w * (L/2)
        Ke += B.T @ C @ B * w * (L * 0.5)

    return Ke


def calculate_B_matrix_flutter(x, L, beta=0.0):
    """
    Construct the (6 x 12) B matrix at position x for Timoshenko beam with
    Hermite interpolation for transverse displacement fields.

    Interpolations (ξ = x/L):
      - Axial displacement u, rotations θx, θy, θz: linear (N1, N2)
      - Transverse displacements v, w: cubic Hermite (H1..H4), where slopes
        are the nodal rotations θz (for v) and θy (for w).

    DOF order (element):
      q_e = [u1 v1 w1 θx1 θy1 θz1  u2 v2 w2 θx2 θy2 θz2]

    Strain vector (6):
      [εx, γxy, γxz, κx, κy, κz]

    Returns
    -------
    B : (6,12) ndarray
    """
    B = np.zeros((6, 12))
    xi = x / L

    # Linear shape functions and derivatives
    N1 = 1.0 - xi
    N2 = xi
    dN1 = -1.0 / L
    dN2 =  1.0 / L

    # Cubic Hermite shape functions for v(ξ) and w(ξ)
    # v(ξ) = H1*v1 + H2*θz1 + H3*v2 + H4*θz2
    # w(ξ) = H1*w1 + H2*θy1 + H3*w2 + H4*θy2
    H1 = 1.0 - 3.0*xi**2 + 2.0*xi**3
    H2 = L * (xi - 2.0*xi**2 + xi**3)
    H3 = 3.0*xi**2 - 2.0*xi**3
    H4 = L * (-xi**2 + xi**3)

    dH1 = (-6.0*xi + 6.0*xi**2) / L
    dH2 = (1.0 - 4.0*xi + 3.0*xi**2)
    dH3 = ( 6.0*xi - 6.0*xi**2) / L
    dH4 = (-2.0*xi + 3.0*xi**2)

    # Indices (for readability)
    u1, v1, w1, thx1, thy1, thz1 = 0, 1, 2, 3, 4, 5
    u2, v2, w2, thx2, thy2, thz2 = 6, 7, 8, 9, 10, 11

    # --- εx = du/dx - beta*θx
    # du/dx
    B[0, u1] = dN1
    B[0, u2] = dN2
    # - beta*θx
    B[0, thx1] = -beta * N1
    B[0, thx2] = -beta * N2

    # --- γxy = dv/dx - θz
    # dv/dx via Hermite interpolation
    B[1, v1]  = dH1
    B[1, thz1] = dH2
    B[1, v2]  = dH3
    B[1, thz2] = dH4
    # - θz
    B[1, thz1] += -N1
    B[1, thz2] += -N2

    # --- γxz = dw/dx - θy
    # dw/dx via Hermite interpolation
    B[2, w1]  = dH1
    B[2, thy1] = dH2
    B[2, w2]  = dH3
    B[2, thy2] = dH4
    # - θy
    B[2, thy1] += -N1
    B[2, thy2] += -N2

    # --- κx = dθx/dx - beta*du/dx
    # dθx/dx
    B[3, thx1] = dN1
    B[3, thx2] = dN2
    # - beta*du/dx
    B[3, u1] += -beta * dN1
    B[3, u2] += -beta * dN2

    # --- κy = dθy/dx
    B[4, thy1] = dN1
    B[4, thy2] = dN2

    # --- κz = dθz/dx
    B[5, thz1] = dN1
    B[5, thz2] = dN2

    return B

def assemble_global_stiffness_matrix(beam_model, flutter_benchmark, agard_theory, sonata, K):
    total_dof = len(beam_model["nodes"])*6
    K_global = np.zeros((total_dof, total_dof))

    nodes = beam_model["nodes"]

    for element in beam_model["elements"]:
        # Get node indices for this element
        n1_idx, n2_idx = element["nodes"]
        
        # Get the actual node objects
        n1 = nodes[n1_idx]
        n2 = nodes[n2_idx]
        
        element_length = element["length"]
        
        if agard_theory == True:
            # For AGARD theory, use the provided K matrix
            Ke_loc = element["stiffness"]

        if flutter_benchmark == True:
            # For normal operation, calculate from element properties
            element_stiffness = element["stiffness"]
            Ke_loc = calculate_element_stiffness_matrix_flutter(element_stiffness, element_length)

        if sonata == True:
            # For normal operation, calculate from element properties
            element_stiffness = element["stiffness"]
            Ke_loc = calculate_element_stiffness_matrix(element_stiffness, element_length)
            
        # Transform to global coordinates
        # Use node indices (n1_idx, n2_idx) instead of node objects (n1, n2)
        R, T = _element_transform(beam_model["nodes"], n1_idx, n2_idx, sonata, flutter_benchmark)
        Ke = T.T @ Ke_loc @ T
        #Ke = Ke_loc

        # Assemble into global matrix
        for i in range(12):
            # Use node indices (n1_idx, n2_idx) for global DOF calculation
            gi = n1_idx * 6 + (i % 6) if i < 6 else n2_idx * 6 + (i % 6)
            for j in range(12):
                gj = n1_idx * 6 + (j % 6) if j < 6 else n2_idx * 6 + (j % 6)
                K_global[gi, gj] += Ke[i, j]

    return K_global

def compute_mode_strain_energy_contributions(K_global, mode_shape, n_nodes):
    """
    Compute the strain energy contributions (axial, torsion, bending y, bending z) for a mode.

    Args:
        K_global: Global stiffness matrix (total_dof × total_dof)
        mode_shape: Mode shape vector (total_dof,)
        n_nodes: Number of nodes

    Returns:
        A dictionary with strain energies:
        {
            "Axial": U_axial,
            "Torsion": U_torsion,
            "Bending Y": U_bending_y,
            "Bending Z": U_bending_z,
            "Total": U_total
        }
    """
    dof_per_node = 6
    
    # DOF indices per deformation type
    axial_dofs = [n * dof_per_node + 0 for n in range(n_nodes)]
    torsion_dofs = [n * dof_per_node + 3 for n in range(n_nodes)]
    bending_y_dofs = [n * dof_per_node + 1 for n in range(n_nodes)] + \
                     [n * dof_per_node + 5 for n in range(n_nodes)]
    bending_z_dofs = [n * dof_per_node + 2 for n in range(n_nodes)] + \
                     [n * dof_per_node + 4 for n in range(n_nodes)]

    # Helper function to compute strain energy for a DOF group
    def strain_energy(dof_indices):
        u_sub = mode_shape[dof_indices]
        K_sub = K_global[np.ix_(dof_indices, dof_indices)]
        return 0.5 * u_sub.T @ K_sub @ u_sub

    # Compute strain energies
    U_axial = strain_energy(axial_dofs)
    U_torsion = strain_energy(torsion_dofs)
    U_bending_y = strain_energy(bending_y_dofs)
    U_bending_z = strain_energy(bending_z_dofs)
    
    # Total energy (for verification, can also compute with full vector)
    U_total = 0.5 * mode_shape.T @ K_global @ mode_shape
    
    return {
        "Axial": U_axial,
        "Torsion": U_torsion,
        "Bending Y": U_bending_y,
        "Bending Z": U_bending_z,
        "Total": U_total
    }
