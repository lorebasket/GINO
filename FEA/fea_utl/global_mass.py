import numpy as np

def _save_global_matrix_to_csv(M_global, config):

    import csv
    
    print(f"\n*** Saving global K and M matrices to CSV files ***")
    
    # Create output directory: output_plots/{config_name}/
    output_dir = os.path.join('output_plots', config.name)
    os.makedirs(output_dir, exist_ok=True)
    
    config_name = config.name
    
    # ========================================================================
    # SAVE GLOBAL MASS MATRIX (M_global)
    # ========================================================================
    matrix_name = f"{M_global=}"
    m_file = os.path.join(output_dir, f"{config_name}_{matrix_name}.csv")
    
    try:
        np.savetxt(m_file, M_global, delimiter=',', fmt='%.10e')
        print(f"  ✓ Saved M_global matrix (shape {M_global.shape}) to: {m_file}")
    except Exception as e:
        print(f"  ✗ Error saving M_global: {e}")
    
    print(f"  Output directory: {output_dir}\n")

def calculate_shape_functions(x, L):
    """
    Calculate the shape function matrix N for a 2-node beam element aligned along Y.
    DOF ordering (per node): [u, v, w, θx, θy, θz]
    Element local DOFs (12): [u1,v1,w1,θx1,θy1,θz1, u2,v2,w2,θx2,θy2,θz2]

    Returns:
        N_matrix: 6 x 12 matrix such that:
                  [u, v, w, θx, θy, θz]^T_at_x = N_matrix @ nodal_dofs_local(12)
    Notes:
        - For transverse (u,w) we use cubic-Hermite:
            u(x) = H1*u1 + H2*θz1 + H3*u2 + H4*θz2
          (H2 and H4 include factor L so nodal rotations are in radians)
        - θx and θz rows are computed as derivatives of transverse shape functions:
            θz(x) = du/dy = d/dx (u(x))  (here y is beam axis, x is local coordinate along element)
          so we compute derivative w.r.t x using d/dx = (1/L)*d/dξ with ξ = x/L.
        - Axial v and torsion θy use linear shape functions.
    """
    N_matrix = np.zeros((6, 12), dtype=float)
    xi = x / L

    # Linear (axial and torsion)
    N_lin_1 = 1.0 - xi
    N_lin_2 = xi

    # Cubic Hermite displacement shape functions (in ξ = x/L)
    H1 = 1 - 3*xi**2 + 2*xi**3
    H2 = L * (xi - 2*xi**2 + xi**3)    # multiplies nodal rotation (slope) - includes L
    H3 = 3*xi**2 - 2*xi**3
    H4 = L * (-xi**2 + xi**3)          # multiplies nodal rotation (slope) - includes L

    # Derivatives of Hermite w.r.t ξ
    dH1_dxi = -6*xi + 6*xi**2
    dH2_dxi = 1 - 4*xi + 3*xi**2
    dH3_dxi = 6*xi - 6*xi**2
    dH4_dxi = -2*xi + 3*xi**2

    # Convert derivatives to d/dx: d/dx = (1/L) * d/dξ
    dH1_dx = dH1_dxi / L
    dH2_dx = dH2_dxi        # note: H2 had L factor, so d(H2)/dx = (1/L)*d(L*...)/dξ = dH2_dxi
    dH3_dx = dH3_dxi / L
    dH4_dx = dH4_dxi        # same reasoning as H2

    # Row 0: u (transverse X)  -> depends on u1, θz1, u2, θz2
    N_matrix[0, 0]  = H1        # u1
    N_matrix[0, 5]  = H2        # θz1
    N_matrix[0, 6]  = H3        # u2
    N_matrix[0, 11] = H4        # θz2

    # Row 1: v (axial along beam Y) -> linear interpolation between v1 and v2
    N_matrix[1, 1] = N_lin_1    # v1
    N_matrix[1, 7] = N_lin_2    # v2

    # Row 2: w (transverse Z) -> depends on w1, θx1, w2, θx2
    N_matrix[2, 2]  = H1        # w1
    N_matrix[2, 3]  = H2        # θx1
    N_matrix[2, 8]  = H3        # w2
    N_matrix[2, 9]  = H4        # θx2

    # Row 3: θx (rotation about X) -> derivative of w interpolation (θx ≈ dw/dy)
    # If your sign convention is θx = -dw/dy, add a negative sign here.
    N_matrix[3, 2]  = dH1_dx    # from w1
    N_matrix[3, 3]  = dH2_dx    # from θx1 (slope nodal DOF)
    N_matrix[3, 8]  = dH3_dx    # from w2
    N_matrix[3, 9]  = dH4_dx    # from θx2

    # Row 4: θy (torsion about beam axis Y) -> linear interpolation (θy1, θy2)
    N_matrix[4, 4]  = N_lin_1
    N_matrix[4, 10] = N_lin_2

    # Row 5: θz (rotation about Z) -> derivative of u interpolation (θz ≈ du/dy)
    # If sign conv is θz = -du/dy, add a negative sign here.
    N_matrix[5, 0]  = dH1_dx    # from u1
    N_matrix[5, 5]  = dH2_dx    # from θz1
    N_matrix[5, 6]  = dH3_dx    # from u2
    N_matrix[5, 11] = dH4_dx    # from θz2

    return N_matrix


def calculate_element_mass_matrix(element_mass, L):
    """
    Compute a consistent 12x12 element mass matrix:
        M_e = ∫_0^L N^T * element_mass * N dx
    where element_mass is the 6x6 sectional/matrix per-node (sectional density/inertia)
    (Element_mass must be in the same DOF ordering: [u,v,w,θx,θy,θz]).

    Returns:
        M_element: 12x12 numpy array
    """
    M_element = np.zeros((12, 12), dtype=float)

    # 2-point Gauss (exact for polynomials up to degree 3-5 depending on integrand)
    gauss_points = np.array([-1.0/np.sqrt(3), 1.0/np.sqrt(3)])
    gauss_weights = np.array([1.0, 1.0])

    for xi, w in zip(gauss_points, gauss_weights):
        # map xi in [-1,1] to x in [0,L] : x = (L/2)*(xi + 1)
        x = 0.5 * L * (xi + 1.0)
        N = calculate_shape_functions(x, L)   # 6 x 12
        integrand = N.T @ element_mass @ N    # 12 x 12
        M_element += integrand * w * (L/2.0)  # Jacobian factor L/2

    return M_element


def assemble_global_mass_matrix(beam_model, config):
    """
    Assemble the global consistent mass matrix for the beam model.
    beam_model expected to contain:
      - 'nodes' : list of nodes (each with "position" and "index")
      - 'elements': list of elements where each element has keys:
           'nodes' : [i, i+1], 'mass': 6x6 local element_mass (sectional),
           'length': L_e
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
        element_mass = element["mass"]  # 6x6, may be rotated to global frame

        # For multibody rotated sub-beams, T6 is stored on the element.
        # Recover local sectional mass (T orthogonal → T^{-1} = T^T) before
        # computing the consistent element mass matrix (N^T m N), then
        # transform the resulting 12×12 back to global.
        T6 = element.get('T6', None)
        if T6 is not None:
            T6 = np.array(T6, dtype=float)
            m_local = T6.T @ element_mass @ T6
        else:
            m_local = element_mass

        M_e = calculate_element_mass_matrix(m_local, L)  # 12x12 in local frame

        if T6 is not None:
            T12 = np.zeros((12, 12), dtype=float)
            T12[:6, :6] = T6
            T12[6:, 6:] = T6
            M_e = T12 @ M_e @ T12.T

        # Build local->global index map
        local_to_global = np.array([
            node1_idx * dof_per_node + 0,
            node1_idx * dof_per_node + 1,
            node1_idx * dof_per_node + 2,
            node1_idx * dof_per_node + 3,
            node1_idx * dof_per_node + 4,
            node1_idx * dof_per_node + 5,
            node2_idx * dof_per_node + 0,
            node2_idx * dof_per_node + 1,
            node2_idx * dof_per_node + 2,
            node2_idx * dof_per_node + 3,
            node2_idx * dof_per_node + 4,
            node2_idx * dof_per_node + 5,
        ], dtype=int)

        # Assemble using ix_ for clarity and speed
        M_global[np.ix_(local_to_global, local_to_global)] += M_e

    if config.save_global_matrices:
        _save_global_matrix_to_csv(M_global, config)

    return M_global