import os
import numpy as np

from FEA.fea_utl.multibody_assembly import T6_for_element


def _save_global_matrix_to_csv(K_global, config):

    import csv
    
    print(f"\n*** Saving global K matrix to CSV files ***")
    
    # Create output directory: output_data/{config_name}/
    output_dir = os.path.join(config.output_dir, config.name)
    os.makedirs(output_dir, exist_ok=True)
    
    config_name = config.name
    
    # ========================================================================
    # SAVE GLOBAL STIFFNESS MATRIX (K_global)
    # ========================================================================
    k_file = os.path.join(output_dir, f"{config_name}_K_global.csv")
    
    try:
        np.savetxt(k_file, K_global, delimiter=',', fmt='%.10e')
        print(f"  ✓ Saved K_global matrix (shape {K_global.shape}) to: {k_file}")
    except Exception as e:
        print(f"  ✗ Error saving K_global: {e}")
    
    print(f"  Output directory: {output_dir}\n")

def calculate_element_stiffness_matrix(sectional_stiffness, L):
    """
    Bauchau's linear formulation
    Ke = ∫ B(x)^T * C * B(x) dx
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
    B = np.zeros((6, 12))
    xi = x / L

    # Linear
    N1 = 1 - xi
    N2 = xi
    dN1 = -1 / L
    dN2 = 1 / L

    # Hermite
    dH1 = (-6*xi + 6*xi**2) / L
    dH2 = (1 - 4*xi + 3*xi**2)
    dH3 = (6*xi - 6*xi**2) / L
    dH4 = (-2*xi + 3*xi**2)

    # --- γ_xy = du/dy - θz ---
    B[0, 0]  = dH1
    B[0, 5]  = dH2
    B[0, 6]  = dH3
    B[0, 11] = dH4
    B[0, 5]  -= N1
    B[0, 11] -= N2

    # --- ε_y = dv/dy ---
    B[1, 1] = dN1
    B[1, 7] = dN2

    # --- γ_yz = dw/dy + θx ---
    B[2, 2]  = dH1
    B[2, 3]  = dH2
    B[2, 8]  = dH3
    B[2, 9]  = dH4
    B[2, 3]  += N1   # Timoshenko: shear strain includes rotation
    B[2, 9]  += N2   # Timoshenko: shear strain includes rotation

    # --- κ_x = dθx/dy ---
    B[3, 3] = dN1
    B[3, 9] = dN2

    # --- κ_y = dθy/dy ---
    B[4, 4]  = dN1
    B[4, 10] = dN2

    # --- κ_z = dθz/dy ---
    B[5, 5]  = dN1
    B[5, 11] = dN2

    return B

def assemble_global_stiffness_matrix(beam_model, config):
    total_dof = len(beam_model["nodes"])*6
    K_global = np.zeros((total_dof, total_dof))

    nodes = beam_model["nodes"]

    for element in beam_model["elements"]:
        # Get node indices for this element
        n1_idx, n2_idx = element["nodes"]

        element_length = element["length"]
        element_stiffness = element["stiffness"]

        T6 = T6_for_element(element, nodes)
        C_local = T6.T @ element_stiffness @ T6

        Ke_local = calculate_element_stiffness_matrix(C_local, element_length)

        T12 = np.zeros((12, 12), dtype=float)
        T12[:6, :6] = T6
        T12[6:, 6:] = T6
        Ke = T12 @ Ke_local @ T12.T

        # Assemble into global matrix
        for i in range(12):
            gi = n1_idx * 6 + (i % 6) if i < 6 else n2_idx * 6 + (i % 6)
            for j in range(12):
                gj = n1_idx * 6 + (j % 6) if j < 6 else n2_idx * 6 + (j % 6)
                K_global[gi, gj] += Ke[i, j]

    if config.save_global_matrices:
        _save_global_matrix_to_csv(K_global, config)

    return K_global

