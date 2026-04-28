import sys
import numpy as np
import time
import os
import numpy as np
from FEA.fea_utl.old import Bauchau_stiffness_matrix_assembly, mass_matrix_assembly_for_SONATA
import scipy.linalg
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Import from our modules
sys.path.append('/home/lorebasket/FSI/FEA')
sys.path.append('/home/lorebasket/FSI/SONATA/4_NACA0015')
sys.path.append('/home/lorebasket/FSI/')

from hydrofoils.beam_properties import create_beam_model
from fea_utl import analysis, visualization
from csv_export import parser as ps
from PanelAero import fluid_mainv3
from Blade_library import NACA0015 as contourn


def apply_modal_reduction(K_global, M_global_wet, C_global, n_modes=10):
    """
    Apply modal reduction using wet modes for computational efficiency.
    
    Based on paper equations (22):
    q = Ψₖ * uₖ(t)  # Modal expansion
    M̂ₖ = Ψₖᵀ[M + M_w(k)]Ψₖ
    Ĉₖ = Ψₖᵀ[C + C_w(k)]Ψₖ
    K̂ₖ = ΨₖᵀKΨₖ
    
    Parameters:
    -----------
    K_global : ndarray
        Global stiffness matrix
    M_global_wet : ndarray
        Global mass matrix including added mass (M + M_w)
    C_global : ndarray
        Global damping matrix including added damping (C + C_w)
    n_modes : int
        Number of wet modes to retain
    
    Returns:
    --------
    M_reduced, K_reduced, C_reduced : ndarray
        Reduced system matrices
    Psi : ndarray
        Modal matrix (mode shapes)
    frequencies_wet : ndarray
        Wet mode frequencies
    """
    print(f"\n=== MODAL REDUCTION ===")
    print(f"Original system size: {K_global.shape[0]} DOFs")
    print(f"Target reduced size: {n_modes} modes")
    
    # Solve generalized eigenvalue problem for wet modes
    # K * phi = lambda * M_wet * phi
    print("Computing wet modes...")
    try:
        eigenvals, eigenvecs = scipy.linalg.eigh(K_global, M_global_wet)
        
        # Sort by frequency (eigenvalues are ω²)
        freq_indices = np.argsort(eigenvals)
        eigenvals = eigenvals[freq_indices]
        eigenvecs = eigenvecs[:, freq_indices]
        
        # Extract frequencies (ω = sqrt(λ))
        frequencies_wet = np.sqrt(np.abs(eigenvals))
        
        # Select first n_modes
        n_modes = min(n_modes, len(frequencies_wet))
        Psi = eigenvecs[:, :n_modes]  # Modal matrix
        frequencies_wet = frequencies_wet[:n_modes]
        
        print(f"Successfully computed {n_modes} wet modes")
        print(f"Wet frequencies [Hz]: {frequencies_wet/(2*np.pi)}")
        
    except Exception as e:
        print(f"Error in eigenvalue computation: {e}")
        print("Using identity reduction (no modal reduction)")
        Psi = np.eye(K_global.shape[0])[:, :n_modes]
        frequencies_wet = np.ones(n_modes)
    
    # Apply modal reduction: Ψᵀ * Matrix * Ψ
    print("Applying modal reduction...")
    
    M_reduced = Psi.T @ M_global_wet @ Psi
    K_reduced = Psi.T @ K_global @ Psi
    C_reduced = Psi.T @ C_global @ Psi
    
    print(f"Reduced system size: {M_reduced.shape[0]} DOFs")
    print(f"Reduction ratio: {M_reduced.shape[0]/K_global.shape[0]:.2%}")
    
    # Verify modal orthogonality (should be identity for mass matrix)
    modal_mass_check = Psi.T @ M_global_wet @ Psi
    if np.allclose(modal_mass_check, np.diag(np.diag(modal_mass_check)), rtol=1e-3):
        print("✓ Modal orthogonality verified")
    else:
        print("⚠ Warning: Modal orthogonality check failed")
    
    return M_reduced, K_reduced, C_reduced, Psi, frequencies_wet

def wilson_theta_modal(M_reduced, C_reduced, K_reduced, F_modal, dt, n_steps, theta=1.4):
    """
    Wilson-θ time integration for the reduced modal system.
    
    Based on paper's approach for solving the modal equations efficiently.
    
    Parameters:
    -----------
    M_reduced, C_reduced, K_reduced : ndarray
        Reduced system matrices
    F_modal : ndarray
        Modal force vector (n_modes x n_steps)
    dt : float
        Time step
    n_steps : int
        Number of time steps
    theta : float
        Wilson-θ parameter (typically 1.4)
    
    Returns:
    --------
    u_modal : ndarray
        Modal displacement response (n_modes x n_steps)
    """
    n_modes = M_reduced.shape[0]
    
    # Initialize response arrays
    u_modal = np.zeros((n_modes, n_steps))
    v_modal = np.zeros((n_modes, n_steps))
    a_modal = np.zeros((n_modes, n_steps))
    
    # Wilson-θ constants
    dt_theta = theta * dt
    c1 = 6.0 / (dt_theta**2)
    c2 = 3.0 / dt_theta
    c3 = 2.0 * c2
    
    # Effective stiffness matrix
    K_eff = K_reduced + c1 * M_reduced + c2 * C_reduced
    
    try:
        K_eff_inv = np.linalg.inv(K_eff)
        print("Wilson-θ: Effective stiffness matrix inverted successfully")
    except np.linalg.LinAlgError:
        print("Warning: Singular effective stiffness matrix, using pseudo-inverse")
        K_eff_inv = np.linalg.pinv(K_eff)
    
    # Initial acceleration
    if F_modal.ndim == 1:
        F_modal = F_modal.reshape(-1, 1)
    
    if F_modal.shape[1] >= 1:
        F0 = F_modal[:, 0] if F_modal.shape[1] > 0 else np.zeros(n_modes)
        a_modal[:, 0] = np.linalg.solve(M_reduced, F0 - C_reduced @ v_modal[:, 0] - K_reduced @ u_modal[:, 0])
    
    # Time stepping
    for i in range(1, n_steps):
        # Force at time i
        Fi = F_modal[:, i] if F_modal.shape[1] > i else F_modal[:, -1]
        
        # Effective force
        F_eff = Fi + M_reduced @ (c1 * u_modal[:, i-1] + c2 * v_modal[:, i-1] + 2.0 * a_modal[:, i-1])
        F_eff += C_reduced @ (c3 * u_modal[:, i-1] + 2.0 * v_modal[:, i-1] + dt_theta * a_modal[:, i-1])
        
        # Solve for displacement
        u_modal[:, i] = K_eff_inv @ F_eff
        
        # Update velocity and acceleration
        du = u_modal[:, i] - u_modal[:, i-1]
        v_modal[:, i] = c2 * du - 2.0 * v_modal[:, i-1] - dt_theta * a_modal[:, i-1]
        a_modal[:, i] = c1 * du - c2 * v_modal[:, i-1] - 2.0 * a_modal[:, i-1]
    
    return u_modal, v_modal, a_modal

def solve_fsi_with_modal_reduction(K_global, M_global, C_global, M_added, C_added, 
                                 excitation_force, dt, n_steps, n_modes=10):
    """
    Complete FSI solution with modal reduction following the paper's methodology.
    
    This is the main function that integrates everything following paper's Eq. (22).
    
    Parameters:
    -----------
    K_global, M_global, C_global : ndarray
        Structural system matrices
    M_added, C_added : ndarray
        Added mass and damping matrices from fluid
    excitation_force : ndarray
        External force vector (n_dofs x n_steps)
    dt : float
        Time step
    n_steps : int
        Number of time steps
    n_modes : int
        Number of modes for reduction
    
    Returns:
    --------
    response_physical : ndarray
        Physical displacement response (n_dofs x n_steps)
    """
    print("\n=== FSI SOLUTION WITH MODAL REDUCTION ===")
    
    # 1. Combine structural and fluid matrices
    M_global_wet = M_global + M_added
    C_global_wet = C_global + C_added
    
    # 2. Apply modal reduction
    M_reduced, K_reduced, C_reduced, Psi, frequencies_wet = apply_modal_reduction(
        K_global, M_global_wet, C_global_wet, n_modes=n_modes
    )
    
    # 3. Project external force onto modal basis
    F_modal = Psi.T @ excitation_force
    print(f"Projected force shape: {F_modal.shape}")
    
    # 4. Solve reduced system in time domain
    print("Solving reduced system with Wilson-θ integrator...")
    u_modal, v_modal, a_modal = wilson_theta_modal(
        M_reduced, C_reduced, K_reduced, F_modal, dt, n_steps
    )
    
    # 5. Transform modal response back to physical coordinates
    response_physical = Psi @ u_modal
    print(f"Transformed physical response shape: {response_physical.shape}")
    
    # Plot modal participation
    visualization.plot_modal_participation(u_modal, frequencies_wet, dt)
    
    return response_physical

def main():
    # Start timer
    start_time = time.time()
    
    # Blade name
    blade_name = '4_NACA0015'
    beam_length = 4.0 # m
    n_elements = 20

    # Section properties file
    csv_filename = f'/home/lorebasket/FSI/SONATA/{blade_name}/csv_export/{blade_name}_section_data.csv'
    section_props = ps.parse_section_props_csv(csv_filename)

    # Get stiffness and mass matrices
    K = ps.parse_sectional_matrix_csv(f'/home/lorebasket/FSI/SONATA/{blade_name}/csv_export/{blade_name}_anbax_beam_properties_stiff_matrices.csv')
    M = ps.parse_sectional_matrix_csv(f'/home/lorebasket/FSI/SONATA/{blade_name}/csv_export/{blade_name}_anbax_beam_properties_mass_matrices.csv')

    # Transform K,M matrices from NA to SC
    K, M = ps.transform_matrices(csv_filename, K, M, beam_length)
    
    # Create beam model
    beam_model = create_beam_model(K, M, beam_length, n_elements)
    
    # Assemble global matrices
    nodes = beam_model["nodes"]
    K_global = Bauchau_stiffness_matrix_assembly.assemble_global_stiffness_matrix(beam_model)
    M_global = mass_matrix_assembly_for_SONATA.assemble_global_mass_matrix(beam_model)

    # ==============================================================================
    # DRY MODAL ANALYSIS
    # ==============================================================================
    print("\n=== STARTING DRY MODAL ANALYSIS ===")
    total_dof = len(nodes) * 6
    frequencies_dry, _, _, _, _, _ = analysis.modal_analysis(K_global, M_global, total_dof, num_modes=10)
    print("\nDry Frequencies (Hz):", frequencies_dry)
    
    # Fluid properties
    rho_f = 1000  # Fluid density (water)
    
    # ==============================================================================
    # WET MODAL ANALYSIS
    # ==============================================================================
    print("\n=== STARTING WET MODAL ANALYSIS ===")
    
    # Set output directory for added mass components
    output_dir_addedcomp = os.path.join(os.getcwd(), 'output', 'added_components')
    os.makedirs(output_dir_addedcomp, exist_ok=True)
    
    # Aerodynamic grid properties
    nspan = 20  # Number of spanwise panels
    nchord = 10 # Number of chordwise panels
    
    # Blade geometry
    chord_root = 0.1
    chord_tip = 0.05
    beam_length = beam_length
    
    # DLM inputs
    attack_angle = 10 # °grad
    V = [10] # inflow velocity [m/s]
    f = [14] # Driving frequency in Hz, converted to a list

    # Contour coordinates
    contour_coords = contourn.main()
    traslate = False

    # Run fluid main
    M_added, C_added = fluid_mainv3.main(
        blade_name, section_props, traslate, csv_filename, output_dir_addedcomp,
        beam_model, beam_length, rho_f, chord_tip, chord_root, nspan, nchord,
        attack_angle, V, f, contour_coords, fix_Madded=True)

    # After you get M_added and C_added from fluid analysis:
    M_added_real = np.real(M_added)  # Use only real part for stability
    C_added_real = np.real(C_added)  # Use only real part for stability

    C_global = 0.1 * M_global + 0.1 * K_global
    
    # Choose solution method
    USE_MODAL_REDUCTION = False  # Set this flag
    
    if USE_MODAL_REDUCTION:
        print("\n=== SOLVING WITH MODAL REDUCTION ===")
        
        # Solve using modal reduction (much more efficient)
        n_modes_to_use = 10
        
        # Define external force (example: tip load)
        n_dofs = K_global.shape[0]
        n_steps = 200
        dt = 0.01
        excitation_force = np.zeros((n_dofs, n_steps))
        tip_dof = -1 # Last DOF
        excitation_force[tip_dof, :] = 10 * np.sin(2 * np.pi * 5 * np.linspace(0, n_steps*dt, n_steps))
        
        response = solve_fsi_with_modal_reduction(
            K_global, M_global, C_global, M_added_real, C_added_real,
            excitation_force, dt, n_steps, n_modes=n_modes_to_use
        )
        
        # Plot final response
        visualization.plot_response_at_dof(response, tip_dof, dt, 'Tip Displacement (Modal Reduction)')
        
    else:
        print("\n=== SOLVING WITH FULL SYSTEM (DIRECT INTEGRATION) ===")
        
        # Combine structural and fluid matrices
        M_total = M_global + M_added_real
        C_total = C_global + C_added_real
        K_total = K_global
        
        # Check for positive definiteness
        try:
            np.linalg.cholesky(M_total)
            print("✓ Total mass matrix is positive definite.")
        except np.linalg.LinAlgError:
            print("⚠ Warning: Could not check for positive definiteness.")

        # Solve eigenvalue problem for wet modes
        print("\nSolving eigenvalue problem for wet modes...")
        total_dof = len(nodes) * 6
        frequencies_wet, modes_wet, _, _, _, _ = analysis.modal_analysis(K_total, M_total, total_dof, num_modes=10)
        
        print("\nWet Frequencies (Hz):", frequencies_wet)
        
        # Plotting mode shapes
        visualization.plot_mode_shapes(beam_model['nodes'], modes_wet, frequencies_wet, K_global, num_modes_to_plot=5)

    # End timer
    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
