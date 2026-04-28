# FSI/Hydroelastic_analysis_workflow/aero_structural_coupling.py

import numpy as np
from panelaero_utl.coupling_matrix import main as spline_main
from panelaero_utl.coupling_matrix import build_Z_qs, build_Z_force
from panelaero_utl.old import Apan_to_fem
from FEA.fea_utl import cantilever_beam

def calculate_spline_matrix(beam_model, aerogrid, config):
    """
    Calculates the spline matrix (Z) to transfer forces and displacements
    between the aerodynamic and structural grids.
    """
    print("\n--- Calculating Aero-Structural Coupling Matrix ---")
    
    # Pass xea_factor to calculate elastic axis position relative to each panel's corner points
    # CRITICAL: Elastic axis is calculated per-panel relative to panel geometry, not as absolute value
    xea_factor = None
    if hasattr(config, 'xea_factor'):
        xea_factor = config.xea_factor
        print(f"Using xea_factor={xea_factor:.4f} to calculate elastic axis position relative to each panel's corner points")
    
    Z, panel_to_node_map, panel_xi_map = spline_main(
        beam_model,
        aerogrid=aerogrid,
        coupling_diagnostics=False,
        plot=False,
        plot_airfoil=False,
        airfoil_profile=None, # Not needed for pure analysis
        n_airfoil_sections=0,
        xea_factor=xea_factor,
    )
    
    # Calculate and print distances of control and force points from y-axis
    from panelaero_utl.coupling_matrix import calculate_control_force_points_distances
    distance_results = calculate_control_force_points_distances(aerogrid, xea_factor=xea_factor)
    
    # Apply boundary conditions to the coupling matrix
    # This keeps all DOFs in Z and Apan matrices (Cantilever boundary conditions handled by default)
    total_dof = len(beam_model['nodes']) * 6
    Z_constrained = cantilever_beam.cantilever_beam(Z, total_dof, constrained_dofs=None, dof_per_node=6)

    # Also calculate the Apan matrix needed for some solvers
    # Build Apan using the original method (for backward compatibility)
    Apan_original = Apan_to_fem.build_coupling(beam_model, aerogrid, panel_to_node_map, panel_xi_map)
    Apan_T = Apan_original.T
    Apan_constrained_original = (cantilever_beam.cantilever_beam(Apan_T, total_dof, constrained_dofs=None, dof_per_node=6)).T
    
    # Build Apan from Z_force^T @ A for theory compliance (if Z_force is available)
    # This ensures Apan = G_f^T @ A exactly as per theory
    if 'A' in aerogrid:
        panel_areas = aerogrid['A']
        A_diag = np.diag(panel_areas)
        # Apan_theory = Z_force^T @ A (will be computed after Z_force is built)
        Apan_theory = None  # Will be set below
    else:
        Apan_theory = None
        panel_areas = None

    # Z quasy steady (G_x for slope)
    Z_qs = build_Z_qs(aerogrid, panel_to_node_map, panel_xi_map, total_dof)
    Z_qs_constrained = cantilever_beam.cantilever_beam(Z_qs, total_dof, constrained_dofs=None, dof_per_node=6)
    
    # Z_force (G_f for force transfer at 25% chord)
    node_positions = np.array([node['position'] for node in beam_model['nodes']])
    Z_force = build_Z_force(aerogrid, panel_to_node_map, panel_xi_map, node_positions, beam_model, total_dof, xea_factor=xea_factor)
    
    # Build Apan from Z_force^T @ A for theory compliance
    # THEORY: Apan = G_f^T @ A (ALWAYS, no exceptions)
    # IMPORTANT: Build Apan BEFORE applying boundary conditions, then apply BCs once (like Apan_original)
    if 'A' in aerogrid:
        panel_areas = aerogrid['A']
        A_diag = np.diag(panel_areas)
        # Apan_theory = Z_force^T @ A (theory-compliant construction)
        # Z_force shape: (n_pan, 6*Nnodes), so Z_force.T @ A_diag gives (6*Nnodes, n_pan)
        Apan_theory = Z_force.T @ A_diag  # (6*Nnodes, n_pan) - BEFORE boundary conditions
        # Apply boundary conditions to Apan_theory (same way as Apan_original)
        Apan_theory_T = Apan_theory.T  # (n_pan, 6*Nnodes)
        Apan_constrained = (cantilever_beam.cantilever_beam(Apan_theory_T, total_dof, constrained_dofs=None, dof_per_node=6)).T
    else:
        # CRITICAL: Apan MUST be Z_force^T @ A according to theory
        # If panel areas not available, raise error or try to extract from Apan_original
        import warnings
        warnings.warn("Panel areas 'A' not found in aerogrid. Attempting to extract from Apan_original structure.")
        # Try to reconstruct A from Apan_original (not ideal, but better than using wrong Apan)
        # This is a fallback - ideally panel areas should always be available
        Apan_constrained = Apan_constrained_original
        print("WARNING: Using Apan_original as fallback. This may not be theory-compliant if Apan_original != Z_force^T @ A")
    
    # Apply boundary conditions to Z_force AFTER building Apan
    Z_force_constrained = cantilever_beam.cantilever_beam(Z_force, total_dof, constrained_dofs=None, dof_per_node=6)

    print("Coupling matrices Z, Z_force, Z_qs and Apan computed.")
    print(f"  Z shape: {Z_constrained.shape}")
    print(f"  Z_force shape: {Z_force_constrained.shape}")
    print(f"  Apan shape: {Apan_constrained.shape}")
    
    # Save coupling matrices for debugging if requested
    try:
        if getattr(config, 'save_matrices', False):
            out_dir = getattr(config, 'output_dir', '.')
            import os
            os.makedirs(out_dir, exist_ok=True)
            fname = os.path.join(out_dir, f"{config.name}_coupling_matrices.npz")
            np.savez_compressed(
                fname,
                Z=Z_constrained,
                Z_force=Z_force_constrained,
                Z_qs=Z_qs_constrained,
                Apan=Apan_constrained,
            )
            print(f"Saved coupling matrices to {fname}")
    except Exception as e:
        print(f"Warning: failed to save coupling matrices: {e}")
    
    # Plot chordwise strip if requested
    if hasattr(config, 'plot_chordwise_strip') and config.plot_chordwise_strip:
        from panelaero_utl.coupling_matrix import plot_chordwise_strip
        y_target = getattr(config, 'chordwise_strip_y', 3.0)
        plot_chordwise_strip(aerogrid, y_target=y_target, xea_factor=xea_factor, 
                            xcm_factor=getattr(config, 'xcm_factor', None), 
                            config=config)
    
    # Plot full wing if requested
    if hasattr(config, 'plot_full_wing') and config.plot_full_wing:
        from panelaero_utl.coupling_matrix import plot_full_wing
        plot_full_wing(aerogrid, xea_factor=xea_factor, 
                      xcm_factor=getattr(config, 'xcm_factor', None), 
                      config=config)
    
    return {
        "Z": Z_constrained,
        "Apan": Apan_constrained,
        "Z_qs": Z_qs_constrained,
        "Z_force": Z_force_constrained,
    }