# FSI/COUPLING/aero_structural_coupling.py

import numpy as np
from .hydroelastic_utl.aerogrid_coupling_matrix import main as spline_main
from .hydroelastic_utl.aerogrid_coupling_matrix import build_Z_qs, build_Z_force
from .hydroelastic_utl.aerogrid_coupling_matrix import AeroGridToFEM
from panelaero_utl.old import Apan_to_fem
from FEA.fea_utl import cantilever_beam


def _multibody_subgrids(aerogrid):
    """Return [(name, (start, stop)), …] per-surface panel ranges, or None.

    A merged inverted-T aerogrid (built by build_multi_aerogrid.py) carries
    ``subgrid_names`` and ``subgrid_panel_ranges``; other grids return None.
    """
    if 'subgrid_panel_ranges' not in aerogrid or 'subgrid_names' not in aerogrid:
        return None
    names = [str(n) for n in np.asarray(aerogrid['subgrid_names']).tolist()]
    ranges = np.asarray(aerogrid['subgrid_panel_ranges'], dtype=int).reshape(-1, 2)
    return list(zip(names, [tuple(r) for r in ranges]))


def _slice_aerogrid(aerogrid, a, b):
    """Per-panel slice [a:b]; ``cornerpoint_grids`` kept whole (global ID lookup)."""
    sub = {'cornerpoint_grids': aerogrid['cornerpoint_grids']}
    for key in ('offset_j', 'offset_l', 'offset_k', 'N', 'A', 'r',
                'cornerpoint_panels', 'ID'):
        if key in aerogrid:
            sub[key] = np.asarray(aerogrid[key])[a:b]
    sub['n'] = int(b - a)
    return sub


def calculate_spline_matrix_multibody(beam_model, aerogrid, subgrids, xea_factor):
    """Assemble global Z, Z_qs, Z_force for a multibody (inverted-T) model.

    Each aero sub-surface is coupled to its own sub-beam (matched by the shared
    ``submodel`` tag) with the span axis inferred from that sub-beam alone, then
    scattered into the global panel×DOF matrices.  This lets the strut bracket
    along its span axis and the horizontal foils along theirs.
    """
    node_positions = np.array([node['position'] for node in beam_model['nodes']])
    n_nodes = len(beam_model['nodes'])
    n_panels = int(aerogrid['n'])
    total_dof = n_nodes * 6

    # global node indices per submodel tag
    nodes_by_sub = {}
    for i, node in enumerate(beam_model['nodes']):
        nodes_by_sub.setdefault(node.get('submodel'), []).append(i)

    Z = np.zeros((n_panels, total_dof))
    Z_qs = np.zeros((n_panels, total_dof))
    Z_force = np.zeros((n_panels, total_dof))
    panel_to_node_map = [None] * n_panels
    panel_xi_map = [0.5] * n_panels

    coupler = AeroGridToFEM(beam_model)
    for name, (a, b) in subgrids:
        gnodes = nodes_by_sub.get(name)
        if not gnodes:
            raise ValueError(
                f"Multibody coupling: no beam nodes tagged submodel='{name}' "
                f"(available: {sorted(str(k) for k in nodes_by_sub)})"
            )
        gnodes = list(gnodes)
        sub_pos = node_positions[gnodes]

        # Foils attach at the strut tip, which assemble_multibody merges into the
        # strut sub-beam (so the foil root node is tagged 'strut_ref2').  Include
        # that junction node here so foil panels near the root couple to it and
        # transfer their loads into the strut tip DOFs.
        others = [i for i in range(n_nodes) if i not in gnodes]
        if others:
            d_ext = np.linalg.norm(node_positions[others][:, None, :] - sub_pos[None, :, :], axis=2)
            nearest_ext = int(others[int(np.argmin(d_ext.min(axis=1)))])
            min_ext = float(d_ext.min())
            spacing = (float(np.min(np.linalg.norm(np.diff(sub_pos, axis=0), axis=1)))
                       if len(sub_pos) > 1 else np.inf)
            if min_ext <= 1.5 * spacing and nearest_ext not in gnodes:
                gnodes = [nearest_ext] + gnodes
                sub_pos = node_positions[gnodes]
        sub_grid = _slice_aerogrid(aerogrid, a, b)
        normals = sub_grid['N'] / np.linalg.norm(sub_grid['N'], axis=1)[:, None]

        Z_sub, p2n_sub, xi_sub = coupler.build_z_matrix(
            sub_grid, sub_pos, normals, None, debug=False, xea_factor=xea_factor,
        )
        n_sub_dof = 6 * len(gnodes)
        Zqs_sub = build_Z_qs(sub_grid, p2n_sub, xi_sub, n_sub_dof,
                             node_positions=sub_pos, beam_model=None)
        Zf_sub = build_Z_force(sub_grid, p2n_sub, xi_sub, sub_pos, None,
                               n_sub_dof, xea_factor=xea_factor)

        # scatter local (panel, sub-node DOF) → global (panel, node DOF)
        for jl, jg in enumerate(gnodes):
            gcols = slice(6 * jg, 6 * jg + 6)
            lcols = slice(6 * jl, 6 * jl + 6)
            Z[a:b, gcols] = Z_sub[:, lcols]
            Z_qs[a:b, gcols] = Zqs_sub[:, lcols]
            Z_force[a:b, gcols] = Zf_sub[:, lcols]
        for ip in range(b - a):
            n1, n2 = p2n_sub[ip]
            panel_to_node_map[a + ip] = (gnodes[n1], gnodes[n2])
            panel_xi_map[a + ip] = xi_sub[ip]

    return Z, Z_qs, Z_force, panel_to_node_map, panel_xi_map

def calculate_spline_matrix(beam_model, aerogrid, config, constrained_dofs=None):
    """
    Calculates the spline matrix (Z) to transfer forces and displacements
    between the aerodynamic and structural grids.
    """
    print("\n--- Calculating Aero-Structural Coupling Matrix ---")
    
    # Pass xea_factor to calculate elastic axis position relative to each panel's corner points
    # CRITICAL: Elastic axis is calculated per-panel relative to panel geometry, not as absolute value
    from FEA.fea_utl.beam_geometry import (
        aerogrid_xea_factor,
        coupling_xea_factor,
        xea_decoupled_from_aerogrid,
    )
    xea_factor = None
    if hasattr(config, 'xea_factor'):
        xea_factor = coupling_xea_factor(config)
        print(
            f"Using xea_factor={xea_factor:.4f} for coupling EA on aerogrid panels "
            f"(corner1 + xea_factor * l_m)"
        )
        if xea_decoupled_from_aerogrid(config):
            print(
                f"  decouple_xea_from_aerogrid: aerogrid fixed at aerogrid_xea_factor="
                f"{aerogrid_xea_factor(config):.4f}; structural beam shifted chordwise"
            )

    node_positions = np.array([node['position'] for node in beam_model['nodes']])

    # Inverted-T / multibody: couple each aero sub-surface (strut, foils) to its
    # own sub-beam with a per-surface span axis, then assemble global matrices.
    _subgrids = _multibody_subgrids(aerogrid) if beam_model.get('is_multibody') else None
    Z_qs_mb = Z_force_mb = None
    if _subgrids is not None:
        print(f"Multibody aero-structural spline: {len(_subgrids)} sub-surface(s) "
              f"→ {', '.join(n for n, _ in _subgrids)}")
        (Z, Z_qs_mb, Z_force_mb,
         panel_to_node_map, panel_xi_map) = calculate_spline_matrix_multibody(
            beam_model, aerogrid, _subgrids, xea_factor,
        )
    else:
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
    from .hydroelastic_utl.aerogrid_coupling_matrix import calculate_control_force_points_distances
    distance_results = calculate_control_force_points_distances(aerogrid, xea_factor=xea_factor)
    
    # Apply boundary conditions to the coupling matrix (must match structural dry analysis)
    total_dof = len(beam_model['nodes']) * 6
    if constrained_dofs is None:
        constrained_dofs = getattr(config, 'constrained_dofs', None)
    if constrained_dofs is None:
        constrained_dofs = list(range(6))
        print("Coupling BC: default cantilever (root only, DOFs 0-5)")
    else:
        print(f"Coupling BC: {len(constrained_dofs)} constrained DOFs (from structural analysis)")
    Z_constrained = cantilever_beam.cantilever_beam(Z, total_dof, constrained_dofs=constrained_dofs, dof_per_node=6)

    # Also calculate the Apan matrix needed for some solvers
    # Build Apan using the original method (for backward compatibility)
    Apan_original = Apan_to_fem.build_coupling(beam_model, aerogrid, panel_to_node_map, panel_xi_map)
    Apan_T = Apan_original.T
    Apan_constrained_original = (cantilever_beam.cantilever_beam(Apan_T, total_dof, constrained_dofs=constrained_dofs, dof_per_node=6)).T
    
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
    if Z_qs_mb is not None:
        Z_qs = Z_qs_mb
    else:
        Z_qs = build_Z_qs(
            aerogrid, panel_to_node_map, panel_xi_map, total_dof,
            node_positions=node_positions, beam_model=beam_model,
        )
    Z_qs_constrained = cantilever_beam.cantilever_beam(Z_qs, total_dof, constrained_dofs=constrained_dofs, dof_per_node=6)
    
    # Z_force (G_f for force transfer at 25% chord)
    if Z_force_mb is not None:
        Z_force = Z_force_mb
    else:
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
        Apan_constrained = (cantilever_beam.cantilever_beam(Apan_theory_T, total_dof, constrained_dofs=constrained_dofs, dof_per_node=6)).T
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
    Z_force_constrained = cantilever_beam.cantilever_beam(Z_force, total_dof, constrained_dofs=constrained_dofs, dof_per_node=6)

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
        from .hydroelastic_utl.aerogrid_coupling_matrix import plot_chordwise_strip
        y_target = getattr(config, 'chordwise_strip_y', 3.0)
        plot_chordwise_strip(aerogrid, y_target=y_target, xea_factor=xea_factor, 
                            xcm_factor=getattr(config, 'xcm_factor', None), 
                            config=config)
    
    # Plot full wing if requested
    if hasattr(config, 'plot_full_wing') and config.plot_full_wing:
        from .hydroelastic_utl.aerogrid_coupling_matrix import plot_full_wing
        plot_full_wing(aerogrid, xea_factor=xea_factor, 
                      xcm_factor=getattr(config, 'xcm_factor', None), 
                      config=config)
    
    return {
        "Z": Z_constrained,
        "Apan": Apan_constrained,
        "Z_qs": Z_qs_constrained,
        "Z_force": Z_force_constrained,
    }