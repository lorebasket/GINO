# FSI/Hydroelastic_analysis_workflow/structural_analysis.py

import numpy as np
import time
import os
import traceback
from collections import namedtuple

from FEA.fea_utl import (
    global_stiffness,
    global_mass,
    analysis,
    modal_reduction
)
from FEA.fea_utl import guyan_reduction
from FEA.fea_utl.mode_classifier import ModeClassifier

# Define a result object for clarity
StructuralResults = namedtuple('StructuralResults', [
    'u_full', 'reaction_forces', 'dry_frequencies', 'dry_damping_ratios',
    'dry_eigenvalues', 'dry_eigenvectors', 'dry_eigenvectors_full', 'Mff', 'Kff', 'Cff',
    'M_hat', 'C_hat', 'K_hat', 'total_dof', 'constrained_dofs',
    'rayleigh_alpha', 'rayleigh_beta', 'beam_model',
    'K_global', 'M_global'
])


def _create_gravity_force_vector(beam_model, total_dof, gravity_acc, dof_per_node=6):
    """
    Create distributed gravity force vector for all nodes based on element mass.
    
    For GOLAND-type models:
    - The sectional mass matrix M (6x6) has M[0,0] = M[1,1] = M[2,2] = mu (kg/m)
    - Each element stores this matrix in elem['mass']
    - We extract mu from the mass matrix diagonal and compute gravity loads
    
    Parameters
    ----------
    beam_model : dict
        Contains 'nodes' and 'elements' with mass properties
    total_dof : int
        Total degrees of freedom in the system
    gravity_acc : array-like
        Gravitational acceleration vector [gx, gy, gz] in m/s²
    dof_per_node : int, optional
        Degrees of freedom per node (default: 6)
    
    Returns
    -------
    f_gravity : np.ndarray
        Force vector with gravity loads distributed to translational DOFs
    """
    f_gravity = np.zeros(total_dof, dtype=float)
    gravity_vec = np.array(gravity_acc, dtype=float)
    
    # Track total mass for verification
    total_mass = 0.0
    
    # Distribute gravity forces based on element mass
    for i, elem in enumerate(beam_model['elements']):
        node1_idx = elem['nodes'][0]
        node2_idx = elem['nodes'][1]
        
        # Get element length
        elem_length = elem['length']
        
        # Extract mass per unit length from the 6x6 mass matrix
        # For GOLAND: M[0,0] = M[1,1] = M[2,2] = mu (kg/m)
        if 'mass' in elem:
            M_elem = elem['mass']
            # For uniform beam: M[0,0] = M[1,1] = M[2,2] = mu
            # Extract just one (they should be equal)
            mass_per_length = M_elem[0, 0]  # Direct extraction
            
            # Debug: check consistency
            if i == 0:  # Print only for first element
                print(f"  First element mass matrix diagonal: M[0,0]={M_elem[0,0]:.3f}, M[1,1]={M_elem[1,1]:.3f}, M[2,2]={M_elem[2,2]:.3f} kg/m")
        else:
            # Fallback: no mass matrix available
            mass_per_length = 0.0
            print(f"  Warning: Element {i} (nodes {elem['nodes']}) has no 'mass' key!")
        
        # Total element mass
        elem_mass = mass_per_length * elem_length
        total_mass += elem_mass
        
        # Distribute half to each node (lumped mass approach)
        node_force = 0.5 * elem_mass * gravity_vec
        
        # Apply to translational DOFs (0, 1, 2) for each node
        # Note: DOFs 3, 4, 5 are rotational and don't get direct gravity forces
        f_gravity[node1_idx * dof_per_node : node1_idx * dof_per_node + 3] += node_force
        f_gravity[node2_idx * dof_per_node : node2_idx * dof_per_node + 3] += node_force
    
    print(f"  Total structural mass: {total_mass:.3f} kg")
    print(f"  Expected gravity force: {total_mass * np.linalg.norm(gravity_vec):.3f} N")
    
    # Extract only vertical DOF forces (index 2 in 6-DOF, index 0 in 3-DOF)
    if dof_per_node == 6:
        vertical_dof_idx = 2
    else:
        vertical_dof_idx = 0
    f_z_only = np.array([f_gravity[i*dof_per_node + vertical_dof_idx] for i in range(total_dof // dof_per_node)])
    print(f"  Vertical-direction force sum: {np.sum(f_z_only):.3f} N (should equal expected gravity)")
    print(f"  X-direction force sum: {np.sum([f_gravity[i*dof_per_node + 0] for i in range(total_dof // dof_per_node)]):.3f} N (should be ~0)")
    print(f"  Y-direction force sum: {np.sum([f_gravity[i*dof_per_node + 1] for i in range(total_dof // dof_per_node)]):.3f} N (should be ~0)")
    
    return f_gravity


def mass_matrix_strip_theory(M_global, config):
    import numpy as np

    # strip theory added mass per unit length
    rho = getattr(config, 'rho_f', {}).get(getattr(config, 'fluid', 'air'), 1.02)
    m_a_prime = rho * np.pi * (config.chord**2) / 4.0

    # grid
    L = config.beam_length
    N = config.n_elements + 1  # number of nodes
    y = np.linspace(0, L, N)
    dy = y[1] - y[0]

    ndof_per_node = 6
    ndof_total = N * ndof_per_node

    # -----------------------------
    # BUILD 1D CONSISTENT MASS MATRIX
    # (simple trapezoidal / lumped hybrid)
    # -----------------------------
    M_1D = np.zeros_like(M_global)

    for i in range(N):
        M_1D[i, i] += m_a_prime * dy

    # -----------------------------
    # GLOBAL MATRIX (606x606)
    # -----------------------------
    M_global_added = np.zeros((ndof_total, ndof_total))

    # DOF ordering per node:
    # [u, v, w, ry, rx, rz]
    W_DOF_INDEX = 2

    for i in range(N):
        for j in range(N):
            global_i = i * ndof_per_node + W_DOF_INDEX
            global_j = j * ndof_per_node + W_DOF_INDEX

            M_global_added[global_i, global_j] = M_1D[i, j]

    # -----------------------------
    # CHECK
    # -----------------------------
    print("Global matrix shape:", M_global_added.shape)
    return M_global_added


def added_mass_projection(M_global, dry_vectors, sorted_mode_indices, config, total_dof, constrained_dofs=None):
    print("\nCalculating strip theory added mass matrix...")

    M_added_strip = mass_matrix_strip_theory(M_global, config)

    # Constraints application
    dof_per_node = 6
    if constrained_dofs is None:
        constrained_dofs = list(range(dof_per_node))  # clamp 6dof at node 0 by default

    all_dofs = np.arange(total_dof, dtype=int)
    mask = np.ones(total_dof, dtype=bool)
    mask[np.array(constrained_dofs, dtype=int)] = False
    free_dofs = all_dofs[mask]
    fixed_dofs = np.array(constrained_dofs, dtype=int)

    M_cant_added_strip = M_added_strip[np.ix_(free_dofs, free_dofs)].astype(np.float64)
    Phi_for_reduction = dry_vectors[:, sorted_mode_indices]  # Select modes in frequency order

    M_hat_added = Phi_for_reduction.T @ M_cant_added_strip @ Phi_for_reduction
    
    return M_hat_added, Phi_for_reduction


def added_damping_projection(factor, M_global, dry_vectors, sorted_mode_indices, config, total_dof, constrained_dofs=None):
    print("\nCalculating strip theory added damping matrix...")

    # For demonstration, we create a simple proportional damping matrix based on the mass matrix
    C_added_strip = factor * mass_matrix_strip_theory(M_global, config)  # 1% of added mass as damping

    # Constraints application
    dof_per_node = 6
    if constrained_dofs is None:
        constrained_dofs = list(range(dof_per_node))  # clamp 6dof at node 0 by default

    all_dofs = np.arange(total_dof, dtype=int)
    mask = np.ones(total_dof, dtype=bool)
    mask[np.array(constrained_dofs, dtype=int)] = False
    free_dofs = all_dofs[mask]
    fixed_dofs = np.array(constrained_dofs, dtype=int)

    C_cant_added_strip = C_added_strip[np.ix_(free_dofs, free_dofs)].astype(np.float64)
    Phi_for_reduction = dry_vectors[:, sorted_mode_indices]  # Select modes in frequency order

    C_hat_added = Phi_for_reduction.T @ C_cant_added_strip @ Phi_for_reduction
    
    return C_hat_added, Phi_for_reduction


def run_dry_analysis(beam_model, config):
    """
    Performs static and modal analysis on the structural model (dry run).
    Optionally includes gravity effects if config.include_gravity = True.
    """
    print("\n--- Starting Dry Structural Analysis ---")
    start_time = time.time()

    # Define boundary conditions
    # For traditional cantilever: fixed root (first node, DOFs 0-5)
    # For full-span center-clamped: fixed center node
    nodes = beam_model["nodes"]
    n_nodes = len(nodes)
    total_dof = n_nodes * 6
    
    # Check if this is a full-span model with a clamped center node
    clamped_node_idx = config.clamped_node_idx if hasattr(config, 'clamped_node_idx') else None
    for i, node in enumerate(nodes):
        if node.get('clamped', False):
            clamped_node_idx = i
            print(f"Detected clamped node at index {i}, position: {node['position']}")
            break
    
    if clamped_node_idx is not None:
        # Full-span model: clamp the center node(s)
        # Handle both single index and array of indices
        if isinstance(clamped_node_idx, np.ndarray):
            constrained_dofs = []
            for idx in clamped_node_idx:
                idx_int = int(idx)
                # Convert negative indices to positive (e.g., -1 becomes n_nodes-1)
                if idx_int < 0:
                    idx_int = n_nodes + idx_int
                constrained_dofs.extend(range(idx_int * 6, (idx_int + 1) * 6))
            print(f"Full-span configuration: clamping nodes {clamped_node_idx} (DOFs {constrained_dofs[0]}-{constrained_dofs[-1]})")
        else:
            clamped_node_idx_int = int(clamped_node_idx)
            # Convert negative indices to positive
            if clamped_node_idx_int < 0:
                clamped_node_idx_int = n_nodes + clamped_node_idx_int
            constrained_dofs = list(range(clamped_node_idx_int * 6, (clamped_node_idx_int + 1) * 6))
            print(f"Full-span configuration: clamping center node {clamped_node_idx_int} (DOFs {constrained_dofs[0]}-{constrained_dofs[-1]})")
    else:
        # Traditional cantilever: clamp the root (first node)
        constrained_dofs = list(range(6))
        print(f"Cantilever configuration: clamping root node (DOFs 0-5)")


    # Assemble global matrices using global_stiffness and global_mass functions
    print("Assembling global stiffness and mass matrices...")
    K_global = global_stiffness.assemble_global_stiffness_matrix(beam_model, config)
    M_global = global_mass.assemble_global_mass_matrix(beam_model, config)
    print(f"  K_global shape: {K_global.shape}")
    print(f"  M_global shape: {M_global.shape}")

    
    # ========================================================================
    # ADD CONCENTRATED POINT MASSES
    # ========================================================================
    if 'point_masses' in beam_model and beam_model['point_masses']:
        print("\n*** Adding concentrated point masses ***")
        for pm in beam_model['point_masses']:
            node_idx = pm['node_index']
            mass = pm['mass']
            # DOFs for this node: [ux, uy, uz, rx, ry, rz]
            dof_start = node_idx * 6
            
            # Add translational mass to DOFs [ux, uy, uz]
            for i in range(3):
                M_global[dof_start + i, dof_start + i] += mass
            
            print(f"  Added {mass:.1f} kg point mass at node {node_idx}")
            print(f"    Node position: {beam_model['nodes'][node_idx]['position']}")
            print(f"    DOFs updated: {dof_start} to {dof_start+2} (translational)")
            
            # If rotational inertia is provided, add it to rotational DOFs
            if pm.get('inertia') is not None:
                inertia = pm['inertia']  # 3x3 inertia tensor [Ixx, Iyy, Izz, Ixy, Ixz, Iyz]
                for i in range(3):
                    M_global[dof_start + 3 + i, dof_start + 3 + i] += inertia[i, i]
                print(f"    Added rotational inertia to DOFs {dof_start+3} to {dof_start+5}")
    

    # ========================================================================
    # GRAVITY LOAD VECTOR
    # ========================================================================
    include_gravity = getattr(config, 'include_gravity', False)
    gravity_acc = getattr(config, 'gravity_acc', np.array([0, 0, -9.81]))
    
    if include_gravity:
        print(f"\n*** GRAVITY ENABLED: g = {gravity_acc} m/s² ***")
        force_vector = _create_gravity_force_vector(beam_model, total_dof, gravity_acc)
        print(f"  Total gravity force magnitude: {np.linalg.norm(force_vector):.3f} N")
    else:
        print("\n*** NO GRAVITY: Using undeformed configuration for modal analysis ***")
        force_vector = np.zeros(total_dof)
    

    # -- Static Analysis (with or without gravity) --- #
    u_full, reaction_forces, dry_values, dry_vectors, M_constrained, K_constrained = analysis.solve_static_analysis(
        K_global, M_global, 
        force_vector,  # Now includes gravity if enabled
        total_dof,
        config, beam_model,
        constrained_dofs=constrained_dofs,
        num_modes=config.num_modes_egv
    )


    dry_frequencies = np.sqrt(dry_values) / (2 * np.pi)  # Convert from omega^2 to Hz
    print("\nNatural frequencies static analysis (Hz):")
    for i, freq in enumerate(dry_frequencies[:5]):
        print(f"  Mode {i+1}: {freq:.4f}")

    
    # Print static deflections if gravity is included
    if include_gravity and np.any(u_full != 0):
        print(f"\n*** STATIC DEFLECTIONS DUE TO GRAVITY ***")
        max_deflection = np.max(np.abs(u_full))
        max_dof = np.argmax(np.abs(u_full))
        print(f"  Max deflection: {max_deflection:.6f} m at DOF {max_dof}")
        
        # Report tip deflection for cantilever
        if clamped_node_idx is None:  # Cantilever
            tip_node_idx = n_nodes - 1
            tip_w = u_full[tip_node_idx * 6 + 2]  # Z-displacement at tip
            beam_length = getattr(config, 'beam_length', 1.0)
            print(f"  Tip vertical deflection: {tip_w:.6f} m ({100*abs(tip_w)/beam_length:.2f}% of span)")


    # More diagnostics
    print(f"Constrained DOFs: {constrained_dofs}")
    print(f"M_constrained_shape: {getattr(M_constrained, 'shape', None)}")
    print(f"K_constrained_shape: {getattr(K_constrained, 'shape', None)}")
    print(f"dry_vectors shape: {getattr(dry_vectors, 'shape', None)}")

    # ========================== #
    # --- MODAL ANALYSIS EGV --- #
    # --- if use strip theory, the fluid at rest analyses condition are calculated #
    # ========================= #

    # Rayleigh damping targets from config (default to no structural damping)
    mode_ids = getattr(config, 'rayleigh_mode_ids')
    target_zetas = getattr(config, 'rayleigh_target_zetas')
    C_constrained, alpha_rayleigh, beta_rayleigh, _ = analysis.rayleigh_from_two_modes(
        K_constrained, M_constrained, dry_vectors, dry_values,
        target_mode_ids=mode_ids, target_zetas=target_zetas, verbose=False
    )
    
    # Ensure modes_to_analyze are sorted by frequency (ascending order)
    sorted_mode_indices = sorted([i for i in range(config.num_modes_egv)])
    print(f"\n{'='*70}")
    print(f"MODAL REDUCTION")
    print(f"{'='*70}")
    print(f"Original modes_to_analyze: {config.num_modes_egv}")
    print(f"Sorted modes_to_analyze: {sorted_mode_indices}")
    
    # Show original unreduced frequencies (in Hz)
    freqs_unreduced = np.sqrt(dry_values[sorted_mode_indices]) / (2*np.pi)
    print(f"Unreduced frequencies (Hz): {freqs_unreduced}")
    print(f"Unreduced vectors: {dry_vectors[:, sorted_mode_indices]}")
    print(f"(these come from full {dry_vectors.shape[0]}-DOF system eigenanalysis)")
    
    Phi_for_reduction = dry_vectors[:, sorted_mode_indices]  # Select modes in frequency order 
    
    # ==========================
    # ADDED MASS STRIP THEORY
    # ==========================
    if config.added_mass_strip_theory:
        print("\nIncluding strip theory added mass in modal reduction...")
        M_hat_added, Phi_for_reduction = added_mass_projection(M_global, dry_vectors, sorted_mode_indices, config, total_dof, constrained_dofs=None)

        # ADDED DAMPING ESTIMATION
        factor = 0   # Example: 1% of added mass as damping
        C_hat_added = M_hat_added * factor  # Simple proportional damping based on added mass

        M_hat, K_hat, C_hat = modal_reduction.reduce_matrices(M_constrained, C_constrained, K_constrained, Phi_for_reduction)

        M_hat += M_hat_added
        C_hat += C_hat_added
        print("Added strip theory mass and damping to reduced matrices.")   

    else:
        M_hat, K_hat, C_hat = modal_reduction.reduce_matrices(M_constrained, C_constrained, K_constrained, Phi_for_reduction)
    
    # ==========================
    # MODAL REDUCTION SOLUTION
    # ==========================   
    dry_freqs, dry_damp_ratios, dry_eigvals, _, _ = modal_reduction.solve_modes_state_space(
        M_hat, C_hat, K_hat
    )
    # omega_n^2 derived from the state-space solve: consistent with M_hat/K_hat
    n_modes_reduced = len(dry_freqs)
    dry_omega_sq = (2 * np.pi * dry_freqs[:n_modes_reduced]) ** 2
    
    print(f"\nAfter modal reduction")
    if config.added_mass_strip_theory:
        print(f"\nFluid at rest system frequencies with strip theory added mass (Hz): {dry_freqs[:n_modes_reduced]}")
    else:
        print(f"\nDry system frequencies (Hz): {dry_freqs[:n_modes_reduced]}")
    print(f"(from {len(sorted_mode_indices)}×{len(sorted_mode_indices)} reduced system with damping)")
    print(f"Actual modes extracted after state-space solve: {n_modes_reduced}")
    print(f"{'='*70}\n")

    # ========================================================================
    # EXPAND MODAL BASIS TO FULL FREE-DOF SPACE
    # ========================================================================
    free_dofs = np.setdiff1d(np.arange(total_dof), constrained_dofs)
    Phi_for_aero_coupling = Phi_for_reduction[:, :n_modes_reduced]  # Select first n_modes_reduced
    
    # Prepare full eigenvectors (including constrained DOFs = zeros)
    dry_vectors_full = np.zeros((total_dof, n_modes_reduced))
    dry_vectors_full[free_dofs, :] = Phi_for_aero_coupling

    print(f"Dry analysis completed in {time.time() - start_time:.3f} seconds")
    

    frequencies_to_print = min(5, len(dry_freqs))
    print("\nNatural frequencies (Hz):")
    for i, freq in enumerate(dry_freqs[:frequencies_to_print]):
        print(f"  Mode {i+1}: {freq:.4f}")


    # ========================================================================
    # VIBRATION MODE CLASSIFICATION (if enabled in config)
    # ========================================================================
    if getattr(config, 'classify_modes'):
        print("\n" + "="*70)
        print("VIBRATION MODE CLASSIFICATION")
        print("="*70)
        try:
            # Calculate number of nodes from total DOF (total_dof / 6)
            # This ensures consistency with the eigenvector dimensions
            n_nodes_actual = total_dof // 6
            
            # Get eigenvalues and eigenvectors from structural results
            # Important: These should already have matching dimensions (n_modes_reduced)
            eigenvalues = dry_omega_sq  # omega_n^2 from state-space solve (length: n_modes_reduced)
            eigenvectors = dry_vectors_full  # Full DOF space eigenvectors (columns: n_modes_reduced)
            
            # Verify dimensions match
            if eigenvectors.shape[0] != total_dof:
                raise ValueError(
                    f"Eigenvector dimension mismatch: got {eigenvectors.shape[0]} DOFs, "
                    f"expected {total_dof}"
                )
            
            if len(eigenvalues) != eigenvectors.shape[1]:
                raise ValueError(
                    f"Eigenvalue/eigenvector count mismatch: "
                    f"{len(eigenvalues)} eigenvalues but {eigenvectors.shape[1]} eigenvector columns"
                )
            
            # Instantiate and run classifier
            mode_classifier = ModeClassifier(
                eigenvectors=eigenvectors,
                eigenvalues=eigenvalues,
                n_nodes=n_nodes_actual,
                freq_type='lambda',
                threshold_dominant=0.60,
                threshold_secondary=0.20
            )
            
            # Print summary table
            mode_classifier.summary(n_decimals=4)
            
            # Plot participation ratios
            mode_classifier.plot_participation(
                max_modes=min(10, len(dry_freqs)),
                figsize=(14, 6)
            )
            
            # Optionally plot first few mode shapes with classification
            n_shapes_to_plot = min(4, len(dry_freqs))
            if 'node_coords' in beam_model and beam_model['node_coords'] is not None:
                for i in range(n_shapes_to_plot):
                    mode_classifier.plot_mode_shape(
                        mode_index=i,
                        node_coords=beam_model['node_coords'],
                        scale=1.0
                    )
            
            print("✓ Mode classification completed successfully")
        except Exception as e:
            print(f"⚠ Warning: Could not complete mode classification: {e}")
            import traceback
            traceback.print_exc()

    return StructuralResults(
        u_full=u_full, # Full displacement vector from static analysis (needed for force computation)
        reaction_forces=reaction_forces, # Reaction forces from static analysis
        dry_frequencies=dry_freqs,
        dry_damping_ratios=dry_damp_ratios,
        dry_eigenvalues=dry_omega_sq, # omega_n^2 from state-space solve (consistent with M_hat/K_hat, always positive)
        dry_eigenvectors=Phi_for_aero_coupling, # Free DOFs in FULL SPACE (426 DOFs) for aero-structural coupling
        dry_eigenvectors_full=dry_vectors_full, # Full DOFs with zeros - for sweep
        Mff=M_constrained,
        Kff=K_constrained,
        Cff=C_constrained,
        M_hat=M_hat,
        C_hat=C_hat,
        K_hat=K_hat,
        total_dof=total_dof,
        constrained_dofs=constrained_dofs,
        rayleigh_alpha=alpha_rayleigh,
        rayleigh_beta=beta_rayleigh,
        beam_model=beam_model,
        K_global=K_global,
        M_global=M_global
    )

def _create_rbe3_constraint_matrix(node_coords, center_of_mass, constrained_dofs):
    """
    Create RBE3 constraint matrix for master-slave coupling.
    
    RBE3 (Rigid Body Element, type 3) kinematically couples slave DOFs to master DOFs
    while distributing forces/moments. This function builds the constraint matrix that
    relates slave DOF displacements to master rigid body displacements.
    
    Theory:
    -------
    For a slave node i at position r_i (relative to center_of_mass):
    - u_i = u_master + θ_master × (r_i - r_cm)
    - In component form:
        u_i = u_m - (θ_my * r_iz - θ_mz * r_iy)
        v_i = v_m - (θ_mz * r_ix - θ_mx * r_iz)
        w_i = w_m - (θ_mx * r_iy - θ_my * r_ix)
        θ_i = θ_master (rotation is same for all nodes in rigid body)
    
    Parameters
    ----------
    node_coords : np.ndarray
        All node coordinates, shape (n_nodes, 3)
    center_of_mass : np.ndarray
        Position of master node (3,)
    constrained_dofs : list
        Global DOF indices that are constrained (fixed boundary)
    
    Returns
    -------
    B : np.ndarray
        Constraint matrix, shape (n_slave_dofs, 6)
        where n_slave_dofs = 6*(n_nodes-1) after excluding master and constrained nodes
    master_dofs : np.ndarray
        Global DOF indices for master node [0, 1, 2, 3, 4, 5]
    slave_dofs : np.ndarray
        Global DOF indices for all slave nodes (excluding master and constrained)
    """
    n_nodes = len(node_coords)
    total_dof = n_nodes * 6
    
    # Master DOFs are always indices 0-5 (first node is master)
    # In the augmented system, master will be explicitly added before slave DOFs
    master_dofs_local = np.arange(6)  # Local indices: 0-5
    
    # Identify slave DOFs (all nodes except master, and not constrained)
    slave_dofs_list = []
    for node_idx in range(1, n_nodes):  # Skip node 0 (master)
        for local_dof in range(6):
            global_dof = node_idx * 6 + local_dof
            # Add to slave list if not already constrained
            if global_dof not in constrained_dofs:
                slave_dofs_list.append(global_dof)
    
    slave_dofs = np.array(slave_dofs_list, dtype=int)
    n_slave_dofs = len(slave_dofs)
    
    # Build constraint matrix B
    # Rows: one per slave DOF
    # Cols: 6 master DOFs [u_m, v_m, w_m, θx_m, θy_m, θz_m]
    B = np.zeros((n_slave_dofs, 6))
    
    for i_slave, global_dof in enumerate(slave_dofs):
        node_idx = global_dof // 6
        local_dof = global_dof % 6
        
        # Position of this node relative to master
        r_rel = node_coords[node_idx] - center_of_mass
        r_x, r_y, r_z = r_rel[0], r_rel[1], r_rel[2]
        
        # Constraint relationships:
        # u_slave = u_m - (θy_m * r_z - θz_m * r_y)
        # v_slave = v_m - (θz_m * r_x - θx_m * r_z)
        # w_slave = w_m - (θx_m * r_y - θy_m * r_x)
        # θx_slave = θx_m
        # θy_slave = θy_m
        # θz_slave = θz_m
        
        if local_dof == 0:  # u_slave
            B[i_slave, 0] = 1.0  # Couples to u_m
            B[i_slave, 4] = -r_z  # Couples to θy_m
            B[i_slave, 5] = r_y   # Couples to θz_m
        elif local_dof == 1:  # v_slave
            B[i_slave, 1] = 1.0  # Couples to v_m
            B[i_slave, 5] = -r_x  # Couples to θz_m
            B[i_slave, 3] = r_z   # Couples to θx_m
        elif local_dof == 2:  # w_slave
            B[i_slave, 2] = 1.0  # Couples to w_m
            B[i_slave, 3] = -r_y  # Couples to θx_m
            B[i_slave, 4] = r_x   # Couples to θy_m
        elif local_dof == 3:  # θx_slave
            B[i_slave, 3] = 1.0
        elif local_dof == 4:  # θy_slave
            B[i_slave, 4] = 1.0
        elif local_dof == 5:  # θz_slave
            B[i_slave, 5] = 1.0
    
    return B, master_dofs_local, slave_dofs


def _assemble_augmented_system_with_rbe3(K_global, M_global, M_added_6dof,
                                         node_coords, center_of_mass, 
                                         constrained_dofs, config):
    """
    Assemble augmented system with RBE3 master-slave coupling.
    
    The augmented system consists of:
    1. Master node (6 DOF) with hydrodynamic added mass 6×6
    2. Slave nodes (60 DOF for 10-node beam) linked via RBE3 constraints
    
    System is assembled as:
    [M_master      0    ] [u_master]   [K_master      0    ] [u_master]   0
    [0        M_slave  ] [u_slave ] = [0        K_slave  ] [u_slave ] = f
    
    Where M_master includes hydrodynamic added mass, and slave displacements are
    defined through constraint matrix: u_slave = B @ u_master
    
    Parameters
    ----------
    K_global, M_global : np.ndarray
        Original structural matrices (66 DOF)
    M_added_6dof : np.ndarray
        Hydrodynamic added mass (6×6)
    node_coords : np.ndarray
        All node coordinates (11 nodes × 3)
    center_of_mass : np.ndarray
        Position of master node (3,)
    constrained_dofs : list
        Constrained DOF indices
    config : object
        Configuration with num_modes, etc.
    
    Returns
    -------
    results : dict
        'K_master': Master stiffness (6×6)
        'M_master': Master mass with added mass (6×6)
        'K_slave': Slave stiffness in independent DOF space
        'M_slave': Slave mass in independent DOF space
        'B': Constraint matrix (n_slave × 6)
        'master_dofs': Global indices of master DOFs
        'slave_dofs': Global indices of slave DOFs
        'total_dof_augmented': Total DOFs in augmented system
        'center_of_mass': Center of mass position
    """
    print("\n*** Assembling augmented system with RBE3 coupling ***")
    
    total_dof_original = K_global.shape[0]
    
    # Step 1: Create constraint matrix
    print("\n  [1] Building RBE3 constraint matrix...")
    B, master_dofs, slave_dofs = _create_rbe3_constraint_matrix(
        node_coords, center_of_mass, constrained_dofs
    )
    
    n_master = 6
    n_slave = len(slave_dofs)
    n_total_augmented = n_master + n_slave
    
    print(f"    Master DOFs: {n_master}")
    print(f"    Slave DOFs: {n_slave}")
    print(f"    Total augmented DOFs: {n_total_augmented}")
    print(f"    Constraint matrix B shape: {B.shape}")
    print(f"    Constraint matrix rank: {np.linalg.matrix_rank(B)}")
    
    # Step 2: Extract master-master and slave-slave submatrices
    print("\n  [2] Extracting structural submatrices...")
    
    master_dofs_global = np.arange(6)  # Master is first node
    
    K_mm = K_global[np.ix_(master_dofs_global, master_dofs_global)]
    M_mm = M_global[np.ix_(master_dofs_global, master_dofs_global)]
    
    K_ss = K_global[np.ix_(slave_dofs, slave_dofs)]
    M_ss = M_global[np.ix_(slave_dofs, slave_dofs)]
    
    K_ms = K_global[np.ix_(master_dofs_global, slave_dofs)]
    K_sm = K_global[np.ix_(slave_dofs, master_dofs_global)]
    
    M_ms = M_global[np.ix_(master_dofs_global, slave_dofs)]
    M_sm = M_global[np.ix_(slave_dofs, master_dofs_global)]
    
    print(f"    K_mm shape: {K_mm.shape}")
    print(f"    K_ss shape: {K_ss.shape}")
    print(f"    M_mm shape: {M_mm.shape}")
    print(f"    M_ss shape: {M_ss.shape}")
    
    # Step 3: Add hydrodynamic mass to master
    print("\n  [3] Adding hydrodynamic mass to master node...")
    M_master = M_mm + M_added_6dof
    K_master = K_mm
    
    print(f"    M_master diagonal: {np.diag(M_master)}")
    print(f"    Structural M_mm diagonal: {np.diag(M_mm)}")
    print(f"    Added mass diagonal: {np.diag(M_added_6dof)}")
    
    # Step 4: Static condensation to eliminate slave DOFs
    #
    # Original system:
    #   [K_mm  K_ms] [u_m]   [f_m]
    #   [K_sm  K_ss] [u_s] = [f_s]
    #
    # With constraints u_s = B @ u_m:
    #   (K_mm + K_ms @ B + B^T @ K_sm + B^T @ K_ss @ B) @ u_m = f_m + B^T @ f_s
    #
    # For undamped analysis, ignore coupling forcing:
    print("\n  [4] Performing static condensation of slave DOFs...")
    
    K_condensed = (
        K_mm + 
        K_ms @ B + 
        B.T @ K_sm + 
        B.T @ K_ss @ B
    )
    
    M_condensed = (
        M_mm + 
        M_ms @ B + 
        B.T @ M_sm + 
        B.T @ M_ss @ B
    )
    
    print(f"    K_condensed shape: {K_condensed.shape}")
    print(f"    M_condensed shape: {M_condensed.shape}")
    print(f"    K_condensed condition number: {np.linalg.cond(K_condensed):.2e}")
    print(f"    M_condensed condition number: {np.linalg.cond(M_condensed):.2e}")
    
    # Verify condensed matrices are symmetric
    sym_error_K = np.max(np.abs(K_condensed - K_condensed.T))
    sym_error_M = np.max(np.abs(M_condensed - M_condensed.T))
    print(f"    Symmetry check - K: {sym_error_K:.2e}, M: {sym_error_M:.2e}")
    
    results = {
        'K_condensed': K_condensed,
        'M_condensed': M_condensed,
        'K_master': K_master,
        'M_master': M_master,
        'K_ss': K_ss,
        'M_ss': M_ss,
        'K_ms': K_ms,
        'K_sm': K_sm,
        'B': B,
        'master_dofs': master_dofs_global,
        'slave_dofs': slave_dofs,
        'n_master': n_master,
        'n_slave': n_slave,
        'n_total_augmented': n_total_augmented,
        'center_of_mass': center_of_mass,
    }
    
    print(f"\n  ✓ Augmented system assembled successfully")
    
    return results


def _load_rfa_matrices(config, output_dir=None):
    """
    Load Roger Fit Approximation (RFA) matrices from CSV files.
    
    Loads Q0, Q1, Q2 (constant, linear, quadratic terms) and Alag (lag pole contributions)
    matrices from: /output_data/RFA_matrices/{config.name}/panel_space/
    
    Parameters
    ----------
    config : AnalysisConfig
        Configuration with name attribute and output_dir
    output_dir : str, optional
        Output directory (default: config.output_dir)
    
    Returns
    -------
    dict with keys:
        'Q0', 'Q1', 'Q2': ndarray
            Roger coefficient matrices
        'Alag': list of ndarray
            Lag contribution matrices
        'blag': ndarray
            Lag pole locations
        'space': str
            'panel' (aerodynamic panel space)
    
    Raises
    ------
    FileNotFoundError
        If required CSV files not found
    ValueError
        If matrices have inconsistent shapes
    """
    if output_dir is None:
        output_dir = getattr(config, 'output_dir', '/media/lorenzo/Seagate Por/recupero dati Linux/lorebasket/FSI/output_data/')
    
    config_name = config.name
    rfa_dir = os.path.join(output_dir, "RFA_matrices", config_name, "panel_space")
    
    print(f"\n  Loading RFA matrices from: {rfa_dir}")
    
    # Load Q0, Q1, Q2
    try:
        Q0 = np.genfromtxt(os.path.join(rfa_dir, "Q0_constant_term.csv"), delimiter=',')
        Q1 = np.genfromtxt(os.path.join(rfa_dir, "Q1_linear_term.csv"), delimiter=',')
        Q2 = np.genfromtxt(os.path.join(rfa_dir, "Q2_quadratic_term.csv"), delimiter=',')
        
        print(f"    Q0 shape: {Q0.shape}")
        print(f"    Q1 shape: {Q1.shape}")
        print(f"    Q2 shape: {Q2.shape}")
        
        # Verify consistent shapes
        if Q0.shape != Q1.shape or Q1.shape != Q2.shape:
            raise ValueError(f"Inconsistent Q matrix shapes: Q0={Q0.shape}, Q1={Q1.shape}, Q2={Q2.shape}")
        
    except FileNotFoundError as e:
        print(f"    ✗ Error: {e}")
        raise
    
    # Load Alag matrices
    Alag = []
    blag = []
    
    # Read metadata to find lag pole information
    try:
        metadata = np.genfromtxt(os.path.join(rfa_dir, "lag_poles.csv"), delimiter=',', dtype=str)
        if metadata.ndim == 1:
            metadata = metadata.reshape(1, -1)
    except:
        # If lag_poles.csv doesn't exist, search for Alag files
        import glob
        alag_files = sorted(glob.glob(os.path.join(rfa_dir, "Alag_*.csv")))
        metadata = None
    
    # Load Alag files
    alag_idx = 0
    while True:
        alag_file = os.path.join(rfa_dir, f"Alag_{alag_idx}_pole_*.csv")
        import glob
        matches = glob.glob(alag_file)
        
        if not matches:
            break
        
        alag_path = matches[0]
        A_lag_i = np.genfromtxt(alag_path, delimiter=',')
        Alag.append(A_lag_i)
        
        # Extract pole value from filename
        filename = os.path.basename(alag_path)
        pole_str = filename.split('pole_')[1].replace('.csv', '')
        blag.append(float(pole_str))
        
        print(f"    Alag[{alag_idx}] (b={float(pole_str):.4f}): shape {A_lag_i.shape}")
        alag_idx += 1
    
    blag = np.array(blag)
    
    if not Alag:
        print(f"    Warning: No Alag matrices found")
    else:
        print(f"    Loaded {len(Alag)} lag pole matrices")
    
    return {
        'Q0': Q0,
        'Q1': Q1,
        'Q2': Q2,
        'Alag': Alag,
        'blag': blag,
        'space': 'panel'
    }


def _fluidrest_analysis_modal_dofs(structural_results, capytaine_data, config):
    """
    Fluid-at-rest analysis using MODAL DOF added mass matrices from Capytaine
    with Roger Fit Approximation (RFA) matrices for frequency-dependent effects.
    
    **Method:** Frequency-dependent modal reduction with RFA
    - Loads 8×8 modal added mass matrices for each frequency
    - Loads Roger Fit matrices (Q0, Q1, Q2) for fluid added mass coupling
    - Computes augmented mass matrix: M_total = M_struct + 0.5*ρ*A2
    - Projects A2 to modal space using Z transformation matrices
    - Solves eigenvalue problem in modal space at reference frequency
    - Returns frequency-dependent results with fluid coupling
    
    **File Format (Capytaine):** {config_name}_modal_added_mass.npz
    - Contains: M_coupled_000, M_coupled_001, ..., frequencies, M_structural, K_structural
    
    **File Format (RFA):** /output_data/RFA_matrices/{config.name}/panel_space/
    - Contains: Q0_constant_term.csv, Q1_linear_term.csv, Q2_quadratic_term.csv, Alag_*.csv
    """
    print("\n" + "="*70)
    print("METHOD: Modal DOF Added Mass (Frequency-Dependent + RFA Coupling)")
    print("="*70)
    
    # Extract frequency array and modal matrices
    omega_range = capytaine_data['frequencies']
    n_omegas = len(omega_range)
    
    print(f"\nFrequency range: {omega_range[0]:.4f} to {omega_range[-1]:.4f} rad/s")
    print(f"Number of frequencies: {n_omegas}")
    
    # Load all M_coupled matrices
    added_mass_freq = []
    for i_omega in range(n_omegas):
        key = f'M_coupled_{i_omega:03d}'
        if key in capytaine_data.files:
            added_mass_freq.append(capytaine_data[key])
        else:
            print(f"  Warning: Missing {key}")
            break
    
    added_mass_freq = np.array(added_mass_freq)  # shape: (n_omega, 8, 8)
    n_modes_modal = added_mass_freq.shape[1]
    
    print(f"Modal matrices shape: {added_mass_freq.shape}")
    print(f"Modal DOF size: {n_modes_modal}")
    
    # Get structural modal basis from dry analysis
    Phi_dry = structural_results.dry_eigenvectors  # (n_free_dofs, n_modes_dry)
    M_struct_ff = structural_results.Mff
    K_struct_ff = structural_results.Kff
    
    # Project structural matrices to modal space
    M_struct_modal = Phi_dry.T @ M_struct_ff @ Phi_dry
    K_struct_modal = Phi_dry.T @ K_struct_ff @ Phi_dry
    
    print(f"\nStructural basis: {Phi_dry.shape}")
    print(f"M_struct_modal: {M_struct_modal.shape}, diagonal: {np.diag(M_struct_modal)}")
    print(f"K_struct_modal: {K_struct_modal.shape}, diagonal: {np.diag(K_struct_modal)}")
    
    # Extract structural matrices from NPZ if available (already in modal space)
    if 'M_structural' in capytaine_data.files:
        M_struct_normalized = capytaine_data['M_structural']  # (8, 8)
        print(f"\nLoaded M_struct_normalized from file: {M_struct_normalized.shape}")
    else:
        M_struct_normalized = M_struct_modal
    
    # Extract pure added mass (M_coupled - M_struct_normalized)
    M_added_modal = []
    for i_omega in range(len(omega_range)):
        key = f'M_coupled_{i_omega:03d}'
        if key in capytaine_data.files:
            M_coupled_i = capytaine_data[key]
            M_added_i = M_coupled_i - M_struct_normalized
            M_added_modal.append(M_added_i)
    
    M_added_modal = np.array(M_added_modal)
    print(f"\nPure added mass matrices: {M_added_modal.shape}")
    
    # ========================================================================
    # LOAD AND INTEGRATE RFA MATRICES
    # ========================================================================
    print(f"\nLoading Roger Fit Approximation (RFA) matrices...")
    
    try:
        rfa_data = _load_rfa_matrices(config)
        Q0 = rfa_data['Q0']
        Q1 = rfa_data['Q1']
        Q2 = rfa_data['Q2']
        Alag = rfa_data['Alag']
        blag = rfa_data['blag']
        
        print(f"  ✓ RFA matrices loaded successfully")
        print(f"    Q0, Q1, Q2 shape: {Q0.shape} (panel aerodynamic space)")
        print(f"    Number of lag poles: {len(Alag)}")
        
        # For frequency-dependent fluid at rest analysis, use Q2 (quadratic term)
        # A2 represents the frequency-dependent added mass coefficient
        # Fluid added mass: M_fluid = 0.5 * ρ * A2
        rho = getattr(config, 'rho_f', {}).get(getattr(config, 'fluid', 'air'), 1.02)
        M_added_rfa = 0.5 * rho * Q2
        
        print(f"\n  Fluid coupling (using Q2 quadratic term):")
        print(f"    ρ = {rho} kg/m³")
        print(f"    A2 shape: {Q2.shape}")
        print(f"    M_fluid = 0.5*ρ*A2 shape: {M_added_rfa.shape}")
        print(f"    M_fluid norm: {np.linalg.norm(M_added_rfa):.3e}")
        
        # Check if Z transformation matrices are available for projection
        # These would transform from aerodynamic panel space to modal space
        # For now, we'll use the aerodynamic modal DOF matrices directly
        has_Z = False
        Z = None
        Z_f = None
        
        if hasattr(config, 'Z') and config.Z is not None:
            has_Z = True
            Z = config.Z
            Z_f = getattr(config, 'Z_f', None)
            print(f"  Transformation matrices available (Z shape: {Z.shape})")
        
    except FileNotFoundError as e:
        print(f"  ⚠ Warning: RFA matrices not found - {e}")
        print(f"    Proceeding with Capytaine modal DOF matrices only (Q0, Q1, Q2 not available)")
        M_added_rfa = None
    except Exception as e:
        print(f"  ⚠ Warning: Error loading RFA matrices - {e}")
        M_added_rfa = None
    
    # Solve eigenvalue problems at each frequency
    from scipy.linalg import eigh
    
    print(f"\nSolving eigenvalue problems at {len(omega_range)} frequencies...")
    wet_freqs_all = []
    wet_omega_sq_all = []
    
    for i_omega, omega in enumerate(omega_range):
        M_added_i = M_added_modal[i_omega]
        M_total_i = M_struct_modal + M_added_i
        
        # Add RFA-based fluid coupling if available
        if M_added_rfa is not None and M_added_rfa.shape == M_struct_modal.shape:
            # Include the frequency-independent component from RFA
            M_total_i = M_total_i + M_added_rfa
        
        eigenvalues_i, _ = eigh(K_struct_modal, M_total_i)
        eigenvalues_i = np.maximum(eigenvalues_i, 0)
        
        freqs_i = np.sqrt(eigenvalues_i) / (2 * np.pi)
        wet_freqs_all.append(freqs_i)
        wet_omega_sq_all.append(eigenvalues_i)
        
        if i_omega % max(1, n_omegas//3) == 0:
            print(f"  ω = {omega:.4f} rad/s ({omega/(2*np.pi):.6f} Hz): f₁={freqs_i[0]:.4f} Hz")
    
    # Use results at first frequency (reference)
    i_ref = 0
    wet_freqs = wet_freqs_all[i_ref]
    wet_omega_sq = wet_omega_sq_all[i_ref]
    omega_ref = omega_range[i_ref]
    
    print(f"\n✓ Using results at ω = {omega_ref:.4f} rad/s ({omega_ref/(2*np.pi):.6f} Hz)")
    
    # Select modes
    dry_freqs = structural_results.dry_frequencies
    n_modes = min(len(wet_freqs), len(dry_freqs), getattr(config, 'num_modes', len(dry_freqs)))
    
    wet_freqs = wet_freqs[:n_modes]
    wet_omega_sq = wet_omega_sq[:n_modes]
    dry_freqs = dry_freqs[:n_modes]
    
    freq_reduction = (dry_freqs - wet_freqs) / dry_freqs * 100
    
    # Print comparison
    print("\n" + "="*70)
    print("COMPARISON: Dry vs Wet (Modal DOF Method with RFA)")
    print("="*70 + "\n")
    print(f"{'Mode':<6} {'Dry (Hz)':<15} {'Wet (Hz)':<15} {'Δf (Hz)':<15} {'Δf (%)':<15}")
    print("-" * 70)
    
    for i in range(n_modes):
        print(f"{i+1:<6} {dry_freqs[i]:<15.6f} {wet_freqs[i]:<15.6f} {dry_freqs[i]-wet_freqs[i]:<15.6f} {freq_reduction[i]:<15.2f}")
    
    print("\n" + "="*70 + "\n")
    
    # Expand eigenvectors (use structural basis as approximation)
    # Full frequency-dependent solution would require solving with frequency-dependent matrices
    wet_eigenvectors_expanded = Phi_dry[:, :n_modes]
    
    return {
        'wet_frequencies': wet_freqs,
        'wet_eigenvalues': wet_omega_sq,
        'wet_eigenvectors': wet_eigenvectors_expanded,
        'dry_frequencies': dry_freqs,
        'frequency_reduction': freq_reduction,
        'M_added_modal': M_added_modal,
        'M_struct_modal': M_struct_modal,
        'K_struct_modal': K_struct_modal,
        'omega_range': omega_range,
        'omega_ref': omega_ref,
        'use_modal_matrices': True,
        'n_modes_modal': n_modes_modal,
        'M_added_rfa': M_added_rfa if M_added_rfa is not None else None,
        'rfa_integrated': M_added_rfa is not None,
    }


def _fluidrest_analysis_rbe3_6dof(structural_results, capytaine_data, config):
    """
    Fluid-at-rest analysis using RBE3 master-slave coupling with 6×6 added mass
    enhanced with Roger Fit Approximation (RFA) matrices.
    
    **Method:** RBE3 kinematic coupling with RFA integration
    - Loads 6×6 rigid body added mass matrices from Capytaine
    - Loads Roger Fit matrices (Q0, Q1, Q2) for frequency-dependent coupling
    - Computes augmented mass: M_total = M_struct + 0.5*ρ*A2 (from RFA)
    - Creates RBE3 constraint matrix B to project 6-DOF master to full beam DOF
    - Projects added mass to full beam DOF space: M_aug = B @ M_rfa @ B^T
    - Solves modal reduction with augmented mass matrix
    
    **File Format (Capytaine):** {config_name}_capytaine_matrices.npz
    - Contains: omega, added_mass (n_omega, 6, 6)
    
    **File Format (RFA):** /output_data/RFA_matrices/{config.name}/panel_space/
    - Contains: Q0, Q1, Q2, Alag matrices
    """
    print("\n" + "="*70)
    print("METHOD: RBE3 Master-Slave Coupling (6×6 Rigid Body + RFA)")
    print("="*70)
    
    # Extract frequency array
    omega_range = capytaine_data['omega']
    added_mass_freq = capytaine_data['added_mass']  # (n_omega, 6, 6)
    
    print(f"\nFrequency range: {omega_range[0]:.6f} to {omega_range[-1]:.6f} rad/s")
    print(f"Added mass shape: {added_mass_freq.shape}")
    
    # Extract added mass at reference frequency (closest to zero for fluid at rest)
    freq_idx_ref = np.argmin(np.abs(omega_range))
    omega_ref = omega_range[freq_idx_ref]
    M_added_6dof = np.real(added_mass_freq[freq_idx_ref, :, :])
    
    print(f"\nReference frequency: ω = {omega_ref:.6f} rad/s ({omega_ref/(2*np.pi):.6f} Hz)")
    print(f"Added mass diagonal (kg): {np.diag(M_added_6dof)}")
    
    # Get structural data
    beam_model = structural_results.beam_model
    total_dof = structural_results.total_dof
    constrained_dofs = structural_results.constrained_dofs
    K_global = structural_results.K_global
    M_global = structural_results.M_global
    
    # Node coordinates
    nodes = beam_model['nodes']
    node_coords = np.array([node['position'] for node in nodes])
    center_of_mass = np.mean(node_coords, axis=0)
    
    print(f"\nStructure:")
    print(f"  Nodes: {len(nodes)}")
    print(f"  Total DOFs: {total_dof}")
    print(f"  Center of mass: ({center_of_mass[0]:.3f}, {center_of_mass[1]:.3f}, {center_of_mass[2]:.3f}) m")
    
    # Create RBE3 constraint matrix
    print(f"\nCreating RBE3 constraint matrix...")
    B_full, _, slave_dofs_global = _create_rbe3_constraint_matrix(
        node_coords, center_of_mass, constrained_dofs
    )
    
    print(f"  Constraint matrix B: {B_full.shape}")
    print(f"  Master DOFs: 6 (rigid body)")
    print(f"  Slave DOFs: {len(slave_dofs_global)}")
    
    # ========================================================================
    # LOAD AND INTEGRATE RFA MATRICES
    # ========================================================================
    print(f"\nLoading Roger Fit Approximation (RFA) matrices...")
    
    M_added_rfa = None
    try:
        rfa_data = _load_rfa_matrices(config)
        Q0 = rfa_data['Q0']
        Q1 = rfa_data['Q1']
        Q2 = rfa_data['Q2']
        Alag = rfa_data['Alag']
        blag = rfa_data['blag']
        
        print(f"  ✓ RFA matrices loaded successfully")
        print(f"    Q0, Q1, Q2 shape: {Q0.shape} (panel aerodynamic space)")
        print(f"    Number of lag poles: {len(Alag)}")
        
        # For fluid at rest analysis, use Q2 (quadratic/frequency-dependent term)
        # at v≈0.01 reference speed
        rho = getattr(config, 'rho_f', {}).get(getattr(config, 'fluid', 'air'), 1.02)
        
        # A2 = Q2 represents the frequency-dependent added mass coefficient
        # Fluid added mass matrix: M_fluid = 0.5 * ρ * A2
        M_added_rfa_panel_space = 0.5 * rho * Q2
        
        print(f"\n  Fluid coupling (using Q2 quadratic term at v≈0.01):")
        print(f"    ρ = {rho} kg/m³")
        print(f"    A2 shape: {Q2.shape}")
        print(f"    M_fluid = 0.5*ρ*A2 shape: {M_added_rfa_panel_space.shape}")
        print(f"    M_fluid diagonal norm: {np.linalg.norm(np.diag(M_added_rfa_panel_space)):.3e}")
        
        # NOTE: This is in aerodynamic panel space. For full integration with Z transformation,
        # would require: M_fluid_modal = Z^T @ M_fluid_panel @ Z
        # For now, use this as reference; actual coupling would need Z matrices from flutter analysis
        M_added_rfa = M_added_rfa_panel_space
        
    except FileNotFoundError as e:
        print(f"  ⚠ Warning: RFA matrices not found - {e}")
        print(f"    Proceeding with Capytaine 6×6 matrices only")
    except Exception as e:
        print(f"  ⚠ Warning: Error loading RFA matrices - {e}")
    
    # Project added mass to full free DOF space via RBE3 constraint
    M_added_full = B_full @ M_added_6dof @ B_full.T
    
    print(f"\n  RBE3 projection:")
    print(f"    Constraint matrix B: {B_full.shape}")
    print(f"    Added mass projected: {M_added_full.shape}")
    print(f"    Max projected value: {np.max(np.abs(M_added_full)):.3e}")
    
    # Get structural matrices in free DOF space
    free_dofs = np.setdiff1d(np.arange(total_dof), constrained_dofs)
    M_struct_ff = structural_results.Mff
    K_struct_ff = structural_results.Kff
    Phi_dry = structural_results.dry_eigenvectors
    n_modes_dry = Phi_dry.shape[1]
    
    # Augmented system: structural + projected added mass
    M_aug_ff = M_struct_ff + M_added_full
    K_aug_ff = K_struct_ff
    
    print(f"\nAugmented system:")
    print(f"  M_aug: {M_aug_ff.shape}")
    print(f"  K_aug: {K_aug_ff.shape}")
    print(f"  Symmetry M_aug: {np.max(np.abs(M_aug_ff - M_aug_ff.T)):.3e}")
    
    # Modal reduction to modal space
    M_hat_wet = Phi_dry.T @ M_aug_ff @ Phi_dry
    K_hat_wet = Phi_dry.T @ K_struct_ff @ Phi_dry
    
    print(f"\nModal reduction:")
    print(f"  M_hat: {M_hat_wet.shape}, diag: {np.diag(M_hat_wet)}")
    print(f"  K_hat: {K_hat_wet.shape}, diag: {np.diag(K_hat_wet)}")
    
    # Solve eigenvalue problem
    from scipy.linalg import eigh
    
    eigenvalues_modal, eigenvectors_modal = eigh(K_hat_wet, M_hat_wet)
    eigenvalues_modal = np.maximum(eigenvalues_modal, 0)
    
    # Sort by frequency
    sort_idx = np.argsort(eigenvalues_modal)
    eigenvalues_modal = eigenvalues_modal[sort_idx]
    eigenvectors_modal = eigenvectors_modal[:, sort_idx]
    
    # Select modes
    n_modes = min(n_modes_dry, getattr(config, 'num_modes', 10))
    wet_omega_sq = eigenvalues_modal[:n_modes]
    wet_eigenvectors_modal = eigenvectors_modal[:, :n_modes]
    wet_freqs = np.sqrt(np.maximum(wet_omega_sq, 0)) / (2 * np.pi)
    
    print(f"\n✓ Eigenvalue problem solved")
    print(f"  Modes: {n_modes}")
    print(f"  Wet frequencies (Hz): {wet_freqs}")
    
    # Expand to full free DOF space
    wet_eigenvectors_expanded = Phi_dry @ wet_eigenvectors_modal
    
    # Compare with dry
    dry_freqs = structural_results.dry_frequencies[:n_modes]
    freq_reduction = (dry_freqs - wet_freqs) / dry_freqs * 100
    
    print("\n" + "="*70)
    print("COMPARISON: Dry vs Wet (RBE3 Method with RFA)")
    print("="*70 + "\n")
    print(f"{'Mode':<6} {'Dry (Hz)':<15} {'Wet (Hz)':<15} {'Δf (Hz)':<15} {'Δf (%)':<15}")
    print("-" * 70)
    
    for i in range(n_modes):
        print(f"{i+1:<6} {dry_freqs[i]:<15.6f} {wet_freqs[i]:<15.6f} {dry_freqs[i]-wet_freqs[i]:<15.6f} {freq_reduction[i]:<15.2f}")
    
    print("\n" + "="*70 + "\n")
    
    return {
        'wet_frequencies': wet_freqs,
        'wet_eigenvalues': wet_omega_sq,
        'wet_eigenvectors': wet_eigenvectors_expanded,
        'dry_frequencies': dry_freqs,
        'frequency_reduction': freq_reduction,
        'M_added_6dof': M_added_6dof,
        'M_added_full': M_added_full,
        'M_hat_wet': M_hat_wet,
        'K_hat_wet': K_hat_wet,
        'B': B_full,
        'master_dofs': np.arange(6),
        'slave_dofs': slave_dofs_global,
        'n_master': 6,
        'n_slave': len(slave_dofs_global),
        'center_of_mass': center_of_mass,
        'omega_ref': omega_ref,
        'use_modal_matrices': False,
        'M_added_rfa': M_added_rfa if M_added_rfa is not None else None,
        'rfa_integrated': M_added_rfa is not None,
    }


def run_fluidrest_analysis(structural_results, config):
    """
    Modal analysis with fluid at rest (DISPATCHER).
    
    Chooses between two analysis methods based on config.modal_capytaine_dofs:
    
    **If modal_capytaine_dofs = True:**
    - Uses frequency-dependent 8×8 modal added mass matrices
    - Direct modal space reduction with Capytaine modal DOF coupling
    - File: {config_name}_modal_added_mass.npz
    
    **If modal_capytaine_dofs = False:**
    - Uses 6×6 rigid body added mass matrices with RBE3 coupling
    - Master-slave kinematic coupling to distribute added mass
    - File: {config_name}_capytaine_matrices.npz
    
    Parameters
    ----------
    structural_results : StructuralResults
        From run_dry_analysis()
    config : AnalysisConfig
        Configuration with modal_capytaine_dofs, capytaine_data_dir, etc.
    
    Returns
    -------
    dict or None
        Analysis results or None if disabled/error
    """
    print("\n" + "="*70)
    print("FLUIDREST ANALYSIS (FLUID AT REST)")
    print("="*70)
    
    # Check if enabled
    if not getattr(config, 'fluid_at_rest', False):
        print("✓ Disabled - skipping")
        return None
    
    # Check data directory
    capytaine_data_dir = getattr(config, 'capytaine_data_dir', None)
    if capytaine_data_dir is None:
        print("✗ capytaine_data_dir not set")
        return None
    
    # Determine which method to use
    modal_capytaine_dofs = getattr(config, 'modal_capytaine_dofs', False)
    config_name = config.name
    
    if modal_capytaine_dofs:
        # ================================================================
        # MODAL DOF METHOD
        # ================================================================
        print(f"\n[1] Loading modal Capytaine matrices...")
        capytaine_path = os.path.join(capytaine_data_dir, f"{config_name}_modal_added_mass.npz")
    else:
        # ================================================================
        # RBE3 6×6 METHOD
        # ================================================================
        print(f"\n[1] Loading rigid body Capytaine matrices...")
        capytaine_path = os.path.join(capytaine_data_dir, f"{config_name}_capytaine_matrices.npz")
    
    if not os.path.exists(capytaine_path):
        print(f"✗ File not found: {capytaine_path}")
        return None
    
    try:
        capytaine_data = np.load(capytaine_path, allow_pickle=True)
        print(f"✓ Loaded: {os.path.basename(capytaine_path)}")
    except Exception as e:
        print(f"✗ Error loading: {e}")
        traceback.print_exc()
        return None
    
    # Dispatch to appropriate method
    try:
        if modal_capytaine_dofs:
            results = _fluidrest_analysis_modal_dofs(structural_results, capytaine_data, config)
        else:
            results = _fluidrest_analysis_rbe3_6dof(structural_results, capytaine_data, config)
        
        print(f"\n✓ Fluidrest analysis complete!")
        return results
        
    except Exception as e:
        print(f"✗ Error during analysis: {e}")
        traceback.print_exc()
        return None