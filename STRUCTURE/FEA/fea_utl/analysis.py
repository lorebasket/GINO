import numpy as np
from scipy.linalg import solve, eigh, eig


def _save_modal_data_to_csv(beam_model, total_dof, constrained_dofs, dry_values, dry_vectors, 
                            config, output_suffix="_pre_modal_analysis"):
    """
    Save eigenvalues, eigenvectors, and node coordinates to CSV files before modal analysis.
    
    This function creates three CSV files:
    1. <config_name><output_suffix>_eigendata.csv: eigenvalues and eigenvectors
    2. <config_name><output_suffix>_nodes.csv: all node coordinates
    3. <config_name><output_suffix>_constrained_dofs.csv: constrained node information
    4. <config_name><output_suffix>_free_dofs.csv: free DOF information
    
    Parameters
    ----------
    beam_model : dict
        Structural model with 'nodes' and 'elements'
    total_dof : int
        Total degrees of freedom in the system
    constrained_dofs : list
        List of constrained DOF indices
    dry_values : np.ndarray
        Eigenvalues (omega^2) from dry analysis
    dry_vectors : np.ndarray
        Eigenvectors from dry analysis (shape: n_free_dofs × n_modes)
    config : object
        Configuration object with 'name' and optional 'output_dir'
    output_suffix : str, optional
        Suffix for output filenames (default: "_pre_modal_analysis")
    
    Returns
    -------
    None
    """
    import csv
    import os
    
    print(f"\n*** Saving modal data to CSV files ***")
    
    output_dir = getattr(config, 'output_dir', '.')
    os.makedirs(output_dir, exist_ok=True)
    config_name = config.name
    
    # Calculate derived quantities
    n_nodes = total_dof // 6
    n_modes = dry_vectors.shape[1] if len(dry_vectors.shape) > 1 else 1
    free_dofs = np.setdiff1d(np.arange(total_dof), constrained_dofs)
    
    # Convert eigenvalues to frequencies (Hz)
    dry_frequencies = np.sqrt(dry_values) / (2 * np.pi)
    
    # Get node coordinates
    nodes = beam_model['nodes']
    node_coords = np.array([node['position'] for node in nodes])
    
    # ========================================================================
    # 1. SAVE EIGENVALUES AND EIGENVECTORS
    # ========================================================================
    eigendata_file = os.path.join(output_dir, f"{config_name}{output_suffix}_eigendata.csv")
    
    try:
        with open(eigendata_file, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header: Mode metadata + all DOF components for eigenvectors
            header = ['Mode', 'Frequency_Hz', 'Omega_rad_s', 'Omega_squared_rad2_s2']
            # Add columns for each DOF component of the eigenvectors
            for dof_idx in free_dofs:
                node_idx = dof_idx // 6
                local_dof = dof_idx % 6
                dof_names = ['Ux', 'Uy', 'Uz', 'Rx', 'Ry', 'Rz']
                header.append(f'Node_{node_idx}_{dof_names[local_dof]}')
            writer.writerow(header)
            
            # Data rows: one row per mode with full eigenvector
            for mode_idx in range(n_modes):
                freq_hz = dry_frequencies[mode_idx]
                omega_rad_s = np.sqrt(dry_values[mode_idx])
                omega_sq = dry_values[mode_idx]
                
                row = [mode_idx+1, f'{freq_hz:.8f}', f'{omega_rad_s:.8f}', f'{omega_sq:.8f}']
                
                # Add all eigenvector components for this mode
                # dry_vectors shape: (n_free_dofs, n_modes)
                # We want all components of mode_idx
                eigenvector_mode = dry_vectors[:, mode_idx]
                for component in eigenvector_mode:
                    row.append(f'{component:.10e}')
                
                writer.writerow(row)
        
        print(f"  ✓ Saved eigendata to: {eigendata_file}")
    except Exception as e:
        print(f"  ✗ Error saving eigendata: {e}")
    
    # ========================================================================
    # 2. SAVE ALL NODE COORDINATES
    # ========================================================================
    nodes_file = os.path.join(output_dir, f"{config_name}{output_suffix}_nodes.csv")
    
    try:
        with open(nodes_file, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow(['Node_Index', 'X_coord_m', 'Y_coord_m', 'Z_coord_m', 'Node_ID', 'Constrained'])
            
            # Data rows
            for i, node in enumerate(nodes):
                node_dofs = list(range(i*6, (i+1)*6))
                is_constrained = any(dof in constrained_dofs for dof in node_dofs)
                node_id = node.get('id', f"N{i}")
                
                row = [
                    i,
                    f'{node_coords[i, 0]:.8f}',
                    f'{node_coords[i, 1]:.8f}',
                    f'{node_coords[i, 2]:.8f}',
                    node_id,
                    'Yes' if is_constrained else 'No'
                ]
                writer.writerow(row)
        
        print(f"  ✓ Saved node coordinates to: {nodes_file}")
    except Exception as e:
        print(f"  ✗ Error saving node coordinates: {e}")
    
    # ========================================================================
    # 3. SAVE CONSTRAINED NODES INFORMATION
    # ========================================================================
    constrained_file = os.path.join(output_dir, f"{config_name}{output_suffix}_constrained_dofs.csv")
    
    try:
        with open(constrained_file, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow(['Node_Index', 'Constrained_DOF_Indices', 'X_coord_m', 'Y_coord_m', 'Z_coord_m', 'DOF_Type'])
            
            # Group constrained DOFs by node
            constrained_nodes_dict = {}
            for dof in constrained_dofs:
                node_idx = dof // 6
                local_dof = dof % 6
                
                if node_idx not in constrained_nodes_dict:
                    constrained_nodes_dict[node_idx] = []
                constrained_nodes_dict[node_idx].append(local_dof)
            
            # Write constrained nodes
            dof_names = ['Ux', 'Uy', 'Uz', 'Rx', 'Ry', 'Rz']
            for node_idx in sorted(constrained_nodes_dict.keys()):
                local_dofs = sorted(constrained_nodes_dict[node_idx])
                dof_indices_str = ','.join(str(d) for d in local_dofs)
                dof_types_str = ','.join(dof_names[d] for d in local_dofs)
                
                row = [
                    node_idx,
                    dof_indices_str,
                    f'{node_coords[node_idx, 0]:.8f}',
                    f'{node_coords[node_idx, 1]:.8f}',
                    f'{node_coords[node_idx, 2]:.8f}',
                    dof_types_str
                ]
                writer.writerow(row)
        
        print(f"  ✓ Saved constrained DOFs to: {constrained_file}")
    except Exception as e:
        print(f"  ✗ Error saving constrained DOFs: {e}")
    
    # ========================================================================
    # 4. SAVE FREE DOFS INFORMATION
    # ========================================================================
    free_dofs_file = os.path.join(output_dir, f"{config_name}{output_suffix}_free_dofs.csv")
    
    try:
        with open(free_dofs_file, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow(['Global_DOF_Index', 'Node_Index', 'Local_DOF', 'DOF_Type', 'X_coord_m', 'Y_coord_m', 'Z_coord_m'])
            
            # Write all free DOFs
            dof_names = ['Ux', 'Uy', 'Uz', 'Rx', 'Ry', 'Rz']
            for dof_idx in free_dofs:
                node_idx = dof_idx // 6
                local_dof = dof_idx % 6
                dof_type = dof_names[local_dof]
                
                row = [
                    dof_idx,
                    node_idx,
                    local_dof,
                    dof_type,
                    f'{node_coords[node_idx, 0]:.8f}',
                    f'{node_coords[node_idx, 1]:.8f}',
                    f'{node_coords[node_idx, 2]:.8f}'
                ]
                writer.writerow(row)
        
        print(f"  ✓ Saved free DOFs to: {free_dofs_file}")
    except Exception as e:
        print(f"  ✗ Error saving free DOFs: {e}")
    except Exception as e:
        print(f"  ✗ Error saving free DOFs: {e}")
    
    print(f"  Summary: {len(free_dofs)} free DOFs, {len(constrained_dofs)} constrained DOFs")
    print(f"  Output directory: {output_dir}")


def create_force_vector(load_magnitude, total_dof, axis='z', tip_only=True, dof_per_node=6, n_nodes=None):

    if n_nodes is None:
        n_nodes = total_dof // dof_per_node
    
    f = np.zeros(total_dof, dtype=float)
    
    tip = (n_nodes - 1) * dof_per_node
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    
    j = axis_map.get(axis, 2)  # default z
    
    f[tip + j] = load_magnitude
    
    return f


def cantilever_beam(Matrix, total_dof, dof_per_node=6):

    if constrained_dofs is None:
        constrained_dofs = list(range(dof_per_node))  # clamp 6dof at node 0 by default

    all_dofs = np.arange(total_dof, dtype=int)
    mask = np.ones(total_dof, dtype=bool)
    mask[np.array(constrained_dofs, dtype=int)] = False
    free_dofs = all_dofs[mask]
    fixed_dofs = np.array(constrained_dofs, dtype=int)

    # Reduce matrices
    Matrix_reduced = Matrix[np.ix_(free_dofs, free_dofs)]
    
    return Matrix_reduced


def solve_static_analysis(K_global, M_global, force_vector, total_dof, config, beam_model, constrained_dofs=None, dof_per_node=6, num_modes=6):
    """
    Solve static analysis and extract eigenmodes with improved numerical stability.
    """
    if constrained_dofs is None:
        constrained_dofs = list(range(dof_per_node))  # clamp 6dof at node 0 by default

    all_dofs = np.arange(total_dof, dtype=int)
    mask = np.ones(total_dof, dtype=bool)
    mask[np.array(constrained_dofs, dtype=int)] = False
    free_dofs = all_dofs[mask]
    fixed_dofs = np.array(constrained_dofs, dtype=int)

    # Reduce matrices
    K_constrained = K_global[np.ix_(free_dofs, free_dofs)].astype(np.float64)
    M_constrained = M_global[np.ix_(free_dofs, free_dofs)].astype(np.float64)
    F = force_vector[free_dofs].astype(np.float64)


    # Check matrix conditions
    def check_matrix_condition(M, name):
        cond = np.linalg.cond(M)
        print(f"Condition number of {name}: {cond:.2e}")
        if cond > 1e12:
            print(f"Warning: {name} is ill-conditioned!")
    
    check_matrix_condition(K_constrained, "stiffness matrix K_constrained")
    check_matrix_condition(M_constrained, "mass matrix M_constrained")

    # Solve static problem only if there are non-zero forces
    u_full = np.zeros(total_dof, dtype=float)
    if np.any(F != 0):
        try:
            u_f = solve(K_constrained, F, assume_a='pos')
            u_full[free_dofs] = u_f
        except np.linalg.LinAlgError:
            print("Warning: Direct solver failed. Trying least squares solution...")
            u_f = np.linalg.lstsq(K_constrained, F, rcond=None)[0]
            u_full[free_dofs] = u_f
    else:
        print("No applied forces - skipping static solve, proceeding to modal analysis.")

    # Calculate reactions
    R = K_global[np.ix_(fixed_dofs, all_dofs)] @ u_full - force_vector[fixed_dofs]

    # Solve eigenvalue problem for modal analysis
    _M_scale = np.linalg.norm(M_constrained, 'fro')
    _eps_reg  = 1e-10 * (_M_scale if _M_scale > 0 else 1.0)
    M_constrained_reg = M_constrained + _eps_reg * np.eye(M_constrained.shape[0])
    # driver='gv': il default 'gvd' (divide & conquer) con autovettori perde modi quando M e'
    # quasi singolare (link rigidi senza massa): TNZ dava 9.55 Hz al posto di 3.55 e 21.7.
    dry_values, dry_vectors = eigh(K_constrained, M_constrained_reg, driver='gv')
    
    # Sort by eigenvalue magnitude and select first num_modes
    idx = np.argsort(np.abs(dry_values))[:num_modes]
    dry_values = dry_values[idx]
    dry_vectors = dry_vectors[:, idx]

    if config.save_modal_data:
        _save_modal_data_to_csv(
            beam_model, total_dof, constrained_dofs, 
            dry_values, dry_vectors, config,
            output_suffix="_dry_egv"
        )
    
    return u_full, R, dry_values, dry_vectors, M_constrained, K_constrained


def compute_rayleigh_ab_from_targets(omega1, omega2, zeta1, zeta2, cond_limit=1e12):
    """Solve for (alpha, beta) matching zeta at exactly two frequencies."""
    return compute_rayleigh_ab_from_modes(
        (omega1, omega2), (zeta1, zeta2), cond_limit=cond_limit
    )


def compute_rayleigh_ab_from_modes(omegas, zetas, cond_limit=1e12):
    """
    Fit Rayleigh coefficients C = alpha*M + beta*K.

    For each mode: zeta_i = alpha/(2*omega_i) + beta*omega_i/2.
    Two modes → exact solve; more modes → least squares.
    """
    omegas = np.asarray(omegas, dtype=float)
    zetas = np.asarray(zetas, dtype=float)
    if omegas.shape != zetas.shape or omegas.ndim != 1:
        raise ValueError("omegas and zetas must be 1-D arrays of equal length")
    n = len(omegas)
    if n < 1:
        raise ValueError("At least one (omega, zeta) pair is required")
    if np.any(omegas <= 1e-12):
        raise ValueError("All Rayleigh target frequencies must be positive")

    if n == 1:
        w = omegas[0]
        return 2.0 * zetas[0] * w, 0.0

    A = np.column_stack([1.0 / (2.0 * omegas), omegas / 2.0])
    b = zetas
    if n == 2 and np.all(np.isfinite(A)):
        try:
            if np.linalg.cond(A) <= cond_limit:
                return np.linalg.solve(A, b)
        except Exception:
            pass
    return np.linalg.lstsq(A, b, rcond=None)[0]


def build_rayleigh_damping(M, K, omega1, omega2, zeta1, zeta2):
    alpha, beta = compute_rayleigh_ab_from_targets(omega1, omega2, zeta1, zeta2)
    C = alpha * M + beta * K
    return C, alpha, beta


def _pick_two_distinct_omegas(omegas, tol_rel=1e-3, eps=1e-10):
    """
    Return indices of two distinct nonzero omegas. 
    Raises if not found.
    """
    idx = []
    for k, w in enumerate(omegas):
        if w > eps and all(abs(w - omegas[j]) / max(w, omegas[j]) > tol_rel for j in idx):
            idx.append(k)
        if len(idx) == 2:
            break
    if len(idx) < 2:
        raise RuntimeError("Could not find two distinct nonzero modes for Rayleigh fit. Increase num_modes_search or check model.")
    return idx[0], idx[1]


def rayleigh_from_modes(K_ff, M_ff,
                        dry_vectors, dry_eigvals,
                        target_mode_ids=(0, 5),
                        target_zetas=(0.1, 0.1),
                        verbose=True):
    """
    Build global Rayleigh damping from dry modal targets.

    target_mode_ids and target_zetas must have the same length (typically
    num_modes_flutter_egv). Two targets use an exact 2×2 solve; more targets
    use least squares on zeta_i = alpha/(2ω_i) + β ω_i/2.
    """
    mode_ids = tuple(int(i) for i in np.atleast_1d(target_mode_ids))
    zetas = tuple(float(z) for z in np.atleast_1d(target_zetas))
    if len(mode_ids) != len(zetas):
        raise ValueError(
            f"rayleigh mode count ({len(mode_ids)}) must match zeta count ({len(zetas)})"
        )

    if len(mode_ids) == 2:
        i, j = mode_ids
        use_pair = (max(i, j) < len(dry_eigvals) and
                    min(dry_eigvals[i], dry_eigvals[j]) > 1e-10 and
                    abs(dry_eigvals[i] - dry_eigvals[j]) / max(dry_eigvals[i], dry_eigvals[j]) > 1e-3)
        if not use_pair:
            if verbose:
                print("[Rayleigh] Requested mode pair invalid or nearly equal; auto-selecting.")
            i, j = _pick_two_distinct_omegas(dry_eigvals)
        mode_ids = (i, j)

    omegas = []
    for mid in mode_ids:
        if mid < 0 or mid >= len(dry_eigvals):
            raise ValueError(f"Rayleigh mode id {mid} out of range (n_modes={len(dry_eigvals)})")
        ev = dry_eigvals[mid]
        if ev <= 1e-10:
            raise ValueError(f"Rayleigh mode {mid} has non-positive eigenvalue {ev}")
        omegas.append(np.sqrt(ev))
    omegas = np.asarray(omegas, dtype=float)
    zetas_arr = np.asarray(zetas[:len(mode_ids)], dtype=float)

    if len(omegas) < 2:
        i, j = _pick_two_distinct_omegas(dry_eigvals)
        mode_ids = (i, j)
        omegas = np.array([np.sqrt(dry_eigvals[i]), np.sqrt(dry_eigvals[j])])
        zetas_arr = np.asarray(zetas[:2], dtype=float)

    alpha, beta = compute_rayleigh_ab_from_modes(omegas, zetas_arr)
    C_global = alpha * M_ff + beta * K_ff

    if verbose:
        fit = "exact 2-mode" if len(omegas) == 2 else f"LSQ over {len(omegas)} modes"
        print(f"[Rayleigh] {fit}: alpha={alpha:.3e}, beta={beta:.3e}")
        for mid, w, z_tgt in zip(mode_ids, omegas, zetas_arr):
            z_ach = alpha / (2.0 * w) + beta * w / 2.0
            print(
                f"  mode {mid}: omega={w:.6f} rad/s, "
                f"zeta_target={z_tgt:.4g}, zeta_achieved={z_ach:.4g}"
            )

    return C_global, alpha, beta, tuple(float(w) for w in omegas)


def rayleigh_from_two_modes(K_ff, M_ff, dry_vectors, dry_eigvals, **kwargs):
    """Backward-compatible alias for two-mode Rayleigh damping."""
    return rayleigh_from_modes(K_ff, M_ff, dry_vectors, dry_eigvals, **kwargs)


def extract_displacements(u_full, n_nodes, dof_per_node=6):
    
    displacements = []
    for i in range(n_nodes):
        s = i * dof_per_node
        node_disp = {
            'node': i,
            'x': u_full[s + 0],
            'y': u_full[s + 1],
            'z': u_full[s + 2],
            'rx': u_full[s + 3],
            'ry': u_full[s + 4],
            'rz': u_full[s + 5]
        }
        displacements.append(node_disp)

    return displacements


