# FEA/fea_utl/guyan_reduction.py
"""
Guyan Static Reduction (Component Mode Synthesis - CMS)

This module implements the Guyan static condensation technique to reduce
the model size by eliminating massless DOFs (e.g., foil nodes in the TNZ case).

The method partitions the system into:
  - Master DOFs (m): arm nodes + ghost point-mass node (have inertia)
  - Slave DOFs (s):  foil nodes (zero mass, kept for aerodynamic geometry)

The slave DOFs are statically condensed out, producing:
  - Reduced matrices K_star, M_star (master-only)
  - Expansion matrix T_slave for recovery of slave DOF modes
"""

import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import csc_matrix, csr_matrix


def classify_dofs_by_mass(M_free, beam_model, free_dofs, mass_tol=1e-12):
    """
    Classify free DOFs as master (non-zero mass diagonal) or slave (zero mass).
    
    Parameters
    ----------
    M_free : np.ndarray
        Free-DOF submatrix of global mass matrix, shape (n_free, n_free)
    beam_model : dict
        Beam model dict with 'nodes' and 'elements' keys
    free_dofs : np.ndarray
        Global DOF indices that are free (unconstrained)
    mass_tol : float, optional
        Tolerance for considering a diagonal mass entry as non-zero
    
    Returns
    -------
    local_master : np.ndarray
        Local indices (w.r.t. free_dofs) of master DOFs
    local_slave : np.ndarray
        Local indices (w.r.t. free_dofs) of slave DOFs
    global_master : np.ndarray
        Global DOF indices of master DOFs
    global_slave : np.ndarray
        Global DOF indices of slave DOFs
    master_nodes : list
        Node indices (in beam_model) that are masters
    slave_nodes : list
        Node indices (in beam_model) that are slaves
    """
    mass_diag = np.diag(M_free)
    
    # Classify local indices
    local_master = np.where(mass_diag > mass_tol)[0]
    local_slave = np.where(mass_diag <= mass_tol)[0]
    
    # Convert to global DOF indices
    global_master = free_dofs[local_master]
    global_slave = free_dofs[local_slave]
    
    # Extract node indices from DOF indices
    # Each node has 6 DOFs, so node index = DOF_index // 6
    master_nodes = sorted(list(set(global_master // 6)))
    slave_nodes = sorted(list(set(global_slave // 6)))
    
    return (local_master, local_slave, global_master, global_slave, 
            master_nodes, slave_nodes)


def guyan_condense(K_global, M_global, beam_model, constrained_dofs, mass_tol=1e-12):
    """
    Statically condense out slave (massless) DOFs using Guyan reduction.
    
    The condensation is performed on the free DOF submatrix (K_ff, M_ff),
    partitioned into master and slave portions:
    
      K_ff = | K_mm  K_ms |      M_ff = | M_mm   0  |
             | K_sm  K_ss |            |   0    0  |
    
    The static relationship u_s = -K_ss^{-1} K_sm u_m yields the expansion matrix.
    
    Parameters
    ----------
    K_global : np.ndarray
        Global stiffness matrix, shape (n_dof, n_dof)
    M_global : np.ndarray
        Global mass matrix, shape (n_dof, n_dof)
    beam_model : dict
        Beam model dict with 'nodes' and 'elements' keys
    constrained_dofs : list or np.ndarray
        Global DOF indices that are constrained (boundary conditions)
    mass_tol : float, optional
        Tolerance for classifying DOFs as massless (default: 1e-12)
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'K_star': Condensed stiffness matrix (n_master, n_master)
        - 'M_star': Condensed mass matrix (n_master, n_master)
        - 'local_master': Local indices of master DOFs (w.r.t. free_dofs)
        - 'local_slave': Local indices of slave DOFs (w.r.t. free_dofs)
        - 'global_master': Global indices of master DOFs
        - 'global_slave': Global indices of slave DOFs
        - 'master_nodes': Node indices that are masters
        - 'slave_nodes': Node indices that are slaves
        - 'T_slave': Expansion matrix, shape (n_slave, n_master)
        - 'free_dofs': All free DOF indices (unconstrained)
        - 'n_master': Number of master DOFs
        - 'n_slave': Number of slave DOFs
        - 'n_free': Total number of free DOFs
    """
    total_dof = K_global.shape[0]
    all_dofs = np.arange(total_dof)
    constrained_dofs_arr = np.asarray(constrained_dofs)
    free_dofs = np.setdiff1d(all_dofs, constrained_dofs_arr)
    
    # Extract free DOF submatrices
    K_ff = K_global[np.ix_(free_dofs, free_dofs)]
    M_ff = M_global[np.ix_(free_dofs, free_dofs)]
    
    # Classify master and slave DOFs
    (local_master, local_slave, global_master, global_slave,
     master_nodes, slave_nodes) = classify_dofs_by_mass(M_ff, beam_model, free_dofs, mass_tol)
    
    n_master = len(local_master)
    n_slave = len(local_slave)
    n_free = len(free_dofs)
    
    print(f"\n{'='*70}")
    print(f"GUYAN STATIC REDUCTION (Component Mode Synthesis)")
    print(f"{'='*70}")
    print(f"Total DOFs: {total_dof} (constrained: {len(constrained_dofs_arr)}, free: {n_free})")
    print(f"Master DOFs: {n_master}")
    print(f"  Global indices: {global_master[:10]}{'...' if n_master > 10 else ''}")
    print(f"  Master nodes: {master_nodes}")
    print(f"Slave DOFs: {n_slave}")
    print(f"  Global indices: {global_slave[:10]}{'...' if n_slave > 10 else ''}")
    print(f"  Slave nodes: {slave_nodes}")
    print(f"Reduction ratio: {n_master} / {n_free} = {100*n_master/n_free:.1f}%")
    
    # ========================================================================
    # CHECK: If no slave DOFs, return None (no condensation needed)
    # ========================================================================
    if n_slave == 0:
        print(f"\n⚠ No massless (slave) DOFs found - all DOFs have inertia.")
        print(f"  Guyan reduction not applicable. Returning None.")
        print(f"  (This is normal if foil nodes have added mass)")
        return None
    
    # Partition the free DOF matrices
    K_mm = K_ff[np.ix_(local_master, local_master)]
    K_ms = K_ff[np.ix_(local_master, local_slave)]
    K_sm = K_ff[np.ix_(local_slave, local_master)]
    K_ss = K_ff[np.ix_(local_slave, local_slave)]
    M_mm = M_ff[np.ix_(local_master, local_master)]
    
    # Compute expansion matrix: T_slave = -K_ss^{-1} K_sm
    # This represents the static relationship: u_s = T_slave @ u_m
    print(f"\nCondensing slave DOFs...")
    print(f"  Computing K_ss^{{{-1}}} (sparse solve, {K_ss.shape[0]}×{K_ss.shape[0]} matrix)")
    
    try:
        # Use sparse solver for efficiency (even if matrices are dense, this can be numerically stable)
        K_ss_sparse = csc_matrix(K_ss)
        
        # Solve K_ss @ X = K_sm for X
        # This gives X = K_ss^{-1} @ K_sm, hence T_slave = -X = -K_ss^{-1} K_sm
        # K_sm has shape (n_slave, n_master), so X will have shape (n_slave, n_master)
        X = spsolve(K_ss_sparse, csr_matrix(K_sm))
        
        # Ensure output is dense array (spsolve can return sparse or array)
        if hasattr(X, 'toarray'):
            X = X.toarray()
        
        T_slave = -X  # Shape: (n_slave, n_master)
        
        # Check for ill-conditioning
        cond_K_ss = np.linalg.cond(K_ss)
        print(f"  K_ss condition number: {cond_K_ss:.2e}")
        if cond_K_ss > 1e10:
            print(f"  ⚠ WARNING: K_ss is ill-conditioned; results may be inaccurate")
    
    except Exception as e:
        print(f"ERROR during condensation: {e}")
        raise
    
    # Compute condensed stiffness and mass
    K_star = K_mm + K_ms @ T_slave  # Exact static condensation
    M_star = M_mm                     # Slave mass is zero
    
    print(f"  K_star shape: {K_star.shape}")
    print(f"  M_star shape: {M_star.shape}")
    print(f"Condensation complete.\n")
    
    return {
        'K_star': K_star,
        'M_star': M_star,
        'local_master': local_master,
        'local_slave': local_slave,
        'global_master': global_master,
        'global_slave': global_slave,
        'master_nodes': master_nodes,
        'slave_nodes': slave_nodes,
        'T_slave': T_slave,
        'free_dofs': free_dofs,
        'n_master': n_master,
        'n_slave': n_slave,
        'n_free': n_free,
    }


def expand_modes(phi_m, reduction_result, total_dof, constrained_dofs):
    """
    Expand master-only mode shapes back to the full DOF space.
    
    Parameters
    ----------
    phi_m : np.ndarray
        Master-only mode shapes, shape (n_master, n_modes)
    reduction_result : dict
        Output from guyan_condense()
    total_dof : int
        Total degrees of freedom in the full system
    constrained_dofs : list or np.ndarray
        Global DOF indices that are constrained
    
    Returns
    -------
    phi_full : np.ndarray
        Full-DOF mode shapes, shape (total_dof, n_modes), with:
        - Constrained DOFs = 0
        - Free master DOFs = phi_m
        - Free slave DOFs = T_slave @ phi_m
    """
    T_slave = reduction_result['T_slave']
    local_master = reduction_result['local_master']
    local_slave = reduction_result['local_slave']
    free_dofs = reduction_result['free_dofs']
    
    n_free = len(free_dofs)
    n_modes = phi_m.shape[1]
    
    # Expand to free DOF space
    phi_free = np.zeros((n_free, n_modes))
    phi_free[local_master, :] = phi_m
    phi_free[local_slave, :] = T_slave @ phi_m  # Kinematic constraint
    
    # Expand to full DOF space (including constrained DOFs = 0)
    phi_full = np.zeros((total_dof, n_modes))
    phi_full[free_dofs, :] = phi_free
    
    return phi_full


def expand_eigenvectors(dry_eigenvectors, reduction_result, total_dof, constrained_dofs):
    """
    Batch expand all eigenvectors from reduced space to full space.
    
    Parameters
    ----------
    dry_eigenvectors : np.ndarray
        Eigenvectors in reduced (master-only) space, shape (n_master, n_modes)
    reduction_result : dict
        Output from guyan_condense()
    total_dof : int
        Total DOFs in the full system
    constrained_dofs : list or np.ndarray
        Constrained DOF indices
    
    Returns
    -------
    dry_eigenvectors_full : np.ndarray
        Eigenvectors in full space, shape (total_dof, n_modes)
    """
    return expand_modes(dry_eigenvectors, reduction_result, total_dof, constrained_dofs)


def create_expansion_matrix(reduction_result, total_dof, constrained_dofs):
    """
    Create the full expansion transformation matrix T_full.
    
    Maps master DOF vectors to full DOF vectors:
        u_full = T_full @ u_master
    
    Parameters
    ----------
    reduction_result : dict
        Output from guyan_condense()
    total_dof : int
        Total DOFs in the full system
    constrained_dofs : list or np.ndarray
        Constrained DOF indices
    
    Returns
    -------
    T_full : np.ndarray
        Expansion matrix, shape (total_dof, n_master)
    """
    T_slave = reduction_result['T_slave']
    local_master = reduction_result['local_master']
    local_slave = reduction_result['local_slave']
    free_dofs = reduction_result['free_dofs']
    n_master = reduction_result['n_master']
    
    # Build T_full in the free DOF space first
    T_free = np.zeros((len(free_dofs), n_master))
    T_free[local_master, :] = np.eye(n_master)
    T_free[local_slave, :] = T_slave
    
    # Expand to full DOF space
    T_full = np.zeros((total_dof, n_master))
    T_full[free_dofs, :] = T_free
    
    return T_full


def print_reduction_info(reduction_result, beam_model):
    """
    Print detailed information about the reduction (master/slave node separation).
    
    Parameters
    ----------
    reduction_result : dict
        Output from guyan_condense()
    beam_model : dict
        Beam model dictionary
    """
    master_nodes = reduction_result['master_nodes']
    slave_nodes = reduction_result['slave_nodes']
    
    print("\nReduction Node Classification:")
    print("  MASTER NODES (have inertia):")
    for node_idx in master_nodes:
        node = beam_model['nodes'][node_idx]
        print(f"    Node {node_idx}: {node.get('position', 'N/A')}")
    
    print("\n  SLAVE NODES (massless, condensed out):")
    for node_idx in slave_nodes:
        node = beam_model['nodes'][node_idx]
        print(f"    Node {node_idx}: {node.get('position', 'N/A')}")
