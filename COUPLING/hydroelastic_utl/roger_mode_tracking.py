"""
Modal Assurance Criterion (MAC) based mode tracking for Roger's RFA method.

This module implements MAC-driven mode identification and tracking across
velocity steps when using Roger's Rational Function Approximation for flutter
analysis. The methodology mirrors the approach used in the P-K method
(pk_solverv3) but adapted for the augmented state-space formulation of RFA.

Key concepts:
- Extract structural modes from augmented eigenvalue problem (discard lag states)
- Compute MAC between consecutive velocity steps for mode continuity
- Implement mode matching with frequency and MAC weighting
- Provide diagnostic tools for mode tracking quality assessment
"""

import numpy as np
from numpy.linalg import norm


def _mac_mode_shapes(u1, u2, n_structural_dof=None):
    """Optional slice to structural [q; q_dot] block before MAC (Roger augmented states)."""
    u1 = np.asarray(u1, dtype=complex).ravel()
    u2 = np.asarray(u2, dtype=complex).ravel()
    if n_structural_dof is not None and n_structural_dof > 0:
        n = int(n_structural_dof)
        if u1.size >= n:
            u1 = u1[:n]
        if u2.size >= n:
            u2 = u2[:n]
    return u1, u2


def compute_mac(u1, u2, n_structural_dof=None):
    """
    Modal Assurance Criterion (MAC) between two complex vectors.
    
    MAC quantifies the consistency of two mode shapes and ranges from 0 (no
    correlation) to 1 (perfect correlation). It is invariant to scaling and
    phase differences.
    
    Parameters
    ----------
    u1, u2 : (n,) complex ndarray
        Two mode shape vectors (eigenvectors)
    
    Returns
    -------
    mac : float in [0, 1]
        Modal Assurance Criterion value
    
    Notes
    -----
    MAC = |u1^H @ u2|^2 / (u1^H @ u1 * u2^H @ u2)
    
    where ^H denotes conjugate transpose.
    """
    if u1 is None or u2 is None:
        return 0.0

    u1, u2 = _mac_mode_shapes(u1, u2, n_structural_dof)
    numerator = np.abs(u1.conj().T @ u2) ** 2
    denominator = (u1.conj().T @ u1) * (u2.conj().T @ u2)
    
    if np.abs(denominator) < 1e-12:
        return 0.0
    
    mac_value = np.real(numerator / denominator)
    
    # Clamp to [0, 1] range to handle numerical errors
    mac_value = np.clip(mac_value, 0.0, 1.0)
    
    return mac_value


def extract_structural_modes_from_augmented(eigenvalues, eigenvectors, n_modes, omega_n,
                                             threshold_factor=0.1, candidates_multiplier=2,
                                             vecs_previous=None, freqs_previous=None):
    """
    Extract structural modes from the augmented RFA state-space problem.
    
    The augmented state-space includes:
    - n_modes structural modes (complex conjugate pairs for oscillatory motion)
    - N lag state modes (typically real, located at -V*blag)
    
    Strategy:
    1. Filters oscillatory modes (keep imaginary part > threshold)
    2. Keeps positive imaginary parts only (one of each conjugate pair)
    3. Sorts by frequency
    4. Extracts MORE candidates (e.g., 2*n_modes) to handle mode appearance
    5. If previous modes available, selects best n_modes using MAC matching
    6. Otherwise selects first n_modes by frequency
    
    Parameters
    ----------
    eigenvalues : (n_total,) complex ndarray
        All eigenvalues from augmented A_aug matrix
    eigenvectors : (n_total, n_total) complex ndarray
        All eigenvectors from augmented A_aug matrix
    n_modes : int
        Number of structural modes to extract
    omega_n : (n_modes,) float ndarray
        Structural natural frequencies (rad/s)
    threshold_factor : float, optional
        Oscillatory mode threshold as fraction of lowest structural frequency
        Default: 0.1 (discard modes with frequency < 10% of omega_n[0])
    candidates_multiplier : int, optional
        Multiplier for number of mode candidates to extract (e.g., 2 = search for 2*n_modes)
        Default: 2
    vecs_previous, freqs_previous : ndarray, optional
        Previous velocity eigenvectors and frequencies for MAC-based selection
        If provided, best modes selected by MAC; otherwise by frequency order
    
    Returns
    -------
    eigs_structural : (n_modes,) complex ndarray
        Selected eigenvalues corresponding to structural modes
    vecs_structural : (n_total, n_modes) complex ndarray
        Corresponding eigenvectors
    indices_selected : (n_modes,) int ndarray
        Indices into the full eigenvalue/eigenvector arrays
    frequencies : (n_modes,) float ndarray
        Frequencies (rad/s) of selected modes
    damping : (n_modes,) float ndarray
        Damping ratios of selected modes (sigma/|omega|)
    """
    # Extract imaginary and real parts
    imag_parts = np.imag(eigenvalues)
    real_parts = np.real(eigenvalues)
    
    # Step 1: Filter oscillatory modes
    # Discard near-zero eigenvalues (lag poles typically sit at large negative real values)
    threshold = threshold_factor * omega_n[0] if len(omega_n) > 0 else 1.0
    oscillatory_mask = np.abs(imag_parts) > threshold
    
    # Step 2: Keep positive imaginary parts only (one per conjugate pair)
    pos_imag_mask = imag_parts > 0
    
    combined_mask = oscillatory_mask & pos_imag_mask
    indices_osc = np.where(combined_mask)[0]
    
    if len(indices_osc) == 0:
        # Fallback: all modes lost or no oscillatory modes
        # Return zero-frequency dummy modes
        eigs_dummy = np.array([(-0.01 * w + 1j * w) for w in omega_n])
        vecs_dummy = np.zeros((len(eigenvalues), n_modes), dtype=complex)
        indices_dummy = np.arange(n_modes)
        freqs_dummy = omega_n.copy()
        damp_dummy = np.full(n_modes, -0.01)
        
        return eigs_dummy, vecs_dummy, indices_dummy, freqs_dummy, damp_dummy
    
    # Step 3: Sort by frequency
    freqs_osc = np.abs(imag_parts[indices_osc])
    sort_order = np.argsort(freqs_osc)
    indices_osc_sorted = indices_osc[sort_order]
    
    # Step 4: Extract candidate modes (2*n_modes or all available)
    n_candidates = min(len(indices_osc_sorted), n_modes * candidates_multiplier)
    indices_candidates = indices_osc_sorted[:n_candidates]
    
    # Step 5: Select final n_modes using MAC matching or frequency order
    if vecs_previous is not None and freqs_previous is not None:
        # MAC-based selection: choose best n_modes by MAC with previous step
        indices_selected = _select_modes_by_mac(
            eigenvalues[indices_candidates],
            eigenvectors[:, indices_candidates],
            np.abs(imag_parts[indices_candidates]),
            eigenvalues[indices_osc_sorted[:len(indices_osc_sorted)]],
            eigenvectors[:, indices_osc_sorted[:len(indices_osc_sorted)]],
            freqs_previous,
            vecs_previous,
            n_modes
        )
    else:
        # Frequency-based selection: just take first n_modes from candidates
        if len(indices_candidates) >= n_modes:
            indices_selected = indices_candidates[:n_modes]
        else:
            # Pad if necessary
            indices_selected = indices_candidates.copy()
            n_missing = n_modes - len(indices_candidates)
            
            used_set = set(indices_selected)
            all_indices = np.arange(len(eigenvalues))
            unused_indices = np.array([i for i in all_indices if i not in used_set])
            
            if len(unused_indices) > 0:
                unused_freqs = np.abs(imag_parts[unused_indices])
                for j in range(n_missing):
                    struct_freq = omega_n[len(indices_selected) + j]
                    freq_dists = np.abs(unused_freqs - struct_freq)
                    best_match_idx = np.argmin(freq_dists)
                    indices_selected = np.append(indices_selected, unused_indices[best_match_idx])
                    unused_indices = np.delete(unused_indices, best_match_idx)
                    unused_freqs = np.delete(unused_freqs, best_match_idx)
    
    eigs_structural = eigenvalues[indices_selected]
    vecs_structural = eigenvectors[:, indices_selected]
    frequencies = np.abs(np.imag(eigs_structural))
    damping = np.real(eigs_structural) / np.maximum(np.abs(np.imag(eigs_structural)), 1e-12)
    
    return eigs_structural, vecs_structural, indices_selected, frequencies, damping


def is_dummy_eigenpair(eigenvalue, eigenvector, omega_n, zeta_tol=0.005, vec_tol=1e-10):
    """True for padded Roger roots (zero vector, zeta ~ -0.01, f ~ omega_n)."""
    omega_n = np.asarray(omega_n, dtype=float).ravel()
    freq = np.abs(np.imag(eigenvalue))
    zeta = np.real(eigenvalue) / max(freq, 1e-12)
    if norm(np.asarray(eigenvector, dtype=complex).ravel()) > vec_tol:
        return False
    if abs(zeta + 0.01) > zeta_tol:
        return False
    return np.any(np.abs(freq - omega_n) / np.maximum(omega_n, 1e-12) < 0.02)


def should_mark_mode_lost(freq_rad, eigenvalue, eigenvector, omega_n, f_min_hz=2.0):
    """Branch is lost if frequency is below threshold or eigenpair is a dummy placeholder."""
    f_min_rad = float(f_min_hz) * 2.0 * np.pi
    if not np.isfinite(freq_rad) or float(freq_rad) < f_min_rad:
        return True
    return is_dummy_eigenpair(eigenvalue, eigenvector, omega_n)


def _select_modes_by_mac(eigs_candidates, vecs_candidates, freqs_candidates,
                         eigs_all, vecs_all, freqs_previous, vecs_previous, n_modes,
                         n_structural_dof=None):
    """
    Select best n_modes from candidates using MAC matching with previous step.
    
    For each candidate mode, computes MAC with all previous modes and selects
    the n_modes with best overall MAC scores.
    
    Parameters
    ----------
    eigs_candidates : (n_cand,) complex ndarray
        Candidate eigenvalues
    vecs_candidates : (n_dof, n_cand) complex ndarray
        Candidate eigenvectors
    freqs_candidates : (n_cand,) float ndarray
        Candidate frequencies
    eigs_all, vecs_all : complex ndarray
        All oscillatory modes for fallback
    freqs_previous : (n_prev,) float ndarray
        Previous velocity frequencies
    vecs_previous : (n_dof, n_prev) complex ndarray
        Previous velocity eigenvectors
    n_modes : int
        Number of modes to select
    
    Returns
    -------
    indices_selected : (n_modes,) int ndarray
        Indices into eigs_all/vecs_all arrays for selected modes
    """
    n_cand = len(freqs_candidates)
    n_prev = len(freqs_previous)
    
    # Compute MAC matrix: (candidates x previous)
    mac_matrix_cand = np.zeros((n_cand, n_prev))
    for i_cand in range(n_cand):
        for j_prev in range(n_prev):
            u_cand = vecs_candidates[:, i_cand]
            u_prev = vecs_previous[:, j_prev]
            mac_matrix_cand[i_cand, j_prev] = compute_mac(
                u_cand, u_prev, n_structural_dof=n_structural_dof
            )
    
    # For each candidate, get best MAC with any previous mode
    best_mac_per_candidate = np.max(mac_matrix_cand, axis=1)
    
    # Select n_modes with highest best_mac scores
    selected_indices_in_candidates = np.argsort(-best_mac_per_candidate)[:n_modes]
    
    # Map back to full eigs_all/vecs_all indices
    # Get frequencies from all eigenvalues
    all_freqs = np.abs(np.imag(eigs_all))
    
    # Simple matching: find closest frequency in full list for each candidate
    indices_selected = []
    for idx_in_cand in selected_indices_in_candidates:
        freq_target = freqs_candidates[idx_in_cand]
        freq_dists = np.abs(all_freqs - freq_target)
        best_match_idx = np.argmin(freq_dists)
        indices_selected.append(best_match_idx)
    
    return np.array(indices_selected)


def match_modes_between_velocities(eigs_current, vecs_current, freqs_current,
                                    eigs_previous, vecs_previous, freqs_previous,
                                    freq_margin=0.15, w_freq=0.8, w_mac=0.6,
                                    n_structural_dof=None, verbose=False):
    """
    Match structural modes between consecutive velocity steps using MAC and frequency.
    
    Solves an assignment problem to find the optimal one-to-one correspondence between
    modes at V_i and V_{i-1} based on combined MAC and frequency proximity scores.
    
    Parameters
    ----------
    eigs_current, eigs_previous : (n_modes,) complex ndarray
        Eigenvalues (structural modes only) at current and previous velocity
    vecs_current, vecs_previous : (n_structural, n_modes) complex ndarray
        Eigenvectors (structural modes only) at current and previous velocity
    freqs_current, freqs_previous : (n_modes,) float ndarray
        Frequencies at current and previous velocity
    freq_margin : float, optional
        Acceptable fractional frequency deviation for candidate pairing
        Default: 0.15 (±15%)
    w_freq, w_mac : float, optional
        Weights for frequency and MAC in combined score: score = w_freq*f_err + w_mac*(1-MAC)
        Default: w_freq=0.8, w_mac=0.6 (frequency slightly weighted)
    verbose : bool, optional
        Print matching details
    
    Returns
    -------
    reordering_indices : (n_modes,) int ndarray
        Indices that reorder current modes to match previous step order.
        result_reordered[i] = result_current[reordering_indices[i]]
    mac_matrix : (n_modes, n_modes) ndarray
        MAC values between all current and previous mode pairs [i, j]
    matching_info : list of dicts
        Detailed information for each matched pair
    """
    from scipy.optimize import linear_sum_assignment
    
    n_modes = len(eigs_current)
    
    # Build cost matrix (lower is better)
    cost_matrix = np.zeros((n_modes, n_modes))
    mac_matrix = np.zeros((n_modes, n_modes))
    
    for i in range(n_modes):  # current modes
        for j in range(n_modes):  # previous modes
            # MAC component
            # Defensive: handle 1D arrays gracefully
            try:
                u_curr = vecs_current[:, i] if vecs_current.ndim == 2 else vecs_current
                u_prev = vecs_previous[:, j] if vecs_previous.ndim == 2 else vecs_previous
            except (IndexError, TypeError) as e:
                print(f"[ERROR] Eigenvector indexing failed at (i={i}, j={j})")
                print(f"  vecs_current shape: {vecs_current.shape}, ndim: {vecs_current.ndim}")
                print(f"  vecs_previous shape: {vecs_previous.shape}, ndim: {vecs_previous.ndim}")
                raise
            
            mac_val = compute_mac(u_curr, u_prev, n_structural_dof=n_structural_dof)
            mac_matrix[i, j] = mac_val
            
            # Frequency component (normalized error)
            freq_error = np.abs(freqs_current[i] - freqs_previous[j]) / (freqs_previous[j] + 1e-12)
            
            # Combined cost: penalize low MAC and high frequency deviation
            cost = w_freq * freq_error + w_mac * (1.0 - mac_val)
            
            # Apply hard constraint: skip candidates with large frequency deviation
            if freq_error > freq_margin:
                cost = np.inf
            
            cost_matrix[i, j] = cost
    
    # Solve linear assignment problem (now handles rectangular matrices for variable n_modes)
    n_prev = len(freqs_previous)
    
    # For rectangular case, we need to handle it differently
    if n_modes != n_prev:
        # Rectangular case: use rectangular cost matrix matching
        cost_rect = np.full((n_modes, n_prev), np.inf)
        for i in range(n_modes):
            for j in range(n_prev):
                cost_rect[i, j] = cost_matrix[i, j] if j < n_modes else np.inf
        
        try:
            current_indices, prev_indices = linear_sum_assignment(cost_rect)
        except ValueError as e:
            if "infeasible" in str(e):
                print(f"  [WARNING] Cost matrix infeasible. Using greedy MAC-based matching...")
                current_indices, prev_indices = _greedy_mac_matching(mac_matrix[:, :n_prev], cost_rect, n_modes, n_prev)
            else:
                raise
    else:
        # Square case: standard linear assignment
        try:
            current_indices, prev_indices = linear_sum_assignment(cost_matrix)
        except ValueError as e:
            if "infeasible" in str(e):
                print(f"  [WARNING] Cost matrix infeasible. Using greedy MAC-based matching...")
                current_indices, prev_indices = _greedy_mac_matching(mac_matrix, cost_matrix, n_modes, n_modes)
            else:
                raise
    
    # Build reordering: for each previous mode, find its best match in current
    reordering_indices = np.full(n_prev, -1, dtype=int)
    for i_curr, i_prev in zip(current_indices, prev_indices):
        cost_val = cost_matrix[i_curr, i_prev] if i_prev < n_modes else np.inf
        if cost_val < np.inf:
            reordering_indices[i_prev] = i_curr
    
    # Gather matching information for diagnostics
    matching_info = []
    for i_prev in range(n_prev):
        i_curr = reordering_indices[i_prev]
        
        if i_curr >= 0:  # Mode was matched
            mac_val = mac_matrix[i_curr, i_prev]
            cost_val = cost_matrix[i_curr, i_prev]
            freq_prev = freqs_previous[i_prev]
            freq_curr = freqs_current[i_curr]
            freq_err = np.abs(freq_curr - freq_prev) / (freq_prev + 1e-12)
            
            matching_info.append({
                'mode_index': i_prev,
                'index_in_current': i_curr,
                'mac': mac_val,
                'cost': cost_val,
                'freq_prev': freq_prev,
                'freq_curr': freq_curr,
                'freq_error': freq_err,
                'valid': cost_val < np.inf,
                'status': 'matched'
            })
            
            if verbose:
                print(f"  ✓ Mode {i_prev}: MAC={mac_val:.3f}, freq_err={freq_err:.3e}, "
                      f"f_prev={freq_prev:.3f}→f_curr={freq_curr:.3f} rad/s")
        else:
            # Mode was not matched (lost)
            matching_info.append({
                'mode_index': i_prev,
                'index_in_current': -1,
                'mac': 0.0,
                'cost': np.inf,
                'freq_prev': freqs_previous[i_prev],
                'freq_curr': None,
                'freq_error': np.inf,
                'valid': False,
                'status': 'unmatched'
            })
            
            if verbose:
                print(f"  ✗ Mode {i_prev}: UNMATCHED (freq={freqs_previous[i_prev]:.3f} rad/s)")
    
    # Report new modes (current modes not matched to any previous mode)
    matched_current = set(current_indices[current_indices >= 0])
    new_modes = [i for i in range(n_modes) if i not in matched_current]
    
    if new_modes and verbose:
        print(f"  [INFO] {len(new_modes)} new mode(s) detected at indices: {new_modes}")
        for i_new in new_modes:
            best_mac = np.max(mac_matrix[i_new, :n_prev])
            print(f"    → Mode {i_new}: f={freqs_current[i_new]:.3f} rad/s, best_MAC={best_mac:.3f}")
    
    return reordering_indices, mac_matrix, matching_info


def _greedy_mac_matching(mac_matrix, cost_matrix, n_curr, n_prev):
    """
    Fallback greedy matching when linear assignment fails (infeasible cost matrix).
    
    Matches modes based on highest MAC values, handling unequal matrix dimensions.
    
    Parameters
    ----------
    mac_matrix : (n_curr, n_prev) ndarray
        MAC values for all current vs previous pairs
    cost_matrix : (n_curr, n_prev) ndarray
        Cost matrix (may contain inf values)
    n_curr, n_prev : int
        Number of current and previous modes
    
    Returns
    -------
    current_indices : ndarray
        Matched current mode indices
    prev_indices : ndarray
        Matched previous mode indices
    """
    current_indices = []
    prev_indices = []
    
    used_prev = set()
    used_curr = set()
    
    # Sort all (i_curr, j_prev) pairs by MAC value (descending)
    pairs = []
    for i_curr in range(n_curr):
        for j_prev in range(n_prev):
            if cost_matrix[i_curr, j_prev] < np.inf:  # Only consider valid costs
                pairs.append((mac_matrix[i_curr, j_prev], i_curr, j_prev))
    
    # Sort by MAC (descending)
    pairs.sort(reverse=True, key=lambda x: x[0])
    
    # Greedily match pairs with highest MAC
    for mac_val, i_curr, j_prev in pairs:
        if i_curr not in used_curr and j_prev not in used_prev:
            current_indices.append(i_curr)
            prev_indices.append(j_prev)
            used_curr.add(i_curr)
            used_prev.add(j_prev)
    
    return np.array(current_indices), np.array(prev_indices)


def reorder_and_normalize_modes(eigenvalues, eigenvectors, frequencies, damping,
                                 reordering_indices):
    """
    Reorder modes according to matching and ensure consistent normalization.
    
    Parameters
    ----------
    eigenvalues, eigenvectors : complex ndarray
        Mode data at current velocity
    frequencies, damping : float ndarray
        Frequency and damping data
    reordering_indices : (n_modes,) int ndarray
        Indices specifying new order
    
    Returns
    -------
    damping_reordered : float ndarray
        Reordered damping ratios (n_modes,)
    frequencies_reordered : float ndarray
        Reordered frequencies (n_modes,)
    eigenvectors_reordered : complex ndarray
        Reordered eigenvectors (n_dof, n_modes) with consistent normalization
    eigenvalues_reordered : complex ndarray
        Reordered eigenvalues (n_modes,)
    """
    # Validate input shapes
    if eigenvectors.ndim != 2:
        raise ValueError(f"eigenvectors must be 2D, got shape {eigenvectors.shape}")
    
    n_dof = eigenvectors.shape[0]
    n_modes = len(reordering_indices)
    
    eigs_reordered = eigenvalues[reordering_indices]
    vecs_reordered = eigenvectors[:, reordering_indices]
    freqs_reordered = frequencies[reordering_indices]
    damp_reordered = damping[reordering_indices]
    
    # Ensure vecs_reordered stays 2D after reordering
    if vecs_reordered.ndim != 2:
        print(f"[WARNING] vecs_reordered became {vecs_reordered.ndim}D after reordering, reshaping...")
        vecs_reordered = vecs_reordered.reshape(n_dof, n_modes)
    
    # Normalize eigenvectors to unit L2 norm and fix sign for consistency
    for i in range(vecs_reordered.shape[1]):
        vec = vecs_reordered[:, i]
        norm_val = norm(vec)
        if norm_val > 1e-12:
            vecs_reordered[:, i] = vec / norm_val
            
            # Fix sign: make largest magnitude component positive real
            largest_idx = np.argmax(np.abs(vec))
            if np.real(vec[largest_idx]) < 0:
                vecs_reordered[:, i] = -vecs_reordered[:, i]
    
    # Final validation
    if vecs_reordered.ndim != 2:
        raise ValueError(f"vecs_reordered is {vecs_reordered.ndim}D after normalization, expected 2D with shape ({n_dof}, {n_modes})")
    if vecs_reordered.shape != (n_dof, n_modes):
        raise ValueError(f"vecs_reordered has shape {vecs_reordered.shape}, expected ({n_dof}, {n_modes})")
    
    # Return in order: damping, frequencies, eigenvectors, eigenvalues
    return damp_reordered, freqs_reordered, vecs_reordered, eigs_reordered


def compute_mac_matrix(vecs_set1, vecs_set2):
    """
    Compute full MAC matrix between two sets of mode shape vectors.
    
    Parameters
    ----------
    vecs_set1 : (n_dof, n_modes1) complex ndarray
        First set of eigenvectors
    vecs_set2 : (n_dof, n_modes2) complex ndarray
        Second set of eigenvectors
    
    Returns
    -------
    mac_matrix : (n_modes1, n_modes2) ndarray
        MAC[i, j] = MAC between mode i of set1 and mode j of set2
    """
    n_modes1 = vecs_set1.shape[1]
    n_modes2 = vecs_set2.shape[1]
    
    mac_matrix = np.zeros((n_modes1, n_modes2))
    
    for i in range(n_modes1):
        for j in range(n_modes2):
            mac_matrix[i, j] = compute_mac(vecs_set1[:, i], vecs_set2[:, j])
    
    return mac_matrix


def diagnostic_mac_summary(mac_history, freqs_history, V_list, verbose=True):
    """
    Generate diagnostic summary of MAC tracking quality across velocity sweep.
    
    Parameters
    ----------
    mac_history : list of (n_modes, n_modes) ndarray
        MAC matrices at each velocity step (starting from V_2, since V_1 has no prev step)
    freqs_history : list of (n_modes,) ndarray
        Frequencies at each velocity step
    V_list : array-like
        Velocity values
    verbose : bool
        Print summary statistics
    
    Returns
    -------
    diagnostics : dict
        Summary statistics including:
        - 'mean_mac_per_mode': mean MAC diagonal for each mode across all V
        - 'min_mac_per_mode': minimum diagonal MAC for each mode
        - 'mean_freq_continuity': mean frequency continuity across velocities
        - 'freq_jumps': detected frequency discontinuities
    """
    diagnostics = {}
    n_modes = freqs_history[0].shape[0]
    n_velocities = len(freqs_history)
    
    # MAC diagonal statistics (ideally close to 1.0)
    if len(mac_history) > 0:
        mac_diagonals = []
        for mac_mat in mac_history:
            mac_diagonals.append(np.diag(mac_mat))
        
        mac_diagonals = np.array(mac_diagonals)  # (n_velocities-1, n_modes)
        mean_mac = np.mean(mac_diagonals, axis=0)
        min_mac = np.min(mac_diagonals, axis=0)
        
        diagnostics['mean_mac_per_mode'] = mean_mac
        diagnostics['min_mac_per_mode'] = min_mac
        
        if verbose:
            print("\nMAC Tracking Quality Summary:")
            print("=" * 70)
            for mode_idx in range(n_modes):
                print(f"  Mode {mode_idx}: mean_MAC={mean_mac[mode_idx]:.4f}, "
                      f"min_MAC={min_mac[mode_idx]:.4f}")
    
    # Frequency continuity
    freq_jumps = []
    freq_continuity = []
    
    for v_idx in range(1, n_velocities):
        for mode_idx in range(n_modes):
            f_prev = freqs_history[v_idx - 1][mode_idx]
            f_curr = freqs_history[v_idx][mode_idx]
            
            if f_prev > 1e-6:
                rel_change = abs(f_curr - f_prev) / f_prev
                freq_continuity.append(rel_change)
                
                if rel_change > 0.3:  # 30% jump
                    freq_jumps.append({
                        'velocity_pair': (V_list[v_idx-1], V_list[v_idx]),
                        'mode': mode_idx,
                        'freq_prev': f_prev,
                        'freq_curr': f_curr,
                        'rel_change': rel_change
                    })
    
    diagnostics['mean_freq_continuity'] = np.mean(freq_continuity) if len(freq_continuity) > 0 else np.nan
    diagnostics['freq_jumps'] = freq_jumps
    
    if verbose:
        print("\nFrequency Continuity Summary:")
        print("=" * 70)
        print(f"  Mean relative frequency change: {diagnostics['mean_freq_continuity']:.4e}")
        if len(freq_jumps) > 0:
            print(f"  ⚠ Detected {len(freq_jumps)} frequency jumps (> 30%):")
            for jump in freq_jumps[:5]:  # Show first 5
                print(f"    V={jump['velocity_pair'][0]:.2f}→{jump['velocity_pair'][1]:.2f} m/s, "
                      f"Mode {jump['mode']}: {jump['freq_prev']:.3f}→{jump['freq_curr']:.3f} rad/s "
                      f"(Δ={jump['rel_change']*100:.1f}%)")
        else:
            print(f"  ✓ No major frequency jumps detected")
    
    return diagnostics
