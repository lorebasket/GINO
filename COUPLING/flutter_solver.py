# FSI/COUPLING/flutter_solver.py

import numpy as np
from collections import namedtuple
from .hydroelastic_utl import pk_solverv3
from .hydroelastic_utl import vgvf_plotting
from .hydroelastic_utl import roger_mode_tracking
from .hydroelastic_utl import post_processing
from .hydroelastic_utl import roger_fit
import pickle
import os

FlutterResults = namedtuple('FlutterResults', [
    'velocities', 'damping', 'frequencies', 'modes', 'flutter_speed', 'flutter_frequency', 'raw_results', 'pk_solver'
])


def _fit_roger_to_Q_modal(Q_modal_array, k_list, blag, Nm, N):
    """
    Fit Roger RFA coefficients to modal GAF data.

    Parameters
    ----------
    Q_modal_array : (Nk, Nm, Nm) complex ndarray
    k_list        : (Nk,) array of reduced frequencies (k = omega/V)
    blag          : (N,) array of lag pole values (positive reals, same units as k)
    Nm, N         : number of modes and lag poles

    Returns
    -------
    B0, B1, B2 : (Nm, Nm) real ndarray
    Blag       : list of N (Nm, Nm) real ndarray
    """
    Nk = len(k_list)

    rows = []
    for k in k_list:
        k2    = k ** 2
        re_row = [1.0, 0.0, -k2] + [k2 / (k2 + b**2) for b in blag]
        im_row = [0.0, k,   0.0] + [k * b / (k2 + b**2) for b in blag]
        rows.append(re_row)
        rows.append(im_row)
    A_basis = np.array(rows, dtype=float)   # (2*Nk, 3+N)

    B0   = np.zeros((Nm, Nm))
    B1   = np.zeros((Nm, Nm))
    B2   = np.zeros((Nm, Nm))
    Blag = [np.zeros((Nm, Nm)) for _ in range(N)]

    for i in range(Nm):
        for j in range(Nm):
            q_ij      = Q_modal_array[:, i, j]   # (Nk,) complex
            rhs       = np.empty(2 * Nk)
            rhs[0::2] = q_ij.real                # even indices <- re_row
            rhs[1::2] = q_ij.imag                # odd indices  <- im_row
            coeffs, *_ = np.linalg.lstsq(A_basis, rhs, rcond=None)
            B0[i, j]  = coeffs[0]
            B1[i, j]  = coeffs[1]
            B2[i, j]  = coeffs[2]
            for l in range(N):
                Blag[l][i, j] = coeffs[3 + l]

    return B0, B1, B2, Blag


def _reconstruct_Q_modal(k, B0, B1, B2, Blag, blag):
    """Evaluate Q_modal(ik) from Roger coefficients."""
    k2 = k ** 2
    Q  = B0 - k2 * B2 + 1j * k * B1
    for Bl, b in zip(Blag, blag):
        Q += Bl * (k2 + 1j * k * b) / (k2 + b**2)
    return Q


def _build_A_aug(config, V, rho, M_hat, C_hat, K_hat, B0, B1, B2, Blag, blag, Nm, N):
    """
    Assemble the augmented aeroelastic state matrix.

    State vector:  X = [q(Nm), q_dot(Nm), xa_1(Nm), ..., xa_N(Nm)]
    Size:          (2 + N) * Nm

    EOM:
        M_eff . q_ddot = -K_eff . q  -  C_eff . q_dot  +  sum_i q_dyn*Blag[i]*xa_i
        xa_i_dot       = q_dot  -  V*blag[i]*xa_i

    M_eff = M_hat - 0.5*rho*B2
    C_eff = C_hat - 0.5*rho*V*B1
    K_eff = K_hat - 0.5*rho*V^2*B0
    q_dyn = 0.5*rho*V^2
    """
    n     = Nm
    q_dyn = 0.5 * rho * V ** 2
    n_tot = (2 + N) * n

    if config.added_mass_strip_theory:
        M_eff = M_hat 
    else:
        M_eff = M_hat- 0.5 * rho        * B2
    
    C_eff = C_hat - 0.5 * rho * V    * B1
    K_eff = K_hat - 0.5 * rho * V**2 * B0

    M_eff_inv = np.linalg.inv(M_eff)

    A_aug = np.zeros((n_tot, n_tot), dtype=float)

    # [0, 1]: identity
    A_aug[:n, n:2*n] = np.eye(n)

    # [1, 0] and [1, 1]: structural dynamics
    A_aug[n:2*n, :n]    = -M_eff_inv @ K_eff
    A_aug[n:2*n, n:2*n] = -M_eff_inv @ C_eff

    # [1, 2+i]: lag force on structure (positive -- no minus sign)
    for i in range(N):
        c0 = (2 + i) * n
        A_aug[n:2*n, c0:c0+n] = M_eff_inv @ (q_dyn * Blag[i])

    # [2+i, 1]: q_dot drives lag state (velocity block, NOT displacement)
    # [2+i, 2+i]: lag pole -V * blag[i]
    for i in range(N):
        r0 = (2 + i) * n
        A_aug[r0:r0+n, n:2*n]   = np.eye(n)                  # from q_dot
        A_aug[r0:r0+n, r0:r0+n] = -V * blag[i] * np.eye(n)  # pole

    return A_aug


def _extract_flutter_modes(A_aug, Nm, omega_n, return_eigenvectors=False,
                           vecs_previous=None, freqs_previous=None):
    """
    Extract the Nm structural modes from the augmented eigenvalue problem.
    
    This function separates structural modes (oscillatory) from lag state modes
    (typically real poles) and returns their frequencies and damping ratios.
    
    If previous eigenvectors and frequencies are provided, uses MAC-based selection
    to find the best Nm modes among 2*Nm candidates. Otherwise uses frequency-based
    selection (first Nm by ascending frequency).
    
    Optionally returns eigenvectors for MAC-based mode tracking.

    Parameters
    ----------
    A_aug : (n_aug, n_aug) ndarray
        Augmented aeroelastic state matrix from Roger RFA
    Nm : int
        Number of structural modes to extract
    omega_n : (Nm,) ndarray
        Structural natural frequencies (for reference)
    return_eigenvectors : bool, optional
        If True, return eigenvectors for MAC computations
        Default: False
    vecs_previous : (n_aug, Nm) complex ndarray, optional
        Eigenvectors from previous velocity step (for MAC-based selection)
    freqs_previous : (Nm,) float ndarray, optional
        Frequencies from previous velocity step (for MAC-based selection)

    Returns
    -------
    g_V : (Nm,) float ndarray
        Damping ratios, g = sigma/|omega|
    freq_V : (Nm,) float ndarray
        Frequencies in rad/s
    vecs_V : (n_aug, Nm) complex ndarray, optional
        Eigenvectors of selected modes (only if return_eigenvectors=True)
    eigs_V : (Nm,) complex ndarray, optional
        Selected eigenvalues (only if return_eigenvectors=True)
    """
    # Compute eigendecomposition (eigenvectors needed for MAC)
    eigenvalues, eigenvectors = np.linalg.eig(A_aug)

    # Keep oscillatory modes only (discard real lag poles at ~ -V*blag)
    threshold = 0.1 * omega_n[0]
    osc_mask  = np.abs(np.imag(eigenvalues)) > threshold
    pos_mask  = np.imag(eigenvalues) > 0          # one of each conjugate pair
    
    eigs_filt = eigenvalues[osc_mask & pos_mask]
    vecs_filt = eigenvectors[:, osc_mask & pos_mask]
    
    # Ensure vecs_filt is always 2D
    if vecs_filt.ndim == 1:
        vecs_filt = vecs_filt.reshape(-1, 1)

    if len(eigs_filt) == 0:
        if return_eigenvectors:
            return np.zeros(Nm), np.zeros(Nm), np.zeros((len(eigenvalues), Nm)), np.zeros(Nm, dtype=complex)
        else:
            return np.zeros(Nm), np.zeros(Nm)

    # Sort by frequency
    order     = np.argsort(np.abs(np.imag(eigs_filt)))
    eigs_filt = eigs_filt[order]
    vecs_filt = vecs_filt[:, order]
    freqs_filt = np.abs(np.imag(eigs_filt))

    # Choose modes: MAC-based (if previous available) or frequency-based
    if vecs_previous is not None and freqs_previous is not None and len(eigs_filt) >= Nm:
        # MAC-based selection: search among 2*Nm candidates
        from panelaero_utl.pk_method_utl import roger_mode_tracking
        
        n_candidates = min(len(eigs_filt), 2 * Nm)
        indices_best = roger_mode_tracking._select_modes_by_mac(
            eigs_filt[:n_candidates],
            vecs_filt[:, :n_candidates],
            freqs_filt[:n_candidates],
            eigs_filt,
            vecs_filt,
            freqs_previous,
            vecs_previous,
            Nm
        )
        eigs_selected = eigs_filt[indices_best]
        vecs_selected = vecs_filt[:, indices_best]
    else:
        # Frequency-based selection: take first Nm sorted by frequency
        if len(eigs_filt) >= Nm:
            eigs_selected = eigs_filt[:Nm]
            vecs_selected = vecs_filt[:, :Nm]
        else:
            # Pad with dummy stable modes at structural frequencies
            dummy = np.array([-0.01 * w + 1j * w
                              for w in omega_n[len(eigs_filt):Nm]])
            eigs_selected = np.concatenate([eigs_filt, dummy])
            
            # Pad eigenvectors with zeros - ensure proper 2D shape
            n_missing = Nm - len(eigs_filt)
            n_dof = vecs_filt.shape[0]  # Number of DOFs
            vecs_dummy = np.zeros((n_dof, n_missing), dtype=complex)
            vecs_selected = np.column_stack([vecs_filt, vecs_dummy])  # Ensure 2D
    
    # Ensure vecs_selected is always 2D (n_dof, n_modes)
    if vecs_selected.ndim == 1:
        vecs_selected = vecs_selected.reshape(-1, 1)

    # Convention matching pk_solverv3: g = Re(lambda)/|Im(lambda)|
    g_V    = np.real(eigs_selected) / np.maximum(np.abs(np.imag(eigs_selected)), 1e-12)
    g_V_dimensional = np.real(eigs_selected)  # in rad/s, for diagnostics

    # for paper conventions:
    g_V = g_V_dimensional

    freq_V = np.abs(np.imag(eigs_selected))  # frequencies in rad/s

    # g_V sign convention: negative g = stable, positive g = unstable
    g_V = g_V  # Already correct sign from eigenvalue decomposition

    if return_eigenvectors:
        return g_V.real, freq_V.real, vecs_selected, eigs_selected
    else:
        return g_V.real, freq_V.real


def solve(config, structural_results, coupling_results, aerogrid=None):

    print("\n--- Solving for Flutter ---")
    
    if config.pk_method:
        return _solve_pk_method(config, structural_results, coupling_results, aerogrid)
    
    if config.roger_fit:
        return _solve_roger_fit_method(config, config.V_list, structural_results, coupling_results, aerogrid)

    else:
        raise ValueError("No flutter solution method selected in config.")


def _solve_roger_fit_method(config, V_list, structural_results, coupling_results, aerogrid):
    """
    Solve flutter using Roger's Rational Function Approximation (RFA) with MAC-based mode tracking.
    
    Implements the full per-velocity algorithm from RFA_workflow.md Section 3 with enhancements:
    - STEP A: Build k-list for each velocity
    - STEP B: Compute Q_modal data at each k
    - STEP C: Fit Roger to Q_modal data
    - STEP D: Assemble augmented state matrix A_aug
    - STEP E: Extract flutter modes from eigenvalue analysis (with eigenvectors for MAC)
    - STEP F: Apply MAC-based mode matching across velocity steps for continuity tracking
    
    The MAC (Modal Assurance Criterion) ensures mode continuity by measuring eigenvector
    similarity between consecutive velocity steps, matching modes that have highest MAC
    and closest frequency proximity (similar to P-K method implementation).
    """
    
    from Qjj.precompute_qjj import interp_qjj_from_disk_old

    print("Using Roger's Rational Function Approximation (RFA) for direct v-g-f analysis.")
    # Load pre-computed RFA matrices in PANEL SPACE (not projected)
    # Use direct eigenvalue sweep method instead of p-k iteration

    # Unpack structural data
    M_hat   = structural_results.M_hat
    C_hat   = structural_results.C_hat
    K_hat   = structural_results.K_hat
    omega_n = np.sqrt(np.abs(structural_results.dry_eigenvalues))   # rad/s
    Phi     = structural_results.dry_eigenvectors[:, :len(config.modes_to_analyze)]

    # Unpack coupling data
    Z           = coupling_results['Z']
    Z_qs        = coupling_results['Z_qs']
    Z_force     = coupling_results.get('Z_force', None)
    panel_areas = aerogrid['A']
    A_diag      = np.diag(panel_areas)

    rho     = config.rho_f[config.fluid]
    c_sound = config.c_sound[config.fluid]
    n_modes = len(config.modes_to_analyze)
    N       = config.n_lag
    k_data_min = config.k_list[0]
    k_data_max = config.k_list[-1]

    # Pre-compute once outside V loop (does not depend on V)
    Z_force_modal = Z_force @ Phi          # shape: (Np, Nm)

    g_sweep    = []
    freq_sweep = []
    V_sweep    = []
    vecs_sweep = []  # Store eigenvectors for MAC tracking
    eigs_sweep = []  # Store eigenvalues for diagnostics
    mac_history = []  # Store MAC matrices between consecutive velocity steps

    print(f"\nRoger RFA sweep over {len(V_list)} velocities with MAC-based mode tracking...")
    print(f"{'='*70}")
    print(f"  ROGER RFA WITH LAG STATE DYNAMICS AND MAC MODE TRACKING")
    print(f"  STEP A: Build k-list for each V")
    print(f"  STEP B: Compute Q_modal(k) data")
    print(f"  STEP C: Fit Roger coefficients B0, B1, B2, Blag")
    print(f"  STEP D: Assemble augmented state matrix A_aug")
    print(f"  STEP E: Extract flutter modes from eigenvalues + eigenvectors")
    print(f"  STEP F: Apply MAC-based mode matching across velocities")
    print(f"{'='*70}\n")

    # MAC tracking configuration
    freq_margin_rfa = config.rfa_freq_margin
    w_freq_rfa      = config.rfa_w_freq
    w_mac_rfa       = config.rfa_w_mac

    for i_V, V in enumerate(V_list):
        print(f"\n--- VELOCITY STEP {i_V+1}/{len(V_list)}: V = {V:.2f} m/s ---")

        # STEP A: k-list for this V
        omega_phys = np.linspace(0.1 * omega_n[0], 3.0 * omega_n[-1], 100)
        k_list_V   = np.clip(omega_phys / V, k_data_min, k_data_max)
        k_list_V   = np.unique(k_list_V)
        Nk         = len(k_list_V)
        blag = np.logspace(np.log10(k_list_V[1] * 0.3), 
                   np.log10(k_list_V[-1] * 1.5), N)

        print(f"  STEP A: k_list_V has {Nk} points, range [{k_list_V[0]:.6f}, {k_list_V[-1]:.6f}]")
        print(f"  STEP A: lag poles blag range [{blag[0]:.6f}, {blag[-1]:.6f}]")

        # STEP B: Q_modal data at each k
        Q_modal_data = []
        Ma = V / c_sound
        for k in k_list_V:
            Qjj_k   = interp_qjj_from_disk_old(config.qjj_dir, k, Ma)   # (Np, Np)
            ik      = 1j * k
            w_k     = (ik * Z + Z_qs) @ Phi                              # (Np, Nm)
            aero_k  = Qjj_k @ w_k                                        # (Np, Nm)
            Qm_k    = Z_force_modal.T @ A_diag @ aero_k                  # (Nm, Nm)
            Q_modal_data.append(Qm_k)
        Q_modal_array = np.array(Q_modal_data)                           # (Nk, Nm, Nm)

        print(f"  STEP B: Q_modal computed at {Nk} k values, shape {Q_modal_array.shape}")

        # STEP C: Roger fit
        B0, B1, B2, Blag = _fit_roger_to_Q_modal(
            Q_modal_array, k_list_V, blag, n_modes, N)

        print(f"  STEP C: Roger fit complete - B0/B1/B2 shape {B0.shape}, {len(Blag)} lag terms")

        # STEP C.4: Fit quality check
        max_err = 0.0
        for i_k, k in enumerate(k_list_V):
            Q_fit = _reconstruct_Q_modal(k, B0, B1, B2, Blag, blag)
            Q_ref = Q_modal_data[i_k]
            err = np.linalg.norm(Q_fit - Q_ref, 'fro') / np.linalg.norm(Q_ref, 'fro')
            max_err = max(max_err, err)
            if i_k % max(1, Nk // 3) == 0:
                print(f"    k={k:.4f}: fit error = {err:.2e}")
        print(f"  STEP C: Max fit error = {max_err:.2e} (target < 1%)")

        # STEP D: Augmented state matrix
        A_aug = _build_A_aug(
            config, V, rho, M_hat, C_hat, K_hat, B0, B1, B2, Blag, blag, n_modes, N)

        print(f"  STEP D: A_aug assembled, shape {A_aug.shape}")

        # STEP E: Eigenvalue analysis (with eigenvectors for MAC)
        # First extraction is frequency-based; subsequent extractions use MAC to find best modes
        if i_V > 0 and len(vecs_sweep) > 0:
            # Have previous modes: use MAC-based selection to search for modes
            vecs_V_prev = vecs_sweep[-1]
            freq_V_prev = freq_sweep[-1]
            g_V, freq_V, vecs_V, eigs_V = _extract_flutter_modes(
                A_aug, n_modes, omega_n, return_eigenvectors=True,
                vecs_previous=vecs_V_prev, freqs_previous=freq_V_prev
            )
        else:
            # First velocity step: use frequency-based selection
            g_V, freq_V, vecs_V, eigs_V = _extract_flutter_modes(A_aug, n_modes, omega_n, return_eigenvectors=True)
        
        g_V = -g_V

        print(f"  STEP E: Flutter modes extracted")
        print(f"    Frequencies (rad/s): {freq_V}")
        print(f"    Damping ratios:      {g_V}")

        # STEP F: MAC-based mode tracking across velocity steps
        if i_V > 0 and len(vecs_sweep) > 0:
            print(f"  STEP F: Applying MAC-based mode matching...")
            
            # Retrieve previous velocity data
            freq_V_prev = freq_sweep[-1]
            vecs_V_prev = vecs_sweep[-1]
            eigs_V_prev = eigs_sweep[-1]
            
            # Get the DOF size from the augmented matrix
            n_dof_total = A_aug.shape[0]  # Total DOFs including lag states
            
            # Defensive check: ensure eigenvectors are 2D with correct shape
            if vecs_V.ndim != 2 or vecs_V.shape[0] != n_dof_total:
                print(f"  ⚠ Warning: vecs_V shape is {vecs_V.shape}, expected ({n_dof_total}, {n_modes}). Reshaping...")
                try:
                    vecs_V = vecs_V.reshape(n_dof_total, n_modes)
                except ValueError as e:
                    print(f"  ✗ ERROR: Cannot reshape vecs_V to ({n_dof_total}, {n_modes}). Got error: {e}")
                    print(f"    Current shape: {vecs_V.shape}, total elements: {vecs_V.size}")
                    raise
            
            if vecs_V_prev.ndim != 2 or vecs_V_prev.shape[0] != n_dof_total:
                print(f"  ⚠ Warning: vecs_V_prev shape is {vecs_V_prev.shape}, expected ({n_dof_total}, {n_modes}). Reshaping...")
                try:
                    vecs_V_prev = vecs_V_prev.reshape(n_dof_total, n_modes)
                except ValueError as e:
                    print(f"  ✗ ERROR: Cannot reshape vecs_V_prev to ({n_dof_total}, {n_modes}). Got error: {e}")
                    print(f"    Current shape: {vecs_V_prev.shape}, total elements: {vecs_V_prev.size}")
                    raise
            
            # Match modes between current and previous velocity step
            reordering, mac_mat, match_info = roger_mode_tracking.match_modes_between_velocities(
                eigs_V, vecs_V, freq_V,
                eigs_V_prev, vecs_V_prev, freq_V_prev,
                freq_margin=freq_margin_rfa,
                w_freq=w_freq_rfa,
                w_mac=w_mac_rfa,
                verbose=True
            )
            
            # Handle variable number of modes (reordering may contain -1 for unmatched)
            n_modes_matched = np.sum(reordering >= 0)
            n_modes_current = len(freq_V)
            n_modes_previous = len(freq_V_prev)
            
            if n_modes_current != n_modes_previous:
                print(f"  [INFO] Mode count change: {n_modes_previous} → {n_modes_current}")
                print(f"         {n_modes_matched} modes matched, {n_modes_current - n_modes_matched} new, "
                      f"{n_modes_previous - n_modes_matched} lost")
            
            # Reorder only matched modes; keep new modes in their current order
            if n_modes_matched > 0:
                # Build reordering for matched modes
                reordering_valid = reordering[reordering >= 0]
                g_V_new, freq_V_new, vecs_V_new, eigs_V_new = roger_mode_tracking.reorder_and_normalize_modes(
                    eigs_V[reordering_valid], vecs_V[:, reordering_valid], 
                    freq_V[reordering_valid], g_V[reordering_valid], 
                    np.arange(n_modes_matched)
                )
                
                # If there are new modes, append them to the tracked modes
                if n_modes_current > n_modes_matched:
                    new_mode_indices = np.where(reordering < 0)[0]
                    # We'll handle this by just keeping all modes from current step
                    # and using best MAC match for unmatched ones
                    print(f"  [WARNING] {len(new_mode_indices)} new mode(s) detected. Keeping in tracking...")
                    # Use all current modes (new ones will eventually get matched)
                    g_V_new, freq_V_new, vecs_V_new, eigs_V_new = roger_mode_tracking.reorder_and_normalize_modes(
                        eigs_V, vecs_V, freq_V, g_V, np.arange(n_modes_current)
                    )
            else:
                # No matches found - use current modes as-is
                print(f"  [WARNING] No modes matched with previous step. Using current modes as baseline...")
                g_V_new, freq_V_new, vecs_V_new, eigs_V_new = roger_mode_tracking.reorder_and_normalize_modes(
                    eigs_V, vecs_V, freq_V, g_V, np.arange(n_modes_current)
                )
            
            # Validate returned shapes
            if vecs_V_new.ndim != 2:
                print(f"  ✗ ERROR after reordering: vecs_V_new has shape {vecs_V_new.shape}, expected 2D")
                raise ValueError(f"vecs_V_new should be 2D but got shape {vecs_V_new.shape}")
            
            g_V = g_V_new
            freq_V = freq_V_new
            vecs_V = vecs_V_new
            eigs_V = eigs_V_new
            
            mac_history.append(mac_mat)
            
            # Diagnostic summary for this velocity step
            print(f"    MAC matching summary:")
            for info in match_info:
                validity = "✓" if info['valid'] else "✗"
                print(f"      {validity} Mode {info['mode_index']}: MAC={info['mac']:.3f}, "
                      f"freq_change={info['freq_error']*100:.1f}%")
        else:
            # First velocity: normalize but no matching needed
            if vecs_V.ndim == 2 and vecs_V.shape[1] > 0:
                for j in range(vecs_V.shape[1]):
                    vec = vecs_V[:, j]
                    norm_val = np.linalg.norm(vec)
                    if norm_val > 1e-12:
                        vecs_V[:, j] = vec / norm_val
            else:
                print(f"  ⚠ Warning: vecs_V shape is {vecs_V.shape}, skipping normalization")

        # Ensure vecs_V has the correct shape before storage
        if vecs_V.ndim != 2 or vecs_V.shape[1] != n_modes:
            print(f"  ⚠ Warning: Correcting vecs_V shape from {vecs_V.shape} to ({A_aug.shape[0]}, {n_modes})")
            try:
                vecs_V = vecs_V.reshape(A_aug.shape[0], n_modes)
            except ValueError:
                print(f"  ✗ ERROR: Cannot reshape vecs_V. Got {vecs_V.size} elements, need {A_aug.shape[0] * n_modes}")
                raise

        V_sweep.append(V)
        g_sweep.append(g_V)
        freq_sweep.append(freq_V)
        vecs_sweep.append(vecs_V)
        eigs_sweep.append(eigs_V)

    V_arr = np.array(V_sweep)
    g_arr = np.array(g_sweep)    # (Nv, Nm)
    f_arr = np.array(freq_sweep) # (Nv, Nm)

    # Compute flutter speed
    Vf, ff = None, None
    for j in range(g_arr.shape[1]):
        Vf_j, ff_j = vgvf_plotting.first_flutter_crossing(V_arr, g_arr[:, j])
        if Vf_j is not None:
            if Vf is None or Vf_j < Vf:
                Vf, ff = Vf_j, ff_j

    # MAC diagnostics and summary
    print(f"\n{'='*70}")
    print(f"  MAC MODE TRACKING DIAGNOSTICS")
    print(f"{'='*70}")
    diagnostics = roger_mode_tracking.diagnostic_mac_summary(
        mac_history, freq_sweep, V_list, verbose=True
    )

    print(f"\n{'='*70}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"Result arrays shape:")
    print(f"  V shape: {V_arr.shape}")
    print(f"  g shape: {g_arr.shape}  (damping ratios)")
    print(f"  f shape: {f_arr.shape}  (frequencies in rad/s)")
    print(f"Flutter analysis complete. Estimated flutter speed: {Vf} m/s")
    print(f"{'='*70}\n")

    return FlutterResults(
        velocities=V_arr, damping=g_arr, frequencies=f_arr,
        modes=config.modes_to_analyze,
        flutter_speed=Vf, flutter_frequency=ff,
        raw_results=None, pk_solver=None)


def _solve_pk_method(config, structural_results, coupling_results, aerogrid=None):

    print("Using P-K method.")
    
    print("Using classical P-K iterative method.")
    
    Z = coupling_results['Z']
    Apan = coupling_results['Apan']
    Z_qs = coupling_results['Z_qs']
    Z_force = coupling_results.get('Z_force', None)
    
    # Extract panel areas for theory-compliant formula
    panel_areas = None
    panel_areas = aerogrid['A']  # Panel areas for diagonal matrix A

    
    solver = pk_solverv3.PKSolverV3(
        structural_results.M_hat,
        structural_results.C_hat,
        structural_results.K_hat,
        structural_results.dry_eigenvalues,
        k_guess=config.k_guess
    )

    # Note: structural_results.dry_eigenvectors already contains only the selected modes
    # in sorted frequency order (from structural_analysis.py)
    Phi_FSI_analisys = structural_results.dry_eigenvectors  # Already sorted and selected


    # Create the function to calculate generalized aerodynamic forces
    Qg_func = solver.make_Qg_func(
        config,
        config.paths['FSI'],
        Z,
        Apan,
        Z_qs,
        Phi_FSI_analisys,  # Free DOFs only to match Apan/Z dimensions
        structural_results.dry_eigenvalues,
        b=config.chord / 2,
        c_sound=config.c_sound[config.fluid],
        out_dir_klist=config.qjj_dir,
        out_dir_vjj=config.vjj_dir,
        alpha_r=config.alpha_r,
        Z_force=Z_force,  # Pass G_f matrix for theory-compliant force transfer
        panel_areas=panel_areas  # Pass panel areas for diagonal matrix A
    )

    #results_still_fluid = solver.still_fluid_eigenvalues(config, Qg_func)
    #breakpoint()

    # Sweep through the velocity range
    results = solver.sweep(
        V_list=config.V_list,
        rho=config.rho_f[config.fluid],
        b=config.chord / 2,
        Qg_func=Qg_func,
        modes=config.modes_to_analyze,
        dry_vals=structural_results.dry_eigenvalues,
        dry_vecs=structural_results.dry_eigenvectors,

        tol=config.pk_tol,
        fXK0=config.pk_fXK0,
        fRLX=config.pk_fRLX,
        freq_margin=config.pk_freq_margin,
        perturb_k=config.pk_perturb_k,

        w_freq=config.pk_w_freq,
        w_mac=config.pk_w_mac,

        mac_matching=config.mac_matching,
        last_converged_mode_matching=config.last_converged_mode_matching,

        max_iter=config.pk_max_iter
    )

    # Process and return results
    V, g, omega = vgvf_plotting.results_to_arrays(results, config.modes_to_analyze)
    
    Vf, ff = None, None
    for j in range(g.shape[1]):
        Vf_j, ff_j = vgvf_plotting.first_flutter_crossing(V, g[:, j])
        if Vf_j is not None:
            if Vf is None or Vf_j < Vf:
                Vf, ff = Vf_j, ff_j

    # Plot Cp at fixed k values for all modes
    if aerogrid is not None:
        post_processing.plot_cp_at_fixed_k_values(config, aerogrid, results, k_values=[0.001, 0.5, 1.0])

    # Post-process: sample Q_modal entries at convergence for each speed
    try:
        post_processing.plot_qmodal_entries_at_convergence(
            config,
            structural_results,
            coupling_results,
            FlutterResults(
                velocities=V,
                damping=g,
                frequencies=omega,
                modes=config.modes_to_analyze,
                flutter_speed=Vf,
                flutter_frequency=ff,
                raw_results=results,
                pk_solver=solver
            ),
            mode_index=0,
            save_csv=True
        )
    except Exception as e:
        print(f"[Warning] Q_modal post-processing failed: {e}")

    print(f"Flutter analysis complete. Estimated flutter speed: {Vf}")

    return FlutterResults(
        velocities=V,
        damping=g,
        frequencies=omega,
        modes=config.modes_to_analyze,
        flutter_speed=Vf,
        flutter_frequency=ff,
        raw_results=results,  # Store raw results for eigenvalue plotting
        pk_solver=solver  # Store solver instance for contributions plotting
    )