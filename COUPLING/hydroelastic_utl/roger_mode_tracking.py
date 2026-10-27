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


def reference_eigenvector_for_mac(
    vecs_sweep,
    slot,
    *,
    raw_results=None,
    mac_reference="previous_velocity",
):
    """
    Eigenvector used as MAC reference for branch ``slot``.

    ``previous_velocity`` (default): wet eigenvector at V_{i-1} for the same branch
    slot — allows mode shapes to evolve toward coalescence (flutter theory).

    ``last_valid``: latest converged eigenvector before any ``mode_lost`` gap
    (recovery / reacquisition after branch dropout).
    """
    ref = str(mac_reference or "previous_velocity").lower().replace("-", "_")
    if ref in ("previous", "previous_velocity", "prev", "prev_velocity"):
        if vecs_sweep:
            v_prev = vecs_sweep[-1]
            if (
                v_prev is not None
                and hasattr(v_prev, "ndim")
                and v_prev.ndim == 2
                and 0 <= int(slot) < v_prev.shape[1]
            ):
                return v_prev[:, int(slot)]
    return last_valid_branch_eigenvector(vecs_sweep, raw_results, slot)


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


def last_valid_branch_mode(raw_results, slot):
    """Latest converged branch snapshot for slot ``slot`` (search backward in V history)."""
    for step in range(len(raw_results) - 1, -1, -1):
        modes = raw_results[step].get("modes", [])
        if slot >= len(modes):
            continue
        md = modes[slot]
        omega = md.get("omega", np.nan)
        p_val = md.get("p", np.nan)
        if not md.get("converged", False):
            continue
        if not (np.isfinite(omega) and omega > 0.0):
            continue
        if p_val is None or not np.isfinite(np.abs(p_val)):
            continue
        return md, step
    return None, None


def predict_branch_omega_from_history(raw_results, slot):
    """Linear ω extrapolation from the last two valid converged branch states."""
    p_hist = []
    for step in range(len(raw_results) - 1, -1, -1):
        modes = raw_results[step].get("modes", [])
        if slot >= len(modes):
            continue
        md = modes[slot]
        if md.get("mode_lost", False):
            continue
        p_val = md.get("p", None)
        if not md.get("converged", False):
            continue
        if p_val is None or not np.isfinite(np.real(p_val)):
            continue
        p_hist.insert(0, complex(p_val))
        if len(p_hist) >= 2:
            break
    if len(p_hist) >= 2:
        return float(np.abs(np.imag(2.0 * p_hist[-1] - p_hist[-2])))
    if len(p_hist) == 1:
        return float(np.abs(np.imag(p_hist[0])))
    return None


def valid_branch_V_omega_history(
    raw_results,
    slot,
    max_points=4,
    *,
    omega_cap=None,
):
    """Valid (V, ω, step) snapshots for branch ``slot``, newest last."""
    hist = []
    for step in range(len(raw_results) - 1, -1, -1):
        entry = raw_results[step]
        modes = entry.get("modes", [])
        if slot >= len(modes):
            continue
        md = modes[slot]
        if md.get("mode_lost", False) or not md.get("converged", False):
            continue
        omega = float(md.get("omega", np.nan))
        if not (np.isfinite(omega) and omega > 0.0):
            continue
        if omega_cap is not None and np.isfinite(omega_cap) and omega > float(omega_cap):
            continue
        V_step = float(entry.get("V", np.nan))
        if not np.isfinite(V_step):
            continue
        hist.insert(0, (V_step, omega, int(step)))
        if len(hist) >= int(max_points):
            break
    return hist


def branch_lost_streak_length(raw_results, slot):
    """How many consecutive recent velocity steps ended with ``mode_lost`` on ``slot``."""
    if not raw_results:
        return 0
    streak = 0
    for step in range(len(raw_results) - 1, -1, -1):
        modes = raw_results[step].get("modes", [])
        if slot >= len(modes):
            break
        md = modes[slot]
        if md.get("mode_lost", False) or not md.get("converged", False):
            streak += 1
        else:
            break
    return streak


def branch_slot_lost_at_previous(raw_results, slot):
    if not raw_results:
        return False
    modes = raw_results[-1].get("modes", [])
    if slot >= len(modes):
        return False
    return bool(modes[slot].get("mode_lost", False))


def _partner_slot_for(slot, n_slots):
    if int(n_slots) < 2:
        return None
    return int(slot) - 1 if int(slot) > 0 else 1


def _median_coalescence_gap(raw_results, slot, partner_slot, *, omega_cap=None):
    """Median (ω_slot − ω_partner) at steps where both branches were valid."""
    if partner_slot is None or partner_slot < 0:
        return None
    self_hist = valid_branch_V_omega_history(
        raw_results, slot, max_points=8, omega_cap=omega_cap
    )
    partner_hist = valid_branch_V_omega_history(
        raw_results, partner_slot, max_points=8, omega_cap=omega_cap
    )
    if not self_hist or not partner_hist:
        return None
    gaps = []
    for V_s, w_s, step_s in self_hist:
        for V_p, w_p, step_p in partner_hist:
            if int(step_s) == int(step_p):
                gaps.append(float(w_s) - float(w_p))
    if not gaps:
        return None
    return float(np.median(gaps))


def predict_branch_omega_from_partner(
    raw_results,
    slot,
    partner_slot,
    V_current,
    *,
    omega_cap=None,
):
    """ω(slot) ≈ ω_partner(V) + median historical gap at co-stepped valid points."""
    if partner_slot is None or partner_slot < 0:
        return None
    partner_hist = valid_branch_V_omega_history(
        raw_results, partner_slot, max_points=4, omega_cap=omega_cap
    )
    if not partner_hist:
        return None
    gap = _median_coalescence_gap(
        raw_results, slot, partner_slot, omega_cap=omega_cap
    )
    if gap is None:
        return None
    V_p2, w_p2, _ = partner_hist[-1]
    if len(partner_hist) >= 2:
        V_p1, w_p1, _ = partner_hist[-2]
        dVp = float(V_p2) - float(V_p1)
        w_p_pred = (
            float(w_p2) + (float(V_current) - float(V_p2)) * (float(w_p2) - float(w_p1)) / dVp
            if abs(dVp) > 1e-9
            else float(w_p2)
        )
    else:
        w_p_pred = float(w_p2)
    return float(w_p_pred + float(gap))


def _self_history_unreliable(raw_results, slot, partner_slot, hist, *, omega_cap=None):
    """True when linear self extrapolation is likely corrupted (orphan / collapse)."""
    if not hist:
        return True
    if partner_slot is None or partner_slot < 0:
        return False
    partner_hist = valid_branch_V_omega_history(
        raw_results, partner_slot, max_points=4, omega_cap=omega_cap
    )
    if not partner_hist:
        return False
    w_self = float(hist[-1][1])
    w_partner = float(partner_hist[-1][1])
    gap_now = w_self - w_partner
    median_gap = _median_coalescence_gap(
        raw_results, slot, partner_slot, omega_cap=omega_cap
    )
    if median_gap is not None and abs(gap_now) < 0.35 * abs(median_gap):
        return True
    if int(slot) > int(partner_slot) and w_self <= w_partner * 1.02:
        return True
    if int(slot) < int(partner_slot) and w_self >= w_partner * 0.98:
        return True
    if len(hist) >= 2 and len(partner_hist) >= 2:
        V1, w1, _ = hist[-2]
        V2, w2, _ = hist[-1]
        Vp1, wp1, _ = partner_hist[-2]
        Vp2, wp2, _ = partner_hist[-1]
        dV = float(V2) - float(V1)
        dVp = float(Vp2) - float(Vp1)
        if abs(dV) > 1e-9 and abs(dVp) > 1e-9:
            slope_self = abs((float(w2) - float(w1)) / dV)
            slope_partner = abs((float(wp2) - float(wp1)) / dVp)
            if slope_partner > 1e-9 and slope_self > 2.5 * slope_partner:
                return True
    if omega_cap is not None and w_self > 0.92 * float(omega_cap):
        return True
    return False


def predict_branch_omega_at_V(
    raw_results,
    slot,
    V_current,
    *,
    omega_n=None,
    mode_indices_tracked=None,
    max_struct_frac=1.35,
    partner_slot=None,
    prefer_partner_after_streak=2,
):
    """
    Predict wet-branch ω at ``V_current`` from the last valid (V, ω) history.

    History excludes orphan captures above the structural ω cap. When self history
    is unreliable or the branch has been lost for several steps, prefer partner-branch
    coalescence (ω_partner + median historical gap).
    """
    omega_cap = None
    if (
        omega_n is not None
        and mode_indices_tracked is not None
        and slot < len(mode_indices_tracked)
    ):
        omega_n_arr = np.asarray(omega_n, dtype=float).ravel()
        m_struct = int(mode_indices_tracked[slot])
        if 0 <= m_struct < omega_n_arr.size and float(max_struct_frac) > 0.0:
            omega_cap = float(omega_n_arr[m_struct]) * float(max_struct_frac)

    if partner_slot is None:
        partner_slot = _partner_slot_for(slot, 2)

    omega_partner = predict_branch_omega_from_partner(
        raw_results,
        slot,
        partner_slot,
        V_current,
        omega_cap=omega_cap,
    )

    hist = valid_branch_V_omega_history(
        raw_results, slot, max_points=2, omega_cap=omega_cap
    )
    omega_self = None
    if len(hist) >= 2:
        V1, w1, _ = hist[-2]
        V2, w2, _ = hist[-1]
        dV = float(V2) - float(V1)
        omega_self = (
            float(w2 + (float(V_current) - float(V2)) * (w2 - w1) / dV)
            if abs(dV) > 1e-9
            else float(w2)
        )
    elif len(hist) == 1:
        omega_self = float(hist[0][1])

    streak = branch_lost_streak_length(raw_results, slot)
    self_bad = _self_history_unreliable(
        raw_results, slot, partner_slot, hist, omega_cap=omega_cap
    )

    if (
        streak >= int(prefer_partner_after_streak)
        and omega_partner is not None
        and np.isfinite(omega_partner)
        and omega_partner > 0.0
    ):
        omega_pred = float(omega_partner)
    elif self_bad and omega_partner is not None and np.isfinite(omega_partner):
        if omega_self is not None and np.isfinite(omega_self) and omega_self > 0.0:
            omega_pred = float(0.25 * omega_self + 0.75 * omega_partner)
        else:
            omega_pred = float(omega_partner)
    elif omega_self is not None:
        omega_pred = float(omega_self)
    elif omega_partner is not None:
        omega_pred = float(omega_partner)
    else:
        omega_pred = predict_branch_omega_from_history(raw_results, slot)

    if omega_pred is None or not np.isfinite(omega_pred) or omega_pred <= 0.0:
        return omega_pred
    if omega_cap is not None and float(omega_pred) > float(omega_cap):
        omega_pred = float(omega_cap)
    return float(omega_pred)


def last_valid_branch_eigenvector(vecs_sweep, raw_results, slot):
    """Structural eigenvector from the latest valid converged step for ``slot``."""
    _, step_ref = last_valid_branch_mode(raw_results, slot)
    if (
        step_ref is None
        or step_ref < 0
        or step_ref >= len(vecs_sweep)
        or vecs_sweep[step_ref] is None
    ):
        return None
    vecs_step = vecs_sweep[step_ref]
    if vecs_step.ndim != 2 or slot >= vecs_step.shape[1]:
        return None
    return vecs_step[:, slot]


def reacquire_lost_branch_in_pool(
    slot,
    freq_pool,
    vecs_pool,
    raw_results,
    vecs_sweep,
    V_current,
    used_pool_indices,
    *,
    rel_band=0.20,
    rel_band_growth=0.04,
    rel_band_max=0.35,
    min_mac=0.65,
    w_freq=0.8,
    w_mac=0.6,
    n_structural_dof=None,
    omega_n=None,
    mode_indices_tracked=None,
    max_struct_frac=1.35,
    n_slots=2,
    partner_omega=None,
    min_partner_sep_frac=0.08,
    prefer_partner_streak=2,
    mac_reference="previous_velocity",
):
    """
    Search the wet root pool near the predicted ω(V) neighbourhood and pick the
    best MAC match to the reference branch eigenvector (previous V or last valid).

    Returns pool index, or -1 if no acceptable root is found.
    """
    freq_pool = np.asarray(freq_pool, dtype=float).ravel()
    if freq_pool.size == 0:
        return -1

    partner_slot = _partner_slot_for(slot, n_slots)

    omega_cap = None
    omega_struct = None
    if (
        omega_n is not None
        and mode_indices_tracked is not None
        and slot < len(mode_indices_tracked)
    ):
        omega_n_arr = np.asarray(omega_n, dtype=float).ravel()
        m_struct = int(mode_indices_tracked[slot])
        if 0 <= m_struct < omega_n_arr.size:
            omega_struct = float(omega_n_arr[m_struct])
            if float(max_struct_frac) > 0.0:
                omega_cap = float(omega_struct) * float(max_struct_frac)

    omega_pred = predict_branch_omega_at_V(
        raw_results,
        slot,
        V_current,
        omega_n=omega_n,
        mode_indices_tracked=mode_indices_tracked,
        max_struct_frac=max_struct_frac,
        partner_slot=partner_slot,
        prefer_partner_after_streak=int(prefer_partner_streak),
    )
    if omega_pred is None or not np.isfinite(omega_pred) or omega_pred <= 0.0:
        return -1
    if omega_cap is not None and float(omega_pred) > float(omega_cap):
        omega_pred = float(omega_cap)

    streak = branch_lost_streak_length(raw_results, slot)
    band = min(
        float(rel_band) + float(rel_band_growth) * max(0, int(streak) - 1),
        float(rel_band_max),
    )
    median_gap = _median_coalescence_gap(
        raw_results, slot, partner_slot, omega_cap=omega_cap
    )

    u_ref = reference_eigenvector_for_mac(
        vecs_sweep, slot, raw_results=raw_results, mac_reference=mac_reference
    )
    used = {int(i) for i in used_pool_indices if int(i) >= 0}

    partner_w = None
    if partner_omega is not None and np.isfinite(partner_omega) and float(partner_omega) > 0.0:
        partner_w = float(partner_omega)

    best_i = -1
    best_cost = np.inf
    best_mac = 0.0
    for i in range(freq_pool.size):
        if int(i) in used:
            continue
        f_i = float(freq_pool[i])
        if f_i <= 0.0:
            continue
        if omega_cap is not None and f_i > float(omega_cap) * 1.01:
            continue
        if partner_w is not None and partner_slot is not None and float(min_partner_sep_frac) > 0.0:
            if int(slot) > int(partner_slot):
                min_omega = partner_w * (1.0 + float(min_partner_sep_frac))
                if median_gap is not None and median_gap > 0.0:
                    min_omega = max(min_omega, partner_w + 0.45 * float(median_gap))
                if f_i < min_omega:
                    continue
            elif int(slot) < int(partner_slot):
                max_omega = partner_w * (1.0 - float(min_partner_sep_frac))
                if median_gap is not None and median_gap < 0.0:
                    max_omega = min(max_omega, partner_w + 0.45 * float(median_gap))
                if f_i > max_omega:
                    continue
        freq_err = abs(f_i - float(omega_pred)) / max(float(omega_pred), 1e-12)
        if freq_err > band:
            continue
        if f_i > float(omega_pred) * (1.0 + 0.5 * band):
            continue
        mac_i = 0.0
        if u_ref is not None and vecs_pool is not None and vecs_pool.ndim == 2:
            mac_i = compute_mac(vecs_pool[:, i], u_ref, n_structural_dof=n_structural_dof)
        if mac_i < float(min_mac):
            continue
        sep_pen = 0.0
        if partner_w is not None and partner_w > 0.0:
            sep_pen = abs(f_i - partner_w) / max(f_i, partner_w, 1e-6)
            if sep_pen < float(min_partner_sep_frac):
                continue
            if abs(f_i - float(omega_pred)) > 1e-12:
                if abs(f_i - partner_w) < abs(f_i - float(omega_pred)):
                    continue
        cost = float(w_freq) * freq_err + float(w_mac) * (1.0 - mac_i) + 0.15 * sep_pen
        if cost < best_cost:
            best_cost = cost
            best_i = int(i)
            best_mac = float(mac_i)

    if best_i >= 0:
        print(
            f"  ↻ RFA branch {slot} REACQUIRED at V={float(V_current):.2f} m/s: "
            f"ω_pred={float(omega_pred):.3f} rad/s, ω_pick={float(freq_pool[best_i]):.3f} rad/s, "
            f"MAC_ref={best_mac:.3f}, band={band:.3f}, lost_streak={streak}"
            + (
                f", ω_partner={partner_w:.3f}"
                if partner_w is not None
                else ""
            )
        )
    return best_i


def apply_rfa_branch_guard(
    g_V,
    freq_V,
    vecs_V,
    eigs_V,
    raw_results,
    vecs_sweep,
    *,
    V_current=None,
    max_rel_jump=0.30,
    min_mac_prev=0.70,
    max_struct_frac=1.35,
    min_branch_sep_frac=0.08,
    check_neighbor_collapse=True,
    mode_indices_tracked=None,
    omega_n=None,
    n_structural_dof=None,
    enable=True,
    skip_slots=None,
    mac_reference="previous_velocity",
):
    """
    After Roger branch assignment at one V, reject incoherent roots as ``mode_lost``.

    A branch is marked lost when any of:
    - relative frequency jump vs last valid converged step exceeds ``max_rel_jump``;
    - the same jump exceeds ``max_rel_jump`` vs a linear ω prediction from branch history;
    - tracked ω exceeds ``max_struct_frac`` × dry structural ω for that branch slot.
    - branch collapses onto adjacent tracked branch after having been previously separated
      by more than ``min_branch_sep_frac`` (helps avoid wrong-root capture with MAC≈1).
    """
    g_out = np.asarray(g_V, dtype=float).copy()
    f_out = np.asarray(freq_V, dtype=float).copy()
    e_out = np.asarray(eigs_V, dtype=complex).copy()
    v_out = np.array(vecs_V, dtype=complex, copy=True)
    mode_lost_flags = [False] * f_out.size

    if not enable or not raw_results:
        return g_out, f_out, v_out, e_out, mode_lost_flags

    # Reference frequencies from last valid converged state, per tracked slot.
    omega_ref_list = [np.nan] * f_out.size
    for j in range(f_out.size):
        md_ref_j, _ = last_valid_branch_mode(raw_results, j)
        if md_ref_j is not None:
            omega_ref_list[j] = float(md_ref_j["omega"])

    skip_slots = set(skip_slots or [])

    for j in range(f_out.size):
        if j in skip_slots:
            continue
        omega = float(f_out[j])
        if not np.isfinite(omega) or omega <= 0.0:
            continue

        md_ref, step_ref = last_valid_branch_mode(raw_results, j)
        if md_ref is None:
            continue

        recovering = branch_slot_lost_at_previous(raw_results, j)
        omega_ref = float(md_ref["omega"])
        if recovering and V_current is not None:
            partner_slot = j - 1 if j > 0 else (j + 1 if j + 1 < f_out.size else None)
            omega_pred_v = predict_branch_omega_at_V(
                raw_results,
                j,
                V_current,
                omega_n=omega_n,
                mode_indices_tracked=mode_indices_tracked,
                max_struct_frac=max_struct_frac,
                partner_slot=partner_slot,
            )
            if omega_pred_v is not None and np.isfinite(omega_pred_v) and omega_pred_v > 0.0:
                omega_ref = float(omega_pred_v)

        rel_jump = abs(omega - omega_ref) / max(omega_ref, 1e-6)

        u_ref = reference_eigenvector_for_mac(
            vecs_sweep, j, raw_results=raw_results, mac_reference=mac_reference
        )

        mac_prev = (
            compute_mac(v_out[:, j], u_ref, n_structural_dof=n_structural_dof)
            if u_ref is not None
            else float("nan")
        )

        partner_slot_g = j - 1 if j > 0 else (j + 1 if j + 1 < f_out.size else None)
        omega_pred = predict_branch_omega_at_V(
            raw_results,
            j,
            V_current,
            omega_n=omega_n,
            mode_indices_tracked=mode_indices_tracked,
            max_struct_frac=max_struct_frac,
            partner_slot=partner_slot_g,
        ) if V_current is not None else predict_branch_omega_from_history(raw_results, j)
        rel_jump_pred = (
            abs(omega - float(omega_pred)) / max(float(omega_pred), 1e-6)
            if omega_pred is not None and np.isfinite(omega_pred)
            else float("nan")
        )

        incoherent = rel_jump > float(max_rel_jump)
        if not recovering and np.isfinite(rel_jump_pred) and rel_jump_pred > float(max_rel_jump):
            incoherent = True
        if (
            not recovering
            and rel_jump > float(max_rel_jump)
            and np.isfinite(mac_prev)
            and mac_prev < float(min_mac_prev)
        ):
            incoherent = True

        if (
            max_struct_frac is not None
            and float(max_struct_frac) > 0.0
            and mode_indices_tracked is not None
            and omega_n is not None
            and j < len(mode_indices_tracked)
        ):
            omega_n_arr = np.asarray(omega_n, dtype=float).ravel()
            m_struct = int(mode_indices_tracked[j])
            if 0 <= m_struct < omega_n_arr.size:
                omega_cap = float(omega_n_arr[m_struct]) * float(max_struct_frac)
                if omega > omega_cap:
                    neigh_cap = j - 1 if j > 0 else (j + 1 if j + 1 < f_out.size else None)
                    omega_neigh_cap = (
                        float(f_out[neigh_cap])
                        if neigh_cap is not None and np.isfinite(f_out[neigh_cap])
                        else np.nan
                    )
                    # Upper branch well above partner → likely orphan high-ω capture.
                    high_orphan = (
                        neigh_cap is not None
                        and int(j) > int(neigh_cap)
                        and np.isfinite(omega_neigh_cap)
                        and omega_neigh_cap > 0.0
                        and omega > omega_neigh_cap * 1.35
                    )
                    # Lower branch may evolve smoothly above dry ω·frac (flutter stiffening).
                    coherent_rise = (
                        int(j) == 0
                        or (
                            neigh_cap is not None
                            and int(j) < int(neigh_cap)
                        )
                    ) and rel_jump <= float(max_rel_jump) and (
                        not np.isfinite(mac_prev) or mac_prev >= float(min_mac_prev)
                    )
                    if high_orphan or not coherent_rise:
                        incoherent = True

        # Additional modeguard: if branch j collapses onto adjacent branch although it was
        # previously well-separated, treat as wrong-root capture (mark lost).
        if bool(check_neighbor_collapse) and float(min_branch_sep_frac) > 0.0 and f_out.size >= 2:
            neigh = j - 1 if j > 0 else (j + 1 if j + 1 < f_out.size else None)
            if neigh is not None:
                omega_neigh = float(f_out[neigh]) if np.isfinite(f_out[neigh]) else np.nan
                omega_ref_neigh = float(omega_ref_list[neigh]) if np.isfinite(omega_ref_list[neigh]) else np.nan
                if np.isfinite(omega_neigh) and np.isfinite(omega_ref_neigh):
                    sep_now = abs(omega - omega_neigh) / max(omega, omega_neigh, 1e-6)
                    sep_ref = abs(omega_ref - omega_ref_neigh) / max(omega_ref, omega_ref_neigh, 1e-6)
                    if sep_ref > float(min_branch_sep_frac) and sep_now < 0.5 * float(min_branch_sep_frac):
                        incoherent = True

        if incoherent:
            pred_msg = (
                f", Δω/ω_pred={rel_jump_pred:.3f}"
                if np.isfinite(rel_jump_pred)
                else ""
            )
            struct_msg = ""
            if (
                max_struct_frac is not None
                and mode_indices_tracked is not None
                and omega_n is not None
                and j < len(mode_indices_tracked)
            ):
                m_struct = int(mode_indices_tracked[j])
                omega_n_arr = np.asarray(omega_n, dtype=float).ravel()
                if 0 <= m_struct < omega_n_arr.size:
                    struct_msg = (
                        f", ω_struct_cap="
                        f"{float(omega_n_arr[m_struct]) * float(max_struct_frac):.3f}"
                    )
            sep_msg = ""
            if bool(check_neighbor_collapse) and f_out.size >= 2:
                neigh = j - 1 if j > 0 else (j + 1 if j + 1 < f_out.size else None)
                if neigh is not None:
                    omega_neigh = float(f_out[neigh]) if np.isfinite(f_out[neigh]) else np.nan
                    omega_ref_neigh = float(omega_ref_list[neigh]) if np.isfinite(omega_ref_list[neigh]) else np.nan
                    if np.isfinite(omega_neigh) and np.isfinite(omega_ref_neigh):
                        sep_now = abs(omega - omega_neigh) / max(omega, omega_neigh, 1e-6)
                        sep_ref = abs(omega_ref - omega_ref_neigh) / max(omega_ref, omega_ref_neigh, 1e-6)
                        sep_msg = f", sep_now={sep_now:.3f}, sep_ref={sep_ref:.3f}"
            print(
                f"  ✗ RFA branch {j} LOST (branch-incoherent): "
                f"ω={omega:.3f} rad/s, ω_ref={omega_ref:.3f} rad/s, "
                f"Δω/ω_ref={rel_jump:.3f}, MAC_prev={mac_prev:.3f}"
                f"{pred_msg}{struct_msg}{sep_msg}"
            )
            g_out[j] = np.nan
            f_out[j] = np.nan
            e_out[j] = np.nan
            mode_lost_flags[j] = True

    return g_out, f_out, v_out, e_out, mode_lost_flags


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
    reordering_indices : (n_prev,) int ndarray
        For each previous tracked branch j, index into the current candidate pool,
        or -1 if unmatched.
    mac_matrix : (n_curr, n_prev) ndarray
        MAC values between all current and previous mode pairs [i, j]
    matching_info : list of dicts
        Detailed information for each matched pair
    """
    from scipy.optimize import linear_sum_assignment
    
    n_curr = len(eigs_current)
    n_prev = len(eigs_previous)
    
    # Build rectangular cost matrix (n_curr x n_prev): current candidate pool vs tracked branches
    cost_matrix = np.zeros((n_curr, n_prev))
    mac_matrix = np.zeros((n_curr, n_prev))
    
    for i in range(n_curr):
        for j in range(n_prev):
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
            
            freq_error = np.abs(freqs_current[i] - freqs_previous[j]) / (freqs_previous[j] + 1e-12)
            cost = w_freq * freq_error + w_mac * (1.0 - mac_val)
            if freq_error > freq_margin:
                cost = np.inf
            cost_matrix[i, j] = cost
    
    try:
        current_indices, prev_indices = linear_sum_assignment(cost_matrix)
    except ValueError as e:
        if "infeasible" in str(e):
            print(f"  [WARNING] Cost matrix infeasible. Using greedy MAC-based matching...")
            current_indices, prev_indices = _greedy_mac_matching(
                mac_matrix, cost_matrix, n_curr, n_prev
            )
        else:
            raise
    
    # Build reordering: for each previous mode, find its best match in current
    reordering_indices = np.full(n_prev, -1, dtype=int)
    for i_curr, i_prev in zip(current_indices, prev_indices):
        if cost_matrix[i_curr, i_prev] < np.inf:
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
    matched_current = set(int(i) for i in current_indices if int(i) >= 0)
    new_modes = [i for i in range(n_curr) if i not in matched_current]
    
    if new_modes and verbose:
        print(f"  [INFO] {len(new_modes)} new mode(s) detected at indices: {new_modes}")
        for i_new in new_modes:
            best_mac = np.max(mac_matrix[i_new, :])
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
    mac_history : list of ndarray
        MAC matrices at each velocity step (starting from V_2). Each matrix is
        either square (n, n) or rectangular (n_candidates, n_tracked).
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
            if mac_mat is None:
                continue
            if mac_mat.shape[0] == mac_mat.shape[1]:
                mac_diagonals.append(np.diag(mac_mat))
            else:
                # Rectangular pool vs tracked branches: best MAC per tracked column
                mac_diagonals.append(np.max(mac_mat, axis=0))
        
        mac_diagonals = np.array(mac_diagonals) if mac_diagonals else np.empty((0, n_modes))
        if mac_diagonals.size > 0:
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
