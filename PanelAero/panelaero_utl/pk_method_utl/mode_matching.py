import numpy as np


def predict_p_target_from_history(results, s, i):
    """
    Predict p_target for mode s using extrapolation from converged p values.
    
    Uses quadratic extrapolation if 3+ points available, linear if 2 points available.
    
    Parameters
    ----------
    results : list of dicts
        List of results from previous velocity steps. Each dict has 'V' and 'modes'.
    s : int
        Mode index to predict
    i : int
        Current velocity step index
        
    Returns
    -------
    p_target : complex or None
        Predicted eigenvalue, or None if insufficient history
    """
    # Need at least 2 previous velocity steps for extrapolation
    if i < 2:
        return None
    
    # Extract available converged p values for mode s (looking back up to 3 steps)
    p_history = []
    for step_idx in range(max(0, i-3), i):
        if step_idx >= 0 and step_idx < len(results):
            modes_at_step = results[step_idx].get('modes', [])
            if s < len(modes_at_step):
                p_val = modes_at_step[s].get('p', None)
                # Check if mode was converged and not lost
                if p_val is not None and not np.isnan(p_val):
                    p_history.append(p_val)
    
    # Quadratic extrapolation with 3 points: p(n+1) = 3*p(n) - 3*p(n-1) + p(n-2)
    if len(p_history) >= 3:
        # Use last 3 points
        p_target = 3.0 * p_history[-1] - 3.0 * p_history[-2] + p_history[-3]
        return p_target
    
    # Linear extrapolation with 2 points: p(n+1) = 2*p(n) - p(n-1)
    elif len(p_history) == 2:
        p_target = 2.0 * p_history[-1] - p_history[-2]
        return p_target
    
    # Not enough history
    return None


def mode_matching_paper_compliant(vals, vecs, s, j,
                                   last_converged_omegas,
                                   converged_omegas_at_V,
                                   converged_vecs_at_V,
                                   freq_margin, w_freq, w_mac,
                                   omega_n, mode_switch_warnings_ref,
                                   prev_step_vec=None,
                                   prev_iter_vec=None,
                                   p_target=None):
    """
    MAC-driven mode matching / locking with fallback to p_target prediction.

    The tracking centre is determined by the MAC pre-selector when MAC >= 0.7.
    When MAC < 0.7, the function falls back to using p_target (predicted from
    the last 3 converged p values) for mode selection.

    Parameters
    ----------
    vals, vecs              : eigenvalues / eigenvectors at current iteration
    s                       : mode index being tracked
    last_converged_omegas   : converged frequencies from the previous velocity step
    converged_omegas_at_V   : already-converged frequencies at the current velocity
    converged_vecs_at_V     : already-converged eigenvectors at the current velocity
    freq_margin             : fractional frequency search band
    w_freq, w_mac           : weights for combined frequency/MAC score
    omega_n                 : structural natural frequencies array (from PKSolverV3)
    mode_switch_warnings_ref: list of length 1 — increment [0] to count warnings
    prev_step_vec           : converged eigenvector of mode s from the previous
                              velocity step (cross-velocity MAC reference).
    prev_iter_vec           : eigenvector selected at the previous PK iteration
                              within the current velocity step (intra-velocity MAC
                              reference).  Takes priority over prev_step_vec when
                              both are available, because it is more up-to-date.
    p_target                : predicted eigenvalue from last 3 converged p values
                              (used as fallback when MAC < 0.7)

    Returns
    -------
    p_sel, u_sel, j_sel : selected eigenvalue, eigenvector, index (or None, None, None)
    """
    omegas = np.abs(np.imag(vals))

    # ------------------------------------------------------------------ #
    # MAC reference vector priority:                                       #
    #  1. prev_iter_vec  — same velocity, previous PK iteration           #
    #     (intra-velocity tracking, most up-to-date shape).               #
    #  2. prev_step_vec  — previous velocity step, same mode              #
    #     (cross-velocity tracking, available from 2nd speed onward).     #
    #  3. None           — first velocity step, first iteration:          #
    #     MAC is disabled and the window falls back to structural freq.   #
    # ------------------------------------------------------------------ #
    #if j == 0 and s == 0:

    if prev_iter_vec is not None:
        u_ref = prev_iter_vec
        ref_src = "prev_iter"
    elif prev_step_vec is not None:
        u_ref = prev_step_vec
        ref_src = "prev_step"
    else:
        u_ref = None
        ref_src = "none"

    # ------------------------------------------------------------------ #
    # MAC pre-selector: compute MAC for ALL candidates.                   #
    # The best-MAC candidate's frequency becomes the window centre        #
    # (mac_presel_omega) when u_ref is available.                         #
    # ------------------------------------------------------------------ #
    mac_presel_omega = None
    mac_presel_idx   = None
    mac_presel_val   = 0.0

    if u_ref is not None and len(omegas) > 0:
        mac_all = np.array([compute_mac(vecs[:, idx], u_ref) for idx in range(len(omegas))])
        mac_presel_idx = int(np.argmax(mac_all))
        mac_presel_val = mac_all[mac_presel_idx]
        mac_presel_omega = omegas[mac_presel_idx]
        print(f"      MAC pre-selector (ref={ref_src}): best MAC={mac_presel_val:.3f} "
              f"at ω={mac_presel_omega:.3f} rad/s (index {mac_presel_idx})")

    # ------------------------------------------------------------------ #
    # Step 1: Establish frequency window centre and bounds.               #
    #                                                                      #
    # Priority (highest to lowest):                                        #
    #  A. MAC pre-selector available (u_ref is not None)                  #
    #     → centre on mac_presel_omega.                                   #
    #  B. last_converged_omegas available (j==0, no MAC yet)              #
    #     → seed from previous velocity step's converged frequency.       #
    #  C. Neither  →  use structural natural frequency ±20%.              #
    # ------------------------------------------------------------------ #
    if mac_presel_omega is not None:
        # A — MAC-driven centre
        omega_ref = mac_presel_omega
        lower_bound = omega_ref * (1 - freq_margin)
        upper_bound = omega_ref * (1 + freq_margin)
        lower_bound = max(lower_bound, 0.0)
        if len(last_converged_omegas) > 0 and s < len(last_converged_omegas):
            print(f"      Window centre from MAC: ω={omega_ref:.3f} rad/s "
                  f"(prev-step ref={last_converged_omegas[s]:.3f} rad/s)")
        else:
            print(f"      Window centre from MAC: ω={omega_ref:.3f} rad/s")
        print(f"      Frequency bounds: [{lower_bound:.3f}, {upper_bound:.3f}] rad/s")

        # Inter-mode separation constraints from already-converged modes at this velocity
        if len(converged_omegas_at_V) > 0:
            if s > 0 and s - 1 < len(converged_omegas_at_V):
                lower_bound = max(lower_bound,
                                  converged_omegas_at_V[s - 1] * (1 + freq_margin))
            if s < len(converged_omegas_at_V):
                upper_bound = min(upper_bound,
                                  converged_omegas_at_V[s] * (1 + 2 * freq_margin))

    elif len(last_converged_omegas) > 0 and s < len(last_converged_omegas):
        # B — seed from previous velocity step (j==0, MAC not yet available)
        omega_ref = last_converged_omegas[s]
        print(f"      Initial seed from previous step: {omega_ref:.3f} rad/s (no MAC ref yet)")
        lower_bound = omega_ref * (1 - freq_margin)
        upper_bound = omega_ref * (1 + freq_margin)
        lower_bound = max(lower_bound, 0.0)
        print(f"      Frequency bounds: [{lower_bound:.3f}, {upper_bound:.3f}] rad/s")

        if len(converged_omegas_at_V) > 0:
            if s > 0 and s - 1 < len(converged_omegas_at_V):
                lower_bound = max(lower_bound,
                                  converged_omegas_at_V[s - 1] * (1 + freq_margin))
            if s < len(converged_omegas_at_V):
                upper_bound = min(upper_bound,
                                  converged_omegas_at_V[s] * (1 + 2 * freq_margin))
    else:
        # C — very first velocity step, j==0, no history at all
        margin_factor = 0.2  # ±20%
        omega_ref = omega_n[s]
        lower_bound = omega_ref * (1 - margin_factor)
        upper_bound = omega_ref * (1 + margin_factor)
        print(f"      First velocity step: using structural frequency {omega_ref:.3f} rad/s")
        print(f"      Frequency bounds: [{lower_bound:.3f}, {upper_bound:.3f}] rad/s")

    bound_mask = (omegas >= lower_bound) & (omegas <= upper_bound)

    # ------------------------------------------------------------------ #
    # Expansion: when the mode is just outside the window, widen it.     #
    # ------------------------------------------------------------------ #
    if not np.any(bound_mask):
        omega_center = (lower_bound + upper_bound) / 2.0
        closest_idx = int(np.argmin(np.abs(omegas - omega_center)))
        closest_omega = omegas[closest_idx]

        if closest_omega < lower_bound:
            distance_outside = (lower_bound - closest_omega) / max(lower_bound, 1e-12)
            direction = "below"
        else:
            distance_outside = (closest_omega - upper_bound) / max(upper_bound, 1e-12)
            direction = "above"

        expansion_threshold = 0.2
        if distance_outside < expansion_threshold:
            print(f"      ⚠ MODE {s} just outside bounds [{lower_bound:.3f}, {upper_bound:.3f}] rad/s")
            print(f"         Closest mode at {closest_omega:.3f} rad/s ({direction}, {distance_outside*100:.1f}% outside)")
            print(f"         → Expanding search range by {expansion_threshold*100:.0f}% and retrying")
            lower_bound = lower_bound * (1 - expansion_threshold)
            upper_bound = upper_bound * (1 + expansion_threshold)
            bound_mask = (omegas >= lower_bound) & (omegas <= upper_bound)

            if not np.any(bound_mask):
                print(f"      ✗ MODE {s} LOST even after expansion: No modes in bounds [{lower_bound:.3f}, {upper_bound:.3f}] rad/s")
                print(f"         Available modes: {omegas[:5]}... (first 5)")
                mode_switch_warnings_ref[0] += 1
                return None, None, None
            else:
                print(f"         ✓ Found mode after expansion in [{lower_bound:.3f}, {upper_bound:.3f}] rad/s")
        else:
            print(f"      ⚠ MODE {s} LOST: No modes in bounds [{lower_bound:.3f}, {upper_bound:.3f}] rad/s")
            print(f"         Available modes: {omegas[:5]}... (first 5)")
            print(f"         Closest mode at {closest_omega:.3f} rad/s ({direction}, {distance_outside*100:.1f}% outside)")
            mode_switch_warnings_ref[0] += 1
            return None, None, None

    # ------------------------------------------------------------------ #
    # Step 2: Score candidates inside the (possibly expanded) window.    #
    #                                                                      #
    # When u_ref is available: combined MAC + frequency score.            #
    # Otherwise: pure frequency proximity to omega_ref.                   #
    # ------------------------------------------------------------------ #
    valid_indices = np.where(bound_mask)[0]

    if u_ref is not None:
        # MAC-driven scoring: minimise w_freq * freq_err + w_mac * (1 - MAC)
        # omega_target = mac_presel_omega (already the best-MAC frequency)
        omega_target = mac_presel_omega if mac_presel_omega is not None else omega_ref
        mac_all_valid = np.array([compute_mac(vecs[:, idx], u_ref) for idx in valid_indices])
        freq_errs     = np.array([abs(omegas[idx] - omega_target) / max(omega_target, 1e-6)
                                   for idx in valid_indices])
        scores = w_freq * freq_errs + w_mac * (1.0 - mac_all_valid)

        best_local = int(np.argmin(scores))
        j_sel = valid_indices[best_local]

        mac_val_sel  = mac_all_valid[best_local]
        freq_err_sel = freq_errs[best_local]
        score_sel    = scores[best_local]
        
        # MAC constraint: if MAC < 0.7, fall back to p_target prediction
        if mac_val_sel < 0.7:
            if p_target is not None:
                print(f"      ⚠ Low MAC detected ({mac_val_sel:.3f} < 0.7)")
                print(f"         → Switching to p_target-based selection: p_target={p_target:.6f}")
                
                # Find candidate closest to p_target
                p_errors = np.array([abs(vals[idx] - p_target) for idx in valid_indices])
                best_p_local = int(np.argmin(p_errors))
                j_sel_p = valid_indices[best_p_local]
                
                # Calculate MAC for p_target-selected candidate
                mac_val_p = mac_all_valid[best_p_local]
                p_error = p_errors[best_p_local]
                
                print(f"         p_target candidate: ω={omegas[j_sel_p]:.3f} rad/s, "
                      f"MAC={mac_val_p:.3f}, |p-p_target|={p_error:.3e}")
                
                # Use p_target selection
                j_sel = j_sel_p
                mac_val_sel = mac_val_p
                freq_err_sel = abs(omegas[j_sel] - np.abs(np.imag(p_target))) / max(np.abs(np.imag(p_target)), 1e-6)
                score_sel = p_error
                
                print(f"      ✓ Selected (p_target): ω={omegas[j_sel]:.3f} rad/s, "
                      f"MAC={mac_val_sel:.3f}, p_error={p_error:.3e}")
            else:
                # No p_target available, but MAC is low - print warning and continue with MAC
                print(f"      ⚠ Low MAC detected ({mac_val_sel:.3f} < 0.7) but no p_target available")
                print(f"         → Continuing with MAC-based selection (no better option)")
                print(f"      Selected (low MAC): ω={omegas[j_sel]:.3f} rad/s, "
                      f"MAC={mac_val_sel:.3f} (ref={ref_src}), "
                      f"freq_err={freq_err_sel:.3e}, score={score_sel:.3e}")
        else:
            print(f"      Selected: ω={omegas[j_sel]:.3f} rad/s, "
                  f"MAC={mac_val_sel:.3f} (ref={ref_src}), "
                  f"freq_err={freq_err_sel:.3e}, score={score_sel:.3e}")
    else:
        # No MAC reference — pure frequency proximity to omega_ref
        errors = np.abs(omegas[valid_indices] - omega_ref)
        best_local = int(np.argmin(errors))
        j_sel = valid_indices[best_local]
        print(f"      Selected (freq-only): ω={omegas[j_sel]:.3f} rad/s "
              f"(target={omega_ref:.3f} rad/s)")

    return vals[j_sel], vecs[:, j_sel], j_sel


def compute_mac(u1, u2):
    """Modal Assurance Criterion between two complex vectors."""
    if u1 is None or u2 is None:
        return 0.0

    numerator = np.abs(u1.conj().T @ u2) ** 2
    denominator = (u1.conj().T @ u1) * (u2.conj().T @ u2)

    if np.abs(denominator) < 1e-12:
        return 0.0

    return np.real(numerator / denominator)