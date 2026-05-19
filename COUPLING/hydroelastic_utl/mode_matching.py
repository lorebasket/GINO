import numpy as np

# MAC pre-selector treated as perfect at 1.0 within this tolerance.
_PERFECT_MAC_ATOL = 1e-6


def _apply_inter_mode_separation(lower_bound, upper_bound, s, converged_omegas_at_V,
                                 freq_margin):
    """Tighten frequency window using already-converged modes at the current V."""
    if s > 0 and s - 1 < len(converged_omegas_at_V):
        lower_bound = max(lower_bound,
                          converged_omegas_at_V[s - 1] * (1 + freq_margin))
    if s < len(converged_omegas_at_V):
        upper_bound = min(upper_bound,
                          converged_omegas_at_V[s] * (1 + 2 * freq_margin))
    return lower_bound, upper_bound


def _unclaimed_root_mask(
    omegas,
    converged_omegas_at_V,
    claimed_indices=None,
    freq_margin=0.05,
    min_omega_sep_rad=1.0,
):
    """
    Boolean mask of wet roots still available for structural slot ``s``.

    A root is excluded if:
    - its spectrum index was already claimed by a lower slot at this V, or
    - its frequency lies within ``max(freq_margin * ω_conv, min_omega_sep_rad)`` of any
      converged mode at this airspeed (prevents coalescence onto the same branch).
    """
    omegas = np.asarray(omegas, dtype=float).ravel()
    n = len(omegas)
    mask = np.ones(n, dtype=bool)

    if claimed_indices:
        for j in claimed_indices:
            ji = int(j)
            if 0 <= ji < n:
                mask[ji] = False

    # When lower slots already claimed spectrum indices, only those indices are
    # removed — allows physical frequency crossing (roots within ~1 rad/s).
    if claimed_indices:
        return mask

    margin = float(freq_margin)
    min_sep = float(min_omega_sep_rad)
    for om_c in converged_omegas_at_V or []:
        if not np.isfinite(om_c) or float(om_c) <= 0.0:
            continue
        om_c = float(om_c)
        thresh = max(margin * om_c, min_sep)
        mask &= np.abs(omegas - om_c) >= thresh

    return mask


def _tracking_omega_target(s, last_converged_omegas, mac_presel_omega, p_target):
    """Frequency target for scoring: history trend preferred over MAC at crossover."""
    if len(last_converged_omegas) > s and np.isfinite(last_converged_omegas[s]):
        return float(last_converged_omegas[s])
    if p_target is not None and np.isfinite(np.abs(p_target)):
        return float(np.abs(np.imag(p_target)))
    if mac_presel_omega is not None and np.isfinite(mac_presel_omega):
        return float(mac_presel_omega)
    return None


def select_pk_mode_initial_lock(
    vals,
    vecs,
    omegas,
    s,
    omega_n,
    converged_omegas_at_V,
    converged_p_at_V,
    freq_margin=0.05,
    p_target=None,
    claimed_indices=None,
    min_omega_sep_rad=1.0,
):
    """
    First PK iteration root pick (Yuan & Zhang 2023, Sec. 3.2.2).

    For ``s > 0`` at the current airspeed the search window is the intersection of:

    - **Inter-mode separation**: ω above the already converged lower mode(s) at this V;
    - **Structural band**: ω within ±``freq_margin`` of dry ω_n[s].

    The root is the wet eigenvalue in that window closest to ω_struct[s] (the other
    physical branch), not the one whose ``p`` is closest to mode ``s-1`` — that would
    re-select the bending-like root (~70 rad/s) instead of torsion (~116 rad/s).

  Note: this step does **not** use MAC; MAC applies from iteration 1 onward via
    ``mode_matching_paper_compliant`` when a reference eigenvector exists.
    """
    omegas = np.asarray(omegas, dtype=float).ravel()
    n = len(omegas)
    if n == 0:
        return None, None, None

    margin = float(freq_margin)
    omega_struct = float(omega_n[s]) if s < len(omega_n) else float(omegas[0])

    if s > 0 and len(converged_omegas_at_V) >= s:
        # Inter-mode: strictly above lower neighbour at this V (Paper 3.2.2, first bound)
        omega_lo_inter = float(converged_omegas_at_V[s - 1]) * (1.0 + margin)
        omega_hi_inter = np.inf
        if s < len(converged_omegas_at_V) and np.isfinite(converged_omegas_at_V[s]):
            omega_hi_inter = float(converged_omegas_at_V[s]) * (1.0 + 2.0 * margin)

        # Structural band for slot s (identifies bending vs torsion branch)
        omega_lo_struct = omega_struct * (1.0 - margin)
        omega_hi_struct = omega_struct * (1.0 + margin)

        lower_bound = max(omega_lo_inter, omega_lo_struct)
        upper_bound = omega_hi_struct
        if np.isfinite(omega_hi_inter):
            upper_bound = min(upper_bound, omega_hi_inter)

        mask = (omegas >= lower_bound) & (omegas <= upper_bound)
        mask &= _unclaimed_root_mask(
            omegas,
            converged_omegas_at_V,
            claimed_indices=claimed_indices,
            freq_margin=margin,
            min_omega_sep_rad=min_omega_sep_rad,
        )

        if not np.any(mask):
            expand = margin
            for _ in range(3):
                expand += 0.05
                lower_bound = max(omega_lo_inter, omega_struct * (1.0 - expand))
                upper_bound = omega_struct * (1.0 + expand)
                mask = (omegas >= lower_bound) & (omegas <= upper_bound)
                mask &= _unclaimed_root_mask(
                    omegas,
                    converged_omegas_at_V,
                    claimed_indices=claimed_indices,
                    freq_margin=expand,
                    min_omega_sep_rad=min_omega_sep_rad,
                )
                if np.any(mask):
                    print(
                        f"      PK initial lock: expanded structural/inter-mode window to "
                        f"[{lower_bound:.3f}, {upper_bound:.3f}] rad/s"
                    )
                    break

        valid = np.where(mask)[0] if np.any(mask) else np.arange(n)
        om_candidates = [float(omegas[i]) for i in valid]

        if p_target is not None and np.isfinite(np.abs(p_target)):
            omega_tgt = float(np.abs(np.imag(p_target)))
            scores = np.array(
                [
                    abs(omegas[idx] - omega_struct) / max(omega_struct, 1e-6)
                    + 0.25 * abs(omegas[idx] - omega_tgt) / max(omega_tgt, 1e-6)
                    for idx in valid
                ]
            )
            j_sel = int(valid[int(np.argmin(scores))])
            criterion = "nearest ω_struct (+ p_target tie-break)"
        else:
            f_errs = np.abs(omegas[valid] - omega_struct)
            j_sel = int(valid[int(np.argmin(f_errs))])
            criterion = "nearest ω_struct"

        print(
            f"      PK initial lock (s={s}, j=0): "
            f"inter-mode ω>{omega_lo_inter:.3f} (mode {s-1} conv), "
            f"structural band [{omega_lo_struct:.3f}, {omega_hi_struct:.3f}], "
            f"window=[{lower_bound:.3f}, {upper_bound:.3f}], "
            f"candidates={om_candidates} → pick ω={omegas[j_sel]:.3f} ({criterion})"
        )
        return vals[j_sel], vecs[:, j_sel], j_sel

    freq_errors = np.abs(omegas - omega_struct)
    j_sel = int(np.argmin(freq_errors))
    rel = freq_errors[j_sel] / max(omega_struct, 1e-6)
    if rel > 0.3:
        print(
            f"      ⚠ Mode {s}: ω={omegas[j_sel]:.3f} rad/s is {rel * 100:.1f}% "
            f"from ω_struct={omega_struct:.3f} rad/s"
        )
    else:
        print(
            f"      Structural seed: ω_struct={omega_struct:.3f} → "
            f"ω={omegas[j_sel]:.3f} rad/s (index {j_sel})"
        )
    return vals[j_sel], vecs[:, j_sel], j_sel


def predict_and_select_reorder_modes(
    res_V,
    results,
    velocity_index,
    delta=0.025,
    min_omega_sep_rad=1.0,
):
    """
    Predict-and-select reordering at one airspeed (Yuan & Zhang 2023, Sec. 3.2.1 / Fig. 2).

    After all modes are solved at ``V_i``, reassign slot ``s`` to the available root whose
    frequency is closest to ``Im(p_ext)`` with ``p_ext = 2 p^{(i-1)} - p^{(i-2)}``.
    Each wet root is used at most once so branches may cross without coalescing.
    """
    if velocity_index < 2 or not res_V:
        return res_V

    prev_modes = results[velocity_index - 1].get("modes", [])
    prev2_modes = results[velocity_index - 2].get("modes", [])
    if not prev_modes or not prev2_modes:
        return res_V

    n_modes = len(res_V)
    available = list(range(n_modes))
    reordered = []
    min_sep = float(min_omega_sep_rad)

    def _too_close(omega_a, omega_b):
        if not (np.isfinite(omega_a) and np.isfinite(omega_b)):
            return False
        return abs(float(omega_a) - float(omega_b)) < min_sep

    for s in range(n_modes):
        if not available:
            src = res_V[min(s, len(res_V) - 1)]
            reordered.append({**src, "mode": int(s)})
            continue

        if s >= len(prev_modes) or s >= len(prev2_modes):
            pick = available[0]
            reordered.append({**res_V[pick], "mode": int(s)})
            available.remove(pick)
            continue

        p_v1 = prev_modes[s].get("p")
        p_v2 = prev2_modes[s].get("p")
        if p_v1 is None or p_v2 is None or np.isnan(p_v1) or np.isnan(p_v2):
            pick = available[0]
            reordered.append({**res_V[pick], "mode": int(s)})
            available.remove(pick)
            continue

        p_ext = 2.0 * complex(p_v1) - complex(p_v2)
        omega_ext = float(np.abs(np.imag(p_ext)))

        pick = min(
            available,
            key=lambda r: abs(float(res_V[r].get("omega", np.nan)) - omega_ext),
        )
        omega_pick = float(res_V[pick].get("omega", np.nan))
        for prev in reordered:
            if _too_close(omega_pick, prev.get("omega", np.nan)):
                others = [r for r in available if r != pick]
                if others:
                    pick = min(
                        others,
                        key=lambda r: abs(float(res_V[r].get("omega", np.nan)) - omega_ext),
                    )
                    omega_pick = float(res_V[pick].get("omega", np.nan))
                break

        reordered.append({**res_V[pick], "mode": int(s)})
        available.remove(pick)

    if reordered:
        omegas_out = [float(m.get("omega", np.nan)) for m in reordered]
        print(
            f"      Predict-and-select at V step {velocity_index}: "
            f"ω slots → {[f'{w:.3f}' for w in omegas_out]} rad/s"
        )

    return reordered


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
                                   p_target=None,
                                   skip_inter_mode_on_perfect_mac=False,
                                   claimed_indices=None,
                                   min_omega_sep_rad=1.0):
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
    skip_inter_mode_on_perfect_mac : if True, do not apply inter-mode separation
                              when the MAC pre-selector reports MAC = 1.0 (allows
                              crossed wet natural frequencies to be tracked).

    Returns
    -------
    p_sel, u_sel, j_sel : selected eigenvalue, eigenvector, index (or None, None, None)
    """
    omegas = np.abs(np.imag(vals))
    unclaimed = _unclaimed_root_mask(
        omegas,
        converged_omegas_at_V,
        claimed_indices=claimed_indices,
        freq_margin=freq_margin,
        min_omega_sep_rad=min_omega_sep_rad,
    )
    if not np.any(unclaimed):
        print(
            f"      ⚠ MODE {s}: no unclaimed wet roots "
            f"(claimed idx={claimed_indices}, conv ω={converged_omegas_at_V})"
        )
        mode_switch_warnings_ref[0] += 1
        return None, None, None

    if claimed_indices or (converged_omegas_at_V and s > 0):
        excluded = [float(omegas[i]) for i in range(len(omegas)) if not unclaimed[i]]
        if excluded:
            print(
                f"      Unclaimed-root filter (s={s}): exclude ω={excluded} rad/s "
                f"(already assigned at this V)"
            )

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
        pool = np.where(unclaimed)[0]
        mac_all = np.array([compute_mac(vecs[:, idx], u_ref) for idx in pool])
        mac_presel_idx = int(pool[int(np.argmax(mac_all))])
        mac_presel_val = float(np.max(mac_all))
        mac_presel_omega = omegas[mac_presel_idx]
        print(f"      MAC pre-selector (ref={ref_src}): best MAC={mac_presel_val:.3f} "
              f"at ω={mac_presel_omega:.3f} rad/s (index {mac_presel_idx}, unclaimed only)")

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

        if len(converged_omegas_at_V) > 0:
            perfect_mac = mac_presel_val >= 1.0 - _PERFECT_MAC_ATOL
            if skip_inter_mode_on_perfect_mac and perfect_mac:
                print("      Inter-mode separation skipped (MAC pre-selector = 1.0)")
            else:
                lower_bound, upper_bound = _apply_inter_mode_separation(
                    lower_bound, upper_bound, s, converged_omegas_at_V, freq_margin)

    elif len(last_converged_omegas) > 0 and s < len(last_converged_omegas):
        # B — seed from previous velocity step (j==0, MAC not yet available)
        omega_ref = last_converged_omegas[s]
        print(f"      Initial seed from previous step: {omega_ref:.3f} rad/s (no MAC ref yet)")
        lower_bound = omega_ref * (1 - freq_margin)
        upper_bound = omega_ref * (1 + freq_margin)
        lower_bound = max(lower_bound, 0.0)
        print(f"      Frequency bounds: [{lower_bound:.3f}, {upper_bound:.3f}] rad/s")

        if len(converged_omegas_at_V) > 0:
            lower_bound, upper_bound = _apply_inter_mode_separation(
                lower_bound, upper_bound, s, converged_omegas_at_V, freq_margin)
    else:
        # C — very first velocity step, j==0, no history at all
        margin_factor = 0.2  # ±20%
        omega_ref = omega_n[s]
        lower_bound = omega_ref * (1 - margin_factor)
        upper_bound = omega_ref * (1 + margin_factor)
        print(f"      First velocity step: using structural frequency {omega_ref:.3f} rad/s")
        print(f"      Frequency bounds: [{lower_bound:.3f}, {upper_bound:.3f}] rad/s")

    bound_mask = (omegas >= lower_bound) & (omegas <= upper_bound) & unclaimed

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
            bound_mask = (omegas >= lower_bound) & (omegas <= upper_bound) & unclaimed

            if not np.any(bound_mask):
                pool = np.where(unclaimed)[0]
                omega_fb = _tracking_omega_target(
                    s, last_converged_omegas, mac_presel_omega, p_target
                )
                if pool.size > 0 and omega_fb is not None:
                    j_fb = int(pool[int(np.argmin(np.abs(omegas[pool] - omega_fb)))])
                    print(
                        f"      Fallback (unclaimed + ω trend): ω={omegas[j_fb]:.3f} rad/s "
                        f"(target={omega_fb:.3f})"
                    )
                    return vals[j_fb], vecs[:, j_fb], j_fb
                print(f"      ✗ MODE {s} LOST even after expansion: No modes in bounds [{lower_bound:.3f}, {upper_bound:.3f}] rad/s")
                print(f"         Available modes: {omegas[:5]}... (first 5)")
                mode_switch_warnings_ref[0] += 1
                return None, None, None
            else:
                print(f"         ✓ Found mode after expansion in [{lower_bound:.3f}, {upper_bound:.3f}] rad/s")
        else:
            pool = np.where(unclaimed)[0]
            omega_fb = _tracking_omega_target(
                s, last_converged_omegas, mac_presel_omega, p_target
            )
            if pool.size > 0 and omega_fb is not None:
                j_fb = int(pool[int(np.argmin(np.abs(omegas[pool] - omega_fb)))])
                print(
                    f"      Fallback (unclaimed + ω trend): ω={omegas[j_fb]:.3f} rad/s "
                    f"(target={omega_fb:.3f})"
                )
                return vals[j_fb], vecs[:, j_fb], j_fb
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
        # At crossover, track ω from history rather than MAC on the lower branch.
        omega_target = _tracking_omega_target(
            s, last_converged_omegas, mac_presel_omega, p_target
        )
        if omega_target is None:
            omega_target = mac_presel_omega if mac_presel_omega is not None else omega_ref
        elif s > 0 and len(last_converged_omegas) > s:
            print(
                f"      Frequency trend target: ω_track={omega_target:.3f} rad/s "
                f"(prev-step slot {s})"
            )
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
        # No MAC reference — pure frequency proximity to trend / ref
        omega_target = _tracking_omega_target(
            s, last_converged_omegas, mac_presel_omega, p_target
        )
        if omega_target is None:
            omega_target = omega_ref
        errors = np.abs(omegas[valid_indices] - omega_target)
        best_local = int(np.argmin(errors))
        j_sel = valid_indices[best_local]
        print(f"      Selected (freq-only): ω={omegas[j_sel]:.3f} rad/s "
              f"(target={omega_target:.3f} rad/s)")

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