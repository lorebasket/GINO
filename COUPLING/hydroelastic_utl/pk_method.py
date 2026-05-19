"""
Physics-agnostic p-k iteration engine.

Depends only on numpy/scipy and mode_matching — never import aero_system or hydro_system.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import eig

from . import mode_matching


class PkDiagnostics:
    """Optional container mirroring PKSolverV3 diagnostic lists for post-processing."""

    def __init__(self):
        self.contributions_history = []
        self.w_modal_history = []
        self.converged_wet_spectrum_history = []
        self.mode_switch_warnings = 0
        self.convergence_issues = 0
        self.extrapolation_failures = 0
        self.iteration_log = []


def initial_k_for_mode(s, V, results_for_V, omega_n, k_guess, fXK0=0.618):
    """
    Hybrid reduced-frequency initial guess (Yuan & Zhang 2023, Eq. 13–15).

    Uses the converged frequency of mode ``s-1`` at the current airspeed, optionally
    blended with mode ``s-2`` — not a sorted eigenvalue from the current spectrum.
    """
    V = max(float(V), 1e-12)
    if s == 0:
        return max(float(k_guess), 1e-6)

    if len(results_for_V) < s:
        return max(float(omega_n[s]) / V, float(k_guess), 1e-6)

    om_ssm1 = results_for_V[s - 1].get("omega", np.nan)
    if not np.isfinite(om_ssm1) or om_ssm1 <= 0.0:
        return max(float(omega_n[s]) / V, float(k_guess), 1e-6)

    k_ssm1 = float(om_ssm1) / V
    if s >= 2 and len(results_for_V) >= s - 1:
        om_sm1sm1 = results_for_V[s - 2].get("omega", np.nan)
        if np.isfinite(om_sm1sm1) and om_sm1sm1 > 0.0:
            k_sm1sm1 = float(om_sm1sm1) / V
            k_hybrid = k_ssm1 + fXK0 * (k_sm1sm1 - k_ssm1)
            return max(k_hybrid, 1e-6)

    return max(k_ssm1, 1e-6)


def hybrid_k_guess(s, V, omega_s_prev_iter, omega_s_minus_1_converged, k_guess, fXK0=0.618):
    """Deprecated alias: prefer :func:`initial_k_for_mode` for mode onset."""
    if s == 0:
        return k_guess

    if np.isnan(omega_s_minus_1_converged):
        print("      Note: Previous mode lost, using k_base without hybridization")
        return max(omega_s_prev_iter / V, 1e-6)

    k_base = omega_s_prev_iter / V
    k_converged = omega_s_minus_1_converged / V
    k_hybrid = k_base + fXK0 * (k_converged - k_base)

    if k_hybrid < 1e-6:
        print(f"      Warning: k_hybrid too small ({k_hybrid:.2e}), using {k_guess:.2e}")
        k_hybrid = k_guess

    return k_hybrid


def relax_k(k_old, omega_new, V, fRLX=0.618):
    print(f"      Relaxing k: old={k_old:.6g}, new={omega_new/V:.6g}, fRLX={fRLX}")
    k_target = omega_new / V
    k_new = k_old + fRLX * (k_target - k_old)
    k_new = max(k_new, 1e-6)
    return k_new


def predict_eigenvalue_linear(s, results_i_minus_1, results_i_minus_2):
    if results_i_minus_1 is None or results_i_minus_2 is None:
        return None, False

    if s >= len(results_i_minus_1["modes"]) or s >= len(results_i_minus_2["modes"]):
        return None, False

    p_v1 = results_i_minus_1["modes"][s]["p"]
    p_v2 = results_i_minus_2["modes"][s]["p"]

    if np.isnan(p_v1) or np.isnan(p_v2):
        return None, False

    p_ext = 2.0 * p_v1 - p_v2
    return p_ext, True


def check_frequency_continuity(results, max_jump=0.30):
    problems = []

    for i in range(1, len(results)):
        V_curr = results[i]["V"]
        modes_prev = results[i - 1]["modes"]
        modes_curr = results[i]["modes"]

        for s in range(min(len(modes_prev), len(modes_curr))):
            omega_prev = modes_prev[s]["omega"]
            omega_curr = modes_curr[s]["omega"]

            if omega_prev < 1e-6:
                continue

            rel_change = abs(omega_curr - omega_prev) / omega_prev

            if rel_change > max_jump:
                problems.append(
                    {
                        "velocity": V_curr,
                        "mode": s,
                        "omega_prev": omega_prev,
                        "omega_curr": omega_curr,
                        "rel_change": rel_change,
                    }
                )

    return problems


def normalize_and_fix_sign(u, M, reference=None):
    n_half = len(u) // 2
    u_h = u[:n_half]

    norm_factor = np.sqrt(np.abs(u_h.conj().T @ M @ u_h))
    if norm_factor < 1e-12:
        norm_factor = np.linalg.norm(u)

    u_norm = u / norm_factor

    if reference is not None:
        dot_product = np.real(u_norm.conj().T @ reference)
        if dot_product < 0:
            u_norm = -u_norm
    else:
        significant_idx = int(np.argmax(np.abs(u_norm[:n_half])))
        if np.real(u_norm[significant_idx]) < 0:
            u_norm = -u_norm

    return u_norm


def build_wet_spectrum_summary(vals, omega_n, results_for_V, freq_margin):
    """
    Classify sorted wet roots at PK convergence.

    Roots matched to a converged PK branch (``pk_tracked``) are excluded from
    ``untracked``. The lowest-frequency untracked root is the usual orphan branch
    that never enters the structural mode-matching window.
    """
    vals = np.asarray(vals, dtype=complex).ravel()
    omega_n = np.asarray(omega_n, dtype=float).ravel()
    margin = float(freq_margin)

    tracked_omegas = []
    for md in results_for_V:
        if not md.get("converged", False):
            continue
        om = md.get("omega", np.nan)
        if np.isfinite(om) and om > 0.0:
            tracked_omegas.append(float(om))

    roots = []
    for idx, lam in enumerate(vals):
        omega = float(np.abs(np.imag(lam)))
        if omega < 1e-8:
            continue
        sigma = float(np.real(lam))
        gamma = sigma / omega if omega > 1e-6 else 0.0

        pk_tracked = False
        for om_t in tracked_omegas:
            if abs(omega - om_t) <= margin * max(om_t, 1e-6):
                pk_tracked = True
                break

        struct_near = None
        for si, om_s in enumerate(omega_n):
            if abs(omega - om_s) <= margin * max(om_s, 1e-6):
                struct_near = int(si)
                break

        if pk_tracked:
            kind = "pk_tracked"
        elif struct_near is not None:
            kind = "structural_near"
        else:
            kind = "untracked"

        roots.append(
            {
                "index": int(idx),
                "p": complex(lam),
                "omega": omega,
                "sigma": sigma,
                "gamma": float(gamma),
                "kind": kind,
                "structural_near_mode": struct_near,
            }
        )

    untracked = [r for r in roots if r["kind"] == "untracked"]
    lowest_untracked = min(untracked, key=lambda r: r["omega"]) if untracked else None

    return {
        "roots": roots,
        "untracked": untracked,
        "lowest_untracked": lowest_untracked,
        "n_roots": len(roots),
        "n_untracked": len(untracked),
        "omega_structural": [float(w) for w in omega_n],
        "freq_margin": margin,
    }


def pk_solve_at_velocity(
    i,
    V,
    V_list,
    modes,
    omega_n,
    build_matrix_fn,
    k0_guess,
    k_guess,
    tol,
    fXK0,
    fRLX,
    freq_margin,
    w_freq,
    w_mac,
    max_iter,
    results,
    dry_vals,
    dry_vecs,
    perturb_k,
    mac_matching,
    last_converged_mode_matching,
    matrix_iter_state=None,
    on_converge_fn=None,
    diagnostics=None,
    last_converged_omegas_override=None,
    prev_vecs_matrix=None,
    prev_eigs_row=None,
    eigen_positive_imag_only=False,
    skip_inter_mode_on_perfect_mac=False,
    use_k_guess_each_iter=False,
    min_root_sep_rad=1.0,
):
    """
    Single-airspeed p-k solve for all modes in ``modes``.

    ``omega_n`` : (Nm,) structural natural frequencies [rad/s], same semantics as
    ``PKSolverV3``'s ``self.omega_n`` (i.e. sqrt of dry eigenvalues if those are ω²).

    ``build_matrix_fn(k, V)`` returns the state matrix for the current iterate.
    When ``matrix_iter_state`` is provided, the caller must set keys ``p``, ``s``, ``j``
    before each call (the engine updates them each iteration).

    ``on_converge_fn(k, V, s, p_sel, u_sel)`` optional; returns a dict merged into the
    converged mode result (e.g. contributions / w_modal).
    """
    results_for_V = []
    last_sorted_omegas = []
    converged_omegas_at_V = []
    converged_vecs_at_V = []
    converged_p_at_V = []
    converged_wet_indices_at_V = []

    if last_converged_omegas_override is not None:
        last_converged_omegas = list(last_converged_omegas_override)
    else:
        last_converged_omegas = []
        if i > 0 and len(results) > 0 and "modes" in results[i - 1]:
            for s in modes:
                if s < len(results[i - 1]["modes"]):
                    last_converged_omegas.append(results[i - 1]["modes"][s]["omega"])
                else:
                    last_converged_omegas.append(omega_n[s])

    print(f"\n  Solving at V = {V:.2f} m/s (fXK0={fXK0}, fRLX={fRLX})")

    for s in modes:
        print(f"\n  === MODE {s} ===")

        if i > 0 and s < len(results[i - 1]["modes"]):
            prev_mode_data = results[i - 1]["modes"][s]
            if prev_mode_data.get("mode_lost", False):
                print(f"    ⊗ MODE {s} already lost at previous velocity step")
                print("       → Skipping (inserting NaN)")

                results_for_V.append(
                    {
                        "mode": int(s),
                        "p": np.nan,
                        "omega": np.nan,
                        "sigma": np.nan,
                        "gamma": np.nan,
                        "u": None,
                        "k": np.nan,
                        "vecs": None,
                        "converged": False,
                        "iterations": 0,
                        "mode_lost": True,
                    }
                )
                continue

        if s == 0:
            if i == 0:
                k = k_guess
                print(f"    k_initial = {k:.6g} (from structural ω_n[0]={omega_n[0]:.3f} rad/s)")
            else:
                k = k0_guess
                print(f"    k_initial = {k:.6g} (from previous velocity step)")
        else:
            k = initial_k_for_mode(s, V, results_for_V, omega_n, k_guess, fXK0)
            om_ref = results_for_V[s - 1]["omega"] if len(results_for_V) >= s else omega_n[s - 1]
            print(
                f"    k_initial = {k:.6g} (Yuan & Zhang Eq. 15, ω_{{s-1}}^conv={om_ref:.3f} rad/s)"
            )

        p_ext = None
        omega_ext = None
        use_extrapolation = False

        if i >= 2:
            p_ext, success = predict_eigenvalue_linear(s, results[i - 1], results[i - 2])
            if success:
                omega_ext = np.abs(np.imag(p_ext))
                use_extrapolation = True
                print(f"    Using linear extrapolation: p_ext = {p_ext:.6f}, ω_ext = {omega_ext:.3f} rad/s")
            else:
                if diagnostics is not None:
                    diagnostics.extrapolation_failures += 1
                print("    Extrapolation failed, using previous velocity only")

        if not use_extrapolation and i > 0 and s < len(results[i - 1]["modes"]):
            p_ext = results[i - 1]["modes"][s]["p"]
            omega_ext = np.abs(np.imag(p_ext))

        if s == 0:
            p_prev_iter = None
        elif i > 0 and s < len(results[i - 1]["modes"]):
            p_hist = results[i - 1]["modes"][s].get("p", None)
            p_prev_iter = p_hist if p_hist is not None and np.isfinite(np.real(p_hist)) else 1j * omega_n[s]
            print(f"    p_for_Q_guess: same-mode history (V_{{i-1}}), ω={np.abs(np.imag(p_prev_iter)):.3f} rad/s")
        else:
            p_prev_iter = 1j * float(omega_n[s])
            print(f"    p_for_Q_guess: structural ω_n[{s}]={omega_n[s]:.3f} rad/s (j=0)")

        prev_iter_vec = None
        converged = False
        wet_mode_identified = False

        p_sel = None
        j_sel = None
        u_sel = None
        vals = None
        vecs = None
        omega_sel = float("nan")
        sigma_sel = float("nan")
        gamma_sel = float("nan")
        k_new = float("nan")

        for j in range(max_iter):
            if j == 0 or j % 10 == 0:
                print(f"    Iteration {j}")

            try:
                omega_from_k = k * V
            except Exception:
                omega_from_k = float("nan")
            print(f"      [K-DEBUG] Iter {j}: k={k:.6g}, ω_from_k={omega_from_k:.6f} rad/s")

            if matrix_iter_state is not None:
                matrix_iter_state["p"] = p_prev_iter
                matrix_iter_state["s"] = s
                matrix_iter_state["j"] = j

            Ac = build_matrix_fn(k, V)

            vals_all, vecs_all = eig(Ac)

            if eigen_positive_imag_only:
                pos_idx = np.where(np.imag(vals_all) > 1e-8)[0]
            else:
                imag = np.imag(vals_all)
                pos_idx = np.where(imag >= -1e-3)[0]
            vals = vals_all[pos_idx]
            vecs = vecs_all[:, pos_idx]

            omegas = np.abs(np.imag(vals))
            order = np.argsort(omegas)
            vals = vals[order]
            vecs = vecs[:, order]
            omegas = omegas[order]

            omega_phys_min = omega_n[0] * 0.1 if len(omega_n) > 0 else 1.0
            phys_mask = omegas >= omega_phys_min
            n_phys = int(np.sum(phys_mask))
            if n_phys >= len(omega_n):
                n_stripped = len(omegas) - n_phys
                if n_stripped > 0 and j == 0:
                    print(
                        f"      [strip] Removed {n_stripped} near-zero root(s) "
                        f"(ω < {omega_phys_min:.2f} rad/s) from sorted set"
                    )
                vals = vals[phys_mask]
                vecs = vecs[:, phys_mask]
                omegas = omegas[phys_mask]

            last_sorted_omegas = list(omegas)

            if diagnostics is not None:
                diagnostics.iteration_log.append(
                    {
                        "velocity": V,
                        "mode": s,
                        "iteration": j,
                        "k": k,
                        "omegas": omegas.copy(),
                        "n_modes": len(omegas),
                    }
                )

            print(f"      [DEBUG] Wet frequencies at iteration {j}: {omegas[:10]} rad/s (showing first 10)")
            print(f"      [DEBUG] Total number of wet modes: {len(omegas)}")

            mode_freq_margin = freq_margin

            prev_step_vec = None
            if prev_vecs_matrix is not None and s in modes:
                col = modes.index(s)
                if col < prev_vecs_matrix.shape[1]:
                    prev_step_vec = prev_vecs_matrix[:, col]
            if prev_step_vec is None and i > 0 and s < len(results[i - 1]["modes"]):
                prev_step_vec = results[i - 1]["modes"][s].get("u", None)

            if j == 0 and (prev_iter_vec is None and prev_step_vec is None):
                p_target_init = None
                if i >= 2:
                    p_target_init, _ = predict_eigenvalue_linear(s, results[i - 1], results[i - 2])
                    if p_target_init is not None:
                        p_target_init = complex(p_target_init)
                elif i > 0 and s < len(results[i - 1]["modes"]):
                    p_hist = results[i - 1]["modes"][s].get("p", None)
                    if p_hist is not None and np.isfinite(np.real(p_hist)):
                        p_target_init = complex(p_hist)

                p_sel, u_sel, j_sel = mode_matching.select_pk_mode_initial_lock(
                    vals,
                    vecs,
                    omegas,
                    s,
                    omega_n,
                    converged_omegas_at_V,
                    converged_p_at_V,
                    freq_margin=mode_freq_margin,
                    p_target=p_target_init,
                    claimed_indices=converged_wet_indices_at_V,
                    min_omega_sep_rad=min_root_sep_rad,
                )
                wet_mode_identified = p_sel is not None

            elif j == 0:
                p_sel, u_sel, j_sel = mode_matching.select_pk_mode_initial_lock(
                    vals,
                    vecs,
                    omegas,
                    s,
                    omega_n,
                    converged_omegas_at_V,
                    converged_p_at_V,
                    freq_margin=mode_freq_margin,
                    p_target=mode_matching.predict_p_target_from_history(results, s, i),
                    claimed_indices=converged_wet_indices_at_V,
                    min_omega_sep_rad=min_root_sep_rad,
                )
                wet_mode_identified = p_sel is not None

            else:
                if mac_matching:
                    p_target = mode_matching.predict_p_target_from_history(results, s, i)
                    if p_target is None and prev_eigs_row is not None and s in modes:
                        col = modes.index(s)
                        if col < len(prev_eigs_row) and np.isfinite(np.abs(prev_eigs_row[col])):
                            p_target = prev_eigs_row[col]
                    if p_target is not None:
                        print(f"      [P_TARGET] Predicted from history: p={p_target:.6f}, ω={np.abs(np.imag(p_target)):.3f} rad/s")
                    else:
                        print("      [P_TARGET] Not available (need 2+ velocity steps)")

                    _msw = [0]
                    if diagnostics is not None:
                        _msw[0] = diagnostics.mode_switch_warnings

                    p_sel, u_sel, j_sel = mode_matching.mode_matching_paper_compliant(
                        vals,
                        vecs,
                        s,
                        j,
                        last_converged_omegas,
                        converged_omegas_at_V,
                        converged_vecs_at_V,
                        mode_freq_margin,
                        w_freq,
                        w_mac,
                        omega_n,
                        _msw,
                        prev_step_vec=prev_step_vec,
                        prev_iter_vec=prev_iter_vec,
                        p_target=p_target,
                        skip_inter_mode_on_perfect_mac=skip_inter_mode_on_perfect_mac,
                        claimed_indices=converged_wet_indices_at_V,
                        min_omega_sep_rad=min_root_sep_rad,
                    )
                    if diagnostics is not None:
                        diagnostics.mode_switch_warnings = _msw[0]
                else:
                    print("      [MAC disabled] Using frequency-based mode selection")

                    if last_converged_mode_matching and len(last_converged_omegas) > 0 and s < len(last_converged_omegas):
                        omega_ref = last_converged_omegas[s]
                        lower_bound = omega_ref * (1 - mode_freq_margin)
                        upper_bound = omega_ref * (1 + mode_freq_margin)
                        print(
                            f"      Using last converged frequency: ω_ref={omega_ref:.3f} rad/s, bounds=[{lower_bound:.3f}, {upper_bound:.3f}]"
                        )
                    else:
                        omega_ref = omega_n[s]
                        lower_bound = omega_ref * (1 - 0.2)
                        upper_bound = omega_ref * (1 + 0.2)
                        print(
                            f"      Using structural frequency: ω_struct={omega_ref:.3f} rad/s, bounds=[{lower_bound:.3f}, {upper_bound:.3f}]"
                        )

                    bound_mask = (omegas >= lower_bound) & (omegas <= upper_bound)
                    if np.any(bound_mask):
                        valid_indices = np.where(bound_mask)[0]
                        errors = np.abs(omegas[valid_indices] - omega_ref)
                        best_local = int(np.argmin(errors))
                        j_sel = valid_indices[best_local]
                        p_sel = vals[j_sel]
                        u_sel = vecs[:, j_sel]
                        print(f"      Selected: ω={np.abs(np.imag(p_sel)):.3f} rad/s (closest in bounds)")
                    else:
                        errors = np.abs(omegas - omega_ref)
                        j_sel = int(np.argmin(errors))
                        p_sel = vals[j_sel]
                        u_sel = vecs[:, j_sel]
                        print(f"      ⚠ No mode in bounds, selected closest: ω={np.abs(np.imag(p_sel)):.3f} rad/s")

            if p_sel is None:
                print(f"    ✗ MODE {s} NOT FOUND (no eigenvalues in frequency bounds)")
                print("       → Mode has disappeared, inserting NaN")

                results_for_V.append(
                    {
                        "mode": int(s),
                        "p": np.nan,
                        "omega": np.nan,
                        "sigma": np.nan,
                        "gamma": np.nan,
                        "u": None,
                        "k": np.nan,
                        "vecs": None,
                        "converged": False,
                        "iterations": j + 1,
                        "mode_lost": True,
                    }
                )
                converged = True
                break

            omega_sel = float(np.abs(np.imag(p_sel)))
            sigma_sel = float(np.real(p_sel))

            if abs(omega_sel) > 1e-6:
                gamma_sel = sigma_sel / omega_sel
            else:
                gamma_sel = 0.0

            k_new = relax_k(k, omega_sel, V, fRLX)
            print(f"      Updated k: {k_new:.6g} (from {k:.6g})")

            k_change = abs(k_new - k)

            if k_change < tol:
                extra = None
                if on_converge_fn is not None:
                    if matrix_iter_state is not None:
                        matrix_iter_state["p"] = p_sel
                        matrix_iter_state["s"] = s
                        matrix_iter_state["j"] = j
                    extra = on_converge_fn(k_new, V, s, p_sel, u_sel)

                contributions = None
                w_modal_data = None
                if isinstance(extra, dict):
                    contributions = extra.get("contributions")
                    w_modal_data = extra.get("w_modal_data")

                print(f"    ✓ CONVERGED at iteration {j}")
                print(f"      wet_eigenvalues: {vals[:5]}... (first 5)")
                print(f"      k = {k_new:.6g}, ω = {omega_sel:.4f} rad/s, γ = {gamma_sel:.6f}, p = {p_sel:.6f}")

                n_wet_save = min(3, vecs.shape[1])
                wet_evecs_first3 = vecs[:, :n_wet_save].copy()
                wet_evals_first3 = vals[:n_wet_save].copy()
                wet_vals_all = vals.copy()

                results_for_V.append(
                    {
                        "mode": int(s),
                        "p": complex(p_sel),
                        "omega": float(omega_sel),
                        "sigma": float(sigma_sel),
                        "gamma": float(gamma_sel),
                        "u": u_sel,
                        "k": float(k_new),
                        "vecs": vecs,
                        "wet_evecs_first3": wet_evecs_first3,
                        "wet_evals_first3": wet_evals_first3,
                        "wet_vals_all": wet_vals_all,
                        "converged": True,
                        "iterations": j + 1,
                        "contributions": contributions,
                        "w_modal_data": w_modal_data,
                    }
                )

                if diagnostics is not None:
                    diagnostics.converged_wet_spectrum_history.append(
                        {
                            "V": float(V),
                            "structural_mode": int(s),
                            "pk_iteration": int(j),
                            "k": float(k_new),
                            "wet_evecs_first3": wet_evecs_first3,
                            "wet_evals_first3": wet_evals_first3,
                        }
                    )

                converged_omegas_at_V.append(omega_sel)
                converged_vecs_at_V.append(u_sel)
                converged_p_at_V.append(complex(p_sel))
                if j_sel is not None:
                    converged_wet_indices_at_V.append(int(j_sel))
                converged = True
                break

            if use_k_guess_each_iter:
                k = k_guess
                if j == 0 or j % 10 == 0:
                    print(f"      Next iter k reset to k_guess = {k:.6g} (pk_use_k_guess_each_iter)")
            else:
                k = k_new
            p_prev_iter = p_sel
            prev_iter_vec = u_sel

            if j % 10 == 0 and j > 0:
                print(f"      k={k:.6g}, ω={omega_sel:.4f} rad/s, Δk={k_change:.2e}")

        if not converged and p_sel is not None:
            print(f"    ✗ NO CONVERGENCE after {max_iter} iterations")
            print(f"      Final: k={k:.6g}, ω={omega_sel:.4f} rad/s, γ={gamma_sel:.6f}")

            results_for_V.append(
                {
                    "mode": int(s),
                    "p": complex(p_sel),
                    "omega": float(omega_sel),
                    "sigma": float(sigma_sel),
                    "gamma": float(gamma_sel),
                    "u": u_sel,
                    "k": float(k_new),
                    "vecs": vecs,
                    "converged": False,
                    "iterations": max_iter,
                    "warn": "no_convergence",
                }
            )

            converged_omegas_at_V.append(omega_sel)
            if p_sel is not None:
                converged_p_at_V.append(complex(p_sel))
            if diagnostics is not None:
                diagnostics.convergence_issues += 1

    M_struct = []
    M_effective = []
    K_struct = []
    K_aero = []
    K_effective = []
    C_struct = []
    C_aero = []
    C_effective = []
    w_velocity = []
    w_slope = []

    for mode_data in results_for_V:
        if mode_data.get("converged", False):
            contrib = mode_data.get("contributions")
            if contrib is not None:
                M_struct.append(contrib["M_struct"])
                M_effective.append(contrib["M_effective"])
                K_struct.append(contrib["K_struct"])
                K_aero.append(contrib["K_aero"])
                K_effective.append(contrib["K_effective"])
                C_struct.append(contrib["C_struct"])
                C_aero.append(contrib["C_aero"])
                C_effective.append(contrib["C_effective"])

            w_data = mode_data.get("w_modal_data")
            if w_data is not None:
                w_velocity.append(w_data["w_velocity_norm"])
                w_slope.append(w_data["w_slope_norm"])

    if diagnostics is not None and len(K_struct) > 0:
        diagnostics.contributions_history.append(
            {
                "V": float(V),
                "M_struct": M_struct,
                "M_effective": M_effective,
                "K_struct": K_struct,
                "K_aero": K_aero,
                "K_effective": K_effective,
                "C_struct": C_struct,
                "C_aero": C_aero,
                "C_effective": C_effective,
            }
        )

    if diagnostics is not None and len(w_velocity) > 0:
        diagnostics.w_modal_history.append(
            {
                "V": float(V),
                "w_velocity_norm": float(np.mean(w_velocity)),
                "w_slope_norm": float(np.mean(w_slope)),
            }
        )

    return results_for_V


def pk_sweep(
    V_list,
    modes,
    omega_n,
    build_matrix_fn,
    k_guess,
    rho,
    tol,
    fXK0,
    fRLX,
    freq_margin,
    w_freq,
    w_mac,
    max_iter,
    *,
    dry_vals=None,
    dry_vecs=None,
    perturb_k=1e-6,
    mac_matching=True,
    last_converged_mode_matching=True,
    matrix_iter_state=None,
    on_converge_fn=None,
    diagnostics=None,
    initial_results=None,
    last_converged_omegas_override=None,
    prev_vecs_matrix=None,
    prev_eigs_row=None,
    eigen_positive_imag_only=False,
    skip_inter_mode_on_perfect_mac=False,
    use_k_guess_each_iter=False,
    pk_predict_and_select=True,
    min_root_sep_rad=1.0,
    **kwargs,
):
    """
    Velocity sweep using the p-k engine.

    ``omega_n`` : structural natural frequencies [rad/s] (same as ``PKSolverV3.self.omega_n``).
    """
    results = [] if initial_results is None else list(initial_results)
    k0_guess = k_guess

    print("\n" + "=" * 70)
    print("STARTING PK-METHOD FLUTTER SWEEP (pk_method engine)")
    print("=" * 70)
    print(f"Number of velocities: {len(V_list)}")
    print(f"Number of modes: {len(modes)}")
    print(f"Parameters: fXK0={fXK0}, fRLX={fRLX}, rho={rho}")
    if use_k_guess_each_iter:
        print(f"pk_use_k_guess_each_iter=True → k reset to k_guess={k_guess:.6g} after each non-converged iteration")
    print("=" * 70)

    for i, V in enumerate(V_list):
        i_global = len(results)
        print(f"\n{'=' * 70}")
        print(f"VELOCITY STEP {i_global + 1}: V = {V:.2f} m/s")
        print("=" * 70)

        res_V = pk_solve_at_velocity(
            i_global,
            V,
            V_list,
            modes,
            omega_n,
            build_matrix_fn,
            k0_guess,
            k_guess,
            tol,
            fXK0,
            fRLX,
            freq_margin,
            w_freq,
            w_mac,
            max_iter,
            results,
            dry_vals,
            dry_vecs,
            perturb_k,
            mac_matching,
            last_converged_mode_matching,
            matrix_iter_state=matrix_iter_state,
            on_converge_fn=on_converge_fn,
            diagnostics=diagnostics,
            last_converged_omegas_override=last_converged_omegas_override,
            prev_vecs_matrix=prev_vecs_matrix,
            prev_eigs_row=prev_eigs_row,
            eigen_positive_imag_only=eigen_positive_imag_only,
            skip_inter_mode_on_perfect_mac=skip_inter_mode_on_perfect_mac,
            use_k_guess_each_iter=use_k_guess_each_iter,
            min_root_sep_rad=min_root_sep_rad,
        )

        wet_spectrum = None
        primary_mode = modes[0] if modes else 0
        for md in res_V:
            if md.get("mode") != primary_mode:
                continue
            if not md.get("converged", False):
                continue
            wet_vals_all = md.get("wet_vals_all")
            if wet_vals_all is None:
                wet_vals_all = md.get("wet_evals_first3")
            if wet_vals_all is None:
                continue
            wet_spectrum = build_wet_spectrum_summary(
                wet_vals_all, omega_n, res_V, freq_margin
            )
            break

        if pk_predict_and_select and i_global >= 2:
            res_V = mode_matching.predict_and_select_reorder_modes(
                res_V,
                results,
                i_global,
                delta=max(0.025, float(freq_margin)),
                min_omega_sep_rad=min_root_sep_rad,
            )

        results.append({"V": float(V), "modes": res_V, "wet_spectrum": wet_spectrum})

        if len(res_V) > 0 and res_V[0].get("converged", False) and not np.isnan(res_V[0].get("k", np.nan)):
            k0_guess = res_V[0]["k"]

    print("\n" + "=" * 70)
    print("PK ITERATIONS COMPLETED")
    print("=" * 70)

    problems = check_frequency_continuity(results)
    if problems:
        print(f"\n⚠ WARNING: {len(problems)} frequency discontinuities detected:")
        for p in problems[:5]:
            print(
                f"  V={p['velocity']:.2f} m/s, Mode {p['mode']}: "
                f"{p['omega_prev']:.3f} → {p['omega_curr']:.3f} rad/s "
                f"(Δ={100*p['rel_change']:.1f}%)"
            )

    print("\n" + "=" * 70)
    print("FLUTTER SWEEP SUMMARY")
    print("=" * 70)
    if diagnostics is not None:
        print(f"Mode switch warnings: {diagnostics.mode_switch_warnings}")
        print(f"Convergence issues: {diagnostics.convergence_issues}")
        print(f"Extrapolation failures: {diagnostics.extrapolation_failures}")
    print("=" * 70 + "\n")

    return results
