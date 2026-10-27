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
        self.qmodal_iteration_history = []
        self.w_modal_history = []
        self.converged_wet_spectrum_history = []
        self.mac_convergence_history = []
        self.mode_switch_warnings = 0
        self.convergence_issues = 0
        self.extrapolation_failures = 0
        self.iteration_log = []
        # Opt-in: raw (unfiltered) (vals_all, vecs_all) at every build_matrix_fn
        # evaluation, keyed loosely by (V, k) -- for independent MAC-based mode
        # identification in post-processing, bypassing the per-mode selector
        # entirely (diagnostic only; off by default, no effect on tracking).
        self.capture_full_spectrum = False
        self.full_spectrum_log = []
        # Non-oscillatory (real, growing) eigenvalues seen anywhere in the raw
        # spectrum during the sweep -- a divergence-type instability, distinct
        # from flutter. The oscillatory-only V-g/V-f bookkeeping is structurally
        # blind to these (they carry omega=0, never match any frequency-window
        # selector), so without this they are silently dropped even though the
        # coupled system genuinely goes unstable (2026-09-04, GOLAND_water:
        # real eigenvalues +20.86/+17.88 1/s at V=2.53 m/s, k=17.76, well inside
        # the valid GAF table range -- not an extrapolation artifact).
        self.real_instability_events = []
        self._real_instab_warned_keys = set()


def initial_k_for_mode(s, V, results_for_V, omega_n, k_guess, fXK0=0.618, own_prev_omega=None):
    """
    Hybrid reduced-frequency initial guess (Yuan & Zhang 2023, Eq. 13–15).

    Uses the converged frequency of mode ``s-1`` at the current airspeed, optionally
    blended with mode ``s-2`` — not a sorted eigenvalue from the current spectrum.

    ``own_prev_omega``, when given (mode ``s``'s own converged ω from the
    previous velocity step), takes priority over both the s-1 inheritance and
    the dry-frequency fallback: a mode already has an established trajectory
    at that point, and reseeding from a neighbour or the dry frequency can
    point the trial-k evaluation at a region of the spectrum with no
    oscillatory root at all, dropping an already-converging branch (2026-09-03,
    GOLAND_water torsion: lost at the first velocity step after 7 converged
    steps, because the s-1/dry fallback ignored its own history).
    """
    V = max(float(V), 1e-12)
    if s == 0:
        return max(float(k_guess), 1e-6)

    if own_prev_omega is not None and np.isfinite(own_prev_omega) and float(own_prev_omega) > 0.0:
        return max(float(own_prev_omega) / V, 1e-6)

    if len(results_for_V) < s:
        return max(float(omega_n[s]) / V, float(k_guess), 1e-6)

    om_ssm1 = results_for_V[s - 1].get("omega", np.nan)
    if not np.isfinite(om_ssm1) or om_ssm1 <= 0.0:
        return max(float(omega_n[s]) / V, float(k_guess), 1e-6)

    # The s-1 inheritance assumes neighbouring modes are close in frequency
    # (Yuan & Zhang's own worked examples). When the dry structural spacing
    # says otherwise -- e.g. first bending vs. first torsion, often >2x apart --
    # seeding k from mode s-1 evaluates the hydro/aero operators nowhere near
    # mode s's own resonance; with large added mass/damping (water) that can
    # leave mode s overdamped (no oscillatory wet root) at that trial k, so it
    # is never found. Fall back to mode s's own structural frequency instead.
    if s < len(omega_n) and omega_n[s] > 0.0:
        ratio = float(om_ssm1) / float(omega_n[s])
        if ratio < 0.5 or ratio > 2.0:
            print(
                f"      initial_k_for_mode: ω_{{s-1}}^conv={om_ssm1:.3f} rad/s is "
                f"{ratio:.2f}x ω_struct[{s}]={omega_n[s]:.3f} rad/s -- too far to "
                f"inherit, seeding from mode {s}'s own structural frequency instead"
            )
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


def build_wet_spectrum_summary(vals, omega_n, results_for_V, freq_margin, *, omega_phys_min=None, vecs=None):
    """
    Classify sorted wet roots at PK convergence.

    Roots matched to a converged PK branch (``pk_tracked``) are excluded from
    ``untracked``. The lowest-frequency untracked root is the usual orphan branch
    that never enters the structural mode-matching window.

    ``vecs`` (optional) is a sequence aligned with ``vals`` holding the augmented-
    state eigenvector of each root; when supplied it is attached under the ``vec``
    key so untracked branches can be MAC-tracked / post-processed.
    """
    vals = np.asarray(vals, dtype=complex).ravel()
    omega_n = np.asarray(omega_n, dtype=float).ravel()
    if vecs is not None and len(vecs) != vals.size:
        vecs = None
    margin = float(freq_margin)
    if omega_phys_min is None:
        from . import mode_matching
        omega_phys_min = mode_matching.physical_omega_min(omega_n)
    else:
        omega_phys_min = float(omega_phys_min)

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
        if omega < omega_phys_min:
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

        root = {
            "index": int(idx),
            "p": complex(lam),
            "omega": omega,
            "sigma": sigma,
            "gamma": float(gamma),
            "kind": kind,
            "structural_near_mode": struct_near,
        }
        if vecs is not None and vecs[idx] is not None:
            root["vec"] = np.asarray(vecs[idx], dtype=complex).ravel()
        roots.append(root)

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


def merge_pk_wet_spectrum_vals(modes_list, *, return_vecs=False):
    """
    Union of wet eigenvalues from all converged PK modes at one airspeed.

    Each mode is solved at its own k; merging yields a unified spectrum for
    untracked-root diagnostics that includes every physical branch.

    When ``return_vecs=True`` the corresponding augmented-state eigenvectors are
    returned alongside the eigenvalues (``md['vecs'][:, i]`` is aligned with
    ``md['wet_vals_all'][i]``). The returned vectors are a list (one entry per
    kept eigenvalue) so downstream code can MAC-track the untracked branches.
    """
    merged = []
    merged_vecs = []
    for md in modes_list or []:
        if not md.get("converged", False):
            continue
        wet_vals = md.get("wet_vals_all")
        vecs = md.get("vecs")
        if wet_vals is None:
            wet_vals = md.get("wet_evals_first3")
            vecs = md.get("wet_evecs_first3")
        if wet_vals is None:
            continue
        wet_vals = np.asarray(wet_vals, dtype=complex).ravel()
        vec_ok = (
            vecs is not None
            and np.ndim(vecs) == 2
            and np.asarray(vecs).shape[1] >= wet_vals.size
        )
        for i, lam in enumerate(wet_vals):
            if np.isfinite(np.abs(lam)):
                merged.append(complex(lam))
                merged_vecs.append(
                    np.asarray(vecs)[:, i].copy() if vec_ok else None
                )
    if not merged:
        return (None, None) if return_vecs else None
    merged = np.asarray(merged, dtype=complex)
    om = np.abs(np.imag(merged))
    order = np.argsort(om)
    merged = merged[order]
    merged_vecs = [merged_vecs[k] for k in order]
    om = om[order]
    keep = [0]
    for i in range(1, len(merged)):
        if abs(om[i] - om[keep[-1]]) > 0.02 * max(om[i], om[keep[-1]], 1.0):
            keep.append(i)
    keep = np.asarray(keep, dtype=int)
    merged = merged[keep]
    merged_vecs = [merged_vecs[k] for k in keep]
    if return_vecs:
        return merged, merged_vecs
    return merged


def pk_wet_spectrum_for_velocity(modes_list, omega_n, freq_margin, *, omega_phys_min=None):
    """Classified wet spectrum from merged PK eigenvalues at one V."""
    wet_vals, wet_vecs = merge_pk_wet_spectrum_vals(modes_list, return_vecs=True)
    if wet_vals is None:
        return None
    return build_wet_spectrum_summary(
        wet_vals, omega_n, modes_list, freq_margin,
        omega_phys_min=omega_phys_min, vecs=wet_vecs,
    )


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
    omega_phys_min_frac=0.45,
    tracking_policy="paper",
    mac_reference="previous_velocity",
    freq_jump_terminate_frac=None,
    mac_terminate_floor=0.5,
    track_seed_omegas=None,
    min_pk_iter=1,
    pk_discrete_dt=None,
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
    from . import mode_matching
    omega_phys_min = mode_matching.physical_omega_min(omega_n, frac=omega_phys_min_frac)

    for s in modes:
        print(f"\n  === MODE {s} ===")

        # First-velocity seed frequency for this branch. Defaults to the dry
        # natural frequency ω_n[s]; an optional ``track_seed_omegas`` override
        # lets the caller seed on the *wet* frequency instead (e.g. when added
        # mass moves a wet root far from its dry value, so ω_n[s] would land
        # closer to a neighbouring branch and mis-seed frequency continuity).
        seed_omega_s = float(omega_n[s])
        omega_n_for_s = omega_n
        if track_seed_omegas is not None and s < len(track_seed_omegas):
            _cand = float(track_seed_omegas[s])
            if np.isfinite(_cand) and _cand > 0.0:
                seed_omega_s = _cand
                # Also override omega_n[s] wherever the s>0 first-lock machinery
                # (initial_k_for_mode / select_pk_mode_initial_lock) reads the
                # "structural" target for THIS slot -- otherwise those still
                # centre on the dry ω_n[s], which for a strongly hybridised
                # mode can sit far from every available wet root (see
                # track_seed_omegas docstring above) and the search window /
                # k-guess never reach the branch that actually carries it.
                omega_n_for_s = np.array(omega_n, dtype=float, copy=True)
                omega_n_for_s[s] = seed_omega_s

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

        if tracking_policy == "frequency":
            if i > 0 and s < len(results[i - 1]["modes"]):
                om0 = results[i - 1]["modes"][s].get("omega", np.nan)
            else:
                om0 = seed_omega_s
            if not np.isfinite(om0) or float(om0) <= 0.0:
                om0 = seed_omega_s
            k = max(float(om0) / max(float(V), 1e-12), 1e-4)
            print(
                f"    k_initial = {k:.6g} (frequency policy: ω/V from "
                f"{'prev V' if i > 0 else 'seed'}={om0:.3f} rad/s)"
            )
        elif s == 0:
            if i == 0:
                k = k_guess
                print(f"    k_initial = {k:.6g} (from structural ω_n[0]={omega_n[0]:.3f} rad/s)")
            else:
                k = k0_guess
                print(f"    k_initial = {k:.6g} (from previous velocity step)")
        else:
            own_prev_omega = None
            if i > 0 and s < len(results[i - 1]["modes"]):
                own_prev_omega = results[i - 1]["modes"][s].get("omega", None)
            k = initial_k_for_mode(s, V, results_for_V, omega_n_for_s, k_guess, fXK0,
                                    own_prev_omega=own_prev_omega)
            if own_prev_omega is not None and np.isfinite(own_prev_omega) and own_prev_omega > 0.0:
                print(
                    f"    k_initial = {k:.6g} (own history: ω_{{{s}}}^conv(V_prev)="
                    f"{own_prev_omega:.3f} rad/s)"
                )
            else:
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
        u_first_iter = None
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
        omega_track_v = None

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

            mu_all, vecs_all = eig(Ac)
            if pk_discrete_dt is not None:
                from .newmark_discrete import discrete_poles_to_continuous
                vals_all = discrete_poles_to_continuous(mu_all, pk_discrete_dt)
            else:
                vals_all = mu_all

            if diagnostics is not None and getattr(diagnostics, "capture_full_spectrum", False):
                diagnostics.full_spectrum_log.append({
                    "V": float(V), "k": float(k), "mode": int(s), "iteration": int(j),
                    "vals_all": vals_all.copy(), "vecs_all": vecs_all.copy(),
                })

            if diagnostics is not None:
                _real_pos = np.where(
                    (np.abs(np.imag(vals_all)) < 1e-6) & (np.real(vals_all) > 1e-6)
                )[0]
                if _real_pos.size > 0:
                    _growth = np.real(vals_all[_real_pos])
                    diagnostics.real_instability_events.append({
                        "V": float(V), "mode_slot": int(s), "iteration": int(j),
                        "k": float(k), "growth_rates": _growth.copy(),
                    })
                    _warn_key = (round(float(V), 6), int(s))
                    if _warn_key not in diagnostics._real_instab_warned_keys:
                        diagnostics._real_instab_warned_keys.add(_warn_key)
                        print(
                            f"      ⚠ NON-OSCILLATORY INSTABILITY: V={V:.3f} m/s, mode slot {s}, "
                            f"k={k:.4f} -> {_real_pos.size} real eigenvalue(s), growth rate(s) "
                            f"{np.array2string(_growth, precision=3)} 1/s (divergence-type; "
                            f"omega=0, invisible to V-g/V-f frequency-window selection)"
                        )

            if eigen_positive_imag_only or tracking_policy == "frequency":
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

            if tracking_policy == "frequency":
                if omega_track_v is None:
                    if i > 0 and s < len(results[i - 1]["modes"]):
                        om_tr = results[i - 1]["modes"][s].get("omega", np.nan)
                    else:
                        om_tr = seed_omega_s
                    if not np.isfinite(om_tr) or float(om_tr) <= 0.0:
                        om_tr = seed_omega_s
                    omega_track_v = float(om_tr)
                prev_p_track = None
                if i > 0 and s < len(results[i - 1]["modes"]):
                    prev_p_track = results[i - 1]["modes"][s].get("p", None)
                p_sel, u_sel, j_sel = mode_matching.select_pk_mode_frequency_continuity(
                    vals,
                    vecs,
                    omegas,
                    omega_track_v,
                    prev_vec=prev_step_vec,
                    prev_p=prev_p_track,
                )
                if p_sel is not None:
                    print(
                        f"      [frequency tracking] ω_track={omega_track_v:.3f} rad/s → "
                        f"ω={np.abs(np.imag(p_sel)):.3f} rad/s (index {j_sel})"
                    )

                # Damping jump at coalescence: ω continuous but Re(p)/g hops to partner.
                if (
                    p_sel is not None
                    and i > 0
                    and prev_p_track is not None
                    and np.isfinite(np.real(prev_p_track))
                    and np.isfinite(np.real(p_sel))
                ):
                    g_prev = 2.0 * float(np.real(prev_p_track)) / max(
                        float(np.abs(np.imag(prev_p_track))), 1e-12
                    )
                    g_new = 2.0 * float(np.real(p_sel)) / max(
                        float(np.abs(np.imag(p_sel))), 1e-12
                    )
                    mac_prev = float("nan")
                    if prev_step_vec is not None and u_sel is not None:
                        mac_prev = float(mode_matching.compute_mac(u_sel, prev_step_vec))
                    if abs(g_new - g_prev) > 0.5 and (
                        not np.isfinite(mac_prev) or mac_prev < float(mac_terminate_floor)
                    ):
                        print(
                            f"      ⊗ MODE {s} branch terminated: g jump "
                            f"{g_prev:.3f} → {g_new:.3f} at coalescence "
                            f"(MAC={mac_prev:.3f}); refusing partner-root hop."
                        )
                        p_sel, u_sel, j_sel = None, None, None

                # ── Branch-termination guard (shape-gated) ───────────────────
                # When a branch's oscillatory root vanishes (e.g. the bending
                # mode goes overdamped/aperiodic), the nearest remaining root is
                # a *different* mode far away in frequency. Frequency continuity
                # would jump the branch onto it → a spurious discontinuity. If
                # the nearest root lies more than ``freq_jump_terminate_frac``
                # away from the tracked frequency AND its MAC to the branch's
                # previous-velocity shape is below ``mac_terminate_floor`` (i.e.
                # it is a different mode, not a fast-but-continuous shift), the
                # branch is terminated instead of reassigned. Only active for
                # i>0 (never on the very first, seeding, velocity step).
                if (
                    p_sel is not None
                    and freq_jump_terminate_frac is not None
                    and i > 0
                    and np.isfinite(omega_track_v)
                    and omega_track_v > 1e-6
                ):
                    om_pick = float(np.abs(np.imag(p_sel)))
                    jump_frac = abs(om_pick - omega_track_v) / omega_track_v
                    if jump_frac > float(freq_jump_terminate_frac):
                        mac_prev = float("nan")
                        if prev_step_vec is not None and u_sel is not None:
                            mac_prev = float(
                                mode_matching.compute_mac(u_sel, prev_step_vec)
                            )
                        shape_lost = (
                            prev_step_vec is None
                            or not np.isfinite(mac_prev)
                            or mac_prev < float(mac_terminate_floor)
                        )
                        if shape_lost:
                            print(
                                f"      ⊗ MODE {s} branch terminated: nearest root "
                                f"ω={om_pick:.3f} rad/s is {jump_frac * 100:.0f}% from "
                                f"ω_track={omega_track_v:.3f} rad/s (MAC={mac_prev:.3f} "
                                f"< {mac_terminate_floor}); oscillatory root vanished."
                            )
                            p_sel, u_sel, j_sel = None, None, None

                wet_mode_identified = p_sel is not None

            elif j == 0 and (prev_iter_vec is None and prev_step_vec is None):
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
                    omega_n_for_s,
                    converged_omegas_at_V,
                    converged_p_at_V,
                    freq_margin=mode_freq_margin,
                    p_target=p_target_init,
                    claimed_indices=converged_wet_indices_at_V,
                    min_omega_sep_rad=min_root_sep_rad,
                )
                wet_mode_identified = p_sel is not None

            elif j == 0 and tracking_policy != "frequency" and prev_step_vec is None:
                p_sel, u_sel, j_sel = mode_matching.select_pk_mode_initial_lock(
                    vals,
                    vecs,
                    omegas,
                    s,
                    omega_n_for_s,
                    converged_omegas_at_V,
                    converged_p_at_V,
                    freq_margin=mode_freq_margin,
                    p_target=mode_matching.predict_p_target_from_history(results, s, i),
                    claimed_indices=converged_wet_indices_at_V,
                    min_omega_sep_rad=min_root_sep_rad,
                )
                wet_mode_identified = p_sel is not None

            elif j == 0 and tracking_policy != "frequency" and prev_step_vec is not None and mac_matching:
                p_target = mode_matching.predict_p_target_from_history(results, s, i)
                if p_target is not None:
                    print(f"      [P_TARGET] Predicted from history: p={p_target:.6f}, ω={np.abs(np.imag(p_target)):.3f} rad/s")
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
                    prev_iter_vec=None,
                    p_target=p_target,
                    skip_inter_mode_on_perfect_mac=skip_inter_mode_on_perfect_mac,
                    claimed_indices=converged_wet_indices_at_V,
                    min_omega_sep_rad=min_root_sep_rad,
                    omega_phys_min=omega_phys_min,
                    mac_reference=mac_reference,
                )
                wet_mode_identified = p_sel is not None

            elif j == 0 and tracking_policy != "frequency":
                p_sel, u_sel, j_sel = mode_matching.select_pk_mode_initial_lock(
                    vals,
                    vecs,
                    omegas,
                    s,
                    omega_n_for_s,
                    converged_omegas_at_V,
                    converged_p_at_V,
                    freq_margin=mode_freq_margin,
                    p_target=mode_matching.predict_p_target_from_history(results, s, i),
                    claimed_indices=converged_wet_indices_at_V,
                    min_omega_sep_rad=min_root_sep_rad,
                )
                wet_mode_identified = p_sel is not None

            elif tracking_policy == "frequency":
                pass  # already selected above

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
                        omega_phys_min=omega_phys_min,
                        mac_reference=mac_reference,
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

            if p_sel is not None and u_first_iter is None:
                u_first_iter = np.array(u_sel, dtype=complex, copy=True)

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

            # ── Minimum-iteration guard ───────────────────────────────────────
            # Iteration j=0 evaluates the aero downwash on the DRY structural
            # frequency (mode-isolated Yuan & Zhang bootstrap, aero_system.py
            # j==0,s>0 branch), which is internally inconsistent with the wet
            # reduced frequency used for the Qjj table lookup. Iterations j>=1
            # re-evaluate on the converging *wet* frequency (p_conv). Accepting
            # the j=0 bootstrap whenever the k-guess happens to be within tol
            # returns a systematically under-damped root and injects a step
            # discontinuity (Model2t torsion branch ~13.7 kn: γ jumps -0.22→-0.11
            # the moment j=0 acceptance turns on and stays on for all higher V).
            # Requiring at least ``min_pk_iter`` refinement iterations forces every
            # velocity to settle on the fully-coupled wet aero.
            if k_change < tol and j < min_pk_iter:
                print(
                    f"      [min_pk_iter] converged at j={j} (Δk={k_change:.2e}<tol) "
                    f"but forcing refinement to j>={min_pk_iter} (j=0 uses dry-freq bootstrap aero)"
                )

            if k_change < tol and j >= min_pk_iter:
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

                mac_prev_velocity = float("nan")
                mac_first_iter = float("nan")
                if u_sel is not None:
                    if prev_step_vec is not None:
                        mac_prev_velocity = float(
                            mode_matching.compute_mac(u_sel, prev_step_vec)
                        )
                    if u_first_iter is not None:
                        mac_first_iter = float(
                            mode_matching.compute_mac(u_sel, u_first_iter)
                        )

                print(f"    ✓ CONVERGED at iteration {j}")
                print(f"      wet_eigenvalues: {vals[:5]}... (first 5)")
                print(f"      k = {k_new:.6g}, ω = {omega_sel:.4f} rad/s, γ = {gamma_sel:.6f}, p = {p_sel:.6f}")
                if np.isfinite(mac_prev_velocity):
                    print(f"      MAC vs previous V: {mac_prev_velocity:.4f}")
                if np.isfinite(mac_first_iter):
                    print(f"      MAC vs first iter: {mac_first_iter:.4f}")

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
                        "mac_prev_velocity": mac_prev_velocity,
                        "mac_first_iter": mac_first_iter,
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
                    diagnostics.mac_convergence_history.append(
                        {
                            "V": float(V),
                            "mode": int(s),
                            "iteration": int(j),
                            "mac_prev_velocity": mac_prev_velocity,
                            "mac_first_iter": mac_first_iter,
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

            # ── Non-convergence death guard (frequency policy) ────────────────
            # A frequency-tracked branch that never settles is almost always a
            # branch whose oscillatory root has vanished (the mode goes
            # overdamped/aperiodic, e.g. the 1st bending mode near ~11 kn as in
            # Besch & Liu). Once the root disappears the k-iteration flip-flops
            # between the dying root and a distant unrelated root and can never
            # reach a fixed point. Recording the last (arbitrary) iterate injects
            # a spurious spike and, because the slot survives, lets the branch
            # re-attach onto a neighbouring mode at the next velocity (Branch 1
            # then becomes a noisy duplicate of the torsion branch).
            #
            # The MAC-gated guard in the iteration loop only fires on *converged*
            # picks, so it misses this failure mode. Here we terminate the branch
            # when the final non-converged frequency is a large jump from the
            # tracked frequency — a slow-but-continuous mode (final ω near
            # ω_track) is left untouched and simply recorded as non-converged.
            terminate_noconv = False
            if (
                tracking_policy == "frequency"
                and freq_jump_terminate_frac is not None
                and omega_track_v is not None
                and np.isfinite(omega_track_v)
                and float(omega_track_v) > 1e-6
                and np.isfinite(omega_sel)
            ):
                jump_frac = abs(omega_sel - omega_track_v) / omega_track_v
                if jump_frac > float(freq_jump_terminate_frac):
                    terminate_noconv = True
                    print(
                        f"      ⊗ MODE {s} branch terminated (no convergence): "
                        f"final ω={omega_sel:.3f} rad/s is {jump_frac * 100:.0f}% "
                        f"from ω_track={omega_track_v:.3f} rad/s — the oscillatory "
                        f"root vanished (mode overdamped/aperiodic)."
                    )

            if terminate_noconv:
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
                        "iterations": max_iter,
                        "mode_lost": True,
                        "warn": "no_convergence_terminated",
                    }
                )
                if diagnostics is not None:
                    diagnostics.convergence_issues += 1
                continue

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
    M_hydro = []
    M_effective = []
    K_struct = []
    K_aero = []
    K_effective = []
    C_struct = []
    C_aero = []
    C_hydro = []
    C_empirical = []
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
                C_empirical.append(
                    contrib.get(
                        "C_empirical",
                        np.zeros_like(contrib["C_struct"], dtype=float),
                    )
                )
                C_effective.append(contrib["C_effective"])
                if "M_hydro" in contrib:
                    M_hydro.append(contrib["M_hydro"])
                if "C_hydro" in contrib:
                    C_hydro.append(contrib["C_hydro"])

            w_data = mode_data.get("w_modal_data")
            if w_data is not None:
                w_velocity.append(w_data["w_velocity_norm"])
                w_slope.append(w_data["w_slope_norm"])

    if diagnostics is not None and len(K_struct) > 0:
        entry = {
            "V": float(V),
            "M_struct": M_struct,
            "M_effective": M_effective,
            "K_struct": K_struct,
            "K_aero": K_aero,
            "K_effective": K_effective,
            "C_struct": C_struct,
            "C_aero": C_aero,
            "C_empirical": C_empirical,
            "C_effective": C_effective,
        }
        if M_hydro:
            entry["M_hydro"] = M_hydro
        if C_hydro:
            entry["C_hydro"] = C_hydro
        diagnostics.contributions_history.append(entry)

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
    on_velocity_step_fn=None,
    omega_phys_min_frac=0.45,
    tracking_policy="paper",
    mac_reference="previous_velocity",
    freq_jump_terminate_frac=None,
    mac_terminate_floor=0.5,
    track_seed_omegas=None,
    min_pk_iter=1,
    pk_discrete_dt=None,
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
    if pk_discrete_dt is not None:
        print(f"pk_discrete_newmark=True → discrete Ass eigenvalues mapped via s=log(μ)/dt, dt={pk_discrete_dt:g} s")
    print("=" * 70)

    from . import mode_matching
    _omega_phys_min = mode_matching.physical_omega_min(omega_n, frac=omega_phys_min_frac)

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
            omega_phys_min_frac=omega_phys_min_frac,
            tracking_policy=tracking_policy,
            mac_reference=mac_reference,
            freq_jump_terminate_frac=freq_jump_terminate_frac,
            mac_terminate_floor=mac_terminate_floor,
            track_seed_omegas=track_seed_omegas,
            min_pk_iter=min_pk_iter,
            pk_discrete_dt=pk_discrete_dt,
        )

        wet_spectrum = pk_wet_spectrum_for_velocity(
            res_V, omega_n, freq_margin, omega_phys_min=_omega_phys_min,
        )

        if pk_predict_and_select and i_global >= 2:
            res_V = mode_matching.predict_and_select_reorder_modes(
                res_V,
                results,
                i_global,
                delta=max(0.025, float(freq_margin)),
                min_omega_sep_rad=min_root_sep_rad,
            )
            wet_spectrum = pk_wet_spectrum_for_velocity(
                res_V, omega_n, freq_margin, omega_phys_min=_omega_phys_min,
            )

        results.append({"V": float(V), "modes": res_V, "wet_spectrum": wet_spectrum})

        if on_velocity_step_fn is not None:
            try:
                on_velocity_step_fn(float(V), i_global, modes=res_V)
            except TypeError:
                on_velocity_step_fn(float(V), i_global)

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
