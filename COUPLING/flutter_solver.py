# FSI/COUPLING/flutter_solver.py

import numpy as np
from collections import namedtuple
#from .hydroelastic_utl import pk_solverv3
from .hydroelastic_utl import pk_solver_capytaine
from .hydroelastic_utl import pk_method, aero_system, hydro_system, empirical_fluid_damping, aero_q_scaling
from .hydroelastic_utl import orphan_mode_tracking
from .hydroelastic_utl import vgvf_plotting
from .hydroelastic_utl import roger_mode_tracking
from .hydroelastic_utl import post_processing
from .hydroelastic_utl import roger_fit
from .hydroelastic_utl import static_hydroelastic
from .hydroelastic_utl.flow_added_damping_benchmark import compute_resting_fluid_reference
import pickle
import os
import sys
import importlib.util
from types import SimpleNamespace

FlutterResults = namedtuple('FlutterResults', [
    'velocities', 'damping', 'frequencies', 'modes', 'flutter_speed', 'flutter_frequency',
    'raw_results', 'pk_solver', 'dlm_participants_history',
    'resting_ref',  # RestingFluidReference at first low V (see flow_added_damping_benchmark)
    'orphan_damping', 'orphan_frequencies', 'orphan_mac_activated',
    'static_hydroelastic',  # StaticHydroelasticResults from parallel quasi-steady pass
    'converged',  # bool array, same shape as damping/frequencies; None if not tracked by this path
])


def _finalize_flutter_results(
    velocities,
    damping,
    frequencies,
    modes,
    flutter_speed,
    flutter_frequency,
    config,
    *,
    raw_results=None,
    pk_solver=None,
    dlm_participants_history=None,
    orphan_damping=None,
    orphan_frequencies=None,
    orphan_mac_activated=False,
    static_hydroelastic=None,
    converged=None,
):
    """Attach resting-fluid reference from the lowest-speed flutter point."""
    resting_ref = compute_resting_fluid_reference(
        velocities, frequencies, damping, config
    )
    V_arr = np.asarray(velocities, dtype=float)
    g_arr = np.asarray(damping, dtype=float)
    f_arr = np.asarray(frequencies, dtype=float)

    if orphan_damping is not None and getattr(config, "orphan_mac_append_vgvf_branch", True):
        g_orph = np.asarray(orphan_damping, dtype=float).reshape(-1)
        f_orph = np.asarray(orphan_frequencies, dtype=float).reshape(-1)
        if g_orph.shape[0] == V_arr.shape[0] and np.any(np.isfinite(g_orph)):
            if g_arr.ndim == 1:
                g_arr = g_arr.reshape(-1, 1)
            if f_arr.ndim == 1:
                f_arr = f_arr.reshape(-1, 1)
            g_arr = np.column_stack([g_arr, g_orph])
            f_arr = np.column_stack([f_arr, f_orph])

    return FlutterResults(
        velocities=velocities,
        damping=g_arr,
        frequencies=f_arr,
        modes=modes,
        flutter_speed=flutter_speed,
        flutter_frequency=flutter_frequency,
        raw_results=raw_results,
        pk_solver=pk_solver,
        dlm_participants_history=dlm_participants_history,
        resting_ref=resting_ref,
        orphan_damping=orphan_damping,
        orphan_frequencies=orphan_frequencies,
        orphan_mac_activated=bool(orphan_mac_activated),
        static_hydroelastic=static_hydroelastic,
        converged=converged,
    )


def _create_static_hydroelastic_runner(config, structural_results, coupling_results, aerogrid):
    if not getattr(config, "static_hydroelastic_pass", False):
        return None
    if aerogrid is None:
        print("[static hydroelastic] skipped: aerogrid is None")
        return None
    if coupling_results.get("Z_force") is None:
        print("[static hydroelastic] skipped: Z_force not in coupling_results")
        return None
    return static_hydroelastic.StaticHydroelasticSweep(
        config, structural_results, coupling_results, aerogrid
    )


def _create_cp_snapshot_plotter(config, aerogrid, V_list, structural_results=None, coupling_results=None):
    if not getattr(config, "plot_cp_sweep_snapshots", True):
        return None
    if aerogrid is None:
        return None
    return post_processing.CpSweepSnapshotPlotter(
        config,
        aerogrid,
        V_list,
        structural_results=structural_results,
        coupling_results=coupling_results,
    )


def _make_on_velocity_step(static_runner, cp_plotter):
    if static_runner is None and cp_plotter is None:
        return None

    def on_step(V, i_V, modes=None):
        if static_runner is not None:
            static_runner.step(V, i_V)
        if cp_plotter is not None and modes is not None:
            cp_plotter.maybe_plot(V, i_V, modes)

    return on_step


def _finalize_static_hydroelastic_pass(flutter_results, static_runner, config):
    if static_runner is None:
        return flutter_results
    static_results = static_runner.finalize()
    if static_results is not None and getattr(config, "plot_static_hydroelastic_sweep", True):
        try:
            post_processing.plot_static_hydroelastic_sweep(static_results, config=config)
        except Exception as exc:
            print(f"[Warning] Static hydroelastic plot failed: {exc}")
    return flutter_results._replace(static_hydroelastic=static_results)


def _rfa_roger_convention(config):
    """Roger RFA convention: 'panelaero_legacy' or 'paper_zurich' (Kier eq. 16–21)."""
    return str(getattr(config, "rfa_roger_convention", "panelaero_legacy"))


def _is_paper_zurich_roger(config):
    return _rfa_roger_convention(config) == "paper_zurich"


def _roger_semichord_m(config):
    """Reference semichord b = c/2 [m] for paper_zurich convention."""
    chord = float(getattr(config, "chord"))
    ref = getattr(config, "rfa_blag_ref_length", "semichord")
    if isinstance(ref, str):
        ref_key = ref.lower()
        if ref_key in ("semichord", "half_chord", "c/2"):
            return 0.5 * chord
        if ref_key in ("chord", "c"):
            return chord
        raise ValueError(
            "config.rfa_blag_ref_length must be 'semichord', 'chord', or a positive length"
        )
    b = float(ref)
    if b <= 0.0:
        raise ValueError(f"config.rfa_blag_ref_length must be positive, got {b}")
    return b


def _k_weights_for_roger_fit(
    k_list,
    V,
    omega_n,
    sigma_frac=0.6,
    *,
    roger_convention="panelaero_legacy",
    semichord=1.0,
):
    """
    Gaussian weights for Roger LSQ.

    panelaero_legacy: peak at k_pa ≈ ω_n/V.
    paper_zurich: peak at k* ≈ ω_n b/V (k_list remains PanelAero k_pa on disk).
    """
    k_list = np.asarray(k_list, dtype=float)
    w = np.zeros(k_list.size, dtype=float)
    paper = roger_convention == "paper_zurich"
    b = float(semichord)
    for om in np.asarray(omega_n, dtype=float).ravel():
        if om <= 0 or V <= 0:
            continue
        if paper:
            k_star_c = om * b / V
            sigma = max(sigma_frac * k_star_c, 0.05 * b)
            k_star_grid = k_list * b
            w = np.maximum(w, np.exp(-0.5 * ((k_star_grid - k_star_c) / sigma) ** 2))
        else:
            kc = om / V
            sigma = max(sigma_frac * kc, 0.05)
            w = np.maximum(w, np.exp(-0.5 * ((k_list - kc) / sigma) ** 2))
    if np.max(w) < 1e-12:
        w[:] = 1.0
    else:
        w /= np.max(w)
    return w


def _omega_ref_for_roger_fit(i_V, freq_sweep, fit_indices, omega_n, n_modes_tracked):
    """
    Frequencies for Roger LSQ weighting: use tracked wet ω from the previous
    velocity step when available (adaptive band), else dry ω_n.
    """
    fit_indices = np.asarray(fit_indices, dtype=int)
    omega_n = np.asarray(omega_n, dtype=float).ravel()
    if i_V > 0 and freq_sweep:
        prev = np.asarray(freq_sweep[-1], dtype=float).ravel()
        omega_ref = np.empty(fit_indices.size, dtype=float)
        for j, mi in enumerate(fit_indices):
            if j < min(n_modes_tracked, prev.size) and np.isfinite(prev[j]) and prev[j] > 0:
                omega_ref[j] = float(prev[j])
            else:
                omega_ref[j] = float(omega_n[mi])
        return omega_ref
    return omega_n[fit_indices]


def _k_mask_for_roger_fit(k_list, V, omega_ref, config):
    """Boolean mask for k nodes used in Roger LSQ (flutter band + config limits)."""
    k_list = np.asarray(k_list, dtype=float)
    lo = float(getattr(config, "rfa_k_fit_min", 0.0))
    hi = float(getattr(config, "rfa_k_fit_max", np.inf))
    if V > 0 and omega_ref is not None and getattr(config, "rfa_k_fit_adaptive_band", True):
        om = np.asarray(omega_ref, dtype=float).ravel()
        om = om[np.isfinite(om) & (om > 0)]
        if om.size:
            lo_frac = float(getattr(config, "rfa_k_fit_band_lo_frac", 0.35))
            hi_frac = float(getattr(config, "rfa_k_fit_band_hi_frac", 3.5))
            k_lo = lo_frac * float(np.min(om)) / V
            k_hi = hi_frac * float(np.max(om)) / V
            lo = max(lo, k_lo)
            hi = min(hi, k_hi) if np.isfinite(hi) else k_hi
    return (k_list >= lo) & (k_list <= hi)


def _roger_tracked_fit_error(
    Q_modal_data,
    k_list,
    B0,
    B1,
    B2,
    Blag,
    blag,
    fit_indices,
    *,
    include_b2,
    recon_kw,
):
    """Relative Frobenius error on the Roger-fitted modal block only."""
    idx = np.asarray(fit_indices, dtype=int)
    sl = np.ix_(idx, idx)
    err_max = 0.0
    for i_k, k in enumerate(k_list):
        Q_fit = _reconstruct_Q_modal(
            k, B0, B1, B2, Blag, blag, include_b2=include_b2, **recon_kw,
        )[sl]
        Q_ref = Q_modal_data[i_k][sl]
        denom = max(float(np.linalg.norm(Q_ref, "fro")), 1e-30)
        err_max = max(err_max, float(np.linalg.norm(Q_fit - Q_ref, "fro") / denom))
    return err_max


def _structural_state_weight(vec, n_modal):
    """Fraction of eigenvector energy in [q; q_dot] (exclude lag substates)."""
    v = np.asarray(vec, dtype=complex).ravel()
    n_struct = 2 * int(n_modal)
    if v.size < n_struct or n_struct <= 0:
        return 1.0
    e_struct = float(np.sum(np.abs(v[:n_struct]) ** 2))
    e_total = float(np.sum(np.abs(v) ** 2))
    return e_struct / max(e_total, 1e-30)


def _filter_rfa_pool_roots(
    freq_pool,
    g_pool,
    eigs_pool,
    vecs_pool,
    omega_n,
    mode_indices_tracked,
    config,
    *,
    omega_wet_prev=None,
    omega_predicted=None,
    n_slots_min=2,
    n_modal=None,
):
    """
    Drop spurious high-/low-frequency roots before tracking (not unstable flutter roots).

    Band limits use the previous wet tracked frequencies when they are separated;
    always capped by the highest *tracked* dry mode so untracked modes (e.g. dry mode 3
    at ~44 Hz) never enter a 2-mode pool.
    """
    gamma_max = float(getattr(config, "rfa_pool_gamma_abs_max", 5.0))
    hi_frac = float(getattr(config, "rfa_pool_omega_max_frac", 1.35))
    lo_frac = float(getattr(config, "rfa_pool_omega_min_frac", 0.35))
    coalesce_rad = float(getattr(config, "rfa_pool_coalesce_sep_rad", 2.0))
    struct_w_min = float(getattr(config, "rfa_pool_structural_weight_min", 0.20))
    omega_n = np.asarray(omega_n, dtype=float).ravel()
    tracked_dry = omega_n[np.asarray(mode_indices_tracked, dtype=int)]
    omega_hi_dry = float(np.max(tracked_dry)) * hi_frac
    omega_lo_dry = float(np.min(tracked_dry)) * lo_frac
    n_modal_eff = int(n_modal) if n_modal is not None else int(omega_n.size)

    wet = np.asarray(omega_wet_prev, dtype=float).ravel() if omega_wet_prev is not None else np.array([])
    wet = wet[np.isfinite(wet) & (wet > 0)]
    pred = np.asarray(omega_predicted, dtype=float).ravel() if omega_predicted is not None else np.array([])
    pred = pred[np.isfinite(pred) & (pred > 0)]
    reacquire_band = float(getattr(config, "rfa_branch_reacquire_rel_band", 0.20))
    use_wet_band = wet.size >= 2 and (float(np.max(wet)) - float(np.min(wet))) >= coalesce_rad
    wet_lo_frac = float(getattr(config, "rfa_pool_wet_lo_frac", 0.85))

    def _apply(gamma_lim, omega_lo, omega_hi):
        n = len(np.asarray(freq_pool).ravel())
        keep = []
        for i in range(n):
            if np.isfinite(g_pool[i]) and abs(float(g_pool[i])) > gamma_lim:
                continue
            f_i = float(freq_pool[i])
            if f_i < omega_lo or f_i > omega_hi:
                continue
            if (
                struct_w_min > 0.0
                and vecs_pool is not None
                and vecs_pool.ndim == 2
                and i < vecs_pool.shape[1]
                and _structural_state_weight(vecs_pool[:, i], n_modal_eff) < struct_w_min
            ):
                continue
            keep.append(i)
        return keep

    omega_hi = omega_hi_dry
    omega_lo = omega_lo_dry
    if use_wet_band:
        # Extend (not shrink) the band so tracked wet roots above dry ω_max are not
        # dropped from the pool — otherwise MAC matching goes UNMATCHED and picks a wrong root.
        omega_hi = max(omega_hi, float(np.max(wet)) * hi_frac)
        omega_lo = min(omega_lo, float(np.min(wet)) * lo_frac)
    elif wet.size > 0:
        # Drop low-ω orphans when tracked branches sit at higher ω (coalescence band).
        omega_lo = max(omega_lo, float(np.min(wet)) * wet_lo_frac)
    if pred.size > 0:
        pred_hi = float(np.max(pred)) * (1.0 + reacquire_band)
        pred_lo = float(np.min(pred)) * (1.0 - reacquire_band)
        omega_hi = max(omega_hi, min(pred_hi, omega_hi_dry))
        omega_lo = min(omega_lo, max(pred_lo, omega_lo_dry * lo_frac))
        if wet.size == 0:
            use_wet_band = pred.size >= 2 and (
                float(np.max(pred)) - float(np.min(pred))
            ) >= coalesce_rad

    keep = _apply(gamma_max, omega_lo, omega_hi)
    if len(keep) < n_slots_min:
        keep = _apply(gamma_max, omega_lo_dry * 0.5, omega_hi_dry * 1.05)
    if len(keep) < n_slots_min:
        keep = _apply(max(gamma_max, 20.0), omega_lo_dry * 0.25, omega_hi_dry * 1.2)

    n = len(np.asarray(freq_pool).ravel())
    if n == 0 or not keep:
        return freq_pool, g_pool, eigs_pool, vecs_pool
    if len(keep) < n_slots_min and n >= n_slots_min:
        print(
            f"  [RFA pool filter] kept only {len(keep)}/{n} roots "
            f"(need {n_slots_min} for branch tracking) — using full {n}-root pool"
        )
        return freq_pool, g_pool, eigs_pool, vecs_pool
    if len(keep) < n:
        print(
            f"  [RFA pool filter] kept {len(keep)}/{n} roots "
            f"(|g|<={gamma_max:.1f}, ω∈[{omega_lo:.1f}, {omega_hi:.1f}] rad/s"
            f"{', wet band' if use_wet_band else ''})"
        )
    idx = np.asarray(keep, dtype=int)
    return (
        np.asarray(freq_pool, dtype=float).ravel()[idx],
        np.asarray(g_pool, dtype=float).ravel()[idx],
        np.asarray(eigs_pool)[idx],
        vecs_pool[:, idx] if vecs_pool.ndim == 2 else vecs_pool,
    )


def _assign_rfa_branches_predict_and_select(
    freq_pool,
    eigs_pool,
    raw_results,
    i_V,
    n_slots,
    *,
    V_current=None,
    delta=0.025,
    min_omega_sep_rad=1.0,
    mode_indices_tracked=None,
    omega_n=None,
    allow_coalescence=True,
):
    """
    PK-style predict-and-select on the full RFA wet pool at one airspeed.

    Allows branch slots to cross in frequency without coalescing onto the same root.
    """
    freq_pool = np.asarray(freq_pool, dtype=float).ravel()
    n_pool = freq_pool.size
    if i_V < 2 or not raw_results or n_pool < n_slots:
        return None

    prev_modes = raw_results[i_V - 1].get("modes", [])
    prev2_modes = raw_results[i_V - 2].get("modes", [])
    available = list(range(n_pool))
    picked = []
    min_sep = float(min_omega_sep_rad)

    def _too_close(omega_a, omega_b):
        if not (np.isfinite(omega_a) and np.isfinite(omega_b)):
            return False
        return abs(float(omega_a) - float(omega_b)) < min_sep

    for s in range(n_slots):
        if not available:
            picked.append(min(s, n_pool - 1))
            continue

        md_ref, _ = roger_mode_tracking.last_valid_branch_mode(raw_results, s)
        omega_prev = float(md_ref["omega"]) if md_ref is not None else np.nan
        p_v1 = md_ref.get("p") if md_ref is not None else None
        recovering = roger_mode_tracking.branch_slot_lost_at_previous(raw_results, s)

        p_v2 = None
        if i_V >= 2:
            p_hist = []
            for step in range(len(raw_results) - 1, -1, -1):
                modes2 = raw_results[step].get("modes", [])
                if s >= len(modes2):
                    continue
                md2 = modes2[s]
                p2 = md2.get("p", None)
                if md2.get("converged", False) and p2 is not None and np.isfinite(np.real(p2)):
                    p_hist.insert(0, complex(p2))
                if len(p_hist) >= 2:
                    break
            if len(p_hist) >= 2:
                p_v2 = p_hist[-2]

        if (
            recovering
            and V_current is not None
            and np.isfinite(float(V_current))
        ):
            omega_pred_v = roger_mode_tracking.predict_branch_omega_at_V(
                raw_results, s, float(V_current),
                omega_n=omega_n,
                mode_indices_tracked=mode_indices_tracked,
                max_struct_frac=1.35,
                partner_slot=(s - 1 if s > 0 else 1),
            )
            if omega_pred_v is not None and np.isfinite(omega_pred_v) and omega_pred_v > 0.0:
                omega_ext = float(omega_pred_v)
            elif np.isfinite(omega_prev):
                omega_ext = omega_prev
            else:
                pick = available[0]
                picked.append(int(pick))
                available.remove(int(pick))
                continue
        elif (
            p_v1 is not None
            and p_v2 is not None
            and np.isfinite(np.real(p_v1))
            and np.isfinite(np.real(p_v2))
        ):
            omega_ext = float(np.abs(np.imag(2.0 * complex(p_v1) - complex(p_v2))))
        elif np.isfinite(omega_prev):
            omega_ext = omega_prev
        else:
            omega_pred = roger_mode_tracking.predict_branch_omega_at_V(
                raw_results, s, V_current,
                omega_n=omega_n,
                mode_indices_tracked=mode_indices_tracked,
                partner_slot=(s - 1 if s > 0 else 1),
            )
            if omega_pred is None or not np.isfinite(omega_pred):
                omega_pred = roger_mode_tracking.predict_branch_omega_from_history(raw_results, s)
            if omega_pred is not None and np.isfinite(omega_pred):
                omega_ext = float(omega_pred)
            else:
                pick = available[0]
                picked.append(int(pick))
                available.remove(int(pick))
                continue

        def _rel_err(r, omega_ref):
            return abs(float(freq_pool[r]) - omega_ref) / max(omega_ref, 1e-6)

        candidates = [r for r in available if _rel_err(r, omega_ext) <= float(delta)]
        if np.isfinite(omega_prev):
            prev_band = [r for r in available if _rel_err(r, omega_prev) <= float(delta)]
            if prev_band:
                candidates = prev_band if not candidates else [
                    r for r in candidates if r in prev_band
                ] or prev_band

        search_set = candidates if candidates else available
        pick = min(search_set, key=lambda r: _rel_err(r, omega_ext))
        omega_pick = float(freq_pool[pick])

        # When bend/torsion coalesce, nearby roots are physical — do not jump to a
        # distant orphan just because two pool frequencies are closer than min_sep.
        if not allow_coalescence:
            for prev_pick in picked:
                if _too_close(omega_pick, float(freq_pool[prev_pick])):
                    others = [r for r in available if r != pick]
                    if others:
                        if np.isfinite(omega_prev):
                            pick = min(others, key=lambda r: _rel_err(r, omega_prev))
                        else:
                            pick = min(others, key=lambda r: _rel_err(r, omega_ext))
                        omega_pick = float(freq_pool[pick])
                    break

        picked.append(int(pick))
        available.remove(int(pick))

    print(
        f"  [RFA predict-and-select] pool indices {picked} → "
        f"ω = {[f'{float(freq_pool[i]):.3f}' for i in picked]} rad/s"
    )
    return np.asarray(picked, dtype=int)


def _rfa_pas_mac_tiebreak(
    pas_indices,
    freq_pool,
    vecs_pool,
    vecs_prev,
    *,
    degeneracy_rad=3.0,
    mac_margin=0.02,
    n_structural_dof=None,
):
    """
    When two wet roots are nearly degenerate, PAS can swap branch labels.
    Prefer the assignment with higher slot-wise MAC to the previous step.
    """
    pas_indices = np.asarray(pas_indices, dtype=int).ravel()
    if pas_indices.size < 2 or vecs_prev is None:
        return pas_indices
    if vecs_pool.ndim != 2 or vecs_prev.ndim != 2:
        return pas_indices

    omegas = [float(freq_pool[i]) for i in pas_indices]
    if max(omegas) - min(omegas) > float(degeneracy_rad):
        return pas_indices

    def _mac_score(indices):
        score = 0.0
        ncol = min(len(indices), vecs_prev.shape[1])
        for j in range(ncol):
            score += roger_mode_tracking.compute_mac(
                vecs_pool[:, indices[j]],
                vecs_prev[:, j],
                n_structural_dof=n_structural_dof,
            )
        return score

    direct = _mac_score(pas_indices)
    if pas_indices.size == 2:
        swapped = np.array([pas_indices[1], pas_indices[0]], dtype=int)
        swap_score = _mac_score(swapped)
        if swap_score > direct + float(mac_margin):
            print(
                f"  [RFA PAS MAC tie-break] {pas_indices.tolist()} → {swapped.tolist()} "
                f"(MAC {direct:.3f} → {swap_score:.3f})"
            )
            return swapped
    return pas_indices


def _roger_blag_beta_from_modes(V_flutter, f_modes_hz, semichord, n_lag):
    """
    Dimensionless Roger poles beta = k*b placed from modal reduced frequencies
    at the target flutter speed (Bash & Liu Fig. 8 style).

    Returns beta values (not [1/m]); use with rfa_blag_dimensionless=True.
    """
    f = np.asarray(f_modes_hz, dtype=float).ravel()
    if f.size == 0:
        raise ValueError("f_modes_hz must not be empty")
    if V_flutter <= 0 or semichord <= 0:
        raise ValueError("V_flutter and semichord must be positive")
    n_lag = int(n_lag)
    if n_lag < 1:
        raise ValueError("n_lag must be >= 1")

    beta = 2.0 * np.pi * f * float(semichord) / float(V_flutter)
    bmin, bmax = float(np.min(beta)), float(np.max(beta))
    lo = max(0.03, 0.35 * bmin)
    hi = max(lo * 2.0, 1.75 * bmax)

    if n_lag == 1:
        return np.array([np.sqrt(bmin * bmax)])
    if n_lag == 2:
        return np.array([lo, hi])
    if n_lag == 3:
        mid = float(np.sqrt(bmin * bmax))
        return np.array([lo, mid, hi])

    mid_lo = float(np.mean(beta[: max(1, beta.size // 2)]))
    mid_hi = float(np.mean(beta[max(1, beta.size // 2) :]))
    return np.array([lo, mid_lo, mid_hi, hi])


def _resolve_roger_blag_config(config, n_lag):
    """Return dimensionless beta poles from config (manual or auto)."""
    if getattr(config, "rfa_blag_auto", False):
        V_kn = float(getattr(config, "rfa_blag_V_flutter_kn", 83.0))
        V_f = V_kn * 0.514444
        f_ref = getattr(config, "rfa_blag_f_ref_hz", (2.0, 14.0, 27.0, 37.0))
        chord = float(getattr(config, "chord"))
        ref = getattr(config, "rfa_blag_ref_length", "semichord")
        if isinstance(ref, str) and ref.lower() in ("semichord", "half_chord", "c/2"):
            semichord = 0.5 * chord
        elif isinstance(ref, str) and ref.lower() in ("chord", "c"):
            semichord = chord
        else:
            semichord = float(ref)
        beta = _roger_blag_beta_from_modes(V_f, f_ref, semichord, n_lag)
        print(
            f"  [RFA poles] auto from V_f={V_kn:.1f} kn, f_ref={np.asarray(f_ref)} Hz "
            f"→ beta={np.array2string(beta, precision=3)}"
        )
        return beta

    blag = np.asarray(config.blag, dtype=float).ravel()
    if blag.size < n_lag:
        raise ValueError(f"config.blag must have at least {n_lag} entries (n_lag={n_lag})")
    return blag[:n_lag]


def _roger_lag_poles(config, k_list_V, n_lag):
    """
    Return Roger lag poles for state dynamics and (optionally) fit basis.

    Returns
    -------
    blag_state : ndarray
        Poles for lag ODE ``xa_dot = q_dot - V * blag_state * xa`` [1/m].
    ref_length : float or None
        Semichord used for dimensionless conversion (paper_zurich).
    beta_fit : ndarray or None
        Dimensionless β_l for Roger LSQ when convention is paper_zurich; else None.
    """
    paper = _is_paper_zurich_roger(config)

    if getattr(config, 'rfa_adaptive_blag', False):
        if paper:
            print(
                "  [WARN] paper_zurich + rfa_adaptive_blag: adaptive poles use k_pa grid; "
                "prefer fixed dimensionless config.blag."
            )
        k_lo = max(float(k_list_V[1]) if k_list_V.size > 1 else float(k_list_V[0]), 1e-6)
        blag_adaptive = np.logspace(np.log10(k_lo * 0.3), np.log10(k_list_V[-1] * 1.5), n_lag)
        return blag_adaptive, None, None

    beta = _resolve_roger_blag_config(config, n_lag)

    if paper:
        b = _roger_semichord_m(config)
        return beta / b, b, beta

    use_dimensionless = getattr(config, 'rfa_blag_dimensionless', False)
    if not use_dimensionless:
        return beta, None, None

    ref_length = _roger_semichord_m(config)
    return beta / ref_length, ref_length, None


def _capytaine_asymptotic_modal_matrices(omega_h, A_h, B_h, omega_ref, config):
    """
    Mean Capytaine added mass / radiation damping over the high-omega (asymptotic) band.

    Flutter reduced frequencies sit in the flat tail of A(omega), B(omega) (~0 for B).
    """
    omega_h = np.asarray(omega_h, dtype=float).ravel()
    A_h = np.asarray(A_h, dtype=float)
    B_h = np.asarray(B_h, dtype=float)
    omega_ref = np.asarray(omega_ref, dtype=float).ravel()
    om_lo = float(np.min(omega_h))
    om_hi = float(np.max(omega_h))
    frac = float(getattr(config, "capytaine_asymptotic_omega_frac", 0.5))
    w_cut = float(getattr(config, "capytaine_asymptotic_omega_min", np.nan))
    if not np.isfinite(w_cut):
        w_cut = om_lo + frac * (om_hi - om_lo)
    w_cut = max(w_cut, float(np.min(omega_ref)) * 0.8)
    mask = omega_h >= w_cut
    if not np.any(mask):
        n_tail = max(3, omega_h.size // 4)
        mask = np.zeros(omega_h.size, dtype=bool)
        mask[-n_tail:] = True
    A_asy = np.mean(A_h[mask], axis=0)
    B_asy = np.mean(B_h[mask], axis=0)
    print(
        f"[Capytaine asymptotic] ω >= {w_cut:.2f} rad/s ({np.sum(mask)}/{omega_h.size} samples)\n"
        f"  A_asy diagonal: {np.diag(A_asy)}\n"
        f"  B_asy diagonal: {np.diag(B_asy)}"
    )
    return A_asy, B_asy


def _resolve_capytaine_apply_indices(config, n_capy_modes, n_modal):
    """
    Modal DOF indices that receive non-zero Capytaine added mass / radiation damping.

    Default: all dry structural modes from the initial modal analysis
    (``0 .. num_modes_egv-1``), capped by the Capytaine tensor and modal basis size.
    DOF beyond that range (if any) stay zero — they are not physical dry modes.
    """
    raw = getattr(config, "capytaine_apply_mode_indices", None)
    if raw is None:
        n_struct = int(getattr(config, "num_modes_egv", n_modal))
        n_apply = min(n_struct, n_capy_modes, n_modal)
        indices = list(range(n_apply))
    else:
        indices = [int(i) for i in np.asarray(raw, dtype=int).ravel()]

    if not indices:
        raise ValueError(
            "capytaine_apply_mode_indices is empty. Increase num_modes_egv or "
            "set capytaine_apply_mode_indices explicitly."
        )
    if min(indices) < 0 or max(indices) >= n_modal:
        raise ValueError(
            f"capytaine_apply_mode_indices {indices} out of range for "
            f"n_modal={n_modal} (0..{n_modal - 1})"
        )
    if max(indices) >= n_capy_modes:
        raise ValueError(
            f"capytaine_apply_mode_indices {indices} exceed Capytaine matrix "
            f"dimension n_capy={n_capy_modes}"
        )
    return indices


def _resolve_capytaine_source_indices(config, n_capy_modes, apply_indices):
    """Indices inside the Capytaine NPZ (added_mass / added_damping) tensors."""
    raw = getattr(config, "capytaine_source_mode_indices", None)
    if raw is None:
        return list(range(n_capy_modes))
    src = [int(i) for i in np.asarray(raw, dtype=int).ravel()]
    if len(src) != len(apply_indices):
        raise ValueError(
            "capytaine_source_mode_indices must have the same length as "
            f"capytaine_apply_mode_indices ({len(apply_indices)}), got {len(src)}"
        )
    if min(src) < 0 or max(src) >= n_capy_modes:
        raise ValueError(
            f"capytaine_source_mode_indices {src} out of range for n_capy={n_capy_modes}"
        )
    return src


def _embed_capytaine_on_physical_modes(A_capy, B_capy, n_modal, apply_indices, source_indices):
    """
    Embed Capytaine matrices into the Roger modal basis.

    Only ``apply_indices`` rows/columns are filled from ``A_capy`` / ``B_capy``;
    all other DOFs (untracked / purely aeroelastic branches) remain zero.
    """
    A_capy = np.asarray(A_capy, dtype=float)
    B_capy = np.asarray(B_capy, dtype=float)
    n_modal = int(n_modal)
    A_emb = np.zeros((n_modal, n_modal), dtype=float)
    B_emb = np.zeros((n_modal, n_modal), dtype=float)

    for ii, mi in enumerate(apply_indices):
        si = source_indices[ii]
        for jj, mj in enumerate(apply_indices):
            sj = source_indices[jj]
            A_emb[mi, mj] = A_capy[si, sj]
            B_emb[mi, mj] = B_capy[si, sj]

    return A_emb, B_emb


def _capytaine_structural_dest_indices(config, mode_indices_tracked):
    """
    Modal-basis row/col indices where the structural Capytaine block is embedded.

    These are fixed dry-mode coordinates (``modes_to_analyze`` / ``mode_indices_tracked``),
    not MAC eigenvalue-pool indices. Pool tracking only labels which root is branch 1 vs 2
    in V–g output; it must not relocate A_∞ / B_∞ in the EOM matrix.
    """
    n_flutter = int(getattr(config, "num_modes_flutter_egv", len(mode_indices_tracked)))
    tracked = [int(i) for i in np.asarray(mode_indices_tracked, dtype=int).ravel()]
    if len(tracked) < n_flutter:
        raise ValueError(
            f"mode_indices_tracked {tracked} shorter than num_modes_flutter_egv={n_flutter}"
        )
    dest = tracked[:n_flutter]
    return dest


def _capytaine_structural_source_indices(config, n_capy):
    """Row/col indices inside the Capytaine tensor for structural flutter modes."""
    n_flutter = int(getattr(config, "num_modes_flutter_egv", 1))
    raw = getattr(config, "capytaine_source_mode_indices", None)
    if raw is None:
        src = list(range(min(n_flutter, int(n_capy))))
    else:
        src = [int(i) for i in np.asarray(raw, dtype=int).ravel()]
        if len(src) < n_flutter:
            raise ValueError(
                f"capytaine_source_mode_indices has {len(src)} entries but "
                f"num_modes_flutter_egv={n_flutter}"
            )
        src = src[:n_flutter]
    if not src or min(src) < 0 or max(src) >= int(n_capy):
        raise ValueError(
            f"capytaine structural source indices {src} out of range for n_capy={n_capy}"
        )
    return src


def _embed_capytaine_structural_at_pool_indices(
    A_capy,
    B_capy,
    n_modal,
    structural_src_indices,
    pool_dof_indices,
):
    """
    Build ``num_modes_egv × num_modes_egv`` Capytaine matrices filled with zeros.

    Only the ``num_modes_flutter_egv × num_modes_flutter_egv`` structural block is
    copied from ``A_capy`` / ``B_capy`` and placed at modal-basis row/col indices
    given by ``pool_dof_indices`` (typically ``modes_to_analyze`` dry-mode DOFs).
    """
    A_capy = np.asarray(A_capy, dtype=float)
    B_capy = np.asarray(B_capy, dtype=float)
    n_modal = int(n_modal)
    pool_dof_indices = [int(p) for p in np.asarray(pool_dof_indices, dtype=int).ravel()]
    structural_src_indices = [int(s) for s in structural_src_indices]

    if len(pool_dof_indices) != len(structural_src_indices):
        raise ValueError(
            "pool_dof_indices and structural_src_indices must have the same length: "
            f"{len(pool_dof_indices)} vs {len(structural_src_indices)}"
        )
    for p in pool_dof_indices:
        if p < 0 or p >= n_modal:
            raise ValueError(
                f"pool_dof_indices {pool_dof_indices} out of range for n_modal={n_modal}"
            )

    A_emb = np.zeros((n_modal, n_modal), dtype=float)
    B_emb = np.zeros((n_modal, n_modal), dtype=float)
    for si, p_row in zip(structural_src_indices, pool_dof_indices):
        for sj, p_col in zip(structural_src_indices, pool_dof_indices):
            A_emb[p_row, p_col] = A_capy[si, sj]
            B_emb[p_row, p_col] = B_capy[si, sj]
    return A_emb, B_emb


def _bootstrap_capytaine_pool_indices(
    config,
    V,
    rho,
    M_hat,
    C_hat,
    K_hat,
    B0,
    B1,
    B2,
    Blag,
    blag,
    n_modes_modal,
    N,
    omega_n,
    n_search_modes,
    omega_phys_min,
    mode_indices_tracked,
    use_b2,
):
    """
    First-velocity bootstrap: eigen-solve without Capytaine, then map each
    structural mode to the closest wet-pool root (same rule as STEP F init).
    """
    z = np.zeros((n_modes_modal, n_modes_modal), dtype=float)
    A_aug, *_ = _build_A_aug(
        config, V, rho, M_hat, C_hat, K_hat, B0, B1, B2, Blag, blag, n_modes_modal, N,
        M_hydro_add=z, C_hydro_add=z, apply_b2_mass=use_b2,
    )
    _, freq_pool, _, _ = _extract_flutter_modes(
        A_aug,
        n_search_modes,
        omega_n,
        return_eigenvectors=True,
        dimensionless_vgvf_results=config.dimensionless_vgvf_results,
        n_search_modes=n_search_modes,
        omega_phys_min=omega_phys_min,
    )
    return _init_tracking_indices_structural(freq_pool, mode_indices_tracked, omega_n)


def _capytaine_matrices_for_velocity_step(
    config,
    A_full,
    B_full,
    n_modal,
    i_V,
    pool_dof_indices,
    *,
    reason,
):
    """Build per-velocity ``M_hydro_add``, ``C_hydro_add`` for structural modes only."""
    struct_src = _capytaine_structural_source_indices(config, A_full.shape[0])
    pool_dof_indices = [int(p) for p in np.asarray(pool_dof_indices, dtype=int).ravel()]
    A_step, B_step = _embed_capytaine_structural_at_pool_indices(
        A_full, B_full, n_modal, struct_src, pool_dof_indices
    )
    n_flutter = len(struct_src)
    block_pairs = [
        (pool_dof_indices[s], pool_dof_indices[t])
        for s in range(n_flutter)
        for t in range(n_flutter)
    ]
    zero_dof = sorted(set(range(n_modal)) - set(pool_dof_indices))
    struct_map = list(zip(range(n_flutter), pool_dof_indices))
    print(
        f"  [Capytaine @ V step {i_V + 1}] {reason}: "
        f"structural→DOF {struct_map}, "
        f"non-zero block at {block_pairs}, zero DOF {zero_dof}"
    )
    return A_step, B_step, pool_dof_indices


def _load_capytaine_modal_radiation(config):
    capytaine_dir = os.path.join(config.paths["FSI"], "FLUID", "capytaine")
    if capytaine_dir not in sys.path:
        sys.path.insert(0, capytaine_dir)
    from capytaine_workflow import load_capytaine_bundle

    bundle = load_capytaine_bundle(config)
    capy_data = bundle["matrices"]
    capy_npz = bundle["path"]
    source = bundle.get("source_inputs") or {}
    if source:
        n_ev = int(np.asarray(source.get("source_eigenvalues", [])).size)
        n_bm = int(np.asarray(source.get("source_beam_positions", [])).shape[0])
        print(
            f"[Capytaine] Loaded matrices with source fingerprint "
            f"({n_ev} modes, {n_bm} beam nodes): {capy_npz}"
        )
    else:
        print(
            f"[Capytaine] Loaded matrices (no source-input metadata in file): {capy_npz}"
        )
    return capy_data, capy_npz


def _resolve_capytaine_added_mass_source(config) -> str:
    """'modal' (BEM face fields) or 'stripstyle' (rigid A_∞ lumped + Φᵀ M_add Φ)."""
    key = str(getattr(config, "capytaine_added_mass_source", "modal") or "modal").strip().lower()
    if key in ("stripstyle", "strip_style", "strip-style", "capy_strip", "striplike"):
        return "stripstyle"
    return "modal"


def _load_capytaine_stripstyle_matrices(config, structural_results):
    """
  Constant generalized added mass from Capytaine rigid-body ``A_∞`` + strip lumping.

  Returns ``(A_asy, B_asy, export_path_or_rigid_npz)`` with ``B_asy = 0`` (no modal
  radiation damping in this path; use empirical_fluid_damping if needed).
    """
    capytaine_dir = os.path.join(config.paths["FSI"], "FLUID", "capytaine")
    if capytaine_dir not in sys.path:
        sys.path.insert(0, capytaine_dir)
    from capytaine_strip_projection import load_or_compute_stripstyle_M_hat_added

    M_hat_added, _, rigid_path = load_or_compute_stripstyle_M_hat_added(
        structural_results, config, export_npz=True,
    )
    A_asy = np.asarray(M_hat_added, dtype=float)
    B_asy = np.zeros_like(A_asy)
    export_path = os.path.join(
        str(getattr(config, "capytaine_results_dir", "") or os.path.dirname(rigid_path)),
        "capytaine_stripstyle_M_hat_added.npz",
    )
    print(
        f"[Capytaine strip-style] A_∞ from rigid NPZ + strip lumping: {rigid_path}\n"
        f"  M_hat_added diagonal: {np.diag(A_asy)}\n"
        f"  B_∞ = 0 (strip-style path has no frequency-dependent modal radiation damping)"
    )
    return A_asy, B_asy, export_path


def _fit_roger_to_Q_modal(
    Q_modal_array,
    k_list,
    blag,
    Nm,
    N,
    row_weights=None,
    include_b2=True,
    *,
    roger_convention="panelaero_legacy",
    semichord=1.0,
    beta_lag=None,
):
    """
    Fit Roger RFA coefficients to modal GAF data.

    k_list is always PanelAero k_pa = omega/V (Qjj disk index).
    paper_zurich evaluates the rational basis at k* = k_pa * b (Kier eq. 19-20).

    Returns
    -------
    B0, B1, B2 : (Nm, Nm) real ndarray  (B2 is zero when include_b2 is False)
    Blag       : list of N (Nm, Nm) real ndarray
    """
    Nk = len(k_list)
    paper = roger_convention == "paper_zurich"
    b_len = float(semichord)
    lag_poles = np.asarray(beta_lag if (paper and beta_lag is not None) else blag, dtype=float)
    i_b2 = 2 if include_b2 else None
    i_lag0 = 3 if include_b2 else 2

    rows = []
    for k_pa in k_list:
        k = float(k_pa) * b_len if paper else float(k_pa)
        k2 = k ** 2
        if include_b2:
            re_row = [1.0, 0.0, -k2] + [k2 / (k2 + bp**2) for bp in lag_poles]
            im_row = [0.0, k, 0.0] + [k * bp / (k2 + bp**2) for bp in lag_poles]
        else:
            re_row = [1.0, 0.0] + [k2 / (k2 + bp**2) for bp in lag_poles]
            im_row = [0.0, k] + [k * bp / (k2 + bp**2) for bp in lag_poles]
        rows.append(re_row)
        rows.append(im_row)
    A_basis = np.array(rows, dtype=float)

    B0 = np.zeros((Nm, Nm))
    B1 = np.zeros((Nm, Nm))
    B2 = np.zeros((Nm, Nm))
    Blag = [np.zeros((Nm, Nm)) for _ in range(N)]

    for i in range(Nm):
        for j in range(Nm):
            q_ij = Q_modal_array[:, i, j]
            rhs = np.empty(2 * Nk)
            rhs[0::2] = q_ij.real
            rhs[1::2] = q_ij.imag
            if row_weights is not None:
                w_rows = np.repeat(np.asarray(row_weights, dtype=float), 2)
                sw = np.sqrt(np.maximum(w_rows, 1e-12))
                coeffs, *_ = np.linalg.lstsq(A_basis * sw[:, None], rhs * sw, rcond=None)
            else:
                coeffs, *_ = np.linalg.lstsq(A_basis, rhs, rcond=None)
            B0[i, j] = coeffs[0]
            B1[i, j] = coeffs[1]
            if include_b2:
                B2[i, j] = coeffs[i_b2]
            for l in range(N):
                Blag[l][i, j] = coeffs[i_lag0 + l]

    return B0, B1, B2, Blag


def _print_roger_lag_diagnostics(B0, B1, Blag, *, warn_ratio=100.0, warn_abs_max=1e5):
    """
    Check Roger lag-term magnitudes against B0 (stiffness-like term).

    Large Blag with alternating signs vs. small B0 usually indicates RFA
    overfitting — especially harmful when B2 is replaced by Capytaine added mass.
    """
    b0_norm = float(np.linalg.norm(B0, "fro"))
    b0_max = float(np.max(np.abs(B0))) if B0.size else 0.0
    b1_norm = float(np.linalg.norm(B1, "fro"))
    print(f"  STEP C.1: Roger coefficient norms — ||B0||={b0_norm:.4e}, ||B1||={b1_norm:.4e}")
    unhealthy = False
    for l, Bl in enumerate(Blag):
        bl_norm = float(np.linalg.norm(Bl, "fro"))
        bl_max = float(np.max(np.abs(Bl)))
        ratio = bl_norm / max(b0_norm, 1e-30)
        tag = "OK"
        if bl_max >= warn_abs_max or ratio >= warn_ratio:
            tag = "OVERFIT?"
            unhealthy = True
        print(
            f"    Blag[{l}]: ||·||={bl_norm:.4e}, max|·|={bl_max:.4e}, "
            f"||Blag||/||B0||={ratio:.2e}  [{tag}]"
        )
    if unhealthy:
        print(
            "  [WARN] Lag matrices much larger than B0 — likely RFA overfitting. "
            "Reduce n_lag and/or widen config.blag spacing."
        )
    elif b0_max > 0:
        print(f"  STEP C.1: Lag terms within ~{max(float(np.linalg.norm(Bl, 'fro')) / b0_norm for Bl in Blag):.1f}× of ||B0|| — fit looks healthy.")


def _interp_modal_matrix(omega_grid, matrix_grid, omega_query):
    """Linear interpolation of modal matrix over frequency."""
    w = float(np.clip(omega_query, omega_grid[0], omega_grid[-1]))
    i_hi = int(np.searchsorted(omega_grid, w, side="right"))
    if i_hi <= 0:
        return matrix_grid[0]
    if i_hi >= len(omega_grid):
        return matrix_grid[-1]
    i_lo = i_hi - 1
    w0 = omega_grid[i_lo]
    w1 = omega_grid[i_hi]
    alpha = 0.0 if abs(w1 - w0) < 1e-14 else (w - w0) / (w1 - w0)
    return (1.0 - alpha) * matrix_grid[i_lo] + alpha * matrix_grid[i_hi]


def _plot_added_mass_terms_vs_velocity(
    structural_results,
    config,
    V_arr,
    freq_arr,
    omega_h,
    A_h,
    capytaine_added_mass_used=None,
):
    """
    Plot m11, m22, m12, m21 used in the sweep.
    For RFA-PK, pass the asymptotic Capytaine matrix actually added to M_eff.
    """
    try:
        import matplotlib.pyplot as plt
        from FEA.structural_analysis import added_mass_projection as _strip_modal_added_mass

        V_arr = np.asarray(V_arr, dtype=float)
        freq_arr = np.asarray(freq_arr, dtype=float)  # (Nv, Nm)
        if freq_arr.ndim != 2 or freq_arr.shape[1] < 2:
            print("[Info] Skipping added-mass-terms plot: need at least 2 tracked modes.")
            return

        n_mode_cols = int(structural_results.dry_eigenvectors.shape[1])
        mode_idx_plot = sorted({int(i) for i in config.modes_to_analyze if int(i) < n_mode_cols})
        plot_modes = mode_idx_plot[:2]
        if len(plot_modes) < 2:
            print(
                "[Info] Skipping added-mass-terms plot: need two valid mode indices in "
                f"dry_eigenvectors (columns={n_mode_cols}, modes_to_analyze={config.modes_to_analyze})."
            )
            return

        # Strip-theory modal added mass (not used in RFA-PK branch, but plotted for comparison).
        _strip_proj = _strip_modal_added_mass(
            structural_results.M_global,
            structural_results.dry_eigenvectors,
            plot_modes,
            config,
            structural_results.total_dof,
            constrained_dofs=structural_results.constrained_dofs,
            apply_structural_dihedral=bool(getattr(config, "dihedral_rotate_beam", False)),
        )
        if _strip_proj is None:
            from FEA import structural_analysis as _fea_structural_mod

            raise TypeError(
                "added_mass_projection returned None (expected (M_hat_added, Phi_for_reduction)). "
                f"Loaded FEA.structural_analysis from {_fea_structural_mod.__file__}. "
                "Ensure that function ends with: return M_hat_added, Phi_for_reduction"
            )
        M_strip, _ = _strip_proj

        capy_terms = {"m11": [], "m12": [], "m21": [], "m22": []}
        if capytaine_added_mass_used is not None:
            A_used = np.asarray(capytaine_added_mass_used, dtype=float)
            if A_used.shape[0] < 2 or A_used.shape[1] < 2:
                print(
                    "[Info] Skipping added-mass-terms plot: Capytaine matrix must be at least 2x2."
                )
                return
            capy_mats = [A_used] * len(V_arr)
        else:
            # Fallback for frequency-dependent branches: evaluate at the first tracked mode.
            capy_mats = [
                _interp_modal_matrix(omega_h, A_h, float(freq_arr[i_v, 0]))
                for i_v in range(len(V_arr))
            ]

        for A_use in capy_mats:
            capy_terms["m11"].append(float(A_use[0, 0]))
            capy_terms["m12"].append(float(A_use[0, 1]))
            capy_terms["m21"].append(float(A_use[1, 0]))
            capy_terms["m22"].append(float(A_use[1, 1]))

        fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.2), sharex=True, squeeze=False)
        term_layout = [("m11", 0, 0, (0, 0)), ("m12", 0, 1, (0, 1)), ("m21", 1, 0, (1, 0)), ("m22", 1, 1, (1, 1))]

        for term, r, c, (ii, jj) in term_layout:
            ax = axes[r, c]
            ax.plot(V_arr, capy_terms[term], lw=1.2, label=f"{term} Capytaine")
            ax.axhline(float(M_strip[ii, jj]), color="tab:red", lw=1.2, label=f"{term} Strip")
            ax.set_title(term)
            ax.grid(True, alpha=0.3)
            ax.set_ylabel("M_added [kg]")
            ax.legend(fontsize=7)

        axes[1, 0].set_xlabel("Velocity V [m/s]")
        axes[1, 1].set_xlabel("Velocity V [m/s]")
        fig.suptitle("Added-mass terms vs velocity: Capytaine used in RFA-PK vs Strip theory", fontsize=11)

        out_dir = config.output_dir
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{config.name}_Madded_terms_vs_V_capytaine_vs_strip.png")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"✓ Saved added-mass terms vs velocity plot: {out_path}")
    except Exception as e:
        import traceback

        print(f"[Warning] Could not plot added-mass terms vs velocity: {e}")
        traceback.print_exc()


def _reconstruct_Q_modal(
    k,
    B0,
    B1,
    B2,
    Blag,
    blag,
    include_b2=True,
    *,
    roger_convention="panelaero_legacy",
    semichord=1.0,
    beta_lag=None,
):
    """Evaluate Q_modal from Roger coefficients at PanelAero grid index k_pa."""
    paper = roger_convention == "paper_zurich"
    b_len = float(semichord)
    k_eval = float(k) * b_len if paper else float(k)
    lag_poles = np.asarray(beta_lag if (paper and beta_lag is not None) else blag, dtype=float)
    k2 = k_eval ** 2
    Q = B0 + 1j * k_eval * B1
    if include_b2:
        Q -= k2 * B2
    for Bl, bp in zip(Blag, lag_poles):
        Q += Bl * (k2 + 1j * k_eval * bp) / (k2 + bp**2)
    return Q


def _build_A_aug(config, V, rho, M_hat, C_hat, K_hat, B0, B1, B2, Blag, blag, Nm, N,
                 M_hydro_add=None, C_hydro_add=None, apply_b2_mass=None):
    """
    Assemble the augmented aeroelastic state matrix.

    State vector:  X = [q(Nm), q_dot(Nm), xa_1(Nm), ..., xa_N(Nm)]
    Size:          (2 + N) * Nm

    EOM:
        M_eff . q_ddot = -K_eff . q  -  C_eff . q_dot  +  sum_i q_dyn*Blag[i]*xa_i
        xa_i_dot       = q_dot  -  V*blag[i]*xa_i

    paper_zurich: M_eff = M_hat - 0.5*rho*b^2*B2; C_eff = C_hat - s*0.5*rho*V*b*B1 (eq. 16)
    panelaero_legacy: M_eff = M_hat - 0.5*rho*B2; C_eff = C_hat - s*0.5*rho*V*B1*b
    K_eff = K_hat - 0.5*rho*V^2*B0
    q_dyn = 0.5*rho*V^2

    ``aero_im_Q_scale`` multiplies the B1 (Im Q) damping term after the
    unscaled Roger fit. If ``aero_im_Q_scale_lags`` is True, the same factor
    also multiplies the lag-state aerodynamic forcing ``q_dyn * Blag``.
    """
    n     = Nm
    q_dyn = 0.5 * rho * V ** 2
    n_tot = (2 + N) * n
    paper = _is_paper_zurich_roger(config)
    semichord = _roger_semichord_m(config)

    # Hydroelastic correction: added mass in M_hat (strip / Capytaine), not DLM B2.
    # Set rfa_apply_b2_mass=True only for pure-DLM RFA without external added mass.
    apply_b2 = getattr(config, 'rfa_apply_b2_mass', False)
    
    if apply_b2:
        if paper:
            M_eff = M_hat - 0.5 * rho * (semichord ** 2) * B2
        else:
            M_eff = M_hat - 0.5 * rho * B2
    else:
        M_eff = np.array(M_hat, copy=True)
    if M_hydro_add is not None:
        M_eff = M_eff + np.asarray(M_hydro_add, dtype=float)
    aero_im_scale = aero_q_scaling.aero_im_Q_scale(config)
    
    C_eff = C_hat - aero_im_scale * 0.5 * rho * V * semichord * B1
    if C_hydro_add is not None:
        C_eff = C_eff + np.asarray(C_hydro_add, dtype=float)
    C_emp = empirical_fluid_damping.modal_empirical_C(config, V, M_eff)
    if np.any(C_emp != 0.0):
        C_eff = C_eff + C_emp
   
   
    K_eff = K_hat - 0.5 * rho * V**2 * B0
    lag_force_scale = aero_q_scaling.roger_lag_force_scale(config)

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
        A_aug[n:2*n, c0:c0+n] = M_eff_inv @ (lag_force_scale * q_dyn * Blag[i])

    # [2+i, 1]: q_dot drives lag state (velocity block, NOT displacement)
    # [2+i, 2+i]: lag pole -V * blag[i]
    for i in range(N):
        r0 = (2 + i) * n
        A_aug[r0:r0+n, n:2*n]   = np.eye(n)                  # from q_dot
        A_aug[r0:r0+n, r0:r0+n] = -V * blag[i] * np.eye(n)  # pole

    return A_aug, M_eff, C_eff, K_eff


def _roger_fit_mode_indices(config, mode_indices_tracked, n_modes_modal):
    """Modal indices used for Roger LSQ (subset avoids overfitting unused modes)."""
    if not getattr(config, "rfa_fit_tracked_modes_only", True):
        return list(range(int(n_modes_modal)))
    tracked = [int(i) for i in mode_indices_tracked if 0 <= int(i) < int(n_modes_modal)]
    if not tracked:
        return list(range(int(n_modes_modal)))
    if len(tracked) < int(n_modes_modal):
        return tracked
    return list(range(int(n_modes_modal)))


def _embed_roger_coefficients(B0_sub, B1_sub, B2_sub, Blag_sub, fit_indices, n_modal):
    """Place fitted n_fit×n_fit Roger blocks into the full n_modal basis."""
    n_modal = int(n_modal)
    idx = np.asarray(fit_indices, dtype=int)
    B0 = np.zeros((n_modal, n_modal), dtype=float)
    B1 = np.zeros((n_modal, n_modal), dtype=float)
    B2 = np.zeros((n_modal, n_modal), dtype=float)
    Blag = [np.zeros((n_modal, n_modal), dtype=float) for _ in Blag_sub]
    sl = np.ix_(idx, idx)
    B0[sl] = B0_sub
    B1[sl] = B1_sub
    B2[sl] = B2_sub
    for l, Bl in enumerate(Blag_sub):
        Blag[l][sl] = Bl
    return B0, B1, B2, Blag


def _partner_omega_for_reacquire(tracking_indices, partner_slot, freq_pool, freq_V_prev):
    """Current-step partner ω from assigned pool index, else previous-step reference."""
    if (
        partner_slot is not None
        and 0 <= int(partner_slot) < len(tracking_indices)
        and int(tracking_indices[partner_slot]) >= 0
    ):
        return float(freq_pool[int(tracking_indices[partner_slot])])
    if (
        freq_V_prev is not None
        and partner_slot is not None
        and 0 <= int(partner_slot) < len(freq_V_prev)
        and np.isfinite(freq_V_prev[partner_slot])
        and float(freq_V_prev[partner_slot]) > 0.0
    ):
        return float(freq_V_prev[partner_slot])
    return None


def _rfa_reacquire_kwargs(config):
    return dict(
        rel_band=float(getattr(config, "rfa_branch_reacquire_rel_band", 0.20)),
        rel_band_growth=float(
            getattr(config, "rfa_branch_reacquire_rel_band_growth", 0.04)
        ),
        rel_band_max=float(getattr(config, "rfa_branch_reacquire_rel_band_max", 0.35)),
        min_mac=float(getattr(config, "rfa_branch_reacquire_min_mac", 0.65)),
        w_freq=float(getattr(config, "rfa_w_freq", 0.8)),
        w_mac=float(getattr(config, "rfa_w_mac", 0.6)),
        max_struct_frac=float(getattr(config, "rfa_branch_guard_max_struct_frac", 1.35)),
        min_partner_sep_frac=float(
            getattr(config, "rfa_branch_reacquire_min_partner_sep_frac", 0.08)
        ),
        prefer_partner_streak=int(
            getattr(config, "rfa_branch_reacquire_prefer_partner_streak", 1)
        ),
        mac_reference=str(getattr(config, "rfa_mac_reference", "previous_velocity")),
    )


def _rfa_mac_reference_kw(config):
    return dict(mac_reference=getattr(config, "rfa_mac_reference", "previous_velocity"))


def _reject_invalid_rfa_mac_assignments(
    tracking_indices,
    freq_pool,
    freq_V_prev,
    raw_results,
    mode_indices_tracked,
    omega_n,
    config,
):
    """Reject MAC assignments above structural ω cap or with excessive jump while recovering."""
    struct_frac = float(getattr(config, "rfa_branch_guard_max_struct_frac", 1.35))
    max_jump = float(getattr(config, "rfa_branch_guard_max_rel_jump", 0.30))
    omega_n = np.asarray(omega_n, dtype=float).ravel()
    rejected = set()
    for i_prev in range(len(tracking_indices)):
        i_curr = int(tracking_indices[i_prev])
        if i_curr < 0:
            continue
        f_pick = float(freq_pool[i_curr])
        m_struct = int(mode_indices_tracked[i_prev])
        omega_cap = float(omega_n[m_struct]) * struct_frac
        partner_slot = i_prev - 1 if i_prev > 0 else (i_prev + 1 if i_prev + 1 < len(tracking_indices) else None)
        partner_omega = None
        if (
            partner_slot is not None
            and 0 <= int(partner_slot) < len(tracking_indices)
            and int(tracking_indices[partner_slot]) >= 0
        ):
            partner_omega = float(freq_pool[int(tracking_indices[partner_slot])])
        if f_pick > omega_cap * 1.01:
            high_orphan = (
                partner_omega is not None
                and int(i_prev) > int(partner_slot)
                and f_pick > float(partner_omega) * 1.35
            )
            if high_orphan:
                print(
                    f"  [INFO] MAC match rejected branch {i_prev}: "
                    f"ω_pick={f_pick:.3f} > ω_struct_cap={omega_cap:.3f} (orphan above partner)"
                )
                tracking_indices[i_prev] = -1
                rejected.add(int(i_prev))
                continue
        if roger_mode_tracking.branch_slot_lost_at_previous(raw_results, i_prev):
            f_prev = float(freq_V_prev[i_prev])
            if np.isfinite(f_prev) and f_prev > 0.0:
                rel = abs(f_pick - f_prev) / f_prev
                if rel > max_jump:
                    print(
                        f"  [INFO] MAC match rejected branch {i_prev}: "
                        f"Δω/ω_prev={rel:.3f} while recovering from mode_lost"
                    )
                    tracking_indices[i_prev] = -1
                    rejected.add(int(i_prev))
    return rejected


def _init_tracking_indices_structural(freq_pool, mode_indices_tracked, omega_n):
    """First-velocity branch seeds: global min-cost match pool roots → dry ω_n."""
    freq_pool = np.asarray(freq_pool, dtype=float).ravel()
    omega_n = np.asarray(omega_n, dtype=float).ravel()
    tracked = [int(i) for i in mode_indices_tracked]
    n_tracked = len(tracked)
    n_pool = int(freq_pool.size)
    if n_pool == 0 or n_tracked == 0:
        return np.zeros(n_tracked, dtype=int)
    if n_pool < n_tracked:
        return np.arange(n_tracked, dtype=int)

    cost = np.zeros((n_tracked, n_pool), dtype=float)
    for j, mi in enumerate(tracked):
        target = float(omega_n[mi])
        cost[j, :] = np.abs(freq_pool - target) / max(target, 1e-12)

    try:
        from scipy.optimize import linear_sum_assignment
        _, col_ind = linear_sum_assignment(cost)
        return np.asarray(col_ind[:n_tracked], dtype=int)
    except Exception:
        indices = []
        used = set()
        for mode_idx in tracked:
            target = float(omega_n[mode_idx])
            order = np.argsort(np.abs(freq_pool - target))
            pick = next((int(i) for i in order if int(i) not in used), int(order[0]))
            used.add(pick)
            indices.append(pick)
        return np.asarray(indices, dtype=int)


def _finalize_rfa_tracking_indices(
    tracking_indices,
    freq_pool,
    mode_indices_tracked,
    omega_n,
):
    """
    Ensure each tracked branch maps to a distinct pool root when the pool allows it.
    """
    freq_pool = np.asarray(freq_pool, dtype=float).ravel()
    n_tracked = len(mode_indices_tracked)
    idx = np.asarray(tracking_indices, dtype=int).ravel()[:n_tracked]
    if idx.size < n_tracked:
        idx = _init_tracking_indices_structural(freq_pool, mode_indices_tracked, omega_n)
    if len(freq_pool) >= n_tracked and len(set(idx.tolist())) < n_tracked:
        print(
            "  [WARN] Duplicate pool indices in branch assignment; "
            "re-seeding from structural ω_n on current pool"
        )
        idx = _init_tracking_indices_structural(freq_pool, mode_indices_tracked, omega_n)
    return idx


def _pick_pool_index_for_branch(
    branch_idx,
    mode_indices_tracked,
    omega_n,
    freq_pool,
    vecs_pool,
    vecs_prev_branch,
    freq_prev_branch,
    used_current,
    *,
    freq_margin,
    w_freq,
    w_mac,
    n_structural_dof=None,
):
    """
    Assign a candidate pool root to tracked branch ``branch_idx`` (PK-style window).
    """
    freq_pool = np.asarray(freq_pool, dtype=float).ravel()
    mode_struct = int(mode_indices_tracked[branch_idx])
    omega_n = np.asarray(omega_n, dtype=float).ravel()
    omega_ref = float(freq_prev_branch) if np.isfinite(freq_prev_branch) and freq_prev_branch > 0 else float(
        omega_n[mode_struct]
    )

    best_i = None
    best_cost = np.inf
    for i in range(len(freq_pool)):
        if int(i) in used_current:
            continue
        f_i = float(freq_pool[i])
        if f_i <= 0.0:
            continue
        freq_err = abs(f_i - omega_ref) / max(omega_ref, 1e-12)
        cost = float(w_freq) * freq_err
        if vecs_prev_branch is not None and vecs_pool is not None:
            mac_i = roger_mode_tracking.compute_mac(
                vecs_pool[:, i], vecs_prev_branch, n_structural_dof=n_structural_dof
            )
            cost += float(w_mac) * (1.0 - mac_i)
        if freq_err <= float(freq_margin) and cost < best_cost:
            best_cost = cost
            best_i = int(i)

    if best_i is not None:
        return best_i

    # Widen search: closest unused root to reference (never arbitrary leftover index).
    best_i = None
    best_err = np.inf
    for i in range(len(freq_pool)):
        if int(i) in used_current:
            continue
        f_i = float(freq_pool[i])
        if f_i <= 0.0:
            continue
        freq_err = abs(f_i - omega_ref) / max(omega_ref, 1e-12)
        if freq_err < best_err:
            best_err = freq_err
            best_i = int(i)
    if best_i is not None:
        return best_i
    for i in range(len(freq_pool)):
        if int(i) not in used_current:
            return int(i)
    # Pool smaller than branch count: allow closest root even if already used.
    order = np.argsort(np.abs(freq_pool - omega_ref))
    return int(order[0]) if order.size else 0


def _extract_flutter_modes(
    A_aug,
    Nm,
    omega_n,
    return_eigenvectors=False,
    vecs_previous=None,
    freqs_previous=None,
    dimensionless_vgvf_results=False,
    n_search_modes=None,
    omega_phys_min=None,
):


    # Compute eigendecomposition (eigenvectors needed for MAC)
    eigenvalues, eigenvectors = np.linalg.eig(A_aug)

    # Keep oscillatory modes only (discard real lag poles at ~ -V*blag)
    omega_n = np.asarray(omega_n, dtype=float).ravel()
    if omega_phys_min is None:
        omega_phys_min = 0.1 * float(omega_n[0]) if omega_n.size else 1.0
    threshold = max(float(omega_phys_min), 0.1 * float(omega_n[0]) if omega_n.size else 1.0)
    osc_mask = np.abs(np.imag(eigenvalues)) > threshold
    pos_mask = np.imag(eigenvalues) > 0          # one of each conjugate pair

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

    n_search = int(n_search_modes) if n_search_modes is not None else int(Nm)
    n_search = max(int(Nm), n_search)
    n_structural_modes = int(len(np.asarray(omega_n).ravel()))

    # Choose modes: MAC-based (if previous available) or frequency-based
    if vecs_previous is not None and freqs_previous is not None and len(eigs_filt) >= Nm:
        # MAC-based selection: search among 2*Nm candidates
        from COUPLING.hydroelastic_utl import roger_mode_tracking
        
        n_candidates = min(len(eigs_filt), 2 * n_search)
        indices_best = roger_mode_tracking._select_modes_by_mac(
            eigs_filt[:n_candidates],
            vecs_filt[:, :n_candidates],
            freqs_filt[:n_candidates],
            eigs_filt,
            vecs_filt,
            freqs_previous,
            vecs_previous,
            Nm,
            n_structural_dof=2 * n_structural_modes,
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
    g_V    =  np.real(eigs_selected) / np.maximum(np.abs(np.imag(eigs_selected)), 1e-12)
    g_V_dimensional =  np.real(eigs_selected)  # in rad/s, for diagnostics

    # for paper conventions:
    if not dimensionless_vgvf_results:
        g_V = g_V_dimensional

    freq_V = np.abs(np.imag(eigs_selected))  # frequencies in rad/s

    if return_eigenvectors:
        return g_V.real, freq_V.real, vecs_selected, eigs_selected
    else:
        return g_V.real, freq_V.real


def _extract_all_oscillatory_eigenvalues(A_aug, omega_n, omega_phys_min=None):
    """
    Full sorted wet spectrum at one airspeed (all physical oscillatory roots of A_aug).

    Used only for untracked-root diagnostics/plots — not for branch tracking.
    """
    eigenvalues, _ = np.linalg.eig(A_aug)
    omega_n = np.asarray(omega_n, dtype=float).ravel()
    if omega_phys_min is None:
        omega_phys_min = 0.1 * float(omega_n[0]) if omega_n.size else 1.0
    threshold = max(float(omega_phys_min), 0.1 * float(omega_n[0]) if omega_n.size else 1.0)
    imag = np.imag(eigenvalues)
    mask = (np.abs(imag) > threshold) & (imag > 0)
    eigs = eigenvalues[mask]
    if eigs.size == 0:
        return eigs
    order = np.argsort(np.abs(np.imag(eigs)))
    return eigs[order]


def _build_wet_spectrum_with_vectors(
    A_aug,
    omega_n,
    tracked_omegas,
    freq_margin,
    *,
    omega_phys_min=None,
    dimensionless_vgvf_results=True,
):
    """
    Classified wet spectrum at one V with eigenvectors attached to each root.

    ``tracked_omegas`` lists frequencies of the main MAC-tracked branches (rad/s).
    """
    eigenvalues, eigenvectors = np.linalg.eig(A_aug)
    omega_n = np.asarray(omega_n, dtype=float).ravel()
    if omega_phys_min is None:
        omega_phys_min = 0.1 * float(omega_n[0]) if omega_n.size else 1.0
    threshold = max(float(omega_phys_min), 0.1 * float(omega_n[0]) if omega_n.size else 1.0)
    margin = float(freq_margin)
    tracked_omegas = [float(w) for w in np.asarray(tracked_omegas, dtype=float).ravel() if w > 0]

    roots = []
    for idx, lam in enumerate(eigenvalues):
        omega = float(np.abs(np.imag(lam)))
        if omega < threshold or np.imag(lam) <= 0:
            continue
        sigma = float(np.real(lam))
        gamma = sigma / omega if omega > 1e-6 else 0.0
        if not dimensionless_vgvf_results:
            gamma = sigma

        pk_tracked = any(
            abs(omega - om_t) <= margin * max(om_t, 1e-6) for om_t in tracked_omegas
        )
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
                "vec": eigenvectors[:, idx].copy(),
            }
        )

    roots.sort(key=lambda r: r["omega"])
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


def solve(config, structural_results, coupling_results, aerogrid=None):

    print("\n--- Solving for Flutter ---")

    aero_method = str(getattr(config, "aero_method", "DLM")).upper()
    # SHARPy modal M̂/K̂ (and Φ for DLM) whenever the Quvlm NPZ carries them.
    if getattr(config, "quvlm_sharpy_structural_basis", False):
        from .hydroelastic_utl import quvlm_basis
        structural_results = quvlm_basis.apply_sharpy_structural_basis(
            structural_results, config
        )

    if config.pk_method:
        return _solve_pk_method(config, structural_results, coupling_results, aerogrid)
    
    if config.roger_fit:
        return _solve_roger_fit_method(config, config.V_list, structural_results, coupling_results, aerogrid)

    if config.RFA_PK_method:
        return _solve_RFA_PK_method(config, config.V_list, structural_results, coupling_results, aerogrid)

    else:
        raise ValueError("No flutter solution method selected in config.")


def _run_roger_rfa_sweep(
    config,
    V_list,
    structural_results,
    coupling_results,
    aerogrid,
    *,
    M_hydro_add=None,
    C_hydro_add=None,
    capytaine_A_full=None,
    capytaine_B_full=None,
    capytaine_static_apply_idx=None,
    capytaine_static_source_idx=None,
    method_label="Roger RFA",
):
    """
    Direct Roger RFA flutter sweep (eigenvalues of A_aug at each V, MAC tracking).

    Capytaine: pass ``capytaine_A_full`` / ``capytaine_B_full`` for per-velocity
    structural embedding (``num_modes_flutter_egv`` block at tracked pool DOF).
    Legacy constant matrices via ``M_hydro_add`` / ``C_hydro_add``.
    """
    from Qjj.precompute_qjj import interp_qjj_from_disk_old, open_qjj_index_old

    print(f"Using {method_label} — direct eigenvalue sweep (no PK iteration).")
    static_runner = _create_static_hydroelastic_runner(
        config, structural_results, coupling_results, aerogrid
    )
    cp_plotter = _create_cp_snapshot_plotter(
        config, aerogrid, V_list, structural_results, coupling_results
    )

    # Unpack structural data
    M_hat   = structural_results.M_hat
    C_hat   = structural_results.C_hat
    K_hat   = structural_results.K_hat
    omega_n = np.sqrt(np.abs(structural_results.dry_eigenvalues))   # rad/s
    mode_indices_tracked = [int(i) for i in getattr(config, "modes_to_analyze", [])]
    n_modes_total = int(len(np.asarray(structural_results.dry_eigenvalues).ravel()))
    if n_modes_total < 1:
        raise ValueError("No dry modes available for flutter analysis.")
    if not mode_indices_tracked:
        raise ValueError("config.modes_to_analyze is empty. Define at least one tracked mode.")
    if max(mode_indices_tracked) >= n_modes_total or min(mode_indices_tracked) < 0:
        raise ValueError(
            f"modes_to_analyze={mode_indices_tracked} out of range for "
            f"{n_modes_total} dry modes (0..{n_modes_total-1})."
        )
    Phi     = structural_results.dry_eigenvectors[:, :n_modes_total]

    # Unpack coupling data
    Z           = coupling_results['Z']
    Z_qs        = coupling_results['Z_qs']
    Z_force     = coupling_results.get('Z_force', None)
    panel_areas = aerogrid['A']
    A_diag      = np.diag(panel_areas)

    rho     = config.rho_f[config.fluid]
    c_sound = config.c_sound[config.fluid]
    n_modes_tracked = len(mode_indices_tracked)
    n_modes_modal = n_modes_total
    N       = config.n_lag
    n_search_cfg = int(getattr(config, "rfa_pool_n_roots", 0) or 0)
    if n_search_cfg > 0:
        n_search_modes = n_search_cfg
    else:
        # Augmented state has (2+N)*Nm DOFs → search tracked modes + lag band, not only Nm roots.
        n_search_modes = max(n_modes_tracked + int(N), 2 * n_modes_tracked)
    n_search_modes = max(n_modes_tracked, min(n_search_modes, n_modes_modal + int(N)))

    aero_method = str(getattr(config, "aero_method", "DLM")).upper()
    use_uvlm_q = aero_method == "UVLM"
    quvlm_k_list = None
    quvlm_Q_tbl = None
    if use_uvlm_q:
        npz_path = getattr(config, "quvlm_npz_path", None)
        if not npz_path:
            raise ValueError("aero_method='UVLM' with roger_fit requires config.quvlm_npz_path")
        _uq = np.load(npz_path, allow_pickle=True)
        quvlm_k_list = np.asarray(_uq["k_list"], dtype=float).ravel()
        quvlm_Q_tbl = np.asarray(_uq["Q"])
        if quvlm_Q_tbl.shape[1] < n_modes_modal:
            raise ValueError(
                f"Quvlm table has {quvlm_Q_tbl.shape[1]} modes < n_modes_modal={n_modes_modal}"
            )
        quvlm_Q_tbl = quvlm_Q_tbl[:, :n_modes_modal, :n_modes_modal]
        qjj_k_grid = np.unique(quvlm_k_list)
        print(
            f"[RFA UVLM] Q(k) from {npz_path}: {len(qjj_k_grid)} k-points, "
            f"nm={n_modes_modal} (no DLM Qjj / no p-k loop)"
        )
    else:
        qjj_k_grid, qjj_Ma_grid, _, _ = open_qjj_index_old(config.qjj_dir)
        qjj_k_grid = np.asarray(qjj_k_grid, dtype=float)
        qjj_k_grid = np.unique(qjj_k_grid)
    if qjj_k_grid.size < 2:
        raise ValueError("Qjj/Quvlm index must contain at least 2 k nodes.")
    k_data_min = float(qjj_k_grid[0])
    k_data_max = float(qjj_k_grid[-1])

    # Pre-compute once outside V loop (does not depend on V). Only the DLM
    # branch below projects panel forces through Phi; aero_method='UVLM' works
    # entirely in modal coordinates (Quvlm) and never touches this. Skipping it
    # for UVLM also avoids a shape mismatch when quvlm_sharpy_structural_basis
    # replaces Phi with the identity (SHARPy modal coords, not physical DOFs).
    Z_force_modal = None if use_uvlm_q else (Z_force @ Phi)  # shape: (Np, Nm)

    g_sweep    = []
    freq_sweep = []
    V_sweep    = []
    vecs_sweep = []  # Store eigenvectors for MAC tracking
    eigs_sweep = []  # Store eigenvalues for diagnostics
    raw_results = []  # PK-like snapshots used by post-processing plots
    mac_history = []  # Store MAC matrices between consecutive velocity steps
    orphan_tracker = (
        orphan_mode_tracking.OrphanBranchTracker(config)
        if getattr(config, "orphan_conditional_mac", False)
        else None
    )
    orphan_g_sweep = []
    orphan_f_sweep = []
    _collect_rfa_matrix_history = (
        getattr(config, 'plot_DLM_participants', False)
        or getattr(config, 'plot_stiffness_damping_contributions', True)
    )
    dlm_participants_history = [] if _collect_rfa_matrix_history else None

    # Hydroelastic correction: the Roger BASIS always needs the B2 (k^2) term to
    # cleanly separate the non-circulatory DLM inertia from A0/A1/lags — in water
    # Re(Q) is dominated by the -k^2 trend and, without B2 in the basis, it
    # pollutes the circulatory coefficients. Whether the fitted B2 is then USED
    # in M_eff is a separate decision (rfa_apply_b2_mass): with strip/Capytaine
    # added mass in M_hat the DLM B2 is fitted and DISCARDED.
    apply_b2 = getattr(config, 'rfa_apply_b2_mass', False)
    fit_b2 = bool(getattr(config, 'rfa_fit_b2', True))
    use_b2 = apply_b2
    if M_hydro_add is not None and apply_b2:
        print(
            "[RFA WARN] rfa_apply_b2_mass=True with external hydro added mass: "
            "M_eff will include both -0.5*rho*B2 and M_hydro_add. "
            "Use this only for explicit sensitivity checks because it can double-count inertia."
        )
    if config.added_mass_strip_theory and apply_b2:
        print(
            "[RFA WARN] strip/Capytaine added mass is in M_hat but rfa_apply_b2_mass=True — "
            "double-counts inertia. Use rfa_apply_b2_mass=False (hydroelastic correction)."
        )
    if not apply_b2:
        print(
            "[RFA] Hydroelastic correction: M_eff = M_hat (strip/Capytaine AM); "
            f"Roger fit {'WITH' if fit_b2 else 'without'} B2 in the basis (B2 discarded in EOM); "
            "A0/K_aero, A1/C_aero, lags = circulatory DLM."
        )
    else:
        print(
            "[RFA] rfa_apply_b2_mass=True: Roger fit includes B2 and "
            "M_eff = M_hat - 0.5*rho*B2 (+ external hydro AM if provided)."
        )
    if apply_b2 and not fit_b2:
        raise ValueError("rfa_apply_b2_mass=True requires rfa_fit_b2=True (cannot apply a B2 that is not fitted).")

    aero_q_scaling.log_aero_im_Q_scale_once(config, context="Roger RFA sweep")

    roger_conv = _rfa_roger_convention(config)
    semichord_b = _roger_semichord_m(config)
    print(
        f"\n[RFA] Roger convention: {roger_conv} "
        f"(Qjj load/downwash: k_pa=ω/V; fit/EOM: "
        f"{'k*=ωb/V, ŝ=sb/V' if roger_conv == 'paper_zurich' else 'k_pa basis'}, b={semichord_b:.6g} m)"
    )

    print(f"\nRoger RFA sweep over {len(V_list)} velocities with MAC-based mode tracking...")
    print(
        f"  Modal basis (dry transform): {n_modes_modal} modes "
        f"(num_modes_egv={getattr(config, 'num_modes_egv', n_modes_total)}), "
        f"candidate roots searched: {n_search_modes}, "
        f"tracked branches: {n_modes_tracked} -> indices {mode_indices_tracked}"
    )
    if n_search_modes < n_modes_modal:
        print(
            f"  [INFO] Searching only first {n_search_modes} oscillatory roots out of "
            f"{n_modes_modal} modal basis modes."
        )
    else:
        print(
            f"  [INFO] Searching full modal root pool ({n_search_modes} roots)."
        )
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
    fit_indices = _roger_fit_mode_indices(config, mode_indices_tracked, n_modes_modal)
    n_fit = len(fit_indices)
    if n_fit < n_modes_modal:
        print(
            f"  [INFO] Roger fit on {n_fit} tracked mode(s) {fit_indices} "
            f"(embedded in {n_modes_modal}×{n_modes_modal} basis; untracked modes: no DLM Roger terms)."
        )
    omega_phys_min = float(
        np.min(omega_n[np.asarray(mode_indices_tracked, dtype=int)])
    ) * float(getattr(config, "rfa_omega_phys_min_frac", 0.45))
    n_mac_dof = 2 * n_modes_modal if getattr(config, "rfa_mac_structural_dof", True) else None

    use_pool_embedding = bool(getattr(config, "capytaine_structural_pool_embedding", True))
    use_dynamic_capy = (
        use_pool_embedding
        and capytaine_A_full is not None
        and capytaine_B_full is not None
    )
    capy_dest_dof = _capytaine_structural_dest_indices(config, mode_indices_tracked)
    if use_dynamic_capy:
        n_flutter_capy = len(capy_dest_dof)
        print(
            "[RFA+Capytaine] Per-velocity structural embedding enabled "
            f"({n_flutter_capy} mode(s) on modal DOF {capy_dest_dof})"
        )
    elif M_hydro_add is not None and capytaine_A_full is None:
        print(
            "[RFA+Capytaine] Legacy constant embedding "
            f"(capytaine_structural_pool_embedding=False)"
        )
    last_capy_apply_idx = None

    for i_V, V in enumerate(V_list):
        if static_runner is not None:
            static_runner.step(V, i_V)

        print(f"\n--- VELOCITY STEP {i_V+1}/{len(V_list)}: V = {V:.2f} m/s ---")
        reacquired_slots = set()

        # STEP A: k-grid and lag poles for Roger fit
        if getattr(config, 'rfa_adaptive_k_list', False):
            omega_phys = np.linspace(0.1 * omega_n[0], 3.0 * omega_n[-1], 100)
            k_list_V   = np.clip(omega_phys / V, k_data_min, k_data_max)
            k_list_V   = np.unique(k_list_V)
        else:
            use_qjj_nodes = bool(getattr(config, "rfa_use_precomputed_k_nodes", True))
            if use_qjj_nodes:
                k_list_V = qjj_k_grid
            else:
                k_list_V = np.asarray(config.k_list, dtype=float)
                k_list_V = k_list_V[(k_list_V >= k_data_min) & (k_list_V <= k_data_max)]
                k_list_V = np.unique(k_list_V)
            if k_list_V.size < 8:
                print(
                    f"  [WARN] Only {k_list_V.size} k points inside Qjj range "
                    f"[{k_data_min}, {k_data_max}]; widen config.k_list or recompute Qjj."
                )
        Nk = len(k_list_V)

        blag, blag_ref_length, beta_fit = _roger_lag_poles(config, k_list_V, N)

        print(f"  STEP A: k_list_V has {Nk} points, range [{k_list_V[0]:.6f}, {k_list_V[-1]:.6f}] (k_pa=ω/V)")
        if roger_conv == "paper_zurich":
            print(
                f"  STEP A: k* range [{k_list_V[0]*semichord_b:.6f}, {k_list_V[-1]*semichord_b:.6f}]"
            )
        print(f"  STEP A: lag poles blag = {np.array2string(blag, precision=4)} [1/m]")
        if blag_ref_length is not None:
            beta_show = (
                beta_fit
                if beta_fit is not None
                else np.asarray(_resolve_roger_blag_config(config, N), dtype=float).ravel()[:N]
            )
            print(
                "  STEP A: dimensionless lag poles beta = "
                f"{np.array2string(beta_show, precision=4)} converted with L_ref={blag_ref_length:.6g} m"
            )

        # STEP B: Q_modal data at each k
        use_hybrid_q = bool(getattr(config, "aero_hybrid_dlm_re_measured_im", False))
        use_measured_q = bool(getattr(config, "aero_use_abramson_measured_q", False))
        mi = tuple(int(i) for i in mode_indices_tracked[:2])

        if use_uvlm_q:
            # Precomputed SHARPy UVLM modal GAF (same table as PK aero_method=UVLM).
            # No DLM conj(): Quvlm is already PanelAero-convention; apply PK sign flips only.
            def _interp_quvlm(k_red):
                kc = min(max(float(k_red), float(quvlm_k_list[0])), float(quvlm_k_list[-1]))
                i = int(np.searchsorted(quvlm_k_list, kc))
                if i <= 0:
                    return quvlm_Q_tbl[0].copy()
                if i >= len(quvlm_k_list):
                    return quvlm_Q_tbl[-1].copy()
                f = (kc - quvlm_k_list[i - 1]) / max(quvlm_k_list[i] - quvlm_k_list[i - 1], 1e-30)
                return (1.0 - f) * quvlm_Q_tbl[i - 1] + f * quvlm_Q_tbl[i]

            Q_modal_data = [
                aero_q_scaling.apply_aero_q_sign_flips(_interp_quvlm(k), config)
                for k in k_list_V
            ]
            Q_modal_array = np.array(Q_modal_data)
            if getattr(config, "aero_b2_removal", False):
                _karr = np.asarray(k_list_V, dtype=float)
                _idx = int(np.argmin(np.abs(_karr - min(34.0, float(_karr.max())))))
                _kb2 = float(_karr[_idx])
                _B2 = -np.real(Q_modal_array[_idx]) / (_kb2 ** 2)
                _B2 = 0.5 * (_B2 + _B2.T)
                Q_modal_array = Q_modal_array + (_karr[:, None, None] ** 2) * _B2[None, :, :]
                Q_modal_data = [Q_modal_array[i] for i in range(Q_modal_array.shape[0])]
                print(f"  [aero_b2_removal/RFA-UVLM] UVLM apparent mass removed before Roger fit "
                      f"(k_b2={_kb2:.3f}); B2 diag={np.diag(_B2)}")
        elif use_hybrid_q or not use_measured_q:
            Q_modal_data = []
            Ma = V / c_sound
            ma_interp_mode = getattr(config, "qjj_ma_interp_mode", "auto")
            ma_low_threshold = float(getattr(config, "qjj_ma_low_threshold", 0.05))
            for k in k_list_V:
                Qjj_k   = interp_qjj_from_disk_old(
                    config.qjj_dir,
                    k,
                    Ma,
                    ma_interp_mode=ma_interp_mode,
                    ma_low_threshold=ma_low_threshold,
                )   # (Np, Np)
                ik      = 1j * k
                w_k     = (ik * Z + Z_qs) @ Phi                              # (Np, Nm)
                aero_k  = Qjj_k @ w_k                                        # (Np, Nm)
                Qm_k    = Z_force_modal.T @ A_diag @ aero_k                  # (Nm_modal, Nm_modal)
                Q_modal_data.append(Qm_k)
            Q_modal_array = np.array(Q_modal_data)                           # (Nk, Nm_modal, Nm_modal)

            # Align RFA with the PK solver: PK applies Im(Q) as C_eff = C + C_aero,
            # which is the energy-consistent sign for this downwash chain (w = +ik·Z + Z_qs),
            # i.e. PK realizes conj(Q(ik)). Fit the conjugate so B1 and the lag
            # Im-channels carry the same sign as PK while keeping stable lag poles.
            Q_modal_array = np.conj(Q_modal_array)
            Q_modal_data = [np.conj(Qm) for Qm in Q_modal_data]

            # [aero_b2_removal] Hydroelastic correction in the RFA path: cancel the DLM's own
            # apparent mass (-k^2 * B2) out of the modal GAF BEFORE the Roger fit. Fitting the raw
            # GAF (which keeps the steep k^2 apparent-mass term) distorts the circulatory B0/B1/lag
            # (over-stiffens bending in water -> early coalescence); removing it first lets the lag
            # poles fit the purely circulatory aero, so the explicit Capytaine added mass in M_hat is
            # the sole apparent mass. B2 = -Re(Q(k_b2))/k_b2^2 at a high-k node (clipped to ~34, as PK).
            if getattr(config, "aero_b2_removal", False):
                _karr = np.asarray(k_list_V, dtype=float)
                _idx = int(np.argmin(np.abs(_karr - min(34.0, float(_karr.max())))))
                _kb2 = float(_karr[_idx])
                _B2 = -np.real(Q_modal_array[_idx]) / (_kb2 ** 2)
                _B2 = 0.5 * (_B2 + _B2.T)
                Q_modal_array = Q_modal_array + (_karr[:, None, None] ** 2) * _B2[None, :, :]
                Q_modal_data = [Q_modal_array[i] for i in range(Q_modal_array.shape[0])]
                print(f"  [aero_b2_removal/RFA] DLM apparent mass removed before Roger fit "
                      f"(k_b2={_kb2:.3f}); B2 diag={np.diag(_B2)}")

        if not use_uvlm_q and use_hybrid_q:
            from COUPLING.hydroelastic_utl.abramson_measured_q_modal import (
                build_hybrid_q_modal_array,
            )

            Q_modal_array = build_hybrid_q_modal_array(
                Q_modal_array,
                k_list_V,
                config,
                structural_results,
                coupling_results,
                aerogrid,
                n_modes_modal=n_modes_modal,
                mode_indices=mi,
                V_for_k=float(V),
            )
            Q_modal_data = [Q_modal_array[i] for i in range(Q_modal_array.shape[0])]
        elif not use_uvlm_q and use_measured_q:
            from COUPLING.hydroelastic_utl.abramson_measured_q_modal import (
                build_measured_q_modal_array,
            )

            Q_modal_array = build_measured_q_modal_array(
                k_list_V,
                config,
                structural_results,
                coupling_results,
                aerogrid,
                n_modes_modal=n_modes_modal,
                mode_indices=mi,
                V_for_k=float(V),
                apply_conj=True,
            )
            Q_modal_data = [Q_modal_array[i] for i in range(Q_modal_array.shape[0])]

        print(f"  STEP B: Q_modal computed at {Nk} k values, shape {Q_modal_array.shape}")

        # Optional: drop very small k (quasi-steady) that dominate LSQ but are not flutter
        omega_weight = _omega_ref_for_roger_fit(
            i_V, freq_sweep, fit_indices, omega_n, n_modes_tracked
        )
        k_fit_mask = _k_mask_for_roger_fit(k_list_V, V, omega_weight, config)
        if np.count_nonzero(k_fit_mask) < 8:
            print(
                f"  [WARN] Roger k-band leaves only {np.count_nonzero(k_fit_mask)} points; "
                "widening to full k_list_V."
            )
            k_fit_mask = np.ones(k_list_V.shape, dtype=bool)
        k_fit = k_list_V[k_fit_mask]
        Q_fit_data = Q_modal_array[k_fit_mask]
        if i_V > 0 and freq_sweep:
            print(
                f"  STEP C.0: Roger fit weights from tracked ω_prev = "
                f"{np.array2string(omega_weight, precision=3)} rad/s; "
                f"k_fit ∈ [{k_fit[0]:.4f}, {k_fit[-1]:.4f}] ({k_fit.size} pts)"
            )

        row_weights = None
        if getattr(config, 'rfa_weight_fit_k', False) and k_fit.size > 0:
            row_weights = _k_weights_for_roger_fit(
                k_fit, V, omega_weight,
                sigma_frac=float(getattr(config, 'rfa_weight_fit_sigma', 0.6)),
                roger_convention=roger_conv,
                semichord=semichord_b,
            )

        fit_kw = dict(
            roger_convention=roger_conv,
            semichord=semichord_b,
            beta_lag=beta_fit,
        )

        # STEP C: Roger fit (tracked-mode subspace when n_fit < n_modes_modal)
        idx_fit = np.asarray(fit_indices, dtype=int)
        if n_fit < n_modes_modal:
            Q_fit_sub = Q_fit_data[:, idx_fit][:, :, idx_fit]
            B0_sub, B1_sub, B2_sub, Blag_sub = _fit_roger_to_Q_modal(
                Q_fit_sub, k_fit, blag, n_fit, N,
                row_weights=row_weights, include_b2=fit_b2, **fit_kw,
            )
            B0, B1, B2, Blag = _embed_roger_coefficients(
                B0_sub, B1_sub, B2_sub, Blag_sub, fit_indices, n_modes_modal
            )
        else:
            B0, B1, B2, Blag = _fit_roger_to_Q_modal(
                Q_fit_data, k_fit, blag, n_modes_modal, N,
                row_weights=row_weights, include_b2=fit_b2, **fit_kw,
            )

        b2_note = (
            "B2 fitted+applied" if (fit_b2 and apply_b2)
            else "B2 fitted, discarded in EOM (AM in M_hat)" if fit_b2
            else "no B2 (AM in M_hat)"
        )
        print(f"  STEP C: Roger fit complete ({b2_note}) — {len(Blag)} lag terms")
        _print_roger_lag_diagnostics(B0, B1, Blag)

        # STEP C.4: Fit quality check
        max_err = 0.0
        recon_kw = dict(
            roger_convention=roger_conv,
            semichord=semichord_b,
            beta_lag=beta_fit,
        )
        for i_k, k in enumerate(k_list_V):
            Q_fit = _reconstruct_Q_modal(
                k, B0, B1, B2, Blag, blag, include_b2=fit_b2, **recon_kw,
            )
            Q_ref = Q_modal_data[i_k]
            err = np.linalg.norm(Q_fit - Q_ref, 'fro') / np.linalg.norm(Q_ref, 'fro')
            max_err = max(max_err, err)
            if i_k % max(1, Nk // 3) == 0:
                print(f"    k={k:.4f}: fit error = {err:.2e}")
        print(f"  STEP C: Max fit error = {max_err:.2e} (target < 1%)")
        if getattr(config, "rfa_report_tracked_fit_error", True):
            trk_err = _roger_tracked_fit_error(
                Q_fit_data, k_fit, B0, B1, B2, Blag, blag, fit_indices,
                include_b2=fit_b2, recon_kw=recon_kw,
            )
            print(
                f"  STEP C: Tracked-block fit error = {trk_err:.2e} "
                f"(modes {fit_indices})"
            )

        ## STEP C.5: Capytaine embedding for this velocity step
        M_hydro_step = M_hydro_add
        C_hydro_step = C_hydro_add
        if use_dynamic_capy:
            pool_for_capy = np.asarray(capy_dest_dof, dtype=int)
            capy_reason = f"structural modal DOF {capy_dest_dof}"
            M_hydro_step, C_hydro_step, last_capy_apply_idx = _capytaine_matrices_for_velocity_step(
                config,
                capytaine_A_full,
                capytaine_B_full,
                n_modes_modal,
                i_V,
                pool_for_capy,
                reason=capy_reason,
            )

        # STEP D: Augmented state matrix
        A_aug, M_eff, C_eff, K_eff = _build_A_aug(
            config, V, rho, M_hat, C_hat, K_hat, B0, B1, B2, Blag, blag, n_modes_modal, N,
            M_hydro_add=M_hydro_step, C_hydro_add=C_hydro_step, apply_b2_mass=use_b2,
        )


        if dlm_participants_history is not None:
            aero_im_scale = aero_q_scaling.aero_im_Q_scale(config)
            K_hat_arr = np.array(K_hat, copy=True)
            C_hat_arr = np.array(C_hat, copy=True)
            M_hat_arr = np.array(M_hat, copy=True)
            zeros = np.zeros_like(M_hat_arr)
            if use_b2:
                if roger_conv == "paper_zurich":
                    M_aero_dlm = 0.5 * rho * (semichord_b ** 2) * np.asarray(B2, dtype=float)
                else:
                    M_aero_dlm = 0.5 * rho * np.asarray(B2, dtype=float)
            else:
                M_aero_dlm = zeros.copy()
            M_hydro_arr = (
                np.asarray(M_hydro_step, dtype=float)
                if M_hydro_step is not None else zeros.copy()
            )
            K_aero_dlm = 0.5 * rho * V**2 * np.asarray(B0, dtype=float)
            K_hydro_arr = zeros.copy()
            C_aero_dlm = aero_im_scale * 0.5 * rho * V * np.asarray(B1, dtype=float)
            C_hydro_arr = (
                np.asarray(C_hydro_step, dtype=float)
                if C_hydro_step is not None else zeros.copy()
            )
            K_eff_arr = np.array(K_eff, copy=True)
            C_eff_arr = np.array(C_eff, copy=True)
            M_eff_arr = np.array(M_eff, copy=True)
            C_emp_arr = empirical_fluid_damping.modal_empirical_C(config, V, M_eff_arr)
            # Same layout as PK contributions_history (one matrix per mode slot).
            _slot = lambda M: [M, M]
            dlm_participants_history.append({
                'V': float(V),
                'branch_dof_indices': list(capy_dest_dof),
                'C_hat': C_hat_arr,
                'K_hat': K_hat_arr,
                'C_eff': C_eff_arr,
                'K_eff': K_eff_arr,
                'M_eff': M_eff_arr,
                'M_struct': _slot(M_hat_arr),
                'M_aero': _slot(M_aero_dlm),
                'M_hydro': _slot(M_hydro_arr),
                'M_effective': _slot(M_eff_arr),
                'K_struct': _slot(K_hat_arr),
                'K_aero': _slot(K_aero_dlm),
                'K_hydro': _slot(K_hydro_arr),
                'K_effective': _slot(K_eff_arr),
                'C_struct': _slot(C_hat_arr),
                'C_aero': _slot(C_aero_dlm),
                'C_hydro': _slot(C_hydro_arr),
                'C_empirical': _slot(C_emp_arr),
                'C_effective': _slot(C_eff_arr),
            })

        print(f"  STEP D: A_aug assembled, shape {A_aug.shape}")

        # STEP E: extract a pool of oscillatory roots (candidate branches)
        g_pool, freq_pool, vecs_pool, eigs_pool = _extract_flutter_modes(
            A_aug,
            n_search_modes,
            omega_n,
            return_eigenvectors=True,
            dimensionless_vgvf_results=config.dimensionless_vgvf_results,
            n_search_modes=n_search_modes,
            omega_phys_min=omega_phys_min,
        )
        
        print(f"  STEP E: Flutter modes extracted")
        print(f"    Candidate frequencies (rad/s): {freq_pool}")
        print(f"    Candidate damping ratios:      {g_pool}")

        _omega_wet_prev = (
            np.asarray(freq_sweep[-1], dtype=float).ravel()
            if i_V > 0 and freq_sweep else None
        )
        _omega_predicted = None
        if i_V > 0 and _omega_wet_prev is not None and raw_results:
            struct_frac = float(getattr(config, "rfa_branch_guard_max_struct_frac", 1.35))
            _omega_predicted = np.asarray(_omega_wet_prev, dtype=float).ravel().copy()
            for j_br in range(len(_omega_predicted)):
                lost_prev = roger_mode_tracking.branch_slot_lost_at_previous(raw_results, j_br)
                if not (
                    lost_prev
                    or not (
                        np.isfinite(_omega_predicted[j_br])
                        and _omega_predicted[j_br] > 0.0
                    )
                ):
                    continue
                omega_pred = roger_mode_tracking.predict_branch_omega_at_V(
                    raw_results,
                    j_br,
                    float(V),
                    omega_n=omega_n,
                    mode_indices_tracked=mode_indices_tracked,
                    max_struct_frac=struct_frac,
                    partner_slot=roger_mode_tracking._partner_slot_for(
                        j_br, n_modes_tracked
                    ),
                )
                if omega_pred is not None and np.isfinite(omega_pred) and omega_pred > 0.0:
                    _omega_predicted[j_br] = float(omega_pred)
        freq_pool, g_pool, eigs_pool, vecs_pool = _filter_rfa_pool_roots(
            freq_pool, g_pool, eigs_pool, vecs_pool, omega_n, mode_indices_tracked, config,
            omega_wet_prev=_omega_wet_prev,
            omega_predicted=_omega_predicted,
            n_slots_min=n_modes_tracked,
            n_modal=n_modes_modal,
        )
        if len(freq_pool):
            print(f"    Pool after filter (rad/s): {freq_pool}")

        use_rfa_pas = bool(getattr(config, "rfa_predict_and_select", True))

        # STEP F: branch tracking across velocity steps (tracked branches only).
        if (
            use_rfa_pas
            and i_V >= 1
            and len(raw_results) >= 1
        ):
            pas_indices = _assign_rfa_branches_predict_and_select(
                freq_pool,
                eigs_pool,
                raw_results,
                i_V,
                n_modes_tracked,
                V_current=float(V),
                delta=max(0.025, float(freq_margin_rfa)),
                min_omega_sep_rad=float(
                    getattr(config, "rfa_min_root_sep_rad", getattr(config, "pk_min_root_sep_rad", 1.0))
                ),
                mode_indices_tracked=mode_indices_tracked,
                omega_n=omega_n,
                allow_coalescence=bool(getattr(config, "rfa_pas_allow_coalescence", True)),
            )
            if pas_indices is not None:
                n_struct_dof = getattr(config, "rfa_mac_n_structural_dof", None)
                pas_indices = _rfa_pas_mac_tiebreak(
                    pas_indices,
                    freq_pool,
                    vecs_pool,
                    vecs_sweep[-1],
                    degeneracy_rad=float(
                        getattr(config, "rfa_pas_mac_degeneracy_rad", 3.0)
                    ),
                    mac_margin=float(getattr(config, "rfa_pas_mac_margin", 0.02)),
                    n_structural_dof=n_struct_dof,
                )
                g_V, freq_V, vecs_V, eigs_V = roger_mode_tracking.reorder_and_normalize_modes(
                    eigs_pool, vecs_pool, freq_pool, g_pool, pas_indices
                )
                mac_history.append(None)
            else:
                pas_indices = None
        else:
            pas_indices = None

        if pas_indices is None and i_V > 0 and len(vecs_sweep) > 0:
            print(f"  STEP F: Applying MAC-based mode matching...")
            
            # Retrieve previous velocity data
            freq_V_prev = np.asarray(freq_sweep[-1], dtype=float).ravel().copy()
            vecs_V_prev = vecs_sweep[-1]
            eigs_V_prev = eigs_sweep[-1]

            # After mode_lost, restore reference ω / eigenvector for MAC matching.
            struct_frac = float(getattr(config, "rfa_branch_guard_max_struct_frac", 1.35))
            for j_br in range(len(freq_V_prev)):
                lost_prev = roger_mode_tracking.branch_slot_lost_at_previous(raw_results, j_br)
                if not (
                    lost_prev
                    or not (np.isfinite(freq_V_prev[j_br]) and freq_V_prev[j_br] > 0.0)
                ):
                    continue
                omega_pred = roger_mode_tracking.predict_branch_omega_at_V(
                    raw_results,
                    j_br,
                    float(V),
                    omega_n=omega_n,
                    mode_indices_tracked=mode_indices_tracked,
                    max_struct_frac=struct_frac,
                    partner_slot=(j_br - 1 if j_br > 0 else 1),
                )
                m_struct = int(mode_indices_tracked[j_br])
                omega_struct = float(omega_n[m_struct])
                if omega_pred is not None and np.isfinite(omega_pred) and omega_pred > 0.0:
                    freq_V_prev[j_br] = float(omega_pred)
                else:
                    md_ref, _ = roger_mode_tracking.last_valid_branch_mode(raw_results, j_br)
                    if md_ref is not None:
                        omega_ref = float(md_ref["omega"])
                        if omega_ref > omega_struct * struct_frac:
                            omega_ref = omega_struct
                        freq_V_prev[j_br] = omega_ref
                    else:
                        freq_V_prev[j_br] = omega_struct
                u_ref = roger_mode_tracking.reference_eigenvector_for_mac(
                    vecs_sweep,
                    j_br,
                    raw_results=raw_results,
                    **_rfa_mac_reference_kw(config),
                )
                if (
                    u_ref is not None
                    and vecs_V_prev.ndim == 2
                    and j_br < vecs_V_prev.shape[1]
                ):
                    vecs_V_prev[:, j_br] = u_ref
            
            # Get the DOF size from the augmented matrix
            n_dof_total = A_aug.shape[0]  # Total DOFs including lag states
            
            # Defensive check: ensure eigenvectors are 2D with correct shape
            if vecs_pool.ndim != 2 or vecs_pool.shape[0] != n_dof_total:
                print(f"  ⚠ Warning: vecs_pool shape is {vecs_pool.shape}, expected ({n_dof_total}, {n_search_modes}). Reshaping...")
                try:
                    vecs_pool = vecs_pool.reshape(n_dof_total, n_search_modes)
                except ValueError as e:
                    print(f"  ✗ ERROR: Cannot reshape vecs_pool to ({n_dof_total}, {n_search_modes}). Got error: {e}")
                    print(f"    Current shape: {vecs_pool.shape}, total elements: {vecs_pool.size}")
                    raise
            
            if vecs_V_prev.ndim != 2 or vecs_V_prev.shape[0] != n_dof_total:
                print(f"  ⚠ Warning: vecs_V_prev shape is {vecs_V_prev.shape}, expected ({n_dof_total}, {n_modes_tracked}). Reshaping...")
                try:
                    vecs_V_prev = vecs_V_prev.reshape(n_dof_total, n_modes_tracked)
                except ValueError as e:
                    print(f"  ✗ ERROR: Cannot reshape vecs_V_prev to ({n_dof_total}, {n_modes_tracked}). Got error: {e}")
                    print(f"    Current shape: {vecs_V_prev.shape}, total elements: {vecs_V_prev.size}")
                    raise
            
            # Match previous tracked branches to current candidate pool
            reordering, mac_mat, match_info = roger_mode_tracking.match_modes_between_velocities(
                eigs_pool, vecs_pool, freq_pool,
                eigs_V_prev, vecs_V_prev, freq_V_prev,
                freq_margin=freq_margin_rfa,
                w_freq=w_freq_rfa,
                w_mac=w_mac_rfa,
                n_structural_dof=2 * n_modes_modal if getattr(config, 'rfa_mac_structural_dof', True) else None,
                verbose=True
            )
            
            # Handle variable number of modes (reordering may contain -1 for unmatched)
            n_modes_matched = np.sum(reordering >= 0)
            n_modes_current = len(freq_pool)
            n_modes_previous = len(freq_V_prev)
            n_modes_unmatched = n_modes_previous - n_modes_matched
            
            if n_modes_unmatched > 0:
                print(
                    f"  [INFO] Branch replacement: {n_modes_matched}/{n_modes_previous} matched, "
                    f"{n_modes_unmatched} mode_lost (no pool substitution)"
                )
            
            # Build a stable branch ordering aligned with the previous step indices.
            # reordering[i_prev] = i_curr for matched modes, or -1 for unmatched (lost) ones.
            # This prevents branch index "scrambling" (which breaks continuity in V-g/V-f plots).
            branch_unmatched = set()
            if n_modes_matched > 0:
                tracking_indices = np.full(n_modes_previous, -1, dtype=int)
                used_for_assign = set()
                for i_prev in range(n_modes_previous):
                    i_curr = int(reordering[i_prev])
                    if i_curr >= 0:
                        tracking_indices[i_prev] = i_curr
                        used_for_assign.add(i_curr)
                    else:
                        branch_unmatched.add(int(i_prev))
                rejected_mac = _reject_invalid_rfa_mac_assignments(
                    tracking_indices,
                    freq_pool,
                    freq_V_prev,
                    raw_results,
                    mode_indices_tracked,
                    omega_n,
                    config,
                )
                branch_unmatched |= rejected_mac
                for i_prev in range(n_modes_previous):
                    if roger_mode_tracking.branch_slot_lost_at_previous(raw_results, i_prev):
                        branch_unmatched.add(int(i_prev))

                # MAC-unmatched / previously lost: try predict-and-reacquire in ω neighbourhood.
                if branch_unmatched and getattr(config, "rfa_branch_reacquire_enable", True):
                    for i_prev in sorted(branch_unmatched):
                        tracking_indices[i_prev] = -1
                    reacq_kw = _rfa_reacquire_kwargs(config)
                    for i_prev in sorted(branch_unmatched):
                        used_pool = {
                            int(tracking_indices[i])
                            for i in range(n_modes_previous)
                            if int(tracking_indices[i]) >= 0 and i != i_prev
                        }
                        partner_slot = roger_mode_tracking._partner_slot_for(
                            i_prev, n_modes_tracked
                        )
                        partner_omega = _partner_omega_for_reacquire(
                            tracking_indices, partner_slot, freq_pool, freq_V_prev
                        )
                        pick = roger_mode_tracking.reacquire_lost_branch_in_pool(
                            i_prev,
                            freq_pool,
                            vecs_pool,
                            raw_results,
                            vecs_sweep,
                            float(V),
                            used_pool,
                            n_structural_dof=n_mac_dof,
                            omega_n=omega_n,
                            mode_indices_tracked=mode_indices_tracked,
                            n_slots=n_modes_tracked,
                            partner_omega=partner_omega,
                            **reacq_kw,
                        )
                        if pick >= 0:
                            tracking_indices[i_prev] = int(pick)
                            reacquired_slots.add(int(i_prev))
                    branch_unmatched -= reacquired_slots

                if branch_unmatched:
                    used_placeholders = {
                        int(tracking_indices[i])
                        for i in range(n_modes_previous)
                        if int(tracking_indices[i]) >= 0
                    }
                    for i_prev in sorted(branch_unmatched):
                        pick_ph = None
                        for cand in range(len(freq_pool)):
                            if cand not in used_placeholders:
                                pick_ph = cand
                                break
                        if pick_ph is None:
                            pick_ph = int(
                                _init_tracking_indices_structural(
                                    freq_pool, mode_indices_tracked, omega_n
                                )[i_prev]
                            )
                        tracking_indices[i_prev] = int(pick_ph)
                        used_placeholders.add(int(pick_ph))
            else:
                # No MAC matches — try reacquire from predicted ω(V), else re-seed from ω_n.
                print(
                    "  [WARNING] No modes matched with previous step. "
                    "Trying predict-and-reacquire before structural ω_n re-seed..."
                )
                tracking_indices = np.full(n_modes_previous, -1, dtype=int)
                if getattr(config, "rfa_branch_reacquire_enable", True):
                    reacq_kw = _rfa_reacquire_kwargs(config)
                    for i_prev in range(n_modes_previous):
                        used_pool = {
                            int(tracking_indices[i])
                            for i in range(n_modes_previous)
                            if int(tracking_indices[i]) >= 0 and i != i_prev
                        }
                        partner_slot = roger_mode_tracking._partner_slot_for(
                            i_prev, n_modes_tracked
                        )
                        partner_omega = _partner_omega_for_reacquire(
                            tracking_indices, partner_slot, freq_pool, freq_V_prev
                        )
                        pick = roger_mode_tracking.reacquire_lost_branch_in_pool(
                            i_prev,
                            freq_pool,
                            vecs_pool,
                            raw_results,
                            vecs_sweep,
                            float(V),
                            used_pool,
                            n_structural_dof=n_mac_dof,
                            omega_n=omega_n,
                            mode_indices_tracked=mode_indices_tracked,
                            n_slots=n_modes_tracked,
                            partner_omega=partner_omega,
                            **reacq_kw,
                        )
                        if pick >= 0:
                            tracking_indices[i_prev] = int(pick)
                            reacquired_slots.add(int(i_prev))
                if np.any(tracking_indices < 0):
                    fallback = _init_tracking_indices_structural(
                        freq_pool, mode_indices_tracked, omega_n
                    )
                    for i_prev in range(n_modes_previous):
                        if int(tracking_indices[i_prev]) < 0:
                            tracking_indices[i_prev] = int(fallback[i_prev])

            tracking_indices = _finalize_rfa_tracking_indices(
                tracking_indices, freq_pool, mode_indices_tracked, omega_n
            )
            g_V_new, freq_V_new, vecs_V_new, eigs_V_new = roger_mode_tracking.reorder_and_normalize_modes(
                eigs_pool, vecs_pool, freq_pool, g_pool, tracking_indices
            )
            
            # Validate returned shapes
            if vecs_V_new.ndim != 2:
                print(f"  ✗ ERROR after reordering: vecs_V_new has shape {vecs_V_new.shape}, expected 2D")
                raise ValueError(f"vecs_V_new should be 2D but got shape {vecs_V_new.shape}")
            
            g_V = g_V_new
            freq_V = freq_V_new
            vecs_V = vecs_V_new
            eigs_V = eigs_V_new

            for j_unmatched in branch_unmatched:
                if 0 <= j_unmatched < len(freq_V):
                    g_V[j_unmatched] = np.nan
                    freq_V[j_unmatched] = np.nan
                    eigs_V[j_unmatched] = np.nan
                    print(
                        f"  ✗ RFA branch {j_unmatched} LOST (MAC unmatched at V={V:.2f} m/s)"
                    )
            
            mac_history.append(mac_mat)
            
            # Diagnostic summary for this velocity step
            print(f"    MAC matching summary:")
            for info in match_info:
                validity = "✓" if info['valid'] else "✗"
                print(f"      {validity} Mode {info['mode_index']}: MAC={info['mac']:.3f}, "
                      f"freq_change={info['freq_error']*100:.1f}%")
        elif pas_indices is None and i_V == 0:
            # First velocity: lock branches to dry structural frequencies (PK-style).
            init_indices = _init_tracking_indices_structural(
                freq_pool, mode_indices_tracked, omega_n
            )
            print(
                f"  [INFO] Initial branch assignment (structural ω_n): "
                f"{list(zip(mode_indices_tracked, init_indices.tolist()))}"
            )
            g_V, freq_V, vecs_V, eigs_V = roger_mode_tracking.reorder_and_normalize_modes(
                eigs_pool, vecs_pool, freq_pool, g_pool, init_indices
            )
            if vecs_V.ndim == 2 and vecs_V.shape[1] > 0:
                for j in range(vecs_V.shape[1]):
                    vec = vecs_V[:, j]
                    norm_val = np.linalg.norm(vec)
                    if norm_val > 1e-12:
                        vecs_V[:, j] = vec / norm_val
            else:
                print(f"  ⚠ Warning: vecs_V shape is {vecs_V.shape}, skipping normalization")

        # Ensure vecs_V has the correct shape before storage
        if vecs_V.ndim != 2 or vecs_V.shape[1] != n_modes_tracked:
            print(
                f"  ⚠ Warning: vecs_V shape {vecs_V.shape} invalid; "
                f"recovering from structural ω_n on current pool ({len(freq_pool)} roots)"
            )
            if len(freq_pool) < n_modes_tracked:
                raise ValueError(
                    f"Cannot recover {n_modes_tracked} tracked branches from "
                    f"{len(freq_pool)} pool root(s) at V={V:.2f} m/s"
                )
            recover_idx = _init_tracking_indices_structural(
                freq_pool, mode_indices_tracked, omega_n
            )
            g_V, freq_V, vecs_V, eigs_V = roger_mode_tracking.reorder_and_normalize_modes(
                eigs_pool, vecs_pool, freq_pool, g_pool, recover_idx
            )
        if vecs_V.ndim != 2 or vecs_V.shape[1] != n_modes_tracked:
            raise ValueError(
                f"vecs_V shape {vecs_V.shape} after recovery; "
                f"expected ({A_aug.shape[0]}, {n_modes_tracked})"
            )

        mode_lost_flags = [False] * n_modes_tracked
        if getattr(config, "rfa_branch_guard_enable", True) and i_V > 0:
            g_V, freq_V, vecs_V, eigs_V, mode_lost_flags = roger_mode_tracking.apply_rfa_branch_guard(
                g_V,
                freq_V,
                vecs_V,
                eigs_V,
                raw_results,
                vecs_sweep,
                V_current=float(V),
                max_rel_jump=float(getattr(config, "rfa_branch_guard_max_rel_jump", 0.30)),
                min_mac_prev=float(getattr(config, "rfa_branch_guard_min_mac_prev", 0.70)),
                max_struct_frac=getattr(config, "rfa_branch_guard_max_struct_frac", 1.35),
                min_branch_sep_frac=float(getattr(config, "rfa_branch_guard_min_sep_frac", 0.08)),
                check_neighbor_collapse=bool(getattr(config, "rfa_branch_guard_check_neighbor_collapse", True)),
                mode_indices_tracked=mode_indices_tracked,
                omega_n=omega_n,
                n_structural_dof=n_mac_dof,
                enable=True,
                skip_slots=reacquired_slots,
                **_rfa_mac_reference_kw(config),
            )

        V_sweep.append(V)
        g_sweep.append(g_V)
        freq_sweep.append(freq_V)
        vecs_sweep.append(vecs_V)
        eigs_sweep.append(eigs_V)

        # Build a PK-like velocity snapshot so post-processing can classify
        # "tracked" vs "untracked" wet roots also for Roger/RFA sweeps.
        tracked_modes_for_V = []
        for j_mode, mode_idx in enumerate(mode_indices_tracked):
            eig_j = np.nan
            if j_mode < len(eigs_V):
                eig_j = eigs_V[j_mode]
            om_j = np.nan
            if j_mode < len(freq_V):
                om_j = float(freq_V[j_mode])
            tracked_modes_for_V.append(
                {
                    "mode": int(mode_idx),
                    "p": complex(eig_j) if np.isfinite(np.abs(eig_j)) else np.nan,
                    "omega": om_j,
                    "k": float(om_j / V) if V > 1e-12 and np.isfinite(om_j) else np.nan,
                    "converged": bool(np.isfinite(om_j) and om_j > 0.0),
                    "mode_lost": bool(
                        j_mode < len(mode_lost_flags) and mode_lost_flags[j_mode]
                    ),
                    "reacquired": bool(j_mode in reacquired_slots),
                }
            )
        freq_margin_rfa_val = float(getattr(config, "rfa_freq_margin", 0.20))
        tracked_omegas_V = [float(freq_V[j]) for j in range(len(freq_V)) if np.isfinite(freq_V[j])]
        wet_spectrum = _build_wet_spectrum_with_vectors(
            A_aug,
            omega_n,
            tracked_omegas_V,
            freq_margin_rfa_val,
            omega_phys_min=omega_phys_min,
            dimensionless_vgvf_results=config.dimensionless_vgvf_results,
        )
        orphan_track = {}
        if orphan_tracker is not None:
            tracked_spec_idx = set()
            for om_t in tracked_omegas_V:
                for r in wet_spectrum.get("roots", []):
                    if abs(float(r["omega"]) - om_t) <= freq_margin_rfa_val * max(om_t, 1e-6):
                        tracked_spec_idx.add(int(r["index"]))
                        break
            n_mac_dof = (
                2 * n_modes_modal
                if getattr(config, "rfa_mac_structural_dof", True)
                else None
            )
            orphan_track = orphan_tracker.step(
                V,
                wet_spectrum,
                n_structural_dof=n_mac_dof,
                tracked_branch_indices=tracked_spec_idx,
            )
            orphan_g_sweep.append(float(orphan_track.get("g", np.nan)))
            orphan_f_sweep.append(float(orphan_track.get("omega", np.nan)))

        raw_results.append(
            {
                "V": float(V),
                "modes": tracked_modes_for_V,
                "wet_spectrum": wet_spectrum,
                "orphan_track": orphan_track,
            }
        )

        if cp_plotter is not None:
            cp_plotter.maybe_plot(V, i_V, tracked_modes_for_V)

    V_arr = np.array(V_sweep)
    g_arr = np.array(g_sweep)    # (Nv, Nm)
    f_arr = np.array(freq_sweep) # (Nv, Nm)

    # Compute flutter speed (optional freq-band selection for coupled modes)
    Vf, ff = vgvf_plotting.select_primary_flutter(
        V_arr,
        g_arr,
        f_arr,
        selection=getattr(config, "flutter_primary_selection", "min_V"),
        freq_band_hz=getattr(config, "flutter_freq_band_hz", None),
        freq_target_hz=getattr(config, "flutter_freq_target_hz", None),
        V_target_kn=getattr(config, "flutter_V_target_kn", None),
    )

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

    mac_convergence_records = post_processing.build_cross_velocity_mac_records(
        V_sweep,
        vecs_sweep,
        mode_indices_tracked,
        n_structural_dof=n_mac_dof,
    )
    mac_by_vm = {(rec["V"], rec["mode"]): rec for rec in mac_convergence_records}
    for entry in raw_results:
        V_entry = float(entry["V"])
        for md in entry.get("modes", []):
            rec = mac_by_vm.get((V_entry, int(md.get("mode", -1))))
            if rec is not None:
                md["mac_prev_velocity"] = rec["mac_prev_velocity"]
                md["mac_first_iter"] = rec["mac_first_iter"]

    rfa_pk_solver = SimpleNamespace(
        contributions_history=dlm_participants_history,
        mac_convergence_history=mac_convergence_records,
    )

    if orphan_tracker is not None and orphan_tracker.mac_activated:
        print(
            f"  [orphan MAC] Branch active from V={orphan_tracker.V_mac_start:.3g} m/s "
            f"(segment ref V={orphan_tracker.V_segment_start:.3g} m/s)"
        )

    return _finalize_static_hydroelastic_pass(
        _finalize_flutter_results(
            V_arr, g_arr, f_arr,
            config.modes_to_analyze,
            Vf, ff,
            config,
            raw_results=raw_results,
            pk_solver=rfa_pk_solver,
            dlm_participants_history=dlm_participants_history,
            orphan_damping=orphan_g_sweep if orphan_tracker is not None else None,
            orphan_frequencies=orphan_f_sweep if orphan_tracker is not None else None,
            orphan_mac_activated=(
                orphan_tracker.mac_activated if orphan_tracker is not None else False
            ),
        ),
        static_runner,
        config,
    )


def _solve_roger_fit_method(config, V_list, structural_results, coupling_results, aerogrid):
    """Roger RFA + strip (or dry) — same core as RFA_PK without Capytaine matrices."""
    aero_method = str(getattr(config, "aero_method", "DLM")).upper()
    if aero_method == "UVLM":
        label = "Roger RFA (UVLM Quvlm GAF → coupled SS, no p-k)"
    else:
        label = "Roger RFA (DLM circulatory + structural M_hat)"
    return _run_roger_rfa_sweep(
        config, V_list, structural_results, coupling_results, aerogrid,
        method_label=label,
    )


def _solve_RFA_PK_method(config, V_list, structural_results, coupling_results, aerogrid):
    """
    Hydroelastic Roger RFA: same DLM Roger fit / MAC sweep as ``roger_fit``, plus constant
    asymptotic Capytaine added mass and radiation damping (no PK, no omega interpolation).
    """
    n_modes_modal = int(len(np.asarray(structural_results.dry_eigenvalues).ravel()))
    am_source = _resolve_capytaine_added_mass_source(config)
    depth_arr = getattr(config, "depth", 0.0)
    depth_used = float(np.asarray(depth_arr).ravel()[int(getattr(config, "depth_index", 0))])

    if am_source == "stripstyle":
        print(
            f"[RFA+Capytaine strip-style] depth={depth_used:.4f} m; "
            "A_∞ = Φᵀ M_add Φ with m_a' from rigid Capytaine (strip lumping)"
        )
        print(
            "[RFA+Capytaine] Same Roger RFA as roger_fit (fixed k_list/blag; "
            "DLM B2 controlled by rfa_apply_b2_mass); Capytaine A,B = "
            "constants in M_eff, C_eff (no PK iteration)."
        )
        A_asy_full, B_asy_full, capy_npz = _load_capytaine_stripstyle_matrices(
            config, structural_results
        )
        if A_asy_full.shape != (n_modes_modal, n_modes_modal):
            raise ValueError(
                f"Strip-style M_hat_added shape {A_asy_full.shape} != "
                f"({n_modes_modal}, {n_modes_modal})"
            )
        omega_h = None
        A_h = None
    else:
        capy_data, capy_npz = _load_capytaine_modal_radiation(config)
        omega_h = np.asarray(capy_data["omega"], dtype=float).ravel()
        A_h = np.asarray(capy_data["added_mass"], dtype=float)
        B_h = np.asarray(capy_data["added_damping"], dtype=float)
        if A_h.shape[1] != n_modes_modal or A_h.shape[2] != n_modes_modal:
            raise ValueError(
                f"Capytaine modal matrix shape {A_h.shape} not compatible with "
                f"{n_modes_modal} dry/modal modes (num_modes_egv)."
            )
        omega_n = np.sqrt(np.abs(structural_results.dry_eigenvalues))
        print(f"[RFA+Capytaine modal BEM] depth={depth_used:.4f} m, file={capy_npz}")
        print(
            "[RFA+Capytaine] Same Roger RFA as roger_fit (fixed k_list/blag; "
            "DLM B2 controlled by rfa_apply_b2_mass); Capytaine A,B = "
            "asymptotic constants in M_eff, C_eff (no PK iteration)."
        )
        A_asy_full, B_asy_full = _capytaine_asymptotic_modal_matrices(
            omega_h, A_h, B_h, omega_n, config
        )
    n_capy = int(A_asy_full.shape[0])
    apply_idx = _resolve_capytaine_apply_indices(config, n_capy, n_modes_modal)
    source_idx = _resolve_capytaine_source_indices(config, n_capy, apply_idx)
    use_pool_embedding = bool(getattr(config, "capytaine_structural_pool_embedding", True))
    if use_pool_embedding:
        n_flutter = int(getattr(config, "num_modes_flutter_egv", 2))
        struct_src = _capytaine_structural_source_indices(config, n_capy)
        print(
            f"[RFA+Capytaine] Base A_∞, B_∞ from asymptotic Capytaine "
            f"(structural block {n_flutter}×{n_flutter}, tensor indices {struct_src}); "
            f"per-velocity pool embedding: "
            f"{getattr(config, 'capytaine_dynamic_physical_mask', True)}"
        )
    else:
        print(
            f"[RFA+Capytaine] Base A_∞, B_∞ from asymptotic Capytaine "
            f"(legacy constant embedding on dry DOF {apply_idx})"
        )
    print(f"  A_∞ full diagonal: {np.diag(A_asy_full)}")
    print(f"  B_∞ full diagonal: {np.diag(B_asy_full)}")

    M_hydro_legacy, C_hydro_legacy = _embed_capytaine_on_physical_modes(
        A_asy_full, B_asy_full, n_modes_modal, apply_idx, source_idx
    )
    if getattr(config, "empirical_fluid_damping", False):
        M_sample = np.array(structural_results.M_hat, copy=True) + M_hydro_legacy
        V_sample = float(np.median(V_list)) if len(V_list) else 20.0
        C_emp_sample = empirical_fluid_damping.modal_empirical_C(config, V_sample, M_sample)
        print(empirical_fluid_damping.describe_empirical_C(config, V_sample, M_sample, C_emp_sample))

    if config.added_mass_strip_theory:
        print(
            "[RFA+Capytaine WARN] added_mass_strip_theory=True: M_hat may already include "
            "strip AM; Capytaine A_asy is still added (check for double-counting)."
        )

    sweep_kw = dict(
        config=config,
        V_list=V_list,
        structural_results=structural_results,
        coupling_results=coupling_results,
        aerogrid=aerogrid,
        method_label="Roger RFA + Capytaine asymptotic (hydroelastic correction)",
    )
    if use_pool_embedding:
        sweep_kw.update(
            capytaine_A_full=A_asy_full,
            capytaine_B_full=B_asy_full,
            capytaine_static_apply_idx=apply_idx,
            capytaine_static_source_idx=source_idx,
        )
    else:
        sweep_kw.update(M_hydro_add=M_hydro_legacy, C_hydro_add=C_hydro_legacy)
    results = _run_roger_rfa_sweep(**sweep_kw)
    if getattr(config, "plot_added_mass_capytaine_vs_strip", False) and omega_h is not None:
        A_plot, _ = _embed_capytaine_on_physical_modes(
            A_asy_full, B_asy_full, n_modes_modal, apply_idx, source_idx
        )
        _plot_added_mass_terms_vs_velocity(
            structural_results,
            config,
            results.velocities,
            results.frequencies,
            omega_h,
            A_h,
            capytaine_added_mass_used=A_plot,
        )
    return results


def _apply_pk_modal_slice_and_qg(config, M_hat, C_hat, K_hat, omega_n_rad, Qg_func):
    """Optional 2×2 (or sub-block) slice + Pietro-style frozen Qg(s=0,j=0) evaluation."""
    idx = getattr(config, "pk_modal_dof_indices", None)
    if idx is None:
        return M_hat, C_hat, K_hat, omega_n_rad, Qg_func, list(getattr(config, "modes_to_analyze", [0, 1]))

    idx = [int(i) for i in idx]
    ix = np.ix_(idx, idx)
    M_s = np.asarray(M_hat, dtype=float)[ix]
    C_s = np.asarray(C_hat, dtype=float)[ix]
    K_s = np.asarray(K_hat, dtype=float)[ix]
    omega_s = np.asarray(omega_n_rad, dtype=float).ravel()[idx]
    modes_s = list(range(len(idx)))
    qg_mode = str(getattr(config, "pk_qg_eval_mode", "default")).lower()
    orig_qg = Qg_func

    def _slice_q(q):
        q = np.asarray(q, dtype=complex)
        if q.shape[0] == len(idx):
            # Tabella GAF gia' costruita sulla fetta modale (aero_method='UVLM'
            # con una base precalcolata sui soli modi risolti): affettarla di
            # nuovo userebbe gli indici della base intera e andrebbe fuori range.
            return q
        return q[ix]

    if qg_mode == "frozen_j0":
        def Qg_wrapped(k, V, p, s, j, **kw):
            r = orig_qg(k, V, p, 0, 0, **kw)
            if isinstance(r, tuple):
                return (_slice_q(r[0]),) + r[1:]
            return _slice_q(r)
    else:
        def Qg_wrapped(k, V, p, s, j, **kw):
            r = orig_qg(k, V, p, s, j, **kw)
            if isinstance(r, tuple):
                return (_slice_q(r[0]),) + r[1:]
            return _slice_q(r)

    print(
        f"[PK modal slice] DOF indices {idx} → {M_s.shape[0]}×{M_s.shape[0]}; "
        f"Qg eval mode={qg_mode!r}"
    )
    return M_s, C_s, K_s, omega_s, Qg_wrapped, modes_s


def _solve_pk_method(config, structural_results, coupling_results, aerogrid=None):

    print("Using P-K method.")
    aero_q_scaling.log_aero_im_Q_scale_once(config, context="PK method")

    if getattr(config, "pk_capytaine_matrices", False):
        print("PK with frequency-dependent Capytaine added mass / radiation damping.")
    else:
        print("Using classical P-K iterative method.")
    
    Z = coupling_results['Z']
    Apan = coupling_results['Apan']
    Z_qs = coupling_results['Z_qs']
    Z_force = coupling_results.get('Z_force', None)
    
    # Extract panel areas for theory-compliant formula
    panel_areas = None
    panel_areas = aerogrid['A']  # Panel areas for diagonal matrix A

    


    rho_f = config.rho_f[config.fluid]
    M_hat = structural_results.M_hat
    C_hat = structural_results.C_hat
    K_hat = structural_results.K_hat
    Phi_FSI_analisys = structural_results.dry_eigenvectors
    omega_n_rad = np.sqrt(np.asarray(structural_results.dry_eigenvalues, dtype=float)).ravel()

    aero_method = str(getattr(config, "aero_method", "DLM")).upper()
    if getattr(config, "quvlm_sharpy_structural_basis", False):
        from .hydroelastic_utl import quvlm_basis
        structural_results = quvlm_basis.apply_sharpy_structural_basis(structural_results, config)
        M_hat = structural_results.M_hat
        C_hat = structural_results.C_hat
        K_hat = structural_results.K_hat
        Phi_FSI_analisys = structural_results.dry_eigenvectors
        omega_n_rad = np.sqrt(np.asarray(structural_results.dry_eigenvalues, dtype=float)).ravel()
    if aero_method == "UVLM":
        print("PK aero provider: SHARPy linear-UVLM modal GAF table (aero_method='UVLM').")
        Qg_func = aero_system.make_Qg_func_uvlm(config)
        if getattr(config, "quvlm_qs_calibrate", False):
            # Per-mode quasi-steady normalization vs the DLM: both kernels are
            # potential-flow and must agree on the low-k circulatory Re diagonal;
            # a separable d_i*d_j residual is cross-basis modal normalization,
            # not physics (k-dependence, damping and apparent mass stay UVLM's).
            _Qg_dlm_cal = aero_system.make_Qg_func(
                config, config.paths['FSI'], Z, Apan, Z_qs, Phi_FSI_analisys,
                structural_results.dry_eigenvalues, c_sound=config.c_sound[config.fluid],
                out_dir_klist=config.qjj_dir, out_dir_vjj=config.vjj_dir,
                alpha_r=config.alpha_r, Z_force=Z_force, panel_areas=panel_areas,
            )
            _k_cal = float(getattr(config, "quvlm_qs_k_cal", 0.1))
            _V_cal = float(np.median(np.asarray(config.V_list, dtype=float)))
            _Qu = np.atleast_2d(Qg_func(_k_cal, _V_cal, None, 0, 0))
            _Qd = np.atleast_2d(_Qg_dlm_cal(_k_cal, _V_cal, None, 0, 0))
            _ru = np.real(np.diag(_Qu))
            _rd = np.real(np.diag(_Qd))
            _dvec = np.ones(len(_ru))
            _ok = (_ru * _rd) > 0
            _dvec[_ok] = np.sqrt(_rd[_ok] / _ru[_ok])
            if not _ok.all():
                print(f"[quvlm_qs_calibrate WARN] sign mismatch on modes "
                      f"{np.flatnonzero(~_ok)} at k={_k_cal}: those left uncalibrated")
            _D = np.diag(_dvec)
            print(f"[quvlm_qs_calibrate] per-mode QS factors d = "
                  f"{np.array2string(_dvec, precision=4)} (k_cal={_k_cal}, V={_V_cal:.2f})")
            _Qg_uvlm_raw = Qg_func

            def Qg_func(k_red, V_dlm, p_conv, s, j, store_w_modal=False,
                        still_fluid_eigs=False, _f=_Qg_uvlm_raw, _D=_D):
                r = _f(k_red, V_dlm, p_conv, s, j, store_w_modal=store_w_modal,
                       still_fluid_eigs=still_fluid_eigs)
                if isinstance(r, tuple):
                    return (_D @ r[0] @ _D,) + tuple(r[1:])
                return _D @ r @ _D
        if getattr(config, "quvlm_compare_dlm", False):
            # calibration aid: print UVLM vs DLM modal GAF at matched k (j=0,s=0 path)
            _Qg_dlm = aero_system.make_Qg_func(
                config, config.paths['FSI'], Z, Apan, Z_qs, Phi_FSI_analisys,
                structural_results.dry_eigenvalues, c_sound=config.c_sound[config.fluid],
                out_dir_klist=config.qjj_dir, out_dir_vjj=config.vjj_dir,
                alpha_r=config.alpha_r, Z_force=Z_force, panel_areas=panel_areas,
            )
            _Vc = float(np.median(np.asarray(config.V_list, dtype=float)))
            print("[quvlm_compare_dlm] modal GAF comparison (post-flip), V=%.2f m/s" % _Vc)
            for _kc in (0.1, 0.5, 2.0, 8.0, 20.0):
                _Qu = np.atleast_2d(Qg_func(_kc, _Vc, None, 0, 0))
                _Qd = np.atleast_2d(_Qg_dlm(_kc, _Vc, None, 0, 0))
                print(f"  k={_kc:5.2f}  Q_uvlm[0,:2]={np.array2string(_Qu[0, :2], precision=4)}"
                      f"  Q_dlm[0,:2]={np.array2string(_Qd[0, :2], precision=4)}")
                print(f"           Q_uvlm[1,:2]={np.array2string(_Qu[1, :2], precision=4)}"
                      f"  Q_dlm[1,:2]={np.array2string(_Qd[1, :2], precision=4)}")
    else:
        Qg_func = aero_system.make_Qg_func(
            config,
            config.paths['FSI'],
            Z,
            Apan,
            Z_qs,
            Phi_FSI_analisys,
            structural_results.dry_eigenvalues,
            c_sound=config.c_sound[config.fluid],
            out_dir_klist=config.qjj_dir,
            out_dir_vjj=config.vjj_dir,
            alpha_r=config.alpha_r,
            Z_force=Z_force,
            panel_areas=panel_areas,
        )

    M_hat, C_hat, K_hat, omega_n_rad, Qg_func, pk_modes = _apply_pk_modal_slice_and_qg(
        config, M_hat, C_hat, K_hat, omega_n_rad, Qg_func,
    )

    capytaine_hydro = None
    if getattr(config, "pk_capytaine_matrices", False):
        capy_data, capy_npz = _load_capytaine_modal_radiation(config)
        omega_h = np.asarray(capy_data["omega"], dtype=float).ravel()
        A_h = np.asarray(capy_data["added_mass"], dtype=float)
        B_h = np.asarray(capy_data["added_damping"], dtype=float)
        n_modes_modal = int(M_hat.shape[0])
        n_capy = int(A_h.shape[1])
        if A_h.shape[1] != n_modes_modal or A_h.shape[2] != n_modes_modal:
            raise ValueError(
                f"Capytaine modal matrix shape {A_h.shape} not compatible with "
                f"{n_modes_modal} dry/modal modes (num_modes_egv)."
            )
        depth_used = capy_data.get("depth", np.nan)
        depth_used = (
            float(np.asarray(depth_used).ravel()[0])
            if np.asarray(depth_used).size > 0
            else np.nan
        )
        apply_idx = _resolve_capytaine_apply_indices(config, n_capy, n_modes_modal)
        source_idx = _resolve_capytaine_source_indices(config, n_capy, apply_idx)
        capytaine_hydro = {
            "omega_h": omega_h,
            "added_mass_h": A_h,
            "added_damping_h": B_h,
            "apply_indices": apply_idx,
            "source_indices": source_idx,
            "n_modal": n_modes_modal,
        }
        print(
            f"[PK+Capytaine] depth={depth_used:.4f} m, file={capy_npz}\n"
            f"  A(ω), B(ω) interpolated at ω=k·V each PK iteration; "
            f"apply DOF {apply_idx}, Capytaine tensor indices {source_idx}"
        )
        pk_bem_mode = str(getattr(config, "pk_bem_mode", "full")).lower()
        if pk_bem_mode == "delta":
            if aero_method != "UVLM":
                raise ValueError(
                    "pk_bem_mode='delta' requires aero_method='UVLM' "
                    "(intact UVLM GAF + A_capy - A_uvlm injection)."
                )
            if getattr(config, "aero_b2_removal", False):
                print("[pk_bem_mode=delta] forcing aero_b2_removal=False (UVLM GAF stays intact).")
                config.aero_b2_removal = False
            from .hydroelastic_utl import bem_delta

            if getattr(config, "pk_bem_delta_fit_k", None) is not None:
                # massa UVLM NON circolatoria dalla tabella GAF in uso (stessa Q del p-k):
                # fit Re Q = a + (2/rho) A k^2 su k in [lo, hi] (come TFOIL_SIMPLE/uvlm_nc_mass.py)
                A_off, delta_meta = bem_delta.fit_uvlm_nc_mass(
                    Qg_func, config.pk_bem_delta_fit_k, rho_f, n_modes_modal)
            else:
                A_off, delta_meta = bem_delta.resolve_uvlm_offset_embedded(
                    config,
                    capy_npz,
                    n_modes_modal,
                    apply_idx,
                    source_idx,
                )
            capytaine_hydro["A_uvlm_offset"] = A_off
            capytaine_hydro["pk_bem_mode"] = "delta"
            capytaine_hydro["delta_meta"] = delta_meta
            print(
                "[pk_bem_mode=delta] M_hydro = A_capy(ω) - A_uvlm "
                f"(offset from {delta_meta.get('npz', '?')})"
            )
        elif pk_bem_mode not in ("full", "replace", ""):
            raise ValueError(f"Unknown pk_bem_mode={pk_bem_mode!r}; use 'full' or 'delta'.")
        if getattr(config, "added_mass_strip_theory", False):
            print(
                "[PK+Capytaine WARN] added_mass_strip_theory=True: M_hat may already "
                "include strip AM; check for double-counting with Capytaine A(ω)."
            )

    matrix_iter_state = {}
    qmodal_history = []
    pk_discrete = bool(getattr(config, "pk_discrete_newmark", False))
    pk_discrete_dt = None
    if pk_discrete:
        pk_discrete_dt = float(getattr(config, "newmark_dt", 0.1143))
        if pk_discrete_dt <= 0.0:
            raise ValueError(f"pk_discrete_newmark requires positive newmark_dt, got {pk_discrete_dt}")
        nd = float(getattr(config, "newmark_damp", 1e-4))
        print(
            f"[PK discrete Newmark] Ass from newmark_ss parity; "
            f"newmark_damp={nd:g}, dt={pk_discrete_dt:g} s; structural C_hat ignored."
        )
    build_fn = aero_system.make_aero_build_matrix_fn(
        rho_f, M_hat, C_hat, K_hat, Qg_func, matrix_iter_state, config=config,
        qmodal_history=qmodal_history,
        capytaine_hydro=capytaine_hydro,
        pk_discrete_newmark=pk_discrete,
        newmark_dt=pk_discrete_dt,
        newmark_damp=float(getattr(config, "newmark_damp", 1e-4)) if pk_discrete else None,
    )
    on_converge_fn = aero_system.make_aero_on_converge_fn(
        rho_f, M_hat, C_hat, K_hat, Qg_func, matrix_iter_state, config=config,
        qmodal_history=qmodal_history,
        capytaine_hydro=capytaine_hydro,
    )
    solver = pk_method.PkDiagnostics()
    static_runner = _create_static_hydroelastic_runner(
        config, structural_results, coupling_results, aerogrid
    )
    cp_plotter = _create_cp_snapshot_plotter(
        config, aerogrid, config.V_list, structural_results, coupling_results
    )
    on_velocity_step_fn = _make_on_velocity_step(static_runner, cp_plotter)

    results = pk_method.pk_sweep(
        config.V_list,
        pk_modes,
        omega_n_rad,
        build_fn,
        config.k_guess,
        rho_f,
        config.pk_tol,
        config.pk_fXK0,
        config.pk_fRLX,
        config.pk_freq_margin,
        config.pk_w_freq,
        config.pk_w_mac,
        config.pk_max_iter,
        dry_vals=structural_results.dry_eigenvalues,
        dry_vecs=structural_results.dry_eigenvectors,
        perturb_k=config.pk_perturb_k,
        mac_matching=config.mac_matching,
        last_converged_mode_matching=config.last_converged_mode_matching,
        matrix_iter_state=matrix_iter_state,
        on_converge_fn=on_converge_fn,
        diagnostics=solver,
        skip_inter_mode_on_perfect_mac=config.pk_skip_inter_mode_on_perfect_mac,
        pk_predict_and_select=getattr(config, "pk_predict_and_select", True),
        min_root_sep_rad=getattr(config, "pk_min_root_sep_rad", 1.0),
        omega_phys_min_frac=getattr(config, "pk_omega_phys_min_frac", 0.45),
        tracking_policy=str(getattr(config, "pk_tracking_policy", "paper")),
        mac_reference=str(getattr(config, "pk_mac_reference", "previous_velocity")),
        freq_jump_terminate_frac=getattr(config, "pk_freq_jump_terminate_frac", None),
        mac_terminate_floor=getattr(config, "pk_mac_terminate_floor", 0.5),
        track_seed_omegas=getattr(config, "pk_track_seed_omegas", None),
        min_pk_iter=int(getattr(config, "pk_min_iter", 1)),
        on_velocity_step_fn=on_velocity_step_fn,
        pk_discrete_dt=pk_discrete_dt,
    )

    # Non-oscillatory (divergence-type) instabilities: the p-k V-g/V-f
    # bookkeeping only tracks complex/oscillatory roots, so a genuine real,
    # positive-growth-rate eigenvalue (found anywhere in the raw spectrum
    # during the sweep -- see pk_method.PkDiagnostics.real_instability_events)
    # would otherwise be silently dropped, reading as "no flutter found"
    # when the system may actually already be unstable via divergence
    # (2026-09-04, GOLAND_water: verified real eigenvalues at V=2.53 m/s,
    # k well inside the valid GAF table range -- not an extrapolation
    # artifact; see docs/notes/gino_passo0_gaf_table_stale_axes.md #18).
    _instab_events = list(getattr(solver, "real_instability_events", []) or [])
    if _instab_events:
        _instab_Vs = sorted({round(float(e["V"]), 4) for e in _instab_events})
        print(
            f"\n⚠ NON-OSCILLATORY (divergence-type) INSTABILITY detected at "
            f"{len(_instab_Vs)} velocity value(s) during the sweep: "
            f"{_instab_Vs[:15]}{'...' if len(_instab_Vs) > 15 else ''} m/s.\n"
            f"  These are real, positive-growth-rate eigenvalues (omega=0, not "
            f"oscillatory) -- invisible to V-g/V-f by construction. See "
            f"real_instability_events.csv for the full list."
        )
        try:
            import csv as _csv
            _outdir = getattr(config, "output_dir", None) or getattr(config, "modal_dir", ".")
            os.makedirs(_outdir, exist_ok=True)
            _csv_path = os.path.join(_outdir, "real_instability_events.csv")
            with open(_csv_path, "w", newline="") as _f:
                _w = _csv.writer(_f)
                _w.writerow(["Velocity_m_s", "mode_slot", "iteration", "k", "growth_rate_1_s"])
                for _e in _instab_events:
                    for _g in _e["growth_rates"]:
                        _w.writerow([_e["V"], _e["mode_slot"], _e["iteration"], _e["k"], float(_g)])
            print(f"  Saved: {_csv_path}")
        except Exception as _exc:
            print(f"[Warning] Could not save real_instability_events.csv: {_exc}")

    # Process and return results
    V, g, omega, conv = vgvf_plotting.results_to_arrays(results, pk_modes)

    # Copertura in frequenza: ogni radice deve stare dentro la griglia BEM (omega,
    # superficie libera inclusa) e dentro la tabella UVLM (k = omega/V). Fuori,
    # gli interpolatori saturano/estrapolano in silenzio -> va ricalcolato.
    try:
        from .hydroelastic_utl import freq_coverage
        _w = np.abs(np.asarray(omega, float))
        _VV = np.repeat(np.asarray(V, float)[:, None], _w.shape[1], axis=1)
        freq_coverage.report_and_save(
            config, np.c_[_w.ravel(), _VV.ravel()],
            getattr(config, "output_dir", None) or getattr(config, "modal_dir", "."),
            bem_omega=None if capytaine_hydro is None else capytaine_hydro["omega_h"],
            bem_B=None if capytaine_hydro is None else capytaine_hydro["added_damping_h"],
            uvlm_kmax=getattr(config, "_quvlm_kmax_table", None) if aero_method == "UVLM" else None,
            wet_omega=_w[0][np.isfinite(_w[0])],
            omega_cap=2.0 * float(np.max(omega_n_rad)))
    except Exception as _exc:
        print(f"[copertura] controllo non eseguito: {_exc}")
    
    Vf, ff = vgvf_plotting.select_primary_flutter(
        V,
        g,
        omega,
        selection=getattr(config, "flutter_primary_selection", "min_V"),
        freq_band_hz=getattr(config, "flutter_freq_band_hz", None),
        freq_target_hz=getattr(config, "flutter_freq_target_hz", None),
        V_target_kn=getattr(config, "flutter_V_target_kn", None),
    )

    # Legacy end-of-run Cp at fixed k (disabled when sweep snapshots are on)
    if aerogrid is not None and not getattr(config, "plot_cp_sweep_snapshots", True):
        post_processing.plot_cp_at_fixed_k_values(config, aerogrid, results, k_values=[0.001, 0.5, 1.0])

    # Plot all Q_damp entries (Re/Im) used at every PK iteration
    if getattr(config, "plot_qmodal_pk_iterations", True):
        try:
            post_processing.plot_qmodal_pk_iteration_history(config, qmodal_history)
        except Exception as e:
            print(f"[Warning] Q_damp PK-iteration plotting failed: {e}")

    # Post-process: sample Q_modal entries at convergence for each speed
    try:
        post_processing.plot_qmodal_entries_at_convergence(
            config,
            structural_results,
            coupling_results,
            _finalize_flutter_results(
                V, g, omega,
                config.modes_to_analyze,
                Vf, ff,
                config,
                raw_results=results,
                pk_solver=solver,
            ),
            mode_index=0,
            save_csv=True
        )
    except Exception as e:
        print(f"[Warning] Q_modal post-processing failed: {e}")

    # [GINO_NEW] STATIC DIVERGENCE indicator via det(K_eff(k->0)) = 0. This is the CORRECT condition
    # (K_eff singular). Do NOT use the min eigenvalue of the symmetric part of K_eff: the aero makes
    # K_eff non-symmetric, and its symmetric part can go indefinite WITHOUT the matrix being singular,
    # giving a spurious early "divergence". det(K_eff)/det(K) starts at 1 and crosses 0 at divergence.
    try:
        import os as _os
        import numpy.linalg as _la
        _Vlist = np.atleast_1d(config.V_list).astype(float)
        _Vref = float(_Vlist[len(_Vlist) // 2])
        _ReQ0 = np.real(np.atleast_2d(Qg_func(0.05, max(_Vref, 1e-3), None, 0, 0)))  # quasi-steady aero stiffness
        _detK0 = _la.det(K_hat)
        _det_ratio = np.array([_la.det(K_hat - 0.5 * rho_f * V**2 * _ReQ0) / _detK0 for V in _Vlist])
        _V_div = None
        for _i in range(1, len(_Vlist)):
            if _det_ratio[_i - 1] > 0 and _det_ratio[_i] <= 0:
                _V_div = _Vlist[_i]
                break
        if _V_div is not None:
            print(f"Static divergence (det(K_eff)=0) at V = {_V_div:.3f} m/s ({_V_div * 1.94384:.1f} kn)")
        else:
            print("Static divergence (det(K_eff)=0): NONE in the swept range (instability is flutter, not divergence).")
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as _plt
            _outdir = getattr(config, 'output_dir', None) or getattr(config, 'modal_dir', '.')
            _os.makedirs(_outdir, exist_ok=True)
            _kn = 1.94384 if getattr(config, 'v_knots', False) else 1.0
            _xv = _Vlist * _kn
            _fig, _ax = _plt.subplots(figsize=(8, 5))
            _ax.plot(_xv, _det_ratio, '-', color='tab:red', lw=1.9, label='det(K_eff)/det(K0)')
            _ax.axhline(0, color='k', lw=.7)
            if _V_div is not None:
                _ax.plot(_V_div * _kn, 0, 'kv', ms=10, label=f'divergence {_V_div * _kn:.1f}')
            _ax.set_xlabel('V [kn]' if _kn != 1.0 else 'V [m/s]')
            _ax.set_ylabel('det(K_eff)/det(K0)')
            _ax.set_title('Static divergence indicator  det(K_eff(k->0)) = 0')
            _ax.grid(alpha=.3)
            _ax.legend()
            _fig.tight_layout()
            _p = _os.path.join(_outdir, 'divergence_indicator.png')
            _fig.savefig(_p, dpi=110)
            _plt.close(_fig)
            np.savetxt(_os.path.join(_outdir, 'divergence_indicator.csv'),
                       np.column_stack([_Vlist, _det_ratio]),
                       delimiter=',', header='V_mps,det_Keff_over_detK0', comments='')
            print(f"Divergence indicator plot saved to {_p}")
        except Exception as _ep:
            print(f"[Warning] divergence plot failed: {_ep}")
    except Exception as _e:
        print(f"[Warning] divergence indicator failed: {_e}")

    print(f"Flutter analysis complete. Estimated flutter speed: {Vf}")

    return _finalize_static_hydroelastic_pass(
        _finalize_flutter_results(
            V, g, omega,
            config.modes_to_analyze,
            Vf, ff,
            config,
            raw_results=results,
            pk_solver=solver,
            converged=conv,
        ),
        static_runner,
        config,
    )