# FSI/COUPLING/flutter_solver.py

import numpy as np
from collections import namedtuple
from .hydroelastic_utl import pk_solverv3
from .hydroelastic_utl import pk_solver_capytaine
from .hydroelastic_utl import pk_method, aero_system, hydro_system, empirical_fluid_damping, aero_q_scaling
from .hydroelastic_utl import vgvf_plotting
from .hydroelastic_utl import roger_mode_tracking
from .hydroelastic_utl import post_processing
from .hydroelastic_utl import roger_fit
import pickle
import os
import importlib.util
from types import SimpleNamespace

FlutterResults = namedtuple('FlutterResults', [
    'velocities', 'damping', 'frequencies', 'modes', 'flutter_speed', 'flutter_frequency',
    'raw_results', 'pk_solver', 'dlm_participants_history',
])


def _k_weights_for_roger_fit(k_list, V, omega_n, sigma_frac=0.6):
    """Gaussian weights peaking at k ≈ ω_n/V (wet reduced frequencies at this airspeed)."""
    k_list = np.asarray(k_list, dtype=float)
    w = np.zeros(k_list.size, dtype=float)
    for om in np.asarray(omega_n, dtype=float).ravel():
        if om <= 0 or V <= 0:
            continue
        kc = om / V
        sigma = max(sigma_frac * kc, 0.05)
        w = np.maximum(w, np.exp(-0.5 * ((k_list - kc) / sigma) ** 2))
    if np.max(w) < 1e-12:
        w[:] = 1.0
    else:
        w /= np.max(w)
    return w


def _roger_lag_poles(config, k_list_V, n_lag):
    """Return Roger lag poles in PanelAero k units [1/m]."""
    if getattr(config, 'rfa_adaptive_blag', False):
        k_lo = max(float(k_list_V[1]) if k_list_V.size > 1 else float(k_list_V[0]), 1e-6)
        return np.logspace(np.log10(k_lo * 0.3), np.log10(k_list_V[-1] * 1.5), n_lag), None

    blag = np.asarray(config.blag, dtype=float).ravel()
    if blag.size < n_lag:
        raise ValueError(f"config.blag must have at least {n_lag} entries (n_lag={n_lag})")
    blag = blag[:n_lag]

    if not getattr(config, 'rfa_blag_dimensionless', False):
        return blag, None

    ref = getattr(config, 'rfa_blag_ref_length', 'semichord')
    if isinstance(ref, str):
        ref_key = ref.lower()
        chord = float(getattr(config, 'chord'))
        if ref_key in ('semichord', 'half_chord', 'c/2'):
            ref_length = 0.5 * chord
        elif ref_key in ('chord', 'c'):
            ref_length = chord
        else:
            raise ValueError(
                "config.rfa_blag_ref_length must be 'semichord', 'chord', or a positive length"
            )
    else:
        ref_length = float(ref)
    if ref_length <= 0.0:
        raise ValueError(f"config.rfa_blag_ref_length must be positive, got {ref_length}")

    return blag / ref_length, ref_length


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


def _load_capytaine_modal_radiation(config):
    io_path = os.path.join(config.paths["FSI"], "FLUID", "capytaine", "modal_radiation_io.py")
    spec = importlib.util.spec_from_file_location("modal_radiation_io", io_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    capy_dir = config.capytaine_results_dir
    depth_value = float(np.asarray(config.depth).ravel()[int(config.depth_index)])
    depth_tag = f"{depth_value:g}".replace(".", "p").replace("-", "m")
    capy_npz = getattr(
        config,
        "capytaine_modal_radiation_file",
        os.path.join(capy_dir, f"modal_radiation_AB_depth_{depth_tag}.npz"),
    )
    if not os.path.exists(capy_npz):
        legacy = os.path.join(capy_dir, f"modal_radiation_AB_depth_{depth_value:g}.npz")
        if os.path.exists(legacy):
            capy_npz = legacy
    if not os.path.exists(capy_npz):
        raise FileNotFoundError(f"Capytaine modal radiation file not found: {capy_npz}")
    return mod.load_modal_radiation_npz(capy_npz), capy_npz


def _fit_roger_to_Q_modal(Q_modal_array, k_list, blag, Nm, N, row_weights=None, include_b2=True):
    """
    Fit Roger RFA coefficients to modal GAF data.

    Parameters
    ----------
    include_b2 : bool
        If False (hydroelastic correction with strip/Capytaine added mass in M_hat),
        fit only B0, B1 and lag terms — do not allocate DLM added mass to B2.

    Returns
    -------
    B0, B1, B2 : (Nm, Nm) real ndarray  (B2 is zero when include_b2 is False)
    Blag       : list of N (Nm, Nm) real ndarray
    """
    Nk = len(k_list)

    rows = []
    for k in k_list:
        k2 = k ** 2
        if include_b2:
            re_row = [1.0, 0.0, -k2] + [k2 / (k2 + b**2) for b in blag]
            im_row = [0.0, k, 0.0] + [k * b / (k2 + b**2) for b in blag]
        else:
            re_row = [1.0, 0.0] + [k2 / (k2 + b**2) for b in blag]
            im_row = [0.0, k] + [k * b / (k2 + b**2) for b in blag]
        rows.append(re_row)
        rows.append(im_row)
    A_basis = np.array(rows, dtype=float)

    B0 = np.zeros((Nm, Nm))
    B1 = np.zeros((Nm, Nm))
    B2 = np.zeros((Nm, Nm))
    Blag = [np.zeros((Nm, Nm)) for _ in range(N)]
    i_b2 = 2 if include_b2 else None
    i_lag0 = 3 if include_b2 else 2

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
            apply_structural_pitch=bool(getattr(config, "pitch_rotate_beam", False)),
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

        out_dir = os.path.join(config.output_dir, config.name)
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


def _reconstruct_Q_modal(k, B0, B1, B2, Blag, blag, include_b2=True):
    """Evaluate Q_modal(ik) from Roger coefficients."""
    k2 = k ** 2
    Q = B0 + 1j * k * B1
    if include_b2:
        Q -= k2 * B2
    for Bl, b in zip(Blag, blag):
        Q += Bl * (k2 + 1j * k * b) / (k2 + b**2)
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

    M_eff = M_hat - 0.5*rho*B2
    C_eff = C_hat - aero_im_Q_scale*0.5*rho*V*B1
    K_eff = K_hat - 0.5*rho*V^2*B0
    q_dyn = 0.5*rho*V^2

    ``aero_im_Q_scale`` multiplies the B1 (Im Q) damping term after the
    unscaled Roger fit. If ``aero_im_Q_scale_lags`` is True, the same factor
    also multiplies the lag-state aerodynamic forcing ``q_dyn * Blag``.
    """
    n     = Nm
    q_dyn = 0.5 * rho * V ** 2
    n_tot = (2 + N) * n

    # Hydroelastic correction: added mass in M_hat (strip / Capytaine), not DLM B2.
    # Set rfa_apply_b2_mass=True only for pure-DLM RFA without external added mass.
    apply_b2 = getattr(config, 'rfa_apply_b2_mass', False)
    if apply_b2:
        M_eff = M_hat - 0.5 * rho * B2
    else:
        M_eff = np.array(M_hat, copy=True)
    if M_hydro_add is not None:
        M_eff = M_eff + np.asarray(M_hydro_add, dtype=float)
    aero_im_scale = aero_q_scaling.aero_im_Q_scale(config)
    C_eff = C_hat - aero_im_scale * 0.5 * rho * V * B1
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


def _extract_flutter_modes(A_aug, Nm, omega_n, return_eigenvectors=False,
                           vecs_previous=None, freqs_previous=None, dimensionless_vgvf_results=False):


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
        from COUPLING.hydroelastic_utl import roger_mode_tracking
        
        n_candidates = min(len(eigs_filt), 2 * Nm)
        indices_best = roger_mode_tracking._select_modes_by_mac(
            eigs_filt[:n_candidates],
            vecs_filt[:, :n_candidates],
            freqs_filt[:n_candidates],
            eigs_filt,
            vecs_filt,
            freqs_previous,
            vecs_previous,
            Nm,
            n_structural_dof=2 * Nm,
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


def solve(config, structural_results, coupling_results, aerogrid=None):

    print("\n--- Solving for Flutter ---")
    
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
    method_label="Roger RFA",
):
    """
    Direct Roger RFA flutter sweep (eigenvalues of A_aug at each V, MAC tracking).

    Optional constant Capytaine added mass / radiation damping in M_eff, C_eff
  (asymptotic values — no per-step omega interpolation, no PK iteration).
    """
    from Qjj.precompute_qjj import interp_qjj_from_disk_old

    print(f"Using {method_label} — direct eigenvalue sweep (no PK iteration).")
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
    _collect_rfa_matrix_history = (
        getattr(config, 'plot_DLM_participants', False)
        or getattr(config, 'plot_stiffness_damping_contributions', True)
    )
    dlm_participants_history = [] if _collect_rfa_matrix_history else None

    requested_b2 = getattr(config, 'rfa_apply_b2_mass', False)
    use_b2 = requested_b2
    if M_hydro_add is not None and use_b2:
        print(
            "[RFA WARN] rfa_apply_b2_mass=True with external hydro added mass: "
            "M_eff will include both -0.5*rho*B2 and M_hydro_add. "
            "Use this only for explicit sensitivity checks because it can double-count inertia."
        )
    if config.added_mass_strip_theory and use_b2:
        print(
            "[RFA WARN] strip/Capytaine added mass is in M_hat but rfa_apply_b2_mass=True — "
            "double-counts inertia. Use rfa_apply_b2_mass=False (hydroelastic correction)."
        )
    if not use_b2:
        print(
            "[RFA] Hydroelastic correction: M_eff = M_hat (strip/Capytaine AM); "
            "Roger fit without B2; A0/K_aero, A1/C_aero, lags = circulatory DLM."
        )
    else:
        print(
            "[RFA] rfa_apply_b2_mass=True: Roger fit includes B2 and "
            "M_eff = M_hat - 0.5*rho*B2 (+ external hydro AM if provided)."
        )

    aero_q_scaling.log_aero_im_Q_scale_once(config, context="Roger RFA sweep")

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

        # STEP A: k-grid and lag poles for Roger fit
        if getattr(config, 'rfa_adaptive_k_list', False):
            omega_phys = np.linspace(0.1 * omega_n[0], 3.0 * omega_n[-1], 100)
            k_list_V   = np.clip(omega_phys / V, k_data_min, k_data_max)
            k_list_V   = np.unique(k_list_V)
        else:
            k_list_V = np.asarray(config.k_list, dtype=float)
            k_list_V = k_list_V[(k_list_V >= k_data_min) & (k_list_V <= k_data_max)]
            if k_list_V.size < 8:
                print(
                    f"  [WARN] Only {k_list_V.size} k points inside Qjj range "
                    f"[{k_data_min}, {k_data_max}]; widen config.k_list or recompute Qjj."
                )
        Nk = len(k_list_V)

        blag, blag_ref_length = _roger_lag_poles(config, k_list_V, N)

        print(f"  STEP A: k_list_V has {Nk} points, range [{k_list_V[0]:.6f}, {k_list_V[-1]:.6f}]")
        print(f"  STEP A: lag poles blag = {np.array2string(blag, precision=4)} [1/m]")
        if blag_ref_length is not None:
            beta = np.asarray(config.blag, dtype=float).ravel()[:N]
            print(
                "  STEP A: dimensionless lag poles beta = "
                f"{np.array2string(beta, precision=4)} converted with L_ref={blag_ref_length:.6g} m"
            )

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

        # Optional: drop very small k (quasi-steady) that dominate LSQ but are not flutter
        k_fit_min = float(getattr(config, 'rfa_k_fit_min', 0.0))
        if k_fit_min > 0:
            fit_mask = k_list_V >= k_fit_min
            k_fit = k_list_V[fit_mask]
            Q_fit_data = Q_modal_array[fit_mask]
        else:
            k_fit, Q_fit_data = k_list_V, Q_modal_array

        row_weights = None
        if getattr(config, 'rfa_weight_fit_k', False) and k_fit.size > 0:
            row_weights = _k_weights_for_roger_fit(
                k_fit, V, omega_n,
                sigma_frac=float(getattr(config, 'rfa_weight_fit_sigma', 0.6)),
            )

        # STEP C: Roger fit
        B0, B1, B2, Blag = _fit_roger_to_Q_modal(
            Q_fit_data, k_fit, blag, n_modes, N,
            row_weights=row_weights, include_b2=use_b2,
        )

        b2_note = "with B2" if use_b2 else "no B2 (AM in M_hat)"
        print(f"  STEP C: Roger fit complete ({b2_note}) — {len(Blag)} lag terms")

        # STEP C.4: Fit quality check
        max_err = 0.0
        for i_k, k in enumerate(k_list_V):
            Q_fit = _reconstruct_Q_modal(k, B0, B1, B2, Blag, blag, include_b2=use_b2)
            Q_ref = Q_modal_data[i_k]
            err = np.linalg.norm(Q_fit - Q_ref, 'fro') / np.linalg.norm(Q_ref, 'fro')
            max_err = max(max_err, err)
            if i_k % max(1, Nk // 3) == 0:
                print(f"    k={k:.4f}: fit error = {err:.2e}")
        print(f"  STEP C: Max fit error = {max_err:.2e} (target < 1%)")

        ## STEP C.5: capytaine BEM modal analysis
        #if config.capytaine_BEM_modal_analysis:
        #    M_added

        # STEP D: Augmented state matrix
        A_aug, M_eff, C_eff, K_eff = _build_A_aug(
            config, V, rho, M_hat, C_hat, K_hat, B0, B1, B2, Blag, blag, n_modes, N,
            M_hydro_add=M_hydro_add, C_hydro_add=C_hydro_add, apply_b2_mass=use_b2,
        )

        if dlm_participants_history is not None:
            aero_im_scale = aero_q_scaling.aero_im_Q_scale(config)
            K_hat_arr = np.array(K_hat, copy=True)
            C_hat_arr = np.array(C_hat, copy=True)
            M_hat_arr = np.array(M_hat, copy=True)
            zeros = np.zeros_like(M_hat_arr)
            M_aero_dlm = 0.5 * rho * np.asarray(B2, dtype=float)
            M_hydro_arr = (
                np.asarray(M_hydro_add, dtype=float)
                if M_hydro_add is not None else zeros.copy()
            )
            K_aero_dlm = 0.5 * rho * V**2 * np.asarray(B0, dtype=float)
            K_hydro_arr = zeros.copy()
            C_aero_dlm = aero_im_scale * 0.5 * rho * V * np.asarray(B1, dtype=float)
            C_hydro_arr = (
                np.asarray(C_hydro_add, dtype=float)
                if C_hydro_add is not None else zeros.copy()
            )
            K_eff_arr = np.array(K_eff, copy=True)
            C_eff_arr = np.array(C_eff, copy=True)
            M_eff_arr = np.array(M_eff, copy=True)
            C_emp_arr = empirical_fluid_damping.modal_empirical_C(config, V, M_eff_arr)
            # Same layout as PK contributions_history (one matrix per mode slot).
            _slot = lambda M: [M, M]
            dlm_participants_history.append({
                'V': float(V),
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

        # STEP E: Eigenvalue analysis (with eigenvectors for MAC)
        # First extraction is frequency-based; subsequent extractions use MAC to find best modes
        if i_V > 0 and len(vecs_sweep) > 0:
            # Have previous modes: use MAC-based selection to search for modes
            vecs_V_prev = vecs_sweep[-1]
            freq_V_prev = freq_sweep[-1]
            g_V, freq_V, vecs_V, eigs_V = _extract_flutter_modes(
                A_aug, n_modes, omega_n, return_eigenvectors=True,
                vecs_previous=vecs_V_prev, freqs_previous=freq_V_prev, dimensionless_vgvf_results=config.dimensionless_vgvf_results
            )
        else:
            # First velocity step: use frequency-based selection
            g_V, freq_V, vecs_V, eigs_V = _extract_flutter_modes(A_aug, n_modes, omega_n, return_eigenvectors=True, dimensionless_vgvf_results=config.dimensionless_vgvf_results)
        
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
                n_structural_dof=2 * n_modes if getattr(config, 'rfa_mac_structural_dof', True) else None,
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

    rfa_pk_solver = None
    if dlm_participants_history:
        rfa_pk_solver = SimpleNamespace(contributions_history=dlm_participants_history)

    return FlutterResults(
        velocities=V_arr, damping=g_arr, frequencies=f_arr,
        modes=config.modes_to_analyze,
        flutter_speed=Vf, flutter_frequency=ff,
        raw_results=None, pk_solver=rfa_pk_solver,
        dlm_participants_history=dlm_participants_history,
    )


def _solve_roger_fit_method(config, V_list, structural_results, coupling_results, aerogrid):
    """Roger RFA + strip (or dry) — same core as RFA_PK without Capytaine matrices."""
    return _run_roger_rfa_sweep(
        config, V_list, structural_results, coupling_results, aerogrid,
        method_label="Roger RFA (DLM circulatory + structural M_hat)",
    )


def _solve_RFA_PK_method(config, V_list, structural_results, coupling_results, aerogrid):
    """
    Hydroelastic Roger RFA: same DLM Roger fit / MAC sweep as ``roger_fit``, plus constant
    asymptotic Capytaine added mass and radiation damping (no PK, no omega interpolation).
    """
    capy_data, capy_npz = _load_capytaine_modal_radiation(config)
    omega_h = np.asarray(capy_data["omega"], dtype=float).ravel()
    A_h = np.asarray(capy_data["added_mass"], dtype=float)
    B_h = np.asarray(capy_data["added_damping"], dtype=float)
    n_modes = len(config.modes_to_analyze)
    if A_h.shape[1] != n_modes or A_h.shape[2] != n_modes:
        raise ValueError(
            f"Capytaine modal matrix shape {A_h.shape} not compatible with {n_modes} modes."
        )
    omega_n = np.sqrt(np.abs(structural_results.dry_eigenvalues))
    depth_used = capy_data.get("depth", np.nan)
    depth_used = float(np.asarray(depth_used).ravel()[0]) if np.asarray(depth_used).size > 0 else np.nan
    print(f"[RFA+Capytaine] depth={depth_used:.4f} m, file={capy_npz}")
    print(
        "[RFA+Capytaine] Same Roger RFA as roger_fit (fixed k_list/blag; "
        "DLM B2 controlled by rfa_apply_b2_mass); Capytaine A,B = "
        "asymptotic constants in M_eff, C_eff (no PK iteration)."
    )

    A_asy, B_asy = _capytaine_asymptotic_modal_matrices(omega_h, A_h, B_h, omega_n, config)

    if getattr(config, "empirical_fluid_damping", False):
        M_sample = np.array(structural_results.M_hat, copy=True) + A_asy
        V_sample = float(np.median(V_list)) if len(V_list) else 20.0
        C_emp_sample = empirical_fluid_damping.modal_empirical_C(config, V_sample, M_sample)
        print(empirical_fluid_damping.describe_empirical_C(config, V_sample, M_sample, C_emp_sample))

    if config.added_mass_strip_theory:
        print(
            "[RFA+Capytaine WARN] added_mass_strip_theory=True: M_hat may already include "
            "strip AM; Capytaine A_asy is still added (check for double-counting)."
        )

    results = _run_roger_rfa_sweep(
        config,
        V_list,
        structural_results,
        coupling_results,
        aerogrid,
        M_hydro_add=A_asy,
        C_hydro_add=B_asy,
        method_label="Roger RFA + Capytaine asymptotic (hydroelastic correction)",
    )
    if getattr(config, "plot_added_mass_capytaine_vs_strip", False):
        _plot_added_mass_terms_vs_velocity(
            structural_results,
            config,
            results.velocities,
            results.frequencies,
            omega_h,
            A_h,
            capytaine_added_mass_used=A_asy,
        )
    return results


def _solve_pk_method(config, structural_results, coupling_results, aerogrid=None):

    print("Using P-K method.")
    aero_q_scaling.log_aero_im_Q_scale_once(config, context="PK method")

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

    matrix_iter_state = {}
    build_fn = aero_system.make_aero_build_matrix_fn(
        rho_f, M_hat, C_hat, K_hat, Qg_func, matrix_iter_state, config=config
    )
    on_converge_fn = aero_system.make_aero_on_converge_fn(
        rho_f, M_hat, C_hat, K_hat, Qg_func, matrix_iter_state, config=config
    )
    solver = pk_method.PkDiagnostics()
    results = pk_method.pk_sweep(
        config.V_list,
        config.modes_to_analyze,
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
                pk_solver=solver,
                dlm_participants_history=None,
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
        pk_solver=solver,  # Store solver instance for contributions plotting
        dlm_participants_history=None,
    )