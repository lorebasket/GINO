"""
Aeroelastic state-space assembly and Qg factory for the p-k engine.
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import norm


def check_finite(array, name):
    if np.any(np.isnan(array)):
        raise ValueError(f"NaN detected in {name}")
    if np.any(np.isinf(array)):
        raise ValueError(f"Inf detected in {name}")


def modal_aero_mck(V, rho, k, Qhh, M, C, K, config=None, *, store_contributions=False):
    """
    Effective modal (M, C, K) for aeroelastic PK at reduced frequency ``k``.

    Returns ``(M_eff, C_eff, K_eff)`` or, when ``store_contributions=True``, a dict
    of split terms alongside the triple.
    """
    from . import aero_q_scaling

    k_pa = max(float(k), 1e-12)
    qinf = 0.5 * float(rho) * float(V) ** 2

    Q_damp = aero_q_scaling.scale_Qhh_imag_for_damping(Qhh, config)
    K_aero = qinf * np.real(Q_damp)

    conv = (
        str(getattr(config, "rfa_roger_convention", "panelaero_legacy"))
        if config is not None
        else "panelaero_legacy"
    )
    if conv == "paper_zurich":
        chord = float(getattr(config, "chord"))
        ref = getattr(config, "rfa_blag_ref_length", "semichord")
        if isinstance(ref, str):
            ref_key = ref.lower()
            if ref_key in ("semichord", "half_chord", "c/2"):
                b = 0.5 * chord
            elif ref_key in ("chord", "c"):
                b = chord
            else:
                raise ValueError(
                    "config.rfa_blag_ref_length must be 'semichord', 'chord', or a positive length"
                )
        else:
            b = float(ref)
            if b <= 0.0:
                raise ValueError(f"config.rfa_blag_ref_length must be positive, got {b}")
        k_star = k_pa * b
        C_aero = 0.5 * float(rho) * b * float(V) / max(k_star, 1e-12) * np.imag(Q_damp)
    else:
        C_aero = 0.5 * float(rho) * float(V) / k_pa * np.imag(Q_damp)

    m_eff = M
    k_eff = K - K_aero
    c_eff = C - C_aero
    # Smorzamento strutturale ISTERETICO (opzionale): K (1 + i eta) uguale su tutti i modi,
    # nel p-k come viscoso equivalente eta K / omega alla omega corrente (omega = k V).
    eta_s = float(getattr(config, "structural_eta", 0.0) or 0.0) if config is not None else 0.0
    if eta_s and k * V > 0.0:
        c_eff = c_eff + eta_s * K / (k * V)

    if not store_contributions:
        return m_eff, c_eff, k_eff

    return m_eff, c_eff, k_eff, {
        "K_struct": K,
        "K_aero": K_aero,
        "C_struct": C,
        "C_aero": C_aero,
        "M_struct": M,
        "M_effective": m_eff,
        "K_effective": k_eff,
        "C_effective": c_eff,
    }


def Ac(V, rho, k, Qhh, M, C, K, store_contributions=False, config=None):
    """
    Modal aeroelastic companion matrix.

    ``k`` is PanelAero ``k_pa = ω/V`` (Qjj disk index). ``config.rfa_roger_convention``:
    ``paper_zurich`` → ``C_aero = ½ρbV/k*·Im(Q)`` with ``k* = k_pa·b``;
    ``panelaero_legacy`` → ``C_aero = ½ρV/k_pa·Im(Q)``.
    """
    n = M.shape[0]
    I = np.eye(n)
    Z = np.zeros((n, n))

    mck = modal_aero_mck(V, rho, k, Qhh, M, C, K, config=config, store_contributions=store_contributions)
    if store_contributions:
        m_eff, c_eff, k_eff, contributions = mck
    else:
        m_eff, c_eff, k_eff = mck

    RHS_M = m_eff
    RHS_K = k_eff
    RHS_C = c_eff

    # #region agent log
    try:
        import json as _json, time as _time
        Ac._dbgn = getattr(Ac, "_dbgn", 0)
        if Ac._dbgn < 8:
            Ac._dbgn += 1
            with open("/media/lorenzo/Seagate Por1/recupero dati Linux/lorebasket/FSI/.cursor/debug-a75c44.log", "a") as _f:
                _f.write(_json.dumps({"sessionId": "a75c44", "runId": "run2", "hypothesisId": "E,H",
                    "location": "aero_system.py:Ac", "timestamp": int(_time.time() * 1000),
                    "message": "pk damping assembly: C_eff = C - C_aero (GINO_NEW / Pietro pairing)",
                    "data": {"V": float(V), "k_pa": float(k),
                        "C_eff_diag": [float(c_eff[_j, _j]) for _j in range(min(2, n))]}}) + "\n")
    except Exception:
        pass
    # #endregion

    A21 = -np.linalg.solve(RHS_M, RHS_K)
    A22 = -np.linalg.solve(RHS_M, RHS_C)

    Ac_mat = np.block([[Z, I], [A21, A22]])

    if store_contributions:
        return Ac_mat, contributions

    return Ac_mat


def Ad_newmark(
    V,
    rho,
    k,
    Qhh,
    M,
    C,
    K,
    dt,
    newmark_damp,
    config=None,
    *,
    structural_C=None,
):
    """
    Newmark-beta discrete state matrix for modal aeroelastic PK (SHARPy parity).

    ``structural_C`` defaults to zero (algorithmic damping only in ``newmark_damp``).
    Aero damping enters via ``C_aero`` in the effective C passed to ``newmark_ss``.
    """
    from .newmark_discrete import newmark_discrete_ss

    c_struct = (
        np.zeros_like(M, dtype=float)
        if structural_C is None
        else np.asarray(structural_C, dtype=float)
    )
    m_eff, c_eff, k_eff = modal_aero_mck(V, rho, k, Qhh, M, c_struct, K, config=config)
    return newmark_discrete_ss(
        m_eff,
        c_eff,
        k_eff,
        float(dt),
        float(newmark_damp),
    )


def _pk_capytaine_mass_damping(capytaine_hydro, omega_iter):
    """Frequency-interpolated Capytaine A(ω), B(ω) embedded on modal DOFs.

    When ``pk_bem_mode='delta'``, subtract the precomputed UVLM apparent-mass
    offset so ``M_hydro = A_capy(ω) - A_uvlm`` (SHARPy delta injection parity).
    """
    from . import hydro_system

    M_emb, C_emb = hydro_system.capytaine_matrices_at_omega(
        capytaine_hydro["omega_h"],
        capytaine_hydro["added_mass_h"],
        capytaine_hydro["added_damping_h"],
        omega_iter,
        capytaine_hydro["n_modal"],
        capytaine_hydro["apply_indices"],
        capytaine_hydro["source_indices"],
    )
    A_off = capytaine_hydro.get("A_uvlm_offset")
    if A_off is not None:
        M_emb = M_emb - np.asarray(A_off, dtype=float)
    return M_emb, C_emb


def make_aero_build_matrix_fn(rho, M, C, K, Qg_func, matrix_iter_state, config=None,
                              qmodal_history=None, capytaine_hydro=None,
                              *, pk_discrete_newmark=False, newmark_dt=None, newmark_damp=None):
    """
    Returns ``build_matrix(k, V)`` for :func:`pk_method.pk_sweep`.

    ``matrix_iter_state`` must be a dict updated by the PK engine each iteration
    with keys ``p``, ``s``, and ``j`` before calling ``build_matrix``.

    When ``pk_discrete_newmark=True``, returns the Newmark discrete ``Ass`` matrix
    instead of the continuous companion ``Ac``; the PK engine must set
    ``pk_discrete_dt`` to map eigenvalues via ``s = log(mu)/dt``.
    """
    from . import aero_q_scaling

    if pk_discrete_newmark:
        dt_nm = float(newmark_dt if newmark_dt is not None else getattr(config, "newmark_dt", 0.1143))
        nd_nm = float(newmark_damp if newmark_damp is not None else getattr(config, "newmark_damp", 1e-4))
        if dt_nm <= 0.0:
            raise ValueError(f"newmark_dt must be positive for discrete PK, got {dt_nm}")

    def build_matrix(k, V):
        p = matrix_iter_state.get("p")
        s = matrix_iter_state["s"]
        j = matrix_iter_state["j"]
        Qhh = Qg_func(k, V, p, s, j)
        if qmodal_history is not None:
            Q_damp = aero_q_scaling.scale_Qhh_imag_for_damping(
                np.asarray(Qhh, dtype=complex), config
            )
            qmodal_history.append(
                {
                    "V": float(V),
                    "k": float(k),
                    "mode": int(s),
                    "iteration": int(j),
                    "Q_modal": np.array(Qhh, dtype=complex),
                    "Q_damp": np.array(Q_damp, dtype=complex),
                    "converged": False,
                }
            )
        omega_iter = float(max(k * V, 1e-6))
        M_use = M
        C_use = C
        if capytaine_hydro is not None:
            M_hydro, C_hydro = _pk_capytaine_mass_damping(capytaine_hydro, omega_iter)
            M_use = M + M_hydro
            C_use = C + C_hydro
        if config is not None:
            from . import empirical_fluid_damping

            C_emp = empirical_fluid_damping.modal_empirical_C(
                config, V, M_use, omega_ref=omega_iter
            )
            if np.any(C_emp != 0.0):
                C_use = C_use + C_emp
        if pk_discrete_newmark:
            return Ad_newmark(
                V, rho, k, Qhh, M_use, C_use, K,
                dt_nm, nd_nm, config=config, structural_C=None,
            )
        return Ac(V, rho, k, Qhh, M_use, C_use, K, store_contributions=False, config=config)

    return build_matrix


def make_aero_on_converge_fn(rho, M, C, K, Qg_func, matrix_iter_state, config=None,
                             qmodal_history=None, capytaine_hydro=None):
    """
    Returns ``on_converge_fn(k, V, s, p_sel, u_sel) -> dict`` for
    :func:`pk_method.pk_solve_at_velocity` (final Qhh + contribution split).
    """
    from . import aero_q_scaling

    def on_converge_fn(k, V, s, p_sel, u_sel):
        matrix_iter_state["p"] = p_sel
        matrix_iter_state["s"] = s
        j = matrix_iter_state.get("j", 0)
        Qg_result = Qg_func(k, V, p_sel, s, j, store_w_modal=True)
        if isinstance(Qg_result, tuple):
            Qhh_final, w_modal_data = Qg_result
        else:
            Qhh_final = Qg_result
            w_modal_data = None

        if qmodal_history is not None:
            Q_damp_final = aero_q_scaling.scale_Qhh_imag_for_damping(
                np.asarray(Qhh_final, dtype=complex), config
            )
            qmodal_history.append(
                {
                    "V": float(V),
                    "k": float(k),
                    "mode": int(s),
                    "iteration": int(j),
                    "Q_modal": np.array(Qhh_final, dtype=complex),
                    "Q_damp": np.array(Q_damp_final, dtype=complex),
                    "converged": True,
                }
            )

        omega_iter = float(max(k * V, 1e-6))
        M_hydro = np.zeros_like(M, dtype=float)
        C_hydro = np.zeros_like(C, dtype=float)
        M_use = M
        C_use = C
        if capytaine_hydro is not None:
            M_hydro, C_hydro = _pk_capytaine_mass_damping(capytaine_hydro, omega_iter)
            M_use = M + M_hydro
            C_use = C + C_hydro

        C_emp_split = np.zeros_like(C, dtype=float)
        if config is not None:
            from . import empirical_fluid_damping

            C_emp_split = empirical_fluid_damping.modal_empirical_C(
                config, V, M_use, omega_ref=omega_iter
            )
            if np.any(C_emp_split != 0.0):
                C_use = C_use + C_emp_split

        Ac_result = Ac(V, rho, k, Qhh_final, M_use, C_use, K, store_contributions=True, config=config)
        if isinstance(Ac_result, tuple):
            _, contributions = Ac_result
        else:
            contributions = None

        out = {}
        if contributions is not None:
            contributions = dict(contributions)
            contributions["M_struct"] = np.asarray(M, dtype=float)
            contributions["C_struct"] = np.asarray(C, dtype=float)
            contributions["C_empirical"] = np.asarray(C_emp_split, dtype=float)
            if capytaine_hydro is not None:
                contributions["M_hydro"] = np.asarray(M_hydro, dtype=float)
                contributions["C_hydro"] = np.asarray(C_hydro, dtype=float)
            out["contributions"] = contributions
        if w_modal_data is not None:
            out["w_modal_data"] = w_modal_data
        return out

    return on_converge_fn


def make_Qg_func_uvlm(config):
    """Modal-GAF provider from a precomputed SHARPy linear-UVLM table (aero_method='UVLM').

    Loads ``config.quvlm_npz_path`` (written by ``FLUID/sharpy/gino_bem/precompute_quvlm.py``):
    ``Q(k_pa)`` on a dense ``k_list`` grid, already in the GINO/Capytaine modal basis
    and normalized so ``F_modal = qinf * Q * q`` — the same convention as the DLM
    ``_Qg_raw``. Drop-in for ``make_Qg_func``: same signature, same post-processing
    (aero_b2_removal, sign flips), same clip-beyond-kmax behavior. The UVLM GAF is a
    pure function of k, so the DLM's ``p_conv``/``s``/``j`` downwash-seeding subtleties
    do not apply (they only pick the evaluation frequency, which the caller already
    encodes in ``k_red``).

    Debug levers: ``config.quvlm_scale`` (scalar on Q, default 1) and
    ``config.quvlm_mode_signs`` (per-mode ±1 congruence, default None) for
    normalization / mode-sign calibration against the DLM path.
    """
    from . import aero_q_scaling

    npz_path = getattr(config, "quvlm_npz_path", None)
    if not npz_path:
        raise ValueError("aero_method='UVLM' requires config.quvlm_npz_path "
                         "(see FLUID/sharpy/gino_bem/precompute_quvlm.py)")
    d = np.load(npz_path, allow_pickle=True)
    k_list = np.asarray(d["k_list"], dtype=float).ravel()
    Q_tbl = np.asarray(d["Q"])                     # (nk, nm, nm) complex
    nm = Q_tbl.shape[1]
    n_expected = int(getattr(config, "num_modes_egv", nm))
    _pk_idx = getattr(config, "pk_modal_dof_indices", None)
    if _pk_idx is not None:
        # La tabella puo' essere costruita direttamente sul sottoinsieme di modi
        # che il PK risolve (e.g. AC38: i modi aerodinamicamente attivi, senza
        # quelli di flessione nel piano della corda). In quel caso il confronto
        # va fatto con la fetta, non con la base modale intera.
        n_expected = len(list(_pk_idx))
    if nm != n_expected:
        raise ValueError(f"UVLM GAF table has {nm} modes but {n_expected} expected "
                         f"(num_modes_egv / pk_modal_dof_indices); "
                         f"regenerate {npz_path} with --num-modes {n_expected}")

    scale = float(getattr(config, "quvlm_scale", 1.0))
    signs = getattr(config, "quvlm_mode_signs", None)
    if signs is not None:
        S = np.diag(np.asarray(signs, dtype=float))
        Q_tbl = np.einsum("ij,kjl,lm->kim", S, Q_tbl, S)
    if scale != 1.0:
        Q_tbl = scale * Q_tbl
    print(f"[aero_method=UVLM] modal GAF table {npz_path}: {len(k_list)} k-points, "
          f"k in [{k_list[0]:g}, {k_list[-1]:g}] 1/m, nm={nm}, grid={d.get('grid', '?')}, "
          f"scale={scale}" + (f", mode_signs={list(signs)}" if signs is not None else ""))

    _kmax_tbl = float(k_list[-1])
    if config is not None:                # per il controllo di copertura a fine p-k
        config._quvlm_kmax_table = _kmax_tbl

    # Comportamento oltre k_max. Congelare Q (il default storico) e' giusto solo
    # se oltre k_max non resta nulla che dipenda da k. Non e' questo il caso: la
    # massa apparente vive nel termine Re(Q) ~ C k^2, e il p-k valuta k = omega/V,
    # che a bassa velocita' esce dalla tabella. Congelandola, l'aero di un modo a
    # 10 Hz veniva valutata come se oscillasse a 24 Hz: K_eff = K - qinf Re(Q)
    # cambiava segno e il modo compariva come radice REALE POSITIVA (divergenza
    # apparente a 2 nodi), rubando lo slot al ramo vero.
    # Con quvlm_k_extrapolate si prolunga invece l'asintoto fisico, raccordato in
    # k_max per continuita':  Re(Q) = C k^2  (massa apparente costante),
    # Im(Q) = D k  (smorzamento aerodinamico costante, C_aero = 0.5 rho V D).
    _extrap = bool(getattr(config, "quvlm_k_extrapolate", False))
    _C_hi = np.real(Q_tbl[-1]) / _kmax_tbl ** 2
    _D_hi = np.imag(Q_tbl[-1]) / _kmax_tbl
    if _extrap:
        print(f"[aero_method=UVLM] k > {_kmax_tbl:g}: asintoto Re ~ C k^2, Im ~ D k "
              f"(diag C = {np.array2string(np.diag(_C_hi), precision=5)})")

    def _interp_Q(k_red):
        kq = float(k_red)
        if _extrap and kq > _kmax_tbl:
            return _C_hi * kq ** 2 + 1j * _D_hi * kq
        kc = min(max(kq, k_list[0]), _kmax_tbl)
        i = int(np.searchsorted(k_list, kc))
        if i <= 0:
            return Q_tbl[0]
        if i >= len(k_list):
            return Q_tbl[-1]
        f = (kc - k_list[i - 1]) / max(k_list[i] - k_list[i - 1], 1e-30)
        return (1.0 - f) * Q_tbl[i - 1] + f * Q_tbl[i]

    # aero_b2_removal, mirroring the DLM path: fit the UVLM's own apparent mass
    # from the high-k asymptote Re(Q) -> -k^2*B2 and cancel it out of Re(Q) so the
    # explicit added mass (Capytaine / strip) is the sole apparent mass.
    # pk_bem_mode='delta' keeps the UVLM GAF intact (offset subtracted from A_capy).
    _pk_bem = str(getattr(config, "pk_bem_mode", "full")).lower()
    if _pk_bem == "delta":
        _b2rem = False
    else:
        _b2rem = bool(getattr(config, "aero_b2_removal", False))
    _B2 = None
    if _b2rem:
        _kb2 = min(float(getattr(config, "quvlm_k_b2", 34.0)), _kmax_tbl)
        _B2 = -np.real(_interp_Q(_kb2)) / (_kb2 ** 2)
        _B2 = 0.5 * (_B2 + _B2.T)
        _has_ext_am = (
            bool(getattr(config, "added_mass_strip_theory", False))
            or bool(getattr(config, "pk_capytaine_matrices", False))
        )
        print(f"[aero_b2_removal] UVLM apparent mass removed from Re(Q) "
              f"(k_b2={_kb2:.3f}); B2 diagonal: {np.diag(_B2)}")
        if not _has_ext_am:
            print("[aero_b2_removal WARN] no explicit added mass detected "
                  "(added_mass_strip_theory / pk_capytaine_matrices both False): "
                  "removing B2 without replacing it leaves ~zero apparent mass -- "
                  "wet frequencies will drift toward the dry values.")

    def Qg_func(k_red, V_dlm, p_conv, s, j, store_w_modal=False, still_fluid_eigs=False):
        Q = _interp_Q(k_red)
        if _b2rem:
            # oltre k_max, se Q e' prolungata con C k^2, va tolta anche li'
            kc = float(k_red) if _extrap else min(float(k_red), _kmax_tbl)
            Q = Q + (kc ** 2) * _B2
        Qf = aero_q_scaling.apply_aero_q_sign_flips(Q, config)
        check_finite(Qf, "Q_modal_uvlm")
        if store_w_modal:
            return Qf, {"w_velocity_norm": np.nan, "w_slope_norm": np.nan,
                        "Z_force_modal_norm": np.nan}
        return Qf

    return Qg_func


def make_Qg_func(
    config,
    FSI_path,
    Z,
    Apan,
    Z_qs,
    Phi,
    values,
    c_sound,
    out_dir_klist,
    out_dir_vjj,
    alpha_r,
    Z_force=None,
    panel_areas=None,
):
    from Qjj.precompute_qjj import open_qjj_index_old, interp_qjj_from_disk_old
    from Qjj.precompute_qjj_vlm import open_qjj_index_vlm, interp_qjj_vlm_from_disk
    from . import aero_q_scaling

    k_list, Ma_list, shape, dtype = open_qjj_index_old(out_dir_klist)

    def _Qg_raw(k_red, V_dlm, p_conv, s, j, store_w_modal=False, still_fluid_eigs=False):
        struct_freqs = np.sqrt(values)
        omega_dlm = k_red * V_dlm

        # [GINO_NEW FIX] CLIP the reduced frequency to the table max instead of raising. At low V the
        # reduced freq k=omega/V explodes (~100s); beyond kmax the flow is pure non-circulatory
        # apparent mass (~k-independent), so freezing the aero at kmax is physical and avoids the crash.
        _kmax_tbl = float(k_list[-1])
        k_dlm = min(k_red, _kmax_tbl)
        Ma_dlm = V_dlm / c_sound

        Qjj_dlm = interp_qjj_from_disk_old(
            out_dir_klist,
            k_dlm,
            Ma_dlm,
            ma_interp_mode=getattr(config, "qjj_ma_interp_mode", "auto"),
            ma_low_threshold=float(getattr(config, "qjj_ma_low_threshold", 0.05)),
        )

        check_finite(Qjj_dlm, "Qjj_dlm")

        if j == 0 and s == 0:
            omega = k_red * V_dlm
            p_over_V = 1j * omega / V_dlm
            p = 1j * omega
        elif j == 0 and s > 0:
            # Yuan & Zhang (2023): evaluate Qhh at this mode's structural frequency on
            # the first PK iteration — not at the previously converged mode's p.
            p_over_V = 1j * struct_freqs[s] / V_dlm
            p = 1j * struct_freqs[s]
            print(f"using p = {p:.6g} (structural ω_n[{s}] for mode-isolated j=0)")
        elif still_fluid_eigs:
            p_over_V = 1j * struct_freqs[s] / V_dlm
            p = 1j * struct_freqs[s]
            print(f"using p = {p:.6g} (from still fluid eigenvalues)")

        else:
            if p_conv is not None:
                p_over_V = 1j * np.imag(p_conv) / V_dlm
                p = 1j * np.imag(p_conv)
                print(f"using p = {p:.6g} (from wet frequency convergence)")
            else:
                p_over_V = 1j * struct_freqs[s] / V_dlm
                p = 1j * struct_freqs[s]
                print(f"using p = {p:.6g} (from structural frequency)")

        # [GINO_NEW FIX] also clip the unsteady-downwash reduced freq to kmax (keep aero CONSTANT
        # beyond the table), consistent with the Qjj clip above, so nothing blows up at low V.
        if k_red > _kmax_tbl and abs(p_over_V) > 0:
            p_over_V = p_over_V * (_kmax_tbl / k_red)
        jk_over_b = 1j * k_red
        w_velocity_modal = p_over_V * Z @ Phi

        w_slope_modal = Z_qs @ Phi

        w_aero_velocity = (Qjj_dlm) @ w_velocity_modal
        w_aero_slope = (Qjj_dlm) @ w_slope_modal

        w_modal = w_aero_velocity + w_aero_slope

        Z_force_modal = Z_force @ Phi

        A_diag = np.diag(panel_areas)

        Q_modal = Z_force_modal.T @ A_diag @ w_modal

        check_finite(Q_modal, "Q_modal")

        if store_w_modal:
            return Q_modal, {
                "w_velocity_norm": norm(w_aero_velocity),
                "w_slope_norm": norm(w_aero_slope),
                "Z_force_modal_norm": norm(Z_force_modal),
            }

        return Q_modal

    # [aero_b2_removal] Optional removal of the DLM's own apparent (non-circulatory) mass from the
    # P-K generalized aero force, mirroring the GINO_NEW hydroelastic correction. At high reduced
    # frequency the raw GAF behaves as Re(Q) -> -k^2 * B2, i.e. the DLM apparent mass. When the
    # added mass is instead supplied EXPLICITLY in M_hat (strip theory or Capytaine), that k^2 term
    # must be cancelled out of Re(Q) to avoid double-counting. B2 = -Re(Q(k_b2)) / k_b2^2 is taken
    # once from a single high-k sample (clipped to the table max) and symmetrised; Qg_func then adds
    # +k^2 * B2 back to Re(Q) at every evaluation (k clipped to kmax, aero frozen beyond).
    _b2rem = bool(getattr(config, "aero_b2_removal", False))
    _kmax_b2 = float(k_list[-1])
    _B2 = None
    if _b2rem:
        _kb2 = min(34.0, _kmax_b2)
        _Qb2 = _Qg_raw(_kb2, 1.0, None, 0, 0)
        _B2 = -np.real(_Qb2) / (_kb2 ** 2)
        _B2 = 0.5 * (_B2 + _B2.T)
        _has_ext_am = (
            bool(getattr(config, "added_mass_strip_theory", False))
            or bool(getattr(config, "pk_capytaine_matrices", False))
        )
        print(
            f"[aero_b2_removal] DLM apparent mass removed from Re(Q) "
            f"(k_b2={_kb2:.3f}); B2 diagonal: {np.diag(_B2)}"
        )
        if not _has_ext_am:
            print(
                "[aero_b2_removal WARN] no explicit added mass detected "
                "(added_mass_strip_theory / pk_capytaine_matrices both False): "
                "removing B2 without replacing it leaves ~zero apparent mass -- "
                "wet frequencies will drift toward the dry values."
            )

    # Post-process the raw modal GAF for the P-K path:
    #   (0) aero_b2_removal: add +k^2*B2 to Re(Q) (cancel DLM apparent mass; before sign flips)
    #   (1) aero_real_flip: negate Re(Q) (stiffness / K_aero sign)
    #   (2) aero_imag_flip: negate Im(Q) (damping / C_aero sign)
    def Qg_func(k_red, V_dlm, p_conv, s, j, store_w_modal=False, still_fluid_eigs=False):
        r = _Qg_raw(k_red, V_dlm, p_conv, s, j, store_w_modal=store_w_modal, still_fluid_eigs=still_fluid_eigs)
        Q = r[0] if isinstance(r, tuple) else r
        if _b2rem:
            kc = min(float(k_red), _kmax_b2)
            Q = Q + (kc ** 2) * _B2
        Qf = aero_q_scaling.apply_aero_q_sign_flips(Q, config)
        if isinstance(r, tuple):
            return (Qf,) + tuple(r[1:])
        return Qf

    return Qg_func
