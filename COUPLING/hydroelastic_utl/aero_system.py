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


def Ac(V, rho, k, Qhh, M, C, K, store_contributions=False, config=None):
    """Modal aeroelastic companion matrix (paper canonical form)."""
    from . import aero_q_scaling

    n = M.shape[0]
    I = np.eye(n)
    Z = np.zeros((n, n))

    qinf = 0.5 * rho * V**2
    qinf_I = 0.5 * rho * V / k

    Q_damp = aero_q_scaling.scale_Qhh_imag_for_damping(Qhh, config)
    K_aero = qinf * np.real(Q_damp)
    C_aero = qinf_I * np.imag(Q_damp)

    RHS_M = M
    RHS_K = K - K_aero
    RHS_C = C - C_aero

    K_struct = K
    C_struct = C

    A21 = -np.linalg.solve(RHS_M, RHS_K)
    A22 = -np.linalg.solve(RHS_M, RHS_C)

    Ac_mat = np.block([[Z, I], [A21, A22]])

    if store_contributions:
        return Ac_mat, {
            "K_struct": K_struct,
            "K_aero": K_aero,
            "C_struct": C_struct,
            "C_aero": C_aero,
            "M_struct": M,
            "M_effective": RHS_M,
            "K_effective": RHS_K,
            "C_effective": RHS_C,
        }

    return Ac_mat


def make_aero_build_matrix_fn(rho, M, C, K, Qg_func, matrix_iter_state, config=None):
    """
    Returns ``build_matrix(k, V)`` for :func:`pk_method.pk_sweep`.

    ``matrix_iter_state`` must be a dict updated by the PK engine each iteration
    with keys ``p``, ``s``, and ``j`` before calling ``build_matrix``.
    """

    def build_matrix(k, V):
        p = matrix_iter_state.get("p")
        s = matrix_iter_state["s"]
        j = matrix_iter_state["j"]
        Qhh = Qg_func(k, V, p, s, j)
        C_use = C
        if config is not None:
            from . import empirical_fluid_damping

            omega_iter = float(max(k * V, 1e-6))
            C_emp = empirical_fluid_damping.modal_empirical_C(
                config, V, M, omega_ref=omega_iter
            )
            if np.any(C_emp != 0.0):
                C_use = C + C_emp
        return Ac(V, rho, k, Qhh, M, C_use, K, store_contributions=False, config=config)

    return build_matrix


def make_aero_on_converge_fn(rho, M, C, K, Qg_func, matrix_iter_state, config=None):
    """
    Returns ``on_converge_fn(k, V, s, p_sel, u_sel) -> dict`` for
    :func:`pk_method.pk_solve_at_velocity` (final Qhh + contribution split).
    """

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

        C_use = C
        if config is not None:
            from . import empirical_fluid_damping

            omega_iter = float(max(k * V, 1e-6))
            C_emp = empirical_fluid_damping.modal_empirical_C(
                config, V, M, omega_ref=omega_iter
            )
            if np.any(C_emp != 0.0):
                C_use = C + C_emp

        Ac_result = Ac(V, rho, k, Qhh_final, M, C_use, K, store_contributions=True, config=config)
        if isinstance(Ac_result, tuple):
            _, contributions = Ac_result
        else:
            contributions = None

        out = {}
        if contributions is not None:
            out["contributions"] = contributions
        if w_modal_data is not None:
            out["w_modal_data"] = w_modal_data
        return out

    return on_converge_fn


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

    k_list, Ma_list, shape, dtype = open_qjj_index_old(out_dir_klist)

    def Qg_func(k_red, V_dlm, p_conv, s, j, store_w_modal=False, still_fluid_eigs=False):
        struct_freqs = np.sqrt(values)
        omega_dlm = k_red * V_dlm

        if k_red <= 60:
            k_dlm = k_red
            Ma_dlm = V_dlm / c_sound

            Qjj_dlm = interp_qjj_from_disk_old(out_dir_klist, k_dlm, Ma_dlm)

        else:
            raise ValueError(f"k_red={k_red:.6g} > 30, insufficient aerodynamic data")

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

    return Qg_func
