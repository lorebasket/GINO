"""
Hydroelastic (RFA-augmented) system assembly for the p-k engine.
"""

from __future__ import annotations

import numpy as np


def _interp_modal_matrix(omega_grid, matrix_grid, omega_query):
    """Linear interpolation of a modal matrix over frequency."""
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


def _build_augmented_matrix_with_hydro(M_eff, C_eff, K_eff, q_dyn, Blag, blag, V, Nm, N):
    """Build A_aug including frequency-dependent hydro added mass/damping."""
    n = Nm
    n_tot = (2 + N) * n
    A_aug = np.zeros((n_tot, n_tot), dtype=float)
    M_inv = np.linalg.inv(M_eff)

    A_aug[:n, n : 2 * n] = np.eye(n)
    A_aug[n : 2 * n, :n] = -M_inv @ K_eff
    A_aug[n : 2 * n, n : 2 * n] = -M_inv @ C_eff

    for i in range(N):
        c0 = (2 + i) * n
        A_aug[n : 2 * n, c0 : c0 + n] = M_inv @ (q_dyn * Blag[i])

    for i in range(N):
        r0 = (2 + i) * n
        A_aug[r0 : r0 + n, n : 2 * n] = np.eye(n)
        A_aug[r0 : r0 + n, r0 : r0 + n] = -V * blag[i] * np.eye(n)

    return A_aug


def make_hydro_build_matrix_fn(
    rho,
    M_base,
    C_base,
    K_eff,
    Blag,
    blag,
    omega_h,
    added_mass_h,
    added_damping_h,
    config=None,
):
    """
    Returns ``build_matrix(k, V)`` for :func:`pk_method.pk_sweep` / hydroelastic PK.

    Interpolates added mass/damping at ``omega_iter = k * V`` and assembles the
    augmented Roger state matrix for the current airspeed ``V``.
    """

    Nm = M_base.shape[0]
    N = len(Blag)

    def build_matrix(k, V):
        q_dyn = 0.5 * rho * V**2
        omega_iter = float(max(k * V, 1e-6))
        A_add = _interp_modal_matrix(omega_h, added_mass_h, omega_iter)
        B_add = _interp_modal_matrix(omega_h, added_damping_h, omega_iter)
        M_eff = M_base + A_add
        C_eff = C_base + B_add
        if config is not None:
            from . import empirical_fluid_damping

            C_emp = empirical_fluid_damping.modal_empirical_C(
                config, V, M_eff, omega_ref=omega_iter
            )
            if np.any(C_emp != 0.0):
                C_eff += C_emp
        return _build_augmented_matrix_with_hydro(
            M_eff=M_eff,
            C_eff=C_eff,
            K_eff=K_eff,
            q_dyn=q_dyn,
            Blag=Blag,
            blag=blag,
            V=V,
            Nm=Nm,
            N=N,
        )

    return build_matrix


def hydro_pk_chunk_to_flat_arrays(
    pk_chunk_result,
    modes,
    n_state,
    *,
    dimensionless_vgvf_results=False,
):
    """
    Convert a one-velocity ``pk_sweep`` result (list of one ``{'V', 'modes'}`` dict)
    into the six flat arrays historically returned by ``solve_augmented_pk_with_capytaine``.
    """
    n_modes = len(modes)
    vecs_out = np.zeros((n_state, n_modes), dtype=complex)
    g_out = np.full(n_modes, np.nan, dtype=float)
    f_out = np.full(n_modes, np.nan, dtype=float)
    eigs_out = np.full(n_modes, np.nan + 0j, dtype=complex)
    k_out = np.full(n_modes, np.nan, dtype=float)
    converged_out = np.zeros(n_modes, dtype=bool)

    if not pk_chunk_result:
        return g_out, f_out, vecs_out, eigs_out, k_out, converged_out

    entry = pk_chunk_result[0]
    modes_list = entry.get("modes", [])

    for i_mode, mode_idx in enumerate(modes):
        sel = None
        for md in modes_list:
            if md.get("mode", None) == mode_idx:
                sel = md
                break
        if sel is None and i_mode < len(modes_list):
            sel = modes_list[i_mode]

        if sel is None or not sel.get("converged", False):
            continue

        p_sel = sel.get("p", None)
        u_sel = sel.get("u", None)
        if p_sel is None or u_sel is None:
            continue
        if np.isnan(np.real(p_sel)) or np.isnan(np.imag(p_sel)):
            continue

        eigs_out[i_mode] = p_sel
        nu = min(len(u_sel), n_state)
        vecs_out[:nu, i_mode] = u_sel[:nu]
        f_out[i_mode] = float(np.abs(np.imag(p_sel)))
        if dimensionless_vgvf_results:
            if f_out[i_mode] > 1e-14:
                g_out[i_mode] = float(np.real(p_sel) / f_out[i_mode])
            else:
                g_out[i_mode] = float(np.real(p_sel))
        else:
            g_out[i_mode] = float(np.real(p_sel))
        k_out[i_mode] = float(sel.get("k", np.nan))
        converged_out[i_mode] = bool(sel.get("converged", False))

    return g_out, f_out, vecs_out, eigs_out, k_out, converged_out
