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


def embed_capytaine_on_modal_dofs(A_capy, B_capy, n_modal, apply_indices, source_indices):
    """Embed Capytaine square blocks into the full Roger / PK modal basis."""
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


def capytaine_matrices_at_omega(
    omega_h,
    added_mass_h,
    added_damping_h,
    omega_query,
    n_modal,
    apply_indices,
    source_indices,
):
    """
    Interpolate Capytaine modal added mass / radiation damping at ``omega_query``
    and embed into the full modal DOF space.
    """
    # Capytaine BEM can return NaN at ω→0 (the first table sample, e.g. ω=0.1).
    # A(ω), B(ω) are smooth, so drop any non-finite frequency rows and interpolate
    # / clamp within the finite band. Without this, a low-ω query (which the MAC /
    # predict-and-select PK tracking can request) interpolates the NaN row and
    # poisons the eigensolver (scipy.eig: "array must not contain infs or NaNs").
    omega_h = np.asarray(omega_h, dtype=float)
    A_h = np.asarray(added_mass_h, dtype=float)
    B_h = np.asarray(added_damping_h, dtype=float)
    finite = (
        np.isfinite(omega_h)
        & np.isfinite(A_h.reshape(A_h.shape[0], -1)).all(axis=1)
        & np.isfinite(B_h.reshape(B_h.shape[0], -1)).all(axis=1)
    )
    if finite.any() and not finite.all():
        omega_h, A_h, B_h = omega_h[finite], A_h[finite], B_h[finite]
    A_w = _interp_modal_matrix(omega_h, A_h, omega_query)
    B_w = _interp_modal_matrix(omega_h, B_h, omega_query)
    return embed_capytaine_on_modal_dofs(
        A_w, B_w, n_modal, apply_indices, source_indices
    )


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
