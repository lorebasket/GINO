"""
Modal M/K/C contribution extraction for SharPy p-k + Capytaine BEM runs.

Builds a ``contributions_history`` list compatible with
``COUPLING.hydroelastic_utl.post_processing.plot_stiffness_damping_contributions``.

Structural and hydro terms are taken from the beam assembly; aerodynamic
stiffness/damping diagonals are inferred from each converged coupled eigenvalue
(``K_aero = K_struct - K_tot``, ``C_aero = C_struct + C_hydro - C_tot`` with
``M_eff λ² + C_tot λ + K_tot ≈ 0``), matching the GINO sign convention.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from gino_bem.capy_bem_bridge import apply_bem_from_capy, capy_at_omega


def _modal_mats(beam_sys, nm: int, *, t_ref: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dry structural M, K, C in the first ``nm`` modal DOFs at reduced time ``t_ref``."""
    U = np.asarray(beam_sys.U, dtype=float)[:, :nm]
    ref = beam_sys.scaled_reference_matrices or {}
    K_ref = ref.get("K", beam_sys.Kstr)
    C_ref = ref.get("C", beam_sys.Cstr)
    if K_ref is None or K_ref is False:
        K_ref = beam_sys.Kstr
    if C_ref is None or C_ref is False:
        C_ref = beam_sys.Cstr if beam_sys.Cstr is not None else np.zeros_like(K_ref)
    K_modal = U.T @ (np.asarray(K_ref, dtype=float) * t_ref ** 2) @ U
    C_modal = U.T @ (np.asarray(C_ref, dtype=float) * t_ref) @ U
    M_modal = U.T @ np.asarray(beam_sys.Mstr, dtype=float) @ U
    return M_modal, K_modal, C_modal


def _slot(mat: np.ndarray, n_slots: int = 2) -> list[np.ndarray]:
    return [np.asarray(mat, dtype=float) for _ in range(n_slots)]


def extract_contributions_at_velocity(
    ls,
    capy: dict[str, Any],
    T: np.ndarray,
    V: float,
    branch_results: dict[int, dict[str, Any]],
    *,
    nm: int,
    track_modes: tuple[int, ...] = (0, 1),
    bem: bool = True,
    A_offset: np.ndarray | None = None,
) -> dict[str, Any] | None:
    """One velocity sample; returns a GINO-style contribution entry or None."""
    converged = {j: branch_results[j] for j in track_modes
               if j in branch_results and branch_results[j].get("converged")}
    if not converged:
        return None

    ls.update(float(V))
    beam = ls.beam.sys
    t_ref = float(ls.uvlm.sys.ScalingFacts["length"]) / float(V)
    M_struct, K_struct, C_struct = _modal_mats(beam, nm, t_ref=t_ref)

    omegas = [float(converged[j]["omega"]) for j in converged]
    omega_h = float(np.mean(omegas))

    M_hydro = np.zeros((nm, nm), dtype=float)
    C_hydro = np.zeros((nm, nm), dtype=float)
    if bem:
        apply_bem_from_capy(
            ls, capy, omega_h, n_modes=nm, T=T, t_ref=t_ref, A_offset=A_offset,
        )
        A_h, B_h = capy_at_omega(capy, omega_h, mode_slice=slice(0, nm))
        Tn = np.asarray(T, dtype=float)[:nm, :nm]
        A_h = Tn.T @ A_h @ Tn
        B_h = Tn.T @ B_h @ Tn
        M_hydro = np.asarray(A_h, dtype=float)
        C_hydro = float(t_ref) * np.asarray(B_h, dtype=float)

    M_eff = M_struct + M_hydro
    K_aero = np.zeros((nm, nm), dtype=float)
    C_aero = np.zeros((nm, nm), dtype=float)

    for j, res in converged.items():
        omega = float(res["omega"])
        if omega < 1e-6:
            continue
        sigma = float(res.get("g", 0.0)) * omega
        jj = int(j)
        m = float(M_eff[jj, jj])
        K_tot = -m * (sigma ** 2 - omega ** 2)
        C_tot = -2.0 * m * sigma
        K_aero[jj, jj] = float(K_struct[jj, jj] - K_tot)
        C_aero[jj, jj] = float(C_struct[jj, jj] + C_hydro[jj, jj] - C_tot)

    K_eff = K_struct - K_aero
    C_eff = C_struct - C_aero + C_hydro
    M_eff_mat = M_struct + M_hydro

    return {
        "V": float(V),
        "branch_dof_indices": list(track_modes),
        "M_struct": _slot(M_struct),
        "M_hydro": _slot(M_hydro),
        "M_effective": _slot(M_eff_mat),
        "K_struct": _slot(K_struct),
        "K_aero": _slot(K_aero),
        "K_hydro": _slot(np.zeros((nm, nm), dtype=float)),
        "K_effective": _slot(K_eff),
        "C_struct": _slot(C_struct),
        "C_aero": _slot(C_aero),
        "C_hydro": _slot(C_hydro),
        "C_empirical": _slot(np.zeros((nm, nm), dtype=float)),
        "C_effective": _slot(C_eff),
        "omega_rad_s": omega_h,
    }
