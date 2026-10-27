"""
SHARPy modal-basis payload stored in Quvlm NPZ files (GOLAND parity).
"""

from __future__ import annotations

import numpy as np


def load_quvlm_npz_meta(npz_path: str) -> dict:
    d = np.load(npz_path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def _free_node_indices(n_nodes: int, constrained_dofs) -> list[int]:
    cset = set(int(d) for d in constrained_dofs)
    return [i for i in range(n_nodes) if not any((i * 6 + k) in cset for k in range(6))]


def build_sharpy_phi_on_gino(structural_results, config, n_modes: int) -> np.ndarray:
    """Map SHARPy half-wing [W, TH] onto GINO free-DOF modal matrix Φ.

    DOF layout per free node: [Ux, Uy, Uz, Rx, Ry, Rz]. Plunge→Uz, twist→Ry,
    bending slope Rx ≈ −dUz/dy. Columns are mass-normalized w.r.t. ``Mff``.
    """
    shapes_path = getattr(config, "sharpy_basis_mode_shapes_npz", None)
    if not shapes_path:
        raise ValueError(
            "DLM + quvlm_sharpy_structural_basis requires "
            "config.sharpy_basis_mode_shapes_npz (SHARPy W/TH NPZ)"
        )

    z = np.load(shapes_path)
    y_s = np.asarray(z["y"], dtype=float)
    W_s = np.asarray(z["W"], dtype=float)
    TH_s = np.asarray(z["TH"], dtype=float)
    nm = min(int(n_modes), int(W_s.shape[0]))

    beam = structural_results.beam_model
    nodes = beam["nodes"]
    free_nodes = _free_node_indices(len(nodes), structural_results.constrained_dofs)
    n_free = len(free_nodes)
    y_g = np.array([float(nodes[i]["position"][1]) for i in free_nodes], dtype=float)

    order = np.argsort(y_s)
    y_s = y_s[order]
    Mff = np.asarray(structural_results.Mff, dtype=float)
    Phi = np.zeros((6 * n_free, nm), dtype=float)

    for m in range(nm):
        Ww = W_s[m, order]
        THh = TH_s[m, order]
        uz = np.interp(y_g, y_s, Ww)
        ry = np.interp(y_g, y_s, THh)
        if structural_results.dry_eigenvectors.shape[1] > m:
            ry_old = np.asarray(structural_results.dry_eigenvectors[4::6, m], dtype=float)
            if ry_old.size == ry.size and float(np.dot(ry, ry_old)) < 0.0:
                ry = -ry
        rx = np.gradient(-uz, y_g)
        col = np.zeros(6 * n_free, dtype=float)
        col[2::6] = uz
        col[3::6] = rx
        col[4::6] = ry
        nrm = float(np.sqrt(max(float(col @ Mff @ col), 0.0)))
        if nrm < 1e-30:
            raise RuntimeError(f"SHARPy→GINO Phi column {m} has vanishing mass norm")
        Phi[:, m] = col / nrm

    tip = abs(Phi[2::6, 0][-1])
    tip_th = abs(Phi[4::6, 0][-1])
    print(
        f"[quvlm_sharpy_structural_basis] Built SHARPy Φ on GINO mesh "
        f"({nm} modes × {n_free} free nodes) from {shapes_path}"
    )
    print(
        f"[quvlm_sharpy_structural_basis] mode0 tip |Ry|/|Uz|="
        f"{tip_th / max(tip, 1e-30):.4e} (mass-normalized Φ)"
    )
    return Phi


def apply_sharpy_structural_basis(structural_results, config):
    """
    Replace GINO modal structure with a SHARPy-aligned basis.

    * UVLM: use SHARPy ``M_hat``/``K_hat`` from the Quvlm NPZ and ``Φ = I``
      (Q table already lives in that basis).
    * DLM: use the same SHARPy ``M_hat``/``K_hat``/``ω`` from the NPZ as the
      structural model; map SHARPy [W,TH] onto the GINO mesh as ``Φ`` (columns
      mass-normalized w.r.t. GINO ``Mff``) **only** for the panel-aero
      projection. A Ritz projection ``ΦᵀKΦ`` through GINO's FEA is NOT used:
      interpolated shapes are non-conforming and pick up spurious strain
      energy (dry f came out 36/23/180/144 Hz vs SHARPy 7.7/15.2/38.6/54.6).
    """
    if not bool(getattr(config, "quvlm_sharpy_structural_basis", False)):
        return structural_results

    npz_path = getattr(config, "quvlm_npz_path", None)
    if not npz_path:
        raise ValueError("quvlm_sharpy_structural_basis=True requires config.quvlm_npz_path")

    d = np.load(npz_path, allow_pickle=True)
    if not bool(d.get("sharpy_structural_basis", False)):
        raise ValueError(
            f"{npz_path} was not built with --sharpy-structural-basis; "
            "regenerate Quvlm or disable quvlm_sharpy_structural_basis."
        )

    n = int(np.asarray(d["M_hat_sharpy"]).shape[0])
    aero = str(getattr(config, "aero_method", "DLM")).upper()
    total_dof = int(structural_results.total_dof)
    constrained = np.asarray(structural_results.constrained_dofs, dtype=int)
    free = np.setdiff1d(np.arange(total_dof), constrained)
    phi_full = np.zeros((total_dof, n), dtype=float)

    if aero == "UVLM":
        m_hat = np.asarray(d["M_hat_sharpy"], dtype=float)
        k_hat = np.asarray(d["K_hat_sharpy"], dtype=float)
        omega = np.asarray(d["omega_rad_sharpy"], dtype=float).ravel()
        phi = np.eye(n)
        print(
            f"[quvlm_sharpy_structural_basis] UVLM: SHARPy M,K from NPZ "
            f"(n={n}); dry f = {np.array2string(omega / (2 * np.pi), precision=3)} Hz"
        )
    else:
        phi = build_sharpy_phi_on_gino(structural_results, config, n)
        if phi.shape != (free.size, n):
            raise ValueError(
                f"SHARPy Φ shape {phi.shape} != (n_free, n_modes)=({free.size}, {n})"
            )
        m_hat = np.asarray(d["M_hat_sharpy"], dtype=float)
        k_hat = np.asarray(d["K_hat_sharpy"], dtype=float)
        omega = np.asarray(d["omega_rad_sharpy"], dtype=float).ravel()
        phi_full[free, :] = phi
        print(
            f"[quvlm_sharpy_structural_basis] DLM: SHARPy M̂/K̂ from NPZ, "
            f"SHARPy shapes on GINO mesh for panel projection "
            f"(n={n}); dry f = {np.array2string(omega / (2 * np.pi), precision=3)} Hz"
        )

    return structural_results._replace(
        M_hat=m_hat,
        K_hat=k_hat,
        C_hat=np.zeros_like(m_hat),
        dry_eigenvalues=omega ** 2,
        dry_eigenvectors=phi,
        dry_eigenvectors_full=phi_full,
    )
