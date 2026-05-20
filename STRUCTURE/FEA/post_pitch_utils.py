"""
Post-build rigid rotation about global +X by structural pitch (deg).

Workflow: build beam, global K/M, modal data, and load aerogrid in the
reference frame (beam along +Y). Then call apply_structural_pitch_about_x
so global matrices, mode vectors, beam nodes, and/or aerogrid are congruent
with a pitch rotation about +X before aero-structural coupling.

Config flags ``pitch_rotate_beam`` and ``pitch_rotate_aerogrid`` select whether
to rotate only the beam model, only the DLM aerogrid, or both (default).
"""

from __future__ import annotations

import numpy as np


def rotate_aerogrid_about_x(aerogrid: dict, pitch_deg: float) -> dict:
    """Rotate aerogrid point/normal vectors with panelaero_utl convention (rows @ R.T)."""
    from panelaero_utl.rotate_aerogrid import _rotation

    return _rotation(aerogrid, np.deg2rad(float(pitch_deg)), axis="x")


def apply_structural_pitch_about_x(
    beam_model,
    structural_results,
    aerogrid,
    pitch_deg: float,
    *,
    rotate_beam: bool = True,
    rotate_aerogrid: bool = True,
):
    """
    Rotate beam geometry, aerogrid, and/or structural matrices/vectors by pitch about +X.

    Uses congruence K' = T^T K T, M' = T^T M T with T = I_n ⊗ T6_x so natural
    frequencies are unchanged; mode shapes are expressed in the same global
    DOF basis after the rigid rotation of the model.

    Parameters
    ----------
    rotate_beam : bool
        If True, rotate beam nodes, element K/M, and global structural matrices/modes.
    rotate_aerogrid : bool
        If True, rotate DLM panel aerogrid coordinates and normals.
    structural_results : namedtuple StructuralResults (mutable fields replaced via _replace)
    """
    pitch_deg = float(pitch_deg)
    if abs(pitch_deg) < 1e-12:
        return structural_results, beam_model, aerogrid
    if not rotate_beam and not rotate_aerogrid:
        return structural_results, beam_model, aerogrid

    sr = structural_results

    if rotate_beam:
        from FEA.fea_utl.rotate_beams_x import get_T6_x

        Tx6 = get_T6_x([pitch_deg])
        R = Tx6[:3, :3]
        n_nodes = len(beam_model["nodes"])
        total_dof = structural_results.total_dof
        if n_nodes * 6 != total_dof:
            raise ValueError("beam_model node count does not match structural_results.total_dof")

        T_full = np.kron(np.eye(n_nodes, dtype=float), Tx6)

        K_g = T_full.T @ structural_results.K_global @ T_full
        M_g = T_full.T @ structural_results.M_global @ T_full

        constrained = np.asarray(structural_results.constrained_dofs, dtype=int)
        free_dofs = np.setdiff1d(np.arange(total_dof), constrained)
        T_ff = T_full[np.ix_(free_dofs, free_dofs)]

        Mff = T_ff.T @ structural_results.Mff @ T_ff
        Kff = T_ff.T @ structural_results.Kff @ T_ff
        Cff = T_ff.T @ structural_results.Cff @ T_ff

        u_full = T_full.T @ structural_results.u_full
        ev_full = T_full.T @ structural_results.dry_eigenvectors_full
        ev_free = ev_full[free_dofs, :]

        for node in beam_model["nodes"]:
            p = np.asarray(node["position"], dtype=float)
            node["position"] = (R @ p).tolist()

        for elem in beam_model.get("elements", []):
            Ke = np.asarray(elem["stiffness"], dtype=float)
            Me = np.asarray(elem["mass"], dtype=float)
            elem["stiffness"] = Tx6.T @ Ke @ Tx6
            elem["mass"] = Tx6.T @ Me @ Tx6
            elem.pop("T6", None)
            elem.pop("beam_dir_global", None)

        sr = structural_results._replace(
            K_global=K_g,
            M_global=M_g,
            Mff=Mff,
            Kff=Kff,
            Cff=Cff,
            u_full=u_full,
            dry_eigenvectors_full=ev_full,
            dry_eigenvectors=ev_free,
        )

    if rotate_aerogrid:
        aerogrid = rotate_aerogrid_about_x(aerogrid, pitch_deg)

    return sr, beam_model, aerogrid
