"""
Build interpolated modal face-displacement fields for Capytaine radiation studies.

Blends structural mode shapes (beam eigenvectors mapped to hull faces) toward a
rigid-motion target:

  - ``rigid_body`` (default): at α = 1 each mode is a *single* Capytaine rigid-body unit
    field (Surge … Yaw) chosen by MAC vs the structural mode; optional peak-amplitude match.
  - ``span_rigidify``: single (u, θ) at a reference span station applied to all faces.
  - ``rigid6_projection``: least-squares blend of all six rigid unit fields (legacy).

At α = 0 the fields match the standard ``run_modal_radiation`` beam map.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from run_modal_radiation import (  # noqa: E402  (same package directory)
    _RIGID_BODY_DOF_NAMES,
    _beam_to_global_rotation_from_cfg,
    _build_face_displacements_from_beam,
    _build_floating_body,
    _constrained_global_dofs,
    _load_beam_positions,
    _load_phi_full_row,
    _mesh_area_centroid,
    _mode_indices_from_config,
    _phi_normalization_scale,
    _resolve_mesh_loader,
    _resolve_modal_span_axis,
    _rotate_nodal_phi_to_hydro_frame,
    _span_interp,
)

TargetKind = Literal["rigid_body", "span_rigidify", "rigid6_projection"]


def _beam_setup(cfg, modal_dir: str, prefix: str, *, align_beam_to_hydro_mesh: bool = True):
    R = _beam_to_global_rotation_from_cfg(cfg)
    nodes_csv = f"{modal_dir}/{prefix}_nodes.csv"
    constrained_csv = f"{modal_dir}/{prefix}_constrained_dofs.csv"
    beam_pos_raw = _load_beam_positions(nodes_csv)
    constrained = _constrained_global_dofs(constrained_csv)
    n_nodes = int(beam_pos_raw.shape[0])

    if align_beam_to_hydro_mesh and not np.allclose(R, np.eye(3), atol=1e-12, rtol=0.0):
        beam_pos = np.einsum("ij,nj->ni", R, beam_pos_raw)
    else:
        beam_pos = np.asarray(beam_pos_raw, dtype=float).copy()
    return R, beam_pos, constrained, n_nodes, nodes_csv, constrained_csv


def rigid_body_unit_face_displacements(
    cpt,
    mesh,
    *,
    rotation_center: np.ndarray | None = None,
) -> tuple[list[str], np.ndarray]:
    """
    Return Capytaine rigid-body unit face displacement fields.

    Returns
    -------
    names : list of 6 DOF labels (Surge … Yaw).
    disp : (6, n_faces, 3) array [m] per unit generalized coordinate.
    """
    body = _build_floating_body(cpt, mesh, name="rigid_probe")
    if rotation_center is not None:
        rc = np.asarray(rotation_center, dtype=float).ravel()
        if rc.size != 3:
            raise ValueError("rotation_center must have length 3.")
        body.center_of_mass = rc
    elif getattr(body, "center_of_mass", None) is None:
        body.center_of_mass = _mesh_area_centroid(mesh)

    body.add_all_rigid_body_dofs()
    names = list(_RIGID_BODY_DOF_NAMES)
    missing = [n for n in names if n not in body.dofs]
    if missing:
        raise RuntimeError(f"Rigid-body DOF setup incomplete; missing: {missing}")
    disp = np.stack([np.asarray(body.dofs[n], dtype=float) for n in names], axis=0)
    return names, disp


def _rigidify_span_displacement(
    face_centers: np.ndarray,
    beam_pos: np.ndarray,
    phi_full: np.ndarray,
    span_axis: int,
    *,
    ref_frac: float = 0.5,
) -> np.ndarray:
    """Apply a single (u, θ) taken at ``ref_frac`` along the beam span to all faces."""
    y_nodes = beam_pos[:, span_axis]
    y_min = float(y_nodes.min())
    y_max = float(y_nodes.max())
    y_ref = y_min + float(ref_frac) * (y_max - y_min)
    i_lo, i_hi, n1, n2 = _span_interp(y_ref, beam_pos, span_axis)
    q_lo = phi_full[i_lo * 6 : i_lo * 6 + 6]
    q_hi = phi_full[i_hi * 6 : i_hi * 6 + 6]
    u = n1 * q_lo[:3] + n2 * q_hi[:3]
    th = n1 * q_lo[3:6] + n2 * q_hi[3:6]
    x_ref = n1 * beam_pos[i_lo] + n2 * beam_pos[i_hi]
    return u[None, :] + np.cross(th, face_centers - x_ref[None, :])


def _mac_squared(flat_a: np.ndarray, flat_b: np.ndarray) -> float:
    na = float(np.dot(flat_a, flat_a))
    nb = float(np.dot(flat_b, flat_b))
    if na < 1e-30 or nb < 1e-30:
        return 0.0
    return float(np.dot(flat_a, flat_b) ** 2 / (na * nb))


def _peak_face_displacement_norm(d_faces: np.ndarray) -> float:
    return float(np.max(np.linalg.norm(np.asarray(d_faces, dtype=float), axis=1)))


def _rigid_body_target_for_mode(
    d_struct_mode: np.ndarray,
    rigid6: np.ndarray,
    rigid_names: list[str],
    *,
    match_peak_amplitude: bool = True,
) -> tuple[np.ndarray, str, float, int]:
    """
    Pick one Capytaine rigid unit field (whole-body motion) best aligned with a structural mode.

    Returns (d_target, rigid_dof_name, mac_squared, rigid_index).
    """
    v_s = np.asarray(d_struct_mode, dtype=float).reshape(-1)
    best_j = 0
    best_mac = -1.0
    for j in range(rigid6.shape[0]):
        mac = _mac_squared(v_s, rigid6[j].reshape(-1))
        if mac > best_mac:
            best_mac = mac
            best_j = j
    d_tgt = np.asarray(rigid6[best_j], dtype=float).copy()
    if match_peak_amplitude:
        peak_s = _peak_face_displacement_norm(d_struct_mode)
        peak_r = _peak_face_displacement_norm(d_tgt)
        if peak_r > 1e-30:
            d_tgt *= peak_s / peak_r
    return d_tgt, str(rigid_names[best_j]), best_mac, int(best_j)


def _project_onto_rigid6(d_mode: np.ndarray, rigid6: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Least-squares fit of one modal face field onto six unit rigid-body patterns.

    Returns (d_target, coeffs) with d_target shape (n_faces, 3), coeffs shape (6,).
    """
    n_faces = d_mode.shape[0]
    if rigid6.shape[1] != n_faces:
        raise ValueError("rigid6 / d_mode face count mismatch.")
    D = rigid6.transpose(1, 0, 2).reshape(-1, 6)
    d_flat = np.asarray(d_mode, dtype=float).reshape(-1)
    coeffs, _, _, _ = np.linalg.lstsq(D, d_flat, rcond=None)
    d_target = (D @ coeffs).reshape(n_faces, 3)
    return d_target, coeffs


def build_structural_and_target_face_displacements(
    *,
    cfg,
    face_centers: np.ndarray,
    modal_dir: str,
    prefix: str,
    mode_indices: list[int],
    span_axis: int | None,
    phi_normalization: str,
    target: TargetKind = "rigid_body",
    ref_frac: float = 0.5,
    match_peak_amplitude: bool = True,
    align_beam_to_hydro_mesh: bool = True,
    rotation_center: np.ndarray | None = None,
    cpt=None,
    mesh=None,
) -> tuple[list[str], np.ndarray, np.ndarray, dict]:
    """
    Build structural modal face fields and per-mode rigid targets.

    Returns
    -------
    mode_names, d_structural, d_target, meta
    """
    norm_mode = str(phi_normalization or "none").strip().lower()
    normalize_phi = norm_mode not in ("none", "raw", "off", "false", "0")

    mode_names, d_structural, span_axis_eff, _phi_scales = _build_face_displacements_from_beam(
        face_centers=face_centers,
        modal_dir=modal_dir,
        prefix=prefix,
        mode_indices=mode_indices,
        span_axis=span_axis,
        normalize_phi=normalize_phi,
        disp_scale=1.0,
        cfg=cfg,
        align_beam_to_hydro_mesh=align_beam_to_hydro_mesh,
        phi_normalization=phi_normalization,
    )
    n_modes = d_structural.shape[0]
    d_target = np.empty_like(d_structural)
    meta: dict = {
        "target_kind": target,
        "span_axis_used": int(span_axis_eff),
        "ref_frac": float(ref_frac),
        "match_peak_amplitude": bool(match_peak_amplitude),
        "per_mode": [],
    }

    if target in ("rigid_body", "rigid6_projection"):
        if cpt is None or mesh is None:
            raise ValueError(f"{target} requires cpt and mesh.")
        rigid_names, rigid6 = rigid_body_unit_face_displacements(
            cpt, mesh, rotation_center=rotation_center
        )
        meta["rigid6_dof_names"] = list(rigid_names)

    if target == "rigid_body":
        for im in range(n_modes):
            d_tgt, dof_name, mac, j_rigid = _rigid_body_target_for_mode(
                d_structural[im],
                rigid6,
                rigid_names,
                match_peak_amplitude=match_peak_amplitude,
            )
            d_target[im] = d_tgt
            meta["per_mode"].append(
                {
                    "mode_index": int(mode_indices[im]),
                    "mode_name": mode_names[im],
                    "rigid_dof_at_alpha_1": dof_name,
                    "rigid_dof_index": j_rigid,
                    "mac_squared_vs_structural": mac,
                    "shape_residual_rel": float(
                        np.linalg.norm(d_tgt - d_structural[im])
                        / max(np.linalg.norm(d_structural[im]), 1e-30)
                    ),
                }
            )
    elif target == "span_rigidify":
        R, beam_pos, constrained, n_nodes, _, _ = _beam_setup(
            cfg, modal_dir, prefix, align_beam_to_hydro_mesh=align_beam_to_hydro_mesh
        )
        eigendata_csv = f"{modal_dir}/{prefix}_eigendata.csv"
        for im, mode_idx in enumerate(mode_indices):
            phi = _load_phi_full_row(eigendata_csv, n_nodes, constrained, mode_idx)
            if normalize_phi:
                scale = _phi_normalization_scale(phi, norm_mode)
                phi = phi / scale
            if align_beam_to_hydro_mesh and not np.allclose(
                _beam_to_global_rotation_from_cfg(cfg), np.eye(3), atol=1e-12, rtol=0.0
            ):
                phi = _rotate_nodal_phi_to_hydro_frame(phi, n_nodes, R)
            d_tgt = _rigidify_span_displacement(
                face_centers, beam_pos, phi, span_axis_eff, ref_frac=ref_frac
            )
            d_target[im] = d_tgt
            meta["per_mode"].append(
                {
                    "mode_index": int(mode_idx),
                    "mode_name": mode_names[im],
                    "rigid6_coeffs": [],
                    "projection_residual_rel": float(
                        np.linalg.norm(d_tgt - d_structural[im])
                        / max(np.linalg.norm(d_structural[im]), 1e-30)
                    ),
                }
            )
    elif target == "rigid6_projection":
        for im in range(n_modes):
            d_tgt, coeffs = _project_onto_rigid6(d_structural[im], rigid6)
            d_target[im] = d_tgt
            meta["per_mode"].append(
                {
                    "mode_index": int(mode_indices[im]),
                    "mode_name": mode_names[im],
                    "rigid6_coeffs": coeffs.tolist(),
                    "projection_residual_rel": float(
                        np.linalg.norm(d_tgt - d_structural[im])
                        / max(np.linalg.norm(d_structural[im]), 1e-30)
                    ),
                }
            )
    else:
        raise ValueError(
            f"Unknown target={target!r}. Use 'rigid_body', 'span_rigidify', or 'rigid6_projection'."
        )

    return mode_names, d_structural, d_target, meta


def interpolate_face_displacements(
    d_structural: np.ndarray,
    d_target: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Convex blend: (1 − α) structural + α target."""
    a = float(alpha)
    if not (0.0 <= a <= 1.0):
        raise ValueError(f"alpha must be in [0, 1], got {a}")
    return (1.0 - a) * np.asarray(d_structural, dtype=float) + a * np.asarray(d_target, dtype=float)


def face_centers_before_depth_translation(mesh, depth: float) -> np.ndarray:
    """Face centres in the structural frame (undo mesh z-shift used for submergence)."""
    fc = np.asarray(mesh.faces_centers, dtype=float).copy()
    if depth != 0.0:
        fc[:, 2] += float(depth)
    return fc


def default_mode_indices(cfg) -> list[int]:
    return _mode_indices_from_config(cfg)


def resolve_span_axis(cfg, cli_span_axis: int | None = None) -> int | None:
    return _resolve_modal_span_axis(cfg, cli_span_axis)


def load_mesh(cpt, mesh_path: str):
    return _resolve_mesh_loader(cpt, mesh_path)
