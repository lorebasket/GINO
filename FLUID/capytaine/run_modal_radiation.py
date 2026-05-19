"""
Run Capytaine radiation problems for user-defined modal DOFs or rigid-body motion.

Modal input file (.npz), when not using rigid body:
    - mode_names: array-like of shape (n_modes,) with mode labels (optional)
    - face_displacements: shape (n_modes, n_faces, 3) [m]

Rigid body: set ``rigid_body_motion = True`` in the case config (see examples/config.py
GLOBAL_FLAGS) or pass ``--rigid-body``. Then Capytaine's six DOFs (Surge…Yaw) are used
and ``--modes`` must not be passed.

This script:
    1) Loads a hull mesh.
    2) Either injects modal face displacements as custom DOFs in ``body.dofs``, or
       (when rigid-body mode) registers Capytaine's six rigid-body DOFs.
    3) Solves radiation for each (DOF, omega).
    4) Saves added mass A(omega) and radiation damping B(omega) (n×n per omega).

    Optional **modal projection debug** (``--plot-mode-projection-debug``): one PNG per mode with the
    hull, face-center displacement quivers (same as ``body.dofs``), an area-weighted vertex
    extrapolation for a quick shape check, the deformed hull colored by per-triangle mean edge
    stretch (|L_def/L_ref − 1|), and thicker polylines on detected top/bottom caps. Add
    ``--show-mode-projection-debug`` for an interactive 3D window per mode (requires a GUI
    matplotlib backend, e.g. TkAgg). See also ``COUPLING/debug/plot_mode_shape_on_mesh.py`` for
    vertex-wise beam kinematics independent of this script.

    **Beam CSV → hydro mesh:** when ``config.pitch`` / ``alpha_deg`` / ``dihedral_angle`` rotate the
    hull (as in ``build_capytaine_mesh.py``), beam node positions and nodal translation/rotation blocks
    in the eigenvector are rotated by the same ``R`` into the hydro frame, and ``span_axis`` is
    inferred from beam extent unless ``capytaine_modal_span_axis`` or ``--span-axis`` is set.
    Use ``--skip-hydro-beam-alignment`` only to recover the old (often inconsistent) behaviour.

    Rigid-body mode with non-identity ``pitch`` / ``alpha_deg`` / ``dihedral_angle`` (same compound
    rotation as ``build_capytaine_mesh.py``): global-axis Surge…Yaw matrices are also written and
    plotted in a **beam-local** basis via ``M_beam = T^T M_global T`` with ``T = blkdiag(R, R)``,
    ``R = R_x(pitch) R_y(α) R_y(Γ)``. Outputs: ``*_beam_local*`` PNG/CSV and ``added_mass_beam_local``
    keys inside per-depth ``modal_radiation_AB*.npz`` metadata.

Example:
    python FLUID/capytaine/run_modal_radiation.py \
        --mesh FLUID/capytaine/NACA0003_mesh.vtu \
        --modes /path/to/modal_displacements_on_faces.npz \
        --omega-min 0.5 --omega-max 120.0 --n-omega 60 \
        --rho 997.0 \
        --out-dir FLUID/capytaine/results_modal_radiation
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Iterable

import numpy as np

# ── FSI examples.config ────────────────────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_FSI_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
_EXAMPLES_DIR = os.path.join(_FSI_ROOT, "examples")
if _EXAMPLES_DIR not in sys.path:
    sys.path.insert(0, _EXAMPLES_DIR)

from config import get_config  # noqa: E402


def _rotation_y(angle_deg: float) -> np.ndarray:
    """Same convention as ``build_capytaine_mesh.py`` (dihedral / angle of attack about +Y)."""
    t = np.deg2rad(float(angle_deg))
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=float)


def _rotation_x(angle_deg: float) -> np.ndarray:
    """Structural pitch about +X, same as ``build_capytaine_mesh.py`` / ``rotate_beams_x``."""
    t = np.deg2rad(float(angle_deg))
    c, s = np.cos(t), np.sin(t)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=float)


def _beam_to_global_rotation_from_cfg(cfg) -> np.ndarray:
    """
    Rotation ``R`` mapping vectors from the mesher beam frame (X chord, Y span, Z thickness)
    into global axes — same as ``build_capytaine_mesh.py``:

        R = R_x(pitch) @ R_y(alpha) @ R_y(dihedral)
    """
    pitch = float(getattr(cfg, "pitch", 0.0))
    alpha = float(getattr(cfg, "alpha_deg", 0.0))
    dihedral = float(getattr(cfg, "dihedral_angle", 0.0))
    return _rotation_x(pitch) @ _rotation_y(alpha) @ _rotation_y(dihedral)
from modal_radiation_io import save_modal_radiation_npz  # noqa: E402

# Plotting: added mass per-depth files use an absolute floor; multi-depth overlay
# masks values negligible vs the largest |A_ik| in that subplot (solver / float noise).
ADDED_MASS_VS_OMEGA_DEPTH_ABS_FLOOR = 1e-4
ADDED_MASS_MULTIDEPTH_REL_NOISE_FLOOR = 1e-6


def _parse_omega_args(args, cfg) -> np.ndarray:
    cli = str(getattr(args, "omega_list", "") or "").strip()
    if cli:
        omega = np.asarray(
            [float(x.strip()) for x in cli.split(",") if x.strip()],
            dtype=float,
        ).ravel()
    elif hasattr(cfg, "omega_list"):
        omega = np.asarray(cfg.omega_list, dtype=float).ravel()
    else:
        raise ValueError("No omega definition found. Provide --omega-list or set cfg.omega_list.")
    if np.any(omega <= 0.0):
        raise ValueError("All omega values must be > 0.")
    return omega


def _load_modal_displacements(path: str) -> tuple[list[str], np.ndarray]:
    data = np.load(path, allow_pickle=True)
    if "face_displacements" not in data:
        raise KeyError("Missing 'face_displacements' in modal npz file.")

    face_disp = np.asarray(data["face_displacements"], dtype=float)
    if face_disp.ndim != 3 or face_disp.shape[2] != 3:
        raise ValueError(
            "face_displacements must have shape (n_modes, n_faces, 3). "
            f"Got {face_disp.shape}."
        )

    n_modes = face_disp.shape[0]
    if "mode_names" in data:
        mode_names = [str(x) for x in data["mode_names"].tolist()]
    else:
        mode_names = [f"mode_{i + 1}" for i in range(n_modes)]

    if len(mode_names) != n_modes:
        raise ValueError(
            f"mode_names length ({len(mode_names)}) != n_modes ({n_modes})."
        )
    return mode_names, face_disp


def _load_beam_positions(nodes_csv: str) -> np.ndarray:
    rows = []
    with open(nodes_csv, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(
                (
                    int(row["Node_Index"]),
                    float(row["X_coord_m"]),
                    float(row["Y_coord_m"]),
                    float(row["Z_coord_m"]),
                )
            )
    rows.sort(key=lambda t: t[0])
    if not rows:
        raise ValueError(f"No rows in {nodes_csv}.")
    if rows[-1][0] != len(rows) - 1:
        raise ValueError(f"Non-contiguous Node_Index in {nodes_csv}.")
    return np.array([x[1:] for x in rows], dtype=float)


def _constrained_global_dofs(constrained_csv: str) -> set[int]:
    out: set[int] = set()
    with open(constrained_csv, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            ni = int(row["Node_Index"])
            local = [int(x.strip()) for x in row["Constrained_DOF_Indices"].split(",")]
            for li in local:
                out.add(ni * 6 + li)
    return out


def _load_phi_full_row(eigendata_csv: str, n_nodes: int, constrained: set[int], mode_row: int) -> np.ndarray:
    """
    Reconstruct the full (n_nodes * 6) eigenvector from the reduced CSV representation.

    Constrained DOFs (including all root-node DOFs for a clamped root) are left at
    their initialised value of **zero**, so ``phi_full`` satisfies the essential
    boundary conditions exactly.  ``_compute_displacements`` therefore inherits
    correct zero displacement at fully clamped nodes without any extra logic.
    """
    total_dof = n_nodes * 6
    free_dofs = sorted(set(range(total_dof)) - constrained)
    with open(eigendata_csv, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        _ = next(reader)
        n_modes_seen = 0
        for row in reader:
            if len(row) < 5:
                continue
            if n_modes_seen == mode_row:
                vals = np.array(row[4 : 4 + len(free_dofs)], dtype=float)
                if vals.size != len(free_dofs):
                    raise ValueError(
                        f"Mode row length mismatch in {eigendata_csv}: "
                        f"expected {len(free_dofs)}, got {vals.size}."
                    )
                phi = np.zeros(total_dof, dtype=float)
                phi[np.array(free_dofs, dtype=int)] = vals
                return phi
            n_modes_seen += 1
    raise IndexError(f"mode_row={mode_row} out of range (only {n_modes_seen} modes in file)")


def _span_interp(y_colloc: float, beam_pos: np.ndarray, span_axis: int):
    y_nodes = beam_pos[:, span_axis]
    sort_idx = np.argsort(y_nodes)
    y_sorted = y_nodes[sort_idx]
    y_rng = float(y_sorted[-1] - y_sorted[0])
    if y_rng < 1e-10:
        raise ValueError(
            f"Degenerate beam span along axis {span_axis}: coordinate range is {y_rng:g} m. "
            "Beam node CSV likely does not vary along this axis (e.g. mesh pitched with "
            "config.pitch but nodes still unpitched, or wrong --span-axis / capytaine_modal_span_axis)."
        )
    j = int(np.searchsorted(y_sorted, y_colloc))
    j = int(np.clip(j, 1, len(y_sorted) - 1))
    i_lo = int(sort_idx[j - 1])
    i_hi = int(sort_idx[j])
    y1, y2 = float(y_sorted[j - 1]), float(y_sorted[j])
    if abs(y2 - y1) < 1e-14:
        xi = 0.5
    else:
        xi = float(np.clip((y_colloc - y1) / (y2 - y1), 0.0, 1.0))
    return i_lo, i_hi, 1.0 - xi, xi


def _compute_displacements(points: np.ndarray, beam_pos: np.ndarray, phi_full: np.ndarray, span_axis: int) -> np.ndarray:
    """
    Map a beam eigenvector ``phi_full`` to 3-D displacements at arbitrary ``points``.

    Rigid cross-section (Euler-Bernoulli) kinematics
    -------------------------------------------------
    For a point ``p`` on the cross-section whose elastic-axis position is ``x_ea``
    (linearly interpolated between the two bracketing beam nodes), the displacement is

        d(p) = u + theta × (p − x_ea)

    where ``u`` and ``theta`` are the linearly interpolated nodal translation and
    infinitesimal rotation vectors.  The cross-product term is the *rigid* rotation
    of the cross-section about the elastic axis; no in-plane (warping/shear)
    deformation of the section is assumed.

    Root clamping
    -------------
    ``phi_full`` is built by ``_load_phi_full_row``, which places zeros at every
    constrained DOF index (supplied by ``_constrained_global_dofs``).  For a
    fully clamped root node all six of its DOFs are zero, so ``u=0``, ``theta=0``
    there, and the interpolated displacement goes to zero as ``p`` approaches the
    root — no special treatment is needed here.

    Parameters
    ----------
    points :   (N, 3) array — face centres in the UNDEFORMED structural frame
               (i.e., before any depth translation applied to the hydro mesh).
    beam_pos : (n_nodes, 3) array — beam node positions in the same frame as ``points``.
    phi_full : (n_nodes * 6,) array — full eigenvector with constrained DOFs zeroed.
    span_axis: axis index (0/1/2) along which beam nodes are sorted for interpolation.
    """
    out = np.empty_like(points, dtype=float)
    for i in range(points.shape[0]):
        p = points[i]
        i_lo, i_hi, n1, n2 = _span_interp(float(p[span_axis]), beam_pos, span_axis)
        q_lo = phi_full[i_lo * 6 : i_lo * 6 + 6]  # [u_x u_y u_z  th_x th_y th_z] at node i_lo
        q_hi = phi_full[i_hi * 6 : i_hi * 6 + 6]
        u  = n1 * q_lo[:3]  + n2 * q_hi[:3]   # interpolated translation
        th = n1 * q_lo[3:6] + n2 * q_hi[3:6]  # interpolated rotation
        x_ea = n1 * beam_pos[i_lo] + n2 * beam_pos[i_hi]   # interpolated elastic axis
        # Rigid cross-section: d = u + theta × (p - x_ea)
        out[i] = u + np.cross(th, p - x_ea)
    return out


def _infer_span_axis_from_beam_positions(beam_pos: np.ndarray) -> int:
    """Pick axis (0,1,2) along which beam node coordinates have the largest extent."""
    ext = beam_pos.max(axis=0) - beam_pos.min(axis=0)
    return int(np.argmax(ext))


def _rotate_nodal_phi_to_hydro_frame(phi_full: np.ndarray, n_nodes: int, R: np.ndarray) -> np.ndarray:
    """
    Rotate each node's translation and infinitesimal rotation blocks into hydro/global frame.

    Same rigid map as mesh vertices: ``u_g = R @ u``, ``theta_g = R @ theta`` (vectors in R^3).
    """
    R = np.asarray(R, dtype=float).reshape(3, 3)
    out = np.empty_like(phi_full, dtype=float)
    for i in range(int(n_nodes)):
        sl = slice(i * 6, i * 6 + 6)
        u = phi_full[sl][:3]
        th = phi_full[sl][3:6]
        out[sl] = np.concatenate([R @ u, R @ th])
    return out


def _build_face_displacements_from_beam(
    *,
    face_centers: np.ndarray,
    modal_dir: str,
    prefix: str,
    mode_indices: list[int],
    span_axis: int | None,
    normalize_phi: bool,
    disp_scale: float,
    cfg,
    align_beam_to_hydro_mesh: bool = True,
) -> tuple[list[str], np.ndarray, int]:
    nodes_csv = os.path.join(modal_dir, f"{prefix}_nodes.csv")
    constrained_csv = os.path.join(modal_dir, f"{prefix}_constrained_dofs.csv")
    eigendata_csv = os.path.join(modal_dir, f"{prefix}_eigendata.csv")
    for p in (nodes_csv, constrained_csv, eigendata_csv):
        if not os.path.isfile(p):
            raise FileNotFoundError(p)

    R = _beam_to_global_rotation_from_cfg(cfg)
    beam_pos_raw = _load_beam_positions(nodes_csv)
    constrained = _constrained_global_dofs(constrained_csv)
    n_nodes = int(beam_pos_raw.shape[0])

    if align_beam_to_hydro_mesh and not np.allclose(R, np.eye(3), atol=1e-12, rtol=0.0):
        beam_pos = np.einsum("ij,nj->ni", R, beam_pos_raw)
        print(
            "[beam→mesh] Rotated beam node positions with R = R_x(pitch) R_y(α) R_y(Γ) "
            "(same as build_capytaine_mesh.py) so they lie in the hydro mesh frame."
        )
    else:
        beam_pos = np.asarray(beam_pos_raw, dtype=float).copy()

    if span_axis is None:
        span_axis_eff = _infer_span_axis_from_beam_positions(beam_pos)
        ext = beam_pos.max(axis=0) - beam_pos.min(axis=0)
        print(f"[beam→mesh] span_axis inferred={span_axis_eff} from beam extent (Δx,Δy,Δz)={tuple(ext)} m")
    else:
        span_axis_eff = int(span_axis)
        print(f"[beam→mesh] span_axis fixed={span_axis_eff} (capytaine_modal_span_axis or --span-axis).")

    disp_all = []
    mode_names = []
    for mode_idx in mode_indices:
        phi = _load_phi_full_row(eigendata_csv, n_nodes, constrained, mode_idx)
        phi_max = float(np.max(np.abs(phi)))
        if normalize_phi and phi_max > 0.0 and np.isfinite(phi_max):
            phi = phi / phi_max
        if align_beam_to_hydro_mesh and not np.allclose(R, np.eye(3), atol=1e-12, rtol=0.0):
            phi = _rotate_nodal_phi_to_hydro_frame(phi, n_nodes, R)
        d = _compute_displacements(face_centers, beam_pos, phi, span_axis=span_axis_eff)
        disp_all.append(disp_scale * d)
        mode_names.append(f"mode_{mode_idx + 1}")

    return mode_names, np.asarray(disp_all, dtype=float), span_axis_eff


def _save_csv_matrix(path: str, omega: np.ndarray, names: list[str], mat: np.ndarray) -> None:
    n_omega, n_mode_i, n_mode_k = mat.shape
    if n_omega != omega.size:
        raise ValueError("Inconsistent omega and matrix shape while writing CSV.")
    with open(path, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["omega_rad_s", "influenced_mode", "radiating_mode", "value"])
        for io, w in enumerate(omega):
            for i in range(n_mode_i):
                for k in range(n_mode_k):
                    wr.writerow([f"{w:.12e}", names[i], names[k], f"{mat[io, i, k]:.12e}"])


def _plot_modal_matrix_sweep(
    omega: np.ndarray,
    mode_names: list[str],
    mat: np.ndarray,
    *,
    title: str,
    y_label: str,
    out_path: str,
    mask_abs_below: float | None = None,
) -> None:
    import matplotlib.pyplot as plt

    n_omega, n_mode_i, n_mode_k = mat.shape
    if n_omega != omega.size:
        raise ValueError("Inconsistent omega size in plotting.")

    fig, axes = plt.subplots(n_mode_i, n_mode_k, figsize=(3.8 * n_mode_k, 3.0 * n_mode_i), squeeze=False)
    for i in range(n_mode_i):
        for k in range(n_mode_k):
            ax = axes[i][k]
            y = np.asarray(mat[:, i, k], dtype=float).copy()
            if mask_abs_below is not None:
                y[np.abs(y) < mask_abs_below] = np.nan
            ax.plot(omega, y, lw=1.8)
            ax.grid(True, alpha=0.3)
            ax.set_title(f"{mode_names[i]} <- {mode_names[k]}", fontsize=9)
            if i == n_mode_i - 1:
                ax.set_xlabel("omega [rad/s]")
            if k == 0:
                ax.set_ylabel(y_label)

    fig.suptitle(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_multi_depth_modal_matrix_sweep(
    omega: np.ndarray,
    mode_names: list[str],
    mats_by_depth: list[np.ndarray],
    depth_values: list[float],
    *,
    title: str,
    y_label: str,
    out_path: str,
    mask_rel_noise_below: float | None = None,
) -> None:
    import matplotlib.pyplot as plt

    n_omega = omega.size
    n_mode = len(mode_names)
    if len(mats_by_depth) != len(depth_values):
        raise ValueError("Depth/matrix list size mismatch.")

    fig, axes = plt.subplots(n_mode, n_mode, figsize=(4.0 * n_mode, 3.2 * n_mode), squeeze=False)
    for i in range(n_mode):
        for k in range(n_mode):
            ax = axes[i][k]
            panel_peak = 0.0
            if mask_rel_noise_below is not None:
                stacked = np.stack(
                    [np.abs(np.asarray(m[:, i, k], dtype=float)) for m in mats_by_depth],
                    axis=0,
                )
                panel_peak = float(np.nanmax(stacked)) if stacked.size else 0.0
            abs_floor = (
                mask_rel_noise_below * panel_peak
                if mask_rel_noise_below is not None and panel_peak > 0.0
                else None
            )
            for depth, mat in zip(depth_values, mats_by_depth):
                if mat.shape != (n_omega, n_mode, n_mode):
                    raise ValueError(f"Unexpected matrix shape in multi-depth plot: {mat.shape}")
                y = np.asarray(mat[:, i, k], dtype=float).copy()
                if abs_floor is not None:
                    y[np.abs(y) < abs_floor] = np.nan
                ax.plot(omega, y, lw=1.6, label=f"depth={depth:g} m")
            ax.grid(True, alpha=0.3)
            ax.set_title(f"{mode_names[i]} <- {mode_names[k]}", fontsize=9)
            if i == n_mode - 1:
                ax.set_xlabel("omega [rad/s]")
            if k == 0:
                ax.set_ylabel(y_label)
            if i == 0 and k == n_mode - 1:
                ax.legend(loc="best", fontsize=8)

    fig.suptitle(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_mesh_free_surface_debug(
    *,
    mesh,
    free_surface: float,
    water_depth: float,
    out_path: str,
    show: bool = False,
) -> None:
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    v = np.asarray(mesh.vertices, dtype=float)
    f = np.asarray(mesh.faces, dtype=np.int64)

    polys = []
    for row in f:
        ids = list(dict.fromkeys(int(i) for i in row))
        if len(ids) < 3:
            continue
        polys.append(v[np.asarray(ids, dtype=np.int64)])

    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(111, projection="3d")

    coll = Poly3DCollection(polys, alpha=0.22, facecolor="steelblue", edgecolor="k", linewidths=0.15)
    ax.add_collection3d(coll)

    # Bounding box includes body and reference planes to avoid "tight zoom" on body only.
    lo = v.min(axis=0).copy()
    hi = v.max(axis=0).copy()

    xx = np.array([[lo[0], hi[0]], [lo[0], hi[0]]], dtype=float)
    yy = np.array([[lo[1], lo[1]], [hi[1], hi[1]]], dtype=float)

    z_refs = []
    if np.isfinite(free_surface):
        z_refs.append(float(free_surface))
        zz_fs = np.full_like(xx, float(free_surface))
        ax.plot_surface(xx, yy, zz_fs, alpha=0.25, color="cyan", linewidth=0)
        ax.text(hi[0], hi[1], float(free_surface), " free_surface", color="teal")

    if np.isfinite(free_surface) and np.isfinite(water_depth):
        z_bottom = float(free_surface) - float(water_depth)
        z_refs.append(z_bottom)
        zz_b = np.full_like(xx, z_bottom)
        ax.plot_surface(xx, yy, zz_b, alpha=0.25, color="sandybrown", linewidth=0)
        ax.text(hi[0], lo[1], z_bottom, " seabed", color="saddlebrown")

    if z_refs:
        lo[2] = min(lo[2], min(z_refs))
        hi[2] = max(hi[2], max(z_refs))

    # Use generous margins and near-isotropic box aspect to keep all geometry visible.
    span = np.maximum(hi - lo, 1e-9)
    pad = 0.18 * span
    lo -= pad
    hi += pad

    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Debug geometry: translated mesh vs free-surface/seabed")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    if show:
        plt.show(block=True)
    plt.close(fig)


def _vertex_area_weighted_mean_face_disp(mesh, d_face: np.ndarray) -> np.ndarray:
    """
    Area-weighted average of per-face constant displacements onto mesh vertices.

    Used only for a smooth **visual** deformed hull; Capytaine still uses the raw per-face
    vectors in ``body.dofs``.
    """
    v = np.asarray(mesh.vertices, dtype=float)
    n_v = int(v.shape[0])
    f = np.asarray(mesh.faces, dtype=np.int64)
    d_face = np.asarray(d_face, dtype=float)
    if d_face.ndim != 2 or d_face.shape[1] != 3:
        raise ValueError(f"d_face must be (n_faces, 3), got {d_face.shape}.")
    nf = int(f.shape[0])
    if d_face.shape[0] != nf:
        raise ValueError(f"d_face rows {d_face.shape[0]} != mesh faces {nf}.")
    areas = np.asarray(mesh.faces_areas, dtype=float).ravel()
    if areas.size != nf:
        raise ValueError("mesh.faces_areas length must match number of faces.")
    acc = np.zeros((n_v, 3), dtype=float)
    wsum = np.zeros(n_v, dtype=float)
    for fi in range(nf):
        row = f[fi]
        ids = list(dict.fromkeys(int(i) for i in row if int(i) >= 0))
        if len(ids) < 3:
            continue
        w = max(float(areas[fi]), 1e-30)
        for vid in ids:
            if 0 <= vid < n_v:
                acc[vid] += w * d_face[fi]
                wsum[vid] += w
    m = wsum > 1e-18
    out = np.zeros((n_v, 3), dtype=float)
    out[m] = acc[m] / wsum[m, np.newaxis]
    return out


def _mean_edge_relative_stretch(v_ref: np.ndarray, v_def: np.ndarray, f: np.ndarray) -> np.ndarray:
    """
    Per-triangle mean of |L_def / L_ref - 1| over its three edges (finite-strain edge metric).

    Scales like a discrete surface analogue of the deformation gradient magnitude for the
    displayed mapping ref → def (useful when coloring the deformed hull).
    """
    v0 = np.asarray(v_ref, dtype=float)
    v1 = np.asarray(v_def, dtype=float)
    f = np.asarray(f, dtype=np.int64)
    out = np.zeros(f.shape[0], dtype=float)
    for fi, row in enumerate(f):
        ids = list(dict.fromkeys(int(i) for i in row if int(i) >= 0))
        if len(ids) < 3:
            continue
        a, b, c = ids[0], ids[1], ids[2]
        acc = 0.0
        for i, j in ((a, b), (b, c), (c, a)):
            L0 = float(np.linalg.norm(v0[i] - v0[j]))
            L1 = float(np.linalg.norm(v1[i] - v1[j]))
            if L0 > 1e-18:
                acc += abs(L1 / L0 - 1.0)
        out[fi] = acc / 3.0
    return out


def _horizontal_cap_boundary_edges(
    v: np.ndarray,
    f: np.ndarray,
    *,
    top: bool,
    z_band_frac: float = 0.02,
) -> list[tuple[int, int]]:
    """Return undirected boundary edges (vertex index pairs) of the top or bottom horizontal cap."""
    v = np.asarray(v, dtype=float)
    f = np.asarray(f, dtype=np.int64)
    z = v[:, 2]
    zmin, zmax = float(z.min()), float(z.max())
    hz = zmax - zmin
    band = max(hz * float(z_band_frac), 1e-9)

    def face_on_cap(row: np.ndarray) -> bool:
        ids = list(dict.fromkeys(int(i) for i in row if int(i) >= 0))
        if len(ids) < 3:
            return False
        zs = [float(z[i]) for i in ids]
        if top:
            return min(zs) >= zmax - band
        return max(zs) <= zmin + band

    from collections import defaultdict

    edge_count: defaultdict[tuple[int, int], int] = defaultdict(int)
    for row in f:
        if not face_on_cap(row):
            continue
        ids = list(dict.fromkeys(int(i) for i in row if int(i) >= 0))
        if len(ids) < 3:
            continue
        a, b, c = ids[0], ids[1], ids[2]
        for p, q in ((a, b), (b, c), (c, a)):
            e = (p, q) if p < q else (q, p)
            edge_count[e] += 1
    return [e for e, c in edge_count.items() if c == 1]


def _plot_cap_boundaries(
    ax,
    v: np.ndarray,
    f: np.ndarray,
    *,
    color: str,
    linewidth: float,
    z_band_frac: float = 0.02,
) -> None:
    """Draw thicker polylines on top and bottom mesh caps (open hull ends)."""
    v = np.asarray(v, dtype=float)
    for top in (True, False):
        edges = _horizontal_cap_boundary_edges(v, f, top=top, z_band_frac=z_band_frac)
        for i, j in edges:
            p0, p1 = v[i], v[j]
            ax.plot(
                [p0[0], p1[0]],
                [p0[1], p1[1]],
                [p0[2], p1[2]],
                color=color,
                linewidth=linewidth,
                solid_capstyle="round",
            )


def _plot_modal_projection_on_mesh(
    *,
    mesh,
    face_disp: np.ndarray,
    mode_names: list[str],
    out_dir: str,
    depth_suffix: str,
    quiver_scale: float,
    max_arrows: int,
    show: bool = False,
) -> None:
    """
    Debug: visualize modal vectors assigned to ``body.dofs`` (per face, 3 components).

    For each mode: semi-transparent reference hull, deformed hull colored by a discrete
    edge-stretch metric (mean |L_def/L_ref − 1| on each triangle, a finite-difference
    analogue of deformation / shape-function gradient along the surface), subsampled
    quivers at face centers (exact ``face_disp``), and thicker polylines on detected
    top/bottom horizontal caps.
    """
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.cm as mpl_cm
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fc = np.asarray(mesh.faces_centers, dtype=float)
    v0 = np.asarray(mesh.vertices, dtype=float)
    f = np.asarray(mesh.faces, dtype=np.int64)
    face_disp = np.asarray(face_disp, dtype=float)
    n_modes, n_faces, _ = face_disp.shape
    if fc.shape[0] != n_faces:
        raise ValueError(f"faces_centers rows {fc.shape[0]} != n_faces {n_faces}.")

    max_arrows = max(50, int(max_arrows))
    if n_faces > max_arrows:
        idx = np.unique(np.linspace(0, n_faces - 1, max_arrows, dtype=int))
    else:
        idx = np.arange(n_faces, dtype=int)

    os.makedirs(out_dir, exist_ok=True)

    def _polys_and_stretch_for_valid_faces(v_ref: np.ndarray, v_df: np.ndarray) -> tuple[list, list, np.ndarray]:
        polys_a: list[np.ndarray] = []
        polys_b: list[np.ndarray] = []
        s_list: list[float] = []
        stretch_all = _mean_edge_relative_stretch(v_ref, v_df, f)
        for fi, row in enumerate(f):
            ids = list(dict.fromkeys(int(i) for i in row))
            if len(ids) < 3:
                continue
            vid = np.asarray(ids, dtype=np.int64)
            polys_a.append(v_ref[vid])
            polys_b.append(v_df[vid])
            s_list.append(float(stretch_all[fi]))
        return polys_a, polys_b, np.asarray(s_list, dtype=float)

    for im in range(n_modes):
        d = face_disp[im]
        norms = np.linalg.norm(d, axis=1)
        bbox_diag = float(np.linalg.norm(v0.max(axis=0) - v0.min(axis=0)))
        d_ref = float(np.percentile(norms, 95)) if norms.size else 0.0
        d_ref = max(d_ref, 1e-15)
        auto_vis = 0.06 * bbox_diag / d_ref
        vis = auto_vis * float(quiver_scale)
        print(
            f"  Mode projection debug '{mode_names[im]}': "
            f"|d| max={float(np.max(norms)):.6g} m, mean={float(np.mean(norms)):.6g} m; "
            f"auto vis scale={auto_vis:.4g} m (× user {quiver_scale:g} → {vis:.4g} m arrow length scale)"
        )
        v_def = v0 + vis * _vertex_area_weighted_mean_face_disp(mesh, d)
        polys0, polys1, stretch_face = _polys_and_stretch_for_valid_faces(v0, v_def)
        stretch_max = float(np.max(stretch_face)) if stretch_face.size else 0.0
        print(f"    edge-stretch (mean |ΔL/L|) max={stretch_max:.6g} on displayed triangles")

        fig = plt.figure(figsize=(11.5, 7.2))
        ax = fig.add_subplot(111, projection="3d")
        ax.add_collection3d(
            Poly3DCollection(polys0, alpha=0.18, facecolor="steelblue", edgecolor="k", linewidths=0.12)
        )
        if stretch_face.size:
            lo_s, hi_s = np.percentile(stretch_face, [3.0, 97.0])
            if hi_s <= lo_s + 1e-30:
                hi_s = lo_s + 1e-12
            norm = Normalize(vmin=float(lo_s), vmax=float(hi_s))
            fc_stretch = mpl_cm.inferno(norm(stretch_face))
            coll_def = Poly3DCollection(
                polys1,
                facecolors=fc_stretch,
                edgecolor="0.15",
                linewidths=0.08,
                alpha=0.82,
            )
            ax.add_collection3d(coll_def)
            sm = ScalarMappable(norm=norm, cmap=mpl_cm.inferno)
            sm.set_array(stretch_face)
            cbar = fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.02, aspect=22)
            cbar.set_label("mean |ΔL/L| per face (ref → def)")
        else:
            ax.add_collection3d(
                Poly3DCollection(polys1, alpha=0.28, facecolor="coral", edgecolor="darkred", linewidths=0.12)
            )

        _plot_cap_boundaries(ax, v0, f, color="midnightblue", linewidth=3.2, z_band_frac=0.02)
        _plot_cap_boundaries(ax, v_def, f, color="darkred", linewidth=2.8, z_band_frac=0.02)

        dqs = vis * d[idx]
        ax.quiver(
            fc[idx, 0],
            fc[idx, 1],
            fc[idx, 2],
            dqs[:, 0],
            dqs[:, 1],
            dqs[:, 2],
            length=1.0,
            normalize=False,
            color="lightgreen",
            linewidth=0.1,
            arrow_length_ratio=0.12,
        )

        allv = np.vstack([v0, v_def, fc[idx] + dqs])
        lo = allv.min(axis=0)
        hi = allv.max(axis=0)
        span = np.maximum(hi - lo, 1e-9)
        pad = 0.12 * span
        lo -= pad
        hi += pad
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_zlim(lo[2], hi[2])
        ax.set_box_aspect([1, 1, 1])
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_zlabel("Z [m]")
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(mode_names[im]))[:96]
        title = (
            f"Modal DOF on mesh: {mode_names[im]}\n"
            f"blue=reference hull; inferno=deformed hull colored by mean |ΔL/L| per face (ref→def); "
            f"thick navy/red = top/bottom cap outlines (ref / def); arrows=face_disp "
            f"(n={idx.size}/{n_faces}); vis scale ~6% bbox per 95th |d| (max edge-stretch={stretch_max:.3g})"
        )
        ax.set_title(title, fontsize=10)
        out_path = os.path.join(out_dir, f"debug_mode_projection_{safe}{depth_suffix}.png")
        fig.tight_layout()
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        print(f"    wrote {out_path}")
        if show:
            plt.show(block=True)
        plt.close(fig)


def _as_mode_matrix(
    ds,
    key: str,
    omega: np.ndarray,
    mode_names: list[str]):
    
    arr = np.asarray(ds[key].values, dtype=float)
    dims = tuple(ds[key].dims)
    target = ("omega", "influenced_dof", "radiating_dof")
    if dims != target:
        perm = [dims.index(d) for d in target]
        arr = np.transpose(arr, axes=perm)
    if arr.shape != (omega.size, len(mode_names), len(mode_names)):
        raise ValueError(
            f"Unexpected shape for '{key}': {arr.shape}, "
            f"expected {(omega.size, len(mode_names), len(mode_names))}."
        )
    return arr


def _resolve_mesh_loader(cpt, mesh_path: str):
    # Capytaine versions expose either `load_mesh` or IO readers.
    if hasattr(cpt, "load_mesh"):
        return cpt.load_mesh(mesh_path)
    from capytaine.io.mesh_loaders import load_mesh

    return load_mesh(mesh_path)


def _parse_free_surface(value) -> float:
    """Return free-surface elevation or np.inf (no free-surface effects)."""
    if value is None:
        return 0.0
    if isinstance(value, (int, float, np.floating)):
        return float(value)
    txt = str(value).strip().lower()
    if txt in {"inf", "infinite", "none", "off"}:
        return np.inf
    return float(txt)


def _translate_mesh_z(mesh, dz: float) -> None:
    if abs(dz) < 1e-15:
        return
    vertices = np.asarray(mesh.vertices, dtype=float).copy()
    vertices[:, 2] += float(dz)
    mesh.vertices = vertices


_RIGID_BODY_DOF_NAMES: tuple[str, ...] = ("Surge", "Sway", "Heave", "Roll", "Pitch", "Yaw")


def _six_dof_transform_from_rotation(R: np.ndarray) -> np.ndarray:
    """Block-diag(R, R) for stacked [trans; rot] with Capytaine rigid-body DOF order."""
    R = np.asarray(R, dtype=float).reshape(3, 3)
    z = np.zeros((3, 3), dtype=float)
    return np.block([[R, z], [z, R]])


def _transform_rigid_matrices_global_to_beam(
    mat: np.ndarray,
    *,
    R_beam_to_global: np.ndarray,
) -> np.ndarray:
    """
    Congruence transform for added mass / radiation damping in rigid 6-DOF basis.

    Capytaine uses global-axis Surge…Yaw. If ``v_g = T v_b`` with ``T = blkdiag(R, R)``
    and ``R`` maps beam-frame components to global (``v_g = R @ v_b``), then

        M_b = T.T @ M_g @ T

    so generalized forces stay power-consistent for harmonic motion in the beam frame.
    """
    mat = np.asarray(mat, dtype=float)
    if mat.ndim != 3 or mat.shape[1] != mat.shape[2]:
        raise ValueError(f"Expected (n_omega, n, n) array, got {mat.shape}.")
    n = mat.shape[1]
    if n != 6:
        raise ValueError(f"Beam-local transform is defined for 6 rigid DOFs only; got n={n}.")
    T = _six_dof_transform_from_rotation(R_beam_to_global)
    out = np.empty_like(mat)
    for io in range(mat.shape[0]):
        M = mat[io]
        out[io] = T.T @ M @ T
    return out


def _mesh_area_centroid(mesh) -> np.ndarray:
    """Area-weighted centroid of face centers (fallback rotation reference for thin hulls)."""
    fc = np.asarray(mesh.faces_centers, dtype=float)
    fa = np.asarray(mesh.faces_areas, dtype=float).ravel()
    w = float(np.sum(fa))
    if w <= 0.0 or not np.isfinite(w):
        return np.zeros(3, dtype=float)
    return np.sum(fc * fa[:, None], axis=0) / w


def _apply_rigid_body_dofs(body, *, rotation_center: np.ndarray | None) -> tuple[list[str], np.ndarray | None]:
    """
    Replace ``body.dofs`` with six rigid-body DOFs (Capytaine convention).

    Returns
    -------
    mode_names : list of 6 dof labels (same order as ``body.dofs``).
    face_disp : None (no modal face array; omitted from saved metadata).
    """
    body.dofs.clear()
    rc = None if rotation_center is None else np.asarray(rotation_center, dtype=float).ravel()
    if rc is not None and rc.size != 3:
        raise ValueError("rigid_body_rotation_center must be a length-3 sequence (x, y, z) [m].")
    if rc is not None:
        body.center_of_mass = rc
    elif getattr(body, "center_of_mass", None) is None:
        body.center_of_mass = _mesh_area_centroid(body.mesh)
    body.add_all_rigid_body_dofs()
    names = list(body.dofs.keys())
    if names != list(_RIGID_BODY_DOF_NAMES):
        missing = [n for n in _RIGID_BODY_DOF_NAMES if n not in body.dofs]
        if missing:
            raise RuntimeError(f"Rigid-body DOF setup incomplete; missing: {missing}. Got keys: {names}")
        reordered = {n: np.asarray(body.dofs[n], dtype=float) for n in _RIGID_BODY_DOF_NAMES}
        body.dofs.clear()
        body.dofs.update(reordered)
        names = list(_RIGID_BODY_DOF_NAMES)
    return names, None


def _build_floating_body(cpt, mesh, name: str = "modal_body"):
    try:
        return cpt.FloatingBody(mesh=mesh, name=name)
    except RuntimeError as exc:
        msg = str(exc)
        if "compute_connectivity" not in msg and "connectivities" not in msg:
            raise
        print(
            "FloatingBody mesh healing failed on connectivity (likely TE non-manifold topology). "
            "Retrying with heal_mesh() bypassed for this mesh instance."
        )
        try:
            mesh.heal_mesh = lambda *args, **kwargs: None  # type: ignore[attr-defined]
        except Exception as patch_exc:
            raise RuntimeError(
                "Could not patch mesh.heal_mesh for fallback body creation."
            ) from patch_exc
        return cpt.FloatingBody(mesh=mesh, name=name)


def _parse_mode_indices(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _mode_indices_from_config(cfg) -> list[int]:
    """
    First ``num_modes_flutter_egv`` dry modes (0-based rows in ``*_eigendata.csv``).

    Matches the modal basis used by ``main.py`` / ``flutter_solver`` when
    ``modes_to_analyze`` is the default ``list(range(num_modes_flutter_egv))``.
    """
    n = int(getattr(cfg, "num_modes_flutter_egv", 1))
    if n < 1:
        raise ValueError(f"num_modes_flutter_egv must be >= 1, got {n}")
    return list(range(n))


def _count_eigendata_mode_rows(eigendata_csv: str) -> int:
    """Number of mode data rows in a dry ``*_eigendata.csv`` file."""
    if not os.path.isfile(eigendata_csv):
        return 0
    n_modes = 0
    with open(eigendata_csv, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        _ = next(reader, None)
        for row in reader:
            if len(row) >= 5:
                n_modes += 1
    return n_modes


def _defaults_from_config(cfg, case_name: str) -> dict:
    blade_name = getattr(cfg, "blade_name", case_name)
    fluid = getattr(cfg, "fluid", "water")
    rho = float(getattr(cfg, "rho_f", {}).get(fluid, 997.0))
    mode_indices = _mode_indices_from_config(cfg)

    return {
        "mesh": os.path.join(_FSI_ROOT, "FLUID", "capytaine", f"{blade_name}_mesh.vtu"),
        "modal_dir": os.path.join(_FSI_ROOT, "output_data", case_name),
        "prefix": f"{case_name}_dry_egv",
        "mode_indices": mode_indices,
        "rho": rho,
        "water_depth": "inf",
        "free_surface_elevation": getattr(cfg, "free_surface_elevation", 0.0),
        "depth": getattr(cfg, "depth", 0.0),
        "out_dir": os.path.join(_FSI_ROOT, "FLUID", "capytaine", case_name, "results_modal_radiation"),
    }


def _parse_depth_values(raw_depth, cli_depth: float | None) -> list[float]:
    if cli_depth is not None:
        return [float(cli_depth)]
    if isinstance(raw_depth, np.ndarray):
        vals = [float(x) for x in raw_depth.ravel().tolist()]
    elif isinstance(raw_depth, (list, tuple)):
        vals = [float(x) for x in raw_depth]
    else:
        vals = [float(raw_depth)]
    if not vals:
        vals = [0.0]
    return vals


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compute modal radiation A(omega), B(omega) with Capytaine.")
    parser.add_argument("--case-name", "--case_name", dest="case_name", type=str, required=True, help="Case config name under examples/<case-name>.")
    parser.add_argument(
        "--rigid-body",
        action="store_true",
        help="Use six rigid-body DOFs (Surge…Yaw). Incompatible with --modes; can also set cfg.rigid_body_motion.",
    )
    parser.add_argument("--modes", type=str, default="", help="NPZ file with precomputed face_displacements.")
    parser.add_argument(
        "--mode-indices",
        type=str,
        default="",
        help="Optional comma-separated 0-based row indices into *_eigendata.csv "
        "(default: 0..num_modes_flutter_egv-1 from case config).",
    )
    parser.add_argument("--span-axis", type=int, default=1, help="Beam span axis (0=X, 1=Y, 2=Z).")
    parser.add_argument(
        "--raw-eigenvector",
        action="store_true",
        help="Use raw mass-normalized eigenvectors (default rescales each mode by max|phi|=1).",
    )
    parser.add_argument(
        "--disp-scale",
        type=float,
        default=1.0,
        help="Global multiplier applied to mapped displacements before assigning body.dofs.",
    )
    parser.add_argument("--omega-list", type=str, default="", help="Comma-separated omega list [rad/s].")
    parser.add_argument("--rho", type=float, default=None, help="Fluid density [kg/m^3] override.")
    parser.add_argument(
        "--depth",
        type=float,
        default=None,
        help="Translate mesh downward by this depth [m] before solving (z <- z - depth).",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable PNG plots of A(omega) and B(omega).",
    )
    parser.add_argument(
        "--no-debug-geometry-plot",
        action="store_true",
        help="Disable debug plot of translated mesh with free surface and seabed.",
    )
    parser.add_argument(
        "--show-debug-geometry",
        action="store_true",
        help="Show interactive 3D debug geometry plot window (also saves PNG).",
    )
    parser.add_argument(
        "--skip-hydro-beam-alignment",
        action="store_true",
        help="Beam-mapped modal runs only: do not rotate beam node positions / eigenvectors into the "
        "hydro mesh frame (disables matching build_capytaine_mesh pitch, α, Γ). For legacy debugging only.",
    )
    parser.add_argument(
        "--plot-mode-projection-debug",
        action="store_true",
        help="Modal / beam-mapped runs only: save one PNG per mode showing hull, face-center "
        "displacement quivers (same vectors as body.dofs), and an area-weighted vertex "
        "extrapolation (visual only). Ignored for rigid-body mode.",
    )
    parser.add_argument(
        "--show-mode-projection-debug",
        action="store_true",
        help="With --plot-mode-projection-debug: open an interactive 3D window per mode (close each "
        "to continue). Sets an interactive matplotlib backend before importing Capytaine; requires "
        "a desktop session (e.g. TkAgg). Implies saving the PNG as well.",
    )
    parser.add_argument(
        "--mode-projection-quiver-scale",
        type=float,
        default=1.0,
        metavar="S",
        help="Multiplies displayed displacement in mode-projection debug PNGs only (not BEM).",
    )
    parser.add_argument(
        "--mode-projection-max-arrows",
        type=int,
        default=900,
        help="Max quiver arrows per mode in mode-projection debug (uniform face subsampling).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    cfg = get_config(args.case_name)
    rigid_body_motion = bool(args.rigid_body) or bool(getattr(cfg, "rigid_body_motion", False))
    if rigid_body_motion and args.plot_mode_projection_debug:
        print("Note: --plot-mode-projection-debug applies only to modal / beam-mapped face DOFs; skipping for rigid-body.")
    if rigid_body_motion and args.show_mode_projection_debug:
        print("Note: --show-mode-projection-debug applies only with modal mode-projection plots; skipping for rigid-body.")
    if args.show_mode_projection_debug and not args.plot_mode_projection_debug:
        parser.error("--show-mode-projection-debug requires --plot-mode-projection-debug.")
    if rigid_body_motion and args.modes:
        raise ValueError(
            "Rigid-body mode is enabled; do not pass --modes. "
            "Clear cfg.rigid_body_motion / omit --rigid-body, or use rigid-body DOFs only."
        )

    defaults = _defaults_from_config(cfg, args.case_name)

    mesh_path = getattr(cfg, "mesh_path", defaults["mesh"])
    modal_dir = getattr(cfg, "modal_dir", defaults["modal_dir"])
    prefix = getattr(cfg, "prefix", defaults["prefix"])
    out_dir = getattr(cfg, "capytaine_results_dir", defaults["out_dir"])
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    rho = float(args.rho) if args.rho is not None else float(defaults["rho"])
    water_depth_cfg = getattr(cfg, "water_depth", defaults["water_depth"])
    water_depth = np.inf if str(water_depth_cfg).lower() == "inf" else float(water_depth_cfg)
    free_surface_cfg = getattr(cfg, "free_surface_elevation", defaults["free_surface_elevation"])
    free_surface = _parse_free_surface(free_surface_cfg)
    depth_values = _parse_depth_values(defaults["depth"], args.depth)
    omega = _parse_omega_args(args, cfg)

    if args.show_mode_projection_debug:
        import matplotlib

        _backend_set = False
        for be in ("TkAgg", "Qt5Agg", "QtAgg", "Gtk3Agg"):
            try:
                matplotlib.use(be, force=True)
                print(f"Interactive mode-shape plots: matplotlib backend set to {be!r}.")
                _backend_set = True
                break
            except Exception:
                continue
        if not _backend_set:
            print(
                "Warning: could not switch to an interactive matplotlib backend. "
                "Try: export MPLBACKEND=TkAgg before running. Figures may not display."
            )

    import capytaine as cpt

    print(f"Case: {args.case_name}")
    print(f"Radiation basis: {'rigid body (6 DOF)' if rigid_body_motion else 'modal / custom face DOFs'}")
    if not rigid_body_motion and not args.modes:
        n_flutter = int(getattr(cfg, "num_modes_flutter_egv", 1))
        mta = getattr(cfg, "modes_to_analyze", None)
        if mta is not None and list(mta) != list(range(n_flutter)):
            print(
                f"Note: cfg.modes_to_analyze={list(mta)} differs from "
                f"range(num_modes_flutter_egv)={list(range(n_flutter))}; "
                "beam-mapped radiation uses num_modes_flutter_egv unless --mode-indices is set."
            )
        print(
            f"Modal radiation modes: num_modes_flutter_egv={n_flutter} "
            f"-> CSV row indices {defaults['mode_indices']}"
        )
    print(f"Depth sweep: {depth_values}")
    print(
        "Free surface: "
        + ("disabled (inf)" if np.isinf(free_surface) else f"z={free_surface:.6g} m")
    )
    print(f"Omega samples: {omega.size} from {omega.min():.6g} to {omega.max():.6g} rad/s")

    _msax = getattr(cfg, "capytaine_modal_span_axis", None)
    if args.case_name == "ABRAMSON1965" and (not rigid_body_motion) and (not args.modes):
        span_axis_modal = int(_msax) if _msax is not None else None
    else:
        span_axis_modal = int(_msax) if _msax is not None else int(args.span_axis)

    span_axis_used_modal = int(args.span_axis)

    all_A: list[np.ndarray] = []
    all_B: list[np.ndarray] = []
    all_A_beam: list[np.ndarray] = []
    all_B_beam: list[np.ndarray] = []
    mode_names_ref: list[str] | None = None

    R_beam_to_global = _beam_to_global_rotation_from_cfg(cfg)
    rigid_use_beam_local = (
        rigid_body_motion
        and not np.allclose(R_beam_to_global, np.eye(3), atol=1e-10, rtol=0.0)
    )
    if rigid_use_beam_local:
        print(
            "Rigid-body results will also be written in the beam-aligned frame "
            "(A_loc = T^T A T, T=blkdiag(R,R), R = R_x(pitch) R_y(α) R_y(Γ) as in build_capytaine_mesh.py)."
        )

    for idepth, depth in enumerate(depth_values):
        print(f"\n--- Solving depth = {depth:g} m ---")
        mesh = _resolve_mesh_loader(cpt, mesh_path)
        _translate_mesh_z(mesh, -depth)
        body = _build_floating_body(cpt, mesh, name=f"modal_body_d{idepth}")
        face_disp: np.ndarray | None = None

        if rigid_body_motion:
            rigid_rc = getattr(cfg, "rigid_body_rotation_center", None)
            rot_c = None if rigid_rc is None else np.asarray(rigid_rc, dtype=float)
            mode_names, face_disp = _apply_rigid_body_dofs(body, rotation_center=rot_c)
            span_axis_used_modal = -1
            n_modes = len(mode_names)
            ref_pt = getattr(body, "center_of_mass", None)
            ref_pt = (
                np.asarray(ref_pt, dtype=float).ravel()
                if ref_pt is not None
                else _mesh_area_centroid(body.mesh)
            )
            print(f"Rigid-body DOFs: {n_modes} -> {mode_names}")
            print(
                "Rotation reference (body.center_of_mass) [m]: "
                f"({ref_pt[0]:.6g}, {ref_pt[1]:.6g}, {ref_pt[2]:.6g})"
            )
        else:
            # ── Face centres must be in the UNDEFORMED structural frame ───────────
            # body.mesh has already been translated downward by `depth` via
            # _translate_mesh_z(mesh, -depth).  The beam node positions loaded from
            # CSV (and possibly R-rotated into the hydro frame) are NOT depth-
            # translated, so we must undo the z-shift here.  This matters only for
            # the beam-mapped path below; the pre-computed NPZ path ignores
            # face_centers entirely.
            face_centers = np.asarray(body.mesh.faces_centers, dtype=float).copy()
            if depth != 0.0:
                face_centers[:, 2] += depth  # restore pre-translation z

            if args.modes:
                mode_names, face_disp = _load_modal_displacements(args.modes)
                span_axis_used_modal = -1
            else:
                if args.mode_indices:
                    mode_indices = _parse_mode_indices(args.mode_indices)
                else:
                    mode_indices = _mode_indices_from_config(cfg)
                if not mode_indices:
                    raise ValueError(
                        "No mode indices defined. Set cfg.num_modes_flutter_egv >= 1 or pass --mode-indices."
                    )
                eigendata_csv = os.path.join(modal_dir, f"{prefix}_eigendata.csv")
                n_csv = _count_eigendata_mode_rows(eigendata_csv)
                if n_csv < len(mode_indices):
                    raise ValueError(
                        f"{eigendata_csv} has {n_csv} mode row(s), but num_modes_flutter_egv="
                        f"{len(mode_indices)} requires at least that many. "
                        f"Re-run dry analysis with num_modes_egv >= {len(mode_indices)} "
                        "or lower num_modes_flutter_egv in the case config."
                    )
                mode_names, face_disp, span_axis_used_modal = _build_face_displacements_from_beam(
                    face_centers=face_centers,
                    modal_dir=modal_dir,
                    prefix=prefix,
                    mode_indices=mode_indices,
                    span_axis=span_axis_modal,
                    normalize_phi=not bool(args.raw_eigenvector),
                    disp_scale=float(args.disp_scale),
                    cfg=cfg,
                    align_beam_to_hydro_mesh=not bool(args.skip_hydro_beam_alignment),
                )

            assert face_disp is not None
            n_modes, n_faces, _ = face_disp.shape
            if int(body.mesh.nb_faces) != n_faces:
                raise ValueError(
                    "Modal displacement / mesh mismatch: "
                    f"modes file has n_faces={n_faces}, mesh has nb_faces={body.mesh.nb_faces}."
                )
            for im, mode_name in enumerate(mode_names):
                body.dofs[mode_name] = face_disp[im]

            print(f"Loaded modes: {n_modes} -> {mode_names}")

            if args.plot_mode_projection_debug:
                sfx_mp = f"_depth_{depth:g}".replace(".", "p").replace("-", "m")
                print("Mode projection debug plots (--plot-mode-projection-debug) …")
                _plot_modal_projection_on_mesh(
                    mesh=body.mesh,
                    face_disp=face_disp,
                    mode_names=mode_names,
                    out_dir=out_dir,
                    depth_suffix=sfx_mp,
                    quiver_scale=float(args.mode_projection_quiver_scale),
                    max_arrows=int(args.mode_projection_max_arrows),
                    show=bool(args.show_mode_projection_debug),
                )

        if mode_names_ref is None:
            mode_names_ref = mode_names
        elif mode_names_ref != mode_names:
            raise ValueError("Mode names changed across depth sweep; expected identical modal basis.")

        print(f"Loaded mesh faces: {body.mesh.nb_faces}")
        print(f"Mesh z-translation applied: {-depth:.6g} m")
        if not args.no_debug_geometry_plot:
            suffix = f"_depth_{depth:g}".replace(".", "p").replace("-", "m")
            dbg_path = os.path.join(out_dir, f"debug_mesh_free_surface{suffix}.png")
            _plot_mesh_free_surface_debug(
                mesh=body.mesh,
                free_surface=free_surface,
                water_depth=water_depth,
                out_path=dbg_path,
                show=bool(args.show_debug_geometry),
            )
            print(f"Saved debug geometry plot: {dbg_path}")

        solver = cpt.BEMSolver()
        problems = []
        for w in omega:
            for mode_name in mode_names:
                problems.append(
                    cpt.RadiationProblem(
                        body=body,
                        radiating_dof=mode_name,
                        omega=float(w),
                        rho=rho,
                        water_depth=water_depth,
                        free_surface=free_surface,
                    )
                )

        print(f"Solving {len(problems)} radiation problems...")
        results = solver.solve_all(problems)
        ds = cpt.assemble_dataset(results)
        A = _as_mode_matrix(ds, "added_mass", omega, mode_names)
        B = _as_mode_matrix(ds, "radiation_damping", omega, mode_names)

        all_A.append(A)
        all_B.append(B)

        A_beam: np.ndarray | None = None
        B_beam: np.ndarray | None = None
        if rigid_use_beam_local:
            if len(mode_names) != 6:
                raise RuntimeError(
                    "Beam-local rigid transform requires 6 DOFs; "
                    f"got {len(mode_names)} modes: {mode_names}."
                )
            A_beam = _transform_rigid_matrices_global_to_beam(A, R_beam_to_global=R_beam_to_global)
            B_beam = _transform_rigid_matrices_global_to_beam(B, R_beam_to_global=R_beam_to_global)
            all_A_beam.append(A_beam)
            all_B_beam.append(B_beam)

        # Save per-depth files.
        suffix = f"_depth_{depth:g}".replace(".", "p").replace("-", "m")
        npz_path = os.path.join(out_dir, f"modal_radiation_AB{suffix}.npz")
        meta: dict = {
            "rho": rho,
            "water_depth": float(water_depth) if np.isfinite(water_depth) else np.inf,
            "free_surface": float(free_surface) if np.isfinite(free_surface) else np.inf,
            "depth": depth,
            "mesh_path": os.path.abspath(mesh_path),
            "case_name": args.case_name,
            "rigid_body_motion": rigid_body_motion,
            "capytaine_modal_span_axis_used": int(span_axis_used_modal),
        }
        if rigid_body_motion:
            meta["modes_path"] = ""
            meta["modal_dir"] = ""
            meta["modal_prefix"] = ""
            rrc = getattr(cfg, "rigid_body_rotation_center", None)
            meta["rigid_body_rotation_center_cfg"] = (
                np.asarray(rrc, dtype=float) if rrc is not None else np.array([], dtype=float)
            )
            com = getattr(body, "center_of_mass", None)
            meta["rigid_body_rotation_reference"] = (
                np.asarray(com, dtype=float) if com is not None else _mesh_area_centroid(body.mesh)
            )
            meta["pitch_deg_cfg"] = float(getattr(cfg, "pitch", 0.0))
            meta["alpha_deg_cfg"] = float(getattr(cfg, "alpha_deg", 0.0))
            meta["dihedral_deg_cfg"] = float(getattr(cfg, "dihedral_angle", 0.0))
            meta["R_beam_to_global"] = R_beam_to_global.astype(float)
            if A_beam is not None and B_beam is not None:
                meta["added_mass_beam_local"] = A_beam
                meta["added_damping_beam_local"] = B_beam
                meta["beam_local_frame_note"] = (
                    "Rigid 6x6 matrices in beam frame (chord/span/thickness before depth translation): "
                    "M_beam = T^T M_global T with T=blkdiag(R,R), v_global = R @ v_beam for translations "
                    "and angular rates. Same R as build_capytaine_mesh.py."
                )
        else:
            meta["modes_path"] = os.path.abspath(args.modes) if args.modes else ""
            meta["modal_dir"] = os.path.abspath(modal_dir) if modal_dir else ""
            meta["modal_prefix"] = prefix
            meta["face_displacements"] = face_disp

        save_modal_radiation_npz(
            path=npz_path,
            omega=omega,
            mode_names=mode_names,
            added_mass=A,
            added_damping=B,
            metadata=meta,
        )
        _save_csv_matrix(os.path.join(out_dir, f"added_mass{suffix}.csv"), omega, mode_names, A)
        _save_csv_matrix(os.path.join(out_dir, f"radiation_damping{suffix}.csv"), omega, mode_names, B)
        if not args.no_plots:
            basis = "Rigid-body" if rigid_body_motion else "Modal"
            _plot_modal_matrix_sweep(
                omega,
                mode_names,
                A,
                title=f"{basis} added mass A(omega) depth={depth:g} m",
                y_label="A_ik [kg]",
                out_path=os.path.join(out_dir, f"added_mass_vs_omega{suffix}.png"),
                mask_abs_below=ADDED_MASS_VS_OMEGA_DEPTH_ABS_FLOOR,
            )
            _plot_modal_matrix_sweep(
                omega,
                mode_names,
                B,
                title=f"{basis} radiation damping B(omega) depth={depth:g} m",
                y_label="B_ik [kg/s]",
                out_path=os.path.join(out_dir, f"radiation_damping_vs_omega{suffix}.png"),
            )
            if A_beam is not None and B_beam is not None:
                _plot_modal_matrix_sweep(
                    omega,
                    mode_names,
                    A_beam,
                    title=(
                        f"{basis} added mass A(omega) beam-local frame depth={depth:g} m "
                        "(same DOF labels; axes = chord/span/thickness after pitch, α, Γ)"
                    ),
                    y_label="A_ik [kg]",
                    out_path=os.path.join(out_dir, f"added_mass_vs_omega_beam_local{suffix}.png"),
                    mask_abs_below=ADDED_MASS_VS_OMEGA_DEPTH_ABS_FLOOR,
                )
                _plot_modal_matrix_sweep(
                    omega,
                    mode_names,
                    B_beam,
                    title=(
                        f"{basis} radiation damping B(omega) beam-local frame depth={depth:g} m "
                        "(same DOF labels; axes = chord/span/thickness after pitch, α, Γ)"
                    ),
                    y_label="B_ik [kg/s]",
                    out_path=os.path.join(out_dir, f"radiation_damping_vs_omega_beam_local{suffix}.png"),
                )
                _save_csv_matrix(
                    os.path.join(out_dir, f"added_mass_beam_local{suffix}.csv"),
                    omega,
                    mode_names,
                    A_beam,
                )
                _save_csv_matrix(
                    os.path.join(out_dir, f"radiation_damping_beam_local{suffix}.csv"),
                    omega,
                    mode_names,
                    B_beam,
                )

    # Keep backward-compatible aggregate files for single-depth cases.
    if len(depth_values) == 1:
        A = all_A[0]
        B = all_B[0]
        mode_names = mode_names_ref or []
        meta_agg: dict = {
            "depth": depth_values[0],
            "case_name": args.case_name,
            "rigid_body_motion": rigid_body_motion,
            "capytaine_modal_span_axis_used": int(span_axis_used_modal),
        }
        if rigid_body_motion:
            meta_agg["pitch_deg_cfg"] = float(getattr(cfg, "pitch", 0.0))
            meta_agg["alpha_deg_cfg"] = float(getattr(cfg, "alpha_deg", 0.0))
            meta_agg["dihedral_deg_cfg"] = float(getattr(cfg, "dihedral_angle", 0.0))
            meta_agg["R_beam_to_global"] = R_beam_to_global.astype(float)
        if len(all_A_beam) == 1 and len(all_B_beam) == 1:
            meta_agg["added_mass_beam_local"] = all_A_beam[0]
            meta_agg["added_damping_beam_local"] = all_B_beam[0]
            meta_agg["beam_local_frame_note"] = (
                "Rigid 6x6 in beam frame: M_beam = T^T M_global T, T=blkdiag(R,R); same R as build_capytaine_mesh.py."
            )
        save_modal_radiation_npz(
            path=os.path.join(out_dir, "modal_radiation_AB.npz"),
            omega=omega,
            mode_names=mode_names,
            added_mass=A,
            added_damping=B,
            metadata=meta_agg,
        )
        _save_csv_matrix(os.path.join(out_dir, "added_mass.csv"), omega, mode_names, A)
        _save_csv_matrix(os.path.join(out_dir, "radiation_damping.csv"), omega, mode_names, B)
        if len(all_A_beam) == 1:
            _save_csv_matrix(
                os.path.join(out_dir, "added_mass_beam_local.csv"), omega, mode_names, all_A_beam[0]
            )
            _save_csv_matrix(
                os.path.join(out_dir, "radiation_damping_beam_local.csv"),
                omega,
                mode_names,
                all_B_beam[0],
            )

    # Multi-depth combined plots.
    if not args.no_plots and len(depth_values) > 1:
        mode_names = mode_names_ref or []
        basis = "Rigid-body" if rigid_body_motion else "Modal"
        _plot_multi_depth_modal_matrix_sweep(
            omega,
            mode_names,
            all_A,
            depth_values,
            title=f"{basis} added mass A(omega) for multiple depths",
            y_label="A_ik [kg]",
            out_path=os.path.join(out_dir, "added_mass_vs_omega_all_depths.png"),
            mask_rel_noise_below=ADDED_MASS_MULTIDEPTH_REL_NOISE_FLOOR,
        )
        _plot_multi_depth_modal_matrix_sweep(
            omega,
            mode_names,
            all_B,
            depth_values,
            title=f"{basis} radiation damping B(omega) for multiple depths",
            y_label="B_ik [kg/s]",
            out_path=os.path.join(out_dir, "radiation_damping_vs_omega_all_depths.png"),
        )
        if len(all_A_beam) == len(depth_values) and len(all_B_beam) == len(depth_values):
            _plot_multi_depth_modal_matrix_sweep(
                omega,
                mode_names,
                all_A_beam,
                depth_values,
                title=f"{basis} added mass A(omega) beam-local frame (multiple depths)",
                y_label="A_ik [kg]",
                out_path=os.path.join(out_dir, "added_mass_vs_omega_all_depths_beam_local.png"),
                mask_rel_noise_below=ADDED_MASS_MULTIDEPTH_REL_NOISE_FLOOR,
            )
            _plot_multi_depth_modal_matrix_sweep(
                omega,
                mode_names,
                all_B_beam,
                depth_values,
                title=f"{basis} radiation damping B(omega) beam-local frame (multiple depths)",
                y_label="B_ik [kg/s]",
                out_path=os.path.join(out_dir, "radiation_damping_vs_omega_all_depths_beam_local.png"),
            )
        print(f"Saved multi-depth plots in: {out_dir}")

    print("Done.")


if __name__ == "__main__":
    main()