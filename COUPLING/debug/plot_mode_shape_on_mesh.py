#!/usr/bin/env python3
"""
Debug: visualize a dry eigenmode on a Capytaine / VTK surface mesh.

Loads:
  - Surface mesh (``.vtu`` / ``.vtk`` / etc.) via meshio
  - Beam modal CSVs written by ``FEA.fea_utl.analysis._save_modal_data_to_csv``:
      ``*_dry_egv_nodes.csv``, ``*_dry_egv_constrained_dofs.csv``, ``*_dry_egv_eigendata.csv``

For each mesh vertex, kinematics are interpolated along beam span (``Y`` by default), identical in
spirit to ``mesh_coupling_matrix.HydroMeshSurfaceToFEM``:

    u, θ linear between bracketing beam nodes
    x_def = x + scale * ( u + θ × (x - x_ea) )

Eigenvectors saved by the FEA pipeline are typically **mass-normalized**: rotational DOF entries can be
``O(10²)`` while translations are ``O(10⁻³)``. Feeding those straight into ``θ × (x - x_ea)`` treats
``θ`` as radians, so the cross term dominates and the surface looks **ballooned** (sections appear to
shear/stretch). By default this script **rescales** ``φ`` so ``max |φ_i| = 1`` before applying the
formula; translations and rotations stay in proportion so each span station remains a rigid motion of
the section. Use ``--raw-eigenvector`` to skip normalization for debugging.

Example::

    cd FSI
    python COUPLING/debug/plot_mode_shape_on_mesh.py \\
      --mesh FLUID/capytaine/NACA0003_mesh.vtu \\
      --modal-dir output_data/NACA0003 \\
      --prefix NACA0003_dry_egv \\
      --scale 0.5 \\
      --mesh-translate 0 0 100 \\
      --save COUPLING/debug/NACA0003_mode1_mesh.png

Meshes rebuilt with ``FLUID/capytaine/build_capytaine_mesh.py`` default to ``z = 0`` (beam frame).
Older VTUs produced when the builder applied ``config.offset_z`` may still need
``--mesh-translate 0 0 100``. Legacy shifted hulls: rebuild without ``--use-config-offset-z``.

Requires: ``pip install meshio matplotlib``
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

# ── repo root on PYTHONPATH (run from FSI/) ──────────────────────────────────
_FSI_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _FSI_ROOT not in sys.path:
    sys.path.insert(0, _FSI_ROOT)


def load_beam_positions(nodes_csv: str) -> np.ndarray:
    """``positions[i]`` = [x,y,z] for ``Node_Index == i``."""
    rows = []
    with open(nodes_csv, newline="") as f:
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
    if rows[-1][0] != len(rows) - 1:
        raise ValueError(f"Non-contiguous Node_Index in {nodes_csv}")
    return np.array([x[1:] for x in rows], dtype=float)


def constrained_global_dofs(constrained_csv: str) -> set[int]:
    """Interpret ``Constrained_DOF_Indices`` as *local* 0..5 per constrained row."""
    out: set[int] = set()
    with open(constrained_csv, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            ni = int(row["Node_Index"])
            local = [int(x.strip()) for x in row["Constrained_DOF_Indices"].split(",")]
            for li in local:
                out.add(ni * 6 + li)
    return out


def load_phi_full_row(eigendata_csv: str, n_nodes: int, constrained: set[int], mode_row: int) -> np.ndarray:
    """
    Reconstruct full-length eigenvector for one mode.

    ``mode_row``: 0-based index into eigenmodes **after** the header row
                  (``0`` = first mode, frequency column Mode == 1).
    """
    total_dof = n_nodes * 6
    free_dofs = sorted(set(range(total_dof)) - constrained)
    with open(eigendata_csv, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        if len(header) < 5:
            raise ValueError(f"Unexpected eigendata header in {eigendata_csv}")
        n_modes_seen = 0
        for row in reader:
            if len(row) < 5:
                continue
            if n_modes_seen == mode_row:
                vals = np.array(row[4 : 4 + len(free_dofs)], dtype=float)
                if vals.size != len(free_dofs):
                    raise ValueError(
                        f"Mode row length mismatch: expected {len(free_dofs)} dof coeffs, "
                        f"got {vals.size} (check CSV integrity)."
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


def deform_points(points: np.ndarray, beam_pos: np.ndarray, phi_full: np.ndarray, scale: float, span_axis: int):
    """First-order rigid kinematics from interpolated beam DOFs."""
    out = np.empty_like(points, dtype=float)
    for i in range(points.shape[0]):
        v = points[i]
        y = float(v[span_axis])
        i_lo, i_hi, n1, n2 = _span_interp(y, beam_pos, span_axis)
        sl_lo = slice(i_lo * 6, i_lo * 6 + 6)
        sl_hi = slice(i_hi * 6, i_hi * 6 + 6)
        q_lo = phi_full[sl_lo]
        q_hi = phi_full[sl_hi]
        u = n1 * q_lo[:3] + n2 * q_hi[:3]
        th = n1 * q_lo[3:6] + n2 * q_hi[3:6]
        x_ea = n1 * beam_pos[i_lo] + n2 * beam_pos[i_hi]
        r = v - x_ea
        out[i] = v + scale * (u + np.cross(th, r))
    return out


def meshio_faces(mesh) -> list[list[int]]:
    """Flatten meshio cell blocks into face vertex index lists."""
    faces: list[list[int]] = []
    for cb in mesh.cells:
        data = np.asarray(cb.data, dtype=np.int64)
        if cb.type == "triangle":
            for tri in data:
                faces.append([int(tri[0]), int(tri[1]), int(tri[2])])
        elif cb.type == "quad":
            for q in data:
                faces.append([int(q[0]), int(q[1]), int(q[2]), int(q[3])])
        else:
            # skip wedges etc.
            continue
    return faces


def _warn_frame_mismatch(pts: np.ndarray, beam_pos: np.ndarray) -> None:
    dm = np.median(pts, axis=0) - np.median(beam_pos, axis=0)
    if np.linalg.norm(dm) > 0.25:
        print(
            "Warning: median(mesh) − median(beam) = "
            f"[{dm[0]:.4f}, {dm[1]:.4f}, {dm[2]:.4f}] m. "
            "If the plot looks wrong, shift the mesh into the beam frame, e.g.\n"
            f"  --mesh-translate {-dm[0]:.6f} {-dm[1]:.6f} {-dm[2]:.6f}"
        )


def _configure_matplotlib_backend(*, save_path: str | None, show: bool, mpl_backend: str | None) -> None:
    """Pick Agg only for save-only runs; otherwise ensure a GUI backend for ``plt.show()``."""
    import matplotlib

    if save_path and not show:
        matplotlib.use("Agg")
        return

    if not show:
        return

    if mpl_backend:
        matplotlib.use(mpl_backend)
        return

    name = matplotlib.get_backend().lower()
    if "agg" in name:
        for candidate in ("TkAgg", "Qt5Agg", "QtAgg"):
            try:
                matplotlib.use(candidate)
                print(f"Interactive plot: switched matplotlib backend to {candidate!r} (was non-interactive).")
                return
            except Exception:
                continue
        print(
            "Warning: could not enable an interactive backend automatically. "
            "Try:  python ... --mpl-backend TkAgg   or   export MPLBACKEND=TkAgg"
        )


def plot_mesh_mode(
    mesh_path: str,
    modal_dir: str,
    prefix: str,
    mode_index: int,
    scale: float,
    span_axis: int,
    save_path: str | None,
    show: bool,
    mesh_translate: np.ndarray | None = None,
    mpl_backend: str | None = None,
    normalize_phi: bool = True,
):
    _configure_matplotlib_backend(save_path=save_path, show=show, mpl_backend=mpl_backend)

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    try:
        import meshio
    except ImportError as e:
        raise SystemExit("Install meshio:  pip install meshio") from e

    nodes_csv = os.path.join(modal_dir, f"{prefix}_nodes.csv")
    constrained_csv = os.path.join(modal_dir, f"{prefix}_constrained_dofs.csv")
    eigendata_csv = os.path.join(modal_dir, f"{prefix}_eigendata.csv")

    for p in (mesh_path, nodes_csv, constrained_csv, eigendata_csv):
        if not os.path.isfile(p):
            raise FileNotFoundError(p)

    beam_pos = load_beam_positions(nodes_csv)
    n_nodes = beam_pos.shape[0]
    constrained = constrained_global_dofs(constrained_csv)
    phi = load_phi_full_row(eigendata_csv, n_nodes, constrained, mode_index)
    phi_max = float(np.max(np.abs(phi)))
    if normalize_phi and phi_max > 0.0 and np.isfinite(phi_max):
        print(
            f"Eigenvector rescaled by 1/max|φ| (max|φ|={phi_max:.6g}). "
            "CSV modes are usually mass-normalized — disable with --raw-eigenvector."
        )
        phi = phi / phi_max

    mesh = meshio.read(mesh_path)
    pts = np.asarray(mesh.points, dtype=float)
    faces = meshio_faces(mesh)
    if not faces:
        raise RuntimeError(f"No triangle/quad cells found in {mesh_path}")

    trans = np.zeros(3, dtype=float) if mesh_translate is None else np.asarray(mesh_translate, dtype=float).reshape(3)
    if np.any(trans != 0.0):
        print(f"Applying mesh translation [dx,dy,dz] = {trans.tolist()} (before modal deformation)")
    else:
        _warn_frame_mismatch(pts, beam_pos)

    pts = pts + trans
    pts_def = deform_points(pts, beam_pos, phi, scale=scale, span_axis=span_axis)

    polys_orig = []
    polys_def = []
    for f in faces:
        polys_orig.append(pts[np.asarray(f)])
        polys_def.append(pts_def[np.asarray(f)])

    fig = plt.figure(figsize=(14, 6))

    ax0 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1 = fig.add_subplot(1, 2, 2, projection="3d")

    coll0 = Poly3DCollection(
        polys_orig, alpha=0.35, facecolor="lightgray", edgecolor="dimgray", linewidths=0.15
    )
    coll1 = Poly3DCollection(
        polys_def, alpha=0.85, facecolor="coral", edgecolor="black", linewidths=0.2
    )
    ax0.add_collection3d(coll0)
    ax1.add_collection3d(coll1)

    for ax in (ax0, ax1):
        ax.scatter(
            beam_pos[:, 0],
            beam_pos[:, 1],
            beam_pos[:, 2],
            c="blue",
            s=8,
            alpha=0.5,
            zorder=10,
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_box_aspect([1, 1, 1])

    ax0.set_title("Undeformed mesh + beam nodes")
    ax1.set_title(f"Mode {mode_index + 1} × scale={scale}")

    # Left: undeformed mesh + beam readable; right: include deformed extent
    hull0 = np.vstack([pts, beam_pos])
    pad0 = 0.08 * (hull0.max(axis=0) - hull0.min(axis=0) + 1e-9)
    lo0 = hull0.min(axis=0) - pad0
    hi0 = hull0.max(axis=0) + pad0

    hull1 = np.vstack([pts, pts_def, beam_pos])
    pad1 = 0.08 * (hull1.max(axis=0) - hull1.min(axis=0) + 1e-9)
    lo1 = hull1.min(axis=0) - pad1
    hi1 = hull1.max(axis=0) + pad1

    ax0.set_xlim(lo0[0], hi0[0])
    ax0.set_ylim(lo0[1], hi0[1])
    ax0.set_zlim(lo0[2], hi0[2])
    ax1.set_xlim(lo1[0], hi1[0])
    ax1.set_ylim(lo1[1], hi1[1])
    ax1.set_zlim(lo1[2], hi1[2])

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved figure: {save_path}")
    if show:
        plt.show(block=True)
    else:
        plt.close(fig)


def main():
    _epilog = """
Shell notes:
  • Flags are written --mesh, --save, etc. Do not put a backslash before the dashes.
  • For multi-line commands in bash, put one backslash only at the end of each continued line.

Examples:
  python plot_mode_shape_on_mesh.py --save out.png

  # Interactive 3D window (mouse: rotate / zoom). Short flag -i same as --show --interactive
  python plot_mode_shape_on_mesh.py -i --mesh-translate 0 0 100

  # Save PNG, then open interactive window
  python plot_mode_shape_on_mesh.py --mesh-translate 0 0 100 --save out.png -i

  python plot_mode_shape_on_mesh.py \\
    --mesh ../FLUID/capytaine/NACA0003_mesh.vtu \\
    --modal-dir ../../output_data/NACA0003 \\
    --prefix NACA0003_dry_egv --scale 0.5 \\
    --mesh-translate 0 0 100 \\
    --save NACA0003_mode_mesh.png
"""
    ap = argparse.ArgumentParser(
        description="Plot dry eigenmode on VTU/VTK surface mesh.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_epilog,
    )
    ap.add_argument(
        "--mesh",
        default=os.path.join(_FSI_ROOT, "FLUID", "capytaine", "NACA0003_mesh.vtu"),
        help="Path to mesh file (vtu, vtk, …)",
    )
    ap.add_argument(
        "--modal-dir",
        default=os.path.join(_FSI_ROOT, "output_data", "NACA0003"),
        help="Directory containing *_dry_egv_*.csv",
    )
    ap.add_argument(
        "--prefix",
        default="NACA0003_dry_egv",
        help="CSV filename prefix (before _nodes.csv, _eigendata.csv, …)",
    )
    ap.add_argument("--mode", type=int, default=0, help="0-based mode index (0 = first mode)")
    ap.add_argument("--scale", type=float, default=0.5, help="Visual amplification factor")
    ap.add_argument(
        "--raw-eigenvector",
        action="store_true",
        help="Use CSV eigenvector amplitudes as-is (mass-normalized modes blow up θ×r). Default: rescale max|φ|=1.",
    )
    ap.add_argument("--span-axis", type=int, default=1, help="Beam span axis (0=X, 1=Y, 2=Z)")
    ap.add_argument(
        "--save",
        default="",
        help="PNG output path (optional). Example: COUPLING/debug/mode1_mesh.png",
    )
    ap.add_argument(
        "--show",
        "-i",
        "--interactive",
        action="store_true",
        dest="show_plot",
        help="Open interactive 3D window after plotting (mouse drag to rotate). Combine with --save to write PNG then show.",
    )
    ap.add_argument(
        "--mpl-backend",
        default=None,
        metavar="NAME",
        help="Force matplotlib GUI backend before plotting, e.g. TkAgg, Qt5Agg (use if no window appears).",
    )
    ap.add_argument(
        "--mesh-translate",
        nargs=3,
        type=float,
        default=None,
        metavar=("DX", "DY", "DZ"),
        help="Translate mesh vertices before deformation [m] if VTU frame ≠ beam CSV (legacy offset meshes).",
    )
    args = ap.parse_args()

    save_path = args.save if args.save else None
    # Interactive when explicitly requested, or when not saving only to file (no --save).
    show_plot = bool(args.show_plot) or (save_path is None)

    plot_mesh_mode(
        mesh_path=args.mesh,
        modal_dir=args.modal_dir,
        prefix=args.prefix,
        mode_index=args.mode,
        scale=args.scale,
        span_axis=args.span_axis,
        save_path=save_path,
        show=show_plot,
        mesh_translate=(
            np.array(args.mesh_translate, dtype=float) if args.mesh_translate is not None else None
        ),
        mpl_backend=args.mpl_backend,
        normalize_phi=not args.raw_eigenvector,
    )


if __name__ == "__main__":
    main()
