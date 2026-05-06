"""
Run Capytaine radiation problems for user-defined modal DOFs.

Expected modal input file (.npz):
    - mode_names: array-like of shape (n_modes,) with mode labels (optional)
    - face_displacements: shape (n_modes, n_faces, 3) [m]

This script:
    1) Loads a hull mesh.
    2) Injects each modal displacement field as a custom DOF in ``body.dofs``.
    3) Solves radiation for each (mode, omega).
    4) Saves modal added mass A(omega) and damping B(omega).

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
from modal_radiation_io import save_modal_radiation_npz  # noqa: E402


def _parse_omega_args(args: argparse.Namespace, cfg) -> np.ndarray:
    if args.omega_list:
        omega = np.asarray([float(x) for x in args.omega_list.split(",")], dtype=float)
    elif hasattr(cfg, "omega_list"):
        omega = np.asarray(cfg.omega_list, dtype=float).ravel()
    elif hasattr(cfg, "omega_range"):
        omega = np.asarray(cfg.omega_range, dtype=float).ravel()
    else:
        raise ValueError("No omega definition found. Provide --omega-list or set cfg.omega_range.")
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
    out = np.empty_like(points, dtype=float)
    for i in range(points.shape[0]):
        p = points[i]
        i_lo, i_hi, n1, n2 = _span_interp(float(p[span_axis]), beam_pos, span_axis)
        q_lo = phi_full[i_lo * 6 : i_lo * 6 + 6]
        q_hi = phi_full[i_hi * 6 : i_hi * 6 + 6]
        u = n1 * q_lo[:3] + n2 * q_hi[:3]
        th = n1 * q_lo[3:6] + n2 * q_hi[3:6]
        x_ea = n1 * beam_pos[i_lo] + n2 * beam_pos[i_hi]
        out[i] = u + np.cross(th, p - x_ea)
    return out


def _build_face_displacements_from_beam(
    *,
    face_centers: np.ndarray,
    modal_dir: str,
    prefix: str,
    mode_indices: list[int],
    span_axis: int,
    normalize_phi: bool,
    disp_scale: float,
) -> tuple[list[str], np.ndarray]:
    nodes_csv = os.path.join(modal_dir, f"{prefix}_nodes.csv")
    constrained_csv = os.path.join(modal_dir, f"{prefix}_constrained_dofs.csv")
    eigendata_csv = os.path.join(modal_dir, f"{prefix}_eigendata.csv")
    for p in (nodes_csv, constrained_csv, eigendata_csv):
        if not os.path.isfile(p):
            raise FileNotFoundError(p)

    beam_pos = _load_beam_positions(nodes_csv)
    constrained = _constrained_global_dofs(constrained_csv)

    disp_all = []
    mode_names = []
    for mode_idx in mode_indices:
        phi = _load_phi_full_row(eigendata_csv, beam_pos.shape[0], constrained, mode_idx)
        phi_max = float(np.max(np.abs(phi)))
        if normalize_phi and phi_max > 0.0 and np.isfinite(phi_max):
            phi = phi / phi_max
        d = _compute_displacements(face_centers, beam_pos, phi, span_axis=span_axis)
        disp_all.append(disp_scale * d)
        mode_names.append(f"mode_{mode_idx + 1}")

    return mode_names, np.asarray(disp_all, dtype=float)


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
    out_path: str):
    import matplotlib.pyplot as plt

    n_omega, n_mode_i, n_mode_k = mat.shape
    if n_omega != omega.size:
        raise ValueError("Inconsistent omega size in plotting.")

    fig, axes = plt.subplots(n_mode_i, n_mode_k, figsize=(3.8 * n_mode_k, 3.0 * n_mode_i), squeeze=False)
    for i in range(n_mode_i):
        for k in range(n_mode_k):
            ax = axes[i][k]
            ax.plot(omega, mat[:, i, k], lw=1.8)
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
            for depth, mat in zip(depth_values, mats_by_depth):
                if mat.shape != (n_omega, n_mode, n_mode):
                    raise ValueError(f"Unexpected matrix shape in multi-depth plot: {mat.shape}")
                ax.plot(omega, mat[:, i, k], lw=1.6, label=f"depth={depth:g} m")
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


def _defaults_from_config(cfg, case_name: str) -> dict:
    blade_name = getattr(cfg, "blade_name", case_name)
    fluid = getattr(cfg, "fluid", "water")
    rho = float(getattr(cfg, "rho_f", {}).get(fluid, 997.0))
    mode_list = getattr(cfg, "modes_to_analyze", list(range(int(getattr(cfg, "num_modes_flutter_egv", 1)))))

    return {
        "mesh": os.path.join(_FSI_ROOT, "FLUID", "capytaine", f"{blade_name}_mesh.vtu"),
        "modal_dir": os.path.join(_FSI_ROOT, "output_data", case_name),
        "prefix": f"{case_name}_dry_egv",
        "mode_indices": [int(m) for m in mode_list],
        "rho": rho,
        "water_depth": "inf",
        "free_surface_elevation": getattr(cfg, "free_surface_elevation", 0.0),
        "depth": getattr(cfg, "depth", 0.0),
        "out_dir": os.path.join(_FSI_ROOT, "FLUID", "capytaine", "results_modal_radiation", case_name),
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
    parser.add_argument("--modes", type=str, default="", help="NPZ file with precomputed face_displacements.")
    parser.add_argument(
        "--mode-indices",
        type=str,
        default="",
        help="Comma-separated 0-based mode indices override, e.g. 0,1,2.",
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
    args = parser.parse_args(list(argv) if argv is not None else None)

    cfg = get_config(args.case_name)
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

    import capytaine as cpt

    print(f"Case: {args.case_name}")
    print(f"Depth sweep: {depth_values}")
    print(
        "Free surface: "
        + ("disabled (inf)" if np.isinf(free_surface) else f"z={free_surface:.6g} m")
    )
    print(f"Omega samples: {omega.size} from {omega.min():.6g} to {omega.max():.6g} rad/s")

    all_A: list[np.ndarray] = []
    all_B: list[np.ndarray] = []
    mode_names_ref: list[str] | None = None

    for idepth, depth in enumerate(depth_values):
        print(f"\n--- Solving depth = {depth:g} m ---")
        mesh = _resolve_mesh_loader(cpt, mesh_path)
        _translate_mesh_z(mesh, -depth)
        body = _build_floating_body(cpt, mesh, name=f"modal_body_d{idepth}")
        face_centers = np.asarray(body.mesh.faces_centers, dtype=float)

        if args.modes:
            mode_names, face_disp = _load_modal_displacements(args.modes)
        else:
            mode_indices = _parse_mode_indices(args.mode_indices) if args.mode_indices else defaults["mode_indices"]
            if not mode_indices:
                raise ValueError("No mode indices defined. Set cfg.modes_to_analyze or pass --mode-indices.")
            mode_names, face_disp = _build_face_displacements_from_beam(
                face_centers=face_centers,
                modal_dir=modal_dir,
                prefix=prefix,
                mode_indices=mode_indices,
                span_axis=int(args.span_axis),
                normalize_phi=not bool(args.raw_eigenvector),
                disp_scale=float(args.disp_scale),
            )

        n_modes, n_faces, _ = face_disp.shape
        if int(body.mesh.nb_faces) != n_faces:
            raise ValueError(
                "Modal displacement / mesh mismatch: "
                f"modes file has n_faces={n_faces}, mesh has nb_faces={body.mesh.nb_faces}."
            )
        for im, mode_name in enumerate(mode_names):
            body.dofs[mode_name] = face_disp[im]

        if mode_names_ref is None:
            mode_names_ref = mode_names
        elif mode_names_ref != mode_names:
            raise ValueError("Mode names changed across depth sweep; expected identical modal basis.")

        print(f"Loaded mesh faces: {body.mesh.nb_faces}")
        print(f"Loaded modes: {n_modes} -> {mode_names}")
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

        # Save per-depth files.
        suffix = f"_depth_{depth:g}".replace(".", "p").replace("-", "m")
        npz_path = os.path.join(out_dir, f"modal_radiation_AB{suffix}.npz")
        save_modal_radiation_npz(
            path=npz_path,
            omega=omega,
            mode_names=mode_names,
            added_mass=A,
            added_damping=B,
            metadata={
                "rho": rho,
                "water_depth": float(water_depth) if np.isfinite(water_depth) else np.inf,
                "free_surface": float(free_surface) if np.isfinite(free_surface) else np.inf,
                "depth": depth,
                "mesh_path": os.path.abspath(mesh_path),
                "modes_path": os.path.abspath(args.modes) if args.modes else "",
                "modal_dir": os.path.abspath(modal_dir) if modal_dir else "",
                "modal_prefix": prefix,
                "case_name": args.case_name,
                "face_displacements": face_disp,
            },
        )
        _save_csv_matrix(os.path.join(out_dir, f"added_mass{suffix}.csv"), omega, mode_names, A)
        _save_csv_matrix(os.path.join(out_dir, f"radiation_damping{suffix}.csv"), omega, mode_names, B)
        if not args.no_plots:
            _plot_modal_matrix_sweep(
                omega,
                mode_names,
                A,
                title=f"Modal added mass A(omega) depth={depth:g} m",
                y_label="A_ik [kg]",
                out_path=os.path.join(out_dir, f"added_mass_vs_omega{suffix}.png"),
            )
            _plot_modal_matrix_sweep(
                omega,
                mode_names,
                B,
                title=f"Modal radiation damping B(omega) depth={depth:g} m",
                y_label="B_ik [kg/s]",
                out_path=os.path.join(out_dir, f"radiation_damping_vs_omega{suffix}.png"),
            )

    # Keep backward-compatible aggregate files for single-depth cases.
    if len(depth_values) == 1:
        A = all_A[0]
        B = all_B[0]
        mode_names = mode_names_ref or []
        save_modal_radiation_npz(
            path=os.path.join(out_dir, "modal_radiation_AB.npz"),
            omega=omega,
            mode_names=mode_names,
            added_mass=A,
            added_damping=B,
            metadata={"depth": depth_values[0], "case_name": args.case_name},
        )
        _save_csv_matrix(os.path.join(out_dir, "added_mass.csv"), omega, mode_names, A)
        _save_csv_matrix(os.path.join(out_dir, "radiation_damping.csv"), omega, mode_names, B)

    # Multi-depth combined plots.
    if not args.no_plots and len(depth_values) > 1:
        mode_names = mode_names_ref or []
        _plot_multi_depth_modal_matrix_sweep(
            omega,
            mode_names,
            all_A,
            depth_values,
            title="Modal added mass A(omega) for multiple depths",
            y_label="A_ik [kg]",
            out_path=os.path.join(out_dir, "added_mass_vs_omega_all_depths.png"),
        )
        _plot_multi_depth_modal_matrix_sweep(
            omega,
            mode_names,
            all_B,
            depth_values,
            title="Modal radiation damping B(omega) for multiple depths",
            y_label="B_ik [kg/s]",
            out_path=os.path.join(out_dir, "radiation_damping_vs_omega_all_depths.png"),
        )
        print(f"Saved multi-depth plots in: {out_dir}")

    print("Done.")


if __name__ == "__main__":
    main()
