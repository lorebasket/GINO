"""
Build a watertight quadrilateral surface mesh for the NACA0003 example geometry
using Capytaine's Mesh API (vertices + faces).

Coordinate convention (STRUCTURE/FEA beam model / ``create_beam_model``):
    X : chordwise; section shifted so ``xea_factor * chord`` lies at ``X = 0`` (elastic-axis line).
    Y : span from ``0`` to ``beam_length`` (same as beam node ``y`` coordinates).
    Z : airfoil thickness; mid-thickness / chord line at ``Z = 0`` for symmetric sections.

By default the mesh is **not** shifted in ``Z``: beam nodes sit at ``z = 0``, matching ``offset_z``
ignored here for geometry (legacy hydro pipelines used ``offset_z = -100`` only for wavemakers /
numerics — use ``--use-config-offset-z`` if you still want that translation on the hull).

See: https://capytaine.org/stable/user_manual/mesh.html

Usage (from FSI root with sonata-env active):
    python FLUID/capytaine/build_capytaine_mesh.py

Preview: ``--show`` uses VTK (needs ``vtk`` + a working DISPLAY / GUI session). If no window
appears (Wayland, SSH, IDE terminal), use ``--matplotlib`` or open the written ``.vtu`` in ParaView.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

# ── FSI examples.config ────────────────────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_FSI_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
_EXAMPLES_DIR = os.path.join(_FSI_ROOT, "examples")
if _EXAMPLES_DIR not in sys.path:
    sys.path.insert(0, _EXAMPLES_DIR)

from config import get_config  # noqa: E402


def _rotation_y(angle_deg: float) -> np.ndarray:
    t = np.deg2rad(angle_deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=float)


def _resample_closed_contour(raw_coords: np.ndarray, n_points: int) -> np.ndarray:
    """
    Uniformly resample a closed 2D contour by arc length.

    Parameters
    ----------
    raw_coords : (N, 2) array
        Input section coordinates. The contour can be open (first != last) or
        explicitly closed (first == last).
    n_points : int
        Number of perimeter points to generate (without repeating the first point at end).
    """
    if n_points < 3:
        raise ValueError(f"n_points must be >= 3, got {n_points}.")

    p = np.asarray(raw_coords, dtype=float)
    if p.ndim != 2 or p.shape[1] != 2:
        raise ValueError("raw_coords must have shape (N, 2)")
    if p.shape[0] < 3:
        raise ValueError("Need at least 3 input contour points.")

    # Ensure explicit closure once, then parameterize by cumulative arc length.
    if not np.allclose(p[0], p[-1]):
        p = np.vstack([p, p[0]])

    seg = np.linalg.norm(p[1:] - p[:-1], axis=1)
    if not np.all(np.isfinite(seg)):
        raise ValueError("Non-finite segment length in contour coordinates.")
    if float(np.sum(seg)) <= 0.0:
        raise ValueError("Degenerate contour: total perimeter length is zero.")

    s = np.concatenate([[0.0], np.cumsum(seg)])
    s_target = np.linspace(0.0, s[-1], int(n_points), endpoint=False)
    x = np.interp(s_target, s, p[:, 0])
    z = np.interp(s_target, s, p[:, 1])
    return np.column_stack([x, z])


def _symmetric_section_cap_faces(base: int, ile: int, n_per: int, *, tip: bool) -> list[list[int]]:
    """
    Cap mesh using symmetry about the chord line: pair ``ile-k`` with ``ile+k`` (mirror
    stations about the LE) into quad strips instead of fanning from LE.

    Perimeter ordering: TE → upper → LE → lower → … → near TE, LE at ``argmin(x)``.

    Let ``nu = ile`` (vertices on upper strictly before LE) and
    ``nl = n_per - 1 - ile`` (vertices on lower strictly after LE).

    * If ``nu == nl`` (odd ``n_per``): strips alone close the TE at vertex ``0``.
    * If ``|nu - nl| == 1`` (even ``n_per``): one extra vertex on one TE branch; add a final
      quadrilateral ``1 — 0 — (n_per-1) — (n_per-2)`` (indices relative to cap ring).

    Larger imbalances are rejected (would need more than one TE patch).
    Capytaine triangles close with first vertex repeated (``[a,b,c,a]``).
    """
    nu = ile
    nl = n_per - 1 - ile
    if nu < 1 or nl < 1:
        raise ValueError(f"Invalid LE split: ile={ile}, n_per={n_per} (need nu>=1 and nl>=1).")
    imbalance = nu - nl
    if abs(imbalance) > 1:
        raise ValueError(
            "Symmetric TE caps support at most one extra vertex on upper or lower TE branch "
            f"(|nu-nl|<=1). Got nu={nu}, nl={nl}, n_per={n_per}, ile={ile}."
        )

    def iv(j: int) -> int:
        return base + j

    faces: list[list[int]] = []

    # Leading-edge wedge between first mirrored neighbours of LE.
    if tip:
        faces.append([iv(ile), iv(ile - 1), iv(ile + 1), iv(ile)])
    else:
        faces.append([iv(ile), iv(ile + 1), iv(ile - 1), iv(ile)])

    # Quad strips while both sides have symmetric stations (same k on upper and lower).
    n_pair = min(nu, nl)
    for k in range(2, n_pair + 1):
        um, up = iv(ile - k), iv(ile - (k - 1))
        lm, lp = iv(ile + k), iv(ile + (k - 1))
        if tip:
            faces.append([up, um, lm, lp])
        else:
            faces.append([up, lp, lm, um])

    # Single TE quadrilateral when lower and upper branch counts differ by one.
    if nu != nl:
        up, lp, lm, um = iv(1), iv(n_per - 2), iv(n_per - 1), iv(0)
        if tip:
            faces.append([up, um, lm, lp])
        else:
            faces.append([up, lp, lm, um])

    return faces


def build_naca0003_mesh(
    *,
    beam_length: float,
    chord: float,
    raw_coords: np.ndarray,
    xea_factor: float = 0.5,
    n_span: int = 24,
    n_chord: int | None = None,
    alpha_deg: float = 0.0,
    dihedral_deg: float = 0.0,
    offset: np.ndarray | None = None,
    name: str = "NACA0003_mesh"):
    """
    Loft the closed airfoil polygon along span and close with root/tip caps.

    Parameters
    ----------
    raw_coords : (N, 2) array
        Section polygon in chord-normalized coordinates (x/c, y/c), first and
        last row typically duplicate the trailing edge (closed loop).
    n_span : int
        Number of spanwise segments (stations = n_span + 1).
    """
    import capytaine as cpt

    raw_coords = np.asarray(raw_coords, dtype=float)
    if raw_coords.ndim != 2 or raw_coords.shape[1] != 2:
        raise ValueError("raw_coords must have shape (N, 2)")

    if n_chord is not None:
        perimeter = _resample_closed_contour(raw_coords, int(n_chord))
    else:
        perimeter = raw_coords[:-1].copy()
    n_per = perimeter.shape[0]
    if n_per < 3:
        raise ValueError("Need at least 3 perimeter points")

    xea = float(xea_factor) * chord

    y_sta = np.linspace(0.0, beam_length, n_span + 1)
    verts_list = []
    for y in y_sta:
        x_phys = (perimeter[:, 0] * chord) - xea
        z_phys = perimeter[:, 1] * chord
        sec = np.column_stack([x_phys, np.full(n_per, y), z_phys])
        verts_list.append(sec)
    vertices = np.vstack(verts_list)

    # Optional dihedral (rotation about span axis Y), then angle of attack like rotate_beam_model_y
    R_tot = _rotation_y(alpha_deg) @ _rotation_y(dihedral_deg)
    vertices = np.einsum("ij,nj->ni", R_tot, vertices)

    if offset is not None:
        vertices = vertices + np.asarray(offset, dtype=float).reshape(1, 3)

    faces = []

    # Lateral surface (quads along perimeter × span)
    for i in range(n_span):
        for j in range(n_per):
            jn = (j + 1) % n_per
            i00 = i * n_per + j
            i01 = i * n_per + jn
            i10 = (i + 1) * n_per + j
            i11 = (i + 1) * n_per + jn
            faces.append([i00, i01, i11, i10])

    ile = int(np.argmin(perimeter[:, 0]))

    # Root / tip caps: symmetric strips (pair ile-k with ile+k), not LE triangle fan.
    faces.extend(_symmetric_section_cap_faces(0, ile, n_per, tip=False))
    tip0 = n_span * n_per
    faces.extend(_symmetric_section_cap_faces(tip0, ile, n_per, tip=True))

    faces_arr = np.asarray(faces, dtype=np.int64)
    mesh = cpt.Mesh(vertices=vertices, faces=faces_arr, name=name)

    # Do not call heal_normals here: compute_connectivity assumes a conformal
    # manifold; foil meshes often have TE-like topology where an edge is shared
    # by more than two panels, which raises RuntimeError.
    return mesh


def build_ABRAMSON1965_mesh(
    *,
    beam_length: float,
    chord: float,
    raw_coords: np.ndarray,
    xea_factor: float = 0.5,
    n_span: int = 24,
    n_chord: int | None = None,
    alpha_deg: float = 0.0,
    dihedral_deg: float = 0.0,
    offset: np.ndarray | None = None,
    name: str = "ABRAMSON1965_mesh"):
    """
    Loft the closed airfoil polygon along span and close with root/tip caps.

    Parameters
    ----------
    raw_coords : (N, 2) array
        Section polygon in chord-normalized coordinates (x/c, y/c), first and
        last row typically duplicate the trailing edge (closed loop).
    n_span : int
        Number of spanwise segments (stations = n_span + 1).
    """
    import capytaine as cpt

    raw_coords = np.asarray(raw_coords, dtype=float)
    if raw_coords.ndim != 2 or raw_coords.shape[1] != 2:
        raise ValueError("raw_coords must have shape (N, 2)")

    if n_chord is not None:
        perimeter = _resample_closed_contour(raw_coords, int(n_chord))
    else:
        perimeter = raw_coords[:-1].copy()
    n_per = perimeter.shape[0]
    if n_per < 3:
        raise ValueError("Need at least 3 perimeter points")

    xea = float(xea_factor) * chord

    y_sta = np.linspace(0.0, beam_length, n_span + 1)
    verts_list = []
    for y in y_sta:
        x_phys = (perimeter[:, 0] * chord) - xea
        z_phys = perimeter[:, 1] * chord
        sec = np.column_stack([x_phys, np.full(n_per, y), z_phys])
        verts_list.append(sec)
    vertices = np.vstack(verts_list)

    # Optional dihedral (rotation about span axis Y), then angle of attack like rotate_beam_model_y
    R_tot = _rotation_y(alpha_deg) @ _rotation_y(dihedral_deg)
    vertices = np.einsum("ij,nj->ni", R_tot, vertices)

    if offset is not None:
        vertices = vertices + np.asarray(offset, dtype=float).reshape(1, 3)

    faces = []

    # Lateral surface (quads along perimeter × span)
    for i in range(n_span):
        for j in range(n_per):
            jn = (j + 1) % n_per
            i00 = i * n_per + j
            i01 = i * n_per + jn
            i10 = (i + 1) * n_per + j
            i11 = (i + 1) * n_per + jn
            faces.append([i00, i01, i11, i10])

    ile = int(np.argmin(perimeter[:, 0]))

    # Root / tip caps: symmetric strips (pair ile-k with ile+k), not LE triangle fan.
    faces.extend(_symmetric_section_cap_faces(0, ile, n_per, tip=False))
    tip0 = n_span * n_per
    faces.extend(_symmetric_section_cap_faces(tip0, ile, n_per, tip=True))

    faces_arr = np.asarray(faces, dtype=np.int64)
    mesh = cpt.Mesh(vertices=vertices, faces=faces_arr, name=name)

    # Do not call heal_normals here: compute_connectivity assumes a conformal
    # manifold; foil meshes often have TE-like topology where an edge is shared
    # by more than two panels, which raises RuntimeError.
    return mesh


def _try_export(mesh, path: str) -> None:
    """Write using Capytaine mesh_writers when extension is supported; else meshio or .npz."""
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    verts = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)

    try:
        from capytaine.io.mesh_writers import extension_dict, write_mesh

        if ext in extension_dict:
            write_mesh(path, verts, faces, ext)
            print(f"Wrote {path} (capytaine mesh_writers, format={ext})")
            return
    except Exception as e:
        print(f"Capytaine mesh writer failed ({e}); trying meshio / npz")

    try:
        import meshio

        cells = [("quad", faces)] if faces.shape[1] == 4 else [("triangle", faces[:, :3])]
        meshio.write(path, meshio.Mesh(points=verts, cells=cells))
        print(f"Wrote {path} (meshio)")
        return
    except ImportError:
        pass
    except Exception as e:
        print(f"meshio export failed ({e}); falling back to npz")

    npz_path = path if path.endswith(".npz") else path + ".npz"
    np.savez_compressed(npz_path, vertices=verts, faces=faces, name=str(mesh.name))
    print(f"Saved vertices/faces as compressed NumPy archive: {npz_path}")


def _preview_mesh(mesh, use_vtk: bool) -> None:
    """VTK window via mesh.show(), or matplotlib if VTK is missing or fails."""
    if not use_vtk:
        print("Opening matplotlib 3D view (close window to exit)...")
        mesh.show_matplotlib()
        return

    try:
        import vtk  # noqa: F401
    except ImportError:
        print(
            "Package 'vtk' not found; opening matplotlib instead. "
            "Install vtk in sonata-env for the interactive VTK viewer."
        )
        mesh.show_matplotlib()
        return

    print(
        "Opening VTK mesh viewer (requires a desktop session / DISPLAY). "
        "Press q in the viewer to quit.\n"
        "If no window appears, run with --matplotlib or open the .vtu in ParaView."
    )
    try:
        mesh.show()
    except Exception as e:
        print(f"VTK viewer failed ({e!r}). Falling back to matplotlib.")
        mesh.show_matplotlib()


def main():
    parser = argparse.ArgumentParser(description="Build Capytaine mesh for a given hydrofoil example.")
    parser.add_argument("--case_name", type=str, default="NACA0003", help="Example to build mesh for")
    parser.add_argument(
        "--n-span",
        type=int,
        default=None,
        help="Spanwise segments (override; default comes from cfg.mesh_n_span or 24).",
    )
    parser.add_argument(
        "--n-chord",
        type=int,
        default=None,
        help="Perimeter points on interpolated contour (override; default comes from cfg.mesh_n_chord).",
    )
    parser.add_argument(
        "--offset-z",
        type=float,
        default=0.0,
        metavar="DZ",
        help="Extra translation applied to all mesh vertices along Z after build [m]. "
        "Default 0 keeps the hull in the same frame as beam nodes (elastic axis at z=0).",
    )
    parser.add_argument(
        "--use-config-offset-z",
        action="store_true",
        help="Ignore --offset-z and use AnalysisConfig.offset_z instead (e.g. -100 for legacy hydro spacing).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output path (.vtu / .stl / .npz)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="VTK interactive viewer (needs vtk + DISPLAY; use --matplotlib if no window)",
    )
    parser.add_argument(
        "--matplotlib",
        action="store_true",
        help="Matplotlib 3D preview (blocks until figure closed)",
    )
    args = parser.parse_args()
    if args.out is None:
        args.out = os.path.join(_THIS_DIR, f"{args.case_name}/{args.case_name}_mesh.vtu")
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    cfg = get_config(args.case_name)
    n_span = int(args.n_span) if args.n_span is not None else int(getattr(cfg, "mesh_n_span", 24))
    n_chord = int(args.n_chord) if args.n_chord is not None else getattr(cfg, "mesh_n_chord", None)
    n_chord = int(n_chord) if n_chord is not None else None

    if args.use_config_offset_z:
        dz = float(getattr(cfg, "offset_z", 0.0))
        print(f"Mesh Z translation from config offset_z = {dz:g} m (--use-config-offset-z)")
    else:
        dz = float(args.offset_z)
        if dz != 0.0:
            print(f"Mesh Z translation from --offset-z = {dz:g} m")
        else:
            print("Mesh aligned with beam frame: Z translation 0 (elastic-axis line at z=0).")

    offset = np.array([0.0, 0.0, dz], dtype=float)

    if args.case_name == "NACA0003":
        mesh = build_naca0003_mesh(
            beam_length=float(cfg.beam_length),
            chord=float(cfg.chord),
            raw_coords=cfg.raw,
            xea_factor=float(cfg.xea_factor),
            n_span=n_span,
            n_chord=n_chord,
            alpha_deg=float(getattr(cfg, "alpha_deg", 0.0)),
            dihedral_deg=float(getattr(cfg, "dihedral_angle", 0.0)),
            offset=offset,
            name="NACA0003",
        )

    elif args.case_name == "ABRAMSON1965":
        mesh = build_ABRAMSON1965_mesh(
            beam_length=float(cfg.beam_length),
            chord=float(cfg.chord),
            raw_coords=cfg.raw,
            xea_factor=float(cfg.xea_factor),
            n_span=n_span,
            n_chord=n_chord,
            alpha_deg=float(getattr(cfg, "alpha_deg", 0.0)),
            dihedral_deg=float(getattr(cfg, "dihedral_angle", 0.0)),
            offset=offset,
            name="ABRAMSON1965",
        )
    else:
        raise ValueError(f"Mesher not implemented for case: {args.case_name}")

    body = None
    try:
        import capytaine as cpt

        body = cpt.FloatingBody(mesh=mesh, name=args.case_name)
        print(f"FloatingBody: {body}")
    except Exception as e:
        print(f"FloatingBody construction note: {e}")

    print(f"Mesh '{mesh.name}': {mesh.nb_faces} faces, {mesh.nb_vertices} vertices")

    _try_export(mesh, args.out)

    if args.matplotlib:
        _preview_mesh(mesh, use_vtk=False)
    elif args.show:
        _preview_mesh(mesh, use_vtk=True)


if __name__ == "__main__":
    main()
