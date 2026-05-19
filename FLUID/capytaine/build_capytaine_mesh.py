"""
Build a watertight surface mesh for hydrofoil examples (e.g. NACA0003, ABRAMSON1965)
using **Gmsh** to extrude the closed section from ``config.raw`` along the beam span,
then load the hull into Capytaine.

Coordinate convention (STRUCTURE/FEA beam model / ``create_beam_model``):
    X : chordwise; section shifted so ``xea_factor * chord`` lies at ``X = 0`` (elastic-axis line).
    Y : span from ``0`` to ``beam_length`` (same as beam node ``y`` coordinates).
    Z : airfoil thickness; mid-thickness / chord line at ``Z = 0`` for symmetric sections.

    After meshing, vertices are rotated: dihedral and angle of attack about **+Y** (same as
    ``rotate_beam_model_y``), then optional **structural pitch** about **+X** via
    ``config.pitch`` (deg), matching ``post_pitch_utils`` / ``rotate_beams_x``.

By default the mesh is **not** shifted in ``Z``: beam nodes sit at ``z = 0``, matching ``offset_z``
ignored here for geometry (legacy hydro pipelines used ``offset_z = -100`` only for wavemakers /
numerics — use ``--use-config-offset-z`` if you still want that translation on the hull).

See: https://capytaine.org/stable/user_manual/mesh.html

Requires the Gmsh Python API (``pip install gmsh`` / system package) and **meshio** for
reading the generated ``.msh`` into Capytaine (same as Capytaine's MSH v4 path).

Usage (from FSI root with sonata-env active):
    python FLUID/capytaine/build_capytaine_mesh.py

Preview: ``--show`` uses VTK (needs ``vtk`` + a working DISPLAY / GUI session). If no window
appears (Wayland, SSH, IDE terminal), use ``--matplotlib`` or open the written ``.vtu`` in ParaView.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile

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


def _rotation_x(angle_deg: float) -> np.ndarray:
    """Right-handed rotation about +X (structural pitch), consistent with ``rotate_beams_x``."""
    t = np.deg2rad(float(angle_deg))
    c, s = np.cos(t), np.sin(t)
    return np.array(
        [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]],
        dtype=float,
    )


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


def _default_gmsh_lc(
    perimeter: np.ndarray,
    chord: float,
    beam_length: float,
    n_span: int,
    n_chord: int | None,
) -> float:
    """Gmsh characteristic length from perimeter length, ``n_chord`` / ``n_span`` targets."""
    x = perimeter[:, 0] * float(chord)
    z = perimeter[:, 1] * float(chord)
    p2 = np.column_stack([x, z])
    if not np.allclose(p2[0], p2[-1]):
        p2 = np.vstack([p2, p2[0]])
    seg = np.linalg.norm(p2[1:] - p2[:-1], axis=1)
    perim_len = float(np.sum(seg))
    nper = int(perimeter.shape[0])
    nch = int(n_chord) if n_chord is not None else max(nper, 8)
    lc_arc = perim_len / max(2 * nch, 6)
    lc_span = float(beam_length) / max(2 * int(n_span), 4)
    return float(max(min(lc_arc, lc_span), 1e-10))


def _meshio_surface_only(msh_path: str):
    """Read a Gmsh ``.msh`` and keep only triangle/quad hull cells (drop volume elements)."""
    import meshio

    m = meshio.read(msh_path)
    surf = [cb for cb in m.cells if cb.type in ("triangle", "quad")]
    if not surf:
        raise RuntimeError(
            f"No triangle/quad cells in {msh_path!r}. "
            "Gmsh may not have meshed the hull; try a smaller --gmsh-lc."
        )
    return meshio.Mesh(points=m.points, cells=surf)


def build_extruded_airfoil_mesh_gmsh(
    *,
    beam_length: float,
    chord: float,
    raw_coords: np.ndarray,
    xea_factor: float = 0.5,
    n_span: int = 24,
    n_chord: int | None = None,
    alpha_deg: float = 0.0,
    dihedral_deg: float = 0.0,
    pitch_deg: float = 0.0,
    offset: np.ndarray | None = None,
    gmsh_lc: float | None = None,
    name: str = "hydrofoil_mesh",
):
    """
    Extrude the closed section (from ``raw_coords`` / ``cfg.raw``) along ``+Y`` with Gmsh,
    mesh the hull, and build a Capytaine surface ``Mesh``.

    Same physical frame as the previous hand-built loft: section in the ``X``–``Z`` plane at
    ``Y = 0``, scaled by ``chord`` and shifted by ``xea_factor * chord`` along ``X``, then
    dihedral / angle of attack about ``+Y`` and structural pitch about ``+X``.

    Parameters
    ----------
    raw_coords : (N, 2) array
        Section polygon in chord-normalized coordinates (x/c, z/c), first and last row
        typically duplicate the trailing edge (closed loop).
    n_span : int
        Number of spanwise segments (``n_span + 1`` stations along ``Y``).
    pitch_deg : float
        Structural pitch about **+X** [deg], same convention as ``AnalysisConfig.pitch`` and
        ``rotate_beams_x``. Applied **after** Gmsh builds the mesh in the beam frame, together
        with ``alpha_deg`` and ``dihedral_deg`` (compound rotation ``R_x @ R_y(α) @ R_y(Γ)``).
    gmsh_lc : float, optional
        Gmsh mesh size at section control points. If omitted, a default is derived from
        ``n_chord`` / ``n_span`` and the scaled perimeter length.
    """
    try:
        import gmsh
    except ImportError as e:
        raise ImportError(
            "The Gmsh Python API is required for build_extruded_airfoil_mesh_gmsh. "
            "Install with: pip install gmsh   (or use your system gmsh + Python bindings)."
        ) from e

    import capytaine as cpt
    import meshio

    raw_coords = np.asarray(raw_coords, dtype=float)
    if raw_coords.ndim != 2 or raw_coords.shape[1] != 2:
        raise ValueError("raw_coords must have shape (N, 2)")

    if n_chord is not None:
        perimeter = _resample_closed_contour(raw_coords, int(n_chord))
    else:
        perimeter = raw_coords[:-1].copy()
    n_per = int(perimeter.shape[0])
    if n_per < 3:
        raise ValueError("Need at least 3 perimeter points")

    xea = float(xea_factor) * float(chord)
    lc = float(gmsh_lc) if gmsh_lc is not None else _default_gmsh_lc(
        perimeter, chord, beam_length, n_span, n_chord
    )

    x_phys = perimeter[:, 0] * float(chord) - xea
    z_phys = perimeter[:, 1] * float(chord)

    fd, msh_path = tempfile.mkstemp(suffix=".msh", prefix="capytaine_gmsh_")
    os.close(fd)
    try:
        gmsh.initialize()
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
            gmsh.model.add("extruded_section")

            point_tags: list[int] = []
            for xi, zi in zip(x_phys, z_phys):
                point_tags.append(gmsh.model.geo.addPoint(float(xi), 0.0, float(zi), lc))

            n = len(point_tags)
            line_tags: list[int] = []
            for i in range(n):
                j = (i + 1) % n
                line_tags.append(gmsh.model.geo.addLine(point_tags[i], point_tags[j]))
            loop = gmsh.model.geo.addCurveLoop(line_tags)
            surf = gmsh.model.geo.addPlaneSurface([loop])
            gmsh.model.geo.synchronize()

            gmsh.model.geo.extrude(
                [(2, surf)],
                0.0,
                float(beam_length),
                0.0,
                numElements=[int(n_span)],
                recombine=False,
            )
            gmsh.model.geo.synchronize()

            gmsh.model.mesh.generate(2)
            gmsh.write(msh_path)
        finally:
            gmsh.finalize()

        mio = _meshio_surface_only(msh_path)
        meshio.write(msh_path, mio)
        mesh = cpt.load_mesh(msh_path)

        v = np.asarray(mesh.vertices, dtype=float)
        # Structural rotations (same as legacy hand-loft): pitch about +X, then α and Γ about +Y.
        R_tot = _rotation_x(pitch_deg) @ _rotation_y(alpha_deg) @ _rotation_y(dihedral_deg)
        v = np.einsum("ij,nj->ni", R_tot, v)
        if offset is not None:
            v = v + np.asarray(offset, dtype=float).reshape(1, 3)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        mesh = cpt.Mesh(vertices=v, faces=faces, name=name)
    finally:
        try:
            os.remove(msh_path)
        except OSError:
            pass

    return mesh


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
    pitch_deg: float = 0.0,
    offset: np.ndarray | None = None,
    gmsh_lc: float | None = None,
    name: str = "NACA0003_mesh",
):
    """Same as :func:`build_extruded_airfoil_mesh_gmsh` with default mesh name."""
    return build_extruded_airfoil_mesh_gmsh(
        beam_length=beam_length,
        chord=chord,
        raw_coords=raw_coords,
        xea_factor=xea_factor,
        n_span=n_span,
        n_chord=n_chord,
        alpha_deg=alpha_deg,
        dihedral_deg=dihedral_deg,
        pitch_deg=pitch_deg,
        offset=offset,
        gmsh_lc=gmsh_lc,
        name=name,
    )


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
    pitch_deg: float = 0.0,
    offset: np.ndarray | None = None,
    gmsh_lc: float | None = None,
    name: str = "ABRAMSON1965_mesh",
):
    """Same as :func:`build_extruded_airfoil_mesh_gmsh` with default mesh name."""
    return build_extruded_airfoil_mesh_gmsh(
        beam_length=beam_length,
        chord=chord,
        raw_coords=raw_coords,
        xea_factor=xea_factor,
        n_span=n_span,
        n_chord=n_chord,
        alpha_deg=alpha_deg,
        dihedral_deg=dihedral_deg,
        pitch_deg=pitch_deg,
        offset=offset,
        gmsh_lc=gmsh_lc,
        name=name,
    )


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
        "--gmsh-lc",
        type=float,
        default=None,
        metavar="LC",
        help="Gmsh characteristic length at section control points [m]. "
        "If omitted, a value is derived from chord, span, and n_chord / n_span.",
    )
    parser.add_argument(
        "--pitch",
        type=float,
        default=None,
        metavar="DEG",
        help="Structural pitch about +X [deg]. If omitted, uses AnalysisConfig.pitch from the case file.",
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

    if args.pitch is not None:
        pitch_deg = float(args.pitch)
        print(f"Structural pitch about +X: pitch = {pitch_deg:g} deg (--pitch override)")
    else:
        pitch_deg = float(getattr(cfg, "pitch", 0.0))
        if abs(pitch_deg) > 1e-12:
            print(f"Structural pitch about +X: pitch = {pitch_deg:g} deg (from config.pitch)")

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

    gmsh_lc = args.gmsh_lc

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
            pitch_deg=pitch_deg,
            offset=offset,
            gmsh_lc=gmsh_lc,
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
            pitch_deg=pitch_deg,
            offset=offset,
            gmsh_lc=gmsh_lc,
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
