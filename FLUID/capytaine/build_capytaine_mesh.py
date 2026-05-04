"""
Build a watertight quadrilateral surface mesh for the NACA0003 example geometry
using Capytaine's Mesh API (vertices + faces).

Coordinate convention (aligned with STRUCTURE/FEA beam model):
    X : chordwise (elastic-axis reference at mid-chord by default)
    Y : span [0, beam_length]
    Z : thickness (airfoil normal-to-chord in section)

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
    alpha_deg: float = 0.0,
    dihedral_deg: float = 0.0,
    offset: np.ndarray | None = None,
    name: str = "NACA0003_hull",
):
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
    parser = argparse.ArgumentParser(description="Build Capytaine hull mesh for NACA0003 example.")
    parser.add_argument("--n-span", type=int, default=24, help="Spanwise segments")
    parser.add_argument(
        "--out",
        type=str,
        default=os.path.join(_THIS_DIR, "NACA0003_hull.vtu"),
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

    cfg = get_config("NACA0003")

    offset = np.array([0.0, 0.0, float(getattr(cfg, "offset_z", 0.0))], dtype=float)

    mesh = build_naca0003_mesh(
        beam_length=float(cfg.beam_length),
        chord=float(cfg.chord),
        raw_coords=cfg.raw,
        xea_factor=float(cfg.xea_factor),
        n_span=args.n_span,
        alpha_deg=float(getattr(cfg, "alpha_deg", 0.0)),
        dihedral_deg=float(getattr(cfg, "dihedral_angle", 0.0)),
        offset=offset,
        name="NACA0003",
    )

    body = None
    try:
        import capytaine as cpt

        body = cpt.FloatingBody(mesh=mesh, name="NACA0003")
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
