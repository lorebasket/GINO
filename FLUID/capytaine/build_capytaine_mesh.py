"""
Build a watertight surface mesh for hydrofoil examples (e.g. NACA0003, ABRAMSON1965).

Pipeline (default **loft** generator):
    1. Fit a closed spline through ``config.raw`` section coordinates (x/c, z/c).
    2. Map to physical X–Z with the elastic axis at the origin; pitch the section in that plane
       (angle of attack + wing dihedral as rotation about **+Y** through ``(0, 0, 0)``).
    3. Optionally plot root/tip section wires in 3D (``--plot-sections`` / ``--matplotlib``).
    4. Place the pitched section at ``Y = 0`` … ``beam_length`` (planform sweep shifts X per station).
    5. Loft the skin and mesh the hull.
    6. Apply structural dihedral about **+X** and global ``offset`` only (no second AoA rotation).

Coordinate convention (STRUCTURE/FEA beam model / ``create_beam_model``):
    X : chordwise; section shifted so ``xea_factor * chord`` lies at ``X = 0`` (elastic-axis line).
    Y : span from ``0`` to ``beam_length`` (same as beam node ``y`` coordinates).
    Z : airfoil thickness; mid-thickness / chord line at ``Z = 0`` for symmetric sections.

    Pitch (``alpha_deg``, ``wing_dihedral_angle``) is applied **before** lofting so every span station
    rotates about the same Y-axis through the origin. Post-loft ``R_x(dihedral_angle)`` matches the beam model.

Usage (from FSI root with sonata-env / GINO active):
    python FLUID/capytaine/build_capytaine_mesh.py --case_name NACA0003

Optional Gmsh OCC loft (``pip install gmsh``):
    python FLUID/capytaine/build_capytaine_mesh.py --case_name NACA0003 --generator gmsh
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

_STRUCTURE_PATH = os.path.join(_FSI_ROOT, "STRUCTURE")
if _STRUCTURE_PATH not in sys.path:
    sys.path.insert(0, _STRUCTURE_PATH)
from FEA.fea_utl.beam_geometry import planform_x_at_span  # noqa: E402


def _rotation_y(angle_deg: float) -> np.ndarray:
    """Right-handed rotation about +Y (nose-up positive α in the X–Z plane)."""
    t = np.deg2rad(angle_deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=float)


def _rotate_xz_about_origin_y(
    x: np.ndarray,
    z: np.ndarray,
    angle_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Rotate chordwise–thickness coordinates about +Y through the origin.

    Equivalent to ``p' = R_y(angle) @ [x, 0, z]`` with Y unchanged.
    """
    if abs(float(angle_deg)) < 1e-12:
        return np.asarray(x, dtype=float), np.asarray(z, dtype=float)
    r = _rotation_y(angle_deg)
    p = np.column_stack([np.asarray(x, dtype=float), np.zeros_like(x), np.asarray(z, dtype=float)])
    pr = np.einsum("ij,nj->ni", r, p)
    return pr[:, 0], pr[:, 2]


def _rotation_x(angle_deg: float) -> np.ndarray:
    """Right-handed rotation about +X (structural dihedral), consistent with ``rotate_beams_x``."""
    t = np.deg2rad(float(angle_deg))
    c, s = np.cos(t), np.sin(t)
    return np.array(
        [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]],
        dtype=float,
    )


def _prepare_closed_loop(raw_coords: np.ndarray) -> np.ndarray:
    """Return an open loop (first point not repeated at end)."""
    p = np.asarray(raw_coords, dtype=float)
    if p.ndim != 2 or p.shape[1] != 2:
        raise ValueError("raw_coords must have shape (N, 2)")
    if p.shape[0] < 3:
        raise ValueError("Need at least 3 input contour points.")
    if np.allclose(p[0], p[-1]):
        p = p[:-1]
    return p


def _has_blunt_te_block(loop: np.ndarray) -> bool:
    """
    True when the contour contains a blunt TE block (>= 3 points at max x).

    In that case, periodic spline smoothing tends to round/collapse the TE.
    """
    x = np.asarray(loop[:, 0], dtype=float)
    x_max = float(np.max(x))
    tol = max(1e-12, 1e-9 * max(abs(x_max), 1.0))
    return int(np.sum(np.abs(x - x_max) <= tol)) >= 3


def _has_sharp_te_pair(loop: np.ndarray) -> bool:
    """
    True when the contour has separate upper/lower vertices at the trailing edge.

    A periodic spline collapses this thin TE slit to one point and leaves a chordwise
    gap in the lofted skin (open mesh). The slit is exactly two coincident-x vertices;
    a blunt TE cap has three or more (see :func:`_has_blunt_te_block`) and must be
    routed to the blunt path instead.
    """
    x = np.asarray(loop[:, 0], dtype=float)
    x_max = float(np.max(x))
    tol = max(1e-12, 1e-9 * max(abs(x_max), 1.0))
    te_idx = np.flatnonzero(np.abs(x - x_max) <= tol)
    return int(te_idx.size) == 2


def _resample_open_polyline(poly: np.ndarray, n_points: int) -> np.ndarray:
    """Uniform arc-length samples on an open polyline (endpoints included)."""
    p = np.asarray(poly, dtype=float)
    if p.shape[0] < 2:
        raise ValueError("Need at least 2 polyline points.")
    if int(n_points) < 2:
        raise ValueError(f"n_points must be >= 2, got {n_points}.")
    seg = np.linalg.norm(p[1:] - p[:-1], axis=1)
    if not np.all(np.isfinite(seg)) or float(np.sum(seg)) <= 0.0:
        raise ValueError("Degenerate open polyline.")
    s = np.concatenate([[0.0], np.cumsum(seg)])
    s_target = np.linspace(0.0, s[-1], int(n_points))
    x = np.interp(s_target, s, p[:, 0])
    z = np.interp(s_target, s, p[:, 1])
    return np.column_stack([x, z])


def _sharp_te_closed_section(loop: np.ndarray, n_points: int) -> tuple[np.ndarray, int]:
    """
    Closed section with explicit TE-upper / TE-lower vertices for sharp trailing edges.

    Upper and lower branches are resampled separately so the thin TE segment is
    preserved; ``ile = n_per // 2`` with index ``0`` at TE upper and ``n_per - 1``
    at TE lower.
    """
    n_per = int(n_points)
    if n_per < 4:
        raise ValueError(f"n_points must be >= 4, got {n_per}.")
    ile = n_per // 2
    nu = ile
    nl = n_per - 1 - ile
    if abs(nu - nl) > 1:
        raise ValueError(
            f"Sharp-TE loft needs |nu-nl|<=1; got nu={nu}, nl={nl}, n_per={n_per}. "
            "Use an odd mesh_n_chord or adjust n_perimeter."
        )
    ile_raw = int(np.argmin(loop[:, 0]))
    upper = loop[: ile_raw + 1]
    lower = loop[ile_raw:]
    upper_pts = _resample_open_polyline(upper, nu + 1)[:-1]
    lower_pts = _resample_open_polyline(lower, nl + 1)
    section = np.vstack([upper_pts, lower_pts])
    if section.shape[0] != n_per:
        raise ValueError(f"Sharp-TE section size mismatch: got {section.shape[0]}, expected {n_per}.")
    return section, ile


def _closed_spline_section(raw_coords: np.ndarray, n_points: int) -> np.ndarray:
    """
    Closed 2D section (x/c, z/c) as a periodic spline through ``raw_coords``.

    Uses ``scipy.interpolate.splprep`` when available; otherwise uniform arc-length
    resampling along the polyline (smooth enough for dense ``raw`` tables).
    """
    if n_points < 3:
        raise ValueError(f"n_points must be >= 3, got {n_points}.")

    loop = _prepare_closed_loop(raw_coords)

    if _has_sharp_te_pair(loop):
        section, _ = _sharp_te_closed_section(loop, int(n_points))
        return section

    if _has_blunt_te_block(loop):
        # Preserve piecewise-flat/blunt TE geometry exactly.
        return _resample_closed_polyline(loop, int(n_points))

    try:
        from scipy.interpolate import splev, splprep

        x = loop[:, 0]
        z = loop[:, 1]
        # Periodic B-spline; s=0 interpolates the input points.
        tck, _ = splprep([x, z], per=True, s=0.0)
        u = np.linspace(0.0, 1.0, int(n_points), endpoint=False)
        xn, zn = splev(u, tck)
        return np.column_stack([np.asarray(xn, dtype=float), np.asarray(zn, dtype=float)])
    except ImportError:
        return _resample_closed_polyline(loop, int(n_points))


def _resample_closed_polyline(loop: np.ndarray, n_points: int) -> np.ndarray:
    """Uniform arc-length samples on a closed polyline (open loop input)."""
    p = np.vstack([loop, loop[0]])
    seg = np.linalg.norm(p[1:] - p[:-1], axis=1)
    if not np.all(np.isfinite(seg)) or float(np.sum(seg)) <= 0.0:
        raise ValueError("Degenerate section polyline.")
    s = np.concatenate([[0.0], np.cumsum(seg)])
    s_target = np.linspace(0.0, s[-1], int(n_points), endpoint=False)
    x = np.interp(s_target, s, p[:, 0])
    z = np.interp(s_target, s, p[:, 1])
    return np.column_stack([x, z])


def _leading_edge_index(section: np.ndarray) -> int:
    """Index of the leading edge (minimum chordwise coordinate) on a closed section."""
    x = np.asarray(section[:, 0], dtype=float)
    x_min = float(np.min(x))
    tol = max(1e-12, 1e-9 * max(abs(x_min), 1.0))
    candidates = np.flatnonzero(x <= x_min + tol)
    if candidates.size == 1:
        return int(candidates[0])
    # LE may appear twice when the loop starts/ends there; pick a balanced split.
    target = len(section) // 2
    return int(candidates[np.argmin(np.abs(candidates - target))])


def _roll_section_for_symmetric_caps(
    section: np.ndarray,
    ile: int,
) -> tuple[np.ndarray, int]:
    """
    Roll a closed section so the LE lies between upper/lower branches for tip/root caps.

    Contours that start at the LE (``ile == 0`` after spline sampling) break
    ``_symmetric_section_cap_faces``, which expects upper stations at ``ile-1, …``
    and lower at ``ile+1, …``.
    """
    n = int(len(section))
    if n < 4:
        raise ValueError(f"section needs >= 4 points, got {n}.")
    if 1 <= ile <= n - 2:
        return section, ile

    target = n // 2
    rolled = np.roll(section, target - ile, axis=0)
    new_ile = _leading_edge_index(rolled)
    if not (1 <= new_ile <= n - 2):
        raise ValueError(
            f"Could not roll section for cap split: ile={new_ile}, n_per={n}. "
            "Check raw contour or increase mesh_n_chord."
        )
    return rolled, new_ile


def _plot_raw_section_2d(raw_coords: np.ndarray) -> None:
    """Optional 2D preview of ``config.raw`` (used with CLI ``--matplotlib``)."""
    import matplotlib.pyplot as plt

    x = np.asarray(raw_coords[:, 0], dtype=float)
    z = np.asarray(raw_coords[:, 1], dtype=float)
    plt.figure(figsize=(10, 3))
    plt.plot(np.r_[x, x[0]], np.r_[z, z[0]], "-o")
    plt.axis("equal")
    plt.grid(True)
    plt.xlabel("x/c")
    plt.ylabel("z/c")
    plt.title("Section coordinates (raw)")
    plt.show()


def _prepare_loft_section_xz(
    raw_coords: np.ndarray,
    *,
    chord: float,
    xea_factor: float,
    n_perimeter: int | None,
    section_z_shift: float = 0.0,
    pitch_about_y_deg: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """
    Spline ``raw`` → physical section (X, Z) with EA at origin, then pitch about +Y.

    Returns ``x_sec``, ``z_sec`` [m], leading-edge index ``ile``, and ``n_per``.
    """
    raw_loop = _prepare_closed_loop(raw_coords)
    n_per_req = int(n_perimeter) if n_perimeter is not None else max(raw_loop.shape[0], 8)
    n_per = max(n_per_req, int(raw_loop.shape[0])) if _has_blunt_te_block(raw_loop) else n_per_req
    if _has_sharp_te_pair(raw_loop):
        section, ile = _sharp_te_closed_section(raw_loop, n_per)
    else:
        section = _closed_spline_section(raw_loop, n_per)
        ile = _leading_edge_index(section)
        section, ile = _roll_section_for_symmetric_caps(section, ile)
    x_sec, z_sec = _section_xyz(
        section,
        chord=chord,
        xea_factor=xea_factor,
        z_shift=section_z_shift,
    )
    x_sec, z_sec = _rotate_xz_about_origin_y(x_sec, z_sec, pitch_about_y_deg)
    return x_sec, z_sec, ile, n_per


def _plot_loft_sections_3d(
    x_sec: np.ndarray,
    z_sec: np.ndarray,
    *,
    beam_length: float,
    sweep_angle_deg: float = 0.0,
    beam_axis_x_offset: float = 0.0,
    pitch_about_y_deg: float = 0.0,
    block: bool = True,
) -> None:
    """3D preview of pitched sections at root and tip before lofting."""
    import matplotlib.pyplot as plt

    y_root, y_tip = 0.0, float(beam_length)
    x_off = planform_x_at_span(
        np.array([y_root, y_tip], dtype=float),
        sweep_angle_deg,
        beam_axis_x_offset,
    )
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(
        x_sec + x_off[0],
        np.full_like(x_sec, y_root),
        z_sec,
        "-",
        color="C0",
        lw=1.8,
        label=f"root  (pitch Y={pitch_about_y_deg:g}°)",
    )
    ax.plot(
        x_sec + x_off[1],
        np.full_like(x_sec, y_tip),
        z_sec,
        "--",
        color="C1",
        lw=1.2,
        label="tip (same section + sweep)",
    )
    ax.scatter([0.0], [0.0], [0.0], c="red", s=50, zorder=5, label="EA / Y-axis origin")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m] (span)")
    ax.set_zlabel("Z [m]")
    ax.set_title("Sections after pitch about +Y at origin (before loft)")
    ax.legend(loc="upper left")
    _set_axes_equal_3d(ax)
    plt.tight_layout()
    if block:
        plt.show()
    else:
        plt.show(block=False)


def _set_axes_equal_3d(ax) -> None:
    """Equal aspect ratio on a 3D matplotlib axis."""
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    center = limits.mean(axis=1)
    radius = 0.5 * np.max(limits[:, 1] - limits[:, 0])
    ax.set_xlim3d(center[0] - radius, center[0] + radius)
    ax.set_ylim3d(center[1] - radius, center[1] + radius)
    ax.set_zlim3d(center[2] - radius, center[2] + radius)


def _symmetric_section_cap_faces(
    base: int,
    ile: int,
    n_per: int,
    *,
    tip: bool,
) -> list[list[int]]:
    """
    Root/tip caps: mirror upper/lower stations about the leading edge (``argmin(x)``).

    The LE cusp is a triangle encoded as a degenerate quad (Capytaine requires 4-ples).
    When ``nu != nl`` (even ``n_per`` with sharp TE), the last ring pair and a small TE
    triangle are emitted separately so cap quads do not overlap at the trailing edge.
    """
    nu = ile
    nl = n_per - 1 - ile
    if nu < 1 or nl < 1:
        raise ValueError(f"Invalid LE split: ile={ile}, n_per={n_per}.")
    if abs(nu - nl) > 1:
        raise ValueError(
            f"Symmetric TE caps need |nu-nl|<=1; got nu={nu}, nl={nl}, n_per={n_per}."
        )

    def iv(j: int) -> int:
        return base + j

    faces: list[list[int]] = []
    # LE triangle (ile-1, ile, ile+1) as a degenerate quad.
    if tip:
        faces.append([iv(ile), iv(ile - 1), iv(ile + 1), iv(ile)])
    else:
        faces.append([iv(ile), iv(ile + 1), iv(ile - 1), iv(ile)])

    n_pair = min(nu, nl)
    k_last = n_pair + 1 if nu == nl else n_pair
    for k in range(2, k_last):
        um, up = iv(ile - k), iv(ile - (k - 1))
        lm, lp = iv(ile + k), iv(ile + (k - 1))
        if tip:
            faces.append([up, um, lm, lp])
        else:
            faces.append([up, lp, lm, um])

    if nu != nl:
        # Penultimate ring up to TE-lower / TE-upper neighbour, then the TE-upper wedge.
        um, up, lm, lp = iv(ile - n_pair), iv(ile - (n_pair - 1)), iv(ile + n_pair), iv(ile + (n_pair - 1))
        if tip:
            faces.append([up, um, lm, lp])
        else:
            faces.append([up, lp, lm, um])
        te_u, te_n, te_l = iv(0), iv(1), iv(n_per - 1)
        if tip:
            faces.append([te_u, te_n, te_l, te_u])
        else:
            faces.append([te_u, te_l, te_n, te_u])

    return faces


def _section_xyz(
    section_norm: np.ndarray,
    *,
    chord: float,
    xea_factor: float,
    z_shift: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Chord-normalized (x/c, z/c) loop → physical X, Z at one span station."""
    xea = float(xea_factor) * float(chord)
    x = section_norm[:, 0] * float(chord) - xea
    z = section_norm[:, 1] * float(chord) + float(z_shift)
    return x, z


def _default_mesh_lc(
    section_norm: np.ndarray,
    chord: float,
    beam_length: float,
    n_span: int,
    n_perimeter: int,
    section_z_shift: float = 0.0,
) -> float:
    """Target Gmsh edge length from section perimeter and span."""
    x, z = _section_xyz(section_norm, chord=chord, xea_factor=0.5, z_shift=section_z_shift)
    p2 = np.column_stack([x, z])
    p2 = np.vstack([p2, p2[0]])
    perim_len = float(np.sum(np.linalg.norm(p2[1:] - p2[:-1], axis=1)))
    lc_perim = perim_len / max(int(n_perimeter), 8)
    lc_span = float(beam_length) / max(int(n_span), 4)
    return float(max(min(lc_perim, lc_span), 1e-10))


def _apply_frame_transform(
    vertices: np.ndarray,
    *,
    alpha_deg: float,
    wing_dihedral_deg: float,
    dihedral_deg: float,
    offset: np.ndarray | None,
) -> np.ndarray:
    """
    Post-loft rigid transform.

    By default ``alpha_deg`` and ``wing_dihedral_deg`` are zero here because pitch is applied
    on the section before lofting (see ``_prepare_loft_section_xz``). Only structural dihedral
    about +X and the global translation remain.
    """
    r_y = float(alpha_deg) + float(wing_dihedral_deg)
    if abs(r_y) > 1e-12:
        import warnings

        warnings.warn(
            "Post-loft R_y rotation is non-zero while pitch is normally applied pre-loft; "
            "check for double-counting angle of attack.",
            stacklevel=2,
        )
    r_tot = _rotation_x(dihedral_deg) @ _rotation_y(r_y)
    v = np.einsum("ij,nj->ni", r_tot, np.asarray(vertices, dtype=float))
    if offset is not None:
        v = v + np.asarray(offset, dtype=float).reshape(1, 3)
    return v


def _meshio_surface_hull(msh_path: str):
    """Read Gmsh output and return a single watertight triangle surface."""
    import meshio

    m = meshio.read(msh_path)
    tris = [cb.data for cb in m.cells if cb.type == "triangle"]
    if not tris:
        raise RuntimeError(
            f"No triangles in {msh_path!r}. Try --generator gmsh with a smaller --mesh-lc."
        )
    return meshio.Mesh(points=m.points, cells=[("triangle", np.vstack(tris))])


def build_lofted_hydrofoil_mesh_structured(
    *,
    beam_length: float,
    chord: float,
    raw_coords: np.ndarray,
    xea_factor: float = 0.5,
    n_span: int = 24,
    n_perimeter: int | None = None,
    alpha_deg: float = 0.0,
    wing_dihedral_deg: float = 0.0,
    dihedral_deg: float = 0.0,
    sweep_angle_deg: float = 0.0,
    beam_axis_x_offset: float = 0.0,
    section_z_shift: float = 0.0,
    span_start: float = 0.0,
    local_section_offset: np.ndarray | None = None,
    offset: np.ndarray | None = None,
    plot_sections: bool = False,
    name: str = "hydrofoil_mesh",
):
    """
    Structured loft: spline section × uniform span stations → conformal quad hull.

    ``n_span`` : number of span segments (``n_span + 1`` stations along Y).
    ``n_perimeter`` : spline evaluation / mesh nodes around the section (default: len(raw)−1).
    ``span_start`` : first span station [m]; use the wetted distance from the root so the
        loft covers only the submerged span ``[span_start, beam_length]`` (default 0 = root).
    """
    import capytaine as cpt

    pitch_deg = float(alpha_deg) + float(wing_dihedral_deg)
    x0, z0, ile, n_per = _prepare_loft_section_xz(
        raw_coords,
        chord=chord,
        xea_factor=xea_factor,
        n_perimeter=n_perimeter,
        section_z_shift=section_z_shift,
        pitch_about_y_deg=pitch_deg,
    )
    if plot_sections:
        _plot_loft_sections_3d(
            x0,
            z0,
            beam_length=beam_length,
            sweep_angle_deg=sweep_angle_deg,
            beam_axis_x_offset=beam_axis_x_offset,
            pitch_about_y_deg=pitch_deg,
        )

    n_stations = int(n_span) + 1
    if not (0.0 <= float(span_start) < float(beam_length)):
        raise ValueError(
            f"span_start must satisfy 0 <= span_start < beam_length; "
            f"got span_start={span_start}, beam_length={beam_length}."
        )
    # ``span_start`` / ``beam_length`` are arc-lengths measured ALONG the swept elastic axis.
    # Project them onto the vertical span (Y here, global Z after the +X dihedral) by
    # cos(sweep) so the loft stations sit at the same heights as the beam nodes. Both the loft
    # and the beam offset the chord via ``planform_x_at_span`` evaluated at these projected
    # heights, so the mesh elastic axis is collinear with the structural EA (X = y·tan(sweep),
    # Y = arc·cos(sweep) ⇒ chord offset = arc·sin(sweep)). The first/last stations land on the
    # horizontal waterline (``span_start``) and tip planes: this is the swept section extruded
    # along the EA and trimmed by two horizontal planes, capped at both ends.
    cos_sweep = float(np.cos(np.deg2rad(float(sweep_angle_deg))))
    arc_stations = np.linspace(float(span_start), float(beam_length), n_stations)
    y_stations = arc_stations * cos_sweep

    # Vertex grid: station j, perimeter i  →  index j * n_per + i
    vertices = np.empty((n_stations * n_per, 3), dtype=float)
    x_at_station = planform_x_at_span(y_stations, sweep_angle_deg, beam_axis_x_offset)

    for j, y in enumerate(y_stations):
        sl = slice(j * n_per, (j + 1) * n_per)
        vertices[sl, 0] = x0 + x_at_station[j]
        vertices[sl, 1] = y
        vertices[sl, 2] = z0

    if local_section_offset is not None:
        vertices = vertices + np.asarray(local_section_offset, dtype=float).reshape(1, 3)

    def vid(j: int, i: int) -> int:
        return (j % n_stations) * n_per + (i % n_per)

    faces: list[list[int]] = []

    # Lateral skin (ruled loft between identical sections)
    for j in range(n_span):
        for i in range(n_per):
            i1 = (i + 1) % n_per
            faces.append([vid(j, i), vid(j, i1), vid(j + 1, i1), vid(j + 1, i)])

    faces.extend(_symmetric_section_cap_faces(0, ile, n_per, tip=False))
    faces.extend(_symmetric_section_cap_faces(n_span * n_per, ile, n_per, tip=True))

    vertices = _apply_frame_transform(
        vertices,
        alpha_deg=0.0,
        wing_dihedral_deg=0.0,
        dihedral_deg=dihedral_deg,
        offset=offset,
    )
    faces_arr = np.asarray(faces, dtype=np.int64)
    return cpt.Mesh(vertices=vertices, faces=faces_arr, name=name)


def build_lofted_hydrofoil_mesh_gmsh(
    *,
    beam_length: float,
    chord: float,
    raw_coords: np.ndarray,
    xea_factor: float = 0.5,
    n_span: int = 24,
    n_perimeter: int | None = None,
    alpha_deg: float = 0.0,
    wing_dihedral_deg: float = 0.0,
    dihedral_deg: float = 0.0,
    sweep_angle_deg: float = 0.0,
    beam_axis_x_offset: float = 0.0,
    section_z_shift: float = 0.0,
    span_start: float = 0.0,
    local_section_offset: np.ndarray | None = None,
    offset: np.ndarray | None = None,
    mesh_lc: float | None = None,
    plot_sections: bool = False,
    name: str = "hydrofoil_mesh",
):
    """
    Gmsh OCC: closed B-spline wire at root and tip → ``addThruSections`` loft → 2D mesh.

    Root/tip caps plus one ruled lateral surface (3 surfaces total). ``span_start`` is the
    first span station [m] (wetted distance from the root) so the loft covers only the
    submerged span ``[span_start, beam_length]`` (default 0 = root).
    """
    if not (0.0 <= float(span_start) < float(beam_length)):
        raise ValueError(
            f"span_start must satisfy 0 <= span_start < beam_length; "
            f"got span_start={span_start}, beam_length={beam_length}."
        )
    try:
        import gmsh
    except ImportError as e:
        raise ImportError(
            "Gmsh is required for --generator gmsh. Install with: pip install gmsh"
        ) from e

    import capytaine as cpt
    import meshio

    pitch_deg = float(alpha_deg) + float(wing_dihedral_deg)
    x_sec, z_sec, _, n_per = _prepare_loft_section_xz(
        raw_coords,
        chord=chord,
        xea_factor=xea_factor,
        n_perimeter=n_perimeter,
        section_z_shift=section_z_shift,
        pitch_about_y_deg=pitch_deg,
    )
    if plot_sections:
        _plot_loft_sections_3d(
            x_sec,
            z_sec,
            beam_length=beam_length,
            sweep_angle_deg=sweep_angle_deg,
            beam_axis_x_offset=beam_axis_x_offset,
            pitch_about_y_deg=pitch_deg,
        )
    local_off = (
        np.asarray(local_section_offset, dtype=float).reshape(3)
        if local_section_offset is not None
        else np.zeros(3, dtype=float)
    )
    if mesh_lc is not None:
        lc = float(mesh_lc)
    else:
        p2 = np.column_stack([x_sec, z_sec])
        p2 = np.vstack([p2, p2[0]])
        perim_len = float(np.sum(np.linalg.norm(p2[1:] - p2[:-1], axis=1)))
        lc_perim = perim_len / max(int(n_per), 8)
        lc_span = float(beam_length) / max(int(n_span), 4)
        lc = float(max(min(lc_perim, lc_span), 1e-10))
    # Avoid spurious edge subdivision: lc not smaller than shortest spline segment.
    p3 = np.column_stack([x_sec, np.zeros_like(x_sec), z_sec])
    seg = np.linalg.norm(np.diff(np.vstack([p3, p3[0]]), axis=0), axis=1)
    lc = float(max(lc, float(np.min(seg)) * 0.98))

    fd, msh_path = tempfile.mkstemp(suffix=".msh", prefix="capytaine_loft_")
    os.close(fd)
    try:
        gmsh.initialize()
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
            gmsh.model.add("hydrofoil_loft")

            def _wire_at_y(y: float) -> int:
                pt_tags = [
                    gmsh.model.occ.addPoint(
                        float(xi + local_off[0]),
                        float(y + local_off[1]),
                        float(zi + local_off[2]),
                        lc,
                    )
                    for xi, zi in zip(x_sec, z_sec)
                ]
                curve = gmsh.model.occ.addBSpline(pt_tags + [pt_tags[0]])
                return gmsh.model.occ.addWire([curve])

            # span_start / beam_length are arc-lengths along the swept EA; project to the
            # vertical span by cos(sweep) so wires sit at the beam-node heights (see the
            # structured builder). The chordwise offset below uses planform_x_at_span on the
            # already-projected Y, giving X = arc·sin(sweep) at the tip — collinear with the beam.
            cos_sweep = float(np.cos(np.deg2rad(float(sweep_angle_deg))))
            wire_root = _wire_at_y(float(span_start) * cos_sweep)
            wire_tip = _wire_at_y(float(beam_length) * cos_sweep)
            gmsh.model.occ.addPlaneSurface([wire_root])
            gmsh.model.occ.addPlaneSurface([wire_tip])
            gmsh.model.occ.addThruSections(
                [wire_root, wire_tip],
                makeSolid=False,
                makeRuled=True,
            )
            gmsh.model.occ.synchronize()

            gmsh.model.mesh.generate(2)
            gmsh.write(msh_path)
        finally:
            gmsh.finalize()

        mio = _meshio_surface_hull(msh_path)
        meshio.write(msh_path, mio)
        mesh = cpt.load_mesh(msh_path)

        v = np.asarray(mesh.vertices, dtype=float)
        if abs(sweep_angle_deg) > 1e-12 or abs(beam_axis_x_offset) > 1e-12:
            v[:, 0] += planform_x_at_span(v[:, 1], sweep_angle_deg, beam_axis_x_offset)
        v = _apply_frame_transform(
            v,
            alpha_deg=0.0,
            wing_dihedral_deg=0.0,
            dihedral_deg=dihedral_deg,
            offset=offset,
        )
        faces = np.asarray(mesh.faces, dtype=np.int64)
        if faces.shape[1] == 3:
            # Capytaine Mesh expects 4-ples; degenerate quad per triangle.
            faces = np.column_stack([faces[:, 0], faces[:, 1], faces[:, 2], faces[:, 2]])
        mesh = cpt.Mesh(vertices=v, faces=faces, name=name)
    finally:
        try:
            os.remove(msh_path)
        except OSError:
            pass

    return mesh


def build_lofted_hydrofoil_mesh(
    *,
    beam_length: float,
    chord: float,
    raw_coords: np.ndarray,
    xea_factor: float = 0.5,
    n_span: int = 24,
    n_perimeter: int | None = None,
    alpha_deg: float = 0.0,
    wing_dihedral_deg: float = 0.0,
    dihedral_deg: float = 0.0,
    sweep_angle_deg: float = 0.0,
    beam_axis_x_offset: float = 0.0,
    section_z_shift: float = 0.0,
    span_start: float = 0.0,
    local_section_offset: np.ndarray | None = None,
    offset: np.ndarray | None = None,
    mesh_lc: float | None = None,
    plot_sections: bool = False,
    generator: str = "loft",
    name: str = "hydrofoil_mesh",
):
    """
    Build a closed hydrofoil hull mesh.

    Parameters
    ----------
    generator : ``'loft'`` | ``'structured'`` | ``'gmsh'``
        ``loft`` (default): structured quad loft; falls back if only Gmsh requested and missing.
        ``structured``: conformal quad grid without Gmsh.
        ``gmsh``: OCC spline loft + Gmsh surface mesh.
    """
    gen = str(generator).strip().lower()
    common = dict(
        beam_length=beam_length,
        chord=chord,
        raw_coords=raw_coords,
        xea_factor=xea_factor,
        n_span=n_span,
        n_perimeter=n_perimeter,
        alpha_deg=alpha_deg,
        wing_dihedral_deg=wing_dihedral_deg,
        dihedral_deg=dihedral_deg,
        sweep_angle_deg=sweep_angle_deg,
        beam_axis_x_offset=beam_axis_x_offset,
        section_z_shift=section_z_shift,
        span_start=span_start,
        local_section_offset=local_section_offset,
        offset=offset,
        plot_sections=plot_sections,
        name=name,
    )
    if gen in ("structured", "loft", "auto", "default"):
        return build_lofted_hydrofoil_mesh_structured(**common)
    if gen == "gmsh":
        return build_lofted_hydrofoil_mesh_gmsh(**common, mesh_lc=mesh_lc)
    raise ValueError(f"Unknown mesh generator {generator!r}. Use 'loft', 'structured', or 'gmsh'.")


def build_naca0003_mesh(**kwargs):
    return build_lofted_hydrofoil_mesh(name=kwargs.pop("name", "NACA0003"), **kwargs)


def build_ABRAMSON1965_mesh(**kwargs):
    return build_lofted_hydrofoil_mesh(name=kwargs.pop("name", "ABRAMSON1965"), **kwargs)


def _span_axis_in_hydro_mesh(cfg, dihedral_deg: float) -> int:
    """Global axis index along the beam span after the structural dihedral is applied."""
    explicit = getattr(cfg, "capytaine_pod_span_axis", None)
    if explicit is not None:
        return int(explicit)
    if abs(float(dihedral_deg) + 90.0) < 1.0:
        return 2  # loft along +Y, R_x(-90°) → span on global Z (Model2T / Abramson)
    if abs(float(dihedral_deg)) < 1e-6:
        return 1
    return int(getattr(cfg, "beam_span_axis", 1))


def _tip_pod_attach_from_strut_mesh(
    strut_vertices: np.ndarray,
    *,
    span_ax: int,
    chord_ax: int = 0,
    lateral_ax: int = 1,
    n_span_hint: int = 30,
) -> tuple[float, float, float, np.ndarray]:
    """
  Tip attachment data from the strut-only mesh (before merging the pod).

  Returns ``(span_tip, x_le_tip, lateral_mid, tip_slab)`` where ``span_tip`` is the
  tip station along ``span_ax`` (minimum for the usual R_x(-90°) vertical strut).
    """
    verts = np.asarray(strut_vertices, dtype=float)
    span = verts[:, span_ax]
    span_tip = float(np.min(span))
    span_tol = max(5e-4, float(np.ptp(span)) / max(2 * int(n_span_hint), 2))
    tip_slab = verts[span <= span_tip + span_tol]
    if tip_slab.size == 0:
        tip_slab = verts
    x_le_tip = float(tip_slab[:, chord_ax].min())
    lateral_mid = float(np.mean(tip_slab[:, lateral_ax]))
    return span_tip, x_le_tip, lateral_mid, tip_slab


def _resolve_pod_gap(cfg, tip_slab: np.ndarray, *, chord_ax: int = 0) -> float:
    """
    Clearance between the strut tip cap and the pod top [m].

    - ``capytaine_pod_gap`` omitted / ``None`` → auto ≈ 40 % of a chordwise panel
      width at the tip (typically 1–4 mm).
    - ``capytaine_pod_gap = 0`` → welded (pod top exactly at the strut tip station).
    - ``capytaine_pod_gap > 0`` → explicit clearance [m].
    """
    if bool(getattr(cfg, "capytaine_pod_weld", False)):
        return 0.0
    gap_cfg = getattr(cfg, "capytaine_pod_gap", None)
    if gap_cfg is not None:
        gap = float(gap_cfg)
        if gap <= 0.0:
            return 0.0
        return gap
    chord_ext = float(np.ptp(tip_slab[:, chord_ax])) if tip_slab.size else float(getattr(cfg, "chord", 0.1))
    n_ch = int(getattr(cfg, "mesh_n_chord", 30) or 30)
    return max(0.001, 0.4 * chord_ext / max(n_ch, 8))


def build_tip_pod_cylinder_mesh(
    cfg,
    strut_vertices,
    *,
    dihedral_deg=None,
    n_span_hint=None,
    name="pod",
    pod_gap_override=None,
):
    """
    Build a capped-cylinder tip pod mesh positioned from the strut-only vertices.

    Placement follows Besch & Liu Fig. 20: streamwise centre
    ``x_LE_tip − nose_ahead + L/2``, lateral mid from the tip cap ring.

    Spanwise placement (along ``span_ax``):

    - **gap** (default from ``capytaine_pod_gap``): pod top at ``span_tip − gap``.
      Used for the merged strut+pod mesh so the two bodies do not share panels.
    - **flush weld** (``pod_gap_override=0`` without ``capytaine_pod_weld``): pod top
      exactly at ``span_tip``. Used for pod-only BEM in the superposition workflow.
    - **overlap weld** (``capytaine_pod_weld=True``): cylinder centre on ``span_tip``;
      overlapping strut panels are clipped when merging.
    """
    from capytaine.meshes.predefined import mesh_horizontal_cylinder

    pod_diam = float(getattr(cfg, "capytaine_pod_diameter", 0.0) or 0.0)
    if pod_diam <= 0.0:
        raise ValueError("capytaine_pod_diameter must be > 0 to build a tip pod mesh")

    if dihedral_deg is None:
        dihedral_deg = float(getattr(cfg, "dihedral_angle", 0.0))
    if n_span_hint is None:
        n_span_hint = int(getattr(cfg, "mesh_n_span", 30))

    pod_len = float(getattr(cfg, "capytaine_pod_length", pod_diam))
    nose_ahead = float(getattr(cfg, "capytaine_pod_nose_ahead_of_le", 0.0))
    pod_res = tuple(int(x) for x in getattr(cfg, "capytaine_pod_resolution", (4, 20, 48)))
    pod_r = 0.5 * pod_diam

    strut_verts = np.asarray(strut_vertices, dtype=float)
    span_ax = _span_axis_in_hydro_mesh(cfg, dihedral_deg)
    span_tip, x_le_tip, lateral_mid, tip_slab = _tip_pod_attach_from_strut_mesh(
        strut_verts,
        span_ax=span_ax,
        n_span_hint=n_span_hint,
    )
    overlap_weld = bool(getattr(cfg, "capytaine_pod_weld", False))
    if pod_gap_override is not None:
        pod_gap = max(0.0, float(pod_gap_override))
        overlap_weld = False
    else:
        pod_gap = _resolve_pod_gap(cfg, tip_slab)
    pod_cx = float(x_le_tip - nose_ahead + 0.5 * pod_len)

    pod_center = np.zeros(3, dtype=float)
    pod_center[0] = pod_cx
    pod_center[1] = lateral_mid
    if overlap_weld:
        pod_center[span_ax] = float(span_tip)
    else:
        pod_center[span_ax] = float(span_tip - pod_r - pod_gap)

    pod = mesh_horizontal_cylinder(
        length=pod_len,
        radius=pod_r,
        center=tuple(pod_center.tolist()),
        resolution=pod_res,
        name=name,
    )
    info = {
        "span_tip": span_tip,
        "x_le_tip": x_le_tip,
        "lateral_mid": lateral_mid,
        "pod_gap": pod_gap,
        "welded": overlap_weld,
        "pod_center": pod_center,
        "span_ax": span_ax,
    }
    return pod, pod_center, info


# Backward-compatible aliases
build_extruded_airfoil_mesh_gmsh = build_lofted_hydrofoil_mesh_gmsh


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
        "--generator",
        type=str,
        default="loft",
        choices=("loft", "structured", "gmsh"),
        help="Mesh pipeline: 'loft'/'structured' = spline section + structured quad loft; "
        "'gmsh' = OCC spline loft + Gmsh surface mesh.",
    )
    parser.add_argument(
        "--n-span",
        type=int,
        default=None,
        help="Spanwise mesh segments (default: cfg.mesh_n_span or 24).",
    )
    parser.add_argument(
        "--n-perimeter",
        type=int,
        default=None,
        help="Section spline evaluation / perimeter nodes (default: cfg.mesh_n_chord or raw loop size).",
    )
    parser.add_argument(
        "--mesh-lc",
        type=float,
        default=None,
        metavar="LC",
        help="Gmsh target edge length [m] (--generator gmsh only).",
    )
    parser.add_argument(
        "--gmsh-lc",
        type=float,
        default=None,
        metavar="LC",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--n-chord",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--dihedral",
        type=float,
        default=None,
        metavar="DEG",
        help="Structural dihedral about +X [deg]. If omitted, uses AnalysisConfig.dihedral_angle from the case file.",
    )
    parser.add_argument(
        "--offset-z",
        type=float,
        default=0.0,
        metavar="DZ",
        help="Extra translation applied to all mesh vertices along Z after build [m].",
    )
    parser.add_argument(
        "--use-config-offset-z",
        action="store_true",
        help="Ignore --offset-z and use AnalysisConfig.offset_z instead.",
    )
    parser.add_argument(
        "--no-pod",
        action="store_true",
        help="Skip the tip-pod merge even when cfg.capytaine_pod is set "
        "(strut-only mesh, e.g. for the superposition workflow or mesh-convergence studies).",
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
    parser.add_argument(
        "--plot-sections",
        action="store_true",
        help="Plot pitched root/tip section wires in 3D before lofting (about +Y at origin)",
    )
    args = parser.parse_args()
    if args.out is None:
        args.out = os.path.join(_THIS_DIR, f"{args.case_name}/{args.case_name}_mesh.vtu")
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    cfg = get_config(args.case_name)
    n_span = int(args.n_span) if args.n_span is not None else int(getattr(cfg, "mesh_n_span", 24))
    n_per = args.n_perimeter
    if n_per is None:
        legacy = args.n_chord if args.n_chord is not None else getattr(cfg, "mesh_n_chord", None)
        n_per = int(legacy) if legacy is not None else None

    mesh_lc = args.mesh_lc if args.mesh_lc is not None else args.gmsh_lc

    if args.dihedral is not None:
        dihedral_deg = float(args.dihedral)
        print(f"Structural dihedral about +X: dihedral_angle = {dihedral_deg:g} deg (--dihedral override)")
    else:
        dihedral_deg = float(getattr(cfg, "dihedral_angle", 0.0))
        if abs(dihedral_deg) > 1e-12:
            print(f"Structural dihedral about +X: dihedral_angle = {dihedral_deg:g} deg (from config)")

    if args.use_config_offset_z:
        dz = float(getattr(cfg, "offset_z", 0.0))
        print(f"Mesh Z translation from config offset_z = {dz:g} m (--use-config-offset-z)")
    else:
        dz = float(args.offset_z)
        if dz != 0.0:
            print(f"Mesh Z translation from --offset-z = {dz:g} m")
        else:
            print("Mesh aligned with beam frame: Z translation 0 (elastic-axis line at z=0).")

    section_z_shift = 0.0
    local_section_offset = np.zeros(3, dtype=float)
    alpha_deg = float(getattr(cfg, "alpha_deg", 0.0))
    wing_dihedral_deg = float(getattr(cfg, "wing_dihedral_angle", 0.0))
    pitch_deg = alpha_deg + wing_dihedral_deg
    if abs(pitch_deg) > 1e-12:
        print(
            f"Pre-loft pitch about +Y at origin: alpha_deg={alpha_deg:g}° "
            f"+ wing_dihedral={wing_dihedral_deg:g}° → {pitch_deg:g}°"
        )

    if hasattr(cfg, "mesh_depth"):
        # Shift section before loft; pitch is already in (x,z). Only dihedral remains post-loft.
        mesh_depth = float(getattr(cfg, "mesh_depth"))
        r_post = _rotation_x(dihedral_deg)
        global_shift = np.array([0.0, 0.0, -mesh_depth], dtype=float)
        local_section_offset = r_post.T @ global_shift
        print(
            "Section pre-loft shift from mesh_depth (target global z): "
            f"mesh_depth={mesh_depth:g} m -> "
            f"local_offset=({local_section_offset[0]:.6g}, {local_section_offset[1]:.6g}, {local_section_offset[2]:.6g}) m."
        )

    span_start = float(getattr(cfg, "wetted_distance_from_root", 0.0))
    if span_start > 0.0:
        print(
            f"Submerged loft: span starts at wetted_distance_from_root = {span_start:g} m "
            f"(dry root segment above the waterline is not meshed)."
        )

    plot_sections = bool(args.plot_sections or args.matplotlib)
    offset = np.array([0.0, 0.0, dz], dtype=float)
    gen = args.generator
    if gen == "loft":
        gen_label = "structured spline loft (conformal quads)"
    elif gen == "structured":
        gen_label = gen
    else:
        gen_label = "Gmsh OCC spline loft + surface mesh"
    print(f"Mesh generator: {gen_label}")

    common = dict(
        beam_length=float(cfg.beam_length),
        chord=float(cfg.chord),
        raw_coords=cfg.raw,
        xea_factor=float(cfg.xea_factor),
        n_span=n_span,
        n_perimeter=n_per,
        alpha_deg=alpha_deg,
        wing_dihedral_deg=wing_dihedral_deg,
        dihedral_deg=dihedral_deg,
        sweep_angle_deg=float(getattr(cfg, "sweep_angle_deg", 0.0)),
        beam_axis_x_offset=float(getattr(cfg, "beam_axis_x_offset", 0.0)),
        section_z_shift=section_z_shift,
        span_start=span_start,
        local_section_offset=local_section_offset,
        offset=offset,
        mesh_lc=mesh_lc,
        plot_sections=plot_sections,
        generator=gen,
    )

    if args.matplotlib and not plot_sections:
        _plot_raw_section_2d(cfg.raw)

    if args.case_name in ("NACA0003", "NACA0015_POM", "NACA0015_LiHua2020"):
        mesh = build_naca0003_mesh(**common, name=args.case_name)
    elif args.case_name == "ABRAMSON1965":
        mesh = build_ABRAMSON1965_mesh(**common, name="ABRAMSON1965")
    elif args.case_name == "Model1":
        mesh = build_ABRAMSON1965_mesh(**common, name="Model1")
    elif args.case_name == "Model2t":
        mesh = build_ABRAMSON1965_mesh(**common, name="Model2t")
    elif args.case_name == "Modelref2":
        mesh = build_ABRAMSON1965_mesh(**common, name="Modelref2")
    elif args.case_name == "GOLAND_water":
        mesh = build_ABRAMSON1965_mesh(**common, name="GOLAND_water")
    else:
        raise ValueError(f"Mesher not implemented for case: {args.case_name}")

    # ── Optional tip pod/bulb: merge a capped cylinder at the strut tip ──────
    # Model 2T (Besch & Liu Fig. 20): a 2-in-Ø pod is welded to the strut tip.
    # Mesh is built from the strut loft only; pod placement uses the tip-cap ring on
    # that mesh (fixed wet geometry). Pod top sits at span_tip − gap with gap ≪ panel
    # size (auto when capytaine_pod_gap ≤ 0). Beam→panel projection clamps span coords
    # beyond the tip to the tip node, so pod panels follow tip rigid motion.
    pod_diam = float(getattr(cfg, "capytaine_pod_diameter", 0.0) or 0.0)
    if args.no_pod:
        print("Tip pod merge skipped (--no-pod): strut-only mesh.")
    if bool(getattr(cfg, "capytaine_pod", False)) and pod_diam > 0.0 and not args.no_pod:
        strut_verts = np.asarray(mesh.vertices, dtype=float)
        n_span_hint = int(n_span) if n_span is not None else int(getattr(cfg, "mesh_n_span", 30))
        pod, pod_center, pod_info = build_tip_pod_cylinder_mesh(
            cfg, strut_verts, dihedral_deg=dihedral_deg, n_span_hint=n_span_hint, name="pod",
        )
        pod_gap = pod_info["pod_gap"]
        overlap_weld = pod_info["welded"]
        pod_center = pod_info["pod_center"]
        span_tip = pod_info["span_tip"]
        pod_r = 0.5 * pod_diam
        pod_len = float(getattr(cfg, "capytaine_pod_length", pod_diam))
        nose_ahead = float(getattr(cfg, "capytaine_pod_nose_ahead_of_le", 0.0))
        span_ax = pod_info["span_ax"]
        pod_cx = float(pod_center[0])

        if overlap_weld:
            sv = np.asarray(mesh.vertices, dtype=float)
            sf = np.asarray(mesh.faces, dtype=np.int64)
            fc = sv[sf].mean(axis=1)  # strut face centroids
            lateral_ax = 1
            rad2 = (fc[:, lateral_ax] - pod_center[lateral_ax]) ** 2 + (
                fc[:, span_ax] - pod_center[span_ax]
            ) ** 2
            inside = (
                (rad2 < (0.995 * pod_r) ** 2)
                & (fc[:, 0] > pod_cx - 0.5 * pod_len)
                & (fc[:, 0] < pod_cx + 0.5 * pod_len)
            )
            if bool(np.any(inside)):
                import capytaine as _cpt

                mesh = _cpt.Mesh(vertices=sv, faces=sf[~inside], name=mesh.name)
                print(f"Welded pod: clipped {int(inside.sum())} strut panels inside the pod.")
        gap_label = "welded (overlap+clip)" if overlap_weld else f"{pod_gap * 1e3:.2f} mm"
        print(
            f"Tip pod: Ø{pod_diam * 1e3:.1f} mm × {pod_len * 1e3:.0f} mm capped cylinder "
            f"({pod.nb_faces} faces), center=({pod_center[0]:.4f}, {pod_center[1]:.4f}, "
            f"{pod_center[2]:.4f}) m, gap={gap_label} (strut tip z={span_tip:.4f}), "
            f"nose {nose_ahead / 0.0254:.1f} in ahead of tip LE."
        )
        mesh = mesh + pod
        print(f"Merged strut+pod mesh: {mesh.nb_faces} faces")

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
