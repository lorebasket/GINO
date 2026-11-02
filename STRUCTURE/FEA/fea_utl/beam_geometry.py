"""
Reference-frame beam / wing axis geometry (before alpha, structural dihedral, etc.).

Convention
----------
- Span along global **+Y** from ``y = 0`` (root) to ``y = beam_length`` (tip).
- For flutter benchmarks (``beam_on_shear_center=True``), nodes lie on the **shear center** with
  the root at ``(beam_axis_x_offset, 0, beam_axis_z_offset)``; default ``(0, 0, 0)``.
  ``xea_factor`` / ``xcm_factor`` are chordwise fractions from the LE for section ``M`` and
  PanelAero coupling (``corner1 + xea_factor * l_m``), not an extra shift of beam nodes
  unless ``decouple_xea_from_aerogrid`` is True (see below).
- When ``decouple_xea_from_aerogrid=True``, the precomputed aerogrid stays at the reference
  ``aerogrid_xea_factor`` (LE at ``-aerogrid_xea_factor * chord``) while ``xea_factor`` drives
  section ``M``/``K`` and the coupling EA point. Beam nodes shift chordwise by
  ``(xea_factor - aerogrid_xea_factor) * chord`` so the structural shear center aligns with the
  fixed wing geometry.
- CG is offset from the shear center only in the section ``M`` matrix (``cg_offset_e``), not by
  translating node coordinates.
- For midchord-referenced cases (``beam_on_shear_center=False``), nodes follow the same geometry
  rule but plots/coupling may express ``xea_factor`` relative to midchord.
- Planform sweep [deg]: beam line in the X–Y plane, ``x(y) = beam_axis_x_offset + y * tan(sweep)``.
  ``sweep_angle_deg = 0`` gives a straight, unswept beam (``tan(0) = 0``).

Section 6×6 ``K`` / ``M`` use DOF order ``[u, v, w, θx, θy, θz]`` with **u** chordwise,
**v** spanwise, **w** thickness. By default they are mapped to global coordinates per element
via ``T6`` (see ``orient_element_section_from_local``): **v** follows the member (swept
elastic axis when ``sweep_angle_deg`` ≠ 0, with optional chord-locked **u** along +X).

When ``section_axes_along_global_span`` is True (``create_beam_model`` / config), **u**, **v**,
**w** stay aligned with global +X, +Y, +Z while nodes may still follow planform sweep.
"""

from __future__ import annotations

import numpy as np


def span_stations_y(n_stations: int, beam_length: float, sweep_angle_deg: float = 0.0) -> np.ndarray:
    """Spanwise coordinates for ``n_stations`` points from root to tip."""
    n = int(n_stations)
    if n < 1:
        raise ValueError(f"n_stations must be >= 1, got {n}")
    return np.linspace(0.0, float(beam_length) * np.cos(np.deg2rad(sweep_angle_deg)), n)


def planform_x_at_span(
    y: np.ndarray,
    sweep_angle_deg: float,
    beam_axis_x_offset: float = 0.0,
) -> np.ndarray:
    """Chordwise X coordinate of the beam elastic axis at span station(s) ``y``."""
    y = np.asarray(y, dtype=float)
    return float(beam_axis_x_offset) + y * np.tan(np.deg2rad(float(sweep_angle_deg)))


def beam_reference_node_positions(
    n_nodes: int,
    beam_length: float,
    *,
    sweep_angle_deg: float = 0.0,
    beam_axis_x_offset: float = 0.0,
    beam_axis_z_offset: float = 0.0,
) -> np.ndarray:
    """Return ``(n_nodes, 3)`` node positions in the reference frame."""
    y = span_stations_y(n_nodes, beam_length, sweep_angle_deg)
    x = planform_x_at_span(y, sweep_angle_deg, beam_axis_x_offset)
    z = np.full_like(y, float(beam_axis_z_offset))
    return np.column_stack([x, y, z])


def xea_decoupled_from_aerogrid(config) -> bool:
    """True when structural ``xea_factor`` may differ from the fixed aerogrid reference."""
    return bool(getattr(config, "decouple_xea_from_aerogrid", False))


def aerogrid_xea_factor(config) -> float:
    """Chordwise fraction from LE used to build / align the aerogrid (CAERO1, Qjj)."""
    if xea_decoupled_from_aerogrid(config):
        ref = getattr(config, "aerogrid_xea_factor", None)
        if ref is not None:
            return float(ref)
    return float(config.xea_factor)


def coupling_xea_factor(config) -> float:
    """Chordwise fraction from LE for the elastic axis on the aerogrid panels (Z, G_f)."""
    return float(config.xea_factor)


def structural_xea_chordwise_offset(config) -> float:
    """
    Extra chordwise shift [m] of beam nodes when the aerogrid stays at ``aerogrid_xea_factor``.

    With a fixed aerogrid (LE at ``-aerogrid_xea_factor * c``), the shear center of a section
    with structural ``xea_factor`` lies at ``(xea_factor - aerogrid_xea_factor) * c`` in global X.
    """
    if not xea_decoupled_from_aerogrid(config):
        return 0.0
    chord = float(getattr(config, "chord", 0.0))
    return (coupling_xea_factor(config) - aerogrid_xea_factor(config)) * chord


def beam_geometry_from_config(config) -> dict[str, float]:
    """Read unified beam reference geometry from an ``AnalysisConfig`` instance."""
    x_offset = float(getattr(config, "beam_axis_x_offset", 0.0) or 0.0)
    x_offset += structural_xea_chordwise_offset(config)
    return {
        "sweep_angle_deg": float(getattr(config, "sweep_angle_deg", 0.0)),
        "beam_axis_x_offset": x_offset,
        "beam_axis_z_offset": float(getattr(config, "beam_axis_z_offset", 0.0) or 0.0),
    }


# Cases that use LE-referenced xea_factor as shear-center location (beam at origin = SC).
_SC_REFERENCED_CASE_NAMES = frozenset({
    "ABRAMSON1965",
    "GOLAND",
    "GOLANDfullspan",
    "grid_conv",
    "Model1",
    "Model2-bluntbased",
})


def beam_on_shear_center(config) -> bool:
    """
    True when beam nodes are on the shear-center axis and ``xea_factor`` is the
    chordwise fraction from the leading edge to the shear center (PanelAero convention).
    """
    explicit = getattr(config, "beam_on_shear_center", None)
    if explicit is not None:
        return bool(explicit)
    if getattr(config, "structural_model_type", None) == "custom":
        return True
    return getattr(config, "name", None) in _SC_REFERENCED_CASE_NAMES
