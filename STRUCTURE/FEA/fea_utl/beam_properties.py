import numpy as np

from FEA.fea_utl.beam_geometry import beam_geometry_from_config, beam_reference_node_positions
from FEA.fea_utl.multibody_assembly import (
    orient_element_section_from_local,
    orient_element_section_global_reference,
)


def create_beam_model(
    K,
    M,
    beam_length,
    n_el,
    *,
    sweep_angle_deg: float = 0.0,
    beam_axis_x_offset: float = 0.0,
    beam_axis_z_offset: float = 0.0,
    cg_offset_e: float | None = None,
    beam_on_shear_center: bool = False,
    section_axes_along_global_span: bool = False,
    agard_theory: bool = False,
    flutter_benchmark: bool = False,
    sonata: bool = False,
):
    """
    Create a beam model with given stiffness and mass properties.

    Section ``K`` and ``M`` are defined in the local beam frame (u chordwise, v along the
    beam axis, w thickness). Nodes lie on the **shear center** when ``beam_on_shear_center``
    is True (flutter benchmarks; root at the origin when offsets are zero). CG offset
    ``cg_offset_e`` (from SC to CG along chord) enters only via ``M`` coupling, not node
    positions. Each element's matrices are mapped to global coordinates with ``T6``
    from the member direction and global chord/thickness references (sweep-aware).

    Structural dihedral about +X is applied later in ``main.py`` (``post_dihedral_utils``).

    Parameters
    ----------
    sweep_angle_deg : float
        Planform sweep in the X–Y plane [deg]. Zero → straight spanwise axis.
    beam_axis_x_offset, beam_axis_z_offset : float
        Position of the elastic axis at the root (y = 0) in the reference frame [m].
    cg_offset_e : float, optional
        If given, prints CG offset from the elastic axis along local chord (diagnostics).
    section_axes_along_global_span : bool
        If True, section ``K``/``M`` stay in the global reference frame (u=+X, v=+Y,
        w=+Z) instead of rotating with each swept member. Nodes may still follow
        ``sweep_angle_deg``; only the section-axis map changes.
    """
    n_nodes = n_el + 1

    nodes_stiffness = []
    nodes_mass = []

    is_dict = isinstance(K, dict)

    if is_dict:
        if agard_theory:
            missing_k = [f"e{i}" for i in range(n_el) if f"e{i}" not in K]
            missing_m = [f"e{i}" for i in range(n_el) if f"e{i}" not in M]
            if missing_k or missing_m:
                raise KeyError(
                    f"Per-element K/M missing keys. "
                    f"Missing K: {missing_k}; Missing M: {missing_m}"
                )
            for i in range(n_el):
                nodes_stiffness.append(K[f"e{i}"])
                nodes_mass.append(M[f"e{i}"])
            print(f"AGARD per-element mode: received {n_el} K/M matrices.")

        elif sonata or flutter_benchmark:
            n_portions = len(K)
            portion_size = n_el // n_portions
            remaining_elements = n_el % n_portions
            sorted_keys = sorted(K.keys())

            for i, key in enumerate(sorted_keys):
                n_elements_in_portion = portion_size + (1 if i < remaining_elements else 0)
                for _ in range(n_elements_in_portion):
                    nodes_stiffness.append(K[key])
                    nodes_mass.append(M[key])

            assert len(nodes_stiffness) == n_el, "Mismatch in stiffness assignment"

            print("Beam portions (automatic):")
            for i, key in enumerate(sorted_keys):
                n_elements_in_portion = portion_size + (1 if i < remaining_elements else 0)
                print(f" Portion {i + 1}: {n_elements_in_portion} elements assigned to key '{key}'")
    else:
        if agard_theory or sonata or flutter_benchmark:
            print("Warning: K and M are not dictionaries. Using single matrix for all elements.")

        for _ in range(n_el):
            nodes_stiffness.append(np.asarray(K, dtype=float).copy())
            nodes_mass.append(np.asarray(M, dtype=float).copy())

    positions = beam_reference_node_positions(
        n_nodes,
        beam_length,
        sweep_angle_deg=sweep_angle_deg,
        beam_axis_x_offset=beam_axis_x_offset,
        beam_axis_z_offset=beam_axis_z_offset,
    )

    if cg_offset_e is not None:
        axis = "shear center" if beam_on_shear_center else "beam reference"
        print(
            f"Beam nodes on {axis} (x_offset={beam_axis_x_offset:g} m); "
            f"CG offset e={float(cg_offset_e):.6f} m from SC along local chord in section M"
        )
    # Reference chord/thickness in global axes (pre-dihedral build frame).
    chord_ref = np.array([1.0, 0.0, 0.0])
    thickness_ref = np.array([0.0, 0.0, 1.0])
    use_global_span_axes = bool(section_axes_along_global_span)
    use_chord_locked = (
        not use_global_span_axes and abs(float(sweep_angle_deg)) > 1e-12
    )

    if use_global_span_axes:
        print(
            "Section K/M: fixed global frame (u=+X chord, v=+Y span, w=+Z); "
            "beam nodes may follow planform sweep"
        )
        if abs(float(sweep_angle_deg)) > 1e-12:
            print(
                f"  Planform sweep: {float(sweep_angle_deg):g} deg "
                f"(geometry only; section axes not rotated with elastic axis)"
            )
    elif use_chord_locked:
        print(
            f"Planform sweep: {float(sweep_angle_deg):g} deg "
            f"(section K/M: v along member, u from global chord, w from u×v)"
        )
    elif cg_offset_e is not None:
        axis = "shear center" if beam_on_shear_center else "beam reference"
        print(
            f"Beam on {axis} at root (0,0,0); "
            f"CG offset e={float(cg_offset_e):.6f} m from SC in section M only"
        )

    nodes = []
    for i in range(n_nodes):
        nodes.append({
            "position": positions[i].tolist(),
            "index": i,
            "stiffness": nodes_stiffness[i - 1] if 0 < i <= n_el else None,
            "mass": nodes_mass[i - 1] if 0 < i <= n_el else None,
        })

    elements = []
    for i in range(n_el):
        p1 = positions[i]
        p2 = positions[i + 1]
        L_e = float(np.linalg.norm(p2 - p1))
        K_loc = nodes_stiffness[i]
        M_loc = nodes_mass[i]
        elem = {
            "nodes": [i, i + 1],
            "length": L_e,
        }
        if use_global_span_axes:
            orient_element_section_global_reference(elem, K_loc, M_loc)
        else:
            orient_element_section_from_local(
                elem,
                p1,
                p2,
                K_loc,
                M_loc,
                chord_direction_global=chord_ref if use_chord_locked else None,
                thickness_direction_global=thickness_ref if use_chord_locked else None,
            )
        elements.append(elem)

    geometry = {
        "sweep_angle_deg": float(sweep_angle_deg),
        "beam_axis_x_offset": float(beam_axis_x_offset),
        "beam_axis_z_offset": float(beam_axis_z_offset),
        "chord_direction_global": chord_ref.tolist(),
        "thickness_direction_global": thickness_ref.tolist(),
        "section_axes_chord_locked": use_chord_locked,
        "section_axes_along_global_span": use_global_span_axes,
    }
    if cg_offset_e is not None:
        geometry["cg_offset_e"] = float(cg_offset_e)
    geometry["beam_on_shear_center"] = bool(beam_on_shear_center)

    return {
        "nodes": nodes,
        "elements": elements,
        "geometry": geometry,
    }


def create_beam_model_from_config(K, M, beam_length, n_el, config, **kwargs):
    """Convenience wrapper: geometry from ``config``, other args forwarded."""
    geo = beam_geometry_from_config(config)
    return create_beam_model(K, M, beam_length, n_el, **geo, **kwargs)
