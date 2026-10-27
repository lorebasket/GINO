# FEA/fea_utl/multibody_assembly.py
"""
Multibody beam assembly utility.

Assembles multiple beam_model dicts (each already expressed in the *global*
coordinate frame, with element K/M matrices already rotated) into a single
monolithic beam_model.

Connection between sub-beams is enforced by merging nodes that are
geometrically coincident (distance < tol).  This allows, e.g., the tip of the
arm sub-beam to be merged with the root of the foil sub-beam.

DOF ordering throughout: [u, v, w, θx, θy, θz]  (global X, Y, Z)
"""

import numpy as np


def assemble_multibody(sub_models, tol=1e-6):
    """
    Assemble a list of beam_model dicts into a single global beam_model.

    Parameters
    ----------
    sub_models : list of dict
        Each entry is a beam_model dict with keys 'nodes' and 'elements'.
        Node positions must already be in the **global** coordinate frame.
        Element 'stiffness' and 'mass' (6×6) must already be expressed in the
        **global** frame (i.e. rotated as needed before calling this function).
    tol : float
        Euclidean distance tolerance for merging coincident nodes [m].

    Returns
    -------
    global_model : dict
        Assembled beam_model with merged nodes and re-indexed elements.
        The returned dict has:
          - 'nodes'    : list of node dicts (position, index, clamped flag, …)
          - 'elements' : list of element dicts (nodes, stiffness, mass, length)
          - 'sub_model_node_maps' : list of lists – for each sub-model, the
                                   mapping  local_node_idx → global_node_idx
    """

    global_nodes = []   # accumulated global node list
    global_elements = []
    sub_model_node_maps = []  # one list per sub-model

    for sm_idx, sm in enumerate(sub_models):
        local_nodes = sm['nodes']
        local_elements = sm['elements']

        # ----------------------------------------------------------------
        # Map every local node to a global index, merging coincidents.
        # ----------------------------------------------------------------
        local_to_global = {}
        for loc_idx, node in enumerate(local_nodes):
            pos = np.array(node['position'], dtype=float)
            merged = False

            for g_idx, g_node in enumerate(global_nodes):
                g_pos = np.array(g_node['position'], dtype=float)
                if np.linalg.norm(pos - g_pos) < tol:
                    # Coincident node — reuse existing global node.
                    local_to_global[loc_idx] = g_idx
                    # Propagate the 'clamped' flag if set on the local node.
                    if node.get('clamped', False):
                        global_nodes[g_idx]['clamped'] = True
                    merged = True
                    break

            if not merged:
                g_idx = len(global_nodes)
                # Copy node dict; position is a plain list so dict() is sufficient,
                # but copy it explicitly to avoid aliasing.
                new_node = dict(node)
                new_node['position'] = list(node['position'])
                new_node['index'] = g_idx
                global_nodes.append(new_node)
                local_to_global[loc_idx] = g_idx

        sub_model_node_maps.append(local_to_global)

        # ----------------------------------------------------------------
        # Re-index elements and append to the global list.
        # ----------------------------------------------------------------
        for elem in local_elements:
            loc_n1, loc_n2 = elem['nodes']
            g_n1 = local_to_global[loc_n1]
            g_n2 = local_to_global[loc_n2]

            # Deep-copy numpy arrays so the assembled model is independent of
            # the sub-model dicts (avoids aliased in-place modifications).
            new_elem = {}
            for k, v in elem.items():
                if isinstance(v, np.ndarray):
                    new_elem[k] = v.copy()
                else:
                    new_elem[k] = v
            new_elem['nodes'] = [g_n1, g_n2]
            global_elements.append(new_elem)

    # Re-number node indices to match list position (safety pass).
    for i, node in enumerate(global_nodes):
        node['index'] = i

    print(f"[multibody_assembly] Assembled {len(sub_models)} sub-models → "
          f"{len(global_nodes)} nodes, {len(global_elements)} elements "
          f"(after merging coincident nodes)")

    return {
        'nodes': global_nodes,
        'elements': global_elements,
        'sub_model_node_maps': sub_model_node_maps,
    }


# ---------------------------------------------------------------------------
# Coordinate-frame rotation helpers
# ---------------------------------------------------------------------------

def rotation_matrix_from_to(v_from, v_to):
    """
    Return the 3×3 rotation matrix R that rotates unit-vector *v_from* onto
    unit-vector *v_to* (Rodrigues' formula).

    Parameters
    ----------
    v_from : array_like (3,)  – reference direction (default beam axis = [0,1,0])
    v_to   : array_like (3,)  – actual  beam direction in global frame

    Returns
    -------
    R : np.ndarray (3,3)
    """
    v_from = np.asarray(v_from, dtype=float)
    v_to   = np.asarray(v_to,   dtype=float)
    v_from = v_from / np.linalg.norm(v_from)
    v_to   = v_to   / np.linalg.norm(v_to)

    dot = np.clip(np.dot(v_from, v_to), -1.0, 1.0)

    if np.isclose(dot, 1.0):
        return np.eye(3, dtype=float)

    if np.isclose(dot, -1.0):
        # 180° rotation – find a perpendicular axis
        perp = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(v_from, perp)) > 0.9:
            perp = np.array([0.0, 1.0, 0.0])
        axis = np.cross(v_from, perp)
        axis /= np.linalg.norm(axis)
        # Rodriguez for 180°:  R = 2 * axis⊗axis - I
        return 2.0 * np.outer(axis, axis) - np.eye(3, dtype=float)

    axis = np.cross(v_from, v_to)
    axis_norm = np.linalg.norm(axis)
    axis /= axis_norm
    angle = np.arccos(dot)
    c, s = np.cos(angle), np.sin(angle)
    K_skew = np.array([
        [    0.0, -axis[2],  axis[1]],
        [ axis[2],     0.0, -axis[0]],
        [-axis[1],  axis[0],     0.0],
    ], dtype=float)
    return c * np.eye(3) + s * K_skew + (1 - c) * np.outer(axis, axis)


def section_rotation_matrix(
    beam_direction_global,
    chord_direction_global,
    thickness_direction_global=None,
):
    """
    Orthonormal section basis in global coordinates (before dihedral / α rotations).

    Columns of ``R`` are local unit vectors ``[u, v, w]`` expressed in global axes:
      - **v** : shear / elastic axis (beam axis, spanwise along the member)
      - **u** : chordwise (from ``chord_direction_global``, made ⊥ v)
      - **w** : thickness (``u × v``, or from ``thickness_direction_global`` if given)

    Section ``K`` / ``M`` at build use DOF order ``[u, v, w, θx, θy, θz]`` with this basis.
    """
    v_dir = np.asarray(beam_direction_global, dtype=float)
    v_dir /= np.linalg.norm(v_dir)

    chord = np.asarray(chord_direction_global, dtype=float)
    cn = np.linalg.norm(chord)
    if cn < 1e-14:
        raise ValueError("chord_direction_global must be non-zero")
    chord /= cn

    u_dir = chord - np.dot(chord, v_dir) * v_dir
    un = np.linalg.norm(u_dir)
    if un < 1e-10:
        raise ValueError(
            "chord_direction_global is parallel to the beam axis; "
            "cannot build a chord–span–thickness triad"
        )
    u_dir /= un

    if thickness_direction_global is None:
        w_dir = np.cross(u_dir, v_dir)
    else:
        thick = np.asarray(thickness_direction_global, dtype=float)
        thick /= np.linalg.norm(thick)
        w_dir = thick - np.dot(thick, v_dir) * v_dir - np.dot(thick, u_dir) * u_dir
        wn = np.linalg.norm(w_dir)
        if wn < 1e-10:
            w_dir = np.cross(u_dir, v_dir)
        else:
            w_dir /= wn

    wn = np.linalg.norm(w_dir)
    if wn < 1e-10:
        raise ValueError("degenerate section basis (u, v, w)")
    w_dir /= wn

    return np.column_stack([u_dir, v_dir, w_dir])


def T6_from_section_rotation(R: np.ndarray) -> np.ndarray:
    """6×6 frame map from ``section_rotation_matrix`` (local → global)."""
    R = np.asarray(R, dtype=float)
    T = np.zeros((6, 6), dtype=float)
    T[:3, :3] = R
    T[3:, 3:] = R
    return T


def T6_from_beam_direction(beam_direction_global):
    """
    Build T using only the beam-axis direction (legacy / multibody links).

    Prefer ``T6_from_beam_and_chord`` when planform sweep fixes chord ⊥ span in space.
    """
    R = rotation_matrix_from_to(np.array([0.0, 1.0, 0.0]), beam_direction_global)
    return T6_from_section_rotation(R)


def T6_from_beam_and_chord(
    beam_direction_global,
    chord_direction_global,
    thickness_direction_global=None,
):
    """
    Section map with explicit chord / thickness references (swept-wing builds).

    Locks **u** to the chord reference and **v** to the member direction so ``EIxx``,
    ``EIzz``, ``GJ`` stay in the intended physical directions under planform sweep.
    """
    R = section_rotation_matrix(
        beam_direction_global,
        chord_direction_global,
        thickness_direction_global,
    )
    return T6_from_section_rotation(R)


def orient_element_section_global_reference(
    element: dict,
    K_loc: np.ndarray,
    M_loc: np.ndarray,
) -> None:
    """
    Map sectional ``K_loc``, ``M_loc`` with local axes fixed to global reference:

    **u** = +X (chord), **v** = +Y (span), **w** = +Z (thickness).

    Use when section properties are defined in the horizontal span frame rather
    than rotated to follow a swept elastic axis. Beam nodes may still follow
    planform sweep; element length remains the physical member length.
    """
    T6 = np.eye(6, dtype=float)
    K_loc = np.asarray(K_loc, dtype=float)
    M_loc = np.asarray(M_loc, dtype=float)
    element["section_stiffness_local"] = K_loc.copy()
    element["section_mass_local"] = M_loc.copy()
    element["stiffness"] = K_loc.copy()
    element["mass"] = M_loc.copy()
    element["T6"] = T6
    element["beam_dir_global"] = [0.0, 1.0, 0.0]
    element["chord_dir_global"] = [1.0, 0.0, 0.0]
    element["thickness_dir_global"] = [0.0, 0.0, 1.0]


def orient_element_section_from_local(
    element: dict,
    p1,
    p2,
    K_loc: np.ndarray,
    M_loc: np.ndarray,
    *,
    chord_direction_global=None,
    thickness_direction_global=None,
) -> np.ndarray:
    """
    Map sectional ``K_loc``, ``M_loc`` (local u=chord, v=beam, w=thickness) into global DOFs.

    When ``chord_direction_global`` is given, **v** follows ``p2 - p1`` and **u** is built
    from the chord reference (Gram–Schmidt), which preserves sweep / DLM chord alignment.
    Otherwise falls back to aligning only local +Y with the member (legacy).

    Stores ``section_stiffness_local`` / ``section_mass_local``, global ``stiffness`` /
    ``mass``, and ``T6`` / ``beam_dir_global`` on *element* (in place).

    Global assembly uses ``C_local = T6.T @ stiffness @ T6``, which recovers ``K_loc``.
    """
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    d = p2 - p1
    length = float(np.linalg.norm(d))
    if length < 1e-14:
        raise ValueError(
            f"Beam element {element.get('nodes')} has zero length; cannot orient section."
        )
    beam_dir = d / length
    if chord_direction_global is not None:
        T6 = T6_from_beam_and_chord(
            beam_dir,
            chord_direction_global,
            thickness_direction_global,
        )
        R = T6[:3, :3]
        element["chord_dir_global"] = R[:, 0].tolist()
        element["thickness_dir_global"] = R[:, 2].tolist()
    else:
        T6 = T6_from_beam_direction(beam_dir)
    K_loc = np.asarray(K_loc, dtype=float)
    M_loc = np.asarray(M_loc, dtype=float)
    element["section_stiffness_local"] = K_loc.copy()
    element["section_mass_local"] = M_loc.copy()
    element["stiffness"] = T6 @ K_loc @ T6.T
    element["mass"] = T6 @ M_loc @ T6.T
    element["T6"] = T6.copy()
    element["beam_dir_global"] = beam_dir.tolist()
    if chord_direction_global is None:
        element.pop("chord_dir_global", None)
        element.pop("thickness_dir_global", None)
    return beam_dir


def compose_T6_on_beam_model(beam_model, T6_delta: np.ndarray) -> None:
    """
    Left-multiply each element's ``T6`` by ``T6_delta`` (rigid frame change in global space).

    Updates stored global ``stiffness`` / ``mass`` from the unchanged local sections.
    """
    T6_delta = np.asarray(T6_delta, dtype=float)
    R = T6_delta[:3, :3]
    for elem in beam_model.get("elements", []):
        if "section_stiffness_local" not in elem:
            continue
        K_loc = np.asarray(elem["section_stiffness_local"], dtype=float)
        M_loc = np.asarray(elem["section_mass_local"], dtype=float)
        T6 = T6_delta @ np.asarray(elem["T6"], dtype=float)
        elem["T6"] = T6
        elem["stiffness"] = T6 @ K_loc @ T6.T
        elem["mass"] = T6 @ M_loc @ T6.T
        elem["beam_dir_global"] = (R @ np.asarray(elem["beam_dir_global"], dtype=float)).tolist()


def T6_for_element(element, nodes):
    """
    Element 6×6 map: use element['T6'] if set (multibody), else infer from
    (node2 − node1) so local beam +Y matches the physical member direction.
    """
    T6 = element.get("T6", None)
    if T6 is not None:
        return np.asarray(T6, dtype=float)
    n1_idx, n2_idx = element["nodes"]
    p1 = np.asarray(nodes[n1_idx]["position"], dtype=float)
    p2 = np.asarray(nodes[n2_idx]["position"], dtype=float)
    d = p2 - p1
    Lg = float(np.linalg.norm(d))
    if Lg < 1e-14:
        raise ValueError(
            f"Beam element {element.get('nodes')} has zero length; cannot infer T6."
        )
    beam_dir = d / Lg
    return T6_from_beam_direction(beam_dir)


def rotate_beam_model_to_global(beam_model, beam_direction_global):
    """
    Rotate all node positions and element K/M matrices of *beam_model* (which
    was built assuming beam axis = +Y) so that the beam axis is aligned with
    *beam_direction_global* in the global frame.

    Also stores the beam direction and 6×6 T matrix on each element so that
    the global assemblers can apply the full 12×12 frame transformation.

    This modifies *beam_model* **in place** and also returns it.

    Parameters
    ----------
    beam_model : dict
        Standard beam_model dict (nodes + elements).
    beam_direction_global : array_like (3,)
        Desired beam axis direction in the global coordinate frame.

    Returns
    -------
    beam_model : dict  (modified in place)
    """
    beam_dir = np.asarray(beam_direction_global, dtype=float)
    beam_dir = beam_dir / np.linalg.norm(beam_dir)

    T = T6_from_beam_direction(beam_dir)
    R = T[:3, :3]

    # Rotate node positions (they are given in local +Y beam frame).
    for node in beam_model['nodes']:
        pos = np.array(node['position'], dtype=float)
        node['position'] = (R @ pos).tolist()

    # Rotate element matrices: K_glob = T @ K_loc @ T^T (congruence transform).
    # Also store the rotation info so the 12×12 assembler can transform Ke_local.
    for elem in beam_model['elements']:
        K = np.array(elem['stiffness'], dtype=float)
        M = np.array(elem['mass'],      dtype=float)
        elem['stiffness'] = T @ K @ T.T
        elem['mass']      = T @ M @ T.T
        # Store the 6×6 T and beam direction for use by the global assemblers.
        elem['T6']           = T.copy()
        elem['beam_dir_global'] = beam_dir.tolist()

    return beam_model


def translate_beam_model(beam_model, offset):
    """
    Translate all node positions of *beam_model* by *offset*.

    Parameters
    ----------
    beam_model : dict
    offset     : array_like (3,)  – translation vector in the global frame [m]

    Returns
    -------
    beam_model : dict  (modified in place)
    """
    off = np.asarray(offset, dtype=float)
    for node in beam_model['nodes']:
        pos = np.array(node['position'], dtype=float)
        node['position'] = (pos + off).tolist()
    return beam_model
