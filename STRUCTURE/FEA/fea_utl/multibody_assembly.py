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


def T6_from_beam_direction(beam_direction_global):
    """
    Build the 6×6 transformation matrix T that maps DOFs from the local beam
    frame into the global frame.

    Parameters
    ----------
    beam_direction_global : array_like (3,)
        Unit vector of the beam axis in the global frame.

    Returns
    -------
    T : np.ndarray (6,6)
    """
    R = rotation_matrix_from_to(np.array([0.0, 1.0, 0.0]), beam_direction_global)
    T = np.zeros((6, 6), dtype=float)
    T[:3, :3] = R
    T[3:, 3:] = R
    return T


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
