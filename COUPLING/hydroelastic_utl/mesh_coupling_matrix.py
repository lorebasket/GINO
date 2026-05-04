"""
Couple a 3D wet surface mesh (e.g. Capytaine hull) to beam FEM nodal DOFs.

Mirrors :class:`AeroGridToFEM` / ``build_z_matrix`` in ``aerogrid_coupling_matrix.py``: for each mesh
face, take the **normal component** of rigid kinematics at the face centroid, with spanwise
linear interpolation of the elastic-axis point between the two bracketing beam nodes (``Y``
axis by default).

Outputs ``Z_mesh`` with shape ``(n_faces, 6 * n_nodes)`` — same layout as the aerodynamic ``Z``
matrix (one scalar kinematic constraint per face / panel row).

Modal coupling (physical nodal eigenvectors as columns):

    Z_modal = Z_mesh @ Phi_full

with ``Phi_full`` shape ``(6 * n_nodes, n_modes)`` (zeros on constrained DOFs), as produced
internally for flutter (``dry_eigenvectors_full``).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "face_centroids_and_normals",
    "HydroMeshSurfaceToFEM",
    "main_hydro_mesh_coupling",
    "mesh_modal_coupling",
    "save_hydro_mesh_coupling_npz",
]


def _face_vertex_indices(face) -> list[int]:
    """Capytaine-style face row: triangle ``[a,b,c,a]`` or quad ``[a,b,c,d]``."""
    f = np.asarray(face, dtype=np.int64).ravel()
    if len(f) < 3:
        raise ValueError(f"Invalid face connectivity: {face}")
    if f[0] == f[-1]:
        return [int(f[0]), int(f[1]), int(f[2])]
    if len(f) >= 4 and f[2] == f[3]:
        return [int(f[0]), int(f[1]), int(f[2])]
    if len(f) >= 4:
        return [int(f[0]), int(f[1]), int(f[2]), int(f[3])]
    return [int(f[0]), int(f[1]), int(f[2])]


def face_centroids_and_normals(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Face centroid and outward unit normal from connectivity.

    Parameters
    ----------
    vertices : (N, 3) float
    faces : (M, 4) int
        Capytaine convention (quads or closed triangles).

    Returns
    -------
    centers : (M, 3), normals : (M, 3) unit normals (fallback ``[0,0,1]`` if degenerate).
    """
    v = np.asarray(vertices, dtype=float)
    fmat = np.asarray(faces, dtype=np.int64)
    centers = np.zeros((fmat.shape[0], 3), dtype=float)
    normals = np.zeros((fmat.shape[0], 3), dtype=float)
    for i in range(fmat.shape[0]):
        ix = _face_vertex_indices(fmat[i])
        pts = v[ix]
        centers[i] = pts.mean(axis=0)
        if len(ix) >= 3:
            nvec = np.cross(pts[1] - pts[0], pts[2] - pts[0])
            ln = np.linalg.norm(nvec)
            if ln > 1e-18:
                normals[i] = nvec / ln
            else:
                normals[i] = np.array([0.0, 0.0, 1.0])
        else:
            normals[i] = np.array([0.0, 0.0, 1.0])
    return centers, normals


def _spanwise_bracket(y_nodes_sorted_ctx, sort_idx, y_colloc: float, span_axis: int):
    """Return ``(i_node_lo, i_node_hi, xi, N1, N2)`` matching ``aerogrid_coupling_matrix.build_z_matrix``."""
    y_sorted = y_nodes_sorted_ctx
    j = int(np.searchsorted(y_sorted, y_colloc))
    j = int(np.clip(j, 1, len(y_sorted) - 1))
    i_lo = int(sort_idx[j - 1])
    i_hi = int(sort_idx[j])
    y1 = float(y_sorted[j - 1])
    y2 = float(y_sorted[j])
    if abs(y2 - y1) < 1e-12:
        xi = 0.5
    else:
        xi = float(np.clip((y_colloc - y1) / (y2 - y1), 0.0, 1.0))
    N1 = 1.0 - xi
    N2 = xi
    return i_lo, i_hi, xi, N1, N2


def _accumulate_normal_coupling_row(
    Z: np.ndarray,
    i_face: int,
    ctrl: np.ndarray,
    n_unit: np.ndarray,
    ea_point: np.ndarray,
    N1: float,
    N2: float,
    i_node_1: int,
    i_node_2: int,
) -> None:
    """Same rigid projection as ``AeroGridToFEM.build_z_matrix`` (translation + θ×r)."""
    n = np.asarray(n_unit, dtype=float).reshape(3)
    nn = np.linalg.norm(n)
    if nn > 1e-18:
        n = n / nn
    r_vec = np.asarray(ctrl, dtype=float).reshape(3) - np.asarray(ea_point, dtype=float).reshape(3)
    idx1 = i_node_1 * 6
    idx2 = i_node_2 * 6
    Z[i_face, idx1 + 0] += N1 * n[0]
    Z[i_face, idx1 + 1] += N1 * n[1]
    Z[i_face, idx1 + 2] += N1 * n[2]
    Z[i_face, idx2 + 0] += N2 * n[0]
    Z[i_face, idx2 + 1] += N2 * n[1]
    Z[i_face, idx2 + 2] += N2 * n[2]
    Z[i_face, idx1 + 3] += N1 * (n @ np.cross([1.0, 0.0, 0.0], r_vec))
    Z[i_face, idx1 + 4] += N1 * (n @ np.cross([0.0, 1.0, 0.0], r_vec))
    Z[i_face, idx1 + 5] += N1 * (n @ np.cross([0.0, 0.0, 1.0], r_vec))
    Z[i_face, idx2 + 3] += N2 * (n @ np.cross([1.0, 0.0, 0.0], r_vec))
    Z[i_face, idx2 + 4] += N2 * (n @ np.cross([0.0, 1.0, 0.0], r_vec))
    Z[i_face, idx2 + 5] += N2 * (n @ np.cross([0.0, 0.0, 1.0], r_vec))


class HydroMeshSurfaceToFEM:
    """Build ``Z_mesh``: wet face × beam nodal DOF coupling (normal kinematics)."""

    def __init__(self, beam_model: dict | None = None):
        self.beam_model = beam_model
        self.face_to_node_map: list[tuple[int, int]] = []
        self.face_xi_map: list[float] = []

    def build_z_matrix(
        self,
        face_centers: np.ndarray,
        face_normals: np.ndarray,
        node_positions: np.ndarray,
        *,
        beam_model: dict | None = None,
        debug: bool = True,
        span_axis: int = 1,
        ea_point_policy: str = "beam_interp",
    ) -> tuple[np.ndarray, list[tuple[int, int]], list[float]]:
        """
        Parameters
        ----------
        face_centers, face_normals
            ``(n_faces, 3)`` arrays (unit normals).
        node_positions
            ``(n_nodes, 3)`` beam node coordinates (same order as FEM DOFs).
        ea_point_policy
            ``"beam_interp"`` — elastic-axis reference is the spanwise linear interpolate of
            node positions (matches aerogrid fallback when lattice corners are unavailable).
        """
        # beam_model reserved for future per-face EA / chord offsets (cf. aerogrid lattice).

        fc = np.asarray(face_centers, dtype=float)
        fn = np.asarray(face_normals, dtype=float)
        pos = np.asarray(node_positions, dtype=float)
        n_faces = fc.shape[0]
        n_nodes = pos.shape[0]
        Z = np.zeros((n_faces, 6 * n_nodes), dtype=float)

        self.face_to_node_map = []
        self.face_xi_map = []

        span_axis = int(span_axis)
        y_nodes = pos[:, span_axis]
        sort_idx = np.argsort(y_nodes)
        y_sorted = y_nodes[sort_idx]

        for i_face in range(n_faces):
            ctrl = fc[i_face]
            normal = fn[i_face]
            y = float(ctrl[span_axis])
            i_lo, i_hi, xi, N1, N2 = _spanwise_bracket(y_sorted, sort_idx, y, span_axis)

            if ea_point_policy == "beam_interp":
                ea_point = N1 * pos[i_lo] + N2 * pos[i_hi]
            else:
                raise ValueError(f"Unknown ea_point_policy={ea_point_policy!r}")

            _accumulate_normal_coupling_row(Z, i_face, ctrl, normal, ea_point, N1, N2, i_lo, i_hi)
            self.face_to_node_map.append((i_lo, i_hi))
            self.face_xi_map.append(xi)

        if debug:
            distances = []
            for ctrl, (n1, n2), xi in zip(fc, self.face_to_node_map, self.face_xi_map):
                beam_pt = (1.0 - xi) * pos[n1] + xi * pos[n2]
                distances.append(float(np.linalg.norm(ctrl - beam_pt)))
            if distances:
                print("Hydro mesh coupling distance statistics (collocation ↔ beam interpolate):")
                print(f"  Mean: {np.mean(distances):.6f} m  Max: {np.max(distances):.6f} m")

        return Z, self.face_to_node_map, self.face_xi_map


def main_hydro_mesh_coupling(
    beam_model: dict,
    *,
    mesh=None,
    vertices: np.ndarray | None = None,
    faces: np.ndarray | None = None,
    coupling_diagnostics: bool = True,
    span_axis: int = 1,
    ea_point_policy: str = "beam_interp",
):
    """
    Convenience driver: build ``Z_mesh`` from a Capytaine ``Mesh`` or raw ``vertices`` / ``faces``.

    Returns
    -------
    Z_mesh : (n_faces, 6 * n_nodes)
    face_to_node_map, face_xi_map
        Same diagnostic meaning as ``panel_to_node_map`` / ``panel_xi_map`` for the aerogrid path.
    """
    node_positions = np.array([node["position"] for node in beam_model["nodes"]], dtype=float)

    if mesh is not None:
        fc = np.asarray(mesh.faces_centers, dtype=float)
        fn = np.asarray(mesh.faces_normals, dtype=float)
    elif vertices is not None and faces is not None:
        fc, fn = face_centroids_and_normals(vertices, faces)
        fn_row = np.linalg.norm(fn, axis=1, keepdims=True)
        fn_row = np.maximum(fn_row, 1e-18)
        fn = fn / fn_row
    else:
        raise ValueError("Provide either ``mesh`` (Capytaine Mesh) or both ``vertices`` and ``faces``.")

    coupler = HydroMeshSurfaceToFEM(beam_model)
    Z, mp, xi = coupler.build_z_matrix(
        fc,
        fn,
        node_positions,
        beam_model=beam_model,
        debug=coupling_diagnostics,
        span_axis=span_axis,
        ea_point_policy=ea_point_policy,
    )
    return Z, mp, xi


def save_hydro_mesh_coupling_npz(
    path: str,
    Z_mesh: np.ndarray,
    *,
    face_centers: np.ndarray | None = None,
    face_normals: np.ndarray | None = None,
    face_to_node_map: np.ndarray | None = None,
    face_xi: np.ndarray | None = None,
) -> None:
    """Write ``Z_mesh`` and optional diagnostics via ``numpy.savez_compressed``."""
    payload = {"Z_mesh": np.asarray(Z_mesh, dtype=float)}
    if face_centers is not None:
        payload["face_centers"] = np.asarray(face_centers, dtype=float)
    if face_normals is not None:
        payload["face_normals"] = np.asarray(face_normals, dtype=float)
    if face_to_node_map is not None:
        payload["face_to_node_map"] = np.asarray(face_to_node_map, dtype=np.int64)
    if face_xi is not None:
        payload["face_xi"] = np.asarray(face_xi, dtype=float)
    np.savez_compressed(path, **payload)


def mesh_modal_coupling(Z_mesh: np.ndarray, eigenvectors_full: np.ndarray) -> np.ndarray:
    """
    Project physical coupling to modal coordinates:

        Z_modal = Z_mesh @ Phi

    Parameters
    ----------
    Z_mesh : (n_faces, n_phys_dof)
    eigenvectors_full : (n_phys_dof, n_modes)
        Full-space eigenvectors (zeros on constrained DOFs), columns = modes.

    Returns
    -------
    Z_modal : (n_faces, n_modes)
    """
    Z_mesh = np.asarray(Z_mesh, dtype=float)
    Phi = np.asarray(eigenvectors_full, dtype=float)
    if Z_mesh.shape[1] != Phi.shape[0]:
        raise ValueError(
            f"DOF mismatch: Z_mesh columns {Z_mesh.shape[1]} vs Phi rows {Phi.shape[0]}"
        )
    return Z_mesh @ Phi
