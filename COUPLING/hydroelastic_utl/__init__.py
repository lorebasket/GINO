"""
hydroelastic_utl package — PK solver, coupling matrix, post-processing, and RFA utilities.
"""

from .mesh_coupling_matrix import (
    HydroMeshSurfaceToFEM,
    face_centroids_and_normals,
    main_hydro_mesh_coupling,
    mesh_modal_coupling,
    save_hydro_mesh_coupling_npz,
)
