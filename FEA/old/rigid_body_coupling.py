import numpy as np

def skew(vec):
      """Return the skew-symmetric matrix of a 3D vector."""
      x, y, z = vec
      return np.array([
            [ 0, -z,  y],
            [ z,  0, -x],
            [-y,  x,  0]
      ])

def build_rigid_body_coupling_matrix(node_positions, ref_point):
      """
      Build a coupling matrix T such that:
            u_beam = T @ u_rigid
      where:
            - u_beam is a (6*n_nodes,) vector (beam DOFs),
            - u_rigid is a (6,) vector (rigid body DOFs: [ux, uy, uz, rx, ry, rz]).

      Parameters:
            node_positions: ndarray of shape (n_nodes, 3)
            ref_point: ndarray of shape (3,) - reference point for rigid body rotations

      Returns:
            T: ndarray of shape (6 * n_nodes, 6)
      """
      n_nodes = node_positions.shape[0]
      T = np.zeros((6 * n_nodes, 6))

      for i in range(n_nodes):
            r_i = node_positions[i]
            r_rel = r_i - ref_point
            S = skew(r_rel)

            # Fill translational DOFs
            T[6*i : 6*i+3, 0:3] = np.eye(3)       # Rigid body translations
            T[6*i : 6*i+3, 3:6] = -S              # Rotations produce displacement

            # Fill rotational DOFs
            # Assume rigid body rotation applied directly to beam node rotation
            T[6*i+3 : 6*i+6, 3:6] = np.eye(3)

      return T
