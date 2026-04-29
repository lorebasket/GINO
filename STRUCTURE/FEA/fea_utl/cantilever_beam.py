import numpy as np

def cantilever_beam(Matrix, total_dof, constrained_dofs=None, dof_per_node=6):

    if constrained_dofs is None:
        constrained_dofs = list(range(dof_per_node))  # clamp first 6 DOFs by default
    
    # Get all DOF indices
    all_dofs = np.arange(total_dof, dtype=int)
    
    # Create mask for free DOFs
    mask = np.ones(total_dof, dtype=bool)
    mask[np.array(constrained_dofs, dtype=int)] = False
    free_dofs = all_dofs[mask]
    
    # For square matrices, reduce both dimensions
    if Matrix.shape[0] == Matrix.shape[1]:
        Matrix_reduced = Matrix[np.ix_(free_dofs, free_dofs)]
    # For rectangular matrices, only reduce the columns
    elif Matrix.shape[1] == total_dof:
        Matrix_reduced = Matrix[:, free_dofs]
    else:
        raise ValueError(f"Matrix shape {Matrix.shape} is not compatible with total_dof={total_dof}")
    
    return Matrix_reduced