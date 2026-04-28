import numpy as np
import matplotlib.pyplot as plt


def plot_shapes_fullspan(beam_model, dry_vectors):
    """
    Plot mode shapes for full-span beam with center clamping.
    The beam spans from -L to +L with center node clamped at y=0.
    
    Parameters:
        beam_model: Dictionary containing 'nodes' with position information
        dry_vectors: Mode shape vectors (free DOFs only)
    """
    # Extract y-positions from beam model
    nodes = beam_model['nodes']
    n_nodes = len(nodes)
    y_positions = np.array([node['position'][1] for node in nodes])
    
    # Find the clamped node index
    clamped_idx = None
    for i, node in enumerate(nodes):
        if node.get('clamped', False):
            clamped_idx = i
            break
    
    # Create full eigenvector array with zeros at constrained DOFs
    constrained_node_indices = set()
    
    if clamped_idx is not None:
        constrained_node_indices.add(clamped_idx)
        print(f"Full-span plot: Center node {clamped_idx} at y={y_positions[clamped_idx]:.4f} is clamped")
    
    # Build full eigenvectors including constrained DOFs
    dry_vectors_full = np.zeros((n_nodes * 6, dry_vectors.shape[1]))
    
    # Map free DOFs back to full DOF array
    free_dof_idx = 0
    for node_idx in range(n_nodes):
        for local_dof in range(6):
            global_dof = node_idx * 6 + local_dof
            if node_idx in constrained_node_indices:
                # Constrained: leave as zero
                pass
            else:
                # Free: copy from dry_vectors
                dry_vectors_full[global_dof, :] = dry_vectors[free_dof_idx, :]
                free_dof_idx += 1
    
    # =========================== MODE SHAPE ANALYSIS =========================== #
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Mode Shape Analysis (Full-Span, Center-Clamped)', fontsize=16)

    # Plot displacements and rotations
    for mode in range(min(5, dry_vectors.shape[1])):  # First 5 modes or less
        # Extract displacements for all nodes
        xi = dry_vectors_full[0::6, mode]
        yi = dry_vectors_full[1::6, mode]
        zi = dry_vectors_full[2::6, mode]
        
        # Extract rotations for all nodes
        phi_x = dry_vectors_full[3::6, mode]
        phi_y = dry_vectors_full[4::6, mode]
        phi_z = dry_vectors_full[5::6, mode]

        # X displacement
        axs[0, 0].plot(y_positions, xi, label=f'Mode {mode+1}', marker='o', markersize=2)
        axs[0, 0].set_title('X Displacement (u)')
        axs[0, 0].set_xlabel('Spanwise position (m)')
        axs[0, 0].axvline(x=0, color='k', linestyle='--', alpha=0.3, label='Center clamp' if mode == 0 else '')
        axs[0, 0].grid(True)

        # Y displacement
        axs[0, 1].plot(y_positions, yi, label=f'Mode {mode+1}', marker='o', markersize=2)
        axs[0, 1].set_title('Y Displacement (v)')
        axs[0, 1].set_xlabel('Spanwise position (m)')
        axs[0, 1].axvline(x=0, color='k', linestyle='--', alpha=0.3)
        axs[0, 1].grid(True)

        # Z displacement
        axs[0, 2].plot(y_positions, zi, label=f'Mode {mode+1}', marker='o', markersize=2)
        axs[0, 2].set_title('Z Displacement (w)')
        axs[0, 2].set_xlabel('Spanwise position (m)')
        axs[0, 2].axvline(x=0, color='k', linestyle='--', alpha=0.3)
        axs[0, 2].grid(True)

        # X rotation (phi_x)
        axs[1, 0].plot(y_positions, phi_x, label=f'Mode {mode+1}', marker='o', markersize=2)
        axs[1, 0].set_title('X Rotation (φx)')
        axs[1, 0].set_xlabel('Spanwise position (m)')
        axs[1, 0].axvline(x=0, color='k', linestyle='--', alpha=0.3)
        axs[1, 0].grid(True)

        # Y rotation (phi_y)
        axs[1, 1].plot(y_positions, phi_y, label=f'Mode {mode+1}', marker='o', markersize=2)
        axs[1, 1].set_title('Y Rotation (φy)')
        axs[1, 1].set_xlabel('Spanwise position (m)')
        axs[1, 1].axvline(x=0, color='k', linestyle='--', alpha=0.3)
        axs[1, 1].grid(True)

        # Z rotation (phi_z)
        axs[1, 2].plot(y_positions, phi_z, label=f'Mode {mode+1}', marker='o', markersize=2)
        axs[1, 2].set_title('Z Rotation (φz)')
        axs[1, 2].set_xlabel('Spanwise position (m)')
        axs[1, 2].axvline(x=0, color='k', linestyle='--', alpha=0.3)
        axs[1, 2].grid(True)

    # Add a single legend for all subplots
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.98, 0.98))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
