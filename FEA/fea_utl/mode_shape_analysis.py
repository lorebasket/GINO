import numpy as np
import matplotlib.pyplot as plt


def _infer_plot_node_order(beam_model: dict) -> list[int]:
    """Return an ordered list of node indices to use for plotting.

    For simple beams, the node list is already ordered.
    For multibody models (e.g. tnz_multibody) we want to plot only the flexible
    arm spine, not the additional rigid foil nodes.

    Heuristic:
      - If the model provides `sub_beam_names` and contains 'arm_spline', assume
        arm nodes are the initial contiguous block up to the last node that
        belongs to the arm (currently: arm is assembled first).
      - Optionally respect a per-node marker `submodel` if present.
    """
    nodes = beam_model.get('nodes', [])
    if not nodes:
        return []

    # If nodes have an explicit submodel tag, use it.
    # NOTE: for correct plotting we need a *contiguous ordered chain*.
    # It's safer to take the arm nodes in their original order.
    submods = [n.get('submodel') for n in nodes]
    if any(s is not None for s in submods):
        arm_idxs = [i for i, s in enumerate(submods) if s in ('arm_spline', 'arm')]
        if arm_idxs:
            return arm_idxs

    # tnz_multibody convention: arm comes first, foils appended.
    # If the model tells us it’s multibody and includes an arm, take the first
    # block of nodes until we detect the first duplicate position jump pattern
    # typical of appended submodels.
    if beam_model.get('is_multibody') and 'arm_spline' in beam_model.get('sub_beam_names', []):
        # Preferred: the model should tell us how many arm nodes exist.
        arm_n_nodes = beam_model.get('arm_n_nodes')
        arm_n_elements = beam_model.get('arm_n_elements')
        if isinstance(arm_n_nodes, int) and arm_n_nodes > 1:
            return list(range(0, arm_n_nodes))
        if isinstance(arm_n_elements, int) and arm_n_elements > 0:
            return list(range(0, arm_n_elements + 1))

        # Fallback: detect first big discontinuity in inter-node distance.
        # (This is fragile for tnz_multibody because foil roots may be near the arm tip.)
        pos = np.array([n['position'] for n in nodes], dtype=float)
        diffs = np.linalg.norm(np.diff(pos, axis=0), axis=1)
        if len(diffs) > 0 and np.all(np.isfinite(diffs)):
            med = float(np.median(diffs))
            thresh = 20.0 * med if med > 0 else float('inf')
            jumps = np.where(diffs > thresh)[0]
            if len(jumps) > 0:
                cut = int(jumps[0] + 1)
                return list(range(0, cut))

    return list(range(len(nodes)))


def _expand_free_dofs_to_nodes(beam_model: dict, dry_vectors: np.ndarray, dof_per_node: int = 6) -> np.ndarray:
    """Expand eigenvectors defined on free DOFs back to per-node DOFs.

    The structural solver typically constrains DOFs of clamped nodes and returns
    eigenvectors only for free DOFs.

    This function rebuilds a (n_nodes*dof_per_node, n_modes) array in global
    node order, inserting zeros for constrained nodes.
    """
    n_nodes = len(beam_model.get('nodes', []))
    if n_nodes == 0:
        return dry_vectors

    clamped = [bool(n.get('clamped', False)) for n in beam_model['nodes']]
    free_node_indices = [i for i, c in enumerate(clamped) if not c]
    n_free_nodes = len(free_node_indices)
    expected = n_free_nodes * dof_per_node

    if dry_vectors.shape[0] != expected:
        # Don’t guess by padding/truncating silently; warn and fall back.
        print(
            "[plot_shapes] WARNING: dry_vectors length doesn’t match free-node DOF count. "
            f"Got {dry_vectors.shape[0]} rows, expected {expected} (= {n_free_nodes} free nodes × {dof_per_node}). "
            "Falling back to legacy mapping (assumes node0 clamped)."
        )
        return None

    n_modes = dry_vectors.shape[1]
    full = np.zeros((n_nodes * dof_per_node, n_modes), dtype=float)
    for j, node_idx in enumerate(free_node_indices):
        full[node_idx * dof_per_node:(node_idx + 1) * dof_per_node, :] = dry_vectors[
            j * dof_per_node:(j + 1) * dof_per_node, :
        ]
    return full


def plot_shapes(beam_length_or_model=None, n_elements_or_none=None, dry_vectors=None, num_modes=None, title_suffix="", beam_model=None):
    """
    Plot dry mode shapes (displacements and rotations) for each mode.

    Supports two calling conventions:

    Straight beam (original):
        plot_shapes(beam_length, n_elements, dry_vectors, num_modes, title_suffix="")

    Curved / SONATA beam (pass beam_model dict, n_elements_or_none is ignored):
        plot_shapes(beam_model, None, dry_vectors, num_modes, title_suffix="")
        OR (with keyword arguments):
        plot_shapes(beam_model=beam_model, dry_vectors=eigenvectors, num_modes=n_modes, title_suffix="...")

    For a curved beam the horizontal axis is the cumulative arc-length along
    the beam elastic axis (derived from node positions stored in beam_model).
    
    Args:
        beam_length_or_model: Scalar beam length (m) or beam_model dict with nodes
        n_elements_or_none: Number of elements (for straight beam) or None
        dry_vectors: Eigenvector matrix (n_dofs, num_modes)
        num_modes: Number of modes to plot
        title_suffix: Optional string to append to plot title (e.g., "(Pre-Reduction)")
        beam_model: (keyword-only) Alternative way to pass beam_model as named argument
    """
    # Handle keyword-only beam_model argument
    if beam_model is not None:
        beam_length_or_model = beam_model

    plot_node_indices = None
    full_vectors = None

    if isinstance(beam_length_or_model, dict):
        # --- Beam-model path: derive plotting x-axis and DOF mapping from model ---
        beam_model = beam_length_or_model

        # Decide which nodes we plot (arm-only for tnz_multibody)
        plot_node_indices = _infer_plot_node_order(beam_model)
        node_positions_all = np.array([n["position"] for n in beam_model["nodes"]], dtype=float)
        node_positions = node_positions_all[plot_node_indices, :]

        # Cumulative arc-length along the plotted node chain
        diffs = np.diff(node_positions, axis=0)
        seg_lengths = np.linalg.norm(diffs, axis=1)
        x_positions = np.concatenate(([0.0], np.cumsum(seg_lengths)))
        xlabel = "Arc-length along beam [m]"

        # Expand eigenvectors back to per-node DOFs using clamped flags
        full_vectors = _expand_free_dofs_to_nodes(beam_model, dry_vectors, dof_per_node=6)
    else:
        # --- Straight-beam path (original behaviour) ---
        beam_length = beam_length_or_model
        n_elements = n_elements_or_none
        x_positions = np.linspace(0, beam_length, n_elements + 1)
        xlabel = "Span position [m]"


    # =========================== MODE SHAPE ANALYSIS =========================== #
    # dry_vectors has shape (n_free_dofs, num_modes).
    # DOFs are laid out as [u0, v0, w0, φx0, φy0, φz0, u1, v1, ...] for each free node.
    # The root node (node 0) is constrained, so free DOFs start from node 1.
    # We prepend a zero to recover the clamped root value.
    DOF_PER_NODE = 6

    # Legacy fallback for cases where we can’t confidently reconstruct full vectors
    if full_vectors is None:
        full_vectors = dry_vectors
        plot_node_indices = list(range(len(x_positions)))

    # Create a figure with 2 rows (displacements and rotations) and 3 columns (x, y, z)
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    title_text = 'Mode Shape Analysis' + (f' {title_suffix}' if title_suffix else '')
    fig.suptitle(title_text, fontsize=16)

    # Plot displacements in the first row
    for mode in range(num_modes):  # Plot all computed modes

        # Extract per-node DOFs
        u_all = full_vectors[0::DOF_PER_NODE, mode]
        v_all = full_vectors[1::DOF_PER_NODE, mode]
        w_all = full_vectors[2::DOF_PER_NODE, mode]

        # Select only nodes we are plotting
        xi = u_all[plot_node_indices]
        yi = v_all[plot_node_indices]
        zi = w_all[plot_node_indices]

        # X displacement
        axs[0, 0].plot(x_positions, xi, label=f'Mode {mode+1}')
        axs[0, 0].set_title('X Displacement (u)')
        axs[0, 0].set_xlabel(xlabel)
        axs[0, 0].grid(True)

        # Y displacement
        axs[0, 1].plot(x_positions, yi, label=f'Mode {mode+1}')
        axs[0, 1].set_title('Y Displacement (v)')
        axs[0, 1].set_xlabel(xlabel)
        axs[0, 1].grid(True)

        # Z displacement
        axs[0, 2].plot(x_positions, zi, label=f'Mode {mode+1}')
        axs[0, 2].set_title('Z Displacement (w)')
        axs[0, 2].set_xlabel(xlabel)
        axs[0, 2].grid(True)


    # Plot rotations in the second row
    for mode in range(num_modes):

        phix_all = full_vectors[3::DOF_PER_NODE, mode]
        phiy_all = full_vectors[4::DOF_PER_NODE, mode]
        phiz_all = full_vectors[5::DOF_PER_NODE, mode]

        phi_x = phix_all[plot_node_indices]
        phi_y = phiy_all[plot_node_indices]
        phi_z = phiz_all[plot_node_indices]

        # X rotation (phi_x)
        axs[1, 0].plot(x_positions, phi_x, label=f'Mode {mode+1}')
        axs[1, 0].set_title('X Rotation (φx)')
        axs[1, 0].set_xlabel(xlabel)
        axs[1, 0].grid(True)

        # Y rotation (phi_y)
        axs[1, 1].plot(x_positions, phi_y, label=f'Mode {mode+1}')
        axs[1, 1].set_title('Y Rotation (φy)')
        axs[1, 1].set_xlabel(xlabel)
        axs[1, 1].grid(True)

        # Z rotation (phi_z)
        axs[1, 2].plot(x_positions, phi_z, label=f'Mode {mode+1}')
        axs[1, 2].set_title('Z Rotation (φz)')
        axs[1, 2].set_xlabel(xlabel)
        axs[1, 2].grid(True)


    # Add a single legend for all subplots
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.98, 0.98))

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for the suptitle
    
    # Save the plot to output_plots directory
    import os
    output_dir = os.path.join(os.path.dirname(__file__), '../../output_plots')
    os.makedirs(output_dir, exist_ok=True)
    plot_filename = os.path.join(output_dir, f'mode_shapes{title_suffix.replace(" ", "_")}.png')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Mode shapes plot saved to: {plot_filename}")
    
    # Display the plot without blocking execution
    # Use draw() instead of show() to display without blocking
    #plt.draw()
    #plt.pause(0.001)  # Minimal pause to allow event processing
    plt.show()