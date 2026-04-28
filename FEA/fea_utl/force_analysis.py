"""
Utilities for computing reaction forces and displacements from structural analysis.

Key Concepts:
- Reaction forces computed at CONSTRAINTS (clamped root node)
- Displacements extracted along the arm span
- Internal forces are NOT computed (conceptually incorrect for cantilever beams)
"""

import numpy as np
import pandas as pd


def compute_reaction_forces(u_full, K_global, constrained_dofs, beam_model, dof_per_node=6):
    """
    Compute reaction forces at the CONSTRAINED nodes (root support).
    
    This is the correct approach: reaction forces appear only at constraints.
    For a cantilever: F_reaction = K_global @ u_full at the clamped end.
    
    Args:
        u_full: Full displacement vector (total_dof,)
        K_global: Global stiffness matrix (total_dof, total_dof)
        constrained_dofs: List of constrained DOF indices (typically 0-5 for first node)
        beam_model: Beam model dictionary with 'nodes'
        dof_per_node: Degrees of freedom per node (default: 6)
    
    Returns:
        dict: {
            'node_index': clamped node index,
            'position': [x, y, z],
            'reaction_forces': [Fx, Fy, Fz],
            'reaction_moments': [Mx, My, Mz]
        }
    """
    # Compute global reaction force vector: f_reaction = K @ u
    f_global = K_global @ u_full
    
    # Extract reaction forces at constrained DOFs
    reaction_forces = np.zeros(3)
    reaction_moments = np.zeros(3)
    
    # Typically constrained_dofs are [0, 1, 2, 3, 4, 5] for first node
    clamped_node_idx = constrained_dofs[0] // dof_per_node
    
    if len(constrained_dofs) >= 3:
        reaction_forces[0] = f_global[constrained_dofs[0]]  # Fx
        reaction_forces[1] = f_global[constrained_dofs[1]]  # Fy
        reaction_forces[2] = f_global[constrained_dofs[2]]  # Fz
    
    if len(constrained_dofs) >= 6:
        reaction_moments[0] = f_global[constrained_dofs[3]]  # Mx
        reaction_moments[1] = f_global[constrained_dofs[4]]  # My
        reaction_moments[2] = f_global[constrained_dofs[5]]  # Mz
    
    # Get clamped node position
    node_data = beam_model['nodes'][clamped_node_idx]
    if isinstance(node_data, dict):
        node_pos = np.array(node_data['position'])
    else:
        node_pos = np.array(node_data)
    
    return {
        'node_index': clamped_node_idx,
        'position': node_pos,
        'reaction_forces': reaction_forces,
        'reaction_moments': reaction_moments
    }


def compute_displacements_along_span(u_full, beam_model, node_indices=None, dof_per_node=6):
    """
    Extract displacements and rotations at multiple nodes along the arm span.
    
    Args:
        u_full: Full displacement vector (total_dof,)
        beam_model: Beam model dictionary with 'nodes'
        node_indices: List of node indices. If None, compute for all nodes.
        dof_per_node: Degrees of freedom per node (default: 6)
    
    Returns:
        DataFrame with displacements at each node
    """
    if node_indices is None:
        node_indices = range(len(beam_model['nodes']))
    
    results = []
    for node_idx in node_indices:
        dof_start = node_idx * dof_per_node
        dof_end = dof_start + dof_per_node
        
        # Get nodal displacements and rotations
        u_node = u_full[dof_start:dof_end]
        
        # Get node position
        node_data = beam_model['nodes'][node_idx]
        if isinstance(node_data, dict):
            node_pos = np.array(node_data['position'])
        else:
            node_pos = np.array(node_data)
        
        # Get y-position (spanwise)
        y_pos = node_pos[1]
        
        results.append({
            'node_idx': node_idx,
            'y_pos': y_pos,
            'x_pos': node_pos[0],
            'z_pos': node_pos[2],
            'disp_x': u_node[0],
            'disp_y': u_node[1],
            'disp_z': u_node[2],
            'rot_x': u_node[3],
            'rot_y': u_node[4],
            'rot_z': u_node[5],
        })
    
    return pd.DataFrame(results)


def get_tip_displacement(u_full, beam_model, dof_per_node=6):
    """
    Get tip node displacement and rotation.
    
    Args:
        u_full: Full displacement vector
        beam_model: Beam model dictionary
        dof_per_node: Degrees of freedom per node
    
    Returns:
        dict with tip displacement data
    """
    tip_idx = len(beam_model['nodes']) - 1
    
    dof_start = tip_idx * dof_per_node
    dof_end = dof_start + dof_per_node
    
    u_tip = u_full[dof_start:dof_end]
    
    # Get tip position
    node_data = beam_model['nodes'][tip_idx]
    if isinstance(node_data, dict):
        tip_pos = np.array(node_data['position'])
    else:
        tip_pos = np.array(node_data)
    
    return {
        'node_index': tip_idx,
        'position': tip_pos,
        'displacements': u_tip[:3],
        'rotations': u_tip[3:],
        'displacement_magnitude': np.linalg.norm(u_tip[:3]),
        'rotation_magnitude': np.linalg.norm(u_tip[3:])
    }


def print_reaction_forces(u_full, K_global, constrained_dofs, beam_model, config=None):
    """
    Print reaction forces at the clamped root in formatted output.
    
    Args:
        u_full: Full displacement vector
        K_global: Global stiffness matrix
        constrained_dofs: List of constrained DOF indices
        beam_model: Beam model dictionary
        config: Analysis config object
    
    Returns:
        Dictionary with reaction force data
    """
    reaction_data = compute_reaction_forces(u_full, K_global, constrained_dofs, beam_model)
    
    print(f"\n{'-'*70}")
    print(f"REACTION FORCES AT ROOT (CLAMPED SUPPORT)")
    print(f"{'-'*70}")
    print(f"Node index: {reaction_data['node_index']}")
    print(f"Position: {reaction_data['position']}")
    
    print(f"\nREACTION FORCES:")
    print(f"  Fx: {reaction_data['reaction_forces'][0]:+.6e} N")
    print(f"  Fy: {reaction_data['reaction_forces'][1]:+.6e} N")
    print(f"  Fz: {reaction_data['reaction_forces'][2]:+.6e} N")
    print(f"  |F|: {np.linalg.norm(reaction_data['reaction_forces']):.6e} N")
    
    print(f"\nREACTION MOMENTS:")
    print(f"  Mx: {reaction_data['reaction_moments'][0]:+.6e} N⋅m")
    print(f"  My: {reaction_data['reaction_moments'][1]:+.6e} N⋅m")
    print(f"  Mz: {reaction_data['reaction_moments'][2]:+.6e} N⋅m")
    print(f"  |M|: {np.linalg.norm(reaction_data['reaction_moments']):.6e} N⋅m")
    
    print(f"{'-'*70}\n")
    return reaction_data


def print_tip_displacement(u_full, beam_model, config=None):
    """
    Print tip node displacement in formatted output.
    
    Args:
        u_full: Full displacement vector
        beam_model: Beam model dictionary
        config: Analysis config object
    
    Returns:
        Dictionary with tip displacement data
    """
    tip_data = get_tip_displacement(u_full, beam_model)
    
    print(f"\n{'-'*70}")
    print(f"TIP NODE DISPLACEMENTS")
    print(f"{'-'*70}")
    print(f"Node index: {tip_data['node_index']}")
    print(f"Original position: {tip_data['position']}")
    
    print(f"\nDISPLACEMENTS:")
    print(f"  ux: {tip_data['displacements'][0]:+.6e} m")
    print(f"  uy: {tip_data['displacements'][1]:+.6e} m")
    print(f"  uz: {tip_data['displacements'][2]:+.6e} m")
    print(f"  |u|: {tip_data['displacement_magnitude']:.6e} m")
    
    print(f"\nROTATIONS:")
    print(f"  θx: {tip_data['rotations'][0]:+.6e} rad ({np.rad2deg(tip_data['rotations'][0]):+.3f}°)")
    print(f"  θy: {tip_data['rotations'][1]:+.6e} rad ({np.rad2deg(tip_data['rotations'][1]):+.3f}°)")
    print(f"  θz: {tip_data['rotations'][2]:+.6e} rad ({np.rad2deg(tip_data['rotations'][2]):+.3f}°)")
    print(f"  |θ|: {tip_data['rotation_magnitude']:.6e} rad")
    
    print(f"{'-'*70}\n")
    return tip_data


def save_displacements_to_csv(u_full, beam_model, output_path):
    """
    Save nodal displacements and rotations to CSV file.
    
    Args:
        u_full: Full displacement vector
        beam_model: Beam model dictionary
        output_path: Path to save CSV file
    """
    disp_df = compute_displacements_along_span(u_full, beam_model)
    disp_df.to_csv(output_path, index=False)
    print(f"✓ Displacements saved to: {output_path}")
    return disp_df


def plot_displacement_distribution(u_full, beam_model, output_dir=None):
    """
    Create plots showing displacement distribution along arm span.
    
    Args:
        u_full: Full displacement vector
        beam_model: Beam model dictionary
        output_dir: Directory to save plots
    
    Returns:
        DataFrame with displacement data
    """
    import os
    import matplotlib.pyplot as plt
    
    disp_df = compute_displacements_along_span(u_full, beam_model)
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Create figure with subplots for displacements and rotations
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Displacement and Rotation Distribution Along Arm Span', 
                 fontsize=14, fontweight='bold')
    
    # Filter for arm nodes (z_pos < 0 indicates arm, not foil)
    arm_mask = disp_df['z_pos'] < 0
    disp_df_arm = disp_df[arm_mask].copy()
    
    # Compute arc length along arm for proper x-axis
    # Use distance from root node (node 0)
    arm_nodes = disp_df_arm.reset_index(drop=True)
    arm_nodes['arc_length'] = 0.0
    for i in range(1, len(arm_nodes)):
        prev_pos = np.array([arm_nodes.loc[i-1, 'x_pos'], 
                             arm_nodes.loc[i-1, 'y_pos'], 
                             arm_nodes.loc[i-1, 'z_pos']])
        curr_pos = np.array([arm_nodes.loc[i, 'x_pos'], 
                             arm_nodes.loc[i, 'y_pos'], 
                             arm_nodes.loc[i, 'z_pos']])
        distance = np.linalg.norm(curr_pos - prev_pos)
        arm_nodes.loc[i, 'arc_length'] = arm_nodes.loc[i-1, 'arc_length'] + distance
    
    disp_df_arm = arm_nodes
    
    # --- DISPLACEMENTS --- #
    # ux
    axes[0, 0].plot(disp_df_arm['arc_length'], disp_df_arm['disp_x'], 'b-o', linewidth=2, markersize=4)
    axes[0, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[0, 0].set_xlabel('Arc Length Along Arm [m]', fontsize=11)
    axes[0, 0].set_ylabel('ux [m]', fontsize=11)
    axes[0, 0].set_title('Lateral Displacement (X)', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # uy
    axes[0, 1].plot(disp_df_arm['arc_length'], disp_df_arm['disp_y'], 'g-o', linewidth=2, markersize=4)
    axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[0, 1].set_xlabel('Arc Length Along Arm [m]', fontsize=11)
    axes[0, 1].set_ylabel('uy [m]', fontsize=11)
    axes[0, 1].set_title('Spanwise Displacement (Y)', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # uz
    axes[0, 2].plot(disp_df_arm['arc_length'], disp_df_arm['disp_z'], 'r-o', linewidth=2, markersize=4)
    axes[0, 2].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[0, 2].set_xlabel('Arc Length Along Arm [m]', fontsize=11)
    axes[0, 2].set_ylabel('uz [m]', fontsize=11)
    axes[0, 2].set_title('Vertical Displacement (Z)', fontweight='bold')
    axes[0, 2].grid(True, alpha=0.3)
    
    # --- ROTATIONS --- #
    # θx (Roll)
    axes[1, 0].plot(disp_df_arm['arc_length'], np.rad2deg(disp_df_arm['rot_x']), 'b-s', linewidth=2, markersize=4)
    axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[1, 0].set_xlabel('Arc Length Along Arm [m]', fontsize=11)
    axes[1, 0].set_ylabel('θx [deg]', fontsize=11)
    axes[1, 0].set_title('Roll Rotation (X)', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # θy (Pitch)
    axes[1, 1].plot(disp_df_arm['arc_length'], np.rad2deg(disp_df_arm['rot_y']), 'g-s', linewidth=2, markersize=4)
    axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[1, 1].set_xlabel('Arc Length Along Arm [m]', fontsize=11)
    axes[1, 1].set_ylabel('θy [deg]', fontsize=11)
    axes[1, 1].set_title('Pitch Rotation (Y)', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    # θz (Yaw)
    axes[1, 2].plot(disp_df_arm['arc_length'], np.rad2deg(disp_df_arm['rot_z']), 'r-s', linewidth=2, markersize=4)
    axes[1, 2].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[1, 2].set_xlabel('Arc Length Along Arm [m]', fontsize=11)
    axes[1, 2].set_ylabel('θz [deg]', fontsize=11)
    axes[1, 2].set_title('Yaw Rotation (Z)', fontweight='bold')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_dir:
        plot_path = os.path.join(output_dir, 'displacement_distribution.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"✓ Displacement distribution plot saved to: {plot_path}")
    
    plt.show()
    return disp_df_arm


def plot_displacements_by_beam(u_full, beam_model, output_dir=None):
    """
    Plot displacements for each beam in a multibody structure separately.
    
    For tnz_multibody: arm_spline, foil_dx, foil_sx
    Includes undeformed beam geometry visualization.
    
    Args:
        u_full: Full displacement vector
        beam_model: Beam model dictionary
        output_dir: Directory to save plots
    
    Returns:
        dict with displacement DataFrames for each beam
    """
    import os
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    disp_df = compute_displacements_along_span(u_full, beam_model)
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Identify beams by z-coordinate ranges:
    # Arm: z_pos >= 0.0
    # Foils: z_pos <= 0.0
    
    arm_mask = disp_df['z_pos'] >= 0.0
    foil_mask = disp_df['z_pos'] <= 0.0
    
    disp_arm = disp_df[arm_mask].copy()
    disp_foils = disp_df[foil_mask].copy()
    
    # Further separate left/right foils by x-coordinate
    if len(disp_foils) > 0:
        foil_dx_mask = disp_foils['y_pos'] >= 0.001  # Right foil (positive y)
        foil_sx_mask = disp_foils['y_pos'] <= -0.01  # Left foil (negative y)
        
        disp_foil_dx = disp_foils[foil_dx_mask].copy().reset_index(drop=True)
        disp_foil_sx = disp_foils[foil_sx_mask].copy().reset_index(drop=True)
    else:
        disp_foil_dx = pd.DataFrame()
        disp_foil_sx = pd.DataFrame()
    
    disp_arm = disp_arm.copy() if len(disp_arm) > 0 else disp_arm
    disp_foil_dx = disp_foil_dx.copy() if len(disp_foil_dx) > 0 else disp_foil_dx
    disp_foil_sx = disp_foil_sx.copy() if len(disp_foil_sx) > 0 else disp_foil_sx
    
    # Create individual plots for each beam
    beams = {
        'arm_spline': (disp_arm, 'darkblue'),
        'foil_dx': (disp_foil_dx, 'darkgreen'),
        'foil_sx': (disp_foil_sx, 'darkred')
    }
    
    for beam_name, (disp_data, color) in beams.items():
        if len(disp_data) == 0:
            print(f"  ⚠ No nodes found for {beam_name}")
            continue
        
        fig = plt.figure(figsize=(18, 10))
        fig.suptitle(f'Displacement and Rotation - {beam_name.upper()}', 
                     fontsize=14, fontweight='bold')
        
        # Create GridSpec for better layout control
        import matplotlib.gridspec as gridspec
        gs = gridspec.GridSpec(2, 4, figure=fig)
        
        # Convert rotations to degrees
        disp_data['rot_x_deg'] = np.rad2deg(disp_data['rot_x'])
        disp_data['rot_y_deg'] = np.rad2deg(disp_data['rot_y'])
        disp_data['rot_z_deg'] = np.rad2deg(disp_data['rot_z'])
        
        # --- 3D UNDEFORMED GEOMETRY --- #
        ax_3d = fig.add_subplot(gs[:, 0], projection='3d')
        ax_3d.plot(disp_data['x_pos'], disp_data['y_pos'], disp_data['z_pos'], 
                   color=color, linewidth=2.5, marker='o', markersize=3, 
                   label='Undeformed')
        ax_3d.set_xlabel('X [m]', fontsize=10)
        ax_3d.set_ylabel('Y [m]', fontsize=10)
        ax_3d.set_zlabel('Z [m]', fontsize=10)
        ax_3d.set_title('Undeformed Beam Geometry', fontweight='bold', fontsize=11)
        ax_3d.grid(True, alpha=0.3)
        ax_3d.legend()
        
        # --- DISPLACEMENTS --- #
        ax1 = fig.add_subplot(gs[0, 1])
        ax1.plot(disp_data['y_pos'], disp_data['disp_x']*1000, 
                 color=color, marker='o', linewidth=2, markersize=4)
        ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax1.set_xlabel('Spine Position Y [m]', fontsize=10)
        ax1.set_ylabel('ux [mm]', fontsize=10)
        ax1.set_title('Lateral Displacement (X)', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.plot(disp_data['y_pos'], disp_data['disp_y']*1000,
                 color=color, marker='o', linewidth=2, markersize=4)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax2.set_xlabel('Spine Position Y [m]', fontsize=10)
        ax2.set_ylabel('uy [mm]', fontsize=10)
        ax2.set_title('Spanwise Displacement (Y)', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        ax3 = fig.add_subplot(gs[0, 3])
        ax3.plot(disp_data['y_pos'], disp_data['disp_z']*1000,
                 color=color, marker='o', linewidth=2, markersize=4)
        ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax3.set_xlabel('Spine Position Y [m]', fontsize=10)
        ax3.set_ylabel('uz [mm]', fontsize=10)
        ax3.set_title('Vertical Displacement (Z)', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # --- ROTATIONS --- #
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(disp_data['y_pos'], disp_data['rot_x_deg'],
                 color=color, marker='s', linewidth=2, markersize=4)
        ax4.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax4.set_xlabel('Spine Position Y [m]', fontsize=10)
        ax4.set_ylabel('θx [deg]', fontsize=10)
        ax4.set_title('Roll Rotation (X)', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.plot(disp_data['y_pos'], disp_data['rot_y_deg'],
                 color=color, marker='s', linewidth=2, markersize=4)
        ax5.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax5.set_xlabel('Spine Position Y [m]', fontsize=10)
        ax5.set_ylabel('θy [deg]', fontsize=10)
        ax5.set_title('Pitch Rotation (Y)', fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        ax6 = fig.add_subplot(gs[1, 3])
        ax6.plot(disp_data['y_pos'], disp_data['rot_z_deg'],
                 color=color, marker='s', linewidth=2, markersize=4)
        ax6.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax6.set_xlabel('Spine Position Y [m]', fontsize=10)
        ax6.set_ylabel('θz [deg]', fontsize=10)
        ax6.set_title('Yaw Rotation (Z)', fontweight='bold')
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_dir:
            plot_path = os.path.join(output_dir, f'displacement_{beam_name}.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"✓ {beam_name.upper()} plot saved to: {plot_path}")
        
        plt.show()
    
    return {
        'arm_spline': disp_arm,
        'foil_dx': disp_foil_dx,
        'foil_sx': disp_foil_sx
    }
