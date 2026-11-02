import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def _infer_beam_span_axis(node_positions, beam_model=None):
    """Global axis (0,1,2) with largest nodal span; optional beam_model['beam_span_axis'] override."""
    if beam_model is not None and beam_model.get("beam_span_axis") is not None:
        return int(beam_model["beam_span_axis"])
    node_positions = np.asarray(node_positions, dtype=float)
    if node_positions.shape[0] < 2:
        return 1
    return int(np.argmax(np.ptp(node_positions, axis=0)))


class AeroGridToFEM:
    def __init__(self, beam_model):
        self.beam_model = beam_model

    def build_aerogrid(self, builder):
        panel_model = builder
        panel_model.build_aerogrid()
        aerogrid = panel_model.aerogrid
        return aerogrid

    def build_z_matrix(self, aerogrid, node_positions, panel_normals, beam_model, debug=True, xea_factor=None):

        offset_j = aerogrid['offset_j']
        control_points = offset_j

        n_panels = len(control_points)
        n_nodes = len(node_positions)
        Z = np.zeros((n_panels, 6 * n_nodes))
        self.panel_to_node_map = []
        self.panel_xi_map = []  # Store xi for visualization

        span_axis = _infer_beam_span_axis(node_positions, beam_model)
        if debug:
            print(f"  Aero–FEM spline: bracketing beam along global axis {span_axis} (0=X, 1=Y, 2=Z)")

        for i_panel in range(n_panels):
            y_nodes = node_positions[:, span_axis]
            sort_idx = np.argsort(y_nodes)
            y_sorted = y_nodes[sort_idx]
            
            ctrl = control_points[i_panel]
            normal = panel_normals[i_panel]
            y = ctrl[span_axis]
            
            # find bracketing nodes in Y
            j = np.searchsorted(y_sorted, y)
            j = int(np.clip(j, 1, len(y_sorted) - 1))
            i_node_1 = int(sort_idx[j - 1])
            i_node_2 = int(sort_idx[j])
            
            y1 = y_nodes[i_node_1]
            y2 = y_nodes[i_node_2]
            if abs(y2 - y1) < 1e-12:
                xi_best = 0.5
            else:
                xi_best = float(np.clip((y - y1) / (y2 - y1), 0.0, 1.0))
            
            N1 = 1.0 - xi_best
            N2 = xi_best
            
            # Calculate Delta_x: distance from elastic axis to control point (75% chord)
            # CRITICAL FIX: Calculate elastic axis position for THIS panel relative to its corner points
            # The lattice defines the wing geometry, so elastic axis should be relative to each panel
            if xea_factor is not None and 'cornerpoint_panels' in aerogrid and 'cornerpoint_grids' in aerogrid:
                # Get corner points for this panel
                panel_corner_ids = aerogrid['cornerpoint_panels'][i_panel]
                grid_data = aerogrid['cornerpoint_grids']
                grid_ids = grid_data[:, 0]
                grid_coords = grid_data[:, 1:4]
                
                # Create mapping from grid ID to coordinates
                id_to_coords = {int(gid): coord for gid, coord in zip(grid_ids, grid_coords)}
                
                # Get corner coordinates: corner1 (LE root), corner2 (TE root), corner3 (TE tip), corner4 (LE tip)
                corner1 = id_to_coords[int(panel_corner_ids[0])]  # LE root
                corner2 = id_to_coords[int(panel_corner_ids[1])]  # TE root
                corner4 = id_to_coords[int(panel_corner_ids[3])]  # LE tip
                
                # Calculate chordwise vectors (like in build_aeromodel)
                l_1 = corner2 - corner1  # chordwise at root
                l_2 = id_to_coords[int(panel_corner_ids[2])] - corner4  # chordwise at tip (corner3 - corner4)
                l_m = (l_1 + l_2) / 2.0  # mean chordwise vector

                ea_point = corner1 + float(xea_factor) * l_m
            else:
                ea_point = N1 * node_positions[i_node_1] + N2 * node_positions[i_node_2]

            # ---- FULL 3D RIGID BODY PROJECTION ----
            n = normal / np.linalg.norm(normal)  # safety normalization

            r_vec = ctrl - ea_point  # 3D arm vector

            # Node 1
            idx1 = i_node_1 * 6
            # Node 2
            idx2 = i_node_2 * 6
            
            Z[i_panel, idx1 + 0] += N1 * n[0]   # u
            Z[i_panel, idx1 + 1] += N1 * n[1]   # v
            Z[i_panel, idx1 + 2] += N1 * n[2]   # w
            
            Z[i_panel, idx2 + 0] += N2 * n[0]
            Z[i_panel, idx2 + 1] += N2 * n[1]
            Z[i_panel, idx2 + 2] += N2 * n[2]

            ## A. rotation θ × r real 3D behaviour
            Z[i_panel, idx1 + 3] += N1 * ( n @ np.cross([1,0,0], r_vec) )  # θx
            Z[i_panel, idx1 + 4] += N1 * ( n @ np.cross([0,1,0], r_vec) )  # θy
            Z[i_panel, idx1 + 5] += N1 * ( n @ np.cross([0,0,1], r_vec) )  # θz
            
            Z[i_panel, idx2 + 3] += N2 * ( n @ np.cross([1,0,0], r_vec) )
            Z[i_panel, idx2 + 4] += N2 * ( n @ np.cross([0,1,0], r_vec) )
            Z[i_panel, idx2 + 5] += N2 * ( n @ np.cross([0,0,1], r_vec) )

            ## B. For simplicity, we can assume rotation around Y-axis (spanwise) dominates for lift coupling
            #ey = np.array([0.0, 1.0, 0.0])
            #rot_proj = n @ np.cross(ey, r_vec)
            #Z[i_panel, idx1 + 4] += N1 * rot_proj
            #Z[i_panel, idx2 + 4] += N2 * rot_proj

            # store for diagnostics/plotting
            self.panel_to_node_map.append((i_node_1, i_node_2))
            self.panel_xi_map.append(xi_best)

        if debug:
            # Calculate coupling statistics
            distances = []
            for i, (ctrl, (n1, n2), xi) in enumerate(zip(control_points, self.panel_to_node_map, self.panel_xi_map)):
                beam_point = (1-xi) * node_positions[n1] + xi * node_positions[n2]
                dist = np.linalg.norm(ctrl - beam_point)
                distances.append(dist)
            
            print(f"Coupling distance statistics:")
            print(f"  Mean distance: {np.mean(distances):.4f}")
            print(f"  Max distance: {np.max(distances):.4f}")
            print(f"  Xi range: [{np.min(self.panel_xi_map):.3f}, {np.max(self.panel_xi_map):.3f}]")

        # Debug: check rotation contributions (theta_y is at DOF index 4)
        if n_nodes > 0:
            # Check rotation contributions for last node as example
            last_node_idx = (n_nodes - 1) * 6
            rot_contrib = np.linalg.norm(Z[:, last_node_idx+3:last_node_idx+6])
            print(f"Rotation contribution norm (last node): {rot_contrib:.6f}")

        return Z, self.panel_to_node_map, self.panel_xi_map
    



    # PLOTTING FUNCTIONS #

    def visualize_coupling(self, control_points, node_positions, aerogrid=None, enhanced=True, 
                          plot_airfoil_sections=False, airfoil_profile=None, n_sections=10):
        if not hasattr(self, 'panel_to_node_map') or not hasattr(self, 'panel_xi_map'):
            raise ValueError("Run build_z_matrix before visualize_coupling.")

        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot panel contours first (so they appear behind other elements)
        if aerogrid is not None:
            self._plot_panel_contours(ax, aerogrid)
            
        # Plot airfoil profile sections if requested
        if plot_airfoil_sections and airfoil_profile is not None and aerogrid is not None:
            self._plot_airfoil_sections(ax, aerogrid, airfoil_profile, n_sections)
        
        # Plot beam nodes and elements
        ax.scatter(node_positions[:, 0], node_positions[:, 1], node_positions[:, 2],
                   c='blue', s=100, label='FEM nodes', alpha=0.8, zorder=5)
        
        # Plot beam elements as lines
        for i in range(len(node_positions) - 1):
            ax.plot([node_positions[i, 0], node_positions[i+1, 0]],
                   [node_positions[i, 1], node_positions[i+1, 1]],
                   [node_positions[i, 2], node_positions[i+1, 2]], 
                   'b-', alpha=0.6, linewidth=2, zorder=4)
        
        # Plot control points
        ax.scatter(control_points[:, 0], control_points[:, 1], control_points[:, 2],
                   c='red', s=60, label='¾-chord points', alpha=0.8, zorder=5)

        # Plot coupling connections
        for i, ctrl in enumerate(control_points):
            i_node_1, i_node_2 = self.panel_to_node_map[i]
            xi = self.panel_xi_map[i]
            A = node_positions[i_node_1]
            B = node_positions[i_node_2]
            beam_point = (1 - xi) * A + xi * B

            # Connection line
            ax.plot([beam_point[0], ctrl[0]],
                    [beam_point[1], ctrl[1]],
                    [beam_point[2], ctrl[2]], 'k--', alpha=0.5, linewidth=1, zorder=3)
            
            if enhanced:
                # Mark the coupling point on the beam
                ax.scatter([beam_point[0]], [beam_point[1]], [beam_point[2]], 
                          c='green', s=30, alpha=0.7, zorder=5)

        ax.set_title("FEM Beam to ¾-Chord Control Point Coupling")
        ax.set_xlabel("X (Chordwise)")
        ax.set_ylabel("Y (Spanwise)")
        ax.set_zlabel("Z (Vertical)")
        ax.set_box_aspect([1, 1, 1])
        ax.legend()
        plt.tight_layout()
        plt.show()
    
    def _plot_panel_contours(self, ax, aerogrid):
        """
        Plot panel contours from aerogrid data.
        Supports multiple aerogrid formats with corner points.
        """
        # Try different common formats for panel corners
        if 'cornerpoint_panels' in aerogrid and 'cornerpoint_grids' in aerogrid:
            # Format: cornerpoint_panels contains grid IDs, cornerpoint_grids contains [ID, x, y, z]
            # This is the format used by AeroModel from build_aeromodel.py
            panel_corners = aerogrid['cornerpoint_panels']
            grid_data = aerogrid['cornerpoint_grids']
            grid_ids = grid_data[:, 0]
            grid_coords = grid_data[:, 1:4]
            
            # Create a mapping from grid ID to coordinates for fast lookup
            id_to_coords = {int(gid): coord for gid, coord in zip(grid_ids, grid_coords)}
            
            for panel_corner_ids in panel_corners:
                # Get coordinates for the 4 corners of this panel
                coords = [id_to_coords[int(corner_id)] for corner_id in panel_corner_ids]
                coords = np.array(coords)
                # Close the loop by appending first point
                x = np.append(coords[:, 0], coords[0, 0])
                y = np.append(coords[:, 1], coords[0, 1])
                z = np.append(coords[:, 2], coords[0, 2])
                ax.plot(x, y, z, 'blue', alpha=0.4, linewidth=0.8, zorder=1)
        
        elif 'corners' in aerogrid:
            # Format: corners[panel_idx] = [[x1,y1,z1], [x2,y2,z2], [x3,y3,z3], [x4,y4,z4]]
            corners = aerogrid['corners']
            for panel_corners in corners:
                panel_corners = np.array(panel_corners)
                # Close the loop by appending first point
                x = np.append(panel_corners[:, 0], panel_corners[0, 0])
                y = np.append(panel_corners[:, 1], panel_corners[0, 1])
                z = np.append(panel_corners[:, 2], panel_corners[0, 2])
                ax.plot(x, y, z, 'blue', alpha=0.4, linewidth=0.8, zorder=1)
        
        elif all(k in aerogrid for k in ['A', 'B', 'C', 'D']):
            # Format: separate arrays for each corner (VLM style)
            A, B, C, D = aerogrid['A'], aerogrid['B'], aerogrid['C'], aerogrid['D']
            n_panels = len(A)
            for i in range(n_panels):
                # Plot panel edges: A->B->C->D->A
                x = [A[i][0], B[i][0], C[i][0], D[i][0], A[i][0]]
                y = [A[i][1], B[i][1], C[i][1], D[i][1], A[i][1]]
                z = [A[i][2], B[i][2], C[i][2], D[i][2], A[i][2]]
                ax.plot(x, y, z, 'gray', alpha=0.4, linewidth=0.8, zorder=1)
        
        elif 'vertices' in aerogrid and 'panels' in aerogrid:
            # Format: vertices array + panel connectivity
            vertices = np.array(aerogrid['vertices'])
            panels = aerogrid['panels']
            for panel in panels:
                panel_verts = vertices[panel]
                x = np.append(panel_verts[:, 0], panel_verts[0, 0])
                y = np.append(panel_verts[:, 1], panel_verts[0, 1])
                z = np.append(panel_verts[:, 2], panel_verts[0, 2])
                ax.plot(x, y, z, 'gray', alpha=0.4, linewidth=0.8, zorder=1)
        
        else:
            print("Warning: Could not find panel corner data in aerogrid.")
            print(f"Available keys: {aerogrid.keys()}")
            print("Supported formats: 'cornerpoint_panels'+'cornerpoint_grids', 'corners', ['A','B','C','D'], or ['vertices','panels']")
    
    def _plot_airfoil_sections(self, ax, aerogrid, airfoil_profile, n_sections=10):

        if 'cornerpoint_panels' not in aerogrid or 'cornerpoint_grids' not in aerogrid:
            print("Warning: Airfoil sections require 'cornerpoint_panels' and 'cornerpoint_grids' in aerogrid.")
            return
        
        # Get grid data
        grid_data = aerogrid['cornerpoint_grids']
        grid_ids = grid_data[:, 0]
        grid_coords = grid_data[:, 1:4]
        id_to_coords = {int(gid): coord for gid, coord in zip(grid_ids, grid_coords)}
        
        # Extract airfoil profile coordinates
        profile_x = np.array(airfoil_profile['x'])
        profile_z = np.array(airfoil_profile['z'])
        
        # Find the spanwise range
        all_y_coords = grid_coords[:, 2]
        y_min, y_max = np.min(all_y_coords), np.max(all_y_coords)
        
        # Generate spanwise positions for sections
        y_sections = np.linspace(y_min, y_max, n_sections)
        
        # Get panel corners
        panel_corners = aerogrid['cornerpoint_panels']
        
        # For each spanwise section
        for y_section in y_sections:
            # Find panels that bracket this spanwise location
            # We need to find the leading and trailing edge at this y location
            
            # Collect all unique spanwise grid lines
            y_values = sorted(list(set(all_y_coords)))
            
            # Find the two y-values that bracket y_section
            y_idx = np.searchsorted(y_values, y_section)
            if y_idx == 0:
                y_idx = 1
            elif y_idx >= len(y_values):
                y_idx = len(y_values) - 1
            
            y1 = y_values[y_idx - 1]
            y2 = y_values[y_idx]
            
            if abs(y2 - y1) < 1e-10:
                xi = 0.5
            else:
                xi = (y_section - y1) / (y2 - y1)
            
            # Find grids at y1 and y2
            grids_at_y1 = []
            grids_at_y2 = []
            
            for i, coord in enumerate(grid_coords):
                if abs(coord[1] - y1) < 1e-6:
                    grids_at_y1.append((coord[0], coord[2], int(grid_ids[i])))  # (x, z, id)
                elif abs(coord[1] - y2) < 1e-6:
                    grids_at_y2.append((coord[0], coord[2], int(grid_ids[i])))  # (x, z, id)
            
            if len(grids_at_y1) < 2 or len(grids_at_y2) < 2:
                continue
            
            # Sort by x-coordinate to find leading and trailing edge
            grids_at_y1.sort(key=lambda p: p[0])
            grids_at_y2.sort(key=lambda p: p[0])
            
            # Leading edge (minimum x) and trailing edge (maximum x)
            le1 = id_to_coords[grids_at_y1[0][2]]  # Leading edge at y1
            te1 = id_to_coords[grids_at_y1[-1][2]]  # Trailing edge at y1
            le2 = id_to_coords[grids_at_y2[0][2]]  # Leading edge at y2
            te2 = id_to_coords[grids_at_y2[-1][2]]  # Trailing edge at y2
            
            # Interpolate to get LE and TE at y_section
            le_section = (1 - xi) * le1 + xi * le2
            te_section = (1 - xi) * te1 + xi * te2
            
            # Calculate local chord and orientation
            chord_vector = te_section - le_section
            chord_length = np.linalg.norm(chord_vector)
            
            if chord_length < 1e-10:
                continue
            
            # Scale airfoil profile
            scaled_x = profile_x * chord_length
            scaled_z = profile_z * chord_length
            
            # Create 3D coordinates for the airfoil section
            # Position at leading edge and align with chord
            airfoil_3d = np.zeros((len(scaled_x), 3))
            
            # Place airfoil in local coordinate system
            # x_local along chord, z_local is vertical offset
            for i in range(len(scaled_x)):
                # Position along chord from leading edge
                point_along_chord = le_section + scaled_x[i] * chord_vector / chord_length
                # Add vertical offset
                airfoil_3d[i] = point_along_chord + np.array([0, 0, scaled_z[i]])
            
            # Plot the airfoil section
            ax.plot(airfoil_3d[:, 0], airfoil_3d[:, 1], airfoil_3d[:, 2], 
                   'darkgreen', alpha=0.6, linewidth=1.5, zorder=2)



def main(beam_prop, aerogrid, coupling_diagnostics, plot=False, plot_airfoil=False, 
         airfoil_profile=None, n_airfoil_sections=10, xea_factor=None):
    coupler = AeroGridToFEM(beam_prop)
    node_positions = np.array([node['position'] for node in beam_prop['nodes']])
    panel_normals = aerogrid['N']/ np.linalg.norm(aerogrid['N'], axis=1)[:, None]  # Normalize normals
    control_points = aerogrid['offset_j']
    
    Z, panel_to_node_map, panel_xi_map = coupler.build_z_matrix(aerogrid, node_positions, panel_normals, beam_prop, xea_factor=xea_factor)
    #print(f"Z matrix shape: {Z.shape}")

    if plot:
        coupler.visualize_coupling(control_points, node_positions, aerogrid=aerogrid, 
                                  enhanced=True, plot_airfoil_sections=plot_airfoil,
                                  airfoil_profile=airfoil_profile, n_sections=n_airfoil_sections)

    return Z, panel_to_node_map, panel_xi_map


def create_airfoil_profile(x_coords, z_coords):

    return {'x': np.array(x_coords), 'z': np.array(z_coords)}


def plot_chordwise_strip(aerogrid, y_target=3.0, xea_factor=None, xcm_factor=None, config=None, save_path=None):
    """
    Plot control points, force points, shear center, and CoG for a chordwise strip of panels
    near the specified y coordinate.
    
    Parameters:
    -----------
    aerogrid : dict
        Aerodynamic grid dictionary
    y_target : float
        Target y coordinate for the strip (default: 3.0)
    xea_factor : float
        Elastic axis factor (e.g., 0.33 for 33% chord)
    xcm_factor : float
        Center of gravity factor (e.g., 0.43 for 43% chord)
    config : object
        Configuration object (optional, used if xea_factor/xcm_factor not provided)
    save_path : str
        Path to save the plot (optional)
    """
    # Get factors from config if not provided
    if xea_factor is None and config is not None:
        xea_factor = getattr(config, 'xea_factor', None)
    if xcm_factor is None and config is not None:
        xcm_factor = getattr(config, 'xcm_factor', None)
    
    # Get panel data
    offset_j = aerogrid['offset_j']  # Control points (75% chord)?
    offset_l = aerogrid['offset_l']  # Force points (25% chord)?
    
    # Find panels near y_target
    y_coords = offset_j[:, 2]  # y coordinates of control points
    y_diff = np.abs(y_coords - y_target)
    min_diff = np.min(y_diff)
    tolerance = 0.1  # tolerance for finding panels at similar y
    
    # Find all panels within tolerance
    strip_indices = np.where(y_diff <= min_diff + tolerance)[0]
    
    if len(strip_indices) == 0:
        print(f"No panels found near y={y_target}")
        return
    
    # Sort by x coordinate
    strip_indices = strip_indices[np.argsort(offset_j[strip_indices, 0])]
    
    print(f"Found {len(strip_indices)} panels in strip near y={y_target:.3f}")
    print(f"Y range: [{np.min(y_coords[strip_indices]):.3f}, {np.max(y_coords[strip_indices]):.3f}]")
    
    # Extract coordinates for the strip
    control_points_x = offset_j[strip_indices, 0]
    control_points_y = offset_j[strip_indices, 1]
    force_points_x = offset_l[strip_indices, 0]
    force_points_y = offset_l[strip_indices, 1]
    
    # Calculate shear center and CoG positions for the strip
    # CRITICAL: Shear center and CoG are FIXED points for this spanwise section (y constant)
    # They do NOT change for each panel in the chordwise direction
    # They extend only in y direction (spanwise), and at this y they are single points
    
    # Get the y coordinate of the strip (should be approximately y_target)
    y_strip = control_points_y[0] if len(control_points_y) > 0 else y_target
    
    # Extract X1 (LE root) and chord from aerogrid
    # X1 is the leading edge root point, which should be the first corner point of the first panel at root
    # Chord is the total chord length
    if 'cornerpoint_panels' in aerogrid and 'cornerpoint_grids' in aerogrid:
        grid_data = aerogrid['cornerpoint_grids']
        grid_ids = grid_data[:, 0]
        grid_coords = grid_data[:, 1:4]
        id_to_coords = {int(gid): coord for gid, coord in zip(grid_ids, grid_coords)}
        
        # Find first panel at root (y=0 or minimum y)
        # Get all panels and find the one with minimum y coordinate
        all_y_coords = offset_j[:, 2]
        root_panel_idx = np.argmin(all_y_coords)  # Panel closest to root
        
        # Get corner points of root panel
        root_panel_corner_ids = aerogrid['cornerpoint_panels'][root_panel_idx]
        corner1_root = id_to_coords[int(root_panel_corner_ids[0])]  # LE root
        corner2_root = id_to_coords[int(root_panel_corner_ids[1])]  # TE root
        
        # X1 is the LE root point
        X1 = corner1_root
        
        # Calculate total chord from root panel
        # For a panel at root, the chord is the distance from LE to TE
        chord_vector_root = corner2_root - corner1_root
        chord_total = np.linalg.norm(chord_vector_root)
        
        # If we have multiple chordwise panels, we need to sum them
        # Or we can get it from the first and last panel at root
        # For now, use the first panel's chord and multiply by n_chord
        # Actually, better: find the first and last panel at root
        root_panels = np.where(all_y_coords <= np.min(all_y_coords) + 0.01)[0]
        if len(root_panels) > 1:
            # Multiple panels at root - get first and last
            root_panels_sorted = root_panels[np.argsort(offset_j[root_panels, 0])]
            first_root_panel = root_panels_sorted[0]
            last_root_panel = root_panels_sorted[-1]
            
            first_corner_ids = aerogrid['cornerpoint_panels'][first_root_panel]
            last_corner_ids = aerogrid['cornerpoint_panels'][last_root_panel]
            
            first_LE = id_to_coords[int(first_corner_ids[0])]
            last_TE = id_to_coords[int(last_corner_ids[1])]
            
            chord_total = np.linalg.norm(last_TE - first_LE)
            X1 = first_LE
    else:
        # Fallback: use known values for Goland (if aerogrid structure is different)
        print("Warning: Cannot extract X1 and chord from aerogrid. Using fallback values.")
        X1 = np.array([-0.604, 0.0, 0.0])
        chord_total = 1.829
    
    # Calculate shear center position for this spanwise section
    # Shear center is at xea_factor * chord_total from X1 (LE root)
    # x_shear_center = X1[0] + xea_factor * chord_total
    if xea_factor is not None:
        x_shear_center = X1[0] + xea_factor * chord_total
        shear_center_x = [x_shear_center]
        shear_center_y = [y_strip]
    else:
        shear_center_x = None
        shear_center_y = None
    
    # Calculate CoG position for this spanwise section
    # CoG is at xcm_factor * chord_total from X1 (LE root)
    # x_cog = X1[0] + xcm_factor * chord_total
    if xcm_factor is not None:
        x_cog = X1[0] + xcm_factor * chord_total
        cog_x = [x_cog]
        cog_y = [y_strip]
    else:
        cog_x = None
        cog_y = None
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot control points (75% chord)
    ax.scatter(control_points_x, control_points_y, c='red', s=100, marker='o', 
               label='Control Points (75% chord)', alpha=0.8, zorder=5)
    
    # Plot force points (25% chord)
    ax.scatter(force_points_x, force_points_y, c='blue', s=100, marker='s', 
               label='Force Points (25% chord)', alpha=0.8, zorder=5)
    
    # Plot shear center (elastic axis) - SINGLE POINT for this spanwise section
    if shear_center_x is not None and shear_center_y is not None:
        ax.scatter(shear_center_x, shear_center_y, c='green', s=150, marker='^', 
                   label='Shear Center (Elastic Axis)', alpha=0.9, zorder=6, edgecolors='darkgreen', linewidths=2)
    
    # Plot CoG - SINGLE POINT for this spanwise section
    if cog_x is not None and cog_y is not None:
        ax.scatter(cog_x, cog_y, c='orange', s=150, marker='v', 
                   label='CoG (Center of Gravity)', alpha=0.9, zorder=6, edgecolors='darkorange', linewidths=2)
    
    # Plot global reference frame (origin)
    ax.scatter([0], [0], c='black', s=150, marker='+', linewidths=3, 
               label='Global Reference Frame (Origin)', zorder=7)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X coordinate [m]', fontsize=12)
    ax.set_ylabel('Y coordinate [m]', fontsize=12)
    ax.set_title(f'Chordwise Strip Analysis at y ≈ {y_target:.3f} m\n'
                 f'({len(strip_indices)} panels)', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.set_aspect('equal', adjustable='box')
    
    # Add text annotations for first and last panels
    if len(strip_indices) > 0:
        # First panel (LE side)
        ax.annotate('LE', xy=(control_points_x[0], control_points_y[0]), 
                   xytext=(5, 5), textcoords='offset points', fontsize=9, color='red')
        # Last panel (TE side)
        ax.annotate('TE', xy=(control_points_x[-1], control_points_y[-1]), 
                   xytext=(5, 5), textcoords='offset points', fontsize=9, color='red')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()
    
    return fig, ax


def plot_full_wing(aerogrid, xea_factor=None, xcm_factor=None, config=None, save_path=None):
    """
    Plot control points, force points, shear center, and CoG for the entire wing.
    
    Parameters:
    -----------
    aerogrid : dict
        Aerodynamic grid dictionary
    xea_factor : float
        Elastic axis factor (e.g., 0.33 for 33% chord)
    xcm_factor : float
        Center of gravity factor (e.g., 0.43 for 43% chord)
    config : object
        Configuration object (optional, used if xea_factor/xcm_factor not provided)
    save_path : str
        Path to save the plot (optional)
    """
    # Get factors from config if not provided
    if xea_factor is None and config is not None:
        xea_factor = getattr(config, 'xea_factor', None)
    if xcm_factor is None and config is not None:
        xcm_factor = getattr(config, 'xcm_factor', None)
    
    # Get panel data
    offset_j = aerogrid['offset_j']  # Control points (75% chord)
    offset_l = aerogrid['offset_l']  # Force points (25% chord)
    
    # Extract X1 (LE root) and chord from aerogrid
    if 'cornerpoint_panels' in aerogrid and 'cornerpoint_grids' in aerogrid:
        grid_data = aerogrid['cornerpoint_grids']
        grid_ids = grid_data[:, 0]
        grid_coords = grid_data[:, 1:4]
        id_to_coords = {int(gid): coord for gid, coord in zip(grid_ids, grid_coords)}
        
        # Find first and last panel at root (y=0 or minimum y)
        all_y_coords = offset_j[:, 2]
        root_panels = np.where(all_y_coords <= np.min(all_y_coords) + 0.01)[0]
        
        if len(root_panels) > 1:
            # Multiple panels at root - get first and last
            root_panels_sorted = root_panels[np.argsort(offset_j[root_panels, 0])]
            first_root_panel = root_panels_sorted[0]
            last_root_panel = root_panels_sorted[-1]
            
            first_corner_ids = aerogrid['cornerpoint_panels'][first_root_panel]
            last_corner_ids = aerogrid['cornerpoint_panels'][last_root_panel]
            
            first_LE = id_to_coords[int(first_corner_ids[0])]
            last_TE = id_to_coords[int(last_corner_ids[1])]
            
            chord_total = np.linalg.norm(last_TE - first_LE)
            X1 = first_LE
        else:
            # Single panel or fallback
            root_panel_idx = root_panels[0] if len(root_panels) > 0 else 0
            root_panel_corner_ids = aerogrid['cornerpoint_panels'][root_panel_idx]
            corner1_root = id_to_coords[int(root_panel_corner_ids[0])]
            corner2_root = id_to_coords[int(root_panel_corner_ids[1])]
            chord_vector_root = corner2_root - corner1_root
            chord_total = np.linalg.norm(chord_vector_root)
            X1 = corner1_root
    else:
        # Fallback: use known values for Goland
        print("Warning: Cannot extract X1 and chord from aerogrid. Using fallback values.")
        X1 = np.array([-0.604, 0.0, 0.0])
        chord_total = 1.829
    
    # Get span range
    y_min = np.min(offset_j[:, 1])
    y_max = np.max(offset_j[:, 1])
    y_span = np.linspace(y_min, y_max, 100)  # 100 points along span
    
    # Calculate shear center line (extends in y direction)
    if xea_factor is not None:
        x_shear_center = X1[0] + xea_factor * chord_total
        shear_center_x = [x_shear_center] * len(y_span)
        shear_center_y = y_span
    else:
        shear_center_x = None
        shear_center_y = None
    
    # Calculate CoG line (extends in y direction)
    if xcm_factor is not None:
        x_cog = X1[0] + xcm_factor * chord_total
        cog_x = [x_cog] * len(y_span)
        cog_y = y_span
    else:
        cog_x = None
        cog_y = None
    
    # Create plot
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Plot control points (75% chord) - ALL panels
    ax.scatter(offset_j[:, 0], offset_j[:, 1], c='red', s=30, marker='o', 
               label='Control Points (75% chord)', alpha=0.6, zorder=4)
    
    # Plot force points (25% chord) - ALL panels
    ax.scatter(offset_l[:, 0], offset_l[:, 1], c='blue', s=30, marker='s', 
               label='Force Points (25% chord)', alpha=0.6, zorder=4)
    
    # Plot shear center line (elastic axis) - extends in y direction
    if shear_center_x is not None and shear_center_y is not None:
        ax.plot(shear_center_x, shear_center_y, c='green', linewidth=3, 
               label='Shear Center (Elastic Axis)', alpha=0.9, zorder=5)
        # Also plot a marker at root and tip for clarity
        ax.scatter([shear_center_x[0]], [shear_center_y[0]], c='green', s=150, marker='^', 
                   alpha=0.9, zorder=6, edgecolors='darkgreen', linewidths=2)
        ax.scatter([shear_center_x[-1]], [shear_center_y[-1]], c='green', s=150, marker='^', 
                   alpha=0.9, zorder=6, edgecolors='darkgreen', linewidths=2)
    
    # Plot CoG line - extends in y direction
    if cog_x is not None and cog_y is not None:
        ax.plot(cog_x, cog_y, c='orange', linewidth=3, 
               label='CoG (Center of Gravity)', alpha=0.9, zorder=5)
        # Also plot a marker at root and tip for clarity
        ax.scatter([cog_x[0]], [cog_y[0]], c='orange', s=150, marker='v', 
                   alpha=0.9, zorder=6, edgecolors='darkorange', linewidths=2)
        ax.scatter([cog_x[-1]], [cog_y[-1]], c='orange', s=150, marker='v', 
                   alpha=0.9, zorder=6, edgecolors='darkorange', linewidths=2)
    
    # Plot global reference frame (origin)
    ax.scatter([0], [0], c='black', s=200, marker='+', linewidths=4, 
               label='Global Reference Frame (Origin)', zorder=7)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X coordinate [m] (chordwise)', fontsize=12)
    ax.set_ylabel('Y coordinate [m] (spanwise)', fontsize=12)
    ax.set_title(f'Full Wing Analysis\n'
                 f'({len(offset_j)} panels total)', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.set_aspect('equal', adjustable='box')
    
    # Add text annotations
    if shear_center_x is not None:
        ax.text(shear_center_x[0], shear_center_y[-1] + 0.1, 'Shear Center', 
               fontsize=9, color='green', ha='center')
    if cog_x is not None:
        ax.text(cog_x[0], cog_y[-1] + 0.1, 'CoG', 
               fontsize=9, color='orange', ha='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()
    
    return fig, ax


def build_Z_force(aerogrid, panel_to_node_map, panel_xi_map, node_positions, beam_model, n_dofs, xea_factor=None):

    force_points = aerogrid['offset_l']  # Fallback
    
    normals = aerogrid["N"]/ np.linalg.norm(aerogrid["N"], axis=1)[:, None]        # (n_panels, 3)
    n_panels = normals.shape[0]
    print(f"Building Z_force: {n_panels} panels, {n_dofs} DOFs")
    
    Z_force = np.zeros((n_panels, n_dofs), dtype=float)
    
    for ip in range(n_panels):
        force_pt = force_points[ip]
        normal = normals[ip]
        
        # Get the two nodes for this panel and compute shape functions
        i_node_1, i_node_2 = panel_to_node_map[ip]
        xi = panel_xi_map[ip]
        N1 = 1.0 - xi
        N2 = xi
        
        # Calculate Delta_x_fp: distance from elastic axis to force point (25% chord)
        # CRITICAL FIX: Calculate elastic axis position for THIS panel relative to its corner points
        if xea_factor is not None and 'cornerpoint_panels' in aerogrid and 'cornerpoint_grids' in aerogrid:
            # Get corner points for this panel
            panel_corner_ids = aerogrid['cornerpoint_panels'][ip]
            grid_data = aerogrid['cornerpoint_grids']
            grid_ids = grid_data[:, 0]
            grid_coords = grid_data[:, 1:4]
            
            # Create mapping from grid ID to coordinates
            id_to_coords = {int(gid): coord for gid, coord in zip(grid_ids, grid_coords)}
            
            # Get corner coordinates
            corner1 = id_to_coords[int(panel_corner_ids[0])]  # LE root
            corner2 = id_to_coords[int(panel_corner_ids[1])]  # TE root
            corner4 = id_to_coords[int(panel_corner_ids[3])]  # LE tip
            
            # Calculate chordwise vectors
            l_1 = corner2 - corner1  # chordwise at root
            l_2 = id_to_coords[int(panel_corner_ids[2])] - corner4  # chordwise at tip
            l_m = (l_1 + l_2) / 2.0  # mean chordwise vector

            ea_point = corner1 + float(xea_factor) * l_m
        else:
            ea_point = N1 * node_positions[i_node_1] + N2 * node_positions[i_node_2]

        n = normal / np.linalg.norm(normal)

        r_vec = force_pt - ea_point

        # Node 1
        idx1 = i_node_1 * 6
        Z_force[ip, idx1 + 0] += N1 * n[0]
        Z_force[ip, idx1 + 1] += N1 * n[1]
        Z_force[ip, idx1 + 2] += N1 * n[2]

        Z_force[ip, idx1 + 3] += N1 * ( n @ np.cross([1,0,0], r_vec) )
        Z_force[ip, idx1 + 4] += N1 * ( n @ np.cross([0,1,0], r_vec) )
        Z_force[ip, idx1 + 5] += N1 * ( n @ np.cross([0,0,1], r_vec) )

        # Node 2
        idx2 = i_node_2 * 6
        Z_force[ip, idx2 + 0] += N2 * n[0]
        Z_force[ip, idx2 + 1] += N2 * n[1]
        Z_force[ip, idx2 + 2] += N2 * n[2]

        Z_force[ip, idx2 + 3] += N2 * ( n @ np.cross([1,0,0], r_vec) )
        Z_force[ip, idx2 + 4] += N2 * ( n @ np.cross([0,1,0], r_vec) )
        Z_force[ip, idx2 + 5] += N2 * ( n @ np.cross([0,0,1], r_vec) )
    
    return Z_force


def build_Z_qs(aerogrid, panel_to_node_map, panel_xi_map, n_dofs, node_positions=None, beam_model=None):

    normals = aerogrid["N"] / np.linalg.norm(aerogrid["N"], axis=1)[:, None]
    n_panels = normals.shape[0]

    Z_qs = np.zeros((n_panels, n_dofs), dtype=float)

    ex = np.array([1.0, 0.0, 0.0])  # flow direction

    span_axis = _infer_beam_span_axis(node_positions, beam_model) if node_positions is not None else 1
    ey = np.zeros(3)
    ey[span_axis] = 1.0
    theta_dof = 3 + span_axis

    for ip in range(n_panels):

        n = normals[ip]

        i_node_1, i_node_2 = panel_to_node_map[ip]
        xi = panel_xi_map[ip]
        N1 = 1.0 - xi
        N2 = xi

        rot_proj = n @ np.cross(ey, ex)

        idx = i_node_1 * 6
        Z_qs[ip, idx + theta_dof] += N1 * rot_proj

        idx = i_node_2 * 6
        Z_qs[ip, idx + theta_dof] += N2 * rot_proj

    return Z_qs

def calculate_control_force_points_distances(aerogrid, xea_factor=0.33):
    """
    Calculate average distances of control points (75% chord) and force points (25% chord)
    from the y-axis.
    
    Parameters
    ----------
    aerogrid : dict
        Aerogrid dictionary containing offset_j and offset_l
    xea_factor : float, optional
        Elastic axis position as fraction of chord (default: 0.33 for 33% chord)
    verbose : bool, optional
        If True, print detailed results (default: True)
    
    Returns
    -------
    results : dict
        Dictionary containing:
        - 'control_avg_dist': Average distance of control points from y-axis (m)
        - 'force_avg_dist': Average distance of force points from y-axis (m)
        - 'control_distances': Array of all control point distances (m)
        - 'force_distances': Array of all force point distances (m)
    """
    
    # Extract control and force points
    offset_j = aerogrid['offset_j']  # Control points at 75% chord
    offset_l = aerogrid['offset_l']  # Force points at 25% chord
    
    # Calculate distances from y-axis
    # Distance from y-axis is sqrt(x^2 + z^2) in the x-z plane
    control_x = offset_j[:, 0]
    control_z = offset_j[:, 2]
    force_x = offset_l[:, 0]
    force_z = offset_l[:, 2]
    
    # Distance from y-axis for each point
    control_distances = np.sqrt(control_x**2 + control_z**2)
    force_distances = np.sqrt(force_x**2 + force_z**2)
    
    # Calculate statistics
    control_avg_dist = np.mean(control_distances)
    control_min_dist = np.min(control_distances)
    control_max_dist = np.max(control_distances)
    
    force_avg_dist = np.mean(force_distances)
    force_min_dist = np.min(force_distances)
    force_max_dist = np.max(force_distances)
    
    results = {
        'control_avg_dist': control_avg_dist,
        'control_min_dist': control_min_dist,
        'control_max_dist': control_max_dist,
        'force_avg_dist': force_avg_dist,
        'force_min_dist': force_min_dist,
        'force_max_dist': force_max_dist,
        'control_distances': control_distances,
        'force_distances': force_distances,
        'n_control': len(offset_j),
        'n_force': len(offset_l),
    }
    

    print("\n" + "="*70)
    print("Control Points and Force Points Distance from Y-Axis")
    print("="*70)
    print(f"Number of control points (75% chord): {results['n_control']}")
    print(f"Number of force points (25% chord): {results['n_force']}")
    print()
    
    print("Control Points (75% chord):")
    print(f"  Average distance from y-axis: {control_avg_dist:.6f} m")
    print(f"  Min distance: {control_min_dist:.6f} m")
    print(f"  Max distance: {control_max_dist:.6f} m")
    print()
    
    print("Force Points (25% chord):")
    print(f"  Average distance from y-axis: {force_avg_dist:.6f} m")
    print(f"  Min distance: {force_min_dist:.6f} m")
    print(f"  Max distance: {force_max_dist:.6f} m")
    print()
    
    print("Difference:")
    print(f"  Control - Force: {control_avg_dist - force_avg_dist:.6f} m")
    print("="*70 + "\n")
    
    return results