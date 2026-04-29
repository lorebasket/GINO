import numpy as np
import pandas as pd

def create_beam_model(K, M, n_el, sections_props_csv=None, center_beam=False, chord=None):

    n_nodes = n_el + 1

    # -------------------------
    # Build per-element K,M list
    # -------------------------
    nodes_stiffness = []
    nodes_mass = []

    # Check if K and M are dictionaries or arrays
    is_dict = isinstance(K, dict)

    if is_dict:
        # Original "portion" logic for SONATA/flutter benchmark
        n_portions = len(K)
        portion_size = n_el // n_portions
        remaining_elements = n_el % n_portions
        sorted_keys = sorted(K.keys())
    
        element_counter = 0
        for i, key in enumerate(sorted_keys):
            n_elements_in_portion = portion_size + (1 if i < remaining_elements else 0)
            for _ in range(n_elements_in_portion):
                nodes_stiffness.append(K[key])
                nodes_mass.append(M[key])
                element_counter += 1

        assert len(nodes_stiffness) == n_el, "Mismatch in stiffness assignment"

        # Print partitioning info
        print("Beam portions (automatic):")
        for i, key in enumerate(sorted_keys):
            n_elements_in_portion = portion_size + (1 if i < remaining_elements else 0)
            print(f" Portion {i + 1}: {n_elements_in_portion} elements assigned to key '{key}'")

    else:
        print("Warning: K and M are not dictionaries. Using single matrix for all elements.")
        
        # Use the same K and M for all elements
        for _ in range(n_el):
            nodes_stiffness.append(K.copy())
            nodes_mass.append(M.copy())

    # -------------------------
    # Create node coordinates
    # -------------------------
    nodes = []

    df = pd.read_csv(sections_props_csv)
    print("\n[DEBUG] beam_model.py: Head of the sections_props_csv DataFrame:")
    print(df.head().to_string())
    
        # Ensure the required columns exist
    required_cols = ['X', 'Y', 'Z', 'CG_X', 'CG_Y', 'x2_x', 'x2_y', 'x2_z', 'x3_x', 'x3_y', 'x3_z']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"CSV file must contain the columns: {required_cols}")

    # 1. Define the primary spanwise coordinates for each of the n_nodes
    span_coords = np.linspace(df['X'].min(), df['X'].max(), n_nodes)

    # 2. Interpolate all required values at each spanwise coordinate
    ref_X = np.interp(span_coords, df['X'], df['X'])
    ref_Y = np.interp(span_coords, df['X'], df['Y'])
    ref_Z = np.interp(span_coords, df['X'], df['Z'])
    
    # Use center of gravity (CG) instead of shear center (SC) for beam alignment
    cg_x = np.interp(span_coords, df['X'], df['CG_X'])
    cg_y = np.interp(span_coords, df['X'], df['CG_Y'])
    
    x2_x = np.interp(span_coords, df['X'], df['x2_x'])
    x2_y = np.interp(span_coords, df['X'], df['x2_y'])
    x2_z = np.interp(span_coords, df['X'], df['x2_z'])
    
    x3_x = np.interp(span_coords, df['X'], df['x3_x'])
    x3_y = np.interp(span_coords, df['X'], df['x3_y'])
    x3_z = np.interp(span_coords, df['X'], df['x3_z'])

    # 3. Calculate global center of gravity positions using SONATA's formula:
    for i in range(n_nodes):
        ref_pos = np.array([ref_X[i], ref_Y[i], ref_Z[i]])
        x2_vec = np.array([x2_x[i], x2_y[i], x2_z[i]])
        x3_vec = np.array([x3_x[i], x3_y[i], x3_z[i]])
        
        # Calculate CG global position: ref_pos + cg_x * x2_vec + cg_y * x3_vec
        # Note: CG_X in SONATA is typically positive (aft of reference), so we ADD it
        cg_global = ref_pos + cg_x[i] * x2_vec + cg_y[i] * x3_vec
        

        # Swap X and Y to make spanwise direction align with Y-axis
        # Translate chordwise (X after swap) by Y value
        y_offset = ref_Y[i]
        
        # Calculate chordwise position (X after coordinate swap)
        # cg_global[1] is the Y component in SONATA coords (chordwise in local frame)
        x_chordwise = cg_global[1] - y_offset
        
        # If center_beam is True, offset the beam chordwise to center it around X=0
        # SONATA places beam reference line at LE (X=0 to X=chord)
        # To center: shift by -chord/2 so LE is at -chord/2 and TE at +chord/2
        if center_beam:
            if chord is None:
                raise ValueError("chord must be provided when center_beam=True")
            x_chordwise -= chord / 2.0
            
        nodes.append({
            "position": [x_chordwise, cg_global[0], cg_global[2]],
            "index": i,
            "stiffness": nodes_stiffness[i-1] if 0 < i <= n_el else None,
            "mass": nodes_mass[i-1] if 0 < i <= n_el else None,
        })

    print(f"\n[DEBUG] beam_model.py: Created {len(nodes)} nodes.")
    print("[DEBUG] beam_model.py: ALL node positions:")
    for i, node in enumerate(nodes):
        print(f"  Node {i}: {node['position']}")

    # -------------------------
    # Create elements
    # -------------------------
    elements = []
    for i in range(n_el):
        p1 = np.array(nodes[i]["position"], dtype=float)
        p2 = np.array(nodes[i+1]["position"], dtype=float)
        L_e = float(np.linalg.norm(p2 - p1))
        
        # Assign stiffness and mass based on the logic used before
        stiffness = nodes_stiffness[i] if nodes_stiffness else None
        mass = nodes_mass[i] if nodes_mass else None

        elements.append({
            "nodes": [i, i + 1],
            "stiffness": stiffness,
            "mass": mass,
            "length": L_e
        })

    return {"nodes": nodes, "elements": elements}
