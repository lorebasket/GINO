import numpy as np

def create_beam_model(K, M, beam_length, n_el, pitch, agard_theory=False, flutter_benchmark=False, sonata=False, center_beam=False, chord=None, xea=None, xcm=None, case_name=None):
    """
    Create a beam model with given stiffness and mass properties.
    
    Parameters:
        K: Can be either:
           - A dictionary of stiffness matrices (for per-element or portioned properties)
           - A single stiffness matrix to be used for all elements
        M: Can be either:
           - A dictionary of mass matrices (for per-element or portioned properties)
           - A single mass matrix to be used for all elements
        beam_length: Length of the beam
        n_el: Number of elements
        pitch: Pitch angle
        agard_theory: If True, uses AGARD per-element mode
        flutter_benchmark: If True, uses benchmark mode
        sonata: If True, uses SONATA partitioning
        xea: Shear center X coordinate [m] (for positioning nodes)
        xcm: Center of mass X coordinate [m] (for reference)
        case_name: Case name to identify special handling (e.g., 'ABRAMSON1965')
    """
    n_nodes = n_el + 1

    # -------------------------
    # Build per-element K,M list
    # -------------------------
    nodes_stiffness = []
    nodes_mass = []

    # Check if K and M are dictionaries or arrays
    is_dict = isinstance(K, dict)
    
    if is_dict:
        if agard_theory:
            # Validate keys for AGARD theory
            missing_k = [f"e{i}" for i in range(n_el) if f"e{i}" not in K]
            missing_m = [f"e{i}" for i in range(n_el) if f"e{i}" not in M]
            if missing_k or missing_m:
                raise KeyError(
                    f"Per-element K/M missing keys. "
                    f"Missing K: {missing_k}; Missing M: {missing_m}"
                )
            for i in range(n_el):
                nodes_stiffness.append(K[f"e{i}"])
                nodes_mass.append(M[f"e{i}"])
            print(f"AGARD per-element mode: received {n_el} K/M matrices.")
        
        elif sonata or flutter_benchmark:
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
        # Handle case where K and M are single matrices
        if agard_theory or sonata or flutter_benchmark:
            print("Warning: K and M are not dictionaries. Using single matrix for all elements.")
        
        # Use the same K and M for all elements
        for _ in range(n_el):
            nodes_stiffness.append(K.copy())
            nodes_mass.append(M.copy())

    # -------------------------
    # Create node coordinates
    # -------------------------
    element_length   = beam_length / n_el
    element_length_x = element_length * np.cos(np.deg2rad(pitch))
    element_length_y = element_length * np.sin(np.deg2rad(pitch))

    # ── Node positioning: Shear Center for ABRAMSON1965, otherwise origin ──
    if case_name == 'ABRAMSON1965' and xea is not None:
        x_node_offset = xea
        print(f"ABRAMSON1965 detected: Nodes positioned at Shear Center (xea={xea:.6f} m)")
    else:
        x_node_offset = 0.0

    nodes = []
    if pitch != 0:
        if center_beam and chord is not None:
            x_offset = chord / 2
        else:
            x_offset = 0.0
        for i in range(n_nodes):
            y = i * element_length_y  # span-wise
            x = i * element_length_x  # stream-wise
            nodes.append({"position": [x_node_offset + x_offset, y, 0.0], "index": i})
    else:
        if center_beam and chord is not None:
            x_offset = chord / 2
        else:            x_offset = 0.0
        for i in range(n_nodes):
            y = i * element_length
            nodes.append({
                "position": [x_node_offset + x_offset, y, 0.0],  # Align beam along y-axis
                "index": i,
                "stiffness": nodes_stiffness[i-1] if 0 < i <= n_el else None,
                "mass": nodes_mass[i-1] if 0 < i <= n_el else None,
            })

    # -------------------------
    # Create elements
    # -------------------------
    elements = []
    for i in range(n_el):
        p1 = np.array(nodes[i]["position"], dtype=float)
        p2 = np.array(nodes[i+1]["position"], dtype=float)
        L_e = float(np.linalg.norm(p2 - p1))
        elements.append({
            "nodes": [i, i + 1],
            "stiffness": nodes_stiffness[i],
            "mass": nodes_mass[i],
            "length": L_e
        })

    return {"nodes": nodes, "elements": elements}
