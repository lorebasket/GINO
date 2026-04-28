#!/usr/bin/env python

def create_beam_model(K, M, beam_length, n_el):
    """
    Create the beam model with nodes, elements, and automatic stiffness assignments.
    No radial stations needed; beam portions are assigned based on number of matrices.
    """
    n_nodes = n_el + 1
    element_length = beam_length / n_el
    dof_per_node = 6  # 3 displacements + 3 rotations

    # Determine portions automatically
    n_portions = len(K)
    portion_size = n_el // n_portions
    remaining_elements = n_el % n_portions

    sorted_keys = sorted(K.keys())
    
    # Prepare lists for stiffness and mass assignments (per element)
    nodes_stiffness = []
    nodes_mass = []
    element_counter = 0
    for i, key in enumerate(sorted_keys):
        n_elements_in_portion = portion_size + (1 if i < remaining_elements else 0)
        for _ in range(n_elements_in_portion):
            nodes_stiffness.append(K[key])
            nodes_mass.append(M[key])
            element_counter += 1

    assert len(nodes_stiffness) == n_el, "Mismatch in stiffness assignment"
    
    # Create nodes (kept exactly as your original format)
    nodes = []
    for i in range(n_nodes):
        x = i * element_length
        nodes.append({"position": [x, 0, 0], "index": i})

    # Create elements (kept exactly as your original format)
    elements = []
    for i in range(n_el):
        elements.append({
            "nodes": [i, i + 1],
            "stiffness": nodes_stiffness[i],
            "mass": nodes_mass[i],
            "length": element_length
        })

    # Print partitioning info
    print("Beam portions (automatic):")
    for i, key in enumerate(sorted_keys):
        n_elements_in_portion = portion_size + (1 if i < remaining_elements else 0)
        print(f" Portion {i + 1}: {n_elements_in_portion} elements assigned to key '{key}'")

    return {
        "nodes": nodes,
        "elements": elements
    }