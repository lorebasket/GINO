#!/usr/bin/env python
# === CREATE BEAM MODEL === #

def create_beam_model(K, M):
    """
    Create the beam model with nodes, elements and stiffness assignments
    """

    # Beam parameters
    n_el = 60
    n_nodes = n_el + 1

    beam_length = 0.305

    element_length = beam_length / n_el
    dof_per_node = 6  # 3 displacement + 3 rotation
    dof_per_element = 12  # 6 DOF per node * 2 nodes per element
    total_dof = (n_nodes) * dof_per_node


    K1 = K['0.0']
    K2 = K['1.0']

    M1 = M['0.0']
    M2 = M['1.0']

    # Define the beam portions for each stiffness matrix
    first_portion_end = int(n_nodes * 0.5)
    second_portion_end = int(n_nodes * 1)

    first_portion = list(range(first_portion_end))
    second_portion = list(range(first_portion_end, n_nodes))

    print(f"Beam portions: {len(first_portion)} nodes in first, {len(second_portion)} in second")

    # Map nodes to stiffness matrices
    nodes_stiffness = []
    for i in range(n_nodes):
        if i < first_portion_end - 1:
            nodes_stiffness.append(K1)
        elif i < second_portion_end - 1:
            nodes_stiffness.append(K2)

    # Map nodes to mass matrices
    nodes_mass = []
    for i in range(n_nodes):
        if i < first_portion_end - 1:
            nodes_mass.append(M1)
        elif i < second_portion_end - 1:
            nodes_mass.append(M2)


    # Generate node positions
    nodes = []
    for i in range(n_nodes):
        x = i * element_length
        nodes.append({"position": [x, 0, 0], "index": i})

    # Define element connectivity
    elements = []
    for i in range(n_el):
        elements.append({"nodes": [i, i+1], "stiffness":nodes_stiffness[i], "mass": nodes_mass[i], "length": element_length})
    
    # Return model details
    return {
        "nodes": nodes,
        "elements": elements,
        "first_portion_end": first_portion_end,
        "second_portion_end": second_portion_end
    }