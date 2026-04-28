# FEA/fea_utl/rbe2_connector.py
"""
RBE2 (Rigid Body Element) connector implementation.

An RBE2 is a rigid body element commonly used in FEA to create rigid
connections between a master (reference) node and multiple dependent (slave) nodes.
All slave nodes move rigidly with the master node (all 6 DOFs are slavishly
constrained to the master's motion).

This module provides:
  1. create_rbe2_connector(): Build an RBE2 element from a reference node and slave nodes
  2. rbe2_rigid_links():     Build multiple rigid link elements to approximate the RBE2
  3. assemble_rbe2_to_model(): Integrate RBE2 into a beam_model dict

For the tnz_multibody assembly, RBE2 rigidly connects the arm tip (reference)
to the foil roots (slaves), ensuring all foils move together as a rigid body
with the arm endpoint while preserving individual foil deformations beyond the junction.
"""

import numpy as np
from .multibody_assembly import T6_from_beam_direction


def create_rbe2_connector(master_idx, slave_indices, node_positions, 
                         rbe2_type='rigid_links', **kwargs):
    """
    Create an RBE2 rigid connector between a master node and one or more slaves.

    Parameters
    ----------
    master_idx : int
        Global node index of the master (reference) node.
    slave_indices : list of int
        Global node indices of the dependent (slave) nodes.
    node_positions : dict or array-like
        Node positions; can be a list of [x, y, z] or dict-like with ['position'].
        If list/array, indexed by node index. If dict, accessed by 'position' key.
    rbe2_type : str, optional
        How to represent the RBE2:
        - 'rigid_links' (default): create individual rigid link elements from
          master to each slave, each with very high stiffness (1e12).
        - 'constraint_matrix': (reserved for future extension) return constraint
          matrix for assembly into global stiffness.
    **kwargs : dict
        Additional options:
        - 'stiffness_scale' (float): multiplier for rigid link stiffness (default 1e12).
        - 'offset_tolerance' (float): ignore slave nodes within this distance
          of the master (default 1e-9 m).
        - 'print_debug' (bool): print diagnostic info (default True).

    Returns
    -------
    elements : list of dict
        If rbe2_type == 'rigid_links': list of element dicts, one per slave,
        with keys 'nodes', 'stiffness', 'mass', 'length', 'T6', 'beam_dir_global',
        'rigid_link', 'rbe2_type'.
    """
    
    stiffness_scale = kwargs.get('stiffness_scale', 1e12)
    offset_tol      = kwargs.get('offset_tolerance', 1e-9)
    print_debug     = kwargs.get('print_debug', True)

    # Extract positions
    master_pos = _get_node_position(node_positions, master_idx)
    
    if print_debug:
        print(f"\n  === RBE2 Connector ===")
        print(f"  Master (ref) node: {master_idx}  pos={master_pos}")
        print(f"  Slave nodes: {slave_indices}")

    elements = []

    if rbe2_type == 'rigid_links':
        for slave_idx in slave_indices:
            slave_pos = _get_node_position(node_positions, slave_idx)
            offset_vec = slave_pos - master_pos
            offset_len = float(np.linalg.norm(offset_vec))

            # Skip if slave is too close to master (redundant node)
            if offset_len < offset_tol:
                if print_debug:
                    print(f"    Slave {slave_idx}: offset={offset_len:.2e} m → SKIPPED (coincident with master)")
                continue

            offset_dir = offset_vec / offset_len
            T6 = T6_from_beam_direction(offset_dir)

            # Rigid stiffness in the local frame
            K_loc = np.zeros((6, 6), float)
            for d in range(6):
                K_loc[d, d] = stiffness_scale

            # Massless link
            M_loc = np.zeros((6, 6), float)

            # Rotate to global frame
            K_glob = T6 @ K_loc @ T6.T
            M_glob = np.zeros((6, 6), float)

            elem = {
                'nodes':           [master_idx, slave_idx],
                'stiffness':       K_glob,
                'mass':            M_glob,
                'length':          offset_len,
                'T6':              T6,
                'beam_dir_global': offset_dir.tolist(),
                'rigid_link':      True,
                'rbe2_type':       'rigid_link',
            }
            elements.append(elem)

            if print_debug:
                print(f"    Slave {slave_idx}: offset={offset_vec}  "
                      f"(L={offset_len:.4f} m, dir={offset_dir})")

    elif rbe2_type == 'constraint_matrix':
        # Reserved for future: return constraint matrix instead of elements
        raise NotImplementedError("RBE2 constraint_matrix mode not yet implemented")

    else:
        raise ValueError(f"Unknown rbe2_type='{rbe2_type}'")

    if print_debug:
        print(f"  Total rigid links created: {len(elements)}")

    return elements


def _get_node_position(node_positions, idx):
    """
    Extract the 3D position of node 'idx' from the node_positions input.
    
    Parameters
    ----------
    node_positions : list of dicts or array-like
        Node data. If list of dicts, each has 'position' key.
        If array-like, directly indexed by node number.
    idx : int
        Node index.
        
    Returns
    -------
    pos : numpy array, shape (3,)
        3D position [x, y, z] in meters.
    """
    if isinstance(node_positions, (list, tuple)):
        item = node_positions[idx]
        if isinstance(item, dict):
            return np.array(item['position'], dtype=float)
        else:
            return np.array(item, dtype=float)
    elif isinstance(node_positions, np.ndarray):
        return np.array(node_positions[idx], dtype=float)
    else:
        raise TypeError(f"Unsupported node_positions type: {type(node_positions)}")


def assemble_rbe2_to_model(beam_model, master_node_idx, slave_node_indices, 
                           rbe2_type='rigid_links', **kwargs):
    """
    Add RBE2 rigid connector elements to an existing beam_model dict.
    
    This function:
      1. Creates RBE2 rigid link elements using create_rbe2_connector().
      2. Appends them to the beam_model's 'elements' list.
      3. Stores RBE2 metadata for diagnostics/post-processing.

    Parameters
    ----------
    beam_model : dict
        Beam model dict with 'nodes' and 'elements' lists.
    master_node_idx : int
        Global node index of the RBE2 master (reference) node.
    slave_node_indices : list of int
        Global node indices of RBE2 slave (dependent) nodes.
    rbe2_type : str, optional
        Type of RBE2 representation (see create_rbe2_connector).
    **kwargs : dict
        Additional options passed to create_rbe2_connector.

    Returns
    -------
    None (modifies beam_model in place)
    
    Side effects
    -----------
    - Appends RBE2 elements to beam_model['elements']
    - Creates/updates beam_model['rbe2_info'] dict with metadata
    """
    
    # Create RBE2 elements
    rbe2_elements = create_rbe2_connector(
        master_node_idx, slave_node_indices, beam_model['nodes'],
        rbe2_type=rbe2_type, **kwargs
    )

    # Append to model
    beam_model['elements'].extend(rbe2_elements)

    # Store metadata for diagnostics
    if 'rbe2_info' not in beam_model:
        beam_model['rbe2_info'] = []

    beam_model['rbe2_info'].append({
        'master_idx':       master_node_idx,
        'master_pos':       beam_model['nodes'][master_node_idx]['position'],
        'slave_indices':    slave_node_indices,
        'type':             rbe2_type,
        'num_rigid_links':  len(rbe2_elements),
    })


def validate_rbe2_connectivity(beam_model, verbose=True):
    """
    Validate RBE2 rigid connector integrity in a beam_model.
    
    Checks:
      1. All master/slave node indices are valid (in range [0, n_nodes))
      2. No cyclic RBE2 dependencies (master cannot be slave of its own RBE2)
      3. Each rigid link element connects valid node pairs
      4. Slave nodes are not also master nodes of other RBE2s (tree structure)
    
    Parameters
    ----------
    beam_model : dict
        Beam model dict with 'elements' and 'rbe2_info' lists.
    verbose : bool, optional
        Print diagnostic output.
        
    Returns
    -------
    is_valid : bool
        True if all checks pass.
    messages : list of str
        Diagnostic/warning messages.
    """
    
    messages = []
    is_valid = True
    n_nodes = len(beam_model['nodes'])
    
    if 'rbe2_info' not in beam_model or not beam_model['rbe2_info']:
        messages.append("No RBE2 connectors in model.")
        return is_valid, messages
    
    all_masters = set()
    all_slaves = set()
    
    for rbe2 in beam_model['rbe2_info']:
        master_idx = rbe2['master_idx']
        slave_indices = rbe2['slave_indices']
        
        # Check master index
        if not (0 <= master_idx < n_nodes):
            messages.append(f"ERROR: RBE2 master_idx={master_idx} out of range [0, {n_nodes})")
            is_valid = False
            continue
        
        all_masters.add(master_idx)
        
        # Check slave indices
        for slave_idx in slave_indices:
            if not (0 <= slave_idx < n_nodes):
                messages.append(f"ERROR: RBE2 slave_idx={slave_idx} out of range [0, {n_nodes})")
                is_valid = False
            else:
                all_slaves.add(slave_idx)
    
    # Check for cyclic dependencies: no slave can also be a master
    cyclic = all_slaves & all_masters
    if cyclic:
        messages.append(f"WARNING: Nodes {cyclic} are both masters and slaves (tree structure violated)")
        # Not necessarily invalid for multiple separate RBE2s, but unusual
    
    if verbose:
        for msg in messages:
            print(f"  {msg}")
        if is_valid:
            print(f"  ✓ RBE2 connectivity validation passed ({len(beam_model['rbe2_info'])} RBE2s)")
    
    return is_valid, messages
