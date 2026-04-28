    
def build_sonata_beam(sys, FSI_path, blade_name, n_elements, alpha_r, flutter_benchmark, sonata):
    
    import sys

    """
    The nodes of the beam model (FEA) are gonna be centered along the ShearCenter positions
    The ShearCenter positions are evaluated from SONATA.
    Each radial station has a point corresponding to the ShearCenter
    """

    # Blade definition parameters
    chord_tip = 0.368 # [m]
    chord_root = 0.559 # [m]
    b = ( chord_tip + chord_root ) / 4
    beam_length = np.sqrt(2*(0.762)**2) # m
    pitch = 45 # degree
    
    # Parse SONATA stiffness and mass matrices
    K = ps.parse_sectional_matrix_csv(FSI_path + '/SONATA/' + blade_name + '/csv_export/' + blade_name + '_anbax_beam_properties_stiff_matrices.csv')
    M = ps.parse_sectional_matrix_csv(FSI_path + '/SONATA/' + blade_name + '/csv_export/' + blade_name + '_anbax_beam_properties_mass_matrices.csv')
    
    # Transform K,M matrices from BEAM REFERENCE AXIS to SC
    sections_props_csv = FSI_path + '/SONATA/' + blade_name + '/csv_export/' + blade_name + '_section_data.csv'
    K, M = ps.transform_matrices(sections_props_csv, K, M, beam_length, sonata, flutter_benchmark)
    K, M = ps.rotate_matrices_by_angle(K, M, 'z', alpha_r) # why the fuck am i rotating around z?????

    # Create beam model
    beam_model = create_beam_model(K, M, beam_length, n_elements, pitch)
    
    return beam_model, K, M