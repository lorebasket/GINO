import sys
import numpy as np
import time
import os

# Import from our modules
#\import sys
sys.path.append('/home/lorebasket/FSI/FEA')
sys.path.append('/home/lorebasket/FSI/SONATA/4_NACA0015')
sys.path.append('/home/lorebasket/FSI/')

#from .1_beams.beam_properties import create_beam_model, n_nodes, E, Iyy, beam_length, b, h, s
from FEA.fea_utl.old import Bauchau_stiffness_matrix_assembly, mass_matrix_assembly_for_SONATA
from hydrofoils.beam_properties import create_beam_model
from fea_utl import analysis, visualization, debug_matrices
from csv_export import parser as ps
from PanelAero import fluid_mainv2

from Blade_library import NACA0015 as contourn

def main():
    # =========================== BEAM PROPERTIES DEFINITION =========================== #
    print("=== Cantilever Beam Analysis ===")
    print("\nCreating beam model...")
    start_time = time.time()
    blade_name = "4_NACA0015"
    beam_length = 4 # m
    n_elements = 100

    
    # Parse section properties from SONATA output
    sections_props_csv = '/home/lorebasket/FSI/SONATA/' + blade_name + '/csv_export/' + blade_name + '_section_data.csv'
    section_props = ps.parse_section_props_csv(sections_props_csv)

    # Get stiffness matrices
    K = ps.parse_sectional_matrix_csv('/home/lorebasket/FSI/SONATA/' + blade_name + '/csv_export/' + blade_name + '_anbax_beam_properties_stiff_matrices.csv')
    M = ps.parse_sectional_matrix_csv('/home/lorebasket/FSI/SONATA/' + blade_name + '/csv_export/' + blade_name + '_anbax_beam_properties_mass_matrices.csv')
    print(f"Stiffness and mass matrices parsed in {time.time() - start_time:.3f} seconds")
    #print(K)
    #print(M)

    # Transform K,M matrices from NA to SC
    # Sonata local REF_FRAME to FEM REF_FRAME
    K, M = ps.transform_matrices(sections_props_csv, K, M, beam_length)
    print(f"Stiffness and mass matrices transformed in {time.time() - start_time:.3f} seconds")
    #print(K)
    #print(M)

    # Create beam model
    beam_model = create_beam_model(K, M, beam_length, n_elements )
    nodes = beam_model["nodes"]
    n_nodes = len(nodes)
    total_dof = len(nodes)*6
    print(f"Beam model created in {time.time() - start_time:.3f} seconds")
    
    # Assemble global stiffness matrix
    print("\nAssembling global stiffness matrix...")
    start_time = time.time()
    K_global = Bauchau_stiffness_matrix_assembly.assemble_global_stiffness_matrix(beam_model)
    print(f"Stiffness matrix assembled in {time.time() - start_time:.3f} seconds")
    
    # Assemble global mass matrix for modal analysis
    print("\nAssembling global mass matrix...")
    start_time = time.time()
    M_global = mass_matrix_assembly_for_SONATA.assemble_global_mass_matrix(beam_model)
    print(f"Mass matrix assembled in {time.time() - start_time:.3f} seconds")
    
    # =========================== STATIC ANALYSIS =========================== #
    # Create force vector
    #load_magnitude = sin(6*wt)
    load_magnitude_max = 1
    force_vector = analysis.create_force_vector(-load_magnitude_max, total_dof)
    
    # Solve static analysis
    print("\nSolving static analysis...")
    start_time = time.time()
    u_full, reaction_forces = analysis.solve_static_analysis(K_global, force_vector, total_dof)
    print(f"Static analysis completed in {time.time() - start_time:.3f} seconds")
    
    # Extract displacements
    print("\nExtracting displacements...")
    displacements = analysis.extract_displacements(u_full, n_nodes)

    # Print maximum displacement
    tip_disp = displacements[n_nodes-1]["y"]
    print(f"Load magnitude applied at the tip: {load_magnitude_max:.1f} N")
    print(f"Tip vertical displacement: {tip_disp:.10f} m")
    
    # Print reaction forces at the constrained end
    print("\nReaction forces at fixed end:")
    for i, force in enumerate(reaction_forces):
        dof_name = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"][i]
        print(f"  {dof_name}: {force:.2f}")

    # =========================== DRY MODAL ANALYSIS =========================== #
    # Perform modal analysis
    print("\nPerforming modal analysis...")
    start_time = time.time()
    frequencies, mode_shapes, omega_n, eigvecs, M_ff, K_ff = analysis.modal_analysis(K_global, M_global, total_dof, num_modes=5)
    print(f"Modal analysis completed in {time.time() - start_time:.3f} seconds")
    # Print natural frequencies
    print("\nNatural frequencies:")
    for i, freq in enumerate(frequencies):
        print(f"  Mode {i+1}: {freq:.4f} Hz")
    ## Plot mode shapes
    #print("\nPlotting mode shapes...")
    #visualization.plot_mode_shapes(nodes, mode_shapes, frequencies, K_global)


    # Perform modal analysis with damping
    print("\nPerforming modal analysis with damping...")
    start_time = time.time()
    frequencies, mode_shapes, damping_ratios, alpha, beta = analysis.modal_analysis_with_damping(
    K_global, M_global, total_dof, num_modes=6, target_damping=(0.02, 0.02)
    )
    # Plot mode shapes
    #visualization.plot_mode_shapes(nodes, mode_shapes, frequencies, damping_ratios)

    print("\n=== Dry modal analysis completed successfully ===")

    # =========================== WET FLUID ANALYSIS =========================== #
    # Perform fluid analysis
    print("\nPerforming fluid analysis...")
    csv_filename = '/home/lorebasket/FSI/SONATA/' + blade_name + '/csv_export/' + blade_name + '_section_data.csv'
    #extension = ".csv"
    
    output_dir_addedcomp = "/home/lorebasket/FSI/PanelAero/csv_export/added_components/" + blade_name
    #output_filename = '/home/lorebasket/FSI/PanelAero/2_symmetric-blades/' + blade_name + extension
    #output_dir_addedcomp = '/home/lorebasket/FSI/PanelAero/csv_export/added_components'

    # input parameters
    rho_f = 1000 # water density [kg/m3]
    chord_tip = 1 # [m]
    chord_root = 1 # [m]
    nspan = 10 # number of spanwise panels
    nchord = 5 # number of chordwise panels

    # DLM inputs
    attack_angle = 10 # °grad
    V = [25] # inflow velocity [m/s]
    f = 3 # induced frequency [Hz]

    # Contour coordinates
    contour_coords = contourn.main()

    # Run fluid main
    M_added, C_added = fluid_mainv2.main(
        blade_name, section_props, csv_filename, output_dir_addedcomp,
        beam_model, beam_length, rho_f, chord_tip, chord_root, nspan, nchord,
        attack_angle, V, f, contour_coords)

    # Debug prints to check matrices
    print("\nMatrix checks:")
    print("M_added contains NaNs:", np.isnan(M_added).any())
    print("M_added contains infs:", np.isinf(M_added).any())
    print("M_added shape:", M_added.shape)
    print("M_global contains NaNs before addition:", np.isnan(M_global).any())
    print("M_global contains infs before addition:", np.isinf(M_global).any())
    print("M_global shape:", M_global.shape)

    print("C_added contains NaNs:", np.isnan(C_added).any())
    print("C_added contains infs:", np.isinf(C_added).any())
    print("C_added shape:", C_added.shape)

    # Add added mass to global mass matrix
    print(np.linalg.norm(M_added))

    # Replace this block:
    # M_global += M_added
    
    # With this:
    # Debug and fix FSI matrices
    print("\n" + "="*60)
    print("DEBUGGING AND FIXING FSI MATRICES")  
    print("="*60)
    
    # Save original dry mass matrix
    M_global_dry = M_global.copy()
    
    # Debug and fix the matrices
    M_global_wet, M_added_fixed = debug_matrices.debug_and_fix_fsi_matrices(
        M_global_dry, M_added, K_global, Z, aerogrid, beam_model
    )
    
    # Update global mass matrix
    M_global = M_global_wet

    print("\nAfter addition:")
    print("M_global contains NaNs:", np.isnan(M_global).any())
    print("M_global contains infs:", np.isinf(M_global).any())
    print("\Global mass matrix saved to CSV files:")
    np.savetxt('global_mass_matrix.csv', M_global, delimiter=',')
    
    # Perform wet modal analysis
    print("\nPerforming wet modal analysis...")
    start_time = time.time()
    frequencies_wet, mode_shapes_wet, damping_ratios_wet, alpha_wet, beta_wet = analysis.modal_analysis_with_damping(
    K_global, M_global, total_dof, num_modes=6, target_damping=(0.05, 0.05), Cadded=C_added
    )
    print(f"Modal analysis completed in {time.time() - start_time:.3f} seconds")
    
    # Print wet natural frequencies
    print("\nWet natural frequencies:")
    for i, freq in enumerate(frequencies_wet):
        print(f"  Mode {i+1}: {freq:.6f} Hz")
    ## Plot wet mode shapes
    #print("\nPlotting wet mode shapes...")
    #visualization.plot_mode_shapes(nodes, mode_shapes_wet, frequencies_wet, K_global)


if __name__ == "__main__":
    main()