import sys
import numpy as np
import time
import os
import numpy as np
from FEA.fea_utl.old import Bauchau_stiffness_matrix_assembly, mass_matrix_assembly_for_SONATA
import scipy.linalg
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Import from our modules
sys.path.append('/home/lorebasket/FSI/FEA')
sys.path.append('/home/lorebasket/FSI/SONATA/4_NACA0015')
sys.path.append('/home/lorebasket/FSI/')

from hydrofoils.beam_properties import create_beam_model
from fea_utl import analysis, visualization
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
    n_elements = 20

    
    # Parse section properties from SONATA output
    sections_props_csv = '/home/lorebasket/FSI/SONATA/' + blade_name + '/csv_export/' + blade_name + '_section_data.csv'
    section_props = ps.parse_section_props_csv(sections_props_csv)

    # Get stiffness matrices
    K = ps.parse_sectional_matrix_csv('/home/lorebasket/FSI/SONATA/' + blade_name + '/csv_export/' + blade_name + '_anbax_beam_properties_stiff_matrices.csv')
    M = ps.parse_sectional_matrix_csv('/home/lorebasket/FSI/SONATA/' + blade_name + '/csv_export/' + blade_name + '_anbax_beam_properties_mass_matrices.csv')
    print(f"Stiffness and mass matrices parsed in {time.time() - start_time:.3f} seconds")

    # Transform K,M matrices from NA to SC
    # Sonata local REF_FRAME to FEM REF_FRAME
    K, M = ps.transform_matrices(sections_props_csv, K, M, beam_length)
    print(f"Stiffness and mass matrices transformed in {time.time() - start_time:.3f} seconds")

    # Create beam model
    beam_model = create_beam_model(K, M, beam_length, n_elements )
    nodes = beam_model["nodes"]
    n_nodes = len(nodes)
    total_dof = len(nodes) * 6
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


    # =========================== WET FLUID ANALYSIS =========================== #
    # Perform fluid analysis
    print("\nPerforming fluid analysis...")
    csv_filename = '/home/lorebasket/FSI/SONATA/' + blade_name + '/csv_export/' + blade_name + '_section_data.csv'
    
    output_dir_addedcomp = "/home/lorebasket/FSI/PanelAero/csv_export/added_components/" + blade_name

    # input parameters
    rho_f = 1000 # water density [kg/m3]
    chord_tip = 1 # [m]
    chord_root = 1 # [m]
    nspan = 10 # number of spanwise panels
    nchord = 5 # number of chordwise panels

    # DLM inputs
    attack_angle = 10 # °grad
    V = [10] # inflow velocity [m/s]
    f = 14 # induced frequency [Hz]

    # Contour coordinates
    contour_coords = contourn.main()
    traslate = False

    # Run fluid main
    M_added, C_added = fluid_mainv2.main(
        blade_name, section_props, traslate, csv_filename, output_dir_addedcomp,
        beam_model, beam_length, rho_f, chord_tip, chord_root, nspan, nchord,
        attack_angle, V, f, contour_coords, fix_Madded=False)

    # After you get M_added and C_added from fluid analysis:
    M_added_real = np.real(M_added)  # Use only real part for stability
    C_added_real = np.real(C_added)  # Use only real part for stability

    C_global = 0.1 * M_global + 0.1 * K_global
    
    # Choose solution method
    USE_MODAL_REDUCTION = False  # Set this flag
    
    if USE_MODAL_REDUCTION:
        print("\n=== SOLVING WITH MODAL REDUCTION ===")
        
        # Solve using modal reduction (much more efficient)
        displacement_fsi, velocity_fsi, acceleration_fsi, modal_data = solve_fsi_with_modal_reduction(
            K_global=K_global,
            M_global=M_global,
            C_global=C_global,
            M_added=M_added_real,
            C_added=C_added_real,
            excitation_force=excitation_force,
            dt=dt,
            n_steps=n_steps,
            n_modes=10  # Start with 10 modes, increase if needed
        )
        
        # Extract wet frequencies
        frequencies_wet = modal_data['frequencies_wet']
        mode_shapes_wet = modal_data['mode_shapes']
        
    else:
        print("\n=== SOLVING WITH DIRECT METHOD ===")

        # =========================== FIXING ADDED MASS MATRIX =========================== #
        # Debug original matrices
        print("\nOriginal matrix checks:")
        print("M_added contains NaNs:", np.isnan(M_added).any())
        print("M_added contains infs:", np.isinf(M_added).any())
        print("M_added is complex:", np.iscomplexobj(M_added))
        print("M_added shape:", M_added.shape)
        print("M_added min/max:", np.min(M_added), np.max(M_added))
        print("M_added norm:", np.linalg.norm(M_added))

        print("\nC_added contains NaNs:", np.isnan(C_added).any())
        print("C_added contains infs:", np.isinf(C_added).any())
        print("C_added shape:", C_added.shape)

        # Check if global mass matrix is positive definite before addition
        try:
            print("\n" + "="*60)
            print("DRY MODAL ANALYSIS")
            print("="*60)
            
            start_time = time.time()
            frequencies_dry, mode_shapes, omega_n, eigvecs, M_ff, K_ff = analysis.modal_analysis(K_global, M_global, total_dof, num_modes=5)
            print(f"Modal analysis completed in {time.time() - start_time:.3f} seconds")
            # Print natural frequencies
            print("\nNatural frequencies:")
            for i, freq in enumerate(frequencies_dry):
                print(f"  Mode {i+1}: {freq:.4f} Hz")

            # Perform modal analysis with damping
            print("\nPerforming modal analysis with damping...")
            start_time = time.time()
            frequencies, mode_shapes, damping_ratios, alpha, beta = analysis.modal_analysis_with_damping(
            K_global, M_global, total_dof, num_modes=6, target_damping=(0.02, 0.02)
            )

            print("\n" + "="*60)
            print("WET MODAL ANALYSIS")
            print("="*60)

            M_global += M_added

            # Perform wet modal analysis with damping
            print("\nPerforming modal analysis with damping...")
            start_time = time.time()
            frequencies_wet, mode_shapes_wet, damping_ratios_wet, alpha_wet, beta_wet = analysis.modal_analysis_with_damping(
            K_global, M_global, total_dof, num_modes=6, target_damping=(0.02, 0.02, C_added)
            )

            print("\nNatural frequencies:")
            for i, freq in enumerate(frequencies_wet):
                print(f"  Mode {i+1}: {freq:.4f} Hz")
            
            print("\n" + "="*60)
            print("DRY-WET COMPARISON")
            print("="*60)

            # Compare with dry frequencies
            print(f"\nFrequency comparison:")
            for i in range(min(len(frequencies_dry), len(frequencies_wet))):
                change_pct = ((frequencies_wet[i] - frequencies_dry[i])/frequencies_dry[i]*100)
                print(f"  Mode {i+1}: {frequencies_dry[i]:.6f} Hz → {frequencies_wet[i]:.6f} Hz ({change_pct:+.2f}%)")

            
            print(f"Success! Wet natural frequencies computed with direct method.")

        except Exception as e:
            print(f"Could not compute eigenvalues: {e}")

        ## Add fixed mass matrix
        #M_global_wet = M_global + M_added
        #
        ## Check combined matrix
        #print("\nChecking combined wet mass matrix...")
        #try:
        #    eigs_wet = scipy.linalg.eigh(K_global, M_global_wet)
        #    min_eig_wet = np.min(np.real(eigs_wet))
        #    print(f"Wet mass matrix min eigenvalue: {min_eig_wet:.2e}")
        #    print(f"Wet mass matrix is positive definite: {min_eig_wet > 0}")
        #    print(f"Wet mass matrix condition number: {np.linalg.cond(M_global_wet):.2e}")
        #
        #    wet_frequencies = np.sqrt(eigs_wet)
        #    print(f"Success! Wet natural frequencies computed with direct method.")
        #    for i, freq in enumerate(wet_frequencies):
        #            print(f"  Mode {i+1}: {freq:.6f} Hz")
        #            
        #except Exception as e:
        #    print(f"Could not compute wet eigenvalues: {e}")

        # Save matrices for debugging
        print("\nSaving matrices to CSV files...")
        np.savetxt('dry_mass_matrix.csv', M_global - M_added, delimiter=',')
        np.savetxt('wet_mass_matrix.csv', M_global, delimiter=',')

        ## =========================== ROBUST WET MODAL ANALYSIS =========================== #
        #print("\n" + "="*60)
        #print("ROBUST WET MODAL ANALYSIS")
        #print("="*60)
        #
        #start_time = time.time()
        #frequencies_wet, mode_shapes_wet, damping_ratios_wet, alpha_wet, beta_wet = analysis.robust_modal_analysis_with_damping(
        #    K_global, M_global, total_dof, num_modes=6, target_damping=(0.05, 0.05), Cadded=C_added
        #)
        #
        #print(f"Modal analysis completed in {time.time() - start_time:.3f} seconds")
        #
        ## Print results
        #if len(frequencies_wet) > 0:
        #    print(f"\nSuccessfully computed {len(frequencies_wet)} wet natural frequencies:")
        #    for i, freq in enumerate(frequencies_wet):
        #        if i < len(damping_ratios_wet):
        #            print(f"  Mode {i+1}: {freq:.6f} Hz (damping: {damping_ratios_wet[i]:.4f})")
        #        else:
        #            print(f"  Mode {i+1}: {freq:.6f} Hz")
        #
        #    # Compare with dry frequencies
        #    if len(frequencies) > 0:
        #        print(f"\nFrequency comparison:")
        #        for i in range(min(len(frequencies), len(frequencies_wet))):
        #            change_pct = ((frequencies_wet[i] - frequencies[i])/frequencies[i]*100)
        #            print(f"  Mode {i+1}: {frequencies[i]:.6f} Hz → {frequencies_wet[i]:.6f} Hz ({change_pct:+.2f}%)")
        #
        #    # Print summary
        #    if len(frequencies_wet) > 0 and len(frequencies) > 0:
        #        avg_change = np.mean([((frequencies_wet[i] - frequencies[i])/frequencies[i]*100) 
        #                            for i in range(min(len(frequencies), len(frequencies_wet)))])
        #        print(f"\nAverage frequency change due to added mass: {avg_change:+.2f}%")
        #
        #else:
        #    print("\nERROR: No wet frequencies computed successfully!")
        #    print("The added mass matrix may still have issues.")
        ##
        ##    # Try fallback strategy
        ##    print("\nTrying fallback strategy with minimal added mass...")
        ##    try:
        ##        # Scale down added mass significantly
        ##        M_added_minimal = M_added_fixed * 0.01
        ##        M_global_minimal = M_global - M_added_fixed + M_added_minimal
        ##
        ##        frequencies_minimal, _, _, _, _ = analysis.robust_modal_analysis_with_damping(
        ##            K_global, M_global_minimal, total_dof, num_modes=3, target_damping=None
        ##        )
        ##
        ##        if len(frequencies_minimal) > 0:
        ##            print("Success with minimal added mass:")
        ##            for i, freq in enumerate(frequencies_minimal):
        ##                print(f"  Mode {i+1}: {freq:.6f} Hz")
        ##
        ##    except Exception as e:
        ##        print(f"Fallback strategy also failed: {e}")

        print("\n=== Wet modal analysis completed ===")

if __name__ == "__main__":
    main()