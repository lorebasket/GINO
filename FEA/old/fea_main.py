
#from PanelAero.panelaero_utl.coupling_matrix import create_airfoil_profile


from FEA.fea_utl.old import Bauchau_stiffness_matrix_assembly, mass_matrix_assembly_for_SONATA


def main():
    
    import sys
    import numpy as np
    import time
    import pandas as pd
    import matplotlib.pyplot as plt

    #hard_drive = True

    FSI_path = '/media/lorenzo/Seagate Por/recupero dati Linux/lorebasket/FSI'

    # Add all necessary paths to sys.path
    sys.path.extend([
        FSI_path + '/FEA',
        FSI_path + '/FEA/fea_utl',  # Add fea_utl directory
        FSI_path + '/SONATA/6_AGARD445.6',
        FSI_path + '/SONATA/6_AGARD445.6',  # Add parent directory of csv_export
        FSI_path + '/',
        FSI_path + '/PanelAero',
        FSI_path + '/PanelAero/panelaero_utl',
        FSI_path + '/FEA/1_beams',
        FSI_path + '/PanelAero/panelaero_utl/pk_method_utl'
    ])

    from hydrofoils.beam_properties import create_beam_model
    from hydrofoils import agard_4456, sonata_foils
    from rotate_beams import rotate_beam_model_y

    # Import all necessary modules from fea_utl
    from fea_utl import (analysis,
                         visualization,
                         modal_reduction,
                         rigid_body_coupling,
                         estimate_I_wing,
                        )
    
    # Import the parser module using the full path
    import sys
    import os
    
    path = os.path.join(FSI_path, 'SONATA', '6_AGARD445.6', 'csv_export')
    sys.path.append(path)
    
    import parser as ps
    from PanelAero import fluid_main


    print("=== Cantilever Beam Analysis ===")

    # ========= DEFINE TYPE OF ANALYSIS ======== #
    f_spectrum = False # True to calculate Aeroelastic components over a spectrum
    flutter_analysis = True # True for flutter analysis detection
    flutter_benchmark = True # True if running sectional data goland
    sonata = False # True if running analysis with sonata
    agard_theory = False # True if running analysis with agard theory
    structural_analysis = True # True for structural analysis only
    
    # ==================== BEAM PROPERTIES DEFINITION ==================== #
    print("\nCreating beam model...")
    start_time = time.time()
    
    blade_name = "6_AGARD445.6"
    yaml_path = FSI_path + '/SONATA/' + blade_name + '/' + blade_name + '.yaml'
    airfoil_name = "AGARD445.60"  # This must match the name in the YAML file

    n_elements = 40      # number of beam elements
    nspan = 40           # number of spanwise panels
    nchord = 30          # number of chordwise panels

    # DLM inputs
    rho_f = 1.02        # air density [kg/m3]
    #rho_f = 1000        # water density [kg/m3]
    
    # angle of attack
    attack_angle = [0.05]   # °grad
    attack_angle_deg = attack_angle[0]
    alpha_r = np.deg2rad(attack_angle[0]) # angle of attack in radians
    
    # inflow velocities
    V = [140]            # inflow velocity [m/s]  NACA0015 paper
    # Frequencies
    f = [11]             # NACA0015 paper

    # Velocity range for flutter analysis
    V_list = np.linspace(120, 180, 30)


    if agard_theory == True:

        # external module to be checked
        beam_model, K, M, alpha_k, alpha_m, alpha_EIy, alpha_EIz, alpha_GJ = ...
        agard_4456.build_beam(yaml_path, airfoil_name, n_elements)

    if sonata == True:
        
        # external modulee to be checked
        blade_name = "6_AGARD445.6"
        beam_model, K, M = ...
        sonata_foils.build_sonata_beam(sys, FSI_path, blade_name, n_elements, nspan, nchord, rho_f, alpha_r, V, f, flutter_analysis, flutter_benchmark, sonata, agard_theory)

    if flutter_benchmark == True:
        alpha_k = 1
        alpha_m = 1

        blade_name = "GOLAND"
        name = blade_name
        fluid = "air"

        # Reference values from Goland wing
        span_goland = 6.096  # m
        beam_length = span_goland
        chord = 1.8288  # m (6 ft)

        # Material properties
        rho = 2700  # kg/m³ ASSUMPTION
        mu = 35.709721  # kg/m (mass per unit length)
        v = 0.31

        # Calculate other properties based on reference values
        i11 = 8.64
        i22 = 0.1 * i11
        i33 = 0.9 * i11

        #A = mu / rho  # Cross-sectional area
        
        # Reference stiffness values from Goland wing
        EIyy = 9.77221e6
        GJ = 9.87581e5    # N·m²             # make E consistent with EIyy & geometry
        EIzz = 9.77221e6 
        EA  = 1.0e12
        GAy = 1.0e12
        GAz = 1.0e12

        # stiffness
        K = np.zeros((6,6), float)
        K[0,0] = EA          # EA
        K[1,1] = GAy   # GA_y ≈ G * A_sy
        K[2,2] = GAz  # GA_z
        K[3,3] = GJ             # GJ (given)
        K[4,4] = EIyy
        K[5,5] = EIzz*100           # now geometric, not mass-inertia derived

        xcm = 0.43*chord        # m (center of mass, from leading edge)
        xea = 0.33*chord        # m (elastic axis, from leading edge)
        e = (0.43 - 0.33) * chord

        # mass
        M = np.zeros((6,6), float)
        # translations
        M[0,0] = mu;  M[1,1] = mu;  M[2,2] = mu
        # cross blocks
        M[0,5] = -mu*e; M[5,0] = -mu*e
        M[2,3] = +mu*e; M[3,2] = +mu*e
        # rotations
        M[3,3] = i11 + mu*e**2
        M[4,4] = i22
        M[5,5] = i33 + mu*e**2

        pitch = 0  # degrees

        beam_model = create_beam_model(K, M, beam_length, n_elements, pitch, 
                                     agard_theory, flutter_benchmark, sonata)
        beam_model = rotate_beam_model_y(beam_model, attack_angle)
    
    if structural_analysis == True:

        nodes = beam_model["nodes"]
        n_nodes = len(nodes)
        total_dof = len(nodes) * 6
        print(f"Beam model created in {time.time() - start_time:.3f} seconds")

        # Assemble global stiffness and mass matrix
        K_global = alpha_k * Bauchau_stiffness_matrix_assembly.assemble_global_stiffness_matrix(beam_model, flutter_benchmark, agard_theory, sonata, K)
        M_global = alpha_m * mass_matrix_assembly_for_SONATA.assemble_global_mass_matrix(beam_model, flutter_benchmark, agard_theory, M)

        # Print matrix ranks for debugging
        print(f"Rank of K_global: {np.linalg.matrix_rank(K_global)} out of {K_global.shape[0]}")

        # =========================== DRY STATIC ANALYSIS =========================== #

        # Create force vector (negative for downward force in z-direction)
        load_magnitude_max = -10.0  # Negative for downward force in z-direction

        # Define boundary conditions - fix all 6 DOFs at the root (first node)
        constrained_dofs = list(range(6))  # Fix all 6 DOFs at node 0
        print(f"Constrained DOFs: {constrained_dofs}")
        force_vector = analysis.create_force_vector(load_magnitude_max, total_dof, axis='z')

        # Solve static analysis
        print("\nSolving static analysis...")
        start_time = time.time()
        u_full, reaction_forces, dry_values, dry_vectors, Mff, Kff = analysis.solve_static_analysis(
            K_global, M_global, 
            force_vector, 
            total_dof,
            constrained_dofs=constrained_dofs,
            num_modes=20
        )

        # Mode shape analysis
        import mode_shape_analysis as shapes
        #shapes.plot_shapes(beam_length, n_elements, dry_vectors)

        # Print natural frequencies
        freqs = np.sqrt(np.abs(dry_values)) / (2 * np.pi)                    # Natural frequencies in Hz
        frequencies_to_print = 5

        print(f"Phi dry: {dry_vectors.shape}")
        print("\nNatural frequencies:")
        for i, freq in enumerate(freqs[:frequencies_to_print]):
            print(f"  Mode {i+1}: {freq:.4f} Hz")

        # Extract displacements
        print("\nExtracting displacements...")
        displacements = analysis.extract_displacements(u_full, n_nodes)      # Displacements at each node
        tip_disp = displacements[n_nodes-1]["z"]                             # Tip displacement in z-direction
        print(f"Load applied at the tip: {load_magnitude_max:.1f} N")
        print(f"Tip vertical displacement: {tip_disp:.10f} m")

        # Print reaction forces at the constrained end
        print("\nReaction forces at fixed end:")                             # Reaction forces at the root
        for i, force in enumerate(reaction_forces):
            dof_name = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"][i]
            print(f"  {dof_name}: {force:.2f}")


        # Perform modal analysis
        print("\nPerforming modal analysis...")
        start_time = time.time()

        # ==== Rayleigh damping computation ==== #
        Cff, alpha, beta, (w1, w2) = analysis.rayleigh_from_two_modes(
                                                    Kff, Mff,
                                                    dry_vectors, dry_values,
                                                    target_mode_ids=(0, 1),
                                                    target_zetas=(0, 0),
                                                    verbose=True
                                                    )

        # ==== Modal analysis ==== #
        print("Mff_shape:", Mff.shape); print("Cff_shape:", Cff.shape); print("Kff_shape:", Kff.shape)
        M_hat, K_hat, C_hat = modal_reduction.reduce_matrices(Mff, Cff, Kff, dry_vectors)
        print("M_hat shape:", M_hat.shape); print("K_hat shape:", K_hat.shape); print("C_hat shape:", C_hat.shape)

        # Get both physical mode shapes and state-space eigenvectors
        dry_freqs, dry_damp_ratios, dry_eigvals, dry_physical_modes, dry_eigvecs = modal_reduction.solve_modes_state_space(M_hat, C_hat, K_hat)

        print(f"Modal analysis completed in {time.time() - start_time:.3f} seconds")

        # Print natural frequencies
        print("\nNatural frequencies:")
        for i, freq in enumerate(dry_freqs[:frequencies_to_print]):
            print(f"  Mode {i+1}: {freq:.4f} Hz")


        print("\n=== Analysis completed successfully ===")

    # =========================== RUN FLUID ANALYSIS =========================== #
    start_time = time.time()
    blade_name = "6_AGARD445.6"
    output_dir = FSI_path + '/PanelAero/2_symmetric-blades'
    wing = output_dir + '/' + blade_name + '.CAERO1' # Read CAERO1 card
    output_dir_ac = FSI_path + '/PanelAero/csv_export/added_components/' + blade_name


    # === Perform fluid analysis over a spectrum, for added components' results === #
    if f_spectrum == True:
        sections_props_csv = 0

        ## ==== MODULE FOR CAERO CARD ==== ##
        results = fluid_main.main(flutter_benchmark, blade_name, wing, output_dir, output_dir_ac, sections_props_csv,
        beam_model, beam_length, rho_f, chord_tip, chord_root, nspan, nchord, attack_angle, V, f)

        print(f"Fluid results calculated in {time.time() - start_time:.3f} seconds")

        # List to collect output data
        start_time = time.time()

        # ==== Output analysis for CAERO and BLADE modules ==== #
        for result in results:
            f = result["f"]
            M_added = result["M_added"]
            C_added = result["C_added"]
            M = M_global + M_added; C = C_global + C_added; K = K_global

            # === Perform wet modal analysis === #
            M_hat, C_hat, K_hat = modal_reduction.reduce_matrices(M, C, K, Psi)
            wet_freqs, wet_damp_ratios, wet_eigvals, wet_mode_shapes = modal_reduction.solve_modes_state_space(M_hat, C_hat, K_hat)

            ## ==== ADDED MASS FOR PIETRO ==== ##
            nodes_position = np.array([node["position"] for node in nodes])  # Convert to NumPy array
            ref_point = np.array([beam_length / 2, 0.0, 0.0])  # Also ensure ref_point is a NumPy array
            Matrix = rigid_body_coupling.build_rigid_body_coupling_matrix(nodes_position, ref_point)

            visualization.excel_results(Matrix, M_added, C_added, M_global, C_global, result, dry_freqs, wet_freqs, n_nodes, nspan, nchord, blade_name, damp_ratios, wet_eigvals)
            print(f"F_spectrum analysis completed in {time.time() - start_time:.3f} seconds")
            print(f"Results exported in {output_dir_ac}")


    # === Perform flutter analysis === #
    if flutter_analysis == True:
        panelareo_utl_path = "/home/lore/FSI/PanelAero/panelaero_utl"
        sys.path.append(panelareo_utl_path)
        import pk_solver
        import aerogrid_generator
        import coupling_matrix as coupler
        import coupling_diagnostics
        import DLM, VLM
        import vgvf_plotting
        import rotate_aerogrid

        import numpy as np, yaml

        sections_props_csv = FSI_path + '/SONATA/' + blade_name + '/csv_export/' + blade_name + '_section_data.csv'

        # ==== Blade geometry ==== #
        if agard_theory == True:
            with open(yaml_path, "r") as f:
                yml = yaml.safe_load(f)
            af = [a for a in yml["airfoils"] if a["name"] == airfoil_name][0]
            
            chord_vals = np.array(yml["components"]["blade"]["outer_shape_bem"]["chord"]["values"], float)
            chord_root = chord_vals[0]
            chord_tip = chord_vals[-1]
        
        elif flutter_benchmark == True:
            chord_root = 1.829
            chord_tip = 1.829

        # ==== Aerogrid generation ==== #
        c_air = 343.0 # [m/s] speed of sound in air
        c_water = 1484.0 # [m/s] speed of sound in water
        c_sound = c_air

        rho_air = 1.02 # [kg/m^3] density of air
        rho_water = 997 # [kg/m^3] density of water
        rho_f = rho_air

        b = chord_tip / 2 # [m] semichord


        #### ==== Aerogrid generation ==== ####
        # old way
        aerogrid = aerogrid_generator.main(FSI_path, blade_name, wing, sections_props_csv, 
        nspan, nchord, chord_tip, chord_root, agard_theory, flutter_benchmark, beam_length)
        
        aerogrid = rotate_aerogrid._rotation(aerogrid, alpha_r, axis='y') # rotate aerogrid
        
        # new way
        from flutter_analysis_workflow import aerodynamic_model
        
        #qjj_dir = f"{FSI_path}/PanelAero/Qjj/qjj_precomputed/{name}_{fluid}_alpha{attack_angle_deg}_nspan{nspan}_nchord{nchord}_klist_new"
        #aerogrid_path = f"{qjj_dir}/aerogrid.npz"
        #aerogrid = aerodynamic_model.build(aerogrid_path)

        # Diagnostic: LIFT
        diagnostic_lift = True
        if diagnostic_lift == True:
            # known inputs
            rho = 1.02             # kg/m^3
            c = 340.0               # m/s
            Uinf = 160.0            # m/s
            qinf = 0.5*rho*Uinf**2
            k = 0.2                 # example reduced frequency
            Ma = Uinf/c

            Qjj_unsteady = DLM.calc_Qjj(aerogrid, k, Ma)     # (n x n)
            Qjj_steady, Bjj = VLM.calc_Qjj(aerogrid, Ma) # check VLM call
            Qjj_total = Qjj_steady + Qjj_unsteady
            print("Qjj_steady shape:", Qjj_steady.shape)

            n = np.ones(aerogrid['n'])

            wj = n * (Uinf * np.sin(alpha_r)) / Uinf
            print("wj shape:", wj.shape)

            dCp_steady = Qjj_steady.dot(wj)                        # (n,)
            dCp_unsteady = Qjj_unsteady.dot(wj)                    # (n,)

            dCp_total = dCp_steady + dCp_unsteady
            print("dCp_total shape:", dCp_total.shape)

            # Panel forces and total lift
            Fxyz = qinf * aerogrid['N'].T * aerogrid['A'] * dCp_total # (n,3)

            print(Fxyz.sum(axis=1)) # total forces in x,y,z in [N]

            #breakpoint()
        
        # ==== Coupling matrix ==== #
        from coupling_matrix import main, create_airfoil_profile
        # Create your NACA 0015 profile (your exact coordinates)
        naca0015 = create_airfoil_profile(
            x_coords=[1.0000, 0.9500, 0.9000, 0.8000, 0.7000, 0.6000, 0.5000, 0.4000, 
                      0.3000, 0.2500, 0.2000, 0.1500, 0.1000, 0.0750, 0.0500, 0.0250, 
                      0.0125, 0.0000, 0.0125, 0.0250, 0.0500, 0.0750, 0.1000, 0.1500, 
                      0.2000, 0.2500, 0.3000, 0.4000, 0.5000, 0.6000, 0.7000, 0.8000, 
                      0.9000, 0.9500, 1.0000],
            z_coords=[0.00158, 0.01008, 0.01810, 0.03279, 0.04580, 0.05704, 0.06617, 
                      0.07254, 0.07502, 0.07427, 0.07172, 0.06682, 0.05853, 0.05250, 
                      0.04443, 0.03268, 0.02367, 0.00000, -0.02367, -0.03268, -0.04443, 
                      -0.05250, -0.05853, -0.06682, -0.07172, -0.07427, -0.07502, 
                      -0.07254, -0.06617, -0.05704, -0.04580, -0.03279, -0.01810, 
                      -0.01008, -0.00158]
        )

        # Now call main with airfoil visualization
        Z, panel_to_node_map, panel_xi_map = main(
            beam_model,
            aerogrid=aerogrid,
            coupling_diagnostics=False,
            plot=False,
            plot_airfoil=False,
            airfoil_profile=naca0015,
            n_airfoil_sections=3
        )
        

        plot_downwash = False
        if plot_downwash == True:

            # Prepare full dry_vectors including constrained DOFs
            dry_vectors_full = np.zeros((total_dof, dry_vectors.shape[1]))
            dry_vectors_full[6:, :] = dry_vectors  # Assuming first 6 DOFs are constrained
            print("dry_vectors_full :", dry_vectors_full)

            print("dry_vectors shape:", dry_vectors.shape)
            print("dry_values shape:", dry_values.shape)
            #print("Z shape:", Z_cantilever.shape)

            # Downwash calculation from modes
            wj = Z @ dry_vectors_full[:, 0]# @ dry_values) # downwash for first mode
            print ("wj before diag shape:", wj.shape)

            wj_diag = np.diag(wj)
            print("wj_diag shape:", wj_diag.shape)
            #breakpoint()

            # Get control point coordinates (you already have this from your code)
            control_points = aerogrid['offset_j'] # control points at 3/4chord
            print("Control points shape:", control_points.shape)

            # Create 3D plot
            fig = plt.figure(figsize=(14, 10))
            ax = fig.add_subplot(111, projection='3d')

            # Plot downwash as colored scatter
            scatter = ax.scatter(control_points[:, 0],  # X (chordwise)
                                 control_points[:, 1],  # Y (spanwise)
                                 control_points[:, 2],  # Z (vertical)
                                 c=wj,                    # Color by downwash
                                 cmap='jet',         # colormap
                                 s=30,                   # Point size
                                 alpha=0.8)

            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
            cbar.set_label('Downwash w (m/s)', rotation=270, labelpad=20)

            # Labels and title
            ax.set_xlabel('X (Chordwise)')
            ax.set_ylabel('Y (Spanwise)')
            ax.set_zlabel('Z (Vertical)')
            ax.set_title('"DRY" Downwash Distribution w = diag(Z @ (vectors @ values)')
            ax.set_box_aspect([1, 2, 0.5])  # Adjust aspect ratio for wing

            plt.tight_layout()
            plt.show()

            # Optional: Print statistics
            print(f"\nDownwash statistics:")
            print(f"  Mean: {np.mean(wj):.6f}")
            print(f"  Std:  {np.std(wj):.6f}")
            print(f"  Min:  {np.min(wj):.6f}")
            print(f"  Max:  {np.max(wj):.6f}")
            
            #breakpoint()


        # === Metodo di ricerca flutter ===
        pk_method   = True   # P–K classico
        eth_method  = False  # ETH / RFA state-space


        # ===== P–K classic method with numerical stabilization ===== #
        if pk_method:
            import pk_solver
            import DLM
            import cantilever_beam
            import Apan_to_fem

            # Coupling matrix
            Z = cantilever_beam.cantilever_beam(Z, total_dof, constrained_dofs=None, dof_per_node=6)
            print("Z shape:", Z.shape)

            # Aerodynamic coupling matrix Apan
            Apan = Apan_to_fem.build_coupling(beam_model, aerogrid, panel_to_node_map, panel_xi_map)
            #print("Apan shape:", Apan.shape)
            Apan_T = Apan.T
            Apan = (cantilever_beam.cantilever_beam(Apan_T, total_dof, constrained_dofs=None, dof_per_node=6)).T
            #print("Apan (cantilever) shape:", Apan.shape)
            #breakpoint()

            # Number of modes to analyze
            modes = [0, 1]#, 2]#, 3, 4]#, 5]#, 6]#, 7, 8]

            # Qjj matrices directory
            out_dir_klist = f"/media/lorenzo/Seagate Por/recupero dati Linux/lorebasket/FSI/PanelAero/Qjj/qjj_precomputed/GOLAND_air_alpha{attack_angle[0]}_nspan{nspan}_nchord{nchord}_klist_new"
            out_dir_ωlist = f"/media/lorenzo/Seagate Por/recupero dati Linux/lorebasket/FSI/PanelAero/Qjj/qjj_precomputed/GOLAND_air_alpha{attack_angle[0]}_nspan{nspan}_nchord{nchord}_ωlist"
            out_dir_vlm = f"/media/lorenzo/Seagate Por/recupero dati Linux/lorebasket/FSI/PanelAero/Qjj/qjj_precomputed/GOLAND_air_alpha{attack_angle[0]}_nspan{nspan}_nchord{nchord}_vlm"
            #out_dir = f"/media/lorenzo/Seagate Por/recupero dati Linux/lorebasket/FSI/PanelAero/Qjj/qjj_precomputed/nspan{nspan}_nchord{nchord}"
            
            # Initialize and run solver
            solver = pk_solver.PKSolverR(M_hat, C_hat, K_hat, dry_values)
            Qg_func = solver.make_Qg_func(FSI_path, Z, Apan, dry_vectors, dry_values, b, c_sound, out_dir_klist, out_dir_vlm, alpha_r)
            
            results = solver.sweep(V_list, rho_f, b, Qg_func, modes, dry_values, dry_vectors, max_iter=1000, tol=1e-6, fXK0=0.65, fRLX=0.5, freq_margin=0.1)

            V, g, ω = vgvf_plotting.results_to_arrays(results, modes)
            vgvf_plotting.plot_vg(V, g, title="Goland – V–g", outfile="vg.png", annotate=True)

            # stima Vf per il primo crossing (se vuoi riportare la stessa verticale anche nel V–f)
            Vf_global = None
            for j in range(g.shape[1]):
                Vf_j, _ = vgvf_plotting.first_flutter_crossing(V, g[:, j])
                if Vf_j is not None:
                    Vf_global = Vf_j if (Vf_global is None or Vf_j < Vf_global) else Vf_global

            vgvf_plotting.plot_vf(V, ω, title="Goland – V–ω", outfile="vf.png", Vf=Vf_global)

        elif eth_method:
            # ===== ETH-style (RFA → state space → V–g/V–f) =====
            import rfa_flutter
            import DLM, VLM

            # Aerodynamic grid and RFA parameters

            # β(dimensionless decay constants)
            betas = [0.1, 0.3, 0.4, 0.5, 0.6, 2.0, 5.0]     # RFA poles (adjust based on frequency range)
            
            k_min, k_max = 0.1, 5.0    # Reduced frequency range
            nk = 200                     # Number of frequency points

            # Calculate modal aerodynamic forces at each frequency
            print("\nCalculating unsteady aerodynamics...")
            print("dry_vectors shape:", dry_vectors.shape)
            Qk_list = []
            for i, k in enumerate(np.logspace(np.log10(k_min), np.log10(k_max), nk)):
                print(f"Calculating frequency {i+1}/{nk} (k={k:.3f})")
                Qk = rfa_flutter.modal_Qk_from_DLM(k, Z, lambda k: DLM.calc_Qjj(aerogrid, 0.01, k), dry_vectors, total_dof)
                Qk_list.append(Qk)

            # Fit RFA
            print("\nFitting RFA...")
            A0, A1, A2, Alist = rfa_flutter.rfa_fit(Qk_list, np.logspace(np.log10(k_min), np.log10(k_max), nk), betas, ridge=1e-6)

            # Validate RFA fit
            print("\nValidating RFA fit...")
            fig_validation = rfa_flutter.validate_rfa_frequency_response(
                aerogrid, Z, dry_vectors, A0, A1, A2, Alist, betas, total_dof, 
                k_min=k_min, k_max=k_max, n_points=nk
            )
            plt.show()

            # Verify RFA fit
            if hasattr(rfa_flutter, 'fit_diagnostics'):
                print("\nRunning RFA diagnostics...")
                rfa_flutter.fit_diagnostics(A0, A1, A2, Alist, Qk_list, 
                                          np.logspace(np.log10(k_min), np.log10(k_max), nk), 
                                          betas)

            # Perform V-g analysis
            print("\nPerforming V-g analysis...")
            M_hat, K_hat, C_hat = modal_reduction.reduce_matrices(Mff, Cff, Kff, dry_vectors)
            eigvals_by_V, eigvecs_by_V = rfa_flutter.sweep(
                V_list, rho_f, b, M_hat, C_hat, K_hat, A0, A1, A2, Alist, betas,
                n_keep=4
            )

            # Plot results
            if hasattr(rfa_flutter, 'plot_vg_vf'):
                print("\nGenerating V-g/V-f plots...")
                rfa_flutter.plot_vg_vf(V_list, eigvals_by_V, eigvecs_by_V,
                       n_structural_modes=5,
                       title_prefix="Goland Wing - ETH Method")
                
                plt.show()

def play_completion_sound():
    """Play a sound when the script finishes."""
    try:
        import os

        # Try to use the system beep
        os.system('echo -n "\a"')  # ASCII bell character
        # If you have 'beep' installed, you can use: os.system('beep -f 1000 -l 1000')
    except Exception as e:
        print(f"Could not play sound: {e}")

if __name__ == "__main__":
    try:
        main()
    finally:
        play_completion_sound()