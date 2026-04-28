def load_rfa_matrices_from_disk(config, velocity, space_type):
    """
    Load pre-computed Roger Fit Approximation (RFA) matrices from disk.
    
    Parameters
    ----------
    config : AnalysisConfig
        Configuration object containing case information (config.name, config.paths['output_plots'])
    space_type : str, optional
        Type of space: 'panel' (aerodynamic panel space) or 'rigid_body' (6 DOF rigid body space)
        Default: 'rigid_body'
    
    Returns
    -------
    Q0 : (n, n) ndarray
        Constant term coefficient matrix
    Q1 : (n, n) ndarray
        Linear term coefficient matrix
    Q2 : (n, n) ndarray
        Quadratic term coefficient matrix
    Alag : list of (n, n) ndarray
        Lag contribution matrices
    blag : ndarray
        Lag pole locations
    
    Raises
    ------
    FileNotFoundError
        If RFA matrices are not found on disk
    ValueError
        If metadata or lag poles file is missing or invalid
    """
    import os
    import numpy as np
    import pandas as pd
    
    # Construct path to RFA matrices
    space_label = "panel_space" if space_type == "panel" else "rigid_body_space"
    output_dir = config.paths.get('output_plots', '/media/lorenzo/Seagate Por/recupero dati Linux/lorebasket/FSI/output_plots')
    rfa_dir = os.path.join(output_dir, "RFA_matrices", config.name, f"V_{velocity:.0f}", space_label)
    
    if not os.path.exists(rfa_dir):
        raise FileNotFoundError(
            f"RFA matrices directory not found: {rfa_dir}\n"
            f"Please run RFA computation first with config.roger_fit = True"
        )
    
    print(f"\n{'='*70}")
    print(f"  LOADING PRE-COMPUTED RFA MATRICES ({space_label.upper()})")
    print(f"{'='*70}\n")
    print(f"Loading from directory: {rfa_dir}\n")
    
    # Load Q0 matrix (constant term)
    csv_Q0 = os.path.join(rfa_dir, "Q0_constant_term.csv")
    Q0 = np.array(pd.read_csv(csv_Q0, header=None))
    print(f"✓ Q0 (constant term) loaded - Shape: {Q0.shape}")
    
    # Load Q1 matrix (linear term)
    csv_Q1 = os.path.join(rfa_dir, "Q1_linear_term.csv")
    Q1 = np.array(pd.read_csv(csv_Q1, header=None))
    print(f"✓ Q1 (linear term) loaded - Shape: {Q1.shape}")
    
    # Load Q2 matrix (quadratic term)
    csv_Q2 = os.path.join(rfa_dir, "Q2_quadratic_term.csv")
    Q2 = np.array(pd.read_csv(csv_Q2, header=None))
    print(f"✓ Q2 (quadratic term) loaded - Shape: {Q2.shape}")
    
    # Load lag poles
    csv_blag = os.path.join(rfa_dir, "lag_poles.csv")
    df_blag = pd.read_csv(csv_blag)
    blag = df_blag['pole_value'].values
    print(f"✓ Lag poles loaded - Count: {len(blag)}, Values: {blag}")
    
    # Load Alag matrices
    Alag = []
    for i, pole in enumerate(blag):
        pole = float(pole)  # Ensure pole is a float for formatting
        csv_Alag = os.path.join(rfa_dir, f"Alag_{i}_pole_{pole:.4f}.csv")
        if not os.path.exists(csv_Alag):
            raise FileNotFoundError(f"Lag matrix file not found: {csv_Alag}")
        A = np.array(pd.read_csv(csv_Alag, header=None))
        Alag.append(A)
        print(f"✓ Alag[{i}] (lag pole b={pole:.4f}) loaded - Shape: {A.shape}")
    
    print(f"\n{'='*70}\n")
    print(f"Successfully loaded {len(Alag)} RFA matrices from disk")
    print(f"Total system dimension: {Q0.shape[0]} x {Q0.shape[1]}\n")
    
    return Q0, Q1, Q2, Alag, blag


def RFA(config, V):
    """
    Roger Fit Approximation for PanelAero aerodynamic matrices.
    
    FORMULAZIONE PANELAERO:
    ========================
    La variabile di Laplace adimensionata è: s_bar = s/V (NO semicorda b)
    
    Le matrici Q(ik) da PanelAero si approssimano come:
        Q(ik) ≈ A0 + A1*(s/V) + A2*(s/V)² + Σ_i Alag_i * (ik/(ik+b_i))
    
    Quando moltiplicate per la pressione dinamica q = 0.5*ρ*V²:
        F_aero = q * Q(ik) = 0.5*ρ*V² * [A0 + A1*(s/V) + A2*(s²/V²)]
                = 0.5*ρ*V²*A0 + 0.5*ρ*V*A1*s + 0.5*ρ*A2*s²
    
    Le matrici aeroelastiche finali (in pk_solverv3.py):
        [M] = [M_s] - 0.5*ρ*[A2]              (NO b² come in ETH)
        [C] = [C_s] - 0.5*ρ*V*[A1]            (NO b come in ETH)
        [K] = [K_s] - 0.5*ρ*V²*[A0]           (come ETH)
    
    Output (Q0, Q1, Q2):
    =====================
    Q0, Q1, Q2 sono le matrici NON ADIMENSIONATE in senso fisico.
    Sono già "corrette" per la formulazione PanelAero poiché vengono
    direttamente adimensionate con la pressione dinamica in pk_solverv3.py.
    """
    import numpy as np
    from Qjj.precompute_qjj import interp_qjj_from_disk_old

    k_list = np.array(config.k_list, dtype=float)
    qjj_dir = f"/media/lorenzo/Seagate Por/recupero dati Linux/lorebasket/FSI/Qjj/qjj_precomputed/{config.name}_{config.fluid}_alpha{config.alpha_deg}_nspan{config.nspan}_nchord{config.nchord}_quartic"

    # Load Q matrices from PanelAero (already in PanelAero formulation with s_bar = s/V)
    Q_list = []
    for k in k_list:
        #k_nd = k / (config.chord * 0.5)
        Ma = config.c_sound[config.fluid] / V  # Mach number based on first velocity in Vlist
        Q = interp_qjj_from_disk_old(qjj_dir, k, Ma)
        Q_list.append(Q)

    Q_list = np.array(Q_list)   # (Nk, n, n) - shape: num_k_values, num_panels, num_panels
    Nk, n, _ = Q_list.shape

    # Lag poles (adimensionali, usati nella funzione razionale)
    blag = config.blag

    # Build the REAL basis matrix by splitting Re/Im parts of each basis function at s=ik
    # Each k contributes 2 rows: one for Re(Q), one for Im(Q)
    # Columns: A0, A1, A2, Alag_1, ..., Alag_nlag
    # 
    # NOTA CRITICA - Formulazione PanelAero:
    # =======================================
    # Stiamo approssimando Q(ik) della forma:
    #   Q(ik) = A0 + A1*(ik/V) + A2*(ik/V)² + Σ_i A_lag_i * ik/(ik+b_i)
    #
    # Nel dominio della frequenza ridotta k = ω/V, la variabile adimensionata è s_bar = s/V.
    # Valutando a s=ik (frequenza immaginaria pura):
    #   - Il termine A0*1 contribuisce Re=A0, Im=0
    #   - Il termine A1*ik contribuisce Re=0, Im=k*A1
    #   - Il termine A2*(ik)² = -k²*A2 contribuisce Re=-k²*A2, Im=0
    #   - Il termine A_lag * ik/(ik+b) = A_lag * k²+ikb / (k²+b²) contribuisce
    #     Re = A_lag * k² / (k²+b²), Im = A_lag * kb / (k²+b²)
    
    rows = []
    for k in k_list:
        k2 = k ** 2

        # Real parts of basis functions at s=ik
        # Re(A0*1) + Re(A1*ik) + Re(A2*(ik)²) + Re(Σ A_lag * ik/(ik+b))
        re_row = [1.0,  0.0,  -k2]
        for b in blag:
            re_row.append(k2 / (k2 + b**2))        # Re( ik/(ik+b) ) = k²/(k²+b²)

        # Imaginary parts of basis functions at s=ik
        # Im(A0*1) + Im(A1*ik) + Im(A2*(ik)²) + Im(Σ A_lag * ik/(ik+b))
        im_row = [0.0,  k,    0.0]
        for b in blag:
            im_row.append((k * b) / (k2 + b**2))   # Im( ik/(ik+b) ) = kb/(k²+b²)

        rows.append(re_row)
        rows.append(im_row)

    A_basis = np.array(rows, dtype=float)   # (2*Nk, 3 + n_lag), real

    # Allocate real output matrices
    # =============================
    
    Q0   = np.zeros((n, n))
    Q1   = np.zeros((n, n))
    Q2   = np.zeros((n, n))
    Alag = [np.zeros((n, n)) for _ in range(config.n_lag)]

    # Fit each (i,j) entry with a real least-squares solve
    for i in range(n):
        for j in range(n):
            # Stack Re and Im of Q[:,i,j] in the same order as A_basis rows
            q_ij = Q_list[:, i, j]                  # complex, shape (Nk,)
            rhs = np.empty(2 * Nk)
            rhs[0::2] = q_ij.real                   # Real parts at even indices
            rhs[1::2] = q_ij.imag                   # Imaginary parts at odd indices

            coeffs, *_ = np.linalg.lstsq(A_basis, rhs, rcond=None)

            Q0[i, j]  = coeffs[0]
            Q1[i, j]  = coeffs[1]
            Q2[i, j]  = coeffs[2]
            for l in range(config.n_lag):
                Alag[l][i, j] = coeffs[3 + l]

    print(f"  ✓ RFA fit completed - Q matrices extracted")
    print(f"    Basis matrix shape: {A_basis.shape} (2*Nk rows, 3+n_lag columns)")
    print(f"    Output matrices: Q0, Q1, Q2 shape {Q0.shape}, Alag count: {len(Alag)}")
    
    # Store reference to config for later use in saving
    RFA.config = config
    RFA.Q0 = Q0
    RFA.Q1 = Q1
    RFA.Q2 = Q2
    RFA.Alag = Alag
    RFA.blag = blag
    
    return Q0, Q1, Q2, Alag, blag


def save_rfa_matrices_to_csv(Q0, Q1, Q2, Alag, blag, config, output_dir, velocity, space_type="panel"):
    """
    Save Roger Fit Approximation (RFA) matrices to CSV files.
    
    Parameters
    ----------
    Q0, Q1, Q2 : (n, n) ndarray
        Roger coefficient matrices (constant, linear, quadratic terms)
    Alag : list of (n, n) ndarray
        Lag contribution matrices
    blag : ndarray
        Lag pole locations
    config : AnalysisConfig
        Configuration object containing case information
    output_dir : str
        Output directory path for CSV files
    velocity : float
        Flight velocity for which RFA matrices are computed (m/s)
    space_type : str
        Type of space: 'panel' (aerodynamic panel space) or 'rigid_body' (6 DOF rigid body space)
    
    Returns
    -------
    None
        Files are saved to disk
    """
    import os
    import pandas as pd
    
    # Create output directory structure: output_plots/RFA_matrices/{config.name}/V_{velocity}/{space_type}/
    space_label = "panel_space" if space_type == "panel" else "rigid_body_space"
    rfa_base_dir = os.path.join(output_dir, "RFA_matrices", config.name, f"V_{velocity:.0f}")
    rfa_dir = os.path.join(rfa_base_dir, space_label)
    os.makedirs(rfa_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"  SAVING RFA MATRICES TO CSV ({space_label.upper()})")
    print(f"{'='*70}\n")
    print(f"Output directory: {rfa_dir}\n")
    
    # Save Q0 matrix
    df_Q0 = pd.DataFrame(Q0)
    csv_Q0 = os.path.join(rfa_dir, "Q0_constant_term.csv")
    df_Q0.to_csv(csv_Q0, index=False, header=False)
    print(f"✓ Q0 (constant term) saved to: Q0_constant_term.csv")
    print(f"  Shape: {Q0.shape}")
    
    # Save Q1 matrix
    df_Q1 = pd.DataFrame(Q1)
    csv_Q1 = os.path.join(rfa_dir, "Q1_linear_term.csv")
    df_Q1.to_csv(csv_Q1, index=False, header=False)
    print(f"✓ Q1 (linear term) saved to: Q1_linear_term.csv")
    print(f"  Shape: {Q1.shape}")
    
    # Save Q2 matrix
    df_Q2 = pd.DataFrame(Q2)
    csv_Q2 = os.path.join(rfa_dir, "Q2_quadratic_term.csv")
    df_Q2.to_csv(csv_Q2, index=False, header=False)
    print(f"✓ Q2 (quadratic term) saved to: Q2_quadratic_term.csv")
    print(f"  Shape: {Q2.shape}")
    
    # Save each Alag matrix
    for l, A in enumerate(Alag):
        df_Alag = pd.DataFrame(A)
        csv_Alag = os.path.join(rfa_dir, f"Alag_{l}_pole_{blag[l]:.4f}.csv")
        df_Alag.to_csv(csv_Alag, index=False, header=False)
        print(f"✓ Alag[{l}] (lag pole b={blag[l]:.4f}) saved to: Alag_{l}_pole_{blag[l]:.4f}.csv")
        print(f"  Shape: {A.shape}")
    
    # Save lag poles information
    df_blag = pd.DataFrame({
        'lag_index': range(len(blag)),
        'pole_value': blag
    })
    csv_blag = os.path.join(rfa_dir, "lag_poles.csv")
    df_blag.to_csv(csv_blag, index=False)
    print(f"✓ Lag poles saved to: lag_poles.csv")
    print(f"  Poles: {blag}")
    
    # Save metadata
    metadata = {
        'case_name': [config.name],
        'velocity': [velocity],
        'fluid': [config.fluid],
        'alpha_deg': [config.alpha_deg],
        'nspan': [config.nspan],
        'nchord': [config.nchord],
        'n_lag': [config.n_lag],
        'space_type': [space_type],
        'space_label': [space_label],
        'Q0_shape': [str(Q0.shape)],
        'Q1_shape': [str(Q1.shape)],
        'Q2_shape': [str(Q2.shape)],
        'matrix_size': [Q0.shape[0]],
        'chord': [config.chord],
        'beam_length': [config.beam_length],
        'rho_fluid': [config.rho_f[config.fluid]],
    }
    df_metadata = pd.DataFrame(metadata)
    csv_metadata = os.path.join(rfa_dir, "metadata.csv")
    df_metadata.to_csv(csv_metadata, index=False)
    print(f"✓ Metadata saved to: metadata.csv")
    print(f"  Velocity: {velocity} m/s")
    print(f"  Space type: {space_label}")
    print(f"  Matrix size: {Q0.shape[0]}")
    
    print(f"\n{'='*70}")
    print(f"  RFA MATRICES SAVED SUCCESSFULLY ({space_label.upper()})")
    print(f"{'='*70}\n")


def load_rfa_matrices_from_csv(case_name, velocity, output_dir, space_type="panel"):
    """
    Load Roger Fit Approximation (RFA) matrices from CSV files.
    
    Parameters
    ----------
    case_name : str
        Configuration case name (e.g., 'NACA0003', 'GOLAND')
    output_dir : str
        Path to output directory containing the RFA matrices
    space_type : str
        Type of space to load: 'panel' (aerodynamic panel space) or 'rigid_body' (6 DOF rigid body space)
    
    Returns
    -------
    Q0, Q1, Q2 : (n, n) ndarray
        Roger coefficient matrices
    Alag : list of (n, n) ndarray
        Lag contribution matrices
    blag : ndarray
        Lag pole locations
    metadata : dict
        Dictionary containing configuration metadata
    """
    import os
    import pandas as pd
    import numpy as np
    
    space_label = "panel_space" if space_type == "panel" else "rigid_body_space"
    rfa_dir = os.path.join(output_dir, "RFA_matrices", case_name, f"V_{velocity:.0f}",space_label)
    
    if not os.path.exists(rfa_dir):
        raise FileNotFoundError(f"RFA matrices directory not found: {rfa_dir}")
    
    print(f"\n{'='*70}")
    print(f"  LOADING RFA MATRICES FROM CSV ({space_label.upper()})")
    print(f"{'='*70}\n")
    print(f"Input directory: {rfa_dir}\n")
    
    # Load Q0, Q1, Q2 matrices
    Q0 = pd.read_csv(os.path.join(rfa_dir, "Q0_constant_term.csv"), header=None).values
    Q1 = pd.read_csv(os.path.join(rfa_dir, "Q1_linear_term.csv"), header=None).values
    Q2 = pd.read_csv(os.path.join(rfa_dir, "Q2_quadratic_term.csv"), header=None).values
    
    print(f"✓ Q0 (constant term) loaded: shape {Q0.shape}")
    print(f"✓ Q1 (linear term) loaded: shape {Q1.shape}")
    print(f"✓ Q2 (quadratic term) loaded: shape {Q2.shape}")
    
    # Load lag poles
    df_blag = pd.read_csv(os.path.join(rfa_dir, "lag_poles.csv"))
    blag = df_blag['pole_value'].values
    
    print(f"✓ Lag poles loaded: {blag}")
    
    # Load Alag matrices
    Alag = []
    for l, b in enumerate(blag):
        csv_file = os.path.join(rfa_dir, f"Alag_{l}_pole_{b:.4f}.csv")
        A = pd.read_csv(csv_file, header=None).values
        Alag.append(A)
        print(f"✓ Alag[{l}] (pole b={b:.4f}) loaded: shape {A.shape}")
    
    # Load metadata
    df_metadata = pd.read_csv(os.path.join(rfa_dir, "metadata.csv"))
    metadata = df_metadata.to_dict('records')[0]
    
    print(f"✓ Metadata loaded")
    print(f"  Case: {metadata['case_name']}")
    print(f"  Fluid: {metadata['fluid']}")
    print(f"  Space type: {metadata['space_label']}")
    print(f"  Grid: nspan={metadata['nspan']}, nchord={metadata['nchord']}")
    print(f"  Matrix size: {metadata['matrix_size']}")
    
    print(f"\n{'='*70}\n")
    
    return Q0, Q1, Q2, Alag, blag, metadata


def build_Qroger(k, Q0, Q1, Q2, Alag, blag):
    """
    Reconstruct Q(ik) from Roger coefficients for PanelAero formulation.
    
    FORMULAZIONE PANELAERO (s_bar = s/V):
    =====================================
    Q(ik) = A0 + A1*(ik/V) + A2*(ik/V)² + Σ_i Alag_i * ik/(ik+b_i)
    
    Expanded form:
    Q(ik) = -k²*A2 + 1j*k*A1 + A0 + Σ_i Alag_i * (k²+1j*k*b_i)/(k²+b_i²)
    
    Parameters:
    -----------
    k : float
        Reduced frequency k = ω/V (adimensionale)
    Q0, Q1, Q2 : (n,n) ndarray
        Roger coefficient matrices (constant, linear, quadratic terms)
    Alag : list of (n,n) ndarray
        Lag contribution matrices
    blag : ndarray
        Lag pole locations (adimensionali)
    
    Returns:
    --------
    Q : (n,n) complex ndarray
        Reconstructed aerodynamic matrix Q(ik) ready for use in pk_solverv3.py
    
    Note:
    -----
    La formula è invariante rispetto alla definizione di s_bar.
    Il fit Roger stesso NON dipende da come s_bar è definito;
    ciò che cambia è come Q viene usato nelle matrici aeroelastiche finali.
    """
    k2 = k ** 2
    Q = -k2 * Q2 + 1j * k * Q1 + Q0      # quadratic + linear + constant
    
    # Add lag contributions: each Alag_i contributes A_i * ik/(ik+b_i)
    for A, b in zip(Alag, blag):
        Q += A * (k2 + 1j * k * b) / (k2 + b**2)   # ik/(ik+b) expanded in Re+Im form
    
    return Q


def build_projection_matrix(panel_centers, ref_node=None, normalize=True):
    import numpy as np
    """
    Build the Np x 6 projection matrix Phi for a flat wing aerogrid.

    Parameters
    ----------
    panel_centers : (Np, 3) array  — [x, y, z] of each panel aerodynamic center
    ref_node      : (3,) array    — reference node position; defaults to aerogrid centroid
    normalize     : bool          — normalize each column to unit norm

    Returns
    -------
    Phi : (Np, 6) real array
          Columns ordered as [Tx, Ty, Tz, Rx, Ry, Rz]
    """
    panel_centers = np.asarray(panel_centers, dtype=float)
    Np = len(panel_centers)

    if ref_node is None:
        ref_node = panel_centers.mean(axis=0)

    dx = panel_centers[:, 0] - ref_node[0]   # (Np,)
    dy = panel_centers[:, 1] - ref_node[1]

    # Normal (out-of-plane, z) displacement produced by each rigid DOF
    Phi = np.zeros((Np, 6))
    Phi[:, 0] = 0.0          # Tx  — in-plane, no normal wash
    Phi[:, 1] = 0.0          # Ty  — in-plane, no normal wash
    Phi[:, 2] = 1.0          # Tz  — uniform heave
    Phi[:, 3] = dy           # Rx  — roll:  w =  dy * theta_x
    Phi[:, 4] = -dx          # Ry  — pitch: w = -dx * theta_y
    Phi[:, 5] = 0.0          # Rz  — in-plane yaw, no normal wash

    if normalize:
        norms = np.linalg.norm(Phi, axis=0)
        # Avoid division by zero for the zero columns (Tx, Ty, Rz)
        norms[norms == 0] = 1.0
        Phi /= norms

    return Phi



def main():
    import numpy as np
    import matplotlib.pyplot as plt

    import sys
    import os

    # RFA TEST

    FSI_path = r'/media/lorenzo/Seagate Por/recupero dati Linux/lorebasket/FSI'

    path = os.path.join(FSI_path)
    sys.path.append(path)
    sys.path.extend([
    ])

    import argparse

    parser = argparse.ArgumentParser(
        description="Fit Roger coefficients to Qjj data",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
        run preliminary analysis to calculate separate Qjj into circulatory and noo circulatory terms
        """
        )

    parser.add_argument(
        "case_name",
        nargs="?",
        default="grid_conv",
        help="Analysis case name (e.g. GOLAND, tnz_multibody, grid_conv). Default: GOLAND",
    )

    args = parser.parse_args()
    case_name = args.case_name

    from flutter_analysis_workflow import config, aerodynamic_model
    from Qjj.precompute_qjj import open_qjj_index_old, interp_qjj_from_disk_old
    _analysis_cfg = config.get_config(case_name)


    # Reference node for projection (e.g. arm tip or aerogrid centroid)
    ref_node = np.array([_analysis_cfg.chord * 0.25, _analysis_cfg.beam_length * 0.5, 0.0])

    # Load aerogrid
    aerogrid_path = f"/media/lorenzo/Seagate Por/recupero dati Linux/lorebasket/FSI/Qjj/qjj_precomputed/{case_name}_{_analysis_cfg.fluid}_alpha{_analysis_cfg.alpha_deg}_nspan{_analysis_cfg.nspan}_nchord{_analysis_cfg.nchord}_quartic/aerogrid.npz"
    aero_source = getattr(_analysis_cfg, "aero_source", "panelaero")
    aerogrid, sharpy_data = aerodynamic_model.build(
            aerogrid_path,
            aero_source=aero_source
        )
    
    control_points = aerogrid['offset_j']  # (Np, 3)


    # ========================================================================
    # 1. FIT ROGER COEFFICIENTS FOR EACH VELOCITY
    # ========================================================================
    print(f"\n{'='*70}")
    print(f"  ROGER FIT APPROXIMATION - PRELIMINARY STUDY")
    print(f"  Case: {case_name}")
    print(f"{'='*70}\n")
    Vlist = _analysis_cfg.V_list
    
    for V in Vlist:
        print(f"\n{'='*70}")
        print(f"  Processing velocity V = {V} m/s")
        print(f"{'='*70}")
        
        # ========================================================================
        # Compute RFA coefficients for this velocity
        # ========================================================================
        Q0, Q1, Q2, Alag, blag = RFA(_analysis_cfg, V)
        print(f"✓ Roger coefficients fitted successfully for V = {V} m/s")
        print(f"  - Q2 shape (mass): {Q2.shape}")
        print(f"  - Lag poles (blag): {blag}")
        
        # ========================================================================
        # Save RFA matrices to CSV (PANEL SPACE - NOT PROJECTED)
        # ========================================================================
        print(f"\n{'='*70}")
        print(f"  SAVING RFA MATRICES IN PANEL SPACE (V = {V} m/s)")
        print(f"{'='*70}\n")
        print(f"Q0 (panel space) shape: {Q0.shape}  (panel aerodynamic space)")
        print(f"Q1 (panel space) shape: {Q1.shape}")
        print(f"Q2 (panel space) shape: {Q2.shape}")
        print(f"Alag (panel space) count: {len(Alag)}, each shape: {Alag[0].shape if Alag else 'N/A'}")
        print(f"blag (poles): {blag}\n")
        save_rfa_matrices_to_csv(Q0, Q1, Q2, Alag, blag, _analysis_cfg, FSI_path + '/output_plots', V, space_type="panel")
        
        print(f"\n{'='*70}")
        print(f"  RFA MATRICES SAVED FOR V = {V} m/s (panel space)")
        print(f"{'='*70}\n")
    
    # ========================================================================
    # 2. BUILD PROJECTION MATRIX (Done once after all velocities)
    # ========================================================================
    print(f"\n{'='*70}")
    print(f"  POST-PROCESSING: BUILD PROJECTION MATRIX")
    print(f"{'='*70}\n")
    Phi = build_projection_matrix(control_points, ref_node, normalize=True)
    print(f"✓ Projection matrix built: shape {Phi.shape}")
    print(f"  - Panel centers: {len(control_points)} panels")
    print(f"  - Reference node: {ref_node}")
    
    # ========================================================================
    # 3. BUILD GEOMETRIC BEAM (structural model)
    # ========================================================================
    n_beam_nodes = _analysis_cfg.n_elements  # Discretization along span
    beam_nodes = np.zeros((n_beam_nodes, 3))

    span = _analysis_cfg.beam_length
    chord = _analysis_cfg.chord
    b = chord / 2 
    beam_nodes[:, 1] = np.linspace(b, span, n_beam_nodes)  # y-coordinate (span)
    # beam_nodes[:, 0] = 0.25 * _analysis_cfg.chord  # x at 25% chord (elastic axis)
    
    print(f"\n✓ Beam geometry created: {n_beam_nodes} nodes along span")
    print(f"  - Span: {_analysis_cfg.beam_length} m")
    
    # ========================================================================
    # 4. SUMMARY
    # ========================================================================
    print(f"\n{'='*70}")
    print(f"  RFA MATRICES IN PANEL SPACE (SUMMARY)")
    print(f"{'='*70}\n")
    print(f"✓ RFA coefficient fitting complete for all velocities:")
    for V in Vlist:
        print(f"  - V = {V} m/s (matrices saved to disk)")
    print(f"\n✓ Panel-space matrices will be projected to modal space in flutter_solver.py")
    print(f"  using structural eigenvectors for correct coupling\n")
    



if __name__ == "__main__":
    main()