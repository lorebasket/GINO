    
def build_beam(yaml_path, airfoil_name, n_elements):
    from agard_area import make_A_of_eta, unit_inertias, calibrate_area_to_total_mass
    import agard_theory_data as agard_data
    from fea_utl.agard_KM_matrices import build_K_M_for_create_beam_model

    # Build A(eta) from geometry (solid-filled; set thin_walled=True if you prefer shell)
    A_of_eta, meta = make_A_of_eta(yaml_path, airfoil_name, thin_walled=False)

    # Target mass from NASA paper
    rho = 381.6 # mahogany density [kg/m3]
    M_target = 0.13
    M_target = M_target * 14.59390294 # from slugs to kg

    if M_target is not None:
        A_of_eta, info = calibrate_area_to_total_mass(A_of_eta, agard_data.length, rho, M_target)
        print(f"[mass calib] scale={info['scale']:.3f}, pre-calib mass={info['M_current']:.3f} kg")

    E_mat = 3.2455e9
    G_mat = 0.4119e9

    # Moment of inertia
    Iy_of_eta, Iz_of_eta = unit_inertias(yaml_path, airfoil_name)

    # Correction factors for natural frequencies fitting, 
    # best fit for Euler Bernoulli beam
    alpha_k = 0.8
    alpha_m = 1.15
    alpha_EIy = 0.35
    alpha_EIz = 0.45
    alpha_GJ = 1

    print("alpha_EIy:", alpha_EIy)
    print("alpha_EIz:", alpha_EIz)
    print("alpha_GJ:", alpha_GJ)

    K, M = build_K_M_for_create_beam_model(
        n_elements, agard_data,
        G13=G_mat, rho=rho,
        Iy_of_eta=Iy_of_eta, Iz_of_eta=Iz_of_eta,
        alpha_EIy=alpha_EIy, alpha_EIz=alpha_EIz, alpha_GJ=alpha_GJ,
        ky=0.85, kz=0.85,
        A=A_of_eta,
        Asy=lambda η: 1*A_of_eta(η),
        Asz=lambda η: 1*A_of_eta(η),
        shear=True,
        use_uniform_EG=False,
        E_mat=E_mat, G_mat=G_mat,
        J_of_eta=None,
        auto_calibrate_EI_to_curve=True,
        r_eff_min=6.0, r_eff_max=8.0,
            use_Euler_Bernoulli=True
    )

    beam_length = agard_data.length
    beam_model = create_beam_model(K, M, agard_data.length, n_elements, agard_data.pitch, agard_theory)

    # Check K[0] 2-norms
    Ke0 = next(iter(K.values()))
    print("L    ocal K[0] 2-norms:",
            "uy/rz-block =", np.linalg.norm(Ke0[np.ix_([1,5,7,11],[1,5,7,11])]),
                "uz/ry-block =", np.linalg.norm(Ke0[np.ix_([2,4,8,10],[2,4,8,10])]))

    #K, M = ps.rotate_matrices_by_angle(K, M, 'z', alpha_r)

    print("EI_Nm2 range:", float(np.min(agard_data.EI_Nm2)), float(np.max(agard_data.EI_Nm2)))
    # quick mass per length estimate
    etas = (np.arange(200)+0.5)/200
    mu   = rho*np.mean(A_of_eta(etas))
    print("mean mu [kg/m]:", mu)
    print("beam_length:", agard_data.length, "eta_span:", getattr(agard_data, "eta_span", None))

    return beam_model, K, M, alpha_k, alpha_m, alpha_EIy, alpha_EIz, alpha_GJ