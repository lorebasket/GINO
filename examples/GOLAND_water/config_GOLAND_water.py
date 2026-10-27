"""
Goland wing fully submerged in water — numerical flutter benchmark.

Geometry matches the air Goland planform.  Reference underwater case
(UVLM + Capytaine BEM) uses GJ×8 and a slightly forward EA
(xea=0.28, xcm=0.38) so that PK shows a smooth g=0 crossing near ~49 kn.
Relative submergence = 1 (entire span wetted from root).

Env overrides: GOLAND_WATER_{STIFF,EI,GJ,MASS,SHEAR}_SCALE, GOLAND_WATER_{XEA,XCM},
GOLAND_WATER_{CHORD,BEAM_LENGTH} (added mass vs. geometry sweep, 2026-09-21;
depth stays at 1 chord by construction, so a chord override moves the
absolute submersion depth too and keeps h/c=1 fixed -- see docs/notes/
gino_passo0_gaf_table_stale_axes.md).
"""
import os
import numpy as np



def _omega_grid():
    """Capytaine frequency grid for the BEM radiation matrices.

    Dense (geometric) below 20 rad/s where A(omega) carries the free-surface depth
    dependence, then linear up to a ceiling that must sit ABOVE the highest p-k root
    the case will produce -- the interpolator clamps beyond it.
    """
    w_max = float(os.environ.get("GOLAND_WATER_OMEGA_MAX", "250.0"))
    n_lo = int(os.environ.get("GOLAND_WATER_N_OMEGA_LO", "30"))
    n_hi = int(os.environ.get("GOLAND_WATER_N_OMEGA_HI", "22"))
    w_split = min(20.0, 0.5 * w_max)
    return np.unique(np.concatenate([
        np.geomspace(0.05, w_split, n_lo),
        np.linspace(w_split, w_max, n_hi),
    ]))


def get_config(AnalysisConfig, FSI_path, QJJ_path):
    attack_angle_deg = 2.0
    fluid = "water"

    n_elements = 80
    nspan = 10
    nchord = 5

    # Reuse the existing coarse Goland aerogrid (same planform; UVLM does not need Qjj).
    qjj_dir = os.path.join(
        QJJ_path, f"GOLAND_air_alpha2_nspan{nspan}_nchord{nchord}_quartic"
    )
    vjj_dir = os.path.join(
        QJJ_path, f"vjj_GOLAND_air_alpha2_nspan{nspan}_nchord{nchord}_quartic"
    )
    aerogrid_path = os.path.join(qjj_dir, "aerogrid.npz")

    beam_length = float(os.environ.get("GOLAND_WATER_BEAM_LENGTH", "6.096"))
    chord = float(os.environ.get("GOLAND_WATER_CHORD", "1.8288"))
    # Reference: smooth flutter ~49 kn (see output_data/.../clean_crossing/xea028_gj8).
    xea_factor = float(os.environ.get("GOLAND_WATER_XEA", "0.28"))
    xcm_factor = float(os.environ.get("GOLAND_WATER_XCM", "0.38"))
    caero_x_le = -xea_factor * chord

    # Fully submerged (relative submergence = 1): entire span wetted.
    # Depth below z=0 (free surface) in metres; default 1 chord. Override:
    # GOLAND_WATER_DEPTH or GOLAND_WATER_DEPTH_CHORD.
    submerged_frac = 1.0
    wetted_distance_from_root = (1.0 - submerged_frac) * beam_length
    if os.environ.get("GOLAND_WATER_DEPTH"):
        submersion_depth = float(os.environ["GOLAND_WATER_DEPTH"])
    elif os.environ.get("GOLAND_WATER_DEPTH_CHORD"):
        submersion_depth = float(os.environ["GOLAND_WATER_DEPTH_CHORD"]) * float(chord)
    else:
        submersion_depth = float(chord)

    # Uniform scale of the air-Goland section (keeps EI/GJ and mode shapes).
    stiffness_scale = float(os.environ.get("GOLAND_WATER_STIFF_SCALE", "1.0"))
    ei_scale = float(os.environ.get("GOLAND_WATER_EI_SCALE", "1.0"))

    gj_scale = float(os.environ.get("GOLAND_WATER_GJ_SCALE", "3.0"))
    mass_scale = float(os.environ.get("GOLAND_WATER_MASS_SCALE", "1.0"))

    shear_scale = float(os.environ.get("GOLAND_WATER_SHEAR_SCALE", "1.0"))

    # Material: aluminum by default; GOLAND_WATER_{RHO_S,E_MOD,NU} to override
    # (e.g. steel 7850 / 200e9 / 0.30). Mass AND stiffness are both derived from
    # this one material, so the two can never disagree about what the foil is
    # made of -- the failure mode found on 2026-09-29, when an unscaled mu from
    # the full-size air-Goland wing implied ~43 g/cm^3 at chord=0.10 m.
    rho_s = float(os.environ.get("GOLAND_WATER_RHO_S", "2700"))
    E_mod = float(os.environ.get("GOLAND_WATER_E_MOD", "69e9"))
    nu_mat = float(os.environ.get("GOLAND_WATER_NU", "0.33"))
    G_mod = E_mod / (2.0 * (1.0 + nu_mat))

    # mu* = (4*0.685*tf/pi) * rho_s/rho_w, so thickness_factor is the ONLY lever
    # on the mass ratio besides material density -- chord and span cancel out.
    thickness_factor = float(os.environ.get("GOLAND_WATER_THICKNESS_FACTOR", "0.12"))
    _t = thickness_factor * chord
    _a_semi, _b_semi = chord / 2.0, _t / 2.0
    _airfoil_area = 0.685 * _t * chord                              # NACA00xx enclosed area
    _Ixx_solid = (np.pi / 4.0) * _a_semi * _b_semi**3               # flapwise 2nd moment of area
    _J_solid = np.pi * _a_semi**3 * _b_semi**3 / (_a_semi**2 + _b_semi**2)  # St-Venant torsion const

    mu = rho_s * _airfoil_area * mass_scale
    e_cg = (xcm_factor - xea_factor) * chord
    # Rotational inertia at the SC reference: thin-lamina centroidal value plus the
    # mandatory parallel-axis term, so the translation/rotation mass sub-blocks stay
    # positive definite for any xea/xcm (without mu*e^2 the eigensolver simply fails).
    _i_centroid = mu * chord**2 / 12.0
    i11 = _i_centroid + mu * e_cg**2
    i22 = 0.1 * _i_centroid
    i33 = 0.9 * _i_centroid + mu * e_cg**2
    # Override di i11 per RIPRODURRE run storici: fino al 2026-08 il caso usava
    # j_tors = 8.64 dello stock Goland invece dell'inerzia derivata. Il termine
    # mu*e^2 (aggiunto 2026-09) alza i11 del 29% sul caso del paper e ne sposta
    # la 2a frequenza da 41.7 a 36.3 Hz. Serve solo per bisezionare la
    # riproducibilita'; NON usarlo per casi nuovi.
    _i11_ovr = os.environ.get("GOLAND_WATER_I11")
    if _i11_ovr is not None:
        i11 = float(_i11_ovr)
    _i22_ovr = os.environ.get("GOLAND_WATER_I22")
    if _i22_ovr is not None:
        i22 = float(_i22_ovr)
    _i33_ovr = os.environ.get("GOLAND_WATER_I33")
    if _i33_ovr is not None:
        i33 = float(_i33_ovr)
    _mu_ovr = os.environ.get("GOLAND_WATER_MU")
    if _mu_ovr is not None:
        mu = float(_mu_ovr)

    # Stiffness. Preferred (physical) path: GOLAND_WATER_{EI,GJ}_FRAC give EIxx and GJ
    # as a FRACTION OF A SOLID BILLET of the same material and section. A real foil is
    # skin+spar on a root flexure, so it is always *less* stiff than solid: frac <= 1 is
    # the buildability test. For reference, the ABRAMSON1965 experiment (which really did
    # flutter, at 48 kn) sits at ei_frac=0.19, gj_frac=0.037 against solid aluminum.
    # Legacy path (GOLAND_WATER_{EI,GJ,STIFF}_SCALE, multiples of the air-Goland section)
    # is kept so earlier runs stay reproducible, but it cannot express buildability and
    # will happily hand you ei_frac=3.7 -- stiffer than solid metal at aluminum's weight.
    ei_frac = os.environ.get("GOLAND_WATER_EI_FRAC")
    gj_frac = os.environ.get("GOLAND_WATER_GJ_FRAC")
    if ei_frac is not None or gj_frac is not None:
        EIxx = float(ei_frac if ei_frac is not None else 1.0) * E_mod * _Ixx_solid
        GJ = float(gj_frac if gj_frac is not None else 1.0) * G_mod * _J_solid
        # Shear/axial rigidity is tied to EI, NOT taken from the solid section. Using the
        # true G*A (~1e8 N against EI~1e3) shear-LOCKS these full-integration elements:
        # measured 2026-09-29, the flapwise frequency rose from the correct 5.94 Hz to
        # 15.5 Hz as GA went up, and EI no longer controlled bending at all (the tx
        # stiffness block stopped scaling with EI). GA = 1e3*EI/L^2 sits on the plateau
        # where the FEM matches the Euler-Bernoulli value to ~1% and every frequency
        # scales as sqrt(EI) again; the legacy path's shear_scale=ei_scale convention
        # lands at ~2e2 here, i.e. the same safe band.
        _GA = 1.0e3 * EIxx / beam_length**2 * shear_scale
        _EA = _GA
    else:
        EIxx = 9.77221e6 * stiffness_scale * ei_scale
        GJ = 0.987581e6 * stiffness_scale * gj_scale
        _EA = 1.0e9
        _GA = 1.0e9 * shear_scale

    # Cover the ~49 kn flutter band with margin (49 kn ≈ 25.2 m/s).
    # Starts near-rest (0.5 m/s, not 0.0 to avoid k=omega/V singularities)
    # so the p-k tracker locks onto each branch close to the simpler
    # structural+added-mass regime before aerodynamic circulatory effects
    # grow with V (user request, 2026-09-03).
    V_list = np.linspace(0.5, 40.0, 40)

    capy_dir = os.path.join(FSI_path, "FLUID", "capytaine", "GOLAND_water")
    # Regenerated on the recalibrated xea=0.28/xcm=0.38, GJ×3 dry modes (was
    # reusing the air-Goland xea=0.33/xcm=0.43 GJ×8 table — Passo 0 of the
    # GINO test plan, see docs/notes/gino_passo0_gaf_table_stale_axes.md).
    # 2026-09-04: M=16 -> M=32 chordwise panels. k_max_usable ~= 1.546*M is a
    # hard Nyquist ceiling of the UVLM ROM's dt=2/M (independent of V_ref);
    # M=16 topped out at k~24.7, too narrow for GOLAND_water's self-consistent
    # k (water's added mass drives k=omega/V far higher than air at the same
    # V). M=32 -> k up to 48, no Nyquist clipping (see update 18/19 in the
    # notes doc).
    quvlm_npz = os.environ.get(
        "GOLAND_WATER_QUVLM_NPZ",
        os.path.join(
            FSI_path, "FLUID", "PanelAero", "Qjj",
            "Quvlm_GOLAND_air_M32N32Ms10_nm4_papernominal_gj8_krylov.npz",
        ),
    )

    return AnalysisConfig(
        name="GOLAND_water",
        blade_name="GOLAND",
        fluid=fluid,
        n_elements=n_elements,

        pk_method=True,
        aero_method="UVLM",
        quvlm_npz_path=quvlm_npz,
        quvlm_sharpy_structural_basis=False,
        quvlm_compare_dlm=False,
        aero_imag_flip=False,
        # GOLAND_WATER_B2_REMOVAL=0: tabella UVLM intatta (riferimento solo UVLM puro, 14/10/2026)
        aero_b2_removal=os.environ.get("GOLAND_WATER_B2_REMOVAL", "1") not in ("0", "false", "False"),
        # "full" adds the Capytaine added mass on top of an intact UVLM GAF, which
        # double-counts apparent mass. On ABRAMSON1965 that suppressed the flutter
        # mechanism outright (no crossing below 100 kn against a measured 48.1 kn);
        # "delta" injects A_capy(w) - A_uvlm instead and recovered the right mode and
        # frequency on Model2t. Override with GOLAND_WATER_PK_BEM_MODE=delta.
        pk_bem_mode=os.environ.get("GOLAND_WATER_PK_BEM_MODE", "full"),
        # delta: massa UVLM non circolatoria dal fit della GAF in uso, k in [14, 24] 1/m
        # (k b >= 2 con b = c/2; k dx <= 0.5 con M32) -- 14/10/2026
        pk_bem_delta_fit_k=tuple(float(x) for x in os.environ.get("GOLAND_WATER_DELTA_FIT", "14,24").split(",")),
        added_mass_strip_theory=False,
        added_mass_strip_torsion=False,
        # Diagnostic switch: with pk_bem_mode="full" the Capytaine torsional added mass
        # is stacked on the UVLM's own apparent mass. Measured 2026-09-29 on the
        # Abramson structure: the wet spectrum came back as [0, 0, 57.6, 244.7, 361.7]
        # rad/s -- the ~87 rad/s torsion root is gone entirely (driven to zero), so no
        # bending-torsion coalescence is possible and NO structure can flutter. Set
        # GOLAND_WATER_PK_CAPYTAINE=0 to drop the BEM matrices and isolate this.
        pk_capytaine_matrices=os.environ.get("GOLAND_WATER_PK_CAPYTAINE", "1") not in ("0", "false", "False"),
        capytaine_auto_recalc=True,
        roger_fit=False,
        RFA_PK_method=False,
        empirical_fluid_damping=False,

        beam_length=beam_length,
        chord=chord,
        chord_root=chord,
        chord_tip=chord,
        # accoppiamento flesso-torsionale (composite tailoring / equivalente freccia)
        bend_twist_psi=float(os.environ.get("GOLAND_WATER_PSI", "0.0")),
        thickness_factor=thickness_factor,

        caero_x1=caero_x_le,
        caero_y1=0.0,
        caero_z1=0.0,
        caero_x4=caero_x_le,
        caero_y4=beam_length,
        caero_z4=0.0,
        caero_id=1,
        caero_pid=1,
        caero_igid=1,
        dihedral_angle=0,
        chordwise_strip_y=3.048,

        rho_s=rho_s,
        mu=mu,
        v=0.31,
        i11=i11,
        i22=i22,
        i33=i33,

        EIxx=EIxx,
        GJ=GJ,
        EIzz=1e2 * EIxx,
        EA=_EA,
        GAx=_GA,
        GAz=_GA,

        beam_on_shear_center=True,
        xcm_factor=xcm_factor,
        xea_factor=xea_factor,
        structural_alpha_rotate=False,

        submerged_frac=submerged_frac,
        wetted_distance_from_root=wetted_distance_from_root,
        # TEST G (T1.6): "inf" rimuove la superficie libera (dominio illimitato),
        # per isolare l'effetto superficie da quello di volume.
        free_surface_elevation=os.environ.get("GOLAND_WATER_FREE_SURFACE", 0.0),
        # 21/10/2026: la massa sezionale ha M[w, theta_y] = +mu*e (e a poppa), cioè una torsione positiva
        # alza i punti a poppa; il BEM usava theta x r (li abbassa) e vedeva l'accoppiamento flesso-torsionale
        # rovesciato (A_BEM = strisce con asse elastico specchiato). -1 allinea il BEM a struttura e UVLM.
        bem_torsion_sign=float(os.environ.get("GOLAND_WATER_BEM_TORSION_SIGN", "-1")),
        water_depth=np.inf,
        depth=[submersion_depth],
        depth_index=0,

        alpha_deg=attack_angle_deg,
        alpha_r=np.deg2rad(attack_angle_deg),
        V_list=V_list,
        # Retain 4: the coupled aeroelastic eigenproblem needs enough states to
        # actually contain a torsion-like oscillatory root for MAC-based
        # selection to find -- num_modes_egv=2 collapses the system to at most
        # 2 complex-conjugate pairs total, which starved the search (docs/notes/
        # gino_passo0_gaf_table_stale_axes.md, 2026-09-03 update 11).
        num_modes_egv=int(os.environ.get("GOLAND_WATER_NUM_MODES_EGV", "4")),
        # Tracking only 2 branches assumes the flutter-critical pair is the two lowest
        # DRY modes. In water that can be false: the Capytaine modal added mass is very
        # uneven across modes (measured A00=0.68 but A11=12.8 on the chord=0.3048 foil),
        # so mode 2's wet frequency can fall below mode 1's and the tracked pair ends up
        # being modes 1 and 3 while the genuinely coalescing pair goes unwatched.
        num_modes_flutter_egv=int(os.environ.get("GOLAND_WATER_NUM_MODES_FLUTTER", "2")),
        rayleigh_target_zetas=(0.01, 0.01),
        structural_damping_model="rayleigh",
        pk_tracking_policy="paper",
        mac_matching=True,
        last_converged_mode_matching=True,
        pk_predict_and_select=True,
        pk_min_root_sep_rad=5.0,
        pk_skip_inter_mode_on_perfect_mac=True,
        # Dry modes 0/1/2 are genuinely, strongly hybridised at rest (V=0
        # BEM-only added-mass, no circulatory contribution yet): dry mode 1
        # (torsion, 41.67 Hz dry) has NO wet root anywhere near its dry
        # value -- its best available host is the 8.668 Hz wet root, which
        # itself is only 42% mode-1 (21% mode 0, 37% mode 2). Centering the
        # slot-1 search window / k-guess on the dry 41.67 Hz value (as the
        # default seed does) finds nothing there and falls back to "closest
        # available root", which is the clean/dominant/unrelated dry-mode-3
        # artifact at 53.86 Hz (0% mode-1 participation) -- not a tracker
        # bug, a wrong search centre. Seed on the actual V=0 wet host
        # instead (see scratchpad v0_ref_current.py / docs/notes/
        # gino_passo0_gaf_table_stale_axes.md); index 0 unaffected (its own
        # host, 1.091 Hz, already dominates cleanly at 99.7%).
        pk_track_seed_omegas=[2 * 3.14159265358979 * 1.091, 2 * 3.14159265358979 * 8.668],

        nspan=nspan,
        nchord=nchord,
        qjj_dir=qjj_dir,
        vjj_dir=vjj_dir,
        aerogrid_path=aerogrid_path,

        prefix="GOLAND_water_dry_egv",
        capytaine_phi_normalization="geometric",
        capytaine_results_dir=os.path.join(capy_dir, "results_modal_radiation"),
        mesh_path=os.path.join(capy_dir, "GOLAND_water_mesh.vtu"),
        # densita' mesh BEM (convergenza superficie libera, 14/10/2026)
        mesh_n_span=int(os.environ.get("GOLAND_WATER_MESH_NS", "12")),
        mesh_n_chord=int(os.environ.get("GOLAND_WATER_MESH_NC", "16")),
        # BEM frequency grid. hydro_system._interp_modal_matrix CLAMPS outside this
        # range, so any p-k root above omega_max sees a frozen A(omega) and a kink in
        # dA/domega at the boundary -- that kink is what makes the k-relaxation limit
        # cycle and drops Converged_Branch_N to 15-50% on soft, light wings whose
        # roots reach 16-20 Hz. The old ceiling was 80 rad/s = 12.7 Hz, far too low.
        # Geometric spacing at the low end also resolves the 0.5-3 Hz band where the
        # free-surface (depth) dependence of A actually lives; the old grid put only
        # 8 points below 8 rad/s and then jumped to 8 rad/s spacing.
        omega_list=_omega_grid(),

        v_knots=True,
        vf_hertz=True,
        plot_vg_vf=True,
        plot_log_decrement_vg=True,
        show_eigen_plot=False,
        plot_pk_wet_eigenvectors=False,
        plot_DLM_participants=False,
        plot_stiffness_damping_contributions=False,
        plot_aerobeam=False,

        # NACA 0012-class section (t/c = 0.12), same table as Abramson.
        raw=np.array([
            [1.0000,     0.00120],
            [0.9500,     0.01415],
            [0.9000,     0.02517],
            [0.8000,     0.04199],
            [0.7000,     0.05269],
            [0.6000,     0.05835],
            [0.5000,     0.06000],
            [0.4000,     0.05855],
            [0.3000,     0.05417],
            [0.2000,     0.04664],
            [0.1500,     0.04135],
            [0.1000,     0.03457],
            [0.0750,     0.03032],
            [0.0500,     0.02509],
            [0.0250,     0.01805],
            [0.0125,     0.01292],
            [0.0000,     0.00000],
            [0.0125,     -0.01292],
            [0.0250,     -0.01805],
            [0.0500,     -0.02509],
            [0.0750,     -0.03032],
            [0.1000,     -0.03457],
            [0.1500,     -0.04135],
            [0.2000,     -0.04664],
            [0.3000,     -0.05417],
            [0.4000,     -0.05855],
            [0.5000,     -0.06000],
            [0.6000,     -0.05835],
            [0.7000,     -0.05269],
            [0.8000,     -0.04199],
            [0.9000,     -0.02517],
            [0.9500,     -0.01415],
            [1.0000,     -0.00120],
        ]),
    )
