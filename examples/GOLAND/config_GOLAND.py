import numpy as np
import os


def get_config(AnalysisConfig, FSI_path, QJJ_path):
    attack_angle_deg = 2
    fluid = 'air'

    # Discretization
    n_elements = 250

    # Grid size for PanelAero
    nspan = 30 #16
    nchord = 15 #8

    qjj_dir  = os.path.join(QJJ_path, f"GOLAND_{fluid}_alpha{attack_angle_deg}_nspan{nspan}_nchord{nchord}_quartic")
    vjj_dir  = os.path.join(QJJ_path, f"vjj_GOLAND_{fluid}_alpha{attack_angle_deg}_nspan{nspan}_nchord{nchord}_quartic")
    aerogrid_path = os.path.join(qjj_dir, "aerogrid.npz")

    V_qjj = np.linspace(1, 180, 179)
    k_list = np.linspace(0.001, 150, 200)

    # SHARPy Goland notebook benchmark: velocity_analysis [100, 180, 81];
    # comparison plots use 120–180 m/s.
    V_list = np.linspace(120.0, 180.0, 81)

    beam_length = 6.096
    chord = 1.8288
    xea_factor = 0.33
    xcm_factor = 0.43
    # LE at -xea·c so the shear center (elastic axis) lies at x = 0 (PanelAero convention).
    caero_x_le = -xea_factor * chord

    return AnalysisConfig(
        name='GOLAND',
        blade_name='GOLAND',
        fluid=fluid,
        n_elements=n_elements,

        # ── Flutter solver (RFA + strip is the Abramson benchmark line) ───────
        pk_method=True,
        added_mass_strip_theory=False,

        # ── Geometry ──────────────────────────────────────────────────────
        beam_length=beam_length,
        chord=chord,
        chord_root=chord,
        chord_tip=chord,

        # ── CAERO1 flat panel (executer.py) ───────────────────────────────
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
        chordwise_strip_y=3.048,    # 50% span

        # ── Material & mass ───────────────────────────────────────────────
        rho_s=2700,
        mu=35.71,
        v=0.31,
        i11=8.64,
        i22=0.1 * 8.64,
        i33=0.9 * 8.64,

        # ── Stiffness ─────────────────────────────────────────────────────
        EIxx=9.77221e6,
        GJ=0.987581e6,
        EIzz=1e2 * 9.77221e6,
        EA=1.0e9,
        GAx=1.0e9,
        GAz=1.0e9,

        # ── EA / CG locations (fractions from LE; beam nodes on SC at x = 0) ─
        beam_on_shear_center=True, #aligns beam nodes onto shear center axis
        xcm_factor=xcm_factor,
        xea_factor=xea_factor,

        # ── SHARPy Goland parity (dry modes + flutter reference) ─────────────
        # Section inertias i11/i22/i33 are at EA/SC; CG offset via skew coupling only.
        structural_alpha_rotate=False,   # α only in aero (BeamLoader quaternion in SHARPy)

        # ── Flutter sweep ─────────────────────────────────────────────────
        alpha_deg=attack_angle_deg,
        alpha_r=np.deg2rad(attack_angle_deg),
        #k_list=k_list,
        V_list=V_list,
        num_modes_flutter_egv=4,

        # ── Structural damping (SHARPy Newmark-β parity) ─────────────────────
        # Algorithmic damping enters the PK loop via pk_discrete_newmark + newmark_damp.
        structural_damping_model='none',
        pk_discrete_newmark=True,
        newmark_damp=0.5e-4,
        # Physical UVLM/Newmark step: c/(M·U_ref). SHARPy's 0.1143 is scaled time
        # (Nyquist ~4.4 Hz) and must NOT be used with GINO's physical M,K.
        newmark_dt=1.8288 / (16.0 * 165.0),  # ≈ 6.927e-4 s @ fine grid / V_ref

        # UVLM parity: plunge-only LSQ congruence (w_lsq) on GINO FEA modal basis
        aero_method='UVLM', # DLM or UVLM
        quvlm_npz_path=os.path.join(
            QJJ_path,
            "Quvlm_GOLAND_air_M16N32Ms10_nm4_krylov_wlsq.npz", # Quvlm_GOLAND_air_M16N32Ms10_nm4_krylov_wlsq.npz or Quvlm_GOLAND_air_M16N32Ms10_nm4_krylov_dlm.npz
        ),
        quvlm_sharpy_structural_basis=False,
        # Quvlm already uses PanelAero harmonic convention (conj in precompute);
        # do not apply the DLM-oriented Im(Q) flip again.
        aero_imag_flip=False,

        # ── Rayleigh damping (unused when structural_damping_model='newmark') ─
        rayleigh_target_zetas=(0.005, 0.005),

        # ── Aerodynamic grid ──────────────────────────────────────────────
        nspan=nspan,
        nchord=nchord,
        qjj_dir=qjj_dir,
        vjj_dir=vjj_dir,
        aerogrid_path=aerogrid_path,
        
        # ── Plotting ──────────────────────────────────────────────────────
        plot_aerobeam=False,

    )
