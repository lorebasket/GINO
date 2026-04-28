import numpy as np

# ------------------------------------------------------------
# 3D Timoshenko beam (2 nodes × 6 dof = 12 dof)
# DOF order per node: [ux, uy, uz, rx, ry, rz]
# This version:
#   • **Builds bending (flexural) and torsional stiffness using EI_SI and GJ_SI directly.**
#   • Uses a very large axial modulus (E_ax) by default, per your request.
#   • Fixes shear-coupling factors phi_y, phi_z to use EIy/EIz and Gs (shear modulus) consistently.
#   • Removes accidental uses of E1/G13 inside the element (those now come from the caller).
#   • Avoids the dimensional mistake EI*Iz in the bending blocks; now uses EIy for (uz, ry) and EIz for (uy, rz).
# ------------------------------------------------------------

def beam3d_stiffness_local(
    EIy, EIz, GJ, A, L, *,
    ky=0.85, kz=0.85, Asy=None, Asz=None,
    Gs=None, E_ax=None, shear=True, debug=False
):
    """
    3D Timoshenko beam (12x12) — local axes:
      DOF order per node: [ux, uy, uz, rx, ry, rz]
      Element DOFs: [0..5] at i, [6..11] at j

    Planes:
      - x–z plane (uy, rz)  → bending rigidity **EIz**
      - x–y plane (uz, ry)  → bending rigidity **EIy**
    """
    import numpy as np
    K = np.zeros((12, 12), dtype=float)

    # --- axial (make it big if you don't care) ---
    if E_ax is None:
        E_ax = 1e12   # or any large number
    k_ax = E_ax * A / L
    dof = [0, 6]
    K[np.ix_(dof, dof)] += k_ax * np.array([[1, -1], [-1, 1]])

    # --- torsion ---
    k_tor = GJ / L
    dof = [3, 9]
    K[np.ix_(dof, dof)] += k_tor * np.array([[1, -1], [-1, 1]])

    # --- shear coupling (Timoshenko) ---
    if shear:
        if Gs is None:
            raise ValueError("shear=True requires Gs (shear modulus).")

        ky_eff = max(ky, 0.95)
        kz_eff = max(kz, 0.95)
        Asy_nom = Asy if Asy is not None else A
        Asz_nom = Asz if Asz is not None else A

        # cap phis so bending stays EI-dominant
        PHI = 0.25
        Asy_min = 12.0 * EIz / (kz_eff * Gs * PHI * L**2)
        Asz_min = 12.0 * EIy / (ky_eff * Gs * PHI * L**2)
        CAP = 10.0
        Asy_eff = min(max(Asy_nom, Asy_min), CAP * A)
        Asz_eff = min(max(Asz_nom, Asz_min), CAP * A)

        phi_z = 12.0 * EIz / (kz_eff * Gs * Asy_eff * L**2)  # plane x–z
        phi_y = 12.0 * EIy / (ky_eff * Gs * Asz_eff * L**2)  # plane x–y
    else:
        phi_y = phi_z = 0.0

    if debug:
        print(f"[elem φ] L={L:.4e}  φy={phi_y:.3f}  φz={phi_z:.3f}")

    # --- bending: use the *correct* rigidity per plane ---
    # x–z plane (uy, rz) → **EIz**
    c = EIz / ((1.0 + phi_z) * L)
    f = 1.0 + phi_z
    Kvy = c * np.array([
        [ 12*f/L**2,   6*f/L,  -12*f/L**2,   6*f/L],
        [  6*f/L,    4.0+phi_z,  -6*f/L,    2.0-phi_z],
        [-12*f/L**2,  -6*f/L,   12*f/L**2,  -6*f/L],
        [  6*f/L,    2.0-phi_z,  -6*f/L,    4.0+phi_z],
    ])
    dof = [1, 5, 7, 11]   # (uy_i, rz_i, uy_j, rz_j)
    K[np.ix_(dof, dof)] += Kvy

    # x–y plane (uz, ry) → **EIy**
    c = EIy / ((1.0 + phi_y) * L)
    f = 1.0 + phi_y
    Kvz = c * np.array([
        [ 12*f/L**2,  -6*f/L,  -12*f/L**2,  -6*f/L],
        [ -6*f/L,    4.0+phi_y,  6*f/L,     2.0-phi_y],
        [-12*f/L**2,   6*f/L,   12*f/L**2,   6*f/L],
        [ -6*f/L,    2.0-phi_y,  6*f/L,     4.0+phi_y],
    ])
    dof = [2, 4, 8, 10]   # (uz_i, ry_i, uz_j, ry_j)
    K[np.ix_(dof, dof)] += Kvz

    return K


def beam3d_mass_local(rho, A, Iy, Iz, L, r_ea2: float = 0.0):
    """Consistent (lumped-rotary) 12×12 mass matrix for a 3D beam.

    Includes:
      • Translational consistent mass (axial + two bending planes)
      • Rotary inertia about y and z (Rayleigh terms using ρIy, ρIz)
      • Torsional rotary inertia about x using ρ(Iy+Iz) per unit length
    """
    M = np.zeros((12, 12), dtype=float)

    # Axial translations (ux)
    m_ax = rho * A * L / 6.0
    Max = m_ax * np.array([[2, 1],
                           [1, 2]], dtype=float)
    dof = [0, 6]
    M[np.ix_(dof, dof)] += Max

    # Torsional rotation (rx): use polar mass moment per unit length: rho*(Iy+Iz)
    Jp_mass = rho * (Iy + Iz + A * r_ea2)   # per unit length
    m_tor = Jp_mass * L / 6.0
    Mtor = m_tor * np.array([[2, 1],
                             [1, 2]], dtype=float)
    dof = [3, 9]
    M[np.ix_(dof, dof)] += Mtor

    # Bending x–z plane (uy, rz): translational + rotary (about z uses rho*Iz)
    c = rho * A * L / 420.0
    Mvy = c * np.array([
        [156,    22*L,   54,   -13*L],
        [22*L,   4*L**2, 13*L,  -3*L**2],
        [54,     13*L,  156,   -22*L],
        [-13*L, -3*L**2,-22*L,  4*L**2],
    ], dtype=float)
    Mrot_z = rho * Iz * np.array([
        [0,   0,    0,   0],
        [0, 4/3,    0, -1/3],
        [0,   0,    0,   0],
        [0, -1/3,   0,  4/3],
    ], dtype=float)
    dof = [1, 5, 7, 11]
    M[np.ix_(dof, dof)] += (Mvy + Mrot_z)

    # Bending x–y plane (uz, ry): translational + rotary (about y uses rho*Iy)
    Mvz = c * np.array([
        [156,   -22*L,   54,    13*L],
        [-22*L,  4*L**2,-13*L,  3*L**2],
        [54,    -13*L,  156,    22*L],
        [13*L,   3*L**2, 22*L,   4*L**2],
    ], dtype=float)
    Mrot_y = rho * Iy * np.array([
        [0,   0,    0,   0],
        [0, 4/3,    0, -1/3],
        [0,   0,    0,   0],
        [0, -1/3,   0,  4/3],
    ], dtype=float)
    dof = [2, 4, 8, 10]
    M[np.ix_(dof, dof)] += (Mvz + Mrot_y)

    return M


# ----------------------- helpers -----------------------

def _asarray(M):
    return np.asarray(M)


def get_EI_GJ_curves_SI(agard):
    EI_tbl = _asarray(getattr(agard, 'EI_lb_in2'))   # (η, lb·in²) – not used, but kept for completeness
    GJ_tbl = _asarray(getattr(agard, 'GJ_lb_in2'))   # (η, lb·in²) – not used, but kept for completeness
    eta_EI = EI_tbl[:, 0].astype(float) if EI_tbl.size else np.array([])
    eta_GJ = GJ_tbl[:, 0].astype(float) if GJ_tbl.size else np.array([])

    # SI curves already provided (per your requirement):
    EI_SI = np.asarray(agard.EI_Nm2, dtype=float).ravel()
    GJ_SI = np.asarray(agard.GJ_Nm2, dtype=float).ravel()

    # If no (η,·) tables are available, assume uniform distribution over [0,1]
    if eta_EI.size == 0:
        eta_EI = np.linspace(0.0, 1.0, EI_SI.size)
    if eta_GJ.size == 0:
        eta_GJ = np.linspace(0.0, 1.0, GJ_SI.size)

    return (eta_EI, EI_SI), (eta_GJ, GJ_SI)


# ----------------------- assembly driver -----------------------

def build_K_M_for_create_beam_model(
    n_el, agard, G13, rho, Iy_of_eta, Iz_of_eta,
    alpha_EIy, alpha_EIz, alpha_GJ,
    ky=0.85, kz=0.85,
    A=1.0, Asy=None, Asz=None, shear=True, E_ax=None,
    *,
    use_uniform_EG: bool = False,   # NEW: drive EI,GJ from E,G and section inertias
    E_mat: float | None = None,     # NEW: Pa
    G_mat: float | None = None,     # NEW: Pa
    J_of_eta=None,                  # NEW: callable for Saint-Venant J(η) (optional)
    alpha_J: float = 1.0,           # NEW: torsion fit factor
    auto_calibrate_EI_to_curve: bool = True, # NEW: match mean EI of paper
    r_eff_min: float = 6.0,         # NEW: bound strong/weak plane ratio
    r_eff_max: float = 10.0,
    use_Euler_Bernoulli: bool = False,        # NEW
):
    (eta_EI, EI_SI), (eta_GJ, GJ_SI) = get_EI_GJ_curves_SI(agard)

    GJ_interp = None
    if use_uniform_EG:
        GJ_interp = lambda eta: np.interp(eta, eta_GJ, GJ_SI)  # N·m² from paper

    # If using the material path, compute a single scalar to match the paper’s average EI
    if use_uniform_EG:
        if (E_mat is None) or (G_mat is None):
            raise ValueError("use_uniform_EG=True requires E_mat and G_mat (Pa).")
        # sample Iy along the span to get a mean geometric inertia
        eta_samples = np.linspace(0, 1, 200)
        Iy_samples = np.array([Iy_of_eta(e) if callable(Iy_of_eta) else float(Iy_of_eta)
                               for e in eta_samples], dtype=float)
        EI_target_mean = float(np.mean(EI_SI)) if (EI_SI is not None and len(EI_SI)>0) else None
        if auto_calibrate_EI_to_curve and (EI_target_mean is not None):
            Iy_mean = float(np.mean(Iy_samples))
            # scalar that maps E*Iy_mean → mean(EI from paper)
            beta_I = EI_target_mean / (E_mat * Iy_mean)   # typically ≪ 1 for solid sections
        else:
            beta_I = 1.0  # no auto-calibration

    # Geometry along span
    Lspan = float(getattr(agard, 'eta_span', agard.length))  # semispan in meters
    # node coordinates (optional, used to compute actual element lengths)
    pitch = np.deg2rad(getattr(agard, 'pitch', 0.0))
    dx = (Lspan / n_el) * np.cos(pitch)
    dy = (Lspan / n_el) * np.sin(pitch)
    dz = 0.0
    node_xyz = np.c_[np.arange(n_el+1)*dx, np.arange(n_el+1)*dy, np.arange(n_el+1)*dz]
    L_e_geom = np.linalg.norm(node_xyz[1:] - node_xyz[:-1], axis=1)

    K_dict, M_dict = {}, {}

    for e in range(n_el):
        eta_mid = (e + 0.5)/n_el

        # geometric inertias for MASS (always geometric here)
        Iy_geom = Iy_of_eta(eta_mid) if callable(Iy_of_eta) else float(Iy_of_eta)
        Iz_geom = Iz_of_eta(eta_mid) if callable(Iz_of_eta) else float(Iz_of_eta)

        if use_uniform_EG:
            # ---- stiffness inertias (EFFECTIVE) ----
            # downscale the solid polygon inertias to match the paper EI level
            IyK = beta_I * Iy_geom
            # keep anisotropy but bounded so phi doesn't explode
            r_geom = Iz_geom / Iy_geom if Iy_geom > 0 else 1.0
            r_eff  = np.clip(r_geom, r_eff_min, r_eff_max)
            IzK = r_eff * IyK

            # build stiffnesses
            EIy = alpha_EIy * (E_mat * IyK)
            EIz = alpha_EIz * (E_mat * IzK)

            # torsion
            if J_of_eta is not None:
                J = float(J_of_eta(eta_mid))
            else:
                # PERFECT: match paper GJ curve level and shape
                GJ_target = float(GJ_interp(eta_mid))
                J = GJ_target / G_mat     # Saint-Venant J implied by paper curve
            GJ = alpha_GJ * (G_mat * J)
        
        elif use_Euler_Bernoulli:
            # uniform moduli
            E = E_mat
            G = G_mat

            # geometry from agard_area: Iy_of_eta(η) = Iyy, Iz_of_eta(η) = Ixx  (about local y and z)
            Iy = Iy_of_eta(eta_mid)   # weak plane (uz, ry)
            Iz = Iz_of_eta(eta_mid)   # strong plane (uy, rz)
            
            # Saint-Venant J(η)
            J = Iy + Iz

            # Euler–Bernoulli bending rigidities
            EIy = alpha_EIy * E * Iy
            EIz = alpha_EIz * E * Iz
            GJ = alpha_GJ * G * J

            ## Torsion: use paper GJ(η) to back out J(η) with your uniform G
            ## (keeps torsion consistent with the paper while using G_mat)
            #GJ = alpha_GJ * np.interp(eta_mid, eta_GJ, GJ_SI)      # from your get_EI_GJ_curves_SI()
            #J  = GJ / G                                 # equivalent St-Venant J(η)

            # element length & properties
            Le    = float(L_e_geom[e])
            A_e   = A(eta_mid)   if callable(A)   else (A if A is not None else 1.0)
            rho_e = rho(eta_mid) if callable(rho) else rho

            # EB stiffness + mass (no shear)
            Kloc = beam3d_stiffness_local_EB(EIy, EIz, GJ, A_e, Le, E_ax=E)   # axial uses E
            Mloc = beam3d_mass_local_EB(rho_e, A_e, Le, add_rayleigh_rot=True, Iy=Iy, Iz=Iz)

            K_dict[f"e{e}"] = Kloc
            M_dict[f"e{e}"] = Mloc
        
        else:
            # ---- original paper-curve path ----
            EIy = alpha_EIy * np.interp(eta_mid, eta_EI, EI_SI)
            GJ  = alpha_GJ  * np.interp(eta_mid, eta_GJ, GJ_SI)
            r_geom = Iz_geom / Iy_geom if Iy_geom > 0 else 1.0
            r_eff  = np.clip(r_geom, r_eff_min, r_eff_max)
            EIz    = r_eff * EIy

        ## hided for Timoshenko beam model
        ## element length
        #Le = float(L_e_geom[e])
#
        ## areas & density possibly varying
        #A_e   = A(eta_mid)   if callable(A)   else float(A)
        #Asy_e = Asy(eta_mid) if callable(Asy) else (float(Asy) if Asy is not None else A_e)
        #Asz_e = Asz(eta_mid) if callable(Asz) else (float(Asz) if Asz is not None else A_e)
        #rho_e = rho(eta_mid) if callable(rho) else float(rho)
#
        ## === DEBUG: compute phi like the element does, to verify it's small ===
        #phi_y = 12.0 * EIy / (max(ky,0.90) * G13 * (Asz_e if Asz_e is not None else A_e) * Le**2)
        #phi_z = 12.0 * EIz / (max(kz,0.90) * G13 * (Asy_e if Asy_e is not None else A_e) * Le**2)
        #if e in (0, n_el//2, n_el-1):
        #    print(f"[phi] e={e:02d}  phiy={phi_y:.2e}  phiz={phi_z:.2e}  "
        #          f"EIy={EIy:.1f}  EIz={EIz:.1f}  beta_I={beta_I:.3e}")
#
        #Kloc = beam3d_stiffness_local(
        #    EIy=EIy, EIz=EIz, GJ=GJ, A=A_e, L=Le,
        #    ky=ky, kz=kz, Asy=Asy_e, Asz=Asz_e,
        #    Gs=G13, E_ax=E_ax, shear=shear,
        #)
        #Mloc = beam3d_mass_local(rho_e, A_e, Iy_geom, Iz_geom, Le) 
#
        #perm = [0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10]
        #Kloc = Kloc[np.ix_(perm, perm)]
        #Mloc = Mloc[np.ix_(perm, perm)]
#
        #K_dict[f"e{e}"] = Kloc
        #M_dict[f"e{e}"] = Mloc

    return K_dict, M_dict


## lets try Euler Bernoulli method
def beam3d_stiffness_local_EB(EIy, EIz, GJ, A, L, E_ax=None):
    """
    3D Euler–Bernoulli beam (12x12), local DOF per node:
      [ux, uy, uz, rx, ry, rz].
    - x–z plane (uy, rz) uses EIz
    - x–y plane (uz, ry) uses EIy
    - no shear deformation (phi_y = phi_z = 0)
    """
    K = np.zeros((12,12), dtype=float)

    # Axial
    if E_ax is None:
        E_ax = 1e12  # large if you don't care about axial compliance
    k_ax = E_ax * A / L
    dof = [0,6]
    K[np.ix_(dof,dof)] += k_ax * np.array([[ 1,-1],[-1, 1]])

    # Torsion
    k_tor = GJ / L
    dof = [3,9]
    K[np.ix_(dof,dof)] += k_tor * np.array([[ 1,-1],[-1, 1]])

    # Bending x–z plane (uy, rz)  →  **EIz**
    c = EIz / L
    Kvy = c*np.array([
        [ 12/L**2,  6/L,  -12/L**2,  6/L],
        [  6/L,     4.0,   -6/L,     2.0],
        [-12/L**2, -6/L,   12/L**2, -6/L],
        [  6/L,     2.0,   -6/L,     4.0],
    ])
    dof = [1,5,7,11]  # (uy_i, rz_i, uy_j, rz_j)
    K[np.ix_(dof,dof)] += Kvy

    # Bending x–y plane (uz, ry)  →  **EIy**
    c = EIy / L
    Kvz = c*np.array([
        [ 12/L**2, -6/L,  -12/L**2, -6/L],
        [ -6/L,     4.0,    6/L,     2.0],
        [-12/L**2,  6/L,   12/L**2,  6/L],
        [ -6/L,     2.0,    6/L,     4.0],
    ])
    dof = [2,4,8,10]  # (uz_i, ry_i, uz_j, ry_j)
    K[np.ix_(dof,dof)] += Kvz

    return K


def beam3d_mass_local_EB(rho, A, L, add_rayleigh_rot=True, Iy=None, Iz=None):
    """
    Consistent mass for Euler–Bernoulli beam.
    - Translational consistent mass (axial + both bending planes)
    - Optional Rayleigh rotary inertia (needs Iy, Iz) to stabilize higher modes.
    """
    M = np.zeros((12,12), dtype=float)

    # Axial translations (ux)
    m_ax = rho * A * L / 6.0
    dof = [0,6]
    M[np.ix_(dof,dof)] += m_ax * np.array([[2,1],[1,2]])

    # Translational (uy) consistent mass (same as EB bending)
    c = rho * A * L / 420.0
    Mvy = c*np.array([
        [156,  22*L,   54,  -13*L],
        [22*L,  4*L**2, 13*L, -3*L**2],
        [54,   13*L,  156,  -22*L],
        [-13*L,-3*L**2,-22*L,  4*L**2],
    ])
    dof = [1,5,7,11]
    M[np.ix_(dof,dof)] += Mvy

    # Translational (uz) consistent mass
    Mvz = c*np.array([
        [156,  -22*L,   54,   13*L],
        [-22*L, 4*L**2, -13*L, 3*L**2],
        [54,   -13*L,  156,   22*L],
        [13*L,  3*L**2, 22*L,  4*L**2],
    ])
    dof = [2,4,8,10]
    M[np.ix_(dof,dof)] += Mvz

    # Optional rotary inertia (Rayleigh) about y & z to avoid unrealistically high modes
    if add_rayleigh_rot and (Iy is not None) and (Iz is not None):
        Mrot_z = rho * Iz * np.array([[0,0,0,0],[0,4/3,0,-1/3],[0,0,0,0],[0,-1/3,0,4/3]])
        Mrot_y = rho * Iy * np.array([[0,0,0,0],[0,4/3,0,-1/3],[0,0,0,0],[0,-1/3,0,4/3]])
        M[np.ix_([1,5,7,11],[1,5,7,11])] += Mrot_z  # rz in x–z plane
        M[np.ix_([2,4,8,10],[2,4,8,10])] += Mrot_y  # ry in x–y plane

    return M