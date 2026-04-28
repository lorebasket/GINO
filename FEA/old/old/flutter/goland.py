import numpy as np

# ===============================
# Goland benchmark parameters
# (from BYU Aeroelasticity.jl example; use SI units)
# ===============================

# ===============================
# Element stiffness builder
# ===============================
def element_stiffness_EB_w_phiy_plus_torsion(EI: float, GJ: float, L: float,
                                             EA_big: float = None, EI_other_big: float = None) -> np.ndarray:
    """
    12x12 Euler–Bernoulli element stiffness:
    - Active bending in x–z plane via [w, φy] with rigidity EI
    - Active torsion about x via φx with rigidity GJ
    Optionally add large-penalty stiffness for:
      - axial [u] with EA_big
      - the other bending plane [v, φz] with EI_other_big
    DOF order per node: [u, v, w, φx, φy, φz]
    """
    K = np.zeros((12, 12), dtype=float)

    # torsion (φx1, φx2)
    K[np.ix_([3, 9], [3, 9])] += (GJ / L) * np.array([[1, -1],
                                                      [-1, 1]], dtype=float)

    # bending in x–z plane (w, φy) on indices [2,4,8,10]
    L2 = L * L
    L3 = L2 * L
    K_bz = (EI / L3) * np.array([
        [12,    6*L,  -12,   -6*L],
        [6*L,  4*L2,  -6*L,   2*L2],
        [-12,  -6*L,   12,    6*L],
        [-6*L,  2*L2,   6*L,  4*L2]
    ], dtype=float)
    K[np.ix_([2, 4, 8, 10], [2, 4, 8, 10])] += K_bz

    # locked dofs, very big stiffness
    if EA_big is not None and EA_big > 0.0:
        K[np.ix_([0, 6], [0, 6])] += (EA_big / L) * np.array([[1, -1],
                                                              [-1, 1]], dtype=float)

    # locked dofs, very big stiffness
    if EI_other_big is not None and EI_other_big > 0.0:
        K_b_other = (EI_other_big / L3) * np.array([
            [12,    6*L,  -12,   -6*L],
            [6*L,  4*L2,  -6*L,   2*L2],
            [-12,  -6*L,   12,    6*L],
            [-6*L,  2*L2,   6*L,  4*L2]
        ], dtype=float)
        K[np.ix_([1, 5, 7, 11], [1, 5, 7, 11])] += K_b_other

    return K

# ===============================
# Section mass (6x6 per-unit-length) and consistent element mass builder
# ===============================
def section_mass_6x6(mu: float, xm2: float, i11: float, i22: float, i33: float) -> np.ndarray:
    """Return the 6x6 sectional mass/inertia (per unit length) in order [u,v,w, φx, φy, φz]."""
    M_sec = np.array([
        [mu,      0.0,      0.0,      0.0,      0.0,  -mu * xm2],
        [0.0,     mu,       0.0,      0.0,      0.0,       0.0],
        [0.0,     0.0,      mu,    mu * xm2,    0.0,       0.0],
        [0.0,     0.0,   mu * xm2,    i11,      0.0,       0.0],
        [0.0,     0.0,      0.0,      0.0,      i22,       0.0],
        [-mu*xm2, 0.0,      0.0,      0.0,      0.0,       i33]
    ], dtype=float)
    
    return M_sec

def _N_lin(xi: float):
    return 0.5 * (1.0 - xi), 0.5 * (1.0 + xi)

def _H_hermite(xi: float):
    # Hermite polynomials in η in [0,1], with η = (1+xi)/2
    eta = 0.5 * (1.0 + xi)
    H1 = 1.0 - 3.0 * eta**2 + 2.0 * eta**3
    H2 =        eta        - 2.0 * eta**2 + eta**3
    H3 =        3.0 * eta**2 - 2.0 * eta**3
    H4 =      - eta**2 + eta**3
    return H1, H2, H3, H4

def S_matrix(xi: float, L: float) -> np.ndarray:
    """6x12 mapping from nodal rates to section rates in order [udot, vdot, wdot, φxdot, φydot, φzdot]."""
    N1, N2 = _N_lin(xi)
    H1, H2, H3, H4 = _H_hermite(xi)
    S = np.zeros((6, 12), dtype=float)

    # DOF order per node: [u, v, w, φx, φy, φz]
    # u linear
    S[0, 0] = N1;  S[0, 6] = N2
    # v, φz (Hermite for v; rotation linear factor sits with L in displacement mapping)
    S[1, 1]  = H1;  S[1, 5]  = L * H2;  S[1, 7]  = H3;  S[1, 11] = L * H4
    # w, φy
    S[2, 2]  = H1;  S[2, 4]  = L * H2;  S[2, 8]  = H3;  S[2, 10] = L * H4
    # φx linear
    S[3, 3] = N1;   S[3, 9] = N2
    # φy linear
    S[4, 4] = N1;   S[4, 10] = N2
    # φz linear
    S[5, 5] = N1;   S[5, 11] = N2
    return S

def element_mass_from_section(M_sec_6x6: np.ndarray, L: float) -> np.ndarray:
    """2-point Gauss integration of S^T M_sec S over element length -> 12x12 consistent element mass."""
    assert M_sec_6x6.shape == (6, 6)
    xi_g = np.array([-1.0/np.sqrt(3.0),  1.0/np.sqrt(3.0)])
    w_g  = np.array([1.0, 1.0])
    M_e = np.zeros((12, 12), dtype=float)
    for xi, w in zip(xi_g, w_g):
        S = S_matrix(xi, L)
        M_e += (L / 2.0) * w * (S.T @ M_sec_6x6 @ S)
    return M_e

# ===============================
# Beam model builders
# ===============================
def beam_model_goland(beam_length, n_el,
                      EI, GJ_val,
                      mu, xm2,
                      i11, i22, i33,
                      add_penalty_stiffness,
                      EA_big, EI_other_big):
    """
    Build a prismatic Goland-like beam model with n_el elements over beam_length.

    Returns a dict:
        {
          "nodes": [{"position":[0,y,0], "index":i}, ...],
          "elements": [{
                "nodes":[i, i+1],
                "stiffness": (12x12),
                "mass":      (12x12),
                "length":    L_e
          }, ...]
        }

    If add_penalty_stiffness=True, axial and the 'other' bending plane receive
    large penalty stiffnesses (EA_big, EI_other_big) to prevent singular K when
    you don't remove inactive DOFs.
    """
    assert n_el >= 1
    L_e = float(beam_length) / n_el

    K_e = element_stiffness_EB_w_phiy_plus_torsion(
        EI=EI, GJ=GJ_val, L=L_e,
        EA_big=(EA_big if add_penalty_stiffness else None),
        EI_other_big=(EI_other_big if add_penalty_stiffness else None)
    )
    M_sec = section_mass_6x6(mu, xm2, i11, i22, i33)
    M_e = element_mass_from_section(M_sec, L_e)

    nodes = [{"position": [0.0, i * L_e, 0.0], "index": i} for i in range(n_el + 1)]
    elements = [{
        "nodes": [i, i + 1],
        "stiffness": K_e.copy(),
        "mass": M_e.copy(),
        "length": L_e
    } for i in range(n_el)]

    return {"nodes": nodes, "elements": elements}, K_e, M_e, M_sec

# ===============================
# Constraint helper for inactive DOFs
# ===============================
def inactive_dofs_all_nodes(n_nodes: int):
    """Return a sorted list of global DOF indices for the inactive DOFs [u, v, φz] at every node."""
    inactive_local = [0, 1, 5]  # u, v, φz in the per-node ordering
    gl = []
    for n in range(n_nodes):
        base = n * 6
        for d in inactive_local:
            gl.append(base + d)
    return sorted(gl)