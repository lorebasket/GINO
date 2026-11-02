"""
Besch & Liu (1973) Report 4012 — lumped tip-pod hydrodynamics.

The paper represents the pod as a prolate spheroid welded to the strut tip:

  * Structural (air): equivalent mass and MOI added at the tip (already in FEA).
  * Hydrodynamic (fluid at rest): added mass and added moment of inertia of a
    submerged prolate spheroid, lumped at the tip and added to the structural
    matrices (Appendix MODEL 2T; notation I* = added MOI of pod about EA).

Reference frame (beam build): X = chord (+aft), Y = span, Z = thickness.
Pod major axis along X; torsion about Y is the flutter-relevant rotation.
"""
from __future__ import annotations

import numpy as np

INCH = 0.0254
LB_TO_KG = 0.45359237


def pod_geometry_from_config(cfg) -> dict:
    """Read pod geometry / CG offset from Model2t-style config."""
    mass = float(getattr(cfg, "tip_mass", 0.0) or 0.0)
    offset = getattr(cfg, "tip_mass_offset", (0.0, 0.0, 0.0))
    offset = tuple(float(v) for v in offset)

    length = float(getattr(cfg, "capytaine_pod_length", 0.0) or 0.0)
    diam = float(getattr(cfg, "capytaine_pod_diameter", 0.0) or 0.0)
    if length <= 0.0 or diam <= 0.0:
        # Fallback: config_Model2T inline definitions
        length = float(getattr(cfg, "pod_length", 25.7 * INCH))
        diam = float(getattr(cfg, "pod_diameter", 2.0 * INCH))

    a = 0.5 * length  # semi-major (along chord X)
    b = 0.5 * diam    # semi-minor
    return {
        "mass_kg": mass,
        "length_m": length,
        "diameter_m": diam,
        "a_m": a,
        "b_m": b,
        "volume_m3": (4.0 / 3.0) * np.pi * a * b**2,
        "cg_offset_m": offset,  # node (EA at tip) → pod CG
    }


def _lamb_alpha0_beta0(a: float, b: float) -> tuple[float, float]:
    """
    Lamb's inertia coefficients α₀ (axial) and β₀ (transverse) for a prolate
    spheroid (major semi-axis a along x, minor semi-axis b), Hydrodynamics §373.

    Limits: sphere (a=b) → α₀=β₀=2/3; slender needle (a≫b) → α₀→0, β₀→1.
    """
    a = float(a)
    b = float(b)
    if a <= 0.0 or b <= 0.0:
        raise ValueError(f"Invalid spheroid semi-axes: a={a}, b={b}")
    if abs(a - b) / a < 1e-9:
        return 2.0 / 3.0, 2.0 / 3.0
    e = float(np.sqrt(1.0 - (b / a) ** 2))  # eccentricity (a > b)
    L = np.log((1.0 + e) / (1.0 - e))       # = 2·arctanh(e)
    alpha0 = (2.0 * (1.0 - e**2) / e**3) * (0.5 * L - e)
    beta0 = 1.0 / e**2 - ((1.0 - e**2) / (2.0 * e**3)) * L
    return float(alpha0), float(beta0)


def prolate_spheroid_coefficients(a: float, b: float) -> tuple[float, float]:
    """
    Translational added-mass coefficients (k_axial, k_transverse) such that
    ``m_added = k · ρV``.

    k_axial = α₀/(2−α₀), k_transverse = β₀/(2−β₀) (Lamb §373).
    For a≫b: k_axial→0 (slender body slips along its axis), k_transverse→1.
    """
    alpha0, beta0 = _lamb_alpha0_beta0(a, b)
    k_axial = alpha0 / (2.0 - alpha0)
    k_trans = beta0 / (2.0 - beta0)
    return float(k_axial), float(k_trans)


def prolate_spheroid_rot_added_moi_coeff(a: float, b: float) -> float:
    """
    Rotational added-MOI coefficient k_rot for a prolate spheroid rotating about a
    transverse axis through its centre, such that ``I*_transverse = k_rot · ρV``
    (units m², so I* has units kg·m²). Lamb §373 / Korotkin.

    I* = (1/5) ρV (a²−b²)² (β₀−α₀) / [ 2(a²−b²) + (a²+b²)(α₀−β₀) ]
    """
    a = float(a)
    b = float(b)
    alpha0, beta0 = _lamb_alpha0_beta0(a, b)
    num = (a**2 - b**2) ** 2 * (beta0 - alpha0)
    den = 2.0 * (a**2 - b**2) + (a**2 + b**2) * (alpha0 - beta0)
    if abs(den) < 1e-30:
        return 0.0
    return float((1.0 / 5.0) * num / den)


def hydrodynamic_pod_inertia_about_cg(
    rho: float,
    a: float,
    b: float,
    *,
    model: str = "paper_prolate",
) -> np.ndarray:
    """
    Added rotational inertia 3×3 about pod CG [kg·m²].

    model:
      - ``paper_prolate``: Lamb transverse added-MOI k_rot·ρV; axial added MOI ≈ 0
        (a body of revolution has negligible added MOI about its own axis).
      - ``paper_prolate_simple``: I_i = (1/5) ρV(a²+b²) transverse (structural
        spheroid form with ρV) — kept for comparison.
      - ``slender_cylinder``: I_yy = (1/12) m* L², m* = ρπb²L (rotation ⊥ axis).
      - ``korvin_transverse``: I_yy = (4/15) πρ a b² (4a² − b²) (major ⊥ y).
    """
    rho = float(rho)
    a, b = float(a), float(b)
    V = (4.0 / 3.0) * np.pi * a * b**2
    m_star = rho * V

    if model == "paper_prolate":
        i_trans = prolate_spheroid_rot_added_moi_coeff(a, b) * rho * V
        i_axial = 0.0  # added MOI about the axis of revolution ≈ 0
        return np.diag([i_axial, i_trans, i_trans])

    if model == "paper_prolate_simple":
        i_trans = (1.0 / 5.0) * m_star * (a**2 + b**2)
        i_axial = (2.0 / 5.0) * m_star * b**2
        return np.diag([i_axial, i_trans, i_trans])

    if model == "slender_cylinder":
        L = 2.0 * a
        m_cyl = rho * np.pi * b**2 * L
        i_perp = (1.0 / 12.0) * m_cyl * L**2
        i_axial = 0.5 * m_cyl * b**2
        return np.diag([i_axial, i_perp, i_perp])

    if model == "korvin_transverse":
        i_yy = (4.0 / 15.0) * np.pi * rho * a * b**2 * (4.0 * a**2 - b**2)
        i_xx = (8.0 / 15.0) * np.pi * rho * a * b**4
        i_zz = i_yy
        return np.diag([i_xx, i_yy, i_zz])

    raise ValueError(f"Unknown hydrodynamic inertia model: {model!r}")


def assemble_tip_lumped_6x6(
    mass_trans: np.ndarray | float,
    inertia_about_cg: np.ndarray,
    offset: np.ndarray,
) -> np.ndarray:
    """
    6×6 lumped mass/inertia at a beam node (same convention as ``structural_analysis``).

    DOFs: [ux, uy, uz, rx, ry, rz]. ``offset`` is node → CG in the global frame.
    """
    mass_trans = np.asarray(mass_trans, dtype=float)
    if mass_trans.ndim == 0:
        m_diag = np.full(3, float(mass_trans))
    elif mass_trans.size == 3:
        m_diag = mass_trans.reshape(3)
    else:
        m_diag = np.full(3, float(mass_trans.ravel()[0]))

    offset = np.asarray(offset, dtype=float).reshape(3)
    rx, ry, rz = offset
    r_skew = np.array([
        [0.0, -rz, ry],
        [rz, 0.0, -rx],
        [-ry, rx, 0.0],
    ], dtype=float)

    M = np.zeros((6, 6), dtype=float)
    t_sl = slice(0, 3)
    r_sl = slice(3, 6)
    for i in range(3):
        M[i, i] = m_diag[i]
    M_tr = m_diag[0] * r_skew  # use chordwise offset mass for coupling (scalar m)
    if not np.allclose(m_diag, m_diag[0]):
        # General: use average for coupling magnitude
        m_couple = float(np.mean(m_diag))
        M_tr = m_couple * r_skew
    else:
        m_couple = m_diag[0]
    M[t_sl, r_sl] += M_tr
    M[r_sl, t_sl] += M_tr.T
    M[r_sl, r_sl] += np.asarray(inertia_about_cg, dtype=float)
    M[r_sl, r_sl] += m_couple * (
        np.dot(offset, offset) * np.eye(3) - np.outer(offset, offset)
    )
    return M


def physical_pod_hydro_matrix_global(
    cfg,
    rho: float | None = None,
    *,
    inertia_model: str = "paper_prolate",
) -> tuple[np.ndarray, dict]:
    """
    Full-DOF-sized zero matrix with 6×6 pod hydrodynamic lump at the tip node.

    Returns (M_add_global, info dict).
    """
    rho = float(rho if rho is not None else getattr(cfg, "rho_water", 997.0))
    geo = pod_geometry_from_config(cfg)
    a, b = geo["a_m"], geo["b_m"]
    k11, k22 = prolate_spheroid_coefficients(a, b)
    V = geo["volume_m3"]
    m_add = np.array([rho * V * k11, rho * V * k22, rho * V * k22], dtype=float)
    i_cg = hydrodynamic_pod_inertia_about_cg(rho, a, b, model=inertia_model)
    m6 = assemble_tip_lumped_6x6(m_add, i_cg, np.asarray(geo["cg_offset_m"]))

    n_nodes = int(getattr(cfg, "n_elements", 200)) + 1
    total_dof = 6 * n_nodes
    tip_dof = (n_nodes - 1) * 6
    m_global = np.zeros((total_dof, total_dof), dtype=float)
    m_global[tip_dof : tip_dof + 6, tip_dof : tip_dof + 6] = m6

    info = {
        **geo,
        "rho": rho,
        "k11": k11,
        "k22": k22,
        "m_added_xyz_kg": m_add,
        "i_added_cg_diag": np.diag(i_cg),
        "inertia_model": inertia_model,
        "m6_diag": np.diag(m6),
    }
    return m_global, info


def project_pod_hydro_to_modal(
    m_global: np.ndarray,
    dry_vectors,
    mode_indices,
    constrained_dofs,
) -> np.ndarray:
    """Φᵀ M_add Φ on free DOFs."""
    total_dof = m_global.shape[0]
    all_dofs = np.arange(total_dof, dtype=int)
    mask = np.ones(total_dof, dtype=bool)
    mask[np.asarray(constrained_dofs, dtype=int)] = False
    free_dofs = all_dofs[mask]
    m_c = m_global[np.ix_(free_dofs, free_dofs)]
    idx = np.asarray(mode_indices, dtype=int).ravel()
    phi = np.asarray(dry_vectors, dtype=float)[:, idx]
    return phi.T @ m_c @ phi
