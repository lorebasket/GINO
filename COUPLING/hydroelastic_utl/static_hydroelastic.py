"""
Quasi-steady static hydroelastic equilibrium at each flutter sweep speed.

PanelAero DLM ``Qjj`` (``config.qjj_dir``) is unsteady-only (steady part removed
inside PanelAero).  Static loads use precomputed **VLM** ``Q_vlm`` (``vjj_dir``).

Incidence (attack, sweep, dihedral) is in the aerogrid used for ``VLM.calc_Qjj``.
The linear VLM kernel ``Q_vlm @ w`` is therefore a **perturbation** about that
reference; by default ``w_ref = 0`` and ``F0 = 0``.

    w = w_ref + Z_qs @ Phi @ eta
    Delta_Cp = Q_vlm @ w
    F(eta) = q * Phi.T @ Apan @ Delta_Cp

Linearized (Apan = Z_force.T @ diag(A)):

    (K_hat - q * Q_modal_vlm) @ eta = F0
    Q_modal_vlm = Phi.T @ Apan @ Q_vlm @ Z_qs @ Phi
    F0          = q * Phi.T @ Apan @ Q_vlm @ w_ref

Flutter p-k / Roger sweeps continue to use DLM ``Qjj`` only.
"""

from __future__ import annotations

import os
from collections import namedtuple

import numpy as np

from .aerogrid_coupling_matrix import _infer_beam_span_axis

StaticHydroelasticStep = namedtuple(
    "StaticHydroelasticStep",
    [
        "V",
        "eta",
        "tip_w",
        "tip_theta",
        "max_w",
        "max_theta",
        "max_slope_downwash",
        "residual_norm",
        "divergence_margin_min_eig",
        "solve_ok",
    ],
)

StaticHydroelasticResults = namedtuple(
    "StaticHydroelasticResults",
    ["steps", "V", "tip_w", "tip_theta", "max_w", "max_theta", "max_slope_downwash"],
)


def check_vlm_data_available(config) -> str:
    """Return ``config.vjj_dir`` if VLM index exists, else raise."""
    vjj_dir = getattr(config, "vjj_dir", None)
    if not vjj_dir:
        raise FileNotFoundError(
            "static_hydroelastic_pass requires config.vjj_dir (VLM precompute path). "
            "Run FLUID/PanelAero/Qjj/executer.py with VLM=True for this case."
        )
    index_path = os.path.join(vjj_dir, "index.json")
    if not os.path.isfile(index_path):
        raise FileNotFoundError(
            f"VLM data not found at {vjj_dir} (missing index.json). "
            "Precompute with executer.py VLM=True; DLM Qjj alone is not sufficient."
        )
    return vjj_dir


def build_w_ref(aerogrid, config) -> np.ndarray:
    """
    Reference normal downwash at eta = 0 (PanelAero ``w`` in ``Delta_Cp = Q_vlm @ w``).

    Default is zero: attack/sweep/dihedral are already in the VLM aerogrid.

    Optional ``static_hydroelastic_w_ref_perturbation_deg``: uniform incremental
    downwash on all panels (e.g. gust / extra Delta-alpha NOT in the lattice).
    """
    n_p = int(np.asarray(aerogrid["A"], dtype=float).ravel().size)
    deg = float(getattr(config, "static_hydroelastic_w_ref_perturbation_deg", 0.0))
    if abs(deg) < 1e-30:
        return np.zeros(n_p, dtype=float)
    return np.full(n_p, np.sin(np.deg2rad(deg)), dtype=float)


def _load_q_vlm(config, V: float):
    from Qjj.precompute_qjj_vlm import interp_qjj_vlm_from_disk

    Ma = float(V) / float(config.c_sound[config.fluid])
    return np.asarray(
        interp_qjj_vlm_from_disk(config.vjj_dir, Ma),
        dtype=float,
    )


def _panel_area_diag(panel_areas) -> np.ndarray:
    return np.diag(np.asarray(panel_areas, dtype=float).ravel())


def validate_static_coupling_shapes(
    Q_vlm,
    Z_qs,
    Z_force,
    Phi,
    panel_areas,
    w_ref,
    *,
    Apan=None,
):
    """Raise ValueError if VLM / coupling dimensions are inconsistent."""
    n_p = int(np.asarray(panel_areas, dtype=float).ravel().size)
    n_free = int(Z_force.shape[1])
    n_m = int(Phi.shape[1])

    if Q_vlm.shape != (n_p, n_p):
        raise ValueError(f"Q_vlm shape {Q_vlm.shape} != ({n_p}, {n_p})")
    if Z_force.shape != (n_p, n_free):
        raise ValueError(f"Z_force shape {Z_force.shape} != ({n_p}, {n_free})")
    if Z_qs.shape != (n_p, n_free):
        raise ValueError(f"Z_qs shape {Z_qs.shape} != ({n_p}, {n_free})")
    if Phi.shape[0] != n_free:
        raise ValueError(f"Phi rows {Phi.shape[0]} != n_free {n_free}")
    if w_ref.shape != (n_p,):
        raise ValueError(f"w_ref shape {w_ref.shape} != ({n_p},)")
    if Apan is not None and Apan.shape != (n_free, n_p):
        raise ValueError(f"Apan shape {Apan.shape} != ({n_free}, {n_p})")


def compute_q_modal_vlm(Z_qs, Z_force, panel_areas, Phi, Q_vlm, *, Apan=None):
    """Modal VLM coupling K_aero/q = Phi.T @ Apan @ Q_vlm @ Z_qs @ Phi  (real, N_m x N_m)."""
    if Apan is None:
        Apan = Z_force.T @ _panel_area_diag(panel_areas)
    w_slope = Z_qs @ Phi
    delta_cp_slope = Q_vlm @ w_slope
    return Phi.T @ Apan @ delta_cp_slope


def generalized_force_from_w_ref(q, Phi, Q_vlm, w_ref, *, Apan=None, Z_force=None, panel_areas=None):
    """F0 = q * Phi.T @ Apan @ Q_vlm @ w_ref  (N_m,)."""
    delta_cp = Q_vlm @ w_ref
    if Apan is None:
        if Z_force is None or panel_areas is None:
            raise ValueError("Apan or (Z_force, panel_areas) required for force projection")
        Apan = Z_force.T @ _panel_area_diag(panel_areas)
    F_phys = Apan @ delta_cp
    return q * (Phi.T @ F_phys)


def extract_deformation_metrics(q_free, structural_results, Z_qs) -> dict:
    """Tip / max translation and rotation from a free-DOF displacement vector."""
    beam_model = structural_results.beam_model
    total_dof = int(structural_results.total_dof)
    constrained = np.asarray(structural_results.constrained_dofs, dtype=int)
    free_dofs = np.setdiff1d(np.arange(total_dof), constrained)
    free_to_global = {i: int(g) for i, g in enumerate(free_dofs)}

    node_positions = np.array([node["position"] for node in beam_model["nodes"]], dtype=float)
    n_nodes = node_positions.shape[0]
    span_axis = _infer_beam_span_axis(node_positions, beam_model)
    theta_dof = 3 + span_axis

    w_vals = []
    theta_vals = []
    for i_node in range(n_nodes):
        base = i_node * 6
        for comp, store in ((2, w_vals), (theta_dof, theta_vals)):
            g_dof = base + comp
            for i_free, g in free_to_global.items():
                if g == g_dof:
                    val = float(q_free[i_free])
                    store.append(val)
                    break

    w_arr = np.asarray(w_vals, dtype=float)
    th_arr = np.asarray(theta_vals, dtype=float)
    tip_idx = int(np.argmax(node_positions[:, span_axis]))

    w_slope = Z_qs @ q_free
    return {
        "tip_w": float(w_arr[tip_idx]) if w_arr.size else 0.0,
        "tip_theta": float(th_arr[tip_idx]) if th_arr.size else 0.0,
        "max_w": float(np.max(np.abs(w_arr))) if w_arr.size else 0.0,
        "max_theta": float(np.max(np.abs(th_arr))) if th_arr.size else 0.0,
        "max_slope_downwash": float(np.max(np.abs(w_slope))) if w_slope.size else 0.0,
    }


def solve_static_equilibrium_at_V(
    config,
    V: float,
    structural_results,
    coupling_results,
    aerogrid,
) -> StaticHydroelasticStep:
    """
    Linear static equilibrium: (K_hat - q K_aero) eta = F0 with VLM only.

    w_ref defaults to zero (incidence in precomputed aerogrid). Loads arise from
    deformation via Z_qs; F0 is nonzero only with an explicit perturbation downwash.
    """
    check_vlm_data_available(config)

    K_hat = np.asarray(structural_results.K_hat, dtype=float)
    n_modes = K_hat.shape[0]
    Phi = np.asarray(structural_results.dry_eigenvectors[:, :n_modes], dtype=float)

    Z_qs = coupling_results["Z_qs"]
    Z_force = coupling_results.get("Z_force")
    if Z_force is None:
        raise ValueError("static_hydroelastic_pass requires coupling_results['Z_force']")
    Apan = coupling_results.get("Apan")
    panel_areas = np.asarray(aerogrid["A"], dtype=float).ravel()

    w_ref = build_w_ref(aerogrid, config)
    rho = float(config.rho_f[config.fluid])
    q = 0.5 * rho * float(V) ** 2
    Q_vlm = _load_q_vlm(config, V)

    validate_static_coupling_shapes(
        Q_vlm, Z_qs, Z_force, Phi, panel_areas, w_ref, Apan=Apan
    )

    Q_modal_vlm = compute_q_modal_vlm(
        Z_qs, Z_force, panel_areas, Phi, Q_vlm, Apan=Apan
    )
    K_aero = q * np.asarray(Q_modal_vlm, dtype=float)
    K_eff = K_hat - K_aero

    F0 = generalized_force_from_w_ref(
        q, Phi, Q_vlm, w_ref, Apan=Apan, Z_force=Z_force, panel_areas=panel_areas
    )

    try:
        eig_min = float(np.min(np.linalg.eigvalsh(0.5 * (K_eff + K_eff.T))))
    except np.linalg.LinAlgError:
        eig_min = np.nan

    solve_ok = True
    eta = np.zeros(n_modes, dtype=float)
    residual_norm = 0.0
    try:
        eta = np.linalg.solve(K_eff, F0)
        residual_norm = float(np.linalg.norm(K_eff @ eta - F0))
    except np.linalg.LinAlgError:
        solve_ok = False
        residual_norm = np.nan

    q_free = Phi @ eta
    metrics = extract_deformation_metrics(q_free, structural_results, Z_qs)

    return StaticHydroelasticStep(
        V=float(V),
        eta=eta,
        tip_w=metrics["tip_w"],
        tip_theta=metrics["tip_theta"],
        max_w=metrics["max_w"],
        max_theta=metrics["max_theta"],
        max_slope_downwash=metrics["max_slope_downwash"],
        residual_norm=residual_norm,
        divergence_margin_min_eig=eig_min,
        solve_ok=solve_ok,
    )


class StaticHydroelasticSweep:
    """Accumulates static equilibrium results during a flutter velocity sweep."""

    def __init__(self, config, structural_results, coupling_results, aerogrid):
        self.config = config
        self.structural_results = structural_results
        self.coupling_results = coupling_results
        self.aerogrid = aerogrid
        self._steps: list[StaticHydroelasticStep] = []

        vjj_dir = check_vlm_data_available(config)
        w_ref = build_w_ref(aerogrid, config)
        w_pert_deg = float(getattr(config, "static_hydroelastic_w_ref_perturbation_deg", 0.0))
        print("\n" + "=" * 70)
        print("STATIC HYDROELASTIC PASS (parallel to flutter sweep)")
        print("=" * 70)
        print("  Steady: VLM (vjj_dir) — DLM Qjj unsteady-only, not used here")
        print(f"  vjj_dir={vjj_dir}")
        print(
            f"  w_ref: default 0 (incidence in aerogrid); "
            f"perturbation={w_pert_deg:.4g} deg, |w_ref| max={np.max(np.abs(w_ref)):.4e}"
        )

    def step(self, V: float, step_index: int | None = None) -> StaticHydroelasticStep:
        result = solve_static_equilibrium_at_V(
            self.config,
            V,
            self.structural_results,
            self.coupling_results,
            self.aerogrid,
        )
        self._steps.append(result)
        label = f" step {step_index + 1}" if step_index is not None else ""
        print(
            f"  [static{label}] V={result.V:.2f} m/s: "
            f"tip w={result.tip_w:.4e} m, tip θ={np.rad2deg(result.tip_theta):.3f} deg, "
            f"max |w|={result.max_w:.4e} m, K_eff min λ={result.divergence_margin_min_eig:.3e}"
        )
        if not result.solve_ok:
            print(f"    ⚠ static solve failed at V={result.V:.2f} m/s (near divergence?)")
        return result

    def finalize(self) -> StaticHydroelasticResults | None:
        if not self._steps:
            return None
        V = np.array([s.V for s in self._steps], dtype=float)
        return StaticHydroelasticResults(
            steps=list(self._steps),
            V=V,
            tip_w=np.array([s.tip_w for s in self._steps], dtype=float),
            tip_theta=np.array([s.tip_theta for s in self._steps], dtype=float),
            max_w=np.array([s.max_w for s in self._steps], dtype=float),
            max_theta=np.array([s.max_theta for s in self._steps], dtype=float),
            max_slope_downwash=np.array([s.max_slope_downwash for s in self._steps], dtype=float),
        )
