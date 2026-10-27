"""GINO-compatible stiffness / mass / damping contribution plots (no COUPLING deps)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_MPS_TO_KNOTS = 1.94384


def _velocity_for_plot(V_m_s: np.ndarray, *, v_knots: bool = False) -> np.ndarray:
    V_m_s = np.asarray(V_m_s, dtype=float)
    return V_m_s * _MPS_TO_KNOTS if v_knots else V_m_s


def plot_stiffness_damping_contributions(
    contributions_history: list[dict[str, Any]],
    save_path: str | Path,
    *,
    v_knots: bool = False,
    apply_b2_mass: bool = False,
    show_plot: bool = False,
) -> None:
    """Plot M/K/C modal diagonal contributions vs velocity (GINO layout)."""
    if not contributions_history:
        return

    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    contributions = contributions_history
    velocities = [c["V"] for c in contributions]
    default_branch_dof = [0, 1]
    if "branch_dof_indices" in contributions[0]:
        default_branch_dof = list(contributions[0]["branch_dof_indices"])

    def extract_entry(contrib_list, key, branch_slot, default=0.0):
        values = []
        for c in contrib_list:
            if key not in c:
                values.append(float(default))
                continue
            dof_indices = c.get("branch_dof_indices", default_branch_dof)
            dof = int(dof_indices[branch_slot]) if branch_slot < len(dof_indices) else branch_slot
            matrices = c[key]
            mat = matrices[branch_slot] if isinstance(matrices, list) else matrices
            values.append(float(mat[dof, dof]))
        return values

    def extract_coupling_entry(contrib_list, key, branch_i, branch_j, default=0.0):
        values = []
        for c in contrib_list:
            if key not in c:
                values.append(float(default))
                continue
            dof_indices = c.get("branch_dof_indices", default_branch_dof)
            di = int(dof_indices[branch_i]) if branch_i < len(dof_indices) else branch_i
            dj = int(dof_indices[branch_j]) if branch_j < len(dof_indices) else branch_j
            matrices = c[key]
            mat = matrices[0] if isinstance(matrices, list) else matrices
            values.append(float(mat[di, dj]))
        return values

    matrix_components = {
        "M": [
            ("structural", "M_struct", "+"),
            *([("aerodynamic_B2", "M_aero", "-")] if apply_b2_mass else []),
            ("hydro_Capytaine", "M_hydro", "+"),
            ("effective", "M_effective", "="),
        ],
        "K": [
            ("structural", "K_struct", "+"),
            ("aerodynamic_B0", "K_aero", "-"),
            ("hydro", "K_hydro", "+"),
            ("effective", "K_effective", "="),
        ],
        "C": [
            ("structural", "C_struct", "+"),
            ("aerodynamic_B1", "C_aero", "-"),
            ("hydro_Capytaine", "C_hydro", "+"),
            ("empirical", "C_empirical", "+"),
            ("effective", "C_effective", "="),
        ],
    }

    entry_indices = [(0, 0), (1, 1), (0, 1), (1, 0)]
    data: dict[str, dict] = {}
    for family, specs in matrix_components.items():
        data[family] = {}
        for label, key, _sign in specs:
            data[family][label] = {}
            for branch_slot in (0, 1):
                entry = (branch_slot, branch_slot)
                data[family][label][entry] = extract_entry(contributions, key, branch_slot)
            for i, j in ((0, 1), (1, 0)):
                data[family][label][(i, j)] = extract_coupling_entry(contributions, key, i, j)

    def balance_residual(family, entry):
        effective = np.asarray(data[family]["effective"][entry], dtype=float)
        rhs = np.zeros_like(effective)
        for label, _key, sign in matrix_components[family]:
            if sign == "=":
                continue
            arr = np.asarray(data[family][label][entry], dtype=float)
            rhs = rhs - arr if sign == "-" else rhs + arr
        return effective - rhs

    csv_cols = {"Velocity_m_s": velocities}
    for family, specs in matrix_components.items():
        for label, _key, _sign in specs:
            for i, j in entry_indices:
                csv_cols[f"{family}_{label}_{i}{j}"] = data[family][label][(i, j)]
        for i, j in entry_indices:
            csv_cols[f"{family}_balance_residual_{i}{j}"] = balance_residual(family, (i, j))
    if contributions and "omega_rad_s" in contributions[0]:
        csv_cols["omega_rad_s"] = [c.get("omega_rad_s", np.nan) for c in contributions]

    contrib_csv = save_path / "stiffness_damping_contributions.csv"
    pd.DataFrame(csv_cols).to_csv(contrib_csv, index=False, float_format="%.6e")

    V_plot = _velocity_for_plot(np.asarray(velocities, dtype=float), v_knots=v_knots)
    v_xlabel = "Airspeed V [knots]" if v_knots else "Airspeed V [m/s]"

    fig, axes = plt.subplots(1, 3, figsize=(22, 5))
    component_style = {
        "structural": ("#1f77b4", "Structural"),
        "aerodynamic_B2": ("#d62728", "Aero B2 (subtracted)"),
        "aerodynamic_B0": ("#d62728", "Aero B0 (subtracted)"),
        "aerodynamic_B1": ("#d62728", "Aero B1 (subtracted)"),
        "hydro_Capytaine": ("#9467bd", "Hydro/Capytaine (added)"),
        "hydro": ("#9467bd", "Hydro (added)"),
        "empirical": ("#8c564b", "Empirical (added)"),
        "effective": ("#2ca02c", "Effective"),
    }
    mode_style = {0: ("-", "o"), 1: ("--", "s")}

    def _plot_family(ax, family, title, ylabel):
        for label, _key, _sign in matrix_components[family]:
            color, label_text = component_style.get(label, ("#7f7f7f", label))
            for mode in (0, 1):
                entry = (mode, mode)
                if entry not in data[family][label]:
                    continue
                linestyle, marker = mode_style.get(mode, ("-", "o"))
                ax.plot(
                    V_plot, data[family][label][entry],
                    color=color, linestyle=linestyle, marker=marker,
                    linewidth=1.1, markersize=2,
                    label=f"{label_text} {family}[{mode},{mode}]",
                )
        ax.set_xlabel(v_xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    m_title = (
        "Mass contributions: M_eff = M_struct - M_aero + M_hydro"
        if apply_b2_mass
        else "Mass contributions: M_eff = M_struct + M_hydro"
    )
    _plot_family(axes[0], "M", m_title, "Mass diagonal [modal units]")
    _plot_family(axes[1], "K", "Stiffness contributions: K_eff = K_struct - K_aero",
                 "Stiffness diagonal [modal units]")
    _plot_family(axes[2], "C", "Damping contributions: C_eff = C_struct - C_aero + C_hydro + C_emp",
                 "Damping diagonal [modal units]")

    plt.tight_layout()
    outfile = save_path / "stiffness_damping_contributions.png"
    fig.savefig(outfile, dpi=150, bbox_inches="tight")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
