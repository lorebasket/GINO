"""
Lamoureux et al. (Flow, 2024) style post-processing: Ωᵢ and ζᵢ,added vs U_R.

Resting-fluid reference frequencies are taken from the first low-speed flutter
point (default V = 0.01 m/s), stored on ``FlutterResults.resting_ref``.
"""

from __future__ import annotations

from collections import namedtuple
import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

RestingFluidReference = namedtuple(
    "RestingFluidReference",
    ["V_ref", "index", "omega_rad_s", "f_hz", "zeta"],
)


def resting_ref_V_from_config(config) -> float:
    """Target airspeed [m/s] for the pseudo resting-fluid reference point."""
    return float(getattr(config, "resting_ref_V", 0.01))


def compute_resting_fluid_reference(velocities, frequencies, damping, config):
    """
    Reference modal state at the velocity closest to ``config.resting_ref_V``.

    Parameters
    ----------
    velocities, frequencies, damping : array-like
        Flutter sweep arrays (frequencies in rad/s; damping = σ/ω if dimensionless).
    config : AnalysisConfig

    Returns
    -------
    RestingFluidReference
    """
    V = np.asarray(velocities, dtype=float).ravel()
    omega = np.asarray(frequencies, dtype=float)
    g = np.asarray(damping, dtype=float)
    if V.size == 0:
        raise ValueError("Empty velocity sweep — cannot define resting-fluid reference.")

    V_target = resting_ref_V_from_config(config)
    i_ref = int(np.argmin(np.abs(V - V_target)))
    V_at = float(V[i_ref])

    if omega.ndim == 1:
        omega_ref = np.asarray([omega[i_ref]], dtype=float)
        zeta_ref = np.asarray([g[i_ref]], dtype=float)
    else:
        omega_ref = np.asarray(omega[i_ref, :], dtype=float)
        zeta_ref = np.asarray(g[i_ref, :], dtype=float)

    f_hz = omega_ref / (2.0 * np.pi)
    print(
        f"\n[resting ref] V_ref = {V_at:.4g} m/s (target {V_target:.4g}, index {i_ref})"
    )
    for j, (om, f, z) in enumerate(zip(omega_ref, f_hz, zeta_ref)):
        if np.isfinite(om):
            print(f"  mode {j}: f_fr = {f:.4f} Hz, ω_fr = {om:.4f} rad/s, ζ = {z:.6f}")

    return RestingFluidReference(
        V_ref=V_at,
        index=i_ref,
        omega_rad_s=omega_ref,
        f_hz=f_hz,
        zeta=zeta_ref,
    )


def hydrofoil_max_thickness_m(config) -> float:
    """Maximum section thickness h [m] for U_R = U / (f_1,fr * h)."""
    if getattr(config, "max_thickness_m", None) is not None:
        return float(config.max_thickness_m)
    raw = getattr(config, "raw", None)
    chord = float(getattr(config, "chord", np.nan))
    if raw is not None and np.isfinite(chord) and chord > 0:
        return float(np.max(np.abs(np.asarray(raw)[:, 1]))) * chord
    raise ValueError("config.max_thickness_m or config.raw + config.chord required for U_R.")


def _rayleigh_zeta_at_omega(omega_rad, structural_results):
    alpha = getattr(structural_results, "rayleigh_alpha", 0.0)
    beta = getattr(structural_results, "rayleigh_beta", 0.0)
    omega_rad = np.maximum(np.asarray(omega_rad, dtype=float), 1e-12)
    return alpha / (2.0 * omega_rad) + beta * omega_rad / 2.0


def benchmark_series(
    flutter_results,
    structural_results,
    config,
    *,
    mode_index: int = 0,
    resting_ref: Optional[RestingFluidReference] = None,
):
    """
    Build Ω(V), ζ_added(V), U_R(V) for one mode (Lamoureux Fig. 4 convention).

    Returns dict with arrays and scalars for plotting / CSV export.
    """
    resting_ref = resting_ref or getattr(flutter_results, "resting_ref", None)
    if resting_ref is None:
        resting_ref = compute_resting_fluid_reference(
            flutter_results.velocities,
            flutter_results.frequencies,
            flutter_results.damping,
            config,
        )

    V = np.asarray(flutter_results.velocities, dtype=float).ravel()
    omega_v = np.asarray(flutter_results.frequencies, dtype=float)
    g_total = np.asarray(flutter_results.damping, dtype=float)
    if omega_v.ndim == 1:
        omega_v = omega_v[:, np.newaxis]
    if g_total.ndim == 1:
        g_total = g_total[:, np.newaxis]

    j = int(mode_index)
    om = omega_v[:, j]
    g = g_total[:, j]

    omega_fr = np.asarray(resting_ref.omega_rad_s, dtype=float).ravel()
    f_fr = np.asarray(resting_ref.f_hz, dtype=float).ravel()
    if j >= omega_fr.size or not np.isfinite(omega_fr[j]) or omega_fr[j] <= 0:
        raise ValueError(f"Invalid resting reference for mode {j}.")

    omega_1_fr = omega_fr[0]
    f_1_fr = f_fr[0]
    h = hydrofoil_max_thickness_m(config)

    Omega = om / omega_1_fr
    U_R = V / (f_1_fr * h)

    zeta_struct = _rayleigh_zeta_at_omega(om, structural_results)
    zeta_added = g - zeta_struct
    i_ref = int(resting_ref.index)
    zeta_added = zeta_added - zeta_added[i_ref]

    return {
        "V_m_s": V,
        "U_R": U_R,
        "Omega": Omega,
        "zeta_added": zeta_added,
        "zeta_total": g,
        "zeta_struct": zeta_struct,
        "omega_rad_s": om,
        "f_hz": om / (2.0 * np.pi),
        "omega_fr_rad_s": omega_fr,
        "f_fr_hz": f_fr,
        "f_1_fr_hz": f_1_fr,
        "h_m": h,
        "mode_index": j,
        "resting_ref": resting_ref,
    }


def _output_name_with_tag(base: str, output_tag: Optional[str]) -> str:
    if not output_tag:
        return base
    stem, ext = os.path.splitext(base)
    return f"{stem}_{output_tag}{ext}"


def plot_flow_added_damping_benchmark(
    flutter_results,
    structural_results,
    config=None,
    save_path="",
    show_plot=False,
    output_tag: Optional[str] = None,
    mode_index: int = 0,
):
    """
    Fig. 4 (a,b) style: Ω₁ and ζ₁,added vs U_R (Lamoureux et al. 2024 / Cupr 2018).
    """
    if config is None:
        raise ValueError("config is required for flow-added-damping benchmark plots.")

    print("\n--- Flow-added damping benchmark (Ω, ζ_added vs U_R) ---")
    data = benchmark_series(
        flutter_results,
        structural_results,
        config,
        mode_index=mode_index,
    )

    if not save_path and getattr(config, "save_plots", False):
        save_path = getattr(config, "output_dir", ".")
    if save_path:
        os.makedirs(save_path, exist_ok=True)

    import pandas as pd

    csv_name = _output_name_with_tag("flow_added_benchmark_mode0.csv", output_tag)
    if save_path:
        pd.DataFrame(
            {
                "Velocity_m_s": data["V_m_s"],
                "U_R": data["U_R"],
                "Omega": data["Omega"],
                "zeta_added": data["zeta_added"],
                "zeta_total": data["zeta_total"],
                "Frequency_Hz": data["f_hz"],
                "f_1_fr_Hz": data["f_1_fr_hz"],
                "h_m": data["h_m"],
            }
        ).to_csv(os.path.join(save_path, csv_name), index=False, float_format="%.8f")
        print(f"Benchmark series saved to {os.path.join(save_path, csv_name)}")

    fig, (ax_omega, ax_zeta) = plt.subplots(1, 2, figsize=(12, 5))
    case = getattr(config, "name", "case")

    ax_omega.plot(data["U_R"], data["Omega"], "b-o", ms=4, label="GINO")
    ax_omega.axhline(1.0, color="k", ls="--", lw=0.8, alpha=0.5)
    ax_omega.set_xlabel(r"Reduced velocity $U_R = U/(f_{1,fr}\,h)$ [–]")
    ax_omega.set_ylabel(r"$\Omega_1 = f_1/f_{1,fr}$ [–]")
    ax_omega.set_title(f"{case} – dimensionless frequency (mode {mode_index})")
    ax_omega.grid(True, alpha=0.3)
    ax_omega.legend()

    ax_zeta.plot(data["U_R"], data["zeta_added"], "r-o", ms=4, label=r"$\zeta_{1,added}$")
    ax_zeta.axhline(0.0, color="k", ls="--", lw=0.8, alpha=0.5)
    ax_zeta.set_xlabel(r"Reduced velocity $U_R$ [–]")
    ax_zeta.set_ylabel(r"Flow-added damping $\zeta_{1,added}$ [–]")
    ax_zeta.set_title(f"{case} – flow-added damping (mode {mode_index})")
    ax_zeta.grid(True, alpha=0.3)
    ax_zeta.legend()

    ref = data["resting_ref"]
    fig.suptitle(
        f"Resting reference at V = {ref.V_ref:.4g} m/s  |  "
        f"$f_{{1,fr}}$ = {data['f_1_fr_hz']:.2f} Hz  |  h = {data['h_m']*1e3:.2f} mm",
        fontsize=10,
    )
    plt.tight_layout()

    png_name = _output_name_with_tag("flow_added_benchmark_fig4.png", output_tag)
    out_png = os.path.join(save_path, png_name) if save_path else png_name
    if save_path:
        plt.savefig(out_png, dpi=150, bbox_inches="tight")
        print(f"Benchmark figure saved to {out_png}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return data


def print_naca0003_geometry_check(config):
    """Compare NACA0003 config against Lamoureux/Cupr benchmark tables."""
    if getattr(config, "name", "") != "NACA0003":
        return

    L = float(config.beam_length)
    c = float(config.chord)
    h_geom = hydrofoil_max_thickness_m(config)
    h_nom = float(getattr(config, "max_thickness_m", h_geom))

    print("\n" + "=" * 70)
    print("NACA0003 geometry check (Lamoureux 2024 / Cupr 2018)")
    print("=" * 70)
    print(f"  Span L           : {L*1e3:.1f} mm  (paper Table 1: 150 mm)")
    print(f"  Chord c          : {c*1e3:.1f} mm  (paper Table 1: 95 mm)")
    print(f"  h from coords    : {h_geom*1e3:.2f} mm  (max |y| × c)")
    print(f"  h for U_R        : {h_nom*1e3:.2f} mm  (config.max_thickness_m or coords)")
    print(f"  TE               : paper uses 1 mm blunt TE; coords are sharp NACA0003")
    print(f"  BC               : clamped–clamped (root + tip)")
    print(f"  Material         : E={config.E/1e9:.0f} GPa, ρ={config.rho_s} kg/m³")
    print(f"  f₁,vac target    : 321 Hz (EIxx calibrated in structural_model)")
    print("=" * 70 + "\n")
