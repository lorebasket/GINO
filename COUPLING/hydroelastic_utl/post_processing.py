# FSI/COUPLING/hydroelastic_utl/post_processing.py

from . import vgvf_plotting
from panelaero_utl.plotting_git import DetailedPlots
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import os
from typing import Optional


def plot_combined_vgvf_params(results_dict, config, param_name, param_label, param_unit):
    """
    Plot combined VG-VF diagrams for multiple parameter values (generic version).
    
    Args:
        results_dict: Dictionary with parameter value as key and (flutter_results, config) as value
        config: Base configuration (optional, for output directory)
        param_name: Name of the parameter being swept (e.g., 'GJ', 'alpha_deg')
        param_label: Label for display (e.g., 'GJ', 'α')
        param_unit: Unit string (e.g., 'N⋅m²', '°')
    """
    print(f"\n--- Generating Combined V-g and V-f plots for {param_name} sweep ---")
    
    # Determine output directory
    if config and hasattr(config, 'save_plots') and config.save_plots:
        output_dir = config.output_dir
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = "output_data"
        os.makedirs(output_dir, exist_ok=True)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    
    # Define colors for different parameter values
    param_values = sorted(results_dict.keys(), key=lambda x: (x if not isinstance(x, tuple) else x[0]))
    colors = plt.cm.viridis(np.linspace(0, 1, len(param_values)))
    
    flutter_summary = []
    
    for i, param_value in enumerate(param_values):
        flutter_results, _ = results_dict[param_value]
        
        if flutter_results is None:
            continue
        
        V = flutter_results.velocities
        g = flutter_results.damping
        omega = flutter_results.frequencies
        
        color = colors[i]
        
        # Format label based on parameter type
        if isinstance(param_value, tuple):
            label = f"{param_label} = {param_value}"
        elif isinstance(param_value, float):
            if abs(param_value) >= 1e6:
                label = f"{param_label} = {param_value/1e6:.2f}e6 {param_unit}"
            elif abs(param_value) >= 1e3:
                label = f"{param_label} = {param_value/1e3:.2f}e3 {param_unit}"
            else:
                label = f"{param_label} = {param_value:.4g} {param_unit}"
        else:
            label = f"{param_label} = {param_value} {param_unit}"
        
        # Plot V-g (damping) for each branch
        for j in range(g.shape[1]):
            linestyle = '-' if j == 0 else '--'
            branch_label = f"{label} (Branch {j+1})" if g.shape[1] > 1 else label
            ax1.plot(V, g[:, j], label=branch_label, color=color, 
                    linestyle=linestyle, linewidth=2)
            
            # Find flutter speed for this branch
            Vf_j, idx = vgvf_plotting.first_flutter_crossing(V, g[:, j])
            if Vf_j is not None:
                flutter_summary.append({
                    'parameter': param_name,
                    'param_value': param_value,
                    'branch': j+1,
                    'flutter_speed': Vf_j,
                    'flutter_frequency': omega[idx, j] if idx is not None else np.nan
                })
                # Mark flutter point
                ax1.plot(Vf_j, 0, 'o', color=color, markersize=8)
        
        # Plot V-f (frequency) for each branch
        for j in range(omega.shape[1]):
            linestyle = '-' if j == 0 else '--'
            branch_label = f"{label} (Branch {j+1})" if omega.shape[1] > 1 else label
            ax2.plot(V, omega[:, j], label=branch_label, color=color,
                    linestyle=linestyle, linewidth=2)
    
    # V-g plot styling
    ax1.axhline(0.0, linestyle="--", linewidth=1, color='black', alpha=0.5)
    ax1.set_xlabel("Airspeed V [m/s]", fontsize=12)
    ax1.set_ylabel("Damping g = σ/ω [–]", fontsize=12)
    case_name = config.name if config else "Flutter Analysis"
    ax1.set_title(f"{case_name} – V-g Diagram ({param_name} sweep)", fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10, loc='best')
    ax1.grid(True, alpha=0.3)
    
    # V-f plot styling
    ax2.set_xlabel("Airspeed V [m/s]", fontsize=12)
    ax2.set_ylabel("Frequency ω [rad/s]", fontsize=12)
    ax2.set_title(f"{case_name} – V-f Diagram ({param_name} sweep)", fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10, loc='best')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Save figure
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    combined_plot = os.path.join(output_dir, f'combined_vgvf_{param_name}_{timestamp}.png')
    plt.savefig(combined_plot, dpi=200, bbox_inches='tight')
    print(f"✓ Combined plot saved to: {combined_plot}")
    
    # Print flutter summary
    if flutter_summary:
        print(f"\n{'='*70}")
        print(f"Flutter Summary - {param_name} Sweep")
        print(f"{'='*70}")
        print(f"{'Param Value':>20} {'Branch':>8} {'V_flutter [m/s]':>16} {'ω_flutter [rad/s]':>18}")
        print("-" * 70)
        for entry in flutter_summary:
            param_str = f"{entry['param_value']}"
            print(f"{param_str:>20} {entry['branch']:>8} "
                  f"{entry['flutter_speed']:>16.2f} {entry['flutter_frequency']:>18.2f}")
        print("=" * 70)
    
    # Save flutter summary to CSV
    if flutter_summary:
        import pandas as pd
        df = pd.DataFrame(flutter_summary)
        summary_csv = os.path.join(output_dir, f'flutter_summary_{param_name}_{timestamp}.csv')
        df.to_csv(summary_csv, index=False, float_format='%.6f')
        print(f"✓ Flutter summary saved to: {summary_csv}")
    
    plt.show()


def plot_combined_vgvf_alpha(results_dict, config=None):
    """
    Plot combined VG-VF diagrams for multiple angles of attack.
    
    Args:
        results_dict: Dictionary with alpha as key and (flutter_results, config) as value
        config: Base configuration (optional, for output directory)
    """
    print("\n--- Generating Combined V-g and V-f plots for multiple α ---")
    
    # Determine output directory
    if config and hasattr(config, 'save_plots') and config.save_plots:
        output_dir = config.output_dir
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = "output_data"
        os.makedirs(output_dir, exist_ok=True)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    
    # Define colors for different alphas
    alphas = sorted(results_dict.keys())
    colors = plt.cm.viridis(np.linspace(0, 1, len(alphas)))
    
    flutter_summary = []
    
    for i, alpha_deg in enumerate(alphas):
        flutter_results, _ = results_dict[alpha_deg]
        
        if flutter_results is None:
            continue
        
        V = flutter_results.velocities
        g = flutter_results.damping
        omega = flutter_results.frequencies
        
        color = colors[i]
        label = f"α = {alpha_deg}°"
        
        # Plot V-g (damping) for each branch
        for j in range(g.shape[1]):
            linestyle = '-' if j == 0 else '--'
            branch_label = f"{label} (Branch {j+1})" if g.shape[1] > 1 else label
            ax1.plot(V, g[:, j], label=branch_label, color=color, 
                    linestyle=linestyle, linewidth=2)
            
            # Find flutter speed for this branch
            Vf_j, idx = vgvf_plotting.first_flutter_crossing(V, g[:, j])
            if Vf_j is not None:
                flutter_summary.append({
                    'alpha_deg': alpha_deg,
                    'branch': j+1,
                    'flutter_speed': Vf_j,
                    'flutter_frequency': omega[idx, j] if idx is not None else np.nan
                })
                # Mark flutter point
                ax1.plot(Vf_j, 0, 'o', color=color, markersize=8)
        
        # Plot V-f (frequency) for each branch
        for j in range(omega.shape[1]):
            linestyle = '-' if j == 0 else '--'
            branch_label = f"{label} (Branch {j+1})" if omega.shape[1] > 1 else label
            ax2.plot(V, omega[:, j], label=branch_label, color=color,
                    linestyle=linestyle, linewidth=2)
    
    # V-g plot styling
    ax1.axhline(0.0, linestyle="--", linewidth=1, color='black', alpha=0.5)
    ax1.set_xlabel("Airspeed V [m/s]", fontsize=12)
    ax1.set_ylabel("Damping g = σ/ω [–]", fontsize=12)
    case_name = config.name if config else "Flutter Analysis"
    ax1.set_title(f"{case_name} – V-g Diagram (Multiple α)", fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10, loc='best')
    ax1.grid(True, alpha=0.3)
    
    # V-f plot styling
    ax2.set_xlabel("Airspeed V [m/s]", fontsize=12)
    ax2.set_ylabel("Frequency ω [rad/s]", fontsize=12)
    ax2.set_title(f"{case_name} – V-f Diagram (Multiple α)", fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10, loc='best')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    combined_plot = os.path.join(output_dir, f'combined_vgvf_alpha_{timestamp}.png')
    plt.savefig(combined_plot, dpi=200, bbox_inches='tight')
    print(f"✓ Combined plot saved to: {combined_plot}")
    
    # Print flutter summary
    if flutter_summary:
        print(f"\n{'='*70}")
        print("Flutter Summary - Multiple Angles of Attack")
        print(f"{'='*70}")
        print(f"{'Alpha [°]':>10} {'Branch':>8} {'V_flutter [m/s]':>16} {'ω_flutter [rad/s]':>18}")
        print("-" * 70)
        for entry in flutter_summary:
            print(f"{entry['alpha_deg']:>10.2f} {entry['branch']:>8} "
                  f"{entry['flutter_speed']:>16.2f} {entry['flutter_frequency']:>18.2f}")
        print("=" * 70)
    
    # Save flutter summary to CSV
    if flutter_summary:
        import pandas as pd
        df = pd.DataFrame(flutter_summary)
        summary_csv = os.path.join(output_dir, f'flutter_summary_alpha_{timestamp}.csv')
        df.to_csv(summary_csv, index=False, float_format='%.6f')
        print(f"✓ Flutter summary saved to: {summary_csv}")
    
    plt.show()


def _output_name_with_tag(basename: str, output_tag: Optional[str]) -> str:
    """Insert ``_tag`` before the extension, e.g. ``vg_data_nlag4.csv``."""
    if not output_tag:
        return basename
    stem, ext = os.path.splitext(basename)
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(output_tag))
    return f"{stem}_{safe}{ext}"


_MPS_TO_KNOTS = 1.94384


def _vgvf_plot_axes(config):
    """Return (v_knots, vf_hertz, v_label, omega_label, v_unit_short)."""
    v_knots = bool(getattr(config, "v_knots", False))
    vf_hertz = bool(getattr(config, "vf_hertz", False))
    v_label = "Airspeed V [knots]" if v_knots else "Airspeed V [m/s]"
    omega_label = "Frequency ω [Hz]" if vf_hertz else "Frequency ω [rad/s]"
    v_unit_short = "kn" if v_knots else "m/s"
    return v_knots, vf_hertz, v_label, omega_label, v_unit_short


def _vgvf_velocity_for_plot(V_m_s, *, v_knots=False):
    V_m_s = np.asarray(V_m_s, dtype=float)
    return V_m_s * _MPS_TO_KNOTS if v_knots else V_m_s


def _vgvf_frequency_for_plot(omega_rad_s, *, vf_hertz=False):
    omega_rad_s = np.asarray(omega_rad_s, dtype=float)
    return omega_rad_s / (2.0 * np.pi) if vf_hertz else omega_rad_s


def _vgvf_scalar_velocity_for_plot(V_m_s, *, v_knots=False):
    if V_m_s is None or not np.isfinite(V_m_s):
        return None
    return float(V_m_s) * _MPS_TO_KNOTS if v_knots else float(V_m_s)


_KNOTS_TO_MPS = 0.514444


def _vgvf_plot_V_min_m_s(config):
    """Minimum airspeed [m/s] for stabilized-band V-g/V-f post-processing plots."""
    if config is None:
        return None
    v_ms = getattr(config, "vgvf_plot_V_min_m_s", None)
    if v_ms is not None:
        return float(v_ms)
    v_kn = getattr(config, "vgvf_plot_V_min_kn", None)
    if v_kn is not None:
        return float(v_kn) * _KNOTS_TO_MPS
    return None


def _vgvf_stabilized_mask(V_m_s, config):
    v_min = _vgvf_plot_V_min_m_s(config)
    if v_min is None:
        return None
    V_m_s = np.asarray(V_m_s, dtype=float)
    return V_m_s >= float(v_min) - 1e-9


def plot_vg_lambda_vf_combined(
    flutter_results,
    config=None,
    save_path="",
    show_plot=False,
    output_tag: Optional[str] = None,
):
    """
    Abramson-style figure: V–δ, V–g, and V–f in one row.

    Left panel plots δ [s⁻¹] = Λ ω/(2π) with the same sign as Λ and V–g
    (see ``abramson_decay_factor_from_storage``). V–g and V–f unchanged.
    Works for any solver that fills ``FlutterResults`` (PK, Roger RFA, RFA+PK).
    """
    if config is None or not getattr(config, "plot_log_decrement_vg", False):
        return

    print("\n--- Generating V–δ / V–g / V–f combined plot ---")

    V_m_s = np.asarray(flutter_results.velocities, dtype=float)
    g = np.asarray(flutter_results.damping, dtype=float)
    omega_rad_s = np.asarray(flutter_results.frequencies, dtype=float)
    Vf_m_s = flutter_results.flutter_speed

    v_knots, vf_hertz, v_label, omega_label, v_unit_short = _vgvf_plot_axes(config)
    V = _vgvf_velocity_for_plot(V_m_s, v_knots=v_knots)
    omega = _vgvf_frequency_for_plot(omega_rad_s, vf_hertz=vf_hertz)
    Vf_plot = _vgvf_scalar_velocity_for_plot(Vf_m_s, v_knots=v_knots)

    dimless = bool(getattr(config, "dimensionless_vgvf_results", True))
    delta_s_inv = vgvf_plotting.abramson_decay_factor_from_storage(
        g, omega_rad_s, dimensionless_vgvf_results=dimless
    )
    Lambda = vgvf_plotting.log_decrement_from_storage(
        g, omega_rad_s, dimensionless_vgvf_results=dimless
    )

    if not save_path and config and getattr(config, "save_plots", False):
        save_path = config.output_dir
    if save_path:
        os.makedirs(save_path, exist_ok=True)

    import pandas as pd

    vlam_data = {"Velocity_m_s": V_m_s}
    if v_knots:
        vlam_data["Velocity_knots"] = V
    for j in range(delta_s_inv.shape[1]):
        vlam_data[f"DecayFactor_s_inv_Branch_{j + 1}"] = delta_s_inv[:, j]
        vlam_data[f"LogDecrement_Branch_{j + 1}"] = Lambda[:, j]
    vlam_df = pd.DataFrame(vlam_data)
    vlam_csv = (
        os.path.join(save_path, _output_name_with_tag("vlambda_data.csv", output_tag))
        if save_path
        else _output_name_with_tag("vlambda_data.csv", output_tag)
    )
    vlam_df.to_csv(vlam_csv, index=False, float_format="%.6f")
    print(f"V–δ data saved to {vlam_csv}")

    fig, (ax_lam, ax_g, ax_f) = plt.subplots(1, 3, figsize=(18, 5))

    n_br = g.shape[1] if g.ndim > 1 else 1
    orphan_mac = bool(getattr(flutter_results, "orphan_mac_activated", False))

    def _branch_label(j):
        if orphan_mac and j == n_br - 1 and n_br >= 3:
            return "Orphan (MAC)"
        return f"Branch {j + 1}"

    for j in range(n_br):
        delta_col = delta_s_inv[:, j] if delta_s_inv.ndim > 1 else delta_s_inv
        ax_lam.plot(V, delta_col, label=_branch_label(j))
    ax_lam.axhline(0.0, linestyle="--", linewidth=1, color="gray")

    for j in range(n_br):
        g_col = g[:, j] if g.ndim > 1 else g
        ax_g.plot(V, g_col, label=_branch_label(j))
    ax_g.axhline(0.0, linestyle="--", linewidth=1, color="gray")

    n_om = omega.shape[1] if omega.ndim > 1 else 1
    for j in range(n_om):
        om_col = omega[:, j] if omega.ndim > 1 else omega
        ax_f.plot(V, om_col, label=_branch_label(j))

    Vf_list_plot = []
    for j in range(n_br):
        g_col = g[:, j] if g.ndim > 1 else g
        Vf_j_m_s, _ = vgvf_plotting.first_flutter_crossing(V_m_s, g_col)
        Vf_j_plot = _vgvf_scalar_velocity_for_plot(Vf_j_m_s, v_knots=v_knots)
        if Vf_j_plot is not None:
            Vf_list_plot.append(Vf_j_plot)
    if Vf_plot is not None:
        Vf_line = Vf_plot
    elif Vf_list_plot:
        Vf_line = min(Vf_list_plot)
    else:
        Vf_line = None

    if Vf_line is not None:
        for ax in (ax_lam, ax_g, ax_f):
            ax.axvline(Vf_line, linestyle="--", linewidth=1, color="red")
        ax_lam.text(
            Vf_line, ax_lam.get_ylim()[1] * 0.85, f"Vf ≈ {Vf_line:.1f} {v_unit_short}",
            rotation=90, ha="right", va="center",
        )

    ax_lam.set_xlabel(v_label)
    ax_lam.set_ylabel(r"Exponential decay factor $\delta$ [s$^{-1}$]")
    ax_lam.set_title(r"Flutter – V–$\delta$ (Abramson units)")
    ax_lam.legend()
    ax_lam.grid(True, alpha=0.3)

    ax_g.set_xlabel(v_label)
    ax_g.set_ylabel("Damping g = σ/ω [–]")
    ax_g.set_title("Flutter – V–g")
    ax_g.legend()
    ax_g.grid(True, alpha=0.3)

    ax_f.set_xlabel(v_label)
    ax_f.set_ylabel(omega_label)
    ax_f.set_title("Flutter – V–f")
    ax_f.legend()
    ax_f.grid(True, alpha=0.3)

    plt.tight_layout()

    combined_name = _output_name_with_tag("vg_lambda_vf_combined.png", output_tag)
    combined_outfile = os.path.join(save_path, combined_name) if save_path else combined_name
    if combined_outfile:
        plt.savefig(combined_outfile, dpi=150, bbox_inches="tight")
        print(f"Combined V–δ / V–g / V–f plot saved to {combined_outfile}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_static_hydroelastic_sweep(
    static_results,
    config=None,
    save_path="",
    show_plot=False,
    output_tag: Optional[str] = None,
):
    """
    Plot quasi-steady equilibrium deflection / twist vs speed from the static pass.
    """
    if static_results is None or len(static_results.V) == 0:
        return

    print("\n--- Plotting static hydroelastic equilibrium sweep ---")

    if not save_path and config and getattr(config, "save_plots", False):
        save_path = config.output_dir
    if save_path:
        os.makedirs(save_path, exist_ok=True)

    import pandas as pd

    V = np.asarray(static_results.V, dtype=float)
    tip_w = np.asarray(static_results.tip_w, dtype=float)
    tip_theta_deg = np.rad2deg(np.asarray(static_results.tip_theta, dtype=float))
    max_w = np.asarray(static_results.max_w, dtype=float)
    max_theta_deg = np.rad2deg(np.asarray(static_results.max_theta, dtype=float))
    max_slope = np.asarray(static_results.max_slope_downwash, dtype=float)

    chord = float(getattr(config, "chord", np.nan)) if config is not None else np.nan
    tip_w_over_c = tip_w / chord if np.isfinite(chord) and chord > 0 else np.full_like(tip_w, np.nan)

    df = pd.DataFrame(
        {
            "Velocity_m_s": V,
            "tip_w_m": tip_w,
            "tip_w_over_chord": tip_w_over_c,
            "tip_theta_deg": tip_theta_deg,
            "max_abs_w_m": max_w,
            "max_abs_theta_deg": max_theta_deg,
            "max_abs_Zqs_downwash": max_slope,
        }
    )
    csv_name = _output_name_with_tag("static_hydroelastic_sweep.csv", output_tag)
    if save_path:
        csv_path = os.path.join(save_path, csv_name)
        df.to_csv(csv_path, index=False)
        print(f"  Static hydroelastic CSV saved to {csv_path}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax_w, ax_th, ax_wc, ax_sl = axes.ravel()

    ax_w.plot(V, tip_w, "-o", markersize=3, label="tip w")
    ax_w.set_xlabel("Velocity [m/s]")
    ax_w.set_ylabel("Tip vertical deflection w [m]")
    ax_w.set_title("Static equilibrium – tip bending")
    ax_w.grid(True, alpha=0.3)
    ax_w.legend(fontsize=9)

    ax_th.plot(V, tip_theta_deg, "-o", markersize=3, color="C1", label="tip θ")
    ax_th.set_xlabel("Velocity [m/s]")
    ax_th.set_ylabel("Tip twist θ [deg]")
    ax_th.set_title("Static equilibrium – tip twist")
    ax_th.grid(True, alpha=0.3)
    ax_th.legend(fontsize=9)

    if np.any(np.isfinite(tip_w_over_c)):
        ax_wc.plot(V, tip_w_over_c, "-o", markersize=3, color="C2")
        ax_wc.set_xlabel("Velocity [m/s]")
        ax_wc.set_ylabel("tip w / chord [–]")
        ax_wc.set_title("Normalized tip deflection")
        ax_wc.grid(True, alpha=0.3)
    else:
        ax_wc.axis("off")
        ax_wc.text(0.5, 0.5, "chord not set", ha="center", va="center", transform=ax_wc.transAxes)

    ax_sl.plot(V, max_slope, "-o", markersize=3, color="C3")
    ax_sl.set_xlabel("Velocity [m/s]")
    ax_sl.set_ylabel(r"max $|Z_{qs}\,q|$ [–]")
    ax_sl.set_title("Quasi-steady slope coupling (panel downwash)")
    ax_sl.grid(True, alpha=0.3)

    plt.tight_layout()
    png_name = _output_name_with_tag("static_hydroelastic_sweep.png", output_tag)
    png_path = os.path.join(save_path, png_name) if save_path else png_name
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"  Static hydroelastic plot saved to {png_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()

    return df


def plot_vg_vf(flutter_results, config=None, save_path="", show_plot=False, output_tag: Optional[str] = None):
    """
    Generates and saves the V-g and V-f plots as subplots in a single figure.
    Also exports the data to CSV files.
    
    Args:
        flutter_results: FlutterResults object with velocities, damping, frequencies, flutter_speed
        config: Configuration object (optional)
        save_path: Path for saving plots and data (optional)
        show_plot: If True, display plot; if False, only save PNG and CSV but don't show (default: False)
        output_tag: If set, file basenames get ``_<tag>`` before the extension (for sweeps).
    """
    print("\n--- Generating V-g and V-f plots ---")

    V_m_s = np.asarray(flutter_results.velocities, dtype=float)
    g = np.asarray(flutter_results.damping, dtype=float)
    omega_rad_s = np.asarray(flutter_results.frequencies, dtype=float)
    Vf_m_s = flutter_results.flutter_speed

    v_knots, vf_hertz, v_label, omega_label, v_unit_short = _vgvf_plot_axes(config)
    V = _vgvf_velocity_for_plot(V_m_s, v_knots=v_knots)
    omega = _vgvf_frequency_for_plot(omega_rad_s, vf_hertz=vf_hertz)
    Vf_plot = _vgvf_scalar_velocity_for_plot(Vf_m_s, v_knots=v_knots)

    # Determine output directory - use provided save_path if available, otherwise use config
    if not save_path and config and config.save_plots:
        output_dir = config.output_dir
        os.makedirs(output_dir, exist_ok=True)
        save_path = output_dir
    
    combined_name = _output_name_with_tag("vg_vf_combined.png", output_tag)
    combined_outfile = os.path.join(save_path, combined_name) if save_path else combined_name
    
    # Export data to CSV files
    import pandas as pd
    
    # CSV: always store SI; add display columns when non-default units are requested
    vg_data = {'Velocity_m_s': V_m_s}
    if v_knots:
        vg_data['Velocity_knots'] = V
    for j in range(g.shape[1]):
        vg_data[f'Damping_Branch_{j+1}'] = g[:, j]
    # Raw decay rate sigma = g*omega [rad/s], alongside the normalised g=
    # sigma/omega ratio above. g blows up whenever omega -> 0 even for an
    # ordinarily-behaved, smoothly-more-damped root (verified 2026-09-22,
    # GOLAND_water: sigma grew perfectly linearly, -17->-58 rad/s, while a
    # branch's g dove to -3.9 purely because its frequency was dropping
    # toward zero -- not a divergence, a normalisation artifact). Readers
    # chasing a "the frequency crashed, is this static divergence?" signal
    # should check this column before g.
    for j in range(g.shape[1]):
        vg_data[f'Sigma_rad_s_Branch_{j+1}'] = g[:, j] * omega_rad_s[:, j]
    # p-k convergence flag per (V, branch), when tracked (see
    # vgvf_plotting.results_to_arrays) -- a non-converged point is whatever
    # k/omega the relaxation loop landed on at max_iter, not a
    # self-consistent root; readers filtering for "clean" trends should
    # mask these out rather than trust them at face value.
    conv = getattr(flutter_results, 'converged', None)
    if conv is not None:
        conv = np.asarray(conv)
        if conv.shape == g.shape:
            for j in range(g.shape[1]):
                vg_data[f'Converged_Branch_{j+1}'] = conv[:, j].astype(int)
    vg_df = pd.DataFrame(vg_data)

    vf_data = {'Velocity_m_s': V_m_s}
    if v_knots:
        vf_data['Velocity_knots'] = V
    for j in range(omega_rad_s.shape[1]):
        vf_data[f'Frequency_rad_s_Branch_{j+1}'] = omega_rad_s[:, j]
        if vf_hertz:
            vf_data[f'Frequency_Hz_Branch_{j+1}'] = omega[:, j]
    vf_df = pd.DataFrame(vf_data)
    
    # Save CSV files
    vg_csv = (
        os.path.join(save_path, _output_name_with_tag("vg_data.csv", output_tag))
        if save_path
        else _output_name_with_tag("vg_data.csv", output_tag)
    )
    vf_csv = (
        os.path.join(save_path, _output_name_with_tag("vf_data.csv", output_tag))
        if save_path
        else _output_name_with_tag("vf_data.csv", output_tag)
    )
    summary_csv = (
        os.path.join(save_path, _output_name_with_tag("flutter_summary.csv", output_tag))
        if save_path
        else _output_name_with_tag("flutter_summary.csv", output_tag)
    )
    
    vg_df.to_csv(vg_csv, index=False, float_format='%.6f')
    vf_df.to_csv(vf_csv, index=False, float_format='%.6f')
    
    print(f"V-g data saved to {vg_csv}")
    print(f"V-f data saved to {vf_csv}")
    
    # Flutter summary in SI (crossings computed on m/s velocities)
    flutter_summary = {
        'Branch': [],
        'Flutter_Speed_m_s': [],
        'Flutter_Frequency_rad_s': [],
    }
    if v_knots:
        flutter_summary['Flutter_Speed_knots'] = []
    if vf_hertz:
        flutter_summary['Flutter_Frequency_Hz'] = []
    for j in range(g.shape[1]):
        Vf_j_m_s, idx = vgvf_plotting.first_flutter_crossing(V_m_s, g[:, j])
        flutter_summary['Branch'].append(j + 1)
        flutter_summary['Flutter_Speed_m_s'].append(Vf_j_m_s if Vf_j_m_s is not None else np.nan)
        if v_knots:
            flutter_summary['Flutter_Speed_knots'].append(
                _vgvf_scalar_velocity_for_plot(Vf_j_m_s, v_knots=True)
                if Vf_j_m_s is not None else np.nan
            )
        if Vf_j_m_s is not None and idx is not None:
            flutter_summary['Flutter_Frequency_rad_s'].append(float(omega_rad_s[idx, j]))
            if vf_hertz:
                flutter_summary['Flutter_Frequency_Hz'].append(float(omega[idx, j]))
        else:
            flutter_summary['Flutter_Frequency_rad_s'].append(np.nan)
            if vf_hertz:
                flutter_summary['Flutter_Frequency_Hz'].append(np.nan)
    
    summary_df = pd.DataFrame(flutter_summary)
    summary_df.to_csv(summary_csv, index=False, float_format='%.6f')
    print(f"Flutter summary saved to {summary_csv}")

    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    
    n_br = g.shape[1] if g.ndim > 1 else 1
    orphan_mac = bool(getattr(flutter_results, "orphan_mac_activated", False))

    def _branch_label(j):
        if orphan_mac and j == n_br - 1 and n_br >= 3:
            return "Orphan (MAC)"
        return f"Branch {j + 1}"

    trim_pk = bool(
        config
        and getattr(config, "pk_method", False)
        and getattr(config, "plot_trim_pk_spikes", True)
    )

    # Plot V-g (left subplot)
    for j in range(n_br):
        col = g[:, j] if g.ndim > 1 else g
        om_col = omega[:, j] if omega.ndim > 1 else omega
        if trim_pk:
            v_plot, om_plot, g_plot = vgvf_plotting.trim_branch_smooth_prefix(
                V, om_col, col, min_v=18.0 if v_knots else 9.0,
            )
        else:
            v_plot, om_plot, g_plot = V, om_col, col
        ax1.plot(v_plot, g_plot, label=_branch_label(j))
    ax1.axhline(0.0, linestyle="--", linewidth=1, color='gray')
    
    # Annotate flutter speed on V-g plot (same display units as x-axis)
    Vf_list_plot = []
    for j in range(n_br):
        g_col = g[:, j] if g.ndim > 1 else g
        Vf_j_m_s, _ = vgvf_plotting.first_flutter_crossing(V_m_s, g_col)
        Vf_j_plot = _vgvf_scalar_velocity_for_plot(Vf_j_m_s, v_knots=v_knots)
        if Vf_j_plot is not None:
            Vf_list_plot.append(Vf_j_plot)
    if Vf_plot is not None:
        Vf_line = Vf_plot
    elif Vf_list_plot:
        Vf_line = min(Vf_list_plot)
    else:
        Vf_line = None
    if Vf_line is not None:
        ax1.axvline(Vf_line, linestyle="--", linewidth=1, color='red')
        ax1.text(
            Vf_line, ax1.get_ylim()[1] * 0.85, f"Vf ≈ {Vf_line:.1f} {v_unit_short}",
            rotation=90, ha="right", va="center",
        )

    ax1.set_xlabel(v_label)
    ax1.set_ylabel("Damping g = σ/ω [–]")
    ax1.set_title("Flutter Analysis – V-g")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot V-f (right subplot)
    n_om = omega.shape[1] if omega.ndim > 1 else 1
    for j in range(n_om):
        col = omega[:, j] if omega.ndim > 1 else omega
        g_col = g[:, j] if g.ndim > 1 else g
        if trim_pk:
            v_plot, om_plot, _ = vgvf_plotting.trim_branch_smooth_prefix(
                V, col, g_col, min_v=18.0 if v_knots else 9.0,
            )
        else:
            v_plot, om_plot = V, col
        ax2.plot(v_plot, om_plot, label=_branch_label(j))
    if Vf_line is not None:
        ax2.axvline(Vf_line, linestyle="--", linewidth=1, color='red')
        ax2.text(
            Vf_line, ax2.get_ylim()[1] * 0.85, f"Vf ≈ {Vf_line:.1f} {v_unit_short}",
            rotation=90, ha="right", va="center",
        )

    ax2.set_xlabel(v_label)
    ax2.set_ylabel(omega_label)
    ax2.set_title("Flutter Analysis – V-f")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # MAC-free orphan branch overlay (does not alter tracked Branch 1/2 curves)
    if config and getattr(config, "plot_orphan_wet_root_on_vgvf", False):
        plot_orphan_wet_root_on_vgvf(
            flutter_results,
            config=config,
            save_path=save_path,
            show_plot=False,
            output_tag=output_tag,
            ax_vg=ax1,
            ax_vf=ax2,
        )
        ax1.legend()
        ax2.legend()
    
    plt.tight_layout()
    
    if combined_outfile:
        plt.savefig(combined_outfile, dpi=150, bbox_inches='tight')
        print(f"Combined V-g and V-f plot saved to {combined_outfile}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()  # Close the figure to free memory

    plot_vg_lambda_vf_combined(
        flutter_results,
        config=config,
        save_path=save_path,
        show_plot=show_plot,
        output_tag=output_tag,
    )

    _plot_vgvf_stabilized_band(
        V_m_s, g, omega_rad_s, flutter_results, config, save_path, output_tag,
    )


def _plot_vgvf_stabilized_band(
    V_m_s, g, omega_rad_s, flutter_results, config, save_path, output_tag,
):
    """Save V-g/V-f (and V-δ/V-g/V-f) figures restricted to V >= vgvf_plot_V_min_*."""
    mask = _vgvf_stabilized_mask(V_m_s, config)
    if mask is None or not np.any(mask):
        return

    v_min_m_s = _vgvf_plot_V_min_m_s(config)
    v_knots, vf_hertz, v_label, omega_label, v_unit_short = _vgvf_plot_axes(config)
    V_all = _vgvf_velocity_for_plot(V_m_s, v_knots=v_knots)
    omega_all = _vgvf_frequency_for_plot(omega_rad_s, vf_hertz=vf_hertz)
    V = V_all[mask]
    g_f = g[mask]
    omega = omega_all[mask]
    V_m_s_f = np.asarray(V_m_s, dtype=float)[mask]
    omega_rad_f = np.asarray(omega_rad_s, dtype=float)[mask]

    v_min_plot = _vgvf_scalar_velocity_for_plot(v_min_m_s, v_knots=v_knots)
    title_suffix = f"V ≥ {v_min_plot:.2g} {v_unit_short} (stabilized UVLM band)"

    n_br = g_f.shape[1] if g_f.ndim > 1 else 1
    orphan_mac = bool(getattr(flutter_results, "orphan_mac_activated", False))

    def _branch_label(j):
        if orphan_mac and j == n_br - 1 and n_br >= 3:
            return "Orphan (MAC)"
        return f"Branch {j + 1}"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    for j in range(n_br):
        col = g_f[:, j] if g_f.ndim > 1 else g_f
        ax1.plot(V, col, label=_branch_label(j))
    ax1.axhline(0.0, linestyle="--", linewidth=1, color="gray")
    ax1.axvline(v_min_plot, linestyle=":", linewidth=1, color="gray", alpha=0.7)
    for j in range(n_br):
        om_col = omega[:, j] if omega.ndim > 1 else omega
        ax2.plot(V, om_col, label=_branch_label(j))
    ax2.axvline(v_min_plot, linestyle=":", linewidth=1, color="gray", alpha=0.7)
    ax1.set_xlabel(v_label)
    ax1.set_ylabel("Damping g = σ/ω [–]")
    ax1.set_title(f"Flutter – V-g ({title_suffix})")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax2.set_xlabel(v_label)
    ax2.set_ylabel(omega_label)
    ax2.set_title(f"Flutter – V-f ({title_suffix})")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    out = (
        os.path.join(save_path, _output_name_with_tag("vg_vf_combined_stabilized.png", output_tag))
        if save_path
        else _output_name_with_tag("vg_vf_combined_stabilized.png", output_tag)
    )
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Stabilized-band V-g/V-f plot saved to {out}")
    plt.close()

    if config is None or not getattr(config, "plot_log_decrement_vg", False):
        return

    dimless = bool(getattr(config, "dimensionless_vgvf_results", True))
    delta_s_inv = vgvf_plotting.abramson_decay_factor_from_storage(
        g_f, omega_rad_f, dimensionless_vgvf_results=dimless
    )
    fig, (ax_lam, ax_g, ax_f) = plt.subplots(1, 3, figsize=(18, 5))
    for j in range(n_br):
        ax_lam.plot(V, delta_s_inv[:, j], label=_branch_label(j))
    ax_lam.axhline(0.0, linestyle="--", linewidth=1, color="gray")
    for j in range(n_br):
        ax_g.plot(V, g_f[:, j], label=_branch_label(j))
    ax_g.axhline(0.0, linestyle="--", linewidth=1, color="gray")
    for j in range(omega.shape[1] if omega.ndim > 1 else 1):
        ax_f.plot(V, omega[:, j], label=_branch_label(j))
    for ax in (ax_lam, ax_g, ax_f):
        ax.axvline(v_min_plot, linestyle=":", linewidth=1, color="gray", alpha=0.7)
        ax.set_xlabel(v_label)
        ax.grid(True, alpha=0.3)
    ax_lam.set_ylabel(r"Exponential decay factor $\delta$ [s$^{-1}$]")
    ax_lam.set_title(rf"Flutter – V–$\delta$ ({title_suffix})")
    ax_g.set_ylabel("Damping g = σ/ω [–]")
    ax_g.set_title(f"Flutter – V-g ({title_suffix})")
    ax_f.set_ylabel(omega_label)
    ax_f.set_title(f"Flutter – V-f ({title_suffix})")
    for ax in (ax_lam, ax_g, ax_f):
        ax.legend()
    plt.tight_layout()
    out_lam = (
        os.path.join(save_path, _output_name_with_tag("vg_lambda_vf_combined_stabilized.png", output_tag))
        if save_path
        else _output_name_with_tag("vg_lambda_vf_combined_stabilized.png", output_tag)
    )
    plt.savefig(out_lam, dpi=150, bbox_inches="tight")
    print(f"Stabilized-band V-δ/V-g/V-f plot saved to {out_lam}")
    plt.close()


def plot_qmodal_entries_at_convergence(config, structural_results, coupling_results, flutter_results,
                                       mode_index=0, save_csv=True):
    """
    For each velocity step, recompute the modal aerodynamic operator Q_modal at the
    converged iteration using the converged eigenvalue p and reduced frequency k
    from the selected flutter branch (mode_index), then extract and plot:

      Re(Q00), Re(Q01), Re(Q10), Re(Q11), Im(Q00), Im(Q11)

    Args:
        config: analysis configuration (to rebuild Qg_func)
        structural_results: provides dry eigenvectors/values and reduced matrices
        coupling_results: dict with Z, Z_qs, Apan
        flutter_results: FlutterResults returned by flutter_solver.solve()
        mode_index: which branch to use for the converged p,k at each V
        save_csv: store the sampled values to CSV in config.output_dir if enabled

    Output:
        - PNG plot saved next to other plots (if enabled)
        - CSV file with columns [V, Re00, Re01, Re10, Re11, Im00, Im11]
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from panelaero_utl import pk_solverv3

    print("\n--- Computing Q_modal entries at convergence ---")

    # Build Qg_func once (uses closure with Z, Z_qs, Apan, Phi)
    Z = coupling_results['Z']
    Apan = coupling_results['Apan']
    Z_qs = coupling_results['Z_qs']

    solver_tmp = pk_solverv3.PKSolverV3(
        structural_results.M_hat,
        structural_results.C_hat,
        structural_results.K_hat,
        structural_results.dry_eigenvalues
    )

    Qg_func = solver_tmp.make_Qg_func(
        config.paths['FSI'],
        Z,
        Apan,
        Z_qs,
        structural_results.dry_eigenvectors,
        structural_results.dry_eigenvalues,
        b=config.chord / 2,
        c_sound=config.c_sound[config.fluid],
        out_dir_klist=config.qjj_dir,
        out_dir_vlm=config.vjj_dir,
        alpha_r=config.alpha_r
    )

    # Iterate velocities using raw results to access converged p and k
    V_list = np.asarray(flutter_results.velocities)
    results_raw = flutter_results.raw_results

    Re00_list, Re01_list, Re10_list, Re11_list = [], [], [], []
    Im00_list, Im11_list = [], []
    V_used = []

    for i, res_v in enumerate(results_raw):
        V = res_v['V']
        modes = res_v.get('modes', [])
        if mode_index >= len(modes):
            # No such mode tracked at this V
            continue
        md = modes[mode_index]
        if not md.get('converged', False):
            # Skip non-converged points
            continue

        k_red = max(md.get('k', np.nan), 1e-9)
        p_conv = md.get('p', None)
        if p_conv is None or np.isnan(np.real(p_conv)) or np.isnan(np.imag(p_conv)):
            continue

        # Recompute Q_modal using converged p and k at this V
        try:
            Q_modal = Qg_func(k_red, V, p_conv, structural_results.dry_eigenvectors, mode_index, md.get('iterations', 0)-1)
        except Exception as e:
            print(f"  Warning: could not compute Q_modal at V={V:.2f}: {e}")
            continue

        # Ensure size is at least 2x2
        if Q_modal.shape[0] < 2 or Q_modal.shape[1] < 2:
            print(f"  Warning: Q_modal is {Q_modal.shape}, need at least 2x2; skipping V={V:.2f}")
            continue

        Re00_list.append(np.real(Q_modal[0, 0]))
        Re01_list.append(np.real(Q_modal[0, 1]))
        Re10_list.append(np.real(Q_modal[1, 0]))
        Re11_list.append(np.real(Q_modal[1, 1]))
        Im00_list.append(np.imag(Q_modal[0, 0]))
        Im11_list.append(np.imag(Q_modal[1, 1]))
        V_used.append(V)

    if len(V_used) == 0:
        print("No converged Q_modal entries found to plot.")
        return

    V_arr = np.asarray(V_used)

    # CSV export
    save_path = ""
    if getattr(config, 'save_plots', False):
        os.makedirs(config.output_dir, exist_ok=True)
        save_path = config.output_dir

    if save_csv:
        import pandas as pd
        df = pd.DataFrame({
            'Velocity_m_s': V_arr,
            'Re00': Re00_list,
            'Re01': Re01_list,
            'Re10': Re10_list,
            'Re11': Re11_list,
            'Im00': Im00_list,
            'Im11': Im11_list,
        })
        csv_file = os.path.join(save_path, "qmodal_entries.csv") if save_path else "qmodal_entries.csv"
        df.to_csv(csv_file, index=False, float_format='%.6f')
        print(f"Q_modal entries saved to {csv_file}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    ax1.plot(V_arr, Re00_list, label='Re00')
    ax1.plot(V_arr, Re01_list, label='Re01')
    ax1.plot(V_arr, Re10_list, label='Re10')
    ax1.plot(V_arr, Re11_list, label='Re11')
    ax1.set_xlabel('Airspeed V [m/s]')
    ax1.set_ylabel('Re(Q_modal[i,j]) [N]')
    ax1.set_title('Real parts of Q_modal entries (2x2) at convergence')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(V_arr, Im00_list, label='Im00')
    ax2.plot(V_arr, Im11_list, label='Im11')
    ax2.set_xlabel('Airspeed V [m/s]')
    ax2.set_ylabel('Im(Q_modal[i,i]) [N]')
    ax2.set_title('Imaginary parts of Q_modal entries (diagonal) at convergence')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    outfile = os.path.join(save_path, "qmodal_entries.png") if save_path else "qmodal_entries.png"
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    print(f"Q_modal entries plot saved to {outfile}")
    plt.show()


def plot_qmodal_pk_iteration_history(config, qmodal_history, save_csv=True, show_plot=False):
    """
    Plot every entry of Q_damp (Re and Im) used at each PK iteration.

    ``qmodal_history`` is the list filled by
    :func:`aero_system.make_aero_build_matrix_fn` /
    :func:`aero_system.make_aero_on_converge_fn`; each record holds the
    ``Q_damp = scale_Qhh_imag_for_damping(Qhh, config)`` matrix actually used
    to assemble K_aero (= q_inf Re(Q_damp)) and C_aero (∝ Im(Q_damp)/k).

    Produces two figures (Re and Im), each an n×n grid of subplots — one per
    matrix entry — with the value vs. the global PK call index. Lines are
    colored by tracked mode branch; converged calls are marked with a star;
    dotted vertical lines mark velocity changes (V annotated on the top row).
    Optionally dumps the full history to CSV.
    """
    if not qmodal_history:
        print("[Q_damp history] Empty PK Q_modal history, nothing to plot.")
        return

    from . import aero_q_scaling

    Q = np.stack([
        np.asarray(rec.get("Q_damp", rec["Q_modal"]), dtype=complex)
        for rec in qmodal_history
    ])
    n_calls, n, _ = Q.shape
    idx = np.arange(n_calls)
    V_arr = np.array([rec["V"] for rec in qmodal_history], dtype=float)
    k_arr = np.array([rec["k"] for rec in qmodal_history], dtype=float)
    mode_arr = np.array([rec["mode"] for rec in qmodal_history], dtype=int)
    iter_arr = np.array([rec["iteration"] for rec in qmodal_history], dtype=int)
    conv_arr = np.array([bool(rec.get("converged", False)) for rec in qmodal_history])

    scale = aero_q_scaling.aero_im_Q_scale(config)
    print(f"\n--- Plotting Q_damp entries per PK iteration "
          f"({n_calls} calls, {n}x{n} matrix, aero_im_Q_scale={scale:g}) ---")

    save_path = ""
    if config is not None and getattr(config, 'output_dir', None):
        os.makedirs(config.output_dir, exist_ok=True)
        save_path = config.output_dir

    # Velocity segment boundaries (indices where V changes)
    v_change = np.where(np.diff(V_arr) != 0.0)[0] + 1
    seg_starts = np.concatenate([[0], v_change])
    seg_ends = np.concatenate([v_change, [n_calls]])
    seg_mid = 0.5 * (seg_starts + seg_ends - 1)
    seg_V = V_arr[seg_starts]

    modes_tracked = sorted(set(mode_arr.tolist()))
    cmap = plt.cm.tab10
    mode_color = {s: cmap(i % 10) for i, s in enumerate(modes_tracked)}

    for part_name, part_fn in (("Re", np.real), ("Im", np.imag)):
        fig, axes = plt.subplots(
            n, n,
            figsize=(max(3.2 * n + 2, 10), max(2.4 * n + 1.5, 7)),
            sharex=True, squeeze=False,
        )
        for i in range(n):
            for jc in range(n):
                ax = axes[i][jc]
                vals = part_fn(Q[:, i, jc])
                for s in modes_tracked:
                    m = mode_arr == s
                    ax.plot(idx[m], vals[m], '.-', ms=3, lw=0.8,
                            color=mode_color[s],
                            label=f"mode {s}" if (i == 0 and jc == 0) else None)
                    mc = m & conv_arr
                    if np.any(mc):
                        ax.plot(idx[mc], vals[mc], '*', ms=9,
                                color=mode_color[s], mec='k', mew=0.4, zorder=5)
                for b in v_change:
                    ax.axvline(b - 0.5, color='gray', ls=':', lw=0.6, alpha=0.7)
                ax.set_title(f"{part_name}(Q[{i},{jc}])", fontsize=9)
                ax.grid(True, alpha=0.25)
                ax.tick_params(labelsize=7)
                if i == n - 1:
                    ax.set_xlabel("PK call index (all iterations)", fontsize=8)

        # Annotate velocities on the top row (thinned to avoid clutter)
        step = max(1, len(seg_V) // 10)
        for jc in range(n):
            ax_top = axes[0][jc]
            for mid, Vv in zip(seg_mid[::step], seg_V[::step]):
                ax_top.annotate(
                    f"V={Vv:g}", xy=(mid, 1.02), xycoords=('data', 'axes fraction'),
                    ha='center', va='bottom', fontsize=6, rotation=45, color='dimgray',
                )

        handles, labels = axes[0][0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc='upper right', fontsize=8, ncol=min(len(handles), 4))
        fig.suptitle(
            f"{part_name}(Q_damp) per PK iteration  "
            f"(aero_im_Q_scale={scale:g}; star = converged)",
            fontsize=11, y=1.0,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.96))

        fname = f"qmodal_pk_iterations_{part_name}.png"
        outfile = os.path.join(save_path, fname) if save_path else fname
        fig.savefig(outfile, dpi=150, bbox_inches='tight')
        print(f"Q_damp PK-iteration plot saved to {outfile}")
        if show_plot:
            plt.show()
        else:
            plt.close(fig)

    if save_csv:
        import pandas as pd
        data = {
            'call_index': idx,
            'Velocity_m_s': V_arr,
            'k_red': k_arr,
            'mode': mode_arr,
            'iteration': iter_arr,
            'converged': conv_arr,
        }
        for i in range(n):
            for jc in range(n):
                data[f'ReQ{i}{jc}'] = np.real(Q[:, i, jc])
        for i in range(n):
            for jc in range(n):
                data[f'ImQ{i}{jc}'] = np.imag(Q[:, i, jc])
        csv_file = (os.path.join(save_path, "qmodal_pk_iterations.csv")
                    if save_path else "qmodal_pk_iterations.csv")
        pd.DataFrame(data).to_csv(csv_file, index=False, float_format='%.6e')
        print(f"Q_damp PK-iteration history saved to {csv_file}")


def plot_damping_debug(structural_results, flutter_results, config=None):
    """
    Debug plot to visualize structural (Rayleigh) damping vs mode and aerodynamic-only damping vs velocity.

    - Structural damping per dry mode: zeta_i = alpha/(2*omega_i) + beta*omega_i/2 with omega_i = sqrt(lambda_i)
    - Aerodynamic-only damping: g_aero(V, j) = g_total(V, j) - zeta_struct(omega(V, j))

    Saves to output_dir if config.save_plots is True.
    """
    print("\n--- Debug: Plotting Structural and Aerodynamic Damping ---")

    V = flutter_results.velocities
    g_total = flutter_results.damping  # shape: (nV, nBranches)
    omega_v = flutter_results.frequencies  # shape: (nV, nBranches)

    # Structural Rayleigh coefficients
    alpha = getattr(structural_results, 'rayleigh_alpha', None)
    beta = getattr(structural_results, 'rayleigh_beta', None)

    if alpha is None or beta is None:
        print("[Debug] Rayleigh coefficients not found in structural_results; skipping aerodynamic separation.")
        return

    # Output directory
    save_path = ""
    if config and getattr(config, 'save_plots', False):
        os.makedirs(config.output_dir, exist_ok=True)
        save_path = config.output_dir

    # Compute structural damping per dry mode
    dry_vals = structural_results.dry_eigenvalues  # omega^2
    omegas_dry = np.sqrt(np.maximum(dry_vals, 0.0))
    zeta_struct_modes = alpha / (2.0 * np.maximum(omegas_dry, 1e-12)) + beta * omegas_dry / 2.0

    # Compute aerodynamic-only damping across V by subtracting Rayleigh contribution at the aeroelastic frequency
    zeta_struct_vs_v = alpha / (2.0 * np.maximum(omega_v, 1e-12)) + beta * omega_v / 2.0
    g_aero = g_total - zeta_struct_vs_v

    # Plot: left structural per mode, right aerodynamic-only V-g
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    # Structural damping per dry mode
    ax1.plot(np.arange(1, len(zeta_struct_modes) + 1), zeta_struct_modes, marker='o')
    ax1.set_xlabel('Dry Mode Index')
    ax1.set_ylabel('Structural Damping Ratio ζ_struct [–]')
    ax1.set_title(f'Rayleigh Damping per Dry Mode\n(alpha={alpha:.3e}, beta={beta:.3e})')
    ax1.grid(True, alpha=0.3)

    # Aerodynamic-only damping curves
    for j in range(g_aero.shape[1]):
        ax2.plot(V, g_aero[:, j], label=f"Branch {j+1}")
    ax2.axhline(0.0, linestyle="--", linewidth=1, color='gray')
    ax2.set_xlabel('Airspeed V [m/s]')
    ax2.set_ylabel('Aerodynamic-only Damping g_aero [–]')
    ax2.set_title('V - g_aero (Total minus Rayleigh)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    outfile = os.path.join(save_path, "damping_debug.png") if save_path else "damping_debug.png"
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    print(f"Damping debug plot saved to {outfile}")
    plt.show()


def plot_damping_matrices(structural_results, coupling_results, flutter_results, config=None, mode_index=0, velocity='flutter'):
    """
    Plot structural damping matrix (modal-reduced C_hat) and an equivalent aerodynamic
    damping matrix evaluated from the imaginary part of the modal aerodynamic operator Qhh
    at a representative velocity and reduced frequency.

    The aerodynamic equivalent viscous damping is constructed as:
      C_aero = (1/4) * rho * c * V / k * Im{Qhh}
    where c = 2*b, k = omega*b/V, and Qhh is the modal aerodynamic matrix.

    Args:
        structural_results: StructuralResults namedtuple with M_hat, C_hat, K_hat, dry_eigenvalues, dry_eigenvectors
        coupling_results: dict with keys 'Z' and 'Apan'
        flutter_results: FlutterResults with velocities, frequencies, damping and raw_results
        config: analysis config (for geometry and fluid properties)
        mode_index: which mode index to use for k computation
        velocity: 'flutter' to pick Vf if available, else 'mid' or a numeric value (m/s)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from panelaero_utl import pk_solverv3

    print("\n--- Plotting Structural and Aerodynamic Damping Matrices ---")

    # Choose sample velocity
    V_array = flutter_results.velocities
    if isinstance(velocity, (int, float)):
        V_samp = float(velocity)
        # pick nearest available velocity index
        idx = int(np.argmin(np.abs(V_array - V_samp)))
    else:
        if velocity == 'flutter' and flutter_results.flutter_speed is not None:
            Vf = flutter_results.flutter_speed
            idx = int(np.argmin(np.abs(V_array - Vf)))
        else:
            idx = len(V_array) // 2
    V_samp = V_array[idx]

    # Pick frequency for mode_index at this velocity
    omega_samp = flutter_results.frequencies[idx, mode_index]
    b = config.chord / 2.0
    rho = config.rho_f[config.fluid]
    c = 2.0 * b
    k_red = max(omega_samp * b / max(V_samp, 1e-6), 1e-6)

    # Build Qg_func (modal aerodynamic operator)
    Z = coupling_results['Z']
    Apan = coupling_results['Apan']
    Z_qs = coupling_results['Z_qs']

    solver_tmp = pk_solverv3.PKSolverV3(
        structural_results.M_hat,
        structural_results.C_hat,
        structural_results.K_hat,
        structural_results.dry_eigenvalues
    )

    Qg_func = solver_tmp.make_Qg_func(
        config.paths['FSI'],
        Z,
        Apan,
        Z_qs,
        structural_results.dry_eigenvectors,  # Free DOFs only to match Apan/Z dimensions
        structural_results.dry_eigenvalues,
        b=config.chord / 2,
        c_sound=config.c_sound[config.fluid],
        out_dir_klist=config.qjj_dir,
        out_dir_vlm=config.vjj_dir,
        alpha_r=config.alpha_r
    )

    # Qhh is complex modal aerodynamic matrix; its imaginary part corresponds to aerodynamic damping term
    # Compute modal aerodynamic operator using k at sampled V
    Qhh = Qg_func(k_red, V_samp, None, structural_results.dry_eigenvectors, mode_index, 0)
    qinf_I = 0.25 * rho * c * V_samp / max(k_red, 1e-6)
    C_aero = qinf_I * np.imag(Qhh)

    # Structural damping matrix (modal reduced)
    C_struct = structural_results.C_hat

    # Plot side-by-side heatmaps
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    im1 = ax1.imshow(C_struct, cmap='RdBu_r', aspect='equal')
    ax1.set_title('Structural Damping Matrix Ĉ (modal)')
    ax1.set_xlabel('Mode')
    ax1.set_ylabel('Mode')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    im2 = ax2.imshow(C_aero, cmap='RdBu_r', aspect='equal')
    ax2.set_title(f'Aerodynamic Damping Matrix C_aero (V={V_samp:.1f} m/s, k={k_red:.3f})')
    ax2.set_xlabel('Mode')
    ax2.set_ylabel('Mode')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    plt.tight_layout()

    # Save
    save_path = ""
    if config and getattr(config, 'save_plots', False):
        os.makedirs(config.output_dir, exist_ok=True)
        save_path = config.output_dir
    outfile = os.path.join(save_path, f"damping_matrices_V{V_samp:.1f}.png") if save_path else f"damping_matrices_V{V_samp:.1f}.png"
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    print(f"Damping matrices plot saved to {outfile}")
    plt.show()


def plot_eigenvalue_trajectory(flutter_results, results_raw, config=None, save_path=""):
    """
    Generates eigenvalue trajectory plot: Real(λ) vs Imag(λ) colored by velocity.
    
    Args:
        flutter_results: FlutterResults namedtuple (for metadata)
        results_raw: Raw results from pk_solver.sweep() containing eigenvalues
        config: Configuration object
        save_path: Path for saving plots
    """
    print("\n--- Generating Eigenvalue Trajectory Plot ---")
    
    # Determine output directory
    if config and config.save_plots:
        output_dir = config.output_dir
        os.makedirs(output_dir, exist_ok=True)
        save_path = output_dir
    
    outfile = os.path.join(save_path, "eigenvalue_trajectory.png") if save_path else "eigenvalue_trajectory.png"
    
    # Extract eigenvalues and velocities from raw results
    eigenvalue_data = []
    
    for result_v in results_raw:
        V = result_v['V']
        modes_list = result_v.get('modes', [])
        
        for mode_data in modes_list:
            p = mode_data.get('p', None)
            if p is not None and not np.isnan(p):
                # p is the complex eigenvalue λ
                real_part = np.real(p)
                imag_part = np.imag(p)
                
                # Only include eigenvalues with positive imaginary part (physical modes)
                if imag_part > 0:
                    eigenvalue_data.append({
                        'V': V,
                        'real': real_part,
                        'imag': imag_part,
                        'mode': mode_data.get('mode', -1)
                    })
    
    if not eigenvalue_data:
        print("No eigenvalue data found to plot.")
        return
    
    # Convert to arrays for plotting
    velocities = np.array([d['V'] for d in eigenvalue_data])
    real_parts = np.array([d['real'] for d in eigenvalue_data])
    imag_parts = np.array([d['imag'] for d in eigenvalue_data])
    modes = np.array([d['mode'] for d in eigenvalue_data])
    
    # Export eigenvalue data to CSV
    import pandas as pd
    eigenvalue_df = pd.DataFrame({
        'Velocity_m_s': velocities,
        'Mode': modes,
        'Real_Part_lambda_rad_s': real_parts,
        'Imag_Part_lambda_rad_s': imag_parts
    })
    
    eigenvalue_csv = os.path.join(save_path, "eigenvalue_data.csv") if save_path else "eigenvalue_data.csv"
    eigenvalue_df.to_csv(eigenvalue_csv, index=False, float_format='%.6f')
    print(f"Eigenvalue data saved to {eigenvalue_csv}")
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Scatter plot with velocity as color
    scatter = ax.scatter(real_parts, imag_parts, c=velocities, 
                        cmap='Blues', s=30, alpha=0.8, edgecolors='none')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, label='Free Stream Velocity, V [m/s]')
    
    # Formatting
    ax.set_xlabel('Real Part, λ [rad/s]', fontsize=12)
    ax.set_ylabel('Imag Part, λ [rad/s]', fontsize=12)
    ax.set_title('Eigenvalue Trajectory', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Set axis limits similar to the reference plot if needed
    # ax.set_xlim([-50, 0])
    # ax.set_ylim([0, 300])
    
    plt.tight_layout()
    
    if outfile:
        plt.savefig(outfile, dpi=150, bbox_inches='tight')
        print(f"Eigenvalue trajectory plot saved to {outfile}")
    
    if config.show_eigen_plot:
        plt.show()


def _dedupe_pk_results_by_velocity(raw_results):
    """Keep the last PK snapshot per airspeed (resting-fluid step may repeat V)."""
    by_v = {}
    for entry in raw_results:
        by_v[float(entry["V"])] = entry
    return [by_v[v] for v in sorted(by_v)]


def extract_wet_spectrum_branch_trajectory(
    raw_results,
    *,
    seed="lowest_untracked",
    omega_n=None,
    freq_margin=0.20,
):
    """
    Track one wet eigenvalue branch across V using wet-spectrum data only (no MAC).

    With ``seed='lowest_untracked'``, each step uses the lowest-frequency untracked
    root when present; afterward the same ``spectrum_index`` is followed. If that
    index vanishes, the closest unused root by frequency is chosen (orphan continuity).

    Returns dict with keys ``V``, ``omega``, ``gamma``, ``spectrum_index``, ``kind``.
    """
    entries = _dedupe_pk_results_by_velocity(raw_results)
    if not entries:
        return None

    if omega_n is None:
        omega_n = np.array([], dtype=float)
    else:
        omega_n = np.asarray(omega_n, dtype=float).ravel()

    tracked_index = None
    omega_prev = None
    out = {
        "V": [],
        "omega": [],
        "gamma": [],
        "spectrum_index": [],
        "kind": [],
    }

    def _pick_root(spec):
        nonlocal tracked_index, omega_prev
        roots = list(spec.get("roots") or [])
        if not roots:
            return None

        if seed == "lowest_untracked":
            low = spec.get("lowest_untracked")
            if low is not None:
                tracked_index = int(low["index"])
                return low

        if tracked_index is not None:
            for r in roots:
                if int(r["index"]) == int(tracked_index):
                    return r

        # Continuation: prefer untracked / structural_near, closest in omega to previous
        candidates = [
            r
            for r in roots
            if r.get("kind") in ("untracked", "structural_near")
        ]
        if not candidates:
            candidates = [r for r in roots if r.get("kind") != "pk_tracked"]
        if omega_prev is not None and omega_prev > 0:
            candidates.sort(
                key=lambda r: abs(float(r["omega"]) - omega_prev) / omega_prev
            )
        else:
            untracked = [r for r in candidates if r.get("kind") == "untracked"]
            if untracked:
                candidates = sorted(untracked, key=lambda r: r["omega"])
            else:
                candidates = sorted(candidates, key=lambda r: r["omega"])
        if not candidates:
            return None
        pick = candidates[0]
        tracked_index = int(pick["index"])
        return pick

    for entry in entries:
        V = float(entry["V"])
        spec = _wet_spectrum_from_velocity_entry(entry, omega_n, freq_margin)
        if spec is None:
            continue
        r = _pick_root(spec)
        if r is None:
            continue
        om = float(r["omega"])
        out["V"].append(V)
        out["omega"].append(om)
        out["gamma"].append(float(r["gamma"]))
        out["spectrum_index"].append(int(r["index"]))
        out["kind"].append(str(r.get("kind", "")))
        omega_prev = om

    if not out["V"]:
        return None
    for key in out:
        out[key] = np.asarray(out[key])
    return out


def plot_orphan_wet_root_on_vgvf(
    flutter_results,
    config=None,
    save_path="",
    show_plot=False,
    output_tag=None,
    ax_vg=None,
    ax_vf=None,
):
    """
    Overlay the MAC-free orphan / lowest-untracked wet root on V-g and V-f plots.

    Does not alter tracked branches — diagnostic only.
    """
    raw = getattr(flutter_results, "raw_results", None)
    if not raw:
        return None

    traj = extract_wet_spectrum_branch_trajectory(raw, seed="lowest_untracked")
    if traj is None:
        print("  Orphan wet-root branch: no trajectory extracted.")
        return None

    v_knots, vf_hertz, v_label, omega_label, _ = _vgvf_plot_axes(config)
    V_plot = _vgvf_velocity_for_plot(traj["V"], v_knots=v_knots)
    omega_plot = _vgvf_frequency_for_plot(traj["omega"], vf_hertz=vf_hertz)

    standalone = ax_vg is None or ax_vf is None
    if standalone:
        fig, (ax_vg, ax_vf) = plt.subplots(1, 2, figsize=(16, 5))

    ax_vg.plot(
        V_plot,
        traj["gamma"],
        "o-",
        color="crimson",
        linewidth=2.0,
        markersize=3,
        label="orphan branch (spectrum track)",
        zorder=6,
    )
    ax_vf.plot(
        V_plot,
        omega_plot,
        "o-",
        color="crimson",
        linewidth=2.0,
        markersize=3,
        label="orphan branch (spectrum track)",
        zorder=6,
    )
    if standalone:
        ax_vg.axhline(0.0, linestyle="--", linewidth=1, color="gray")
        ax_vg.set_xlabel(v_label)
        ax_vg.set_ylabel("Damping g = σ/ω [–]")
        ax_vg.set_title("Orphan wet root — V-g (MAC-free track)")
        ax_vf.set_xlabel(v_label)
        ax_vf.set_ylabel(omega_label)
        ax_vf.set_title("Orphan wet root — V-f (MAC-free track)")
        ax_vg.legend()
        ax_vf.legend()
        ax_vg.grid(True, alpha=0.3)
        ax_vf.grid(True, alpha=0.3)
        plt.tight_layout()
        if not save_path and config and getattr(config, "save_plots", False):
            save_path = config.output_dir
        if save_path:
            os.makedirs(save_path, exist_ok=True)
        out_png = (
            os.path.join(
                save_path,
                _output_name_with_tag("vg_vf_orphan_branch.png", output_tag),
            )
            if save_path
            else _output_name_with_tag("vg_vf_orphan_branch.png", output_tag)
        )
        plt.savefig(out_png, dpi=150, bbox_inches="tight")
        print(f"Orphan-branch V-g/V-f overlay saved to {out_png}")
        import pandas as pd

        csv_out = (
            os.path.join(
                save_path,
                _output_name_with_tag("orphan_wet_root_branch.csv", output_tag),
            )
            if save_path
            else _output_name_with_tag("orphan_wet_root_branch.csv", output_tag)
        )
        pd.DataFrame(
            {
                "Velocity_m_s": traj["V"],
                "omega_rad_s": traj["omega"],
                "gamma": traj["gamma"],
                "spectrum_index": traj["spectrum_index"],
                "kind": traj["kind"],
            }
        ).to_csv(csv_out, index=False, float_format="%.6e")
        print(f"Orphan-branch trajectory saved to {csv_out}")
        if show_plot:
            plt.show()
        else:
            plt.close()

    return traj


def _wet_spectrum_from_velocity_entry(entry, omega_n, freq_margin):
    """Return classified wet spectrum for one velocity step."""
    if entry.get("wet_spectrum") is not None:
        return entry["wet_spectrum"]

    from . import pk_method

    modes_list = entry.get("modes", [])
    if modes_list:
        spec = pk_method.pk_wet_spectrum_for_velocity(modes_list, omega_n, freq_margin)
        if spec is not None:
            return spec
    return None


def plot_pk_converged_wet_eigenvectors(flutter_results, config=None, save_path="", show_plot=False):
    """
    Plot wet roots not assigned to tracked flutter branches (PK or Roger/RFA).

    At each airspeed the full oscillatory spectrum is classified; roots matched to
    tracked branches are excluded. Remaining roots (including emergent / orphan modes)
    are saved to ``pk_untracked_wet_roots.csv`` and plotted as ω(V), g(V).
    """
    print("\n--- Generating Untracked Wet-Root Trend Plot ---")

    raw_results = getattr(flutter_results, "raw_results", None)
    if not raw_results:
        print("No raw modal snapshots available for untracked-root plot.")
        return

    if config and getattr(config, "save_plots", False):
        output_dir = config.output_dir
        os.makedirs(output_dir, exist_ok=True)
        save_path = output_dir

    freq_margin = float(
        getattr(config, "rfa_freq_margin", None)
        or getattr(config, "pk_freq_margin", 0.15)
    )
    omega_n = np.array([], dtype=float)

    entries = _dedupe_pk_results_by_velocity(raw_results)
    spectra = []
    for entry in entries:
        spec = _wet_spectrum_from_velocity_entry(entry, omega_n, freq_margin)
        if spec is not None:
            spectra.append((float(entry["V"]), spec))
            if omega_n.size == 0 and spec.get("omega_structural"):
                omega_n = np.asarray(spec["omega_structural"], dtype=float)
                if spec.get("freq_margin") is not None:
                    freq_margin = float(spec["freq_margin"])

    if not spectra:
        print("No wet-spectrum snapshots found (re-run analysis with updated pk_method).")
        return

    fig, (ax_omega, ax_g) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Tracked branches (PK or Roger/RFA, depending on solver branch)
    V_pk = np.asarray(flutter_results.velocities, dtype=float)
    omega_pk = np.asarray(flutter_results.frequencies, dtype=float)
    g_pk = np.asarray(flutter_results.damping, dtype=float)
    branch_label_prefix = "tracked branch"
    for j in range(omega_pk.shape[1] if omega_pk.ndim > 1 else 1):
        om_col = omega_pk[:, j] if omega_pk.ndim > 1 else omega_pk
        g_col = g_pk[:, j] if g_pk.ndim > 1 else g_pk
        ax_omega.plot(V_pk, om_col, "--", linewidth=1.2, alpha=0.7, label=f"{branch_label_prefix} {j + 1}")
        ax_g.plot(V_pk, g_col, "--", linewidth=1.2, alpha=0.7, label=f"{branch_label_prefix} {j + 1}")

    if omega_n.size > 0:
        for si, om_s in enumerate(omega_n):
            ax_omega.axhline(
                float(om_s),
                color="gray",
                linestyle=":",
                linewidth=0.9,
                alpha=0.6,
                label=f"dry ω_{si}" if si < 2 else None,
            )

    V_orphan, om_orphan, g_orphan = [], [], []
    csv_rows = []
    vec_rows = []
    untracked_branches = {}

    for V, spec in spectra:
        lowest = spec.get("lowest_untracked")
        if lowest is not None:
            V_orphan.append(V)
            om_orphan.append(lowest["omega"])
            g_orphan.append(lowest["gamma"])

        for r in spec.get("untracked", []):
            key = int(r["index"])
            untracked_branches.setdefault(key, {"V": [], "omega": [], "gamma": []})
            untracked_branches[key]["V"].append(V)
            untracked_branches[key]["omega"].append(r["omega"])
            untracked_branches[key]["gamma"].append(r["gamma"])

        for r in spec.get("roots", []):
            csv_rows.append(
                {
                    "Velocity_m_s": V,
                    "spectrum_index": r["index"],
                    "kind": r["kind"],
                    "omega_rad_s": r["omega"],
                    "gamma": r["gamma"],
                    "sigma_rad_s": r["sigma"],
                    "lambda_re": float(np.real(r["p"])),
                    "lambda_im": float(np.imag(r["p"])),
                    "structural_near_mode": r["structural_near_mode"],
                    "is_lowest_untracked": bool(
                        lowest is not None and r["index"] == lowest["index"]
                    ),
                }
            )

            vec = r.get("vec")
            if vec is not None:
                vec = np.asarray(vec, dtype=complex).ravel()
                row = {
                    "Velocity_m_s": V,
                    "spectrum_index": r["index"],
                    "kind": r["kind"],
                    "omega_rad_s": r["omega"],
                    "gamma": r["gamma"],
                    "sigma_rad_s": r["sigma"],
                    "structural_near_mode": r["structural_near_mode"],
                    "is_lowest_untracked": bool(
                        lowest is not None and r["index"] == lowest["index"]
                    ),
                    "state_dim": int(vec.size),
                }
                for i_c in range(vec.size):
                    row[f"vec_re_{i_c}"] = float(np.real(vec[i_c]))
                    row[f"vec_im_{i_c}"] = float(np.imag(vec[i_c]))
                vec_rows.append(row)

    if V_orphan:
        ax_omega.plot(
            V_orphan,
            om_orphan,
            "o-",
            color="crimson",
            linewidth=2.2,
            markersize=4,
            label="lowest untracked root",
            zorder=5,
        )
        ax_g.plot(
            V_orphan,
            g_orphan,
            "o-",
            color="crimson",
            linewidth=2.2,
            markersize=4,
            label="lowest untracked root",
            zorder=5,
        )

    cmap = plt.cm.plasma
    for bi, (idx, data) in enumerate(sorted(untracked_branches.items())):
        if len(data["V"]) == 0:
            continue
        color = cmap(0.2 + 0.6 * bi / max(len(untracked_branches), 1))
        ax_omega.plot(
            data["V"],
            data["omega"],
            "s--",
            color=color,
            linewidth=1.0,
            markersize=3,
            alpha=0.85,
            label=f"untracked idx {idx}",
        )
        ax_g.plot(
            data["V"],
            data["gamma"],
            "s--",
            color=color,
            linewidth=1.0,
            markersize=3,
            alpha=0.85,
        )

    ax_g.axhline(0.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax_omega.set_ylabel(r"$\omega = |\mathrm{Im}(\lambda)|$ [rad/s]")
    ax_g.set_ylabel(r"$g = \sigma/\omega$ [–]")
    ax_g.set_xlabel("Airspeed V [m/s]")
    ax_omega.legend(fontsize=8, loc="best", ncol=2)
    ax_g.legend(fontsize=8, loc="best", ncol=2)
    ax_omega.grid(True, alpha=0.3)
    ax_g.grid(True, alpha=0.3)

    case_name = config.name if config and hasattr(config, "name") else "Flutter Analysis"
    fig.suptitle(
        f"{case_name} — untracked wet roots (not selected by mode matching)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    outfile = os.path.join(save_path, "pk_untracked_wet_roots.png") if save_path else "pk_untracked_wet_roots.png"
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    print(f"Untracked wet-root plot saved to {outfile}")

    if csv_rows:
        import pandas as pd

        csv_path = os.path.join(save_path, "pk_untracked_wet_roots.csv") if save_path else "pk_untracked_wet_roots.csv"
        pd.DataFrame(csv_rows).to_csv(csv_path, index=False, float_format="%.6e")
        print(f"Untracked wet-root data saved to {csv_path}")

    if vec_rows:
        import pandas as pd

        vec_path = (
            os.path.join(save_path, "pk_untracked_wet_roots_vectors.csv")
            if save_path
            else "pk_untracked_wet_roots_vectors.csv"
        )
        pd.DataFrame(vec_rows).to_csv(vec_path, index=False, float_format="%.6e")
        print(
            f"Untracked wet-root eigenvectors saved to {vec_path} "
            f"({len(vec_rows)} roots, complex components as vec_re_i / vec_im_i)"
        )
    else:
        print(
            "  [note] No eigenvectors attached to wet roots — re-run the flutter "
            "analysis with the updated pk_method to populate 'vec' per root."
        )

    if show_plot:
        plt.show()
    else:
        plt.close()


def build_cross_velocity_mac_records(
    velocities,
    vecs_by_velocity,
    mode_indices,
    *,
    n_structural_dof=None,
    iteration=0,
):
    """
    Build converged MAC records from eigenvector snapshots at each airspeed.

    Used by Roger RFA / RFA+Capytaine (one solve per V). PK reuses the same
    cross-velocity MAC and adds ``mac_first_iter`` in ``pk_method``.
    """
    from .roger_mode_tracking import compute_mac

    records = []
    mode_indices = [int(m) for m in mode_indices]
    for i_v, V in enumerate(velocities):
        vecs_V = vecs_by_velocity[i_v]
        if vecs_V is None or not hasattr(vecs_V, "shape") or vecs_V.ndim != 2:
            continue
        for j_mode, mode_idx in enumerate(mode_indices):
            if j_mode >= vecs_V.shape[1]:
                continue
            u_sel = vecs_V[:, j_mode]
            mac_prev = float("nan")
            if i_v > 0:
                vecs_prev = vecs_by_velocity[i_v - 1]
                if (
                    vecs_prev is not None
                    and hasattr(vecs_prev, "shape")
                    and vecs_prev.ndim == 2
                    and j_mode < vecs_prev.shape[1]
                ):
                    mac_prev = float(
                        compute_mac(
                            u_sel,
                            vecs_prev[:, j_mode],
                            n_structural_dof=n_structural_dof,
                        )
                    )
            records.append(
                {
                    "V": float(V),
                    "mode": int(mode_idx),
                    "iteration": int(iteration),
                    "mac_prev_velocity": mac_prev,
                    "mac_first_iter": float("nan"),
                }
            )
    return records


def _collect_pk_converged_mac_records(flutter_results):
    """Converged MAC: vs previous velocity (+ vs first PK iter when available)."""
    pk_solver = getattr(flutter_results, "pk_solver", None)
    hist = getattr(pk_solver, "mac_convergence_history", None) if pk_solver is not None else None
    if hist:
        return list(hist)

    raw_results = getattr(flutter_results, "raw_results", None)
    if not raw_results:
        return []

    records = []
    for entry in raw_results:
        V = float(entry.get("V", np.nan))
        for md in entry.get("modes", []):
            if not md.get("converged", False):
                continue
            records.append(
                {
                    "V": V,
                    "mode": int(md.get("mode", 0)),
                    "iteration": int(md.get("iterations", 0)) - 1,
                    "mac_prev_velocity": md.get("mac_prev_velocity", np.nan),
                    "mac_first_iter": md.get("mac_first_iter", np.nan),
                }
            )
    return records


def plot_pk_converged_mac_history(flutter_results, config=None, save_path="", show_plot=False):
    """
    Plot converged MAC values from PK, Roger RFA, or RFA+Capytaine sweeps.

    Top panel: MAC vs eigenvector at the previous airspeed (all methods).
    Bottom panel: MAC vs first PK iteration at the same airspeed (PK only).
    """
    print("\n--- Generating Converged MAC Plot ---")

    records = _collect_pk_converged_mac_records(flutter_results)
    if not records:
        print("  Skipped: no converged MAC history (re-run flutter analysis).")
        return

    has_intra_pk = any(np.isfinite(r.get("mac_first_iter", np.nan)) for r in records)

    if not save_path:
        if config and getattr(config, "output_dir", None):
            save_path = config.output_dir
        else:
            save_path = "output_data"
    os.makedirs(save_path, exist_ok=True)

    v_scale = 1.94384 if config and getattr(config, "v_knots", False) else 1.0
    v_label = "V [kn]" if v_scale != 1.0 else "V [m/s]"
    case_name = getattr(config, "case_name", "case") if config else "case"

    modes_tracked = sorted({int(r["mode"]) for r in records})
    cmap = plt.cm.tab10
    mode_color = {s: cmap(i % 10) for i, s in enumerate(modes_tracked)}

    if has_intra_pk:
        fig, (ax_cross, ax_intra) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    else:
        fig, ax_cross = plt.subplots(1, 1, figsize=(10, 4.5))
        ax_intra = None

    csv_rows = []
    for s in modes_tracked:
        subset = [r for r in records if int(r["mode"]) == s]
        subset.sort(key=lambda r: r["V"])
        V = np.array([r["V"] for r in subset], dtype=float) * v_scale
        mac_cross = np.array([r["mac_prev_velocity"] for r in subset], dtype=float)
        mac_intra = np.array([r["mac_first_iter"] for r in subset], dtype=float)

        ax_cross.plot(
            V, mac_cross, "o-", ms=5, lw=1.6, color=mode_color[s], label=f"mode {s}"
        )
        if ax_intra is not None:
            ax_intra.plot(
                V, mac_intra, "s-", ms=5, lw=1.6, color=mode_color[s], label=f"mode {s}"
            )

        for r in subset:
            csv_rows.append(
                {
                    "Velocity_m_s": float(r["V"]),
                    "Mode": int(s),
                    "Iteration_at_convergence": int(r["iteration"]),
                    "MAC_vs_previous_velocity": float(r["mac_prev_velocity"]),
                    "MAC_vs_first_iteration": float(r["mac_first_iter"]),
                }
            )

    def _style_mac_ax(ax, title):
        ax.axhline(1.0, color="k", lw=0.6, alpha=0.35)
        ax.axhline(0.7, color="tab:orange", ls="--", lw=0.8, alpha=0.6, label="MAC = 0.7")
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("MAC")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower left", fontsize=8, ncol=min(4, len(modes_tracked)))

    _style_mac_ax(ax_cross, "MAC vs converged eigenvector at previous airspeed")
    if ax_intra is not None:
        _style_mac_ax(
            ax_intra,
            "MAC vs eigenvector at first PK iteration (same airspeed)",
        )
        ax_intra.set_xlabel(v_label)
    else:
        ax_cross.set_xlabel(v_label)

    fig.suptitle(
        f"{case_name} — converged wet eigenvector MAC",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()

    for png_name in ("converged_mac.png", "pk_converged_mac.png"):
        png_path = os.path.join(save_path, png_name)
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        print(f"Converged MAC plot saved to {png_path}")

    import pandas as pd

    for csv_name in ("converged_mac.csv", "pk_converged_mac.csv"):
        csv_path = os.path.join(save_path, csv_name)
        pd.DataFrame(csv_rows).to_csv(csv_path, index=False, float_format="%.6e")
        print(f"Converged MAC data saved to {csv_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def flutter_cp_snapshot_indices(n_velocities, n_snapshots=5):
    """Indices for Cp snapshots: first, last, and evenly spaced interior points."""
    n = int(n_velocities)
    if n <= 0:
        return []
    ns = max(1, int(n_snapshots))
    if n <= ns:
        return list(range(n))
    return sorted({int(round(x)) for x in np.linspace(0, n - 1, ns)})


def converged_k_from_modes(modes, V, mode_slot=0):
    """PanelAero k_pa = ω/V from PK converged k or RFA tracked ω."""
    if not modes or mode_slot >= len(modes):
        return float("nan")
    md = modes[mode_slot]
    k = md.get("k", np.nan)
    if np.isfinite(k):
        return float(k)
    omega = md.get("omega", np.nan)
    if np.isfinite(omega) and float(V) > 1e-12:
        return float(omega) / float(V)
    return float("nan")


def _save_cp_aerogrid_plot(aerogrid, scalars, title, outfile, colormap="RdBu_r"):
    """3-D panel ΔCp map; symmetric colour limits (flat field when |ΔCp|≈0)."""
    points = aerogrid["cornerpoint_grids"][:, 1:4]
    panels = []
    for panel in aerogrid["cornerpoint_panels"]:
        panel_points = [
            points[np.where(aerogrid["cornerpoint_grids"][:, 0] == pid)[0][0]]
            for pid in panel
        ]
        panels.append(panel_points)

    scalars = np.asarray(scalars, dtype=float)
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111, projection="3d")
    poly3d = Poly3DCollection(panels, edgecolor="black", linewidth=0.12)

    vmax = float(np.max(np.abs(scalars)))
    if vmax < 1e-15:
        vmax = 1.0
    norm = plt.Normalize(vmin=-vmax, vmax=vmax)
    cmap = plt.get_cmap(colormap)
    poly3d.set_facecolor(cmap(norm(scalars)))
    ax.add_collection3d(poly3d)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, shrink=0.6)

    mid = (points.max(axis=0) + points.min(axis=0)) * 0.5
    rng = float((points.max(axis=0) - points.min(axis=0)).max())
    ax.set_xlim(mid[0] - rng * 0.5, mid[0] + rng * 0.5)
    ax.set_ylim(mid[1] - rng * 0.5, mid[1] + rng * 0.5)
    ax.set_zlim(mid[2] - rng * 0.5, mid[2] + rng * 0.5)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=40, azim=-120)
    ax.set_title(title)
    fig.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_modal_panel_downwash(Z, Z_qs, phi, k):
    """
    Panel downwash for harmonic modal motion (same chain as Q_modal in flutter_solver).

    w_j = (i k Z + Z_qs) @ φ_j   with  k = k_pa = ω/V  (PanelAero convention).
    """
    phi = np.asarray(phi, dtype=np.complex128).ravel()
    Z = np.asarray(Z, dtype=np.complex128)
    Z_qs = np.asarray(Z_qs, dtype=np.complex128)
    return (1j * float(k) * Z + Z_qs) @ phi


def converged_mode_index(modes, mode_slot=0, fallback=0):
    if not modes or mode_slot >= len(modes):
        return int(fallback)
    return int(modes[mode_slot].get("mode", mode_slot))


class CpSweepSnapshotPlotter:
    """
    Plot DLM ΔCp = Qjj @ wj at selected velocity steps.

    - unit: wj = 1 (kernel probe)
    - modal: wj = (ik·Z + Z_qs) @ φ_j  (flutter mode shape)
    """

    def __init__(self, config, aerogrid, V_list, structural_results=None, coupling_results=None):
        self.config = config
        self.aerogrid = aerogrid
        n_vel = len(V_list)
        n_snap = int(getattr(config, "plot_cp_sweep_n_snapshots", 5))
        self.snapshot_indices = set(flutter_cp_snapshot_indices(n_vel, n_snap))
        self.mode_slot = int(getattr(config, "plot_cp_sweep_mode_slot", 0))
        self._plotted = set()

        self.Z = None
        self.Z_qs = None
        self.Phi = None
        if coupling_results is not None and structural_results is not None:
            self.Z = coupling_results.get("Z")
            self.Z_qs = coupling_results.get("Z_qs")
            n_modes = int(len(np.asarray(structural_results.dry_eigenvalues).ravel()))
            self.Phi = np.asarray(structural_results.dry_eigenvectors)[:, :n_modes]

        if self.snapshot_indices:
            kinds = ["unit"]
            if getattr(config, "plot_cp_sweep_modal_shape", True) and self.Z is not None:
                kinds.append("modal")
            print(
                f"\n[Cp snapshots] {len(self.snapshot_indices)} steps scheduled at "
                f"V-indices {sorted(self.snapshot_indices)} / {n_vel}  ({', '.join(kinds)})"
            )

    def maybe_plot(self, V, i_V, modes):
        if i_V not in self.snapshot_indices or i_V in self._plotted:
            return
        if self.aerogrid is None:
            return
        if not getattr(self.config, "plot_cp_sweep_snapshots", True):
            return
        if not getattr(self.config, "save_plots", True):
            return

        k = converged_k_from_modes(modes, V, self.mode_slot)
        if not np.isfinite(k) or k <= 0.0:
            print(f"  [Cp snapshot] skip step {i_V + 1}: no converged k at V={V:.2f} m/s")
            return

        self._plotted.add(i_V)
        step_label = f"step{i_V + 1:03d}"

        plot_cp_at_V_k(
            self.config,
            self.aerogrid,
            float(V),
            float(k),
            step_label=step_label,
            wj_mode="unit",
        )

        if (
            getattr(self.config, "plot_cp_sweep_modal_shape", True)
            and self.Z is not None
            and self.Z_qs is not None
            and self.Phi is not None
        ):
            mode_idx = converged_mode_index(modes, self.mode_slot)
            phi = self.Phi[:, mode_idx]
            plot_cp_at_V_k(
                self.config,
                self.aerogrid,
                float(V),
                float(k),
                step_label=f"{step_label}_modal_m{mode_idx}",
                wj_mode="modal",
                Z=self.Z,
                Z_qs=self.Z_qs,
                phi=phi,
            )


def plot_cp_at_V_k(
    config,
    aerogrid,
    V,
    k,
    *,
    step_label="",
    wj_mode="unit",
    Z=None,
    Z_qs=None,
    phi=None,
):
    """
    ΔCp = Qjj(k, Ma) @ wj.

    wj modes:
      unit      — ones(n)  (DLM kernel probe)
      incidence — sin(α)  (PanelAero tutorial)
      modal     — (i k Z + Z_qs) @ φ  (flutter mode downwash)
    """
    from Qjj.precompute_qjj import interp_qjj_from_disk_old

    n = int(aerogrid["n"])
    c_sound = config.c_sound[config.fluid]
    Ma = float(V) / float(c_sound)

    Qjj = interp_qjj_from_disk_old(
        config.qjj_dir,
        float(k),
        Ma,
        ma_interp_mode=getattr(config, "qjj_ma_interp_mode", "auto"),
        ma_low_threshold=float(getattr(config, "qjj_ma_low_threshold", 0.05)),
    )

    if wj_mode == "incidence":
        alpha_deg = float(getattr(config, "alpha_deg", 0.0))
        wj = np.ones(n, dtype=np.complex128) * np.sin(np.deg2rad(alpha_deg))
    elif wj_mode == "modal":
        if Z is None or Z_qs is None or phi is None:
            raise ValueError("modal wj requires Z, Z_qs, and phi")
        wj = build_modal_panel_downwash(Z, Z_qs, phi, k)
    else:
        wj = np.ones(n, dtype=np.complex128)

    cp = Qjj.dot(wj)
    out_dir = getattr(config, "output_dir", ".")
    cp_dir = os.path.join(
        out_dir,
        getattr(config, "cp_aerogrid_subdir", "cp_on_aerogrid"),
    )
    os.makedirs(cp_dir, exist_ok=True)

    tag = f"{step_label}_V{V:g}_k{k:g}" if step_label else f"V{V:g}_k{k:g}"
    wj_tag = {"unit": "wj=1", "modal": "wj=(ikZ+Zqs)φ", "incidence": "wj=sinα"}.get(wj_mode, wj_mode)
    print(
        f"  [Cp snapshot] {tag} ({wj_tag}): "
        f"Re(ΔCp) [{cp.real.min():.3e}, {cp.real.max():.3e}], "
        f"Im(ΔCp) [{cp.imag.min():.3e}, {cp.imag.max():.3e}], "
        f"|wj|_max={np.max(np.abs(wj)):.3e}"
    )

    _save_cp_aerogrid_plot(
        aerogrid,
        cp.real,
        f"Re(ΔCp) {wj_tag}  {tag}",
        os.path.join(cp_dir, f"cp_real_flutter_{tag}.png"),
    )
    _save_cp_aerogrid_plot(
        aerogrid,
        cp.imag,
        f"Im(ΔCp) {wj_tag}  {tag}",
        os.path.join(cp_dir, f"cp_imag_flutter_{tag}.png"),
    )


def plot_cp_at_velocities(config, aerogrid, results):

    import sys
    sys.path.append(config.paths['FSI'] + '/panelaero_utl')
    from precompute_qjj import interp_qjj_from_disk_old
    
    # Select velocity indices: first, middle, last
    n_speeds = len(results)
    if n_speeds == 0:
        return
    
    indices = [0]  # First speed
    if n_speeds > 2:
        indices.append(n_speeds // 2)  # Middle speed
    if n_speeds > 1:
        indices.append(n_speeds - 1)  # Last speed
    
    print("\n--- Plotting Cp at selected velocities ---")
    
    # Create output directory if saving
    if config.save_plots:
        output_dir = config.output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    b = config.chord / 2
    c_sound = config.c_sound[config.fluid]
    
    # Create a model-like structure for DetailedPlots
    class ModelWrapper:
        def __init__(self, aerogrid):
            self.aerogrid = aerogrid
    
    plotter = DetailedPlots(ModelWrapper(aerogrid))
    
    for idx in indices:
        V = results[idx]['V']
        modes_data = results[idx]['modes']
        
        # Use the first mode's frequency for k calculation
        if len(modes_data) > 0:
            omega = modes_data[0]['omega']
            k = omega * b / V
            k_dlm = k / b
            Ma_dlm = V / c_sound
            
            # Get Qjj at this velocity and reduced frequency
            Qjj = interp_qjj_from_disk_old(config.qjj_dir, k_dlm, Ma_dlm)
            
            # Calculate Cp
            cp = Qjj.dot(np.ones(aerogrid['n']))
            
            print(f"  Plotting Cp at V = {V:.2f} m/s (k = {k:.4f}, Ma = {Ma_dlm:.4f})")
            
            # Plot real part
            title_real = f'Cp Real - V = {V:.2f} m/s'
            outfile_real = os.path.join(config.output_dir, f"cp_real_V{V:.2f}.png") if config.save_plots else None
            plotter.plot_aerogrid(title_real, scalars=cp.real, colormap='RdBu_r', outfile=outfile_real)
            
            # Plot imaginary part
            title_imag = f'Cp Imaginary - V = {V:.2f} m/s'
            outfile_imag = os.path.join(config.output_dir, f"cp_imag_V{V:.2f}.png") if config.save_plots else None
            plotter.plot_aerogrid(title_imag, scalars=cp.imag, colormap='RdBu_r', outfile=outfile_imag)


def plot_cp_at_fixed_k_values(config, aerogrid, results, k_values=[0.001, 0.5, 1.0]):

    import sys
    sys.path.append(config.paths['FSI'] + '/panelaero_utl')
    from Qjj.precompute_qjj import interp_qjj_from_disk_old
    
    print("\n--- Plotting Cp at fixed k values for all modes ---")
    
    # Create output directory
    if config.save_plots:
        output_dir = config.output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    c_sound = config.c_sound[config.fluid]
    
    # Find how many modes we're tracking (excluding lost modes)
    n_modes = len(results[0]['modes'])
    
    # Create a model-like structure for DetailedPlots
    class ModelWrapper:
        def __init__(self, aerogrid):
            self.aerogrid = aerogrid
    
    plotter = DetailedPlots(ModelWrapper(aerogrid))
    
    # For each k value
    for k in k_values:
        print(f"\n  Processing k = {k}")
        
        # Create figure with 2 rows x n_modes columns
        fig, axes = plt.subplots(2, n_modes, figsize=(6*n_modes, 10))
        if n_modes == 1:
            axes = axes.reshape(2, 1)
        
        # Use a reference velocity to compute Ma (use first non-zero velocity)
        V_ref = results[1]['V'] if len(results) > 1 else results[0]['V']
        Ma_dlm = V_ref / c_sound
        k_dlm = k
        
        # Get Qjj at this k and Mach number
        try:
            Qjj = interp_qjj_from_disk_old(config.qjj_dir, k_dlm, Ma_dlm)
        except Exception as e:
            print(f"    Warning: Could not load Qjj for k={k}, Ma={Ma_dlm:.4f}: {e}")
            plt.close(fig)
            continue
        
        # Plot for each mode
        for mode_idx in range(n_modes):
            # Calculate Cp for this mode (using unit modal displacement)
            cp = Qjj.dot(np.ones(aerogrid['n']))
            
            # Get aerogrid coordinates from centerpoint locations
            # Extract control points (offset_j is the 75% chord downwash control point)
            if 'offset_j' in aerogrid:
                x_coords = aerogrid['offset_j'][:, 0]  # Chord direction
                y_coords = aerogrid['offset_j'][:, 1]  # Span direction
                z_coords = aerogrid['offset_j'][:, 2]  # Vertical direction
            else:
                # Fallback: use cornerpoint_grids if offset_j not available
                print("    Warning: offset_j not found, attempting to extract from cornerpoint_grids")
                continue
            
            # Plot imaginary part (top row)
            ax_imag = axes[0, mode_idx]
            scatter_imag = ax_imag.scatter(y_coords, x_coords, c=cp.imag, 
                                          cmap='RdBu_r', s=50)
            ax_imag.set_xlabel('Span [m]')
            ax_imag.set_ylabel('Chord [m]')
            ax_imag.set_title(f'Mode {mode_idx} - Cp Imag')
            ax_imag.set_aspect('equal')
            plt.colorbar(scatter_imag, ax=ax_imag, label='Cp Imaginary')
            
            # Plot real part (bottom row)
            ax_real = axes[1, mode_idx]
            scatter_real = ax_real.scatter(y_coords, x_coords, c=cp.real, 
                                          cmap='RdBu_r', s=50)
            ax_real.set_xlabel('Span [m]')
            ax_real.set_ylabel('Chord [m]')
            ax_real.set_title(f'Mode {mode_idx} - Cp Real')
            ax_real.set_aspect('equal')
            plt.colorbar(scatter_real, ax=ax_real, label='Cp Real')
        
        # Add overall title
        fig.suptitle(f'Pressure Coefficient Distribution at k = {k:.3f} (Ma = {Ma_dlm:.4f})', 
                     fontsize=16, fontweight='bold')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        
        # Save figure
        if config.save_plots:
            outfile = os.path.join(config.output_dir, f"cp_k{k:.3f}_allmodes.png")
            plt.savefig(outfile, dpi=300, bbox_inches='tight')
            print(f"    Saved plot to {outfile}")
        
        plt.close(fig)
    
    print("  CP plotting at fixed k values completed")


def _resolve_aero_coupling_kind(config):
    """Return ``'DLM'`` or ``'UVLM'`` for aero-beam schematic overlays.

    Panel control/force points and the spline matrix **Z** apply only to the DLM
    runtime path.  UVLM uses a precomputed modal GAF ``Q(k)`` (no panel Z at PK).
    """
    if config is None:
        return "DLM"
    override = getattr(config, "aero_beam_coupling_method", None)
    if override is not None:
        key = str(override).upper()
        if key == "UVLM":
            return "UVLM"
        if key in ("DLM", "PANELAERO", "PANELAERO_DLM"):
            return "DLM"
    method = str(getattr(config, "aero_method", "DLM")).upper()
    return "UVLM" if method == "UVLM" else "DLM"


def _beam_on_shear_center(config, beam_model=None):
    if beam_model and beam_model.get("geometry"):
        v = beam_model["geometry"].get("beam_on_shear_center")
        if v is not None:
            return bool(v)
    if config is None:
        return False
    from FEA.fea_utl.beam_geometry import beam_on_shear_center
    return beam_on_shear_center(config)


def _cg_offset_m(config, beam_model):
    """CG offset e [m] from shear center along local chord, if available."""
    if beam_model:
        geo = beam_model.get("geometry") or {}
        e = geo.get("cg_offset_e")
        if e is not None:
            return float(e)
    if config is None:
        return None
    xcm = getattr(config, "xcm_factor", None)
    xea = getattr(config, "xea_factor", None)
    chord = getattr(config, "chord", None)
    if xcm is None or xea is None or chord is None:
        return None
    return float(xcm - xea) * float(chord)


def _planform_chordwise_unit(config, beam_model):
    """Unit chordwise direction in the reference frame (for CG offset from EA)."""
    sweep_deg = 0.0
    if beam_model and beam_model.get("geometry"):
        sweep_deg = float(beam_model["geometry"].get("sweep_angle_deg", 0.0))
    elif config is not None:
        sweep_deg = float(getattr(config, "sweep_angle_deg", 0.0))
    sweep = np.deg2rad(sweep_deg)
    if abs(sweep) < 1e-12:
        return np.array([1.0, 0.0, 0.0])
    span_dir = np.array([np.tan(sweep), 1.0, 0.0])
    span_dir /= np.linalg.norm(span_dir)
    chord_dir = np.array([-span_dir[1], span_dir[0], 0.0])
    return chord_dir / np.linalg.norm(chord_dir)


def _local_chord_dirs_at_nodes(beam_model, config):
    """Unit chordwise (+u) direction at each node from element ``T6`` (sweep / dihedral aware)."""
    nodes = [n["position"] for n in beam_model["nodes"]]
    n_nodes = len(nodes)
    chord_dirs = np.zeros((n_nodes, 3), dtype=float)
    counts = np.zeros(n_nodes, dtype=int)

    for elem in beam_model.get("elements", []):
        T6 = elem.get("T6")
        if T6 is None:
            continue
        u_dir = np.asarray(T6, dtype=float)[:3, 0]
        un = np.linalg.norm(u_dir)
        if un < 1e-14:
            continue
        u_dir /= un
        for n_idx in elem["nodes"]:
            chord_dirs[n_idx] += u_dir
            counts[n_idx] += 1

    fallback = _planform_chordwise_unit(config, beam_model)
    for i in range(n_nodes):
        if counts[i] > 0:
            v = chord_dirs[i] / counts[i]
            vn = np.linalg.norm(v)
            chord_dirs[i] = v / vn if vn > 1e-14 else fallback
        elif i > 0:
            chord_dirs[i] = chord_dirs[i - 1]
        else:
            chord_dirs[i] = fallback
    return chord_dirs


def _cog_nodes_from_beam(nodes, e, config, beam_model):
    """Build CG axis node coordinates from beam nodes (on SC) and chordwise offset e."""
    nodes = np.asarray(nodes, dtype=float)
    if beam_model and beam_model.get("elements"):
        chord_dirs = _local_chord_dirs_at_nodes(beam_model, config)
        return nodes + float(e) * chord_dirs
    chord_u = _planform_chordwise_unit(config, beam_model)
    return nodes + float(e) * chord_u


def _point_mass_cg_positions(beam_model):
    """CG positions [m] for concentrated masses in ``beam_model['point_masses']``."""
    out = []
    if not beam_model or not beam_model.get("point_masses"):
        return out
    nodes = [np.asarray(n["position"], dtype=float) for n in beam_model["nodes"]]
    for pm in beam_model["point_masses"]:
        ni = int(pm["node_index"])
        offset = np.asarray(pm.get("offset", (0.0, 0.0, 0.0)), dtype=float)
        cg = nodes[ni] + offset
        out.append({
            "cg": cg,
            "node": nodes[ni],
            "mass": float(pm.get("mass", 0.0)),
            "node_index": ni,
        })
    return out


def _draw_point_masses(ax3d, ax_fr, ax_si, ax_tp, beam_model, config=None):
    """Overlay concentrated mass CG locations (e.g. Model2t pod)."""
    masses = _point_mass_cg_positions(beam_model)
    if not masses:
        return
    for i, pm in enumerate(masses):
        cg = pm["cg"]
        node = pm["node"]
        mass = pm["mass"]
        label = f"Pod mass CG ({mass:.1f} kg)" if i == 0 else "_nolegend_"
        ax3d.scatter([cg[0]], [cg[1]], [cg[2]], c="gold", marker="*", s=220,
                       edgecolors="k", linewidths=0.6, label=label, zorder=12)
        ax_fr.scatter([cg[0]], [cg[2]], c="gold", marker="*", s=220,
                      edgecolors="k", linewidths=0.6, zorder=12)
        ax_si.scatter([cg[1]], [cg[2]], c="gold", marker="*", s=220,
                      edgecolors="k", linewidths=0.6, zorder=12)
        ax_tp.scatter([cg[1]], [cg[0]], c="gold", marker="*", s=220,
                      edgecolors="k", linewidths=0.6, zorder=12)
        # Dashed link: beam attachment node → mass CG (offset vector)
        seg = np.vstack([node, cg])
        ax3d.plot(seg[:, 0], seg[:, 1], seg[:, 2], color="goldenrod", ls=":", lw=1.8, alpha=0.9)
        ax_fr.plot(seg[:, 0], seg[:, 2], color="goldenrod", ls=":", lw=1.8, alpha=0.9)
        ax_si.plot(seg[:, 1], seg[:, 2], color="goldenrod", ls=":", lw=1.8, alpha=0.9)
        ax_tp.plot(seg[:, 1], seg[:, 0], color="goldenrod", ls=":", lw=1.8, alpha=0.9)
    if config is not None:
        case = getattr(config, "name", "")
        print(f"\nplot_aero_beam_model: {len(masses)} point mass(es) drawn ({case or 'case'})")
        for pm in masses:
            print(f"  node {pm['node_index']}: CG={pm['cg']}, m={pm['mass']:.3f} kg")


def plot_aero_beam_model(aerogrid, beam_model, config=None):

    paper_style = bool(getattr(config, "paper_figure_style", False)) if config else False
    single_view = getattr(config, "aero_beam_view", "all") if config else "all"
    outfile_name = getattr(config, "aero_beam_outfile", "aero_beam_model.png") if config else "aero_beam_model.png"
    front_only = paper_style and single_view == "front"
    coupling_kind = _resolve_aero_coupling_kind(config)
    show_panel_pts = getattr(config, "aero_beam_show_panel_points", None)
    if show_panel_pts is None:
        show_panel_pts = coupling_kind == "DLM"
    show_coupling_note = bool(getattr(config, "aero_beam_show_coupling_note", paper_style))

    LABEL_PT = LEGEND_PT = None
    if paper_style:
        import sys
        from pathlib import Path as _Path

        sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "docs" / "paper"))
        from paper_figure_style import (
            LABEL_PT,
            LEGEND_PT,
            LINEWIDTH_IN,
            apply_paper_style,
            style_body_text_panel,
        )

    if front_only:
        from unittest.mock import MagicMock

        apply_paper_style(single_column=True)
        fig = plt.figure(figsize=(LINEWIDTH_IN, LINEWIDTH_IN * 1.05))
        ax_front = fig.add_subplot(111)
        ax_iso = ax_side = ax_top = MagicMock()
    else:
        fig = plt.figure(figsize=(24, 16))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3, width_ratios=[1.5, 1, 1])
        ax_iso   = fig.add_subplot(gs[:, 0], projection='3d')
        ax_front = fig.add_subplot(gs[0, 1])
        ax_side  = fig.add_subplot(gs[0, 2])
        ax_top   = fig.add_subplot(gs[1, 1:])

    # ── helper to compute panel normal ───────────────────────────────────────
    def _compute_panel_normal(coords):
        """Compute outward normal for a quadrilateral panel using cross product."""
        if len(coords) < 3:
            return None
        # Use first three points to compute normal
        v1 = coords[1] - coords[0]
        v2 = coords[2] - coords[0]
        normal = np.cross(v1, v2)
        norm_length = np.linalg.norm(normal)
        if norm_length > 1e-12:
            return normal / norm_length
        return None

    # ── helpers ──────────────────────────────────────────────────────────────
    def _draw_aero(ax3d, ax_fr, ax_si, ax_tp,
                   grid_pts, panels, grid_ids, normal_scale=0.08):
        if ax3d is not None:
            ax3d.scatter(grid_pts[:, 0], grid_pts[:, 1], grid_pts[:, 2],
                         s=10, c='b', marker='o', label='Aero Grid Points')

        for panel in panels:
            coords = []
            for pid in panel:
                idx = np.where(grid_ids == pid)[0]
                if idx.size > 0:
                    coords.append(grid_pts[idx[0]])
            if len(coords) >= 3:
                c = np.vstack(coords + [coords[0]])
                if ax3d is not None:
                    ax3d.plot(c[:, 0], c[:, 1], c[:, 2], color='lightblue', linewidth=0.7)
                if ax_fr is not None:
                    ax_fr.plot(c[:, 0], c[:, 2], color='lightblue', linewidth=0.7)
                if ax_si is not None:
                    ax_si.plot(c[:, 1], c[:, 2], color='lightblue', linewidth=0.7)
                if ax_tp is not None:
                    ax_tp.plot(c[:, 1], c[:, 0], color='lightblue', linewidth=0.7)
                
                normal = _compute_panel_normal(np.array(coords))
                if normal is not None and ax3d is not None:
                    panel_center = np.mean(coords, axis=0)
                    if ax3d is not None:
                        ax3d.quiver(panel_center[0], panel_center[1], panel_center[2],
                                   normal[0], normal[1], normal[2],
                                   color='orange', arrow_length_ratio=0.4, linewidth=0.5, alpha=0.6, length=normal_scale)
                    if ax_fr is not None:
                        ax_fr.arrow(panel_center[0], panel_center[2],
                                   normal[0]*normal_scale, normal[2]*normal_scale,
                                   head_width=0.02, head_length=0.02, fc='orange', ec='orange', alpha=0.6, linewidth=0.5)
                    if ax_si is not None:
                        ax_si.arrow(panel_center[1], panel_center[2],
                                   normal[1]*normal_scale, normal[2]*normal_scale,
                                   head_width=0.02, head_length=0.02, fc='orange', ec='orange', alpha=0.6, linewidth=0.5)
                    if ax_tp is not None:
                        ax_tp.arrow(panel_center[1], panel_center[0],
                                   normal[1]*normal_scale, normal[0]*normal_scale,
                                   head_width=0.02, head_length=0.02, fc='orange', ec='orange', alpha=0.6, linewidth=0.5)

    def _draw_ctrl_pts(ax3d, ax_fr, ax_si, ax_tp, pts, color, marker, size, label):
        if ax3d is not None:
            ax3d.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                         c=color, marker=marker, s=size, label=label, alpha=0.8)
        if ax_fr is not None:
            ax_fr.scatter(pts[:, 0], pts[:, 2], c=color, marker=marker, s=size, alpha=0.8)
        if ax_si is not None:
            ax_si.scatter(pts[:, 1], pts[:, 2], c=color, marker=marker, s=size, alpha=0.8)
        if ax_tp is not None:
            ax_tp.scatter(pts[:, 1], pts[:, 0], c=color, marker=marker, s=size, alpha=0.8)

    def _draw_dlm_spline_hints(ax_fr, nodes, ctrl_pts):
        """A few dashed links (front X–Z) from beam to ¾-chord control points."""
        if ax_fr is None or ctrl_pts is None or len(ctrl_pts) == 0 or len(nodes) < 2:
            return
        n_hint = min(3, len(ctrl_pts))
        idxs = np.linspace(0, len(ctrl_pts) - 1, n_hint, dtype=int)
        drawn = False
        for ip in idxs:
            ctrl = ctrl_pts[ip]
            dists = np.linalg.norm(nodes[:, [0, 2]] - ctrl[[0, 2]], axis=1)
            j = int(np.argmin(dists))
            beam_pt = nodes[j]
            ax_fr.plot([beam_pt[0], ctrl[0]], [beam_pt[2], ctrl[2]],
                       "m--", lw=1.1, alpha=0.75, zorder=2,
                       label="Spline Z (downwash)" if not drawn else "_nolegend_")
            drawn = True

    def _draw_coupling_note(ax_fr):
        if ax_fr is None or not show_coupling_note:
            return
        if coupling_kind == "UVLM":
            txt = (
                "UVLM coupling: modal $\\mathbf{Q}(k)$\n"
                "(precomputed; no panel $\\mathbf{Z}$ at PK)"
            )
            fc = "#f3eef8"
        else:
            txt = (
                "DLM coupling: $\\mathbf{Z}$, $\\mathbf{Z}_{\\mathrm{qs}}$, "
                "$\\mathbf{Z}_{\\mathrm{force}}$\n"
                "at $3c/4$ / $c/4$ panel points"
            )
            fc = "#fff8e6"
        ax_fr.text(
            0.02, 0.98, txt, transform=ax_fr.transAxes,
            va="top", ha="left", fontsize=LABEL_PT if paper_style else 7.5,
            bbox=dict(boxstyle="round,pad=0.35", fc=fc, ec="#999999", lw=0.5),
            zorder=20,
        )

    def _draw_beam(ax3d, ax_fr, ax_si, ax_tp, nodes, beam_model, sc_nodes=None, cog_nodes=None,
                   case_name=None, on_shear_center=False, node_groups=None):
        beam_label = 'Beam (SC)' if on_shear_center else 'Beam (EA)'
        # Contiguous node-index groups per sub-body (multibody inverted-T etc.).
        # SC/CG axes are drawn as polylines *within* each group so they are never
        # connected across sub-bodies (e.g. port-foil tip → starboard-foil root).
        if node_groups is None:
            node_groups = [list(range(len(nodes)))]
        elements_plotted = False
        for elem in beam_model['elements']:
            node_indices = elem['nodes']
            if len(node_indices) == 2:
                n1, n2 = node_indices
                ec = np.array([nodes[n1], nodes[n2]])
                kw3d = dict(linewidth=2, label=beam_label if not elements_plotted else '_nolegend_')
                ax3d.plot(ec[:, 0], ec[:, 1], ec[:, 2], 'r-', **kw3d)
                ax_fr.plot(ec[:, 0], ec[:, 2], 'r-', linewidth=2,
                           label=beam_label if not elements_plotted else '_nolegend_')
                ax_si.plot(ec[:, 1], ec[:, 2], 'r-', linewidth=2,
                           label=beam_label if not elements_plotted else '_nolegend_')
                ax_tp.plot(ec[:, 1], ec[:, 0], 'r-', linewidth=2,
                           label=beam_label if not elements_plotted else '_nolegend_')
                elements_plotted = True

        ax3d.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2],
                     c='r', marker='o', s=10, alpha=0.5)
        ax_fr.scatter(nodes[:, 0], nodes[:, 2], c='r', marker='o', s=10, alpha=0.5)
        ax_si.scatter(nodes[:, 1], nodes[:, 2], c='r', marker='o', s=10, alpha=0.5)
        ax_tp.scatter(nodes[:, 1], nodes[:, 0], c='r', marker='o', s=10, alpha=0.5)

        # ── Shear center axis (only when beam is not already on SC) ──────────
        if (
            not on_shear_center
            and sc_nodes is not None
            and len(sc_nodes) > 0
            and (config is None or getattr(config, 'name', None) != 'tnz_multibody')
        ):
            sc_nodes = np.array(sc_nodes)
            _first_sc = True
            for grp in node_groups:
                for k in range(len(grp) - 1):
                    i, j = grp[k], grp[k + 1]
                    sc_elem = np.array([sc_nodes[i], sc_nodes[j]])
                    lab = 'Shear Center' if _first_sc else '_nolegend_'
                    ax3d.plot(sc_elem[:, 0], sc_elem[:, 1], sc_elem[:, 2], 'g--', linewidth=2.5,
                              label=lab, alpha=0.8)
                    ax_fr.plot(sc_elem[:, 0], sc_elem[:, 2], 'g--', linewidth=2.5, label=lab, alpha=0.8)
                    ax_si.plot(sc_elem[:, 1], sc_elem[:, 2], 'g--', linewidth=2.5, label=lab, alpha=0.8)
                    ax_tp.plot(sc_elem[:, 1], sc_elem[:, 0], 'g--', linewidth=2.5, label=lab, alpha=0.8)
                    _first_sc = False

            ax3d.scatter(sc_nodes[:, 0], sc_nodes[:, 1], sc_nodes[:, 2],
                         c='g', marker='^', s=20, alpha=0.8)
            ax_fr.scatter(sc_nodes[:, 0], sc_nodes[:, 2], c='g', marker='^', s=20, alpha=0.8)
            ax_si.scatter(sc_nodes[:, 1], sc_nodes[:, 2], c='g', marker='^', s=20, alpha=0.8)
            ax_tp.scatter(sc_nodes[:, 1], sc_nodes[:, 0], c='g', marker='^', s=20, alpha=0.8)

        # ── Center of gravity axis (when coordinates are available) ──────────
        if cog_nodes is not None and len(cog_nodes) > 0:
            cog_nodes = np.array(cog_nodes)
            _first_cg = True
            for grp in node_groups:
                for k in range(len(grp) - 1):
                    i, j = grp[k], grp[k + 1]
                    cog_elem = np.array([cog_nodes[i], cog_nodes[j]])
                    lab = 'Center of Gravity' if _first_cg else '_nolegend_'
                    ax3d.plot(cog_elem[:, 0], cog_elem[:, 1], cog_elem[:, 2], color='navy',
                              linewidth=2.5, linestyle='-.', label=lab, alpha=0.9)
                    ax_fr.plot(cog_elem[:, 0], cog_elem[:, 2], color='navy', linewidth=2.5,
                               linestyle='-.', label=lab, alpha=0.9)
                    ax_si.plot(cog_elem[:, 1], cog_elem[:, 2], color='navy', linewidth=2.5,
                               linestyle='-.', label=lab, alpha=0.9)
                    ax_tp.plot(cog_elem[:, 1], cog_elem[:, 0], color='navy', linewidth=2.5,
                               linestyle='-.', label=lab, alpha=0.9)
                    _first_cg = False

            ax3d.scatter(cog_nodes[:, 0], cog_nodes[:, 1], cog_nodes[:, 2],
                         c='navy', marker='s', s=15, alpha=0.9)
            ax_fr.scatter(cog_nodes[:, 0], cog_nodes[:, 2], c='navy', marker='s', s=15, alpha=0.9)
            ax_si.scatter(cog_nodes[:, 1], cog_nodes[:, 2], c='navy', marker='s', s=15, alpha=0.9)
            ax_tp.scatter(cog_nodes[:, 1], cog_nodes[:, 0], c='navy', marker='s', s=15, alpha=0.9)

    # ── helper: transparent Capytaine wetted-surface mesh ────────────────────
    def _draw_capytaine_mesh(ax3d, ax_fr, ax_si, ax_tp, mesh_path, alpha=0.10,
                             label='Capytaine mesh', facecolor='steelblue'):
        """Overlay the Capytaine BEM hull (same recipe as run_modal_radiation's
        _plot_mesh_free_surface_debug) at high transparency. Mesh and beam share
        the same frame (e.g. span along -Z for ABRAMSON1965)."""
        try:
            import capytaine as cpt
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            from matplotlib.collections import PolyCollection
        except Exception as e:
            print(f"  Skipping Capytaine mesh overlay (import failed): {e}")
            return
        try:
            mesh = cpt.load_mesh(mesh_path)
        except Exception as e:
            print(f"  Skipping Capytaine mesh overlay (load failed): {e}")
            return

        v = np.asarray(mesh.vertices, dtype=float)
        f = np.asarray(mesh.faces, dtype=np.int64)
        polys = []
        for row in f:
            ids = list(dict.fromkeys(int(i) for i in row))  # drop degenerate (tri stored as quad)
            if len(ids) < 3:
                continue
            polys.append(v[np.asarray(ids, dtype=np.int64)])
        if not polys:
            return

        # 3-D isometric: transparent wetted surface (drawn first → beam/aero on top)
        ax3d.add_collection3d(Poly3DCollection(
            polys, alpha=alpha, facecolor=facecolor,
            edgecolor=facecolor, linewidths=0.1, label=label))

        # 2-D views: high-transparency projected faces (PolyCollection = fast)
        for ax, cols in ((ax_fr, (0, 2)), (ax_si, (1, 2)), (ax_tp, (1, 0))):
            ax.add_collection(PolyCollection(
                [p[:, cols] for p in polys], facecolors=facecolor,
                edgecolors=facecolor, alpha=alpha, linewidths=0.1))
        print(f"  Capytaine mesh overlay: {len(polys)} faces from {os.path.basename(mesh_path)}")

    _cpt_mesh_path = getattr(config, 'mesh_path', None) if config else None
    _strut_mesh = getattr(config, 'capytaine_strut_mesh_path', None) if config else None
    _pod_mesh = getattr(config, 'capytaine_pod_mesh_path', None) if config else None
    if getattr(config, 'plot_capytaine_mesh', True):
        if (_strut_mesh and _pod_mesh
                and os.path.exists(_strut_mesh) and os.path.exists(_pod_mesh)):
            _draw_capytaine_mesh(ax_iso, ax_front, ax_side, ax_top, _strut_mesh,
                                 alpha=0.10, label='Capytaine strut')
            _draw_capytaine_mesh(ax_iso, ax_front, ax_side, ax_top, _pod_mesh,
                                 alpha=0.18, label='Capytaine pod', facecolor='cornflowerblue')
        elif (_cpt_mesh_path and os.path.exists(_cpt_mesh_path)):
            _draw_capytaine_mesh(ax_iso, ax_front, ax_side, ax_top, _cpt_mesh_path)

    ctrl_pts = None

    # ── Plot Aerodynamic Grid ────────────────────────────────────────────────
    if aerogrid:
        grid_pts = aerogrid['cornerpoint_grids'][:, 1:4]
        panels   = aerogrid['cornerpoint_panels']
        grid_ids = aerogrid['cornerpoint_grids'][:, 0]

        _draw_aero(ax_iso, ax_front, ax_side, ax_top, grid_pts, panels, grid_ids)

        ctrl_pts = None

        if show_panel_pts and 'offset_j' in aerogrid:
            oj = np.asarray(aerogrid['offset_j'])
            if oj.size > 0:
                ctrl_pts = oj
                _draw_ctrl_pts(ax_iso, ax_front, ax_side, ax_top,
                               oj, 'magenta', 'x', 40,
                               'Control pts (offset_j @ 75%c)')
        if show_panel_pts and 'offset_l' in aerogrid:
            ol = np.asarray(aerogrid['offset_l'])
            if ol.size > 0:
                _draw_ctrl_pts(ax_iso, ax_front, ax_side, ax_top,
                               ol, 'green', '^', 40,
                               'Force pts (offset_l @ 25%c)')
        elif coupling_kind == "UVLM" and front_only and ax_front is not None:
            # Legend entry: lattice shown for geometry only (no DLM panel operators).
            ax_front.plot([], [], color='lightblue', lw=1.0,
                          label='Aero lattice (geometry)')

    # ── Plot Beam Model ──────────────────────────────────────────────────────
    if beam_model:
        node_coords = [node['position'] for node in beam_model['nodes']]
        nodes = np.array(node_coords)

        # Shear center and CG axes from config / beam_model geometry
        sc_nodes = None
        cog_nodes = None
        case_name = config.name if config else None
        on_sc = _beam_on_shear_center(config, beam_model)

        if not on_sc and config and hasattr(config, 'xea_factor') and hasattr(config, 'chord'):
            # Legacy: beam on midchord; SC at xea_factor from LE
            xea_offset = (config.xea_factor - 0.5) * config.chord
            sc_nodes = np.array([
                [node[0] + xea_offset, node[1], node[2]] for node in nodes
            ])
        e_cg = _cg_offset_m(config, beam_model)
        if e_cg is not None:
            cog_nodes = _cog_nodes_from_beam(nodes, e_cg, config, beam_model)
            ref = "shear center" if on_sc else "beam reference"
            print(
                f"\nplot_aero_beam_model: CG axis offset e={e_cg:.6f} m "
                f"from {ref} ({case_name or 'unknown case'})"
            )
            if len(cog_nodes) > 0:
                print(f"  First beam node: {nodes[0]}; first CG node: {cog_nodes[0]}")
        else:
            print(
                f"\nERROR plot_aero_beam_model: CG axis not available for "
                f"'{case_name or 'unknown case'}' — set xcm_factor, xea_factor, and chord "
                f"on the config, or pass cg_offset_e when building the beam model."
            )

        # Group node indices by sub-body so SC/CG axes are not connected across
        # sub-bodies (multibody inverted-T: strut, foil_port, foil_starboard).
        node_groups = None
        if beam_model.get('is_multibody'):
            groups = {}
            for idx, nd in enumerate(beam_model['nodes']):
                groups.setdefault(nd.get('submodel'), []).append(idx)
            node_groups = list(groups.values())

        _draw_beam(
            ax_iso, ax_front, ax_side, ax_top, nodes, beam_model,
            sc_nodes, cog_nodes, case_name, on_shear_center=on_sc,
            node_groups=node_groups,
        )
        _draw_point_masses(ax_iso, ax_front, ax_side, ax_top, beam_model, config)

    if coupling_kind == "DLM" and show_panel_pts and beam_model and aerogrid:
        node_coords = [node['position'] for node in beam_model['nodes']]
        nodes_arr = np.array(node_coords)
        if ctrl_pts is None and 'offset_j' in aerogrid:
            oj = np.asarray(aerogrid['offset_j'])
            ctrl_pts = oj if oj.size > 0 else None
        if ctrl_pts is not None:
            _draw_dlm_spline_hints(ax_front, nodes_arr, ctrl_pts)

    _draw_coupling_note(ax_front)

    # ── Equal-aspect helper for 3-D ─────────────────────────────────────────
    def _set_equal_3d(ax):
        xlim = ax.get_xlim3d(); ylim = ax.get_ylim3d(); zlim = ax.get_zlim3d()
        ranges = [xlim[1]-xlim[0], ylim[1]-ylim[0], zlim[1]-zlim[0]]
        r = 0.5 * max(ranges)
        ax.set_xlim3d([np.mean(xlim)-r, np.mean(xlim)+r])
        ax.set_ylim3d([np.mean(ylim)-r, np.mean(ylim)+r])
        ax.set_zlim3d([np.mean(zlim)-r, np.mean(zlim)+r])

    def _set_equal_2d(ax):
        ax.set_aspect('equal', adjustable='datalim')

    # ── Labels & formatting ──────────────────────────────────────────────────
    if not front_only:
        ax_iso.set_xlabel('X'); ax_iso.set_ylabel('Y'); ax_iso.set_zlabel('Z')
        if not paper_style:
            ax_iso.set_title('Isometric view', fontsize=12, fontweight='bold')
        ax_iso.legend(fontsize=8, loc='upper left')
        _set_equal_3d(ax_iso)

    ax_front.set_xlabel('X (chord)'); ax_front.set_ylabel('Z (height)')
    if not paper_style:
        ax_front.set_title('Front view  (X–Z)', fontsize=12, fontweight='bold')
    leg_fs = LEGEND_PT if paper_style else 8
    ax_front.legend(fontsize=leg_fs, loc='upper left'); ax_front.grid(True, alpha=0.3)
    _set_equal_2d(ax_front)
    if front_only:
        style_body_text_panel(ax_front)

    if not front_only:
        ax_side.set_xlabel('Y (span)'); ax_side.set_ylabel('Z (height)')
        if not paper_style:
            ax_side.set_title('Side view  (Y–Z)', fontsize=12, fontweight='bold')
        ax_side.legend(fontsize=8, loc='upper left'); ax_side.grid(True, alpha=0.3)
        _set_equal_2d(ax_side)

        ax_top.set_xlabel('Y (span)'); ax_top.set_ylabel('X (chord)')
        if not paper_style:
            ax_top.set_title('Top view', fontsize=12, fontweight='bold')
        ax_top.legend(fontsize=8, loc='upper left'); ax_top.grid(True, alpha=0.3)
        ax_top.invert_yaxis()

    if not paper_style:
        fig.suptitle(
            f'Aerodynamic Grid and Beam Model ({coupling_kind} coupling)',
            fontsize=14, fontweight='bold',
        )

    # ── Save ────────────────────────────────────────────────────────────────
    if config and config.save_plots:
        output_dir = config.output_dir
        os.makedirs(output_dir, exist_ok=True)
        outfile = os.path.join(output_dir, outfile_name)
        dpi = 300 if paper_style else 300
        plt.savefig(outfile, dpi=dpi, bbox_inches='tight')
        print(f"Aero-beam model plot saved to {outfile} ({coupling_kind} coupling)")

    plt.show()


def plot_stiffness_damping_contributions(
    pk_solver=None,
    config=None,
    save_path="",
    show_plot=False,
    flutter_results=None,
):
    """
    Plot the structural and aerodynamic contributions to stiffness and damping
    at each converged velocity step.

    Data source (first match):
      - ``pk_solver.contributions_history`` (PK or RFA-PK via SimpleNamespace)
      - ``flutter_results.dlm_participants_history`` with K_struct / C_aero keys

    Args:
        pk_solver: PKSolverV3 or namespace with ``contributions_history``
        config: Configuration object (optional)
        save_path: Path for saving plots (optional)
        show_plot: If True, display plot; if False, only save CSV and PNG but don't show (default: False)
        flutter_results: optional FlutterResults fallback for RFA history
    """
    print("\n--- Generating Stiffness and Damping Contributions Plot ---")

    contributions = None
    if pk_solver is not None and getattr(pk_solver, "contributions_history", None):
        contributions = pk_solver.contributions_history
    if (not contributions) and flutter_results is not None:
        hist = getattr(flutter_results, "dlm_participants_history", None)
        if hist and isinstance(hist[0], dict) and "K_struct" in hist[0]:
            contributions = hist

    if not contributions:
        print(
            "  Skipped: no contributions history "
            "(enable plot_stiffness_damping_contributions for Roger RFA / RFA-PK)."
        )
        return

    # Determine output directory
    if not save_path:
        if config and getattr(config, "output_dir", None):
            save_path = config.output_dir
        elif config and getattr(config, "save_plots", False):
            save_path = config.output_dir
        else:
            save_path = "output_data"
        os.makedirs(save_path, exist_ok=True)
    
    # Extract data - check if matrices or scalars
    velocities = [c['V'] for c in contributions]
    default_branch_dof = [0, 1]
    if contributions and 'branch_dof_indices' in contributions[0]:
        default_branch_dof = list(contributions[0]['branch_dof_indices'])
    
    # Helper: per-velocity list of 2×2 matrices (one per converged structural slot).
    def extract_entry(contrib_list, key, branch_slot, default=0.0):
        """
        Extract diagonal entry for tracked branch ``branch_slot`` at each velocity.

        Uses ``branch_dof_indices[branch_slot]`` when present so plots follow the
        fixed modal-basis DOF of each structural branch, not MAC pool ordering.
        """
        values = []
        for c in contrib_list:
            if key not in c:
                values.append(float(default))
                continue
            dof_indices = c.get('branch_dof_indices', default_branch_dof)
            dof = int(dof_indices[branch_slot]) if branch_slot < len(dof_indices) else branch_slot
            matrices = c[key]
            if isinstance(matrices, list):
                mat = matrices[branch_slot] if branch_slot < len(matrices) else matrices[0]
            else:
                mat = matrices
            values.append(float(mat[dof, dof]))
        return values

    def extract_coupling_entry(contrib_list, key, branch_i, branch_j, default=0.0):
        values = []
        for c in contrib_list:
            if key not in c:
                values.append(float(default))
                continue
            dof_indices = c.get('branch_dof_indices', default_branch_dof)
            di = int(dof_indices[branch_i]) if branch_i < len(dof_indices) else branch_i
            dj = int(dof_indices[branch_j]) if branch_j < len(dof_indices) else branch_j
            matrices = c[key]
            mat = matrices[0] if isinstance(matrices, list) else matrices
            values.append(float(mat[di, dj]))
        return values

    apply_b2_mass = bool(getattr(config, 'rfa_apply_b2_mass', False)) if config else False

    matrix_components = {
        'M': [
            ('structural', 'M_struct', '+'),
            *(
                [('aerodynamic_B2', 'M_aero', '-')]
                if apply_b2_mass
                else []
            ),
            ('hydro_Capytaine', 'M_hydro', '+'),
            ('effective', 'M_effective', '='),
        ],
        'K': [
            ('structural', 'K_struct', '+'),
            ('aerodynamic_B0', 'K_aero', '-'),
            ('hydro', 'K_hydro', '+'),
            ('effective', 'K_effective', '='),
        ],
        'C': [
            ('structural', 'C_struct', '+'),
            ('aerodynamic_B1', 'C_aero', '-'),
            ('hydro_Capytaine', 'C_hydro', '+'),
            ('empirical', 'C_empirical', '+'),
            ('effective', 'C_effective', '='),
        ],
    }

    entry_indices = [(0, 0), (1, 1), (0, 1), (1, 0)]
    data = {}
    for family, specs in matrix_components.items():
        data[family] = {}
        for label, key, _sign in specs:
            data[family][label] = {}
            for branch_slot in (0, 1):
                entry = (branch_slot, branch_slot)
                data[family][label][entry] = extract_entry(
                    contributions, key, branch_slot
                )
            for i, j in ((0, 1), (1, 0)):
                data[family][label][(i, j)] = extract_coupling_entry(
                    contributions, key, i, j
                )

    def balance_residual(family, entry):
        """Check that effective equals structural - aero + hydro/empirical terms."""
        effective = np.asarray(data[family]['effective'][entry], dtype=float)
        rhs = np.zeros_like(effective)
        for label, _key, sign in matrix_components[family]:
            if sign == '=':
                continue
            arr = np.asarray(data[family][label][entry], dtype=float)
            rhs = rhs - arr if sign == '-' else rhs + arr
        return effective - rhs
    
    # Export to CSV
    import pandas as pd
    
    csv_cols = {'Velocity_m_s': velocities}
    for family, specs in matrix_components.items():
        for label, _key, _sign in specs:
            for i, j in entry_indices:
                csv_cols[f'{family}_{label}_{i}{j}'] = data[family][label][(i, j)]
        for i, j in entry_indices:
            csv_cols[f'{family}_balance_residual_{i}{j}'] = balance_residual(family, (i, j))
    contrib_df = pd.DataFrame(csv_cols)
    
    contrib_csv = os.path.join(save_path, "stiffness_damping_contributions.csv") if save_path else "stiffness_damping_contributions.csv"
    contrib_df.to_csv(contrib_csv, index=False, float_format='%.6e')
    print(f"Contributions data saved to {contrib_csv}")
    
    # Display velocity (optional knots)
    v_knots = bool(getattr(config, "v_knots", False)) if config else False
    V_plot = _vgvf_velocity_for_plot(np.asarray(velocities, dtype=float), v_knots=v_knots)
    v_xlabel = "Airspeed V [knots]" if v_knots else "Airspeed V [m/s]"

    # Create figure with M, K and C system-matrix contributions.
    fig, axes = plt.subplots(1, 3, figsize=(22, 5))
    component_style = {
        'structural': ('#1f77b4', 'Structural'),
        'aerodynamic_B2': ('#d62728', 'Aero B2 (subtracted)'),
        'aerodynamic_B0': ('#d62728', 'Aero B0 (subtracted)'),
        'aerodynamic_B1': ('#d62728', 'Aero B1 (subtracted)'),
        'hydro_Capytaine': ('#9467bd', 'Hydro/Capytaine (added)'),
        'hydro': ('#9467bd', 'Hydro (added)'),
        'empirical': ('#8c564b', 'Empirical (added)'),
        'effective': ('#2ca02c', 'Effective'),
    }
    mode_style = {0: ('-', 'o'), 1: ('--', 's')}

    def _plot_family(ax, family, title, ylabel):
        for label, _key, _sign in matrix_components[family]:
            color, label_text = component_style.get(label, ('#7f7f7f', label))
            for mode in (0, 1):
                entry = (mode, mode)
                if entry not in data[family][label]:
                    continue
                linestyle, marker = mode_style.get(mode, ('-', 'o'))
                series = data[family][label][entry]
                ax.plot(
                    V_plot, series,
                    color=color, linestyle=linestyle, marker=marker,
                    linewidth=1.1, markersize=2,
                    label=f'{label_text} {family}[{mode},{mode}]'
                )
        ax.set_xlabel(v_xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    m_title = (
        'Mass contributions: M_eff = M_struct - M_aero + M_hydro'
        if apply_b2_mass
        else 'Mass contributions: M_eff = M_struct + M_hydro'
    )
    _plot_family(
        axes[0],
        'M',
        m_title,
        'Mass diagonal [modal units]',
    )
    _plot_family(
        axes[1],
        'K',
        'Stiffness contributions: K_eff = K_struct - K_aero',
        'Stiffness diagonal [modal units]',
    )
    _plot_family(
        axes[2],
        'C',
        'Damping contributions: C_eff = C_struct - C_aero + C_hydro + C_emp',
        'Damping diagonal [modal units]',
    )
    
    plt.tight_layout()
    
    # Save figure
    outfile = os.path.join(save_path, "stiffness_damping_contributions.png") if save_path else "stiffness_damping_contributions.png"
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    print(f"Contributions plot saved to {outfile}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()  # Close the figure to free memory


def plot_DLM_participants(flutter_results, config=None, save_path="", show_plot=False):
    """
    Plot modal C_hat, K_hat (structural) and C_eff, K_eff (DLM-augmented) vs velocity.

    Data are collected during the Roger RFA sweep in ``_build_A_aug`` when
    ``config.plot_DLM_participants`` is True.
    """
    history = getattr(flutter_results, 'dlm_participants_history', None)
    if not history:
        print("\nNote: DLM participants plot skipped (no history; enable plot_DLM_participants for Roger RFA).")
        return

    print("\n--- Generating DLM participants plot (C_hat, K_hat, C_eff, K_eff vs V) ---")

    if not save_path and config and getattr(config, 'save_plots', False):
        os.makedirs(config.output_dir, exist_ok=True)
        save_path = config.output_dir
    elif not save_path:
        save_path = "output_data"
        os.makedirs(save_path, exist_ok=True)

    velocities = [step['V'] for step in history]
    n_modes = history[0]['C_hat'].shape[0]

    def _entry_series(key, i, j):
        return [float(step[key][i, j]) for step in history]

    import pandas as pd

    csv_cols = {'Velocity_m_s': velocities}
    for key in ('C_hat', 'K_hat', 'C_eff', 'K_eff'):
        for i in range(n_modes):
            csv_cols[f'{key}_{i}{i}'] = _entry_series(key, i, i)
    contrib_df = pd.DataFrame(csv_cols)
    case_tag = getattr(config, 'name', 'case') if config else 'case'
    contrib_csv = os.path.join(save_path, f"{case_tag}_DLM_participants.csv")
    contrib_df.to_csv(contrib_csv, index=False, float_format='%.6e')
    print(f"DLM participants data saved to {contrib_csv}")

    fig, (ax_k, ax_c) = plt.subplots(1, 2, figsize=(14, 5))
    linestyles_hat = ['-', '--']
    linestyles_eff = ['-', '--']

    for i in range(min(n_modes, 2)):
        ls_hat = linestyles_hat[i % len(linestyles_hat)]
        ls_eff = linestyles_eff[i % len(linestyles_eff)]
        ax_k.plot(velocities, _entry_series('K_hat', i, i),
                  color='C0', linestyle=ls_hat, linewidth=1.5,
                  label=f'K_hat[{i},{i}] (structural)')
        ax_k.plot(velocities, _entry_series('K_eff', i, i),
                  color='C1', linestyle=ls_eff, linewidth=1.5,
                  label=f'K_eff[{i},{i}] (DLM)')
        ax_c.plot(velocities, _entry_series('C_hat', i, i),
                  color='C0', linestyle=ls_hat, linewidth=1.5,
                  label=f'C_hat[{i},{i}] (structural)')
        ax_c.plot(velocities, _entry_series('C_eff', i, i),
                  color='C1', linestyle=ls_eff, linewidth=1.5,
                  label=f'C_eff[{i},{i}] (DLM)')

    ax_k.set_xlabel('Velocity V [m/s]')
    ax_k.set_ylabel('Stiffness diagonal [modal units]')
    ax_k.set_title('Modal stiffness: structural vs effective (DLM)')
    ax_k.legend(fontsize=9)
    ax_k.grid(True, alpha=0.3)

    ax_c.set_xlabel('Velocity V [m/s]')
    ax_c.set_ylabel('Damping diagonal [modal units]')
    ax_c.set_title('Modal damping: structural vs effective (DLM)')
    ax_c.legend(fontsize=9)
    ax_c.grid(True, alpha=0.3)

    plt.tight_layout()
    outfile = os.path.join(save_path, f"{case_tag}_DLM_participants.png")
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    print(f"DLM participants plot saved to {outfile}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_w_modal_components(pk_solver, config=None, save_path=""):
    """
    Plot the velocity and slope components of the modal downwash (after aerodynamic operator)
    at each converged velocity step.
    
    These are the norms of:
    - w_velocity: Qjj_DLM @ w_modal_velocity (velocity-induced downwash)
    - w_slope: Qjj_DLM @ w_modal_slope (slope-induced downwash)
    
    Args:
        pk_solver: PKSolverV3 instance with w_modal_history populated
        config: Configuration object (optional)
        save_path: Path for saving plots (optional)
    """
    print("\n--- Generating Modal Downwash Components Plot ---")
    
    w_modal_data = pk_solver.w_modal_history
    
    if not w_modal_data or len(w_modal_data) == 0:
        print("No w_modal data available to plot.")
        return
    
    # Determine output directory
    if config and getattr(config, 'save_plots', False):
        output_dir = config.output_dir
        os.makedirs(output_dir, exist_ok=True)
        save_path = output_dir
    
    # Extract data
    velocities = [w['V'] for w in w_modal_data]
    w_velocity_norm = [w['w_velocity_norm'] for w in w_modal_data]
    w_slope_norm = [w['w_slope_norm'] for w in w_modal_data]
    
    # Export to CSV
    import pandas as pd
    w_modal_df = pd.DataFrame({
        'Velocity_m_s': velocities,
        'w_velocity_norm': w_velocity_norm,
        'w_slope_norm': w_slope_norm
    })
    
    w_modal_csv = os.path.join(save_path, "w_modal_components.csv") if save_path else "w_modal_components.csv"
    w_modal_df.to_csv(w_modal_csv, index=False, float_format='%.6e')
    print(f"Modal downwash components data saved to {w_modal_csv}")
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    
    # Plot absolute values
    ax1.plot(velocities, w_velocity_norm, 'b-o', label='Velocity component (Qjj_DLM @ w_velocity)', 
             linewidth=1, markersize=2)
    ax1.plot(velocities, w_slope_norm, 'r-s', label='Slope component (Qjj_DLM @ w_slope)', 
             linewidth=1, markersize=2)
    ax1.set_xlabel('Airspeed V [m/s]', fontsize=12)
    ax1.set_ylabel('Downwash Component (norm)', fontsize=12)
    ax1.set_title('Modal Downwash Components vs Velocity', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')  # Log scale for better visualization
    
    # Plot ratio
    w_ratio = [w_slope_norm[i] / w_velocity_norm[i] if w_velocity_norm[i] > 0 else 0 
               for i in range(len(velocities))]
    
    ax2.plot(velocities, w_ratio, 'g-o', linewidth=1, markersize=2)
    ax2.axhline(1.0, linestyle='--', color='gray', linewidth=1, label='Equal contributions')
    ax2.set_xlabel('Airspeed V [m/s]', fontsize=12)
    ax2.set_ylabel('Slope / Velocity Ratio', fontsize=12)
    ax2.set_title('Downwash Component Ratio vs Velocity', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    outfile = os.path.join(save_path, "w_modal_components.png") if save_path else "w_modal_components.png"
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    print(f"Modal downwash components plot saved to {outfile}")
    
    plt.show()


def classify_bend_pitch_modes(Phi, config=None, n_modes_max=8):
    """
    Pick dry-mode indices with dominant translation (bend) and span-axis rotation (pitch).

    Uses 6-DOF blocks [u,v,w,θx,θy,θz] per node in ``Phi`` (free structural DOFs).
    """
    Phi = np.asarray(Phi, dtype=float)
    n_modes = min(Phi.shape[1], int(n_modes_max))
    span_axis = 1
    if config is not None and getattr(config, "beam_span_axis", None) is not None:
        span_axis = int(config.beam_span_axis)
    pitch_dof = 3 + span_axis
    axis_names = ("X", "Y", "Z")

    scores_bend = []
    scores_pitch = []
    for j in range(n_modes):
        col = Phi[:, j]
        if col.size % 6 == 0:
            blocks = col.reshape(-1, 6)
            trans_e = float(np.sum(blocks[:, :3] ** 2))
            pitch_e = float(np.sum(blocks[:, pitch_dof] ** 2))
            rot_e = float(np.sum(blocks[:, 3:] ** 2))
        else:
            third = max(col.size // 3, 1)
            trans_e = float(np.sum(col[:third] ** 2))
            rot_e = float(np.sum(col[third:] ** 2))
            pitch_e = rot_e
        scores_bend.append(trans_e)
        scores_pitch.append(pitch_e if pitch_e > 0.0 else rot_e)

    bend_idx = int(np.argmax(scores_bend))
    pitch_order = np.argsort(scores_pitch)[::-1]
    pitch_idx = int(pitch_order[0])
    if pitch_idx == bend_idx and len(pitch_order) > 1:
        pitch_idx = int(pitch_order[1])

    bend_label = f"mode {bend_idx} (translation, E={scores_bend[bend_idx]:.3e})"
    pitch_label = (
        f"mode {pitch_idx} (θ{axis_names[span_axis]}, E={scores_pitch[pitch_idx]:.3e})"
    )
    return bend_idx, pitch_idx, bend_label, pitch_label


def _downwash_slope_velocity_ratio(Z, Z_qs, Phi, mode_idx, k, V):
    """‖Z_qs φ_j‖ / ‖(ik) Z φ_j‖ with PanelAero k = ω/V."""
    from numpy.linalg import norm

    V = max(float(V), 1e-12)
    k = max(float(k), 1e-12)
    phi = Phi[:, int(mode_idx)]
    w_slope = Z_qs @ phi
    w_vel = (1j * k) * (Z @ phi)
    n_s = float(norm(w_slope))
    n_v = float(norm(w_vel))
    ratio = n_s / n_v if n_v > 1e-30 else np.nan
    return ratio, n_s, n_v


def diagnose_modal_aero_bend_pitch(
    config,
    structural_results,
    coupling_results,
    flutter_results,
    *,
    bend_mode_idx=None,
    pitch_mode_idx=None,
    branch_index=0,
    save_csv=True,
    show_plot=True,
):
    """
    Bend–pitch block of ``Q_modal`` and downwash slope/velocity ratios vs airspeed.

    Recomputes ``Q_modal`` at each converged PK (or Roger) snapshot using the same
    ``make_Qg_func`` path as the flutter solver.  The 2×2 block uses dry-mode indices
    auto-classified from ``Phi`` unless ``bend_mode_idx`` / ``pitch_mode_idx`` are set.
    """
    from COUPLING.hydroelastic_utl import aero_system

    print("\n--- Modal aero bend–pitch diagnostic ---")

    raw = flutter_results.raw_results
    if not raw:
        print("  No raw_results: need PK or Roger sweep snapshots.")
        return None

    Z = coupling_results["Z"]
    Z_qs = coupling_results["Z_qs"]
    Z_force = coupling_results.get("Z_force")
    Phi = structural_results.dry_eigenvectors
    if Z_force is None:
        print("  Z_force missing from coupling_results; cannot form Q_modal.")
        return None

    n_modal = int(Phi.shape[1])
    bend_idx, pitch_idx, bend_lbl, pitch_lbl = classify_bend_pitch_modes(
        Phi, config, n_modes_max=n_modal
    )
    if bend_mode_idx is not None:
        bend_idx = int(bend_mode_idx)
        bend_lbl = f"mode {bend_idx} (user)"
    if pitch_mode_idx is not None:
        pitch_idx = int(pitch_mode_idx)
        pitch_lbl = f"mode {pitch_idx} (user)"
    bend_idx = int(np.clip(bend_idx, 0, n_modal - 1))
    pitch_idx = int(np.clip(pitch_idx, 0, n_modal - 1))
    pair = (bend_idx, pitch_idx)
    print(f"  Bend:  {bend_lbl}")
    print(f"  Pitch: {pitch_lbl}")

    panel_areas = None
    ag_path = getattr(config, "aerogrid_path", None)
    if ag_path and os.path.isfile(ag_path):
        ag = np.load(ag_path)
        if "A" in ag:
            panel_areas = ag["A"]

    Qg_func = aero_system.make_Qg_func(
        config,
        config.paths["FSI"],
        Z,
        coupling_results["Apan"],
        Z_qs,
        Phi,
        structural_results.dry_eigenvalues,
        c_sound=config.c_sound[config.fluid],
        out_dir_klist=config.qjj_dir,
        out_dir_vjj=config.vjj_dir,
        alpha_r=config.alpha_r,
        Z_force=Z_force,
        panel_areas=panel_areas,
    )

    v_list = np.asarray(flutter_results.velocities, dtype=float)
    freq_all = np.asarray(flutter_results.frequencies, dtype=float)

    rows = []
    for res_v in raw:
        V = float(res_v["V"])
        iv = int(np.argmin(np.abs(v_list - V)))

        omega_b = np.nan
        omega_p = np.nan
        if freq_all.ndim >= 2:
            if bend_idx < freq_all.shape[1]:
                omega_b = float(freq_all[iv, bend_idx])
            if pitch_idx < freq_all.shape[1]:
                omega_p = float(freq_all[iv, pitch_idx])
        elif freq_all.ndim == 1 and iv < freq_all.size:
            omega_b = float(freq_all[iv])

        modes = res_v.get("modes", [])
        md = modes[branch_index] if branch_index < len(modes) else {}
        omega_md = float(md.get("omega", np.nan))
        if np.isfinite(omega_md) and omega_md > 0.0:
            if not np.isfinite(omega_b):
                omega_b = omega_md
        if not np.isfinite(omega_b) or omega_b <= 0.0:
            continue
        if not np.isfinite(omega_p) or omega_p <= 0.0:
            omega_p = omega_b

        k_red = md.get("k", omega_b / max(V, 1e-12))
        k_red = max(float(k_red), 1e-9)
        p_conv = md.get("p", None)
        if p_conv is None or not np.isfinite(np.real(p_conv)) or not np.isfinite(np.imag(p_conv)):
            p_conv = 1j * omega_b

        try:
            Q_modal = Qg_func(k_red, V, p_conv, branch_index, 99)
        except Exception as exc:
            print(f"  Warning: Q_modal at V={V:.2f} m/s failed: {exc}")
            continue

        bi, pi = pair
        if Q_modal.shape[0] <= max(bi, pi) or Q_modal.shape[1] <= max(bi, pi):
            print(f"  Warning: Q_modal shape {Q_modal.shape} too small for indices {pair}")
            continue

        r_b, ns_b, nv_b = _downwash_slope_velocity_ratio(Z, Z_qs, Phi, bi, k_red, V)
        r_p, ns_p, nv_p = _downwash_slope_velocity_ratio(Z, Z_qs, Phi, pi, k_red, V)

        def qget(i, j):
            return complex(Q_modal[i, j])

        rows.append(
            {
                "V_m_s": V,
                "k": k_red,
                "omega_bend_rad_s": omega_b,
                "omega_pitch_rad_s": omega_p,
                "Re_Q_bb": np.real(qget(bi, bi)),
                "Re_Q_bp": np.real(qget(bi, pi)),
                "Re_Q_pb": np.real(qget(pi, bi)),
                "Re_Q_pp": np.real(qget(pi, pi)),
                "Im_Q_bb": np.imag(qget(bi, bi)),
                "Im_Q_bp": np.imag(qget(bi, pi)),
                "Im_Q_pb": np.imag(qget(pi, bi)),
                "Im_Q_pp": np.imag(qget(pi, pi)),
                "slope_vel_ratio_bend": r_b,
                "slope_vel_ratio_pitch": r_p,
                "w_slope_norm_bend": ns_b,
                "w_vel_norm_bend": nv_b,
                "w_slope_norm_pitch": ns_p,
                "w_vel_norm_pitch": nv_p,
            }
        )

    if not rows:
        print("  No converged velocity points with valid Q_modal.")
        return None

    import pandas as pd

    df = pd.DataFrame(rows)
    save_path = ""
    if getattr(config, "save_plots", False) or getattr(config, "output_dir", None):
        save_path = getattr(config, "output_dir", "") or ""
        if save_path:
            os.makedirs(save_path, exist_ok=True)

    tag = getattr(config, "name", "case")
    if save_csv:
        csv_path = os.path.join(save_path, f"{tag}_qmodal_bend_pitch.csv") if save_path else f"{tag}_qmodal_bend_pitch.csv"
        df.to_csv(csv_path, index=False, float_format="%.6e")
        print(f"  Saved {csv_path}")

    V_arr = df["V_m_s"].to_numpy()
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    ax = axes[0]
    for key, lbl in (
        ("Re_Q_bb", "Re Q_bb"),
        ("Re_Q_bp", "Re Q_bp"),
        ("Re_Q_pb", "Re Q_pb"),
        ("Re_Q_pp", "Re Q_pp"),
    ):
        ax.plot(V_arr, df[key], "-o", markersize=2, label=lbl)
    ax.set_xlabel("V [m/s]")
    ax.set_ylabel("Re(Q_modal) [N]")
    ax.set_title(f"Real 2×2 block ({bend_lbl} × {pitch_lbl})")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(V_arr, df["Im_Q_bb"], "-o", markersize=2, label="Im Q_bb")
    ax.plot(V_arr, df["Im_Q_pp"], "-s", markersize=2, label="Im Q_pp")
    ax.set_xlabel("V [m/s]")
    ax.set_ylabel("Im(Q_modal) [N]")
    ax.set_title("Imaginary diagonal (damping driver)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    ax = axes[2]
    ax.plot(V_arr, df["slope_vel_ratio_bend"], "-o", markersize=2, label="bend ‖Z_qs φ‖/‖ik Z φ‖")
    ax.plot(V_arr, df["slope_vel_ratio_pitch"], "-s", markersize=2, label="pitch")
    ax.axhline(1.0, color="gray", ls="--", lw=1)
    ax.set_xlabel("V [m/s]")
    ax.set_ylabel("slope / velocity downwash ratio")
    ax.set_title("Pitch (Z_qs) vs unsteady (ik Z) coupling")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    plt.tight_layout()
    png_path = os.path.join(save_path, f"{tag}_qmodal_bend_pitch.png") if save_path else f"{tag}_qmodal_bend_pitch.png"
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"  Plot saved to {png_path}")
    if show_plot:
        plt.show()
    else:
        plt.close()

    # Summary near flutter
    Vf = flutter_results.flutter_speed
    if Vf is not None and np.isfinite(Vf):
        iv = int(np.argmin(np.abs(V_arr - Vf)))
        row = df.iloc[iv]
        print(f"\n  Near flutter (V≈{V_arr[iv]:.2f} m/s, Vf≈{Vf:.2f} m/s):")
        print(
            f"    k={row['k']:.4f}, ω_bend={row['omega_bend_rad_s']:.2f} rad/s, "
            f"slope/vel bend={row['slope_vel_ratio_bend']:.3f}, pitch={row['slope_vel_ratio_pitch']:.3f}"
        )
        print(
            f"    Im(Q_bb)={row['Im_Q_bb']:.3e}, Im(Q_pp)={row['Im_Q_pp']:.3e}, "
            f"Re(Q_bp)={row['Re_Q_bp']:.3e}"
        )

    return df


def plot_deformed_mode_shapes(aerogrid, beam_model, structural_results, coupling_results, 
                               modes_to_plot=None, scale_factor=1.0, config=None, 
                               animate=False, n_frames=20, show_rotations=True):
    """
    Plot the deformed dry mode shapes overlaid on the aerogrid, showing the vibrating structure
    with all 6 DOFs (3 translations + 3 rotations).
    
    Parameters
    ----------
    aerogrid : dict
        Aerodynamic grid with cornerpoint_grids, cornerpoint_panels, offset_j, offset_l
    beam_model : dict
        Structural beam model with nodes
    structural_results : StructuralResults
        Contains dry_eigenvectors_full (full DOFs including constrained) or dry_eigenvectors
    coupling_results : dict
        Contains Z_qs (spline matrix for displacement transfer)
    modes_to_plot : list of int, optional
        List of mode indices to plot (0-indexed). If None, plots first 4 modes.
    scale_factor : float, optional
        Scaling factor for mode shape visualization (default: 1.0)
    config : Config object, optional
        Configuration for saving plots
    animate : bool, optional
        If True, creates animated plots showing oscillation (default: False)
    n_frames : int, optional
        Number of frames for animation (default: 20)
    show_rotations : bool, optional
        If True, visualizes rotations using cross-section lines (default: True)
    """
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from mpl_toolkits.mplot3d import Axes3D
    
    print("\n--- Plotting Deformed Mode Shapes on Aerogrid ---")
    
    # Determine which modes to plot
    if modes_to_plot is None:
        n_modes = min(4, structural_results.dry_eigenvectors.shape[1])
        modes_to_plot = list(range(n_modes))
    
    # Get spline matrices
    Z_qs = coupling_results['Z_qs']
    
    # Extract node positions
    node_coords = np.array([node['position'] for node in beam_model['nodes']])
    n_nodes = len(beam_model['nodes'])
    
    # Use free eigenvectors (without constrained DOFs) for coupling with Z_qs
    # Z_qs expects free DOFs only
    eigenvectors = structural_results.dry_eigenvectors
    print(f"  Using free eigenvectors (shape: {eigenvectors.shape}) for Z_qs coupling")
    
    # For extracting nodal displacements, we need full DOFs (with zeros at constrained)
    if hasattr(structural_results, 'dry_eigenvectors_full') and structural_results.dry_eigenvectors_full is not None:
        eigenvectors_full = structural_results.dry_eigenvectors_full
        print(f"  Using full eigenvectors (shape: {eigenvectors_full.shape}) for nodal displacements")
    else:
        # If full not available, use free (will have issues at constrained nodes)
        eigenvectors_full = eigenvectors
        print(f"  Warning: full eigenvectors not available, using free eigenvectors")
    
    # Get aerogrid points
    aero_grid_pts = aerogrid['cornerpoint_grids'][:, 1:4]  # [x, y, z]
    aero_panels = aerogrid['cornerpoint_panels']
    aero_grid_ids = aerogrid['cornerpoint_grids'][:, 0]
    
    # Get control and force points if available
    has_control_pts = 'offset_j' in aerogrid and len(aerogrid['offset_j']) > 0
    has_force_pts = 'offset_l' in aerogrid and len(aerogrid['offset_l']) > 0
    
    if has_control_pts:
        control_pts = np.asarray(aerogrid['offset_j'])
    if has_force_pts:
        force_pts = np.asarray(aerogrid['offset_l'])
    
    # Create plots for each mode
    for mode_idx in modes_to_plot:
        if mode_idx >= eigenvectors.shape[1]:
            print(f"  Warning: Mode {mode_idx} not available, skipping")
            continue
        
        # Get mode vectors (free DOFs for coupling, full DOFs for nodal displacements)
        mode_vector_free = eigenvectors[:, mode_idx]
        mode_vector_full = eigenvectors_full[:, mode_idx]
        
        # Get natural frequency for this mode
        eigenvalue = structural_results.dry_eigenvalues[mode_idx]
        omega_n = np.sqrt(np.abs(eigenvalue))
        freq_hz = omega_n / (2 * np.pi)
        
        print(f"\n  Mode {mode_idx}: ω = {omega_n:.3f} rad/s, f = {freq_hz:.3f} Hz")
        
        # Extract ALL 6 DOFs from FULL mode shape (for nodal visualization)
        # Mode vector has 6 DOFs per node: [u, v, w, θx, θy, θz]
        u_mode = mode_vector_full[0::6][:n_nodes]  # X-displacement (chordwise)
        v_mode = mode_vector_full[1::6][:n_nodes]  # Y-displacement (spanwise/axial)
        w_mode = mode_vector_full[2::6][:n_nodes]  # Z-displacement (vertical)
        theta_x_mode = mode_vector_full[3::6][:n_nodes]  # Rotation about X
        theta_y_mode = mode_vector_full[4::6][:n_nodes]  # Rotation about Y (torsion!)
        theta_z_mode = mode_vector_full[5::6][:n_nodes]  # Rotation about Z
        
        # Print DOF statistics to understand mode shape character (ACTUAL VALUES)
        print(f"    DOF Magnitudes (actual physical values from eigenvector):")
        print(f"      Translation: u={np.max(np.abs(u_mode)):.3e}, v={np.max(np.abs(v_mode)):.3e}, w={np.max(np.abs(w_mode)):.3e}")
        print(f"      Rotation:    θx={np.max(np.abs(theta_x_mode)):.3e}, θy={np.max(np.abs(theta_y_mode)):.3e}, θz={np.max(np.abs(theta_z_mode)):.3e}")
        
        # Identify dominant DOF
        dof_names = ['u (chordwise)', 'v (spanwise)', 'w (vertical)', 'θx', 'θy (torsion)', 'θz']
        dof_values = [
            np.max(np.abs(u_mode)),
            np.max(np.abs(v_mode)),
            np.max(np.abs(w_mode)),
            np.max(np.abs(theta_x_mode)),
            np.max(np.abs(theta_y_mode)),
            np.max(np.abs(theta_z_mode))
        ]
        dominant_idx = np.argmax(dof_values)
        print(f"    Dominant DOF: {dof_names[dominant_idx]}")
        
        # USE RAW VALUES - Apply only user scale factor (no normalization!)
        # This shows the actual physical deformation from the eigenvector
        u_scaled = u_mode * scale_factor
        v_scaled = v_mode * scale_factor
        w_scaled = w_mode * scale_factor
        theta_x_scaled = theta_x_mode * scale_factor
        theta_y_scaled = theta_y_mode * scale_factor
        theta_z_scaled = theta_z_mode * scale_factor
        
        print(f"    Applied scale factor: {scale_factor}")
        print(f"    Max deformations after scaling:")
        print(f"      u={np.max(np.abs(u_scaled)):.3e} m, w={np.max(np.abs(w_scaled)):.3e} m, θy={np.max(np.abs(theta_y_scaled)):.3e} rad")
        
        # Compute deformed node positions
        nodes_deformed = node_coords.copy()
        nodes_deformed[:, 0] += u_scaled
        nodes_deformed[:, 1] += v_scaled
        nodes_deformed[:, 2] += w_scaled
        
        # Transfer deformations to aerogrid using spline matrix Z_qs
        # Z_qs @ q_structural = q_aerodynamic (displacements at aero points)
        # Use FREE DOFs for coupling (Z_qs expects free DOFs only)
        q_structural = mode_vector_free
        
        # Deform control points if available - INCLUDING TORSIONAL ROTATION
        if has_control_pts:
            n_aero_pts = control_pts.shape[0]
            
            # Z_qs is (n_aero_panels, n_structural_dofs_free)
            # Result q_aero has shape (n_aero_panels,) with each panel's downwash/displacement
            q_aero = Z_qs @ q_structural
            
            # Extract vertical displacement from downwash
            w_aero = q_aero
            
            # Apply scale factor directly (no normalization)
            w_aero_scaled = w_aero * scale_factor
            
            # Initialize deformed control points
            control_pts_deformed = control_pts.copy()
            u_aero = np.zeros(n_aero_pts)
            v_aero = np.zeros(n_aero_pts)
            
            # Apply torsional rotation to each control point based on interpolated beam rotation
            # For each control point, find nearest beam node and interpolate torsion angle
            for i_aero in range(n_aero_pts):
                aero_pos = control_pts[i_aero]
                
                # Find the closest beam node(s) for interpolation (based on Y-coordinate, spanwise)
                y_aero = aero_pos[1]
                
                # Find bracketing nodes in Y direction
                y_nodes = node_coords[:, 1]
                if y_aero <= y_nodes[0]:
                    # Before first node, use first node values
                    theta_y_interp = theta_y_scaled[0]
                    beam_center_undeformed = node_coords[0]
                    beam_displacement = np.array([u_scaled[0], v_scaled[0], w_scaled[0]])
                elif y_aero >= y_nodes[-1]:
                    # After last node, use last node values
                    theta_y_interp = theta_y_scaled[-1]
                    beam_center_undeformed = node_coords[-1]
                    beam_displacement = np.array([u_scaled[-1], v_scaled[-1], w_scaled[-1]])
                else:
                    # Linear interpolation between nodes
                    idx_upper = np.searchsorted(y_nodes, y_aero)
                    idx_lower = idx_upper - 1
                    
                    # Interpolation weight
                    y_lower = y_nodes[idx_lower]
                    y_upper = y_nodes[idx_upper]
                    weight = (y_aero - y_lower) / (y_upper - y_lower + 1e-12)
                    
                    # Interpolate torsion angle
                    theta_y_interp = theta_y_scaled[idx_lower] * (1 - weight) + theta_y_scaled[idx_upper] * weight
                    
                    # Interpolate beam center position (undeformed)
                    beam_center_undeformed = node_coords[idx_lower] * (1 - weight) + node_coords[idx_upper] * weight
                    
                    # Interpolate beam displacement
                    u_interp = u_scaled[idx_lower] * (1 - weight) + u_scaled[idx_upper] * weight
                    v_interp = v_scaled[idx_lower] * (1 - weight) + v_scaled[idx_upper] * weight
                    w_interp = w_scaled[idx_lower] * (1 - weight) + w_scaled[idx_upper] * weight
                    beam_displacement = np.array([u_interp, v_interp, w_interp])
                
                # Apply torsional rotation about UNDEFORMED beam axis (Y-axis)
                # Vector from undeformed beam center to aero point (in XZ plane)
                dx = aero_pos[0] - beam_center_undeformed[0]
                dz = aero_pos[2] - beam_center_undeformed[2]
                
                # Rotate this vector by theta_y (torsion angle about Y-axis)
                cos_theta = np.cos(theta_y_interp)
                sin_theta = np.sin(theta_y_interp)
                
                dx_rot = dx * cos_theta + dz * sin_theta
                dz_rot = -dx * sin_theta + dz * cos_theta
                
                # Update control point position: 
                # 1. Start from undeformed beam center
                # 2. Add rotated offset from beam center
                # 3. Add beam translation
                # 4. Add vertical displacement from downwash
                control_pts_deformed[i_aero, 0] = beam_center_undeformed[0] + dx_rot + beam_displacement[0]
                control_pts_deformed[i_aero, 1] = beam_center_undeformed[1] + beam_displacement[1]  # Y with beam displacement
                control_pts_deformed[i_aero, 2] = beam_center_undeformed[2] + dz_rot + beam_displacement[2] + w_aero_scaled[i_aero]
                
                # Store u and v for animation
                u_aero[i_aero] = control_pts_deformed[i_aero, 0] - control_pts[i_aero, 0]
                v_aero[i_aero] = control_pts_deformed[i_aero, 1] - control_pts[i_aero, 1]
        
        # Create the plot
        if not animate:
            # Static plot
            fig = plt.figure(figsize=(14, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            _plot_mode_shape_frame(ax, aero_grid_pts, aero_panels, aero_grid_ids,
                                   node_coords, nodes_deformed,
                                   control_pts if has_control_pts else None,
                                   control_pts_deformed if has_control_pts else None,
                                   force_pts if has_force_pts else None,
                                   mode_idx, freq_hz, omega_n,
                                   rotations=(theta_x_scaled, theta_y_scaled, theta_z_scaled) if show_rotations else None,
                                   config=config)
            
            # Save plot
            if config and config.save_plots:
                os.makedirs(config.output_dir, exist_ok=True)
                outfile = os.path.join(config.output_dir, f"mode_shape_{mode_idx}_deformed.png")
                plt.savefig(outfile, dpi=300, bbox_inches='tight')
                print(f"  Mode {mode_idx} deformed shape saved to {outfile}")
            
            plt.show()
        
        else:
            # Animated plot
            fig = plt.figure(figsize=(14, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            def animate_frame(frame):
                ax.clear()
                # Oscillate between -1 and +1
                phase = np.cos(2 * np.pi * frame / n_frames)
                
                # Compute deformed positions at this phase
                nodes_anim = node_coords.copy()
                nodes_anim[:, 0] += u_scaled * phase
                nodes_anim[:, 1] += v_scaled * phase
                nodes_anim[:, 2] += w_scaled * phase
                
                control_anim = None
                if has_control_pts:
                    # Recompute torsional deformation for each aero point at this phase
                    control_anim = control_pts.copy()
                    
                    for i_aero in range(n_aero_pts):
                        aero_pos = control_pts[i_aero]
                        y_aero = aero_pos[1]
                        
                        # Find bracketing nodes in Y direction
                        y_nodes = node_coords[:, 1]
                        if y_aero <= y_nodes[0]:
                            theta_y_interp = theta_y_scaled[0] * phase
                            beam_center_undeformed = node_coords[0]
                            beam_disp = np.array([u_scaled[0], v_scaled[0], w_scaled[0]]) * phase
                        elif y_aero >= y_nodes[-1]:
                            theta_y_interp = theta_y_scaled[-1] * phase
                            beam_center_undeformed = node_coords[-1]
                            beam_disp = np.array([u_scaled[-1], v_scaled[-1], w_scaled[-1]]) * phase
                        else:
                            idx_upper = np.searchsorted(y_nodes, y_aero)
                            idx_lower = idx_upper - 1
                            y_lower = y_nodes[idx_lower]
                            y_upper = y_nodes[idx_upper]
                            weight = (y_aero - y_lower) / (y_upper - y_lower + 1e-12)
                            
                            theta_y_interp = (theta_y_scaled[idx_lower] * (1 - weight) + 
                                            theta_y_scaled[idx_upper] * weight) * phase
                            beam_center_undeformed = node_coords[idx_lower] * (1 - weight) + node_coords[idx_upper] * weight
                            
                            u_interp = (u_scaled[idx_lower] * (1 - weight) + u_scaled[idx_upper] * weight) * phase
                            v_interp = (v_scaled[idx_lower] * (1 - weight) + v_scaled[idx_upper] * weight) * phase
                            w_interp = (w_scaled[idx_lower] * (1 - weight) + w_scaled[idx_upper] * weight) * phase
                            beam_disp = np.array([u_interp, v_interp, w_interp])
                        
                        # Apply torsional rotation about undeformed beam center
                        dx = aero_pos[0] - beam_center_undeformed[0]
                        dz = aero_pos[2] - beam_center_undeformed[2]
                        
                        cos_theta = np.cos(theta_y_interp)
                        sin_theta = np.sin(theta_y_interp)
                        
                        dx_rot = dx * cos_theta + dz * sin_theta
                        dz_rot = -dx * sin_theta + dz * cos_theta
                        
                        control_anim[i_aero, 0] = beam_center_undeformed[0] + dx_rot + beam_disp[0]
                        control_anim[i_aero, 1] = beam_center_undeformed[1] + beam_disp[1]
                        control_anim[i_aero, 2] = beam_center_undeformed[2] + dz_rot + beam_disp[2] + w_aero_scaled[i_aero] * phase
                
                _plot_mode_shape_frame(ax, aero_grid_pts, aero_panels, aero_grid_ids,
                                       node_coords, nodes_anim,
                                       control_pts if has_control_pts else None,
                                       control_anim,
                                       force_pts if has_force_pts else None,
                                       mode_idx, freq_hz, omega_n,
                                       title_suffix=f" (Phase: {phase:.2f})",
                                       rotations=(theta_x_scaled * phase, theta_y_scaled * phase, theta_z_scaled * phase) if show_rotations else None,
                                       config=config,
                                       add_colorbar=False)  # Don't add colorbar in animations
            
            anim = animation.FuncAnimation(fig, animate_frame, frames=n_frames, 
                                          interval=50, repeat=True)
            
            # Save animation
            if config and config.save_plots:
                os.makedirs(config.output_dir, exist_ok=True)
                outfile = os.path.join(config.output_dir, f"mode_shape_{mode_idx}_animated.gif")
                anim.save(outfile, writer='pillow', fps=20, dpi=150)
                print(f"  Mode {mode_idx} animation saved to {outfile}")
            
            plt.show()


def _plot_mode_shape_frame(ax, aero_grid_pts, aero_panels, aero_grid_ids,
                           node_coords, nodes_deformed,
                           control_pts, control_pts_deformed,
                           force_pts, mode_idx, freq_hz, omega_n,
                           title_suffix="", rotations=None, config=None, add_colorbar=True):
    """
    Helper function to plot a single frame of the mode shape visualization.
    
    Parameters
    ----------
    rotations : tuple of arrays or None
        (theta_x, theta_y, theta_z) rotation arrays for each node. If provided,
        visualizes rotations using cross-section indicators.
    add_colorbar : bool
        If True, adds a colorbar to the plot. Set to False for animations to avoid multiple colorbars.
    """
    # Plot undeformed aerogrid (light gray, transparent)
    for panel in aero_panels:
        coords = []
        for pid in panel:
            idx = np.where(aero_grid_ids == pid)[0]
            if idx.size > 0:
                coords.append(aero_grid_pts[idx[0]])
        if len(coords) >= 3:
            coords = np.vstack(coords + [coords[0]])
            ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], 
                   color='lightgray', linewidth=0.5, alpha=0.3)
    
    # Plot aerogrid corner points (undeformed, small gray dots)
    ax.scatter(aero_grid_pts[:, 0], aero_grid_pts[:, 1], aero_grid_pts[:, 2], 
              s=3, c='gray', marker='o', alpha=0.2)
    
    # Plot deformed control points (if available) as a colored surface
    # Color represents deformation magnitude: blue = less, through rainbow to red = more
    if control_pts_deformed is not None and control_pts is not None:
        # Calculate deformation magnitude at each control point
        deformation = np.sqrt(
            (control_pts_deformed[:, 0] - control_pts[:, 0])**2 +
            (control_pts_deformed[:, 1] - control_pts[:, 1])**2 +
            (control_pts_deformed[:, 2] - control_pts[:, 2])**2
        )
        
        # Create scatter plot with rainbow colormap (blue -> cyan -> green -> yellow -> red)
        scatter = ax.scatter(control_pts_deformed[:, 0], control_pts_deformed[:, 1], 
                  control_pts_deformed[:, 2], 
                  c=deformation, cmap='jet', 
                  s=20, marker='o', alpha=0.9, edgecolors='none',
                  vmin=0, vmax=np.max(deformation),
                  label='Deformed Aero Control Points')
        
        # Add colorbar to show deformation scale (only if requested, not for animations)
        if add_colorbar:
            cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.6)
            cbar.set_label('Deformation magnitude [m]', fontsize=9)
    
    # Plot force points (undeformed reference)
    if force_pts is not None:
        ax.scatter(force_pts[:, 0], force_pts[:, 1], force_pts[:, 2], 
                  c='darkgreen', marker='^', s=15, alpha=0.3, label='Force Points')
    
    # Plot undeformed beam (dashed gray line)
    ax.plot(node_coords[:, 0], node_coords[:, 1], node_coords[:, 2], 
           'k--', linewidth=1.5, alpha=0.4, label='Undeformed Beam')
    
    # Plot deformed beam (solid colored line)
    ax.plot(nodes_deformed[:, 0], nodes_deformed[:, 1], nodes_deformed[:, 2], 
           'r-o', linewidth=2.5, markersize=6, alpha=0.9, label='Deformed Beam')
    
    # Visualize rotations using cross-section indicators
    if rotations is not None:
        theta_x, theta_y, theta_z = rotations
        n_nodes = len(nodes_deformed)
        
        # Get chord length from config or estimate
        if config and hasattr(config, 'chord'):
            chord = config.chord
        else:
            # Estimate from aerogrid range in X direction
            chord = np.max(aero_grid_pts[:, 0]) - np.min(aero_grid_pts[:, 0])
        
        # Visualize rotations at selected nodes (not all to avoid clutter)
        step = max(1, n_nodes // 10)  # Show ~10 cross-sections
        
        for i in range(0, n_nodes, step):
            pos = nodes_deformed[i]
            
            # Create a local coordinate cross showing the rotated frame
            # Cross-section line length proportional to chord
            cross_len = 0.3 * chord
            
            # Start with unit vectors in local frame
            # Y-axis is the beam axis (spanwise)
            vec_x = np.array([1, 0, 0])  # Chordwise direction
            vec_z = np.array([0, 0, 1])  # Vertical direction
            
            # Apply rotations (small angle approximation for visualization)
            # Rotation about Y (torsion - most important for flutter)
            if abs(theta_y[i]) > 1e-6:
                # Rotate X and Z vectors about Y axis
                cos_y = np.cos(theta_y[i])
                sin_y = np.sin(theta_y[i])
                vec_x_rot = vec_x * cos_y + vec_z * sin_y
                vec_z_rot = -vec_x * sin_y + vec_z * cos_y
            else:
                vec_x_rot = vec_x
                vec_z_rot = vec_z
            
            # Apply rotation about X (if significant)
            if abs(theta_x[i]) > 1e-6:
                cos_x = np.cos(theta_x[i])
                sin_x = np.sin(theta_x[i])
                # This would rotate in YZ plane, but Y is beam axis
                # For simplicity, skip or apply small correction
                pass
            
            # Apply rotation about Z (if significant)
            if abs(theta_z[i]) > 1e-6:
                cos_z = np.cos(theta_z[i])
                sin_z = np.sin(theta_z[i])
                # Rotate X and Y vectors about Z
                # For beam along Y, this affects chordwise orientation
                pass
            
            # Draw cross-section indicator (chordwise line)
            p1 = pos - vec_x_rot * cross_len / 2
            p2 = pos + vec_x_rot * cross_len / 2
            
            # Color based on torsion magnitude
            torsion_color = plt.cm.coolwarm(0.5 + 0.5 * np.tanh(theta_y[i] * 5))
            
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
                   color=torsion_color, linewidth=2, alpha=0.7)
            
            # Optionally add vertical line to show Z-rotation effect
            if abs(theta_x[i]) > 1e-5 or abs(theta_z[i]) > 1e-5:
                p3 = pos - vec_z_rot * cross_len / 4
                p4 = pos + vec_z_rot * cross_len / 4
                ax.plot([p3[0], p4[0]], [p3[1], p4[1]], [p3[2], p4[2]], 
                       color='orange', linewidth=1.5, alpha=0.5)
        
        # Add rotation info to title
        max_torsion = np.max(np.abs(theta_y))
        max_theta_x = np.max(np.abs(theta_x))
        max_theta_z = np.max(np.abs(theta_z))
        rotation_text = f"\n[Max Rotations: θy={max_torsion:.3f} rad, θx={max_theta_x:.3f}, θz={max_theta_z:.3f}]"
        ax.text2D(0.5, 0.95, rotation_text, transform=ax.transAxes, 
                 ha='center', va='top', fontsize=9, color='darkred')
    
    ax.set_xlabel('X [m]', fontsize=11)
    ax.set_ylabel('Y [m]', fontsize=11)
    ax.set_zlabel('Z [m]', fontsize=11)
    ax.set_title(f'Mode {mode_idx}: f = {freq_hz:.3f} Hz (ω = {omega_n:.3f} rad/s){title_suffix}', 
                fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    
    # Set equal aspect ratio
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
    
    ax.grid(True, alpha=0.2)

