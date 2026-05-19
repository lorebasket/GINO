# FSI/COUPLING/hydroelastic_utl/post_processing.py

from . import vgvf_plotting
from panelaero_utl.plotting_git import DetailedPlots
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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


def plot_vg_lambda_vf_combined(
    flutter_results,
    config=None,
    save_path="",
    show_plot=False,
    output_tag: Optional[str] = None,
):
    """
    Abramson-style figure: V–Λ (log decrement), V–g, and V–f in one row.

    Λ = 2π (σ/ω) from stored damping/frequency arrays (see ``log_decrement_from_storage``).
    Works for any solver that fills ``FlutterResults`` (PK, Roger RFA, RFA+PK).
    """
    if config is None or not getattr(config, "plot_log_decrement_vg", False):
        return

    print("\n--- Generating V–Λ / V–g / V–f combined plot ---")

    V_m_s = np.asarray(flutter_results.velocities, dtype=float)
    g = np.asarray(flutter_results.damping, dtype=float)
    omega_rad_s = np.asarray(flutter_results.frequencies, dtype=float)
    Vf_m_s = flutter_results.flutter_speed

    v_knots, vf_hertz, v_label, omega_label, v_unit_short = _vgvf_plot_axes(config)
    V = _vgvf_velocity_for_plot(V_m_s, v_knots=v_knots)
    omega = _vgvf_frequency_for_plot(omega_rad_s, vf_hertz=vf_hertz)
    Vf_plot = _vgvf_scalar_velocity_for_plot(Vf_m_s, v_knots=v_knots)

    dimless = bool(getattr(config, "dimensionless_vgvf_results", True))
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
    for j in range(Lambda.shape[1]):
        vlam_data[f"LogDecrement_Branch_{j + 1}"] = Lambda[:, j]
    vlam_df = pd.DataFrame(vlam_data)
    vlam_csv = (
        os.path.join(save_path, _output_name_with_tag("vlambda_data.csv", output_tag))
        if save_path
        else _output_name_with_tag("vlambda_data.csv", output_tag)
    )
    vlam_df.to_csv(vlam_csv, index=False, float_format="%.6f")
    print(f"V–Λ data saved to {vlam_csv}")

    fig, (ax_lam, ax_g, ax_f) = plt.subplots(1, 3, figsize=(18, 5))

    for j in range(Lambda.shape[1]):
        ax_lam.plot(V, Lambda[:, j], label=f"Branch {j + 1}")
    ax_lam.axhline(0.0, linestyle="--", linewidth=1, color="gray")

    for j in range(g.shape[1]):
        ax_g.plot(V, g[:, j], label=f"Branch {j + 1}")
    ax_g.axhline(0.0, linestyle="--", linewidth=1, color="gray")

    for j in range(omega.shape[1]):
        ax_f.plot(V, omega[:, j], label=f"Branch {j + 1}")

    Vf_list_plot = []
    for j in range(g.shape[1]):
        Vf_j_m_s, _ = vgvf_plotting.first_flutter_crossing(V_m_s, g[:, j])
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
    ax_lam.set_ylabel(r"Log decrement $\Lambda = 2\pi(\sigma/\omega)$ [–]")
    ax_lam.set_title("Flutter – V–Λ")
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
        print(f"Combined V–Λ / V–g / V–f plot saved to {combined_outfile}")

    if show_plot:
        plt.show()
    else:
        plt.close()


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
    
    # Plot V-g (left subplot)
    for j in range(g.shape[1]):
        ax1.plot(V, g[:, j], label=f"Branch {j+1}")
    ax1.axhline(0.0, linestyle="--", linewidth=1, color='gray')
    
    # Annotate flutter speed on V-g plot (same display units as x-axis)
    Vf_list_plot = []
    for j in range(g.shape[1]):
        Vf_j_m_s, _ = vgvf_plotting.first_flutter_crossing(V_m_s, g[:, j])
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
    for j in range(omega.shape[1]):
        ax2.plot(V, omega[:, j], label=f"Branch {j+1}")
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


def _wet_spectrum_from_velocity_entry(entry, omega_n, freq_margin):
    """Return classified wet spectrum for one velocity step."""
    if entry.get("wet_spectrum") is not None:
        return entry["wet_spectrum"]

    from . import pk_method

    primary_mode = 0
    modes_list = entry.get("modes", [])
    if modes_list:
        primary_mode = int(modes_list[0].get("mode", 0))

    for md in modes_list:
        if md.get("mode") != primary_mode or not md.get("converged", False):
            continue
        wet_vals = md.get("wet_vals_all")
        if wet_vals is None:
            wet_vals = md.get("wet_evals_first3")
        if wet_vals is None:
            continue
        return pk_method.build_wet_spectrum_summary(
            wet_vals, omega_n, modes_list, freq_margin
        )
    return None


def plot_pk_converged_wet_eigenvectors(flutter_results, config=None, save_path="", show_plot=False):
    """
    Plot wet roots that the PK loop does not track (orphan / untracked branches).

    At each airspeed, the full sorted wet spectrum is taken from the converged state
    of structural mode 0. Roots already matched to a converged PK branch are excluded;
    the remaining roots (typically the lowest-frequency one at low speed) are plotted
    as ω(V) and g(V) trends alongside the PK-tracked branches for reference.
    """
    print("\n--- Generating PK Untracked Wet-Root Trend Plot ---")

    if config is not None and not getattr(config, "plot_pk_wet_eigenvectors", False):
        print("PK untracked-root plot disabled (plot_pk_wet_eigenvectors=False).")
        return

    raw_results = getattr(flutter_results, "raw_results", None)
    if not raw_results:
        print("No raw PK results available for untracked-root plot.")
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

    # PK-tracked branches (from flutter_results arrays)
    V_pk = np.asarray(flutter_results.velocities, dtype=float)
    omega_pk = np.asarray(flutter_results.frequencies, dtype=float)
    g_pk = np.asarray(flutter_results.damping, dtype=float)
    for j in range(omega_pk.shape[1] if omega_pk.ndim > 1 else 1):
        om_col = omega_pk[:, j] if omega_pk.ndim > 1 else omega_pk
        g_col = g_pk[:, j] if g_pk.ndim > 1 else g_pk
        ax_omega.plot(V_pk, om_col, "--", linewidth=1.2, alpha=0.7, label=f"PK branch {j + 1}")
        ax_g.plot(V_pk, g_col, "--", linewidth=1.2, alpha=0.7, label=f"PK branch {j + 1}")

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
        f"{case_name} — untracked wet roots (not selected by PK mode matching)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    outfile = os.path.join(save_path, "pk_untracked_wet_roots.png") if save_path else "pk_untracked_wet_roots.png"
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    print(f"PK untracked-root plot saved to {outfile}")

    if csv_rows:
        import pandas as pd

        csv_path = os.path.join(save_path, "pk_untracked_wet_roots.csv") if save_path else "pk_untracked_wet_roots.csv"
        pd.DataFrame(csv_rows).to_csv(csv_path, index=False, float_format="%.6e")
        print(f"PK untracked-root data saved to {csv_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


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


def plot_aero_beam_model(aerogrid, beam_model, config=None):

    fig = plt.figure(figsize=(24, 16))
    
    # Use GridSpec with custom width ratios to give isometric view more space
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3, width_ratios=[1.5, 1, 1])
    
    # ── 4-panel layout: isometric (3-D), front (X-Z), side (Y-Z), top (X-Y) ──
    ax_iso   = fig.add_subplot(gs[:, 0], projection='3d')  # isometric (spans both rows)
    ax_front = fig.add_subplot(gs[0, 1])                    # front view  (X → chord, Z → altitude)
    ax_side  = fig.add_subplot(gs[0, 2])                    # side view   (Y → span,  Z → altitude)
    ax_top   = fig.add_subplot(gs[1, 1:])                   # top/plan    (spans two columns at bottom)

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
                ax3d.plot(c[:, 0], c[:, 1], c[:, 2], color='lightblue', linewidth=0.7)
                ax_fr.plot(c[:, 0], c[:, 2], color='lightblue', linewidth=0.7)   # X vs Z
                ax_si.plot(c[:, 1], c[:, 2], color='lightblue', linewidth=0.7)   # Y vs Z
                ax_tp.plot(c[:, 1], c[:, 0], color='lightblue', linewidth=0.7)   # Y vs X (rotated 90°)
                
                # Compute and draw panel normals
                normal = _compute_panel_normal(np.array(coords))
                if normal is not None:
                    panel_center = np.mean(coords, axis=0)
                    normal_end = panel_center + normal_scale * normal
                    
                    # Draw normal on 3D view
                    ax3d.quiver(panel_center[0], panel_center[1], panel_center[2],
                               normal[0], normal[1], normal[2],
                               color='orange', arrow_length_ratio=0.4, linewidth=0.5, alpha=0.6, length=normal_scale)
                    
                    # Draw normals on 2D views
                    ax_fr.arrow(panel_center[0], panel_center[2],
                               normal[0]*normal_scale, normal[2]*normal_scale,
                               head_width=0.02, head_length=0.02, fc='orange', ec='orange', alpha=0.6, linewidth=0.5)
                    ax_si.arrow(panel_center[1], panel_center[2],
                               normal[1]*normal_scale, normal[2]*normal_scale,
                               head_width=0.02, head_length=0.02, fc='orange', ec='orange', alpha=0.6, linewidth=0.5)
                    ax_tp.arrow(panel_center[1], panel_center[0],
                               normal[1]*normal_scale, normal[0]*normal_scale,
                               head_width=0.02, head_length=0.02, fc='orange', ec='orange', alpha=0.6, linewidth=0.5)

    def _draw_ctrl_pts(ax3d, ax_fr, ax_si, ax_tp, pts, color, marker, size, label):
        ax3d.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                     c=color, marker=marker, s=size, label=label, alpha=0.8)
        ax_fr.scatter(pts[:, 0], pts[:, 2], c=color, marker=marker, s=size, alpha=0.8)
        ax_si.scatter(pts[:, 1], pts[:, 2], c=color, marker=marker, s=size, alpha=0.8)
        ax_tp.scatter(pts[:, 1], pts[:, 0], c=color, marker=marker, s=size, alpha=0.8)

    def _draw_beam(ax3d, ax_fr, ax_si, ax_tp, nodes, beam_model, sc_nodes=None, cog_nodes=None, case_name=None):
        elements_plotted = False
        for elem in beam_model['elements']:
            node_indices = elem['nodes']
            if len(node_indices) == 2:
                n1, n2 = node_indices
                ec = np.array([nodes[n1], nodes[n2]])
                kw3d = dict(linewidth=2, label='Beam (EA)' if not elements_plotted else '_nolegend_')
                ax3d.plot(ec[:, 0], ec[:, 1], ec[:, 2], 'r-', **kw3d)
                ax_fr.plot(ec[:, 0], ec[:, 2], 'r-', linewidth=2,
                           label='Beam (EA)' if not elements_plotted else '_nolegend_')
                ax_si.plot(ec[:, 1], ec[:, 2], 'r-', linewidth=2,
                           label='Beam (EA)' if not elements_plotted else '_nolegend_')
                ax_tp.plot(ec[:, 1], ec[:, 0], 'r-', linewidth=2,
                           label='Beam (EA)' if not elements_plotted else '_nolegend_')
                elements_plotted = True

        ax3d.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2],
                     c='r', marker='o', s=10, alpha=0.5)
        ax_fr.scatter(nodes[:, 0], nodes[:, 2], c='r', marker='o', s=10, alpha=0.5)
        ax_si.scatter(nodes[:, 1], nodes[:, 2], c='r', marker='o', s=10, alpha=0.5)
        ax_tp.scatter(nodes[:, 1], nodes[:, 0], c='r', marker='o', s=10, alpha=0.5)

        # ── For ABRAMSON1965: Draw three axes ────────────────────────────────
        if case_name == 'ABRAMSON1965':
            # 1. Shear Center axis (green)
            if sc_nodes is not None and len(sc_nodes) > 0:
                sc_nodes = np.array(sc_nodes)
                for i in range(len(sc_nodes) - 1):
                    sc_elem = np.array([sc_nodes[i], sc_nodes[i+1]])
                    ax3d.plot(sc_elem[:, 0], sc_elem[:, 1], sc_elem[:, 2], 'g-', linewidth=2.5, 
                             label='Shear Center' if i == 0 else '_nolegend_', alpha=0.9)
                    ax_fr.plot(sc_elem[:, 0], sc_elem[:, 2], 'g-', linewidth=2.5,
                              label='Shear Center' if i == 0 else '_nolegend_', alpha=0.9)
                    ax_si.plot(sc_elem[:, 1], sc_elem[:, 2], 'g-', linewidth=2.5,
                              label='Shear Center' if i == 0 else '_nolegend_', alpha=0.9)
                    ax_tp.plot(sc_elem[:, 1], sc_elem[:, 0], 'g-', linewidth=2.5,
                              label='Shear Center' if i == 0 else '_nolegend_', alpha=0.9)
                
                ax3d.scatter(sc_nodes[:, 0], sc_nodes[:, 1], sc_nodes[:, 2],
                            c='g', marker='^', s=15, alpha=0.9)
            
            # 2. Center of Gravity axis (blue)
            if cog_nodes is not None and len(cog_nodes) > 0:
                cog_nodes = np.array(cog_nodes)
                for i in range(len(cog_nodes) - 1):
                    cog_elem = np.array([cog_nodes[i], cog_nodes[i+1]])
                    ax3d.plot(cog_elem[:, 0], cog_elem[:, 1], cog_elem[:, 2], 'b-', linewidth=2.5, 
                             label='Center of Gravity' if i == 0 else '_nolegend_', alpha=0.9)
                    ax_fr.plot(cog_elem[:, 0], cog_elem[:, 2], 'b-', linewidth=2.5,
                              label='Center of Gravity' if i == 0 else '_nolegend_', alpha=0.9)
                    ax_si.plot(cog_elem[:, 1], cog_elem[:, 2], 'b-', linewidth=2.5,
                              label='Center of Gravity' if i == 0 else '_nolegend_', alpha=0.9)
                    ax_tp.plot(cog_elem[:, 1], cog_elem[:, 0], 'b-', linewidth=2.5,
                              label='Center of Gravity' if i == 0 else '_nolegend_', alpha=0.9)
                
                ax3d.scatter(cog_nodes[:, 0], cog_nodes[:, 1], cog_nodes[:, 2],
                            c='b', marker='s', s=15, alpha=0.9)
        
        # ── For other cases: Draw shear center beam if provided ───────────────
        elif config.name != 'tnz_multibody':
            if sc_nodes is not None and len(sc_nodes) > 0:
                sc_nodes = np.array(sc_nodes)
                # Connect shear center nodes with lines
                for i in range(len(sc_nodes) - 1):
                    sc_elem = np.array([sc_nodes[i], sc_nodes[i+1]])
                    ax3d.plot(sc_elem[:, 0], sc_elem[:, 1], sc_elem[:, 2], 'g--', linewidth=2.5, 
                             label='Shear Center' if i == 0 else '_nolegend_', alpha=0.8)
                    ax_fr.plot(sc_elem[:, 0], sc_elem[:, 2], 'g--', linewidth=2.5,
                              label='Shear Center' if i == 0 else '_nolegend_', alpha=0.8)
                    ax_si.plot(sc_elem[:, 1], sc_elem[:, 2], 'g--', linewidth=2.5,
                              label='Shear Center' if i == 0 else '_nolegend_', alpha=0.8)
                    ax_tp.plot(sc_elem[:, 1], sc_elem[:, 0], 'g--', linewidth=2.5,
                              label='Shear Center' if i == 0 else '_nolegend_', alpha=0.8)

                # Scatter shear center nodes
                ax3d.scatter(sc_nodes[:, 0], sc_nodes[:, 1], sc_nodes[:, 2],
                            c='g', marker='^', s=20, alpha=0.8)
                ax_fr.scatter(sc_nodes[:, 0], sc_nodes[:, 2], c='g', marker='^', s=20, alpha=0.8)
                ax_si.scatter(sc_nodes[:, 1], sc_nodes[:, 2], c='g', marker='^', s=20, alpha=0.8)
                ax_tp.scatter(sc_nodes[:, 1], sc_nodes[:, 0], c='g', marker='^', s=20, alpha=0.8)

    # ── Plot Aerodynamic Grid ────────────────────────────────────────────────
    if aerogrid:
        grid_pts = aerogrid['cornerpoint_grids'][:, 1:4]
        panels   = aerogrid['cornerpoint_panels']
        grid_ids = aerogrid['cornerpoint_grids'][:, 0]

        _draw_aero(ax_iso, ax_front, ax_side, ax_top, grid_pts, panels, grid_ids)

        if 'offset_j' in aerogrid:
            oj = np.asarray(aerogrid['offset_j'])
            if oj.size > 0:
                _draw_ctrl_pts(ax_iso, ax_front, ax_side, ax_top,
                               oj, 'magenta', 'x', 40,
                               'Control pts (offset_j @ 75%c)')
        if 'offset_l' in aerogrid:
            ol = np.asarray(aerogrid['offset_l'])
            if ol.size > 0:
                _draw_ctrl_pts(ax_iso, ax_front, ax_side, ax_top,
                               ol, 'green', '^', 40,
                               'Force pts (offset_l @ 25%c)')

    # ── Plot Beam Model ──────────────────────────────────────────────────────
    if beam_model:
        node_coords = [node['position'] for node in beam_model['nodes']]
        nodes = np.array(node_coords)

        print("\n[DEBUG] post_processing.py: Shape of the final 'nodes' array for plotting:", nodes.shape)
        print("[DEBUG] post_processing.py: ALL nodes for plotting:")
        for i in range(nodes.shape[0]):
            print(f"  Node {i}: {nodes[i]}")

        # Compute shear center nodes and CoG nodes (ONLY for ABRAMSON1965)
        sc_nodes = None
        cog_nodes = None
        case_name = config.name if config else None
        
        if case_name == 'ABRAMSON1965' and config and hasattr(config, 'xea_factor') and hasattr(config, 'xcm_factor') and hasattr(config, 'chord'):
            xea_offset = config.xea_factor * config.chord  # Shear center position from LE
            xcm_offset = config.xcm_factor * config.chord  # Center of Gravity position from LE
            
            print(f"\n[DEBUG] ABRAMSON1965 detected:")
            print(f"  xea_factor = {config.xea_factor}, xea_offset = {xea_offset:.6f} m from LE")
            print(f"  xcm_factor = {config.xcm_factor}, xcm_offset = {xcm_offset:.6f} m from LE")
            print(f"  Offset (CoG - EA) = {xcm_offset - xea_offset:.6f} m")
            print(f"  IMPORTANT: Aerogrid is centered at x=0 at CoG (xcm_offset)")
            print(f"    → LE is at x = -{xcm_offset:.6f} m")
            print(f"    → EA (SC) is at x = {xea_offset - xcm_offset:.6f} m (relative to CoG center)")
            
            # In aerogrid coordinate system: x=0 is at CoG
            # Beam nodes are positioned at SC (xea_offset from LE in global coords)
            # We need to express them in aerogrid coords: x_aero = x_global - xcm_offset
            
            # Convert beam nodes to aerogrid coordinates
            nodes_aero = nodes.copy()
            nodes_aero[:, 0] = nodes[:, 0] - xcm_offset
            
            # Build shear center nodes (identical to beam nodes since beam is at SC)
            sc_nodes = nodes_aero.copy()
            
            # Build CoG nodes (in aerogrid coordinates, x=0 always)
            cog_nodes = nodes_aero.copy()
            cog_nodes[:, 0] = 0.0
            
            if len(sc_nodes) > 0:
                print(f"[DEBUG] Computed {len(sc_nodes)} shear center nodes and {len(cog_nodes)} CoG nodes")
                print(f"[DEBUG] First SC node in aerogrid coords: {sc_nodes[0]}")
                print(f"[DEBUG] First CoG node in aerogrid coords: {cog_nodes[0]}")
            
            # Use aerogrid-transformed nodes for plotting
            nodes = nodes_aero
        
        elif config and hasattr(config, 'xea_factor') and hasattr(config, 'chord'):
            # For other cases (backward compatibility)
            xea_offset = (config.xea_factor - 0.5) * config.chord  # Offset from midchord (LE is at x=0)
            sc_nodes = []
            for node in nodes:
                # Node format: [x, y, z]
                # Shear center is offset in x direction by xea_offset
                sc_node = np.array([node[0] + xea_offset, node[1], node[2]])
                sc_nodes.append(sc_node)
            if len(sc_nodes) > 0:
                print(f"[DEBUG] post_processing.py: Computed {len(sc_nodes)} shear center nodes with xea_offset={xea_offset:.6f}")

        _draw_beam(ax_iso, ax_front, ax_side, ax_top, nodes, beam_model, sc_nodes, cog_nodes, case_name)

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
    ax_iso.set_xlabel('X'); ax_iso.set_ylabel('Y'); ax_iso.set_zlabel('Z')
    ax_iso.set_title('Isometric view', fontsize=12, fontweight='bold')
    ax_iso.legend(fontsize=8, loc='upper left')
    _set_equal_3d(ax_iso)

    ax_front.set_xlabel('X (chord)'); ax_front.set_ylabel('Z (height)')
    ax_front.set_title('Front view  (X–Z)', fontsize=12, fontweight='bold')
    ax_front.legend(fontsize=8, loc='upper left'); ax_front.grid(True, alpha=0.3)
    _set_equal_2d(ax_front)

    ax_side.set_xlabel('Y (span)'); ax_side.set_ylabel('Z (height)')
    ax_side.set_title('Side view  (Y–Z)', fontsize=12, fontweight='bold')
    ax_side.legend(fontsize=8, loc='upper left'); ax_side.grid(True, alpha=0.3)
    _set_equal_2d(ax_side)

    ax_top.set_xlabel('Y (span)'); ax_top.set_ylabel('X (chord)')
    ax_top.set_title('Top view', fontsize=12, fontweight='bold')
    ax_top.legend(fontsize=8, loc='upper left'); ax_top.grid(True, alpha=0.3)
    ax_top.invert_yaxis()  # Invert Y-axis to show top view (not bottom view)
    _set_equal_2d(ax_top)

    fig.suptitle('Aerodynamic Grid and Beam Model', fontsize=14, fontweight='bold')

    # ── Save ────────────────────────────────────────────────────────────────
    if config and config.save_plots:
        output_dir = config.output_dir
        os.makedirs(output_dir, exist_ok=True)
        outfile = os.path.join(output_dir, "aero_beam_model.png")
        plt.savefig(outfile, dpi=300, bbox_inches='tight')
        print(f"Aero-beam model plot saved to {outfile}")

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
    
    # Helper: per-velocity list of 2×2 matrices (one per converged structural slot).
    def extract_entry(contrib_list, key, i, j, mode_slot=None, default=0.0):
        """
        Extract [i,j] from contribution matrices at each velocity.

        When ``mode_slot`` is set (0 or 1), use the matrix stored at the end of that
        mode's PK solve (Q_modal evaluated at that mode's converged k, p). When None,
        average across slots (legacy behaviour).
        """
        values = []
        for c in contrib_list:
            if key not in c:
                values.append(float(default))
                continue
            matrices = c[key]
            if isinstance(matrices, list):
                if mode_slot is not None and mode_slot < len(matrices):
                    values.append(float(matrices[mode_slot][i, j]))
                elif len(matrices) > 0:
                    values.append(float(np.mean([mat[i, j] for mat in matrices])))
                else:
                    values.append(float(default))
            else:
                values.append(float(matrices[i, j]))
        return values

    matrix_components = {
        'M': [
            ('structural', 'M_struct', '+'),
            ('aerodynamic_B2', 'M_aero', '-'),
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
            for i, j in entry_indices:
                mode_slot = i if i == j else i
                data[family][label][(i, j)] = extract_entry(
                    contributions, key, i, j, mode_slot=mode_slot
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

    _plot_family(
        axes[0],
        'M',
        'Mass contributions: M_eff = M_struct - M_aero + M_hydro',
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
        output_dir = os.path.join(config.output_dir, getattr(config, 'name', ''))
        os.makedirs(output_dir, exist_ok=True)
        save_path = output_dir
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

