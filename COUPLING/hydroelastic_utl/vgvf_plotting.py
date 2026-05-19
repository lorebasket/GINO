import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment


# == PLOTTING == #
def results_to_arrays(results, mode_indices, omega_min=1e-2):
    V = np.array([r['V'] for r in results], dtype=float)
    n_modes = len(mode_indices)
    g = np.full((len(V), n_modes), np.nan, dtype=float)
    ω = np.full((len(V), n_modes), np.nan, dtype=float)
    # Build arrays by explicitly mapping tracked mode indices (mode_indices)
    # to the per-velocity 'modes' entries. This avoids relying on the
    # enumeration order inside each results entry which may not be stable.
    for iV, rV in enumerate(results):
        modes_list = rV.get('modes', [])

        # Precompute candidate sigmas/omegas for convenience
        cand_sigma = []
        cand_omega = []

        for md in modes_list:
            p = md.get('p', None)
            if p is not None:
                cand_sigma.append(float(np.real(p)))
                cand_omega.append(float(abs(np.imag(p))))
            else:
                cand_sigma.append(float(md.get('sigma', np.nan)))
                cand_omega.append(float(md.get('omega', np.nan)))

        cand_sigma = np.array(cand_sigma, dtype=float)
        cand_omega = np.array(cand_omega, dtype=float)

        for j, s in enumerate(mode_indices):
            selected_idx = None

            # 1) Try to find an explicit mode identifier in the stored dict
            for k, md in enumerate(modes_list):
                if 'mode' in md and md['mode'] == s:
                    selected_idx = k
                    break
                if 's' in md and md['s'] == s:
                    selected_idx = k
                    break

            # 2) Fallback: choose the candidate closest to previous velocity's value
            if selected_idx is None and iV > 0 and np.isfinite(ω[iV-1, j]):
                ref = ω[iV-1, j]
                if cand_omega.size > 0 and np.any(np.isfinite(cand_omega)):
                    diffs = np.abs(cand_omega - ref)
                    selected_idx = int(np.nanargmin(diffs))

            # 3) Fallback: choose the candidate closest to structural/dry frequency
            if selected_idx is None and 'dry_vals' in rV:
                try:
                    dry_vals = rV['dry_vals']
                    if j < len(dry_vals) and np.isfinite(dry_vals[j]):
                        ref = abs(np.imag(dry_vals[j])) if np.iscomplexobj(dry_vals[j]) else dry_vals[j]
                        if cand_omega.size > 0:
                            diffs = np.abs(cand_omega - ref)
                            selected_idx = int(np.nanargmin(diffs))
                except Exception:
                    pass

            # 4) Last resort: pick same index j if available
            if selected_idx is None:
                if j < cand_omega.size:
                    selected_idx = j
                else:
                    # nothing to assign for this mode at this V
                    continue

            sigma = cand_sigma[selected_idx]
            omega = cand_omega[selected_idx]

            if not (np.isfinite(sigma) and np.isfinite(omega)):
                continue
            if omega < omega_min:
                continue

            # Damping ratio g = σ/ω  (standard V-g convention)
            g[iV, j] = sigma / omega
            ω[iV, j] = omega

    return V, g, ω


def damping_ratio_from_storage(damping, omega_rad, *, dimensionless_vgvf_results=True):
    """
    Damping ratio ζ = σ/ω from stored PK arrays.

    If ``dimensionless_vgvf_results`` is True, ``damping`` is already σ/ω.
    Otherwise ``damping`` is σ [rad/s] and is divided by ``omega_rad``.
    """
    zeta = np.asarray(damping, dtype=float)
    if not dimensionless_vgvf_results:
        omega_rad = np.maximum(np.asarray(omega_rad, dtype=float), 1e-12)
        zeta = zeta / omega_rad
    return zeta


def log_decrement_from_storage(damping, omega_rad, *, dimensionless_vgvf_results=True):
    """
    Logarithmic decrement per cycle Λ ≈ 2π ζ = 2π (σ/ω).

    For σ > 0 (stable in the PK convention used here), Λ > 0 (increasing motion),
    aligning with Abramson's experimental δ trend (positive below flutter).
    """
    zeta = damping_ratio_from_storage(
        damping, omega_rad, dimensionless_vgvf_results=dimensionless_vgvf_results
    )
    return 2.0 * np.pi * zeta


def first_flutter_crossing(V, g_row):
    for i in range(len(V)-1):
        g0, g1 = g_row[i], g_row[i+1]
        if np.isnan(g0) or np.isnan(g1):
            continue
        if g0 == 0.0:
            return V[i], i
        if g0 * g1 < 0.0:
            t = -g0 / (g1 - g0)
            Vf = V[i] + t * (V[i+1] - V[i])
            return Vf, i
    return None, None

def plot_vg(V, g, title="V–g Diagram", outfile=None, annotate=True, v_knots=False):
    V_plot = np.asarray(V, dtype=float)
    v_unit = "m/s"
    if v_knots:
        V_plot = V_plot * 1.94384
        v_unit = "kn"
    plt.figure(figsize=(9,5))
    for j in range(g.shape[1]):
        plt.plot(V_plot, g[:, j], label=f"Branch {j+1}")
    plt.axhline(0.0, linestyle="--", linewidth=1)
    if annotate:
        Vf_list = []
        for j in range(g.shape[1]):
            Vf, _ = first_flutter_crossing(V, g[:, j])
            if Vf is not None:
                Vf_list.append(Vf * 1.94384 if v_knots else Vf)
        if Vf_list:
            Vf = min(Vf_list)
            plt.axvline(Vf, linestyle="--", linewidth=1)
            plt.text(Vf, plt.ylim()[1]*0.85, f"Vf ≈ {Vf:.1f} {v_unit}", rotation=90,
                        ha="right", va="center")
    plt.xlabel("Airspeed V [knots]" if v_knots else "Airspeed V [m/s]")
    plt.ylabel("Damping ratio $\\zeta$ [-]")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if outfile:
        plt.savefig(outfile, dpi=150)
    plt.show()

def plot_vf(V, ω, title="V–ω Diagram", outfile=None, Vf=None, v_knots=False, vf_hertz=False):
    V_plot = np.asarray(V, dtype=float)
    omega_plot = np.asarray(ω, dtype=float)
    v_unit = "m/s"
    if v_knots:
        V_plot = V_plot * 1.94384
        v_unit = "kn"
    if vf_hertz:
        omega_plot = omega_plot / (2.0 * np.pi)
    Vf_plot = None
    if Vf is not None and np.isfinite(Vf):
        Vf_plot = float(Vf) * 1.94384 if v_knots else float(Vf)
    plt.figure(figsize=(8,4))
    for j in range(omega_plot.shape[1]):
        plt.plot(V_plot, omega_plot[:, j], label=f"Branch {j+1}")
    if Vf_plot is not None:
        plt.axvline(Vf_plot, linestyle="--", linewidth=1)
        plt.text(Vf_plot, plt.ylim()[1]*0.85, f"Vf ≈ {Vf_plot:.1f} {v_unit}", rotation=90,
                    ha="right", va="center")
    plt.xlabel("Airspeed V [knots]" if v_knots else "Airspeed V [m/s]")
    plt.ylabel("Frequency ω [Hz]" if vf_hertz else "Frequency ω [rad/s]")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if outfile:
        plt.savefig(outfile, dpi=150)
    plt.show()