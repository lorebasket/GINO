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

            # Damping ratio g = -σ/ω  (standard V-g convention)
            g[iV, j] =  sigma / omega
            ω[iV, j] = omega

    return V, g, ω

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

def plot_vg(V, g, title="V–g Diagram", outfile=None, annotate=True):
    plt.figure(figsize=(9,5))
    for j in range(g.shape[1]):
        plt.plot(V, g[:, j], label=f"Branch {j+1}")
    plt.axhline(0.0, linestyle="--", linewidth=1)
    if annotate:
        Vf_list = []
        for j in range(g.shape[1]):
            Vf, _ = first_flutter_crossing(V, g[:, j])
            if Vf is not None:
                Vf_list.append(Vf)
        if Vf_list:
            Vf = min(Vf_list)
            plt.axvline(Vf, linestyle="--", linewidth=1)
            plt.text(Vf, plt.ylim()[1]*0.85, f"Vf ≈ {Vf:.1f} m/s", rotation=90,
                        ha="right", va="center")
    plt.xlabel("Airspeed V [m/s]")
    plt.ylabel("Damping ratio $\\zeta$ [-]")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if outfile:
        plt.savefig(outfile, dpi=150)
    plt.show()



def plot_vf(V, ω, title="V–ω Diagram", outfile=None, Vf=None):
    plt.figure(figsize=(8,4))
    for j in range(ω.shape[1]):
        plt.plot(V, ω[:, j], label=f"Branch {j+1}")
    if Vf is not None:
        plt.axvline(Vf, linestyle="--", linewidth=1)
        plt.text(Vf, plt.ylim()[1]*0.85, f"Vf ≈ {Vf:.1f} m/s", rotation=90,
                    ha="right", va="center")
    plt.xlabel("Airspeed V [m/s]")
    # ω is returned in rad/s. If you want cycles/s (Hz) use ω/(2π) and label [Hz].
    plt.ylabel("Frequency ω [rad/s]")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if outfile:
        plt.savefig(outfile, dpi=150)
    plt.show()