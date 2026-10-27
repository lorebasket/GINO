import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment


# == PLOTTING == #
def results_to_arrays(results, mode_indices, omega_min=1e-2):
    V = np.array([r['V'] for r in results], dtype=float)
    n_modes = len(mode_indices)
    g = np.full((len(V), n_modes), np.nan, dtype=float)
    ω = np.full((len(V), n_modes), np.nan, dtype=float)
    # Per-(V, mode) p-k convergence flag (see md['converged'] in
    # pk_method.pk_solve_at_velocity) -- False/NaN-defaulted so a caller that
    # ignores this array still gets well-defined damping/frequency values;
    # a non-converged point is whatever k/omega the relaxation loop landed
    # on at max_iter, not a self-consistent root (2026-09-21, see
    # docs/notes/gino_pk_convergence_filtering.md).
    conv = np.zeros((len(V), n_modes), dtype=bool)
    # Build arrays by explicitly mapping tracked mode indices (mode_indices)
    # to the per-velocity 'modes' entries. This avoids relying on the
    # enumeration order inside each results entry which may not be stable.
    for iV, rV in enumerate(results):
        modes_list = rV.get('modes', [])

        # Precompute candidate sigmas/omegas for convenience
        cand_sigma = []
        cand_omega = []
        cand_converged = []

        for md in modes_list:
            p = md.get('p', None)
            if p is not None:
                cand_sigma.append(float(np.real(p)))
                cand_omega.append(float(abs(np.imag(p))))
            else:
                cand_sigma.append(float(md.get('sigma', np.nan)))
                cand_omega.append(float(md.get('omega', np.nan)))
            cand_converged.append(bool(md.get('converged', False)))

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
            if selected_idx < len(cand_converged):
                conv[iV, j] = cand_converged[selected_idx]

    return V, g, ω, conv


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

    Stable PK roots (σ < 0) give Λ < 0.
    """
    zeta = damping_ratio_from_storage(
        damping, omega_rad, dimensionless_vgvf_results=dimensionless_vgvf_results
    )
    return 2.0 * np.pi * zeta


def abramson_decay_factor_from_storage(
    damping, omega_rad, *, dimensionless_vgvf_results=True
):
    """
    Decay factor δ [s⁻¹] with the same sign as Λ = 2π(σ/ω) (PK / V–g convention).

    δ = ζ ω = σ = Λ ω/(2π): stable roots (σ < 0) give δ < 0; flutter at δ = 0.
    Magnitude matches Abramson Fig. 5 units; sign is not flipped to paper's
    positive-stable convention.
    """
    zeta = damping_ratio_from_storage(
        damping, omega_rad, dimensionless_vgvf_results=dimensionless_vgvf_results
    )
    omega_rad = np.maximum(np.asarray(omega_rad, dtype=float), 1e-12)
    return zeta * omega_rad


def _crossing_on_segment(V, g_row, f_row, i):
    """Interpolate g=0 crossing on segment [i, i+1]; optional frequency at crossing [rad/s]."""
    g0, g1 = float(g_row[i]), float(g_row[i + 1])
    if np.isnan(g0) or np.isnan(g1):
        return None
    if g0 == 0.0:
        ff = float(f_row[i]) if f_row is not None else None
        return float(V[i]), i, ff
    if g0 * g1 >= 0.0:
        return None
    t = -g0 / (g1 - g0)
    Vf = float(V[i] + t * (V[i + 1] - V[i]))
    ff = None
    if f_row is not None:
        ff = float(f_row[i] + t * (f_row[i + 1] - f_row[i]))
    return Vf, i, ff


def all_flutter_crossings(V, g_row, f_row=None, *, branch=1):
    """All g=0 crossings along one branch (earliest to latest V)."""
    V = np.asarray(V, dtype=float)
    g_row = np.asarray(g_row, dtype=float)
    f_row = None if f_row is None else np.asarray(f_row, dtype=float)
    out = []
    for i in range(len(V) - 1):
        hit = _crossing_on_segment(V, g_row, f_row, i)
        if hit is None:
            continue
        Vf, idx, omega_rad = hit
        f_hz = omega_rad / (2.0 * np.pi) if omega_rad is not None else None
        out.append(
            {
                "Vf": Vf,
                "idx": idx,
                "omega_rad": omega_rad,
                "f_hz": f_hz,
                "branch": int(branch),
            }
        )
    return out


def first_flutter_crossing(V, g_row, f_row=None):
    """First g=0 crossing along V (legacy behaviour)."""
    V = np.asarray(V, dtype=float)
    g_row = np.asarray(g_row, dtype=float)
    f_row = None if f_row is None else np.asarray(f_row, dtype=float)
    for i in range(len(V) - 1):
        hit = _crossing_on_segment(V, g_row, f_row, i)
        if hit is not None:
            return hit[0], hit[1]
    return None, None


def select_primary_flutter(
    V,
    g,
    omega_rad_s=None,
    *,
    selection="min_V",
    freq_band_hz=None,
    freq_target_hz=None,
    V_target_kn=None,
):
    """
    Choose primary flutter speed/frequency from all branch crossings.

    selection:
      'min_V'            — earliest crossing on any branch (legacy default)
      'freq_band'        — in-band crossing closest to freq_target_hz; tiebreak min V
      'freq_band_f_first' — in-band: min |f−f_tgt|, then min |V−V_tgt|
      'freq_band_best'   — in-band: min |ΔV_kn| + 2|Δf| (joint Abramson target)
    """
    g = np.atleast_2d(np.asarray(g, dtype=float))
    omega = None
    if omega_rad_s is not None:
        omega = np.atleast_2d(np.asarray(omega_rad_s, dtype=float))

    crossings = []
    for j in range(g.shape[1]):
        f_col = omega[:, j] if omega is not None and omega.shape[1] > j else None
        crossings.extend(all_flutter_crossings(V, g[:, j], f_col, branch=j + 1))

    if not crossings:
        return None, None

    def _pick(pool):
        if freq_target_hz is not None:
            return min(
                pool,
                key=lambda c: (
                    abs((c["f_hz"] or 0.0) - float(freq_target_hz)),
                    c["Vf"],
                ),
            )
        return min(pool, key=lambda c: c["Vf"])

    if selection in ("freq_band", "freq_band_f_first") and freq_band_hz is not None:
        f_lo, f_hi = float(freq_band_hz[0]), float(freq_band_hz[1])
        in_band = [
            c for c in crossings
            if c["f_hz"] is not None and f_lo <= c["f_hz"] <= f_hi
        ]
        if in_band:
            if selection == "freq_band_f_first":
                v_tgt = float(V_target_kn if V_target_kn is not None else 48.1)
                f_tgt = float(freq_target_hz or 17.5)
                chosen = min(
                    in_band,
                    key=lambda c: (
                        abs(c["f_hz"] - f_tgt),
                        abs(c["Vf"] * 1.94384 - v_tgt),
                    ),
                )
            else:
                chosen = _pick(in_band)
            return chosen["Vf"], chosen["omega_rad"]

    if selection == "freq_band_best" and freq_band_hz is not None:
        f_lo, f_hi = float(freq_band_hz[0]), float(freq_band_hz[1])
        v_tgt = float(V_target_kn if V_target_kn is not None else 48.1)
        f_tgt = float(freq_target_hz or 17.5)
        in_band = [
            c for c in crossings
            if c["f_hz"] is not None and f_lo <= c["f_hz"] <= f_hi
        ]
        if in_band:
            chosen = min(
                in_band,
                key=lambda c: abs(c["Vf"] * 1.94384 - v_tgt) + 2.0 * abs(c["f_hz"] - f_tgt),
            )
            return chosen["Vf"], chosen["omega_rad"]

    if selection == "coalescence" and omega is not None and g.shape[1] >= 2:
        f_hz = omega / (2.0 * np.pi)
        sep = np.abs(f_hz[:, 0] - f_hz[:, 1])
        g_near = np.minimum(np.abs(g[:, 0]), np.abs(g[:, 1]))
        score = sep + 2.0 * g_near
        mask = np.isfinite(score)
        if np.any(mask):
            i_best = int(np.nanargmin(np.where(mask, score, np.inf)))
            # Refine: linear interp g→0 on branch with smaller |g| at i_best
            j = 0 if abs(g[i_best, 0]) <= abs(g[i_best, 1]) else 1
            if i_best < len(V) - 1:
                hit = _crossing_on_segment(V, g[:, j], omega[:, j], i_best)
                if hit is not None:
                    return hit[0], hit[2]
            omega_rad = float(omega[i_best, j])
            return float(V[i_best]), omega_rad

    chosen = min(crossings, key=lambda c: c["Vf"])
    return chosen["Vf"], chosen["omega_rad"]

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
    plt.figure(figsize=(8, 4))
    for j in range(omega_plot.shape[1]):
        plt.plot(V_plot, omega_plot[:, j], label=f"Branch {j+1}")
    if Vf_plot is not None:
        plt.axvline(Vf_plot, linestyle="--", linewidth=1)
        plt.text(
            Vf_plot, plt.ylim()[1] * 0.85, f"Vf ≈ {Vf_plot:.1f} {v_unit}",
            rotation=90, ha="right", va="center",
        )
    plt.xlabel("Airspeed V [knots]" if v_knots else "Airspeed V [m/s]")
    plt.ylabel("Frequency ω [Hz]" if vf_hertz else "Frequency ω [rad/s]")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if outfile:
        plt.savefig(outfile, dpi=150)
    plt.show()


def trim_branch_smooth_prefix(
    v,
    freq,
    damp=None,
    *,
    freq_jump_hz: float = 3.0,
    rel_freq_jump: float = 0.28,
    damp_g_jump: float = 1.2,
    damp_delta_jump: float = 6.0,
    min_v: float = 18.0,
    damp_is_log_decrement: bool = False,
):
    """
    Return (v, freq, damp) truncated before the first post-flutter PK tracking spike.

    ``damp`` may be PK g = σ/ω (dimensionless) or log decrement δ [s⁻¹] when
    ``damp_is_log_decrement=True``.
    """
    v = np.asarray(v, dtype=float)
    freq = np.asarray(freq, dtype=float)
    damp_arr = None if damp is None else np.asarray(damp, dtype=float)
    n = v.size
    if n < 2:
        return v, freq, damp_arr

    order = np.argsort(v)
    v = v[order]
    freq = freq[order]
    if damp_arr is not None:
        damp_arr = damp_arr[order]

    cut = n
    for i in range(1, n):
        if v[i] < min_v:
            continue
        df = abs(freq[i] - freq[i - 1])
        ref_f = max(abs(freq[i - 1]), 0.5)
        if df > freq_jump_hz or df / ref_f > rel_freq_jump:
            cut = i
            break
        if damp_arr is not None:
            dd = abs(damp_arr[i] - damp_arr[i - 1])
            thr = damp_delta_jump if damp_is_log_decrement else damp_g_jump
            if dd > thr:
                cut = i
                break

    if damp_arr is None:
        return v[:cut], freq[:cut], None
    return v[:cut], freq[:cut], damp_arr[:cut]