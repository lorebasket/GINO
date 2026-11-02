"""
Compare Im(Q_modal) from PanelAero/DLM vs Abramson measured 3-D coefficients (curve 3).

At each reduced frequency k (or U = 1/k), evaluates Q_modal from the same coupling path as
the flutter solver and builds a 2×2 reference from measured L/M derivatives (mid-span, TR-4).
Reports Im overestimation factors Im(Q_DLM)/Im(Q_meas,calibrated).
"""

from __future__ import annotations

import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import abramson_measured_coefficients as amc


def compute_Q_modal_at_k(
    k: float,
    V: float,
    config,
    structural_results,
    coupling_results,
    aerogrid,
    *,
    mode_indices=(0, 1),
) -> np.ndarray:
    """
    Modal aerodynamic matrix Q_modal(k) [Nm × Nm] — same path as Roger RFA STEP B.
    """
    from Qjj.precompute_qjj import interp_qjj_from_disk_old

    k = float(k)
    V = max(float(V), 1e-6)
    n_modes = int(len(np.asarray(structural_results.dry_eigenvalues).ravel()))
    Phi = structural_results.dry_eigenvectors[:, :n_modes]

    Z = coupling_results["Z"]
    Z_qs = coupling_results["Z_qs"]
    Z_force = coupling_results.get("Z_force")
    if Z_force is None:
        raise ValueError("coupling_results must contain Z_force")

    panel_areas = aerogrid["A"]
    A_diag = np.diag(panel_areas)
    Z_force_modal = Z_force @ Phi

    Ma = V / float(config.c_sound[config.fluid])
    Qjj_k = interp_qjj_from_disk_old(
        config.qjj_dir,
        k,
        Ma,
        ma_interp_mode=getattr(config, "qjj_ma_interp_mode", "auto"),
        ma_low_threshold=float(getattr(config, "qjj_ma_low_threshold", 0.05)),
    )
    ik = 1j * k
    w_k = (ik * Z + Z_qs) @ Phi
    aero_k = Qjj_k @ w_k
    Qm = Z_force_modal.T @ A_diag @ aero_k

    idx = [int(i) for i in mode_indices]
    return np.asarray(Qm, dtype=complex)[np.ix_(idx, idx)]


def _modal_scale_from_dlm_and_measured(
    Q_dlm: np.ndarray,
    H_meas: np.ndarray,
    *,
    k_ref_label: str = "inv_k=1.0",
    min_im: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calibrate measured H → Q units using |Q_DLM_ij|/|H_ij| at reference k.

    Returns (Q_meas_calibrated, scale_matrix).
    """
    scale = np.zeros((2, 2), dtype=float)
    for i in range(2):
        for j in range(2):
            h_ij = H_meas[i, j]
            q_ij = Q_dlm[i, j]
            den = max(abs(h_ij), 1e-30)
            scale[i, j] = abs(q_ij) / den
    Q_meas = scale * H_meas
    return Q_meas, scale


def build_comparison_table(
    config,
    structural_results,
    coupling_results,
    aerogrid,
    *,
    inv_k_list: np.ndarray | None = None,
    V_for_k: float = 15.0,
    mode_indices=(0, 1),
    measured_csv: str | None = None,
) -> pd.DataFrame:
    """
    Tabulate Im(Q) DLM vs measured-calibrated at common k = 1/inv_k.

    ``V_for_k`` is the airspeed used in Qjj Mach interpolation (low Ma → weak V dependence).
    """
    df_meas = amc.load_measured_coefficients_span50(measured_csv)
    if inv_k_list is None:
        inv_k_list = df_meas["inv_k"].to_numpy(dtype=float)
        inv_k_list = inv_k_list[inv_k_list > 1e-9]

    b = float(config.chord) / 2.0
    rho = float(config.rho_f[config.fluid])
    inv_k_from_k = lambda k: amc.inv_k_from_k(k, b)
    k_from_inv_k = lambda u: amc.k_from_inv_k(u, b)
    omega_b = float(np.sqrt(abs(structural_results.dry_eigenvalues[mode_indices[0]])))
    omega_p = float(np.sqrt(abs(structural_results.dry_eigenvalues[mode_indices[1]])))

    # Fixed magnitude calibration at U = 1 (k = 1): maps dimensionless H → Q_modal units.
    inv_k_cal = 1.0
    k_cal = k_from_inv_k(inv_k_cal)
    H_cal = amc.interpolate_h_matrix(df_meas, inv_k_cal)
    Q_cal = compute_Q_modal_at_k(
        k_cal, V_for_k, config, structural_results, coupling_results, aerogrid,
        mode_indices=mode_indices,
    )
    _, scale_cal = _modal_scale_from_dlm_and_measured(Q_cal, H_cal)

    rows = []
    for inv_k in inv_k_list:
        inv_k = float(inv_k)
        k = k_from_inv_k(inv_k)
        if not np.isfinite(k) or k > 60.0 or k < 1e-4:
            continue

        H = amc.interpolate_h_matrix(df_meas, inv_k)
        Q_dlm = compute_Q_modal_at_k(
            k, V_for_k, config, structural_results, coupling_results, aerogrid,
            mode_indices=mode_indices,
        )

        Q_meas_k = scale_cal * H
        scale = scale_cal

        def _mag_ratio(im_dlm, im_m):
            den = abs(im_m)
            if den < 1e-12:
                return np.nan
            return float(abs(im_dlm) / den)

        im00_d, im00_m = np.imag(Q_dlm[0, 0]), np.imag(Q_meas_k[0, 0])
        im11_d, im11_m = np.imag(Q_dlm[1, 1]), np.imag(Q_meas_k[1, 1])

        rows.append(
            {
                "inv_k": inv_k,
                "k_rad_per_m": k,
                "V_for_Qjj_m_s": V_for_k,
                "omega_bend_rad_s": omega_b,
                "omega_pitch_rad_s": omega_p,
                "Im_Q00_DLM": im00_d,
                "Im_Q11_DLM": im11_d,
                "Im_Q01_DLM": np.imag(Q_dlm[0, 1]),
                "Im_Q10_DLM": np.imag(Q_dlm[1, 0]),
                "Im_L_hh_meas": np.imag(H[0, 0]),
                "Im_M_aa_meas": np.imag(H[1, 1]),
                "Im_Q00_meas_cal": im00_m,
                "Im_Q11_meas_cal": im11_m,
                "scale_00": scale[0, 0],
                "scale_11": scale[1, 1],
                "ratio_Im_00_DLM_over_meas": (
                    float(im00_d / im00_m) if abs(im00_m) > 1e-12 else np.nan
                ),
                "ratio_Im_11_DLM_over_meas": (
                    float(im11_d / im11_m) if abs(im11_m) > 1e-12 else np.nan
                ),
                "ratio_abs_Im_00": _mag_ratio(im00_d, im00_m),
                "ratio_abs_Im_11": _mag_ratio(im11_d, im11_m),
                "sign_match_00": int(np.sign(im00_d) == np.sign(im00_m)),
                "sign_match_11": int(np.sign(im11_d) == np.sign(im11_m)),
                "C_aero_00_DLM_kg_s": 0.5 * rho * V_for_k / k * im00_d,
                "C_aero_11_DLM_kg_s": 0.5 * rho * V_for_k / k * im11_d,
            }
        )

    return pd.DataFrame(rows)


def _flutter_reference_k(
    V_kn: float = 48.1,
    f_hz: float = 17.5,
    b: float = 0.1524,
) -> tuple[float, float]:
    """Experimental flutter: (k [1/m], U) with U = V/(bω) and k = 1/(U·b)."""
    V = float(V_kn) * 0.514444
    omega = 2.0 * np.pi * float(f_hz)
    U = V / (float(b) * omega)
    k = 1.0 / (U * float(b))
    return k, U


def build_comparison_at_wet_k(
    config,
    structural_results,
    coupling_results,
    aerogrid,
    V_list: np.ndarray,
    omega_per_V: np.ndarray,
    *,
    mode_index: int = 0,
    measured_csv: str | None = None,
) -> pd.DataFrame:
    """
    Compare at k(V) = omega(V)/V (same reduced frequency as PK/RFA at each speed).
    """
    df_meas = amc.load_measured_coefficients_span50(measured_csv)
    b = float(config.chord) / 2.0
    k_from_inv_k = lambda u: amc.k_from_inv_k(u, b)
    inv_k_from_k = lambda k: amc.inv_k_from_k(k, b)
    _, scale_cal = _modal_scale_from_dlm_and_measured(
        compute_Q_modal_at_k(
            k_from_inv_k(1.0), 15.0, config, structural_results, coupling_results, aerogrid,
        ),
        amc.interpolate_h_matrix(df_meas, 1.0),
    )
    rho = float(config.rho_f[config.fluid])
    rows = []
    for V, omega in zip(V_list, omega_per_V):
        V = float(V)
        omega = float(omega)
        if V < 1e-6 or omega < 1e-6:
            continue
        k = omega / V
        inv_k = inv_k_from_k(k)
        H = amc.interpolate_h_matrix(df_meas, inv_k)
        Q_dlm = compute_Q_modal_at_k(
            k, V, config, structural_results, coupling_results, aerogrid,
        )
        Q_meas = scale_cal * H
        mi, mj = mode_index, mode_index
        im_d, im_m = np.imag(Q_dlm[mi, mi]), np.imag(Q_meas[mi, mi])
        rows.append(
            {
                "V_m_s": V,
                "k_at_V": k,
                "inv_k_at_V": inv_k,
                "omega_rad_s": omega,
                "Im_Q_DLM_diag": im_d,
                "Im_Q_meas_cal_diag": im_m,
                "ratio_abs_Im_diag": abs(im_d) / max(abs(im_m), 1e-12),
                "C_aero_DLM_diag_kg_s": 0.5 * rho * V / k * im_d,
            }
        )
    return pd.DataFrame(rows)


def summarize_overestimation(
    table: pd.DataFrame,
    *,
    inv_k_flutter: float | None = None,
) -> dict[str, Any]:
    """Median and flutter-point |Im(Q_DLM)|/|Im(Q_meas,cal)| for diagonals."""
    out: dict[str, Any] = {}
    for tag, col_abs, col_signed in (
        ("bb", "ratio_abs_Im_00", "ratio_Im_00_DLM_over_meas"),
        ("pp", "ratio_abs_Im_11", "ratio_Im_11_DLM_over_meas"),
    ):
        r = table[col_abs].replace([np.inf, -np.inf], np.nan).dropna()
        if r.size:
            out[f"median_ratio_abs_Im_{tag}"] = float(np.median(r))
            out[f"mean_ratio_abs_Im_{tag}"] = float(np.mean(r))
        if inv_k_flutter is not None and "inv_k" in table.columns:
            iv = int(np.argmin(np.abs(table["inv_k"].to_numpy() - float(inv_k_flutter))))
            out[f"ratio_abs_Im_{tag}_at_flutter_U"] = float(table[col_abs].iloc[iv])
            out[f"ratio_signed_Im_{tag}_at_flutter_U"] = float(table[col_signed].iloc[iv])
    return out


def plot_imq_comparison(
    table: pd.DataFrame,
    save_path: str,
    *,
    inv_k_flutter: float | None = 1.48,
    show_plot: bool = False,
) -> str:
    """Im(Q) DLM vs measured-calibrated vs |Im(L_hh)|, |Im(M_aa)|."""
    os.makedirs(save_path, exist_ok=True)
    inv_k = table["inv_k"].to_numpy()

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    ax = axes[0, 0]
    ax.plot(inv_k, table["Im_Q00_DLM"], "o-", ms=4, label="Im Q₀₀ DLM")
    ax.plot(inv_k, table["Im_Q00_meas_cal"], "s--", ms=4, label="Im Q₀₀ meas. (cal.)")
    ax.plot(inv_k, table["Im_L_hh_meas"], "^:", ms=3, alpha=0.7, label="Im L_hh raw")
    ax.set_xlabel("U = V/(bω)")
    ax.set_ylabel("Im (modal / coeff.)")
    ax.set_title("Bend–bend (mode 0)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(inv_k, table["Im_Q11_DLM"], "o-", ms=4, label="Im Q₁₁ DLM")
    ax.plot(inv_k, table["Im_Q11_meas_cal"], "s--", ms=4, label="Im Q₁₁ meas. (cal.)")
    ax.plot(inv_k, table["Im_M_aa_meas"], "^:", ms=3, alpha=0.7, label="Im M_αα raw")
    ax.set_xlabel("U = V/(bω)")
    ax.set_title("Pitch–pitch (mode 1)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.plot(inv_k, table["ratio_abs_Im_00"], "o-", ms=4, color="C0", label="|Im Q₀₀| ratio")
    ax.plot(inv_k, table["ratio_abs_Im_11"], "s-", ms=4, color="C1", label="|Im Q₁₁| ratio")
    ax.axhline(1.0, color="gray", ls="--", lw=1)
    if inv_k_flutter is not None:
        ax.axvline(float(inv_k_flutter), color="k", ls=":", lw=1, label=f"exp. flutter U≈{inv_k_flutter:g}")
    ax.set_xlabel("U = 1/k")
    ax.set_ylabel("|Im(Q_DLM)| / |Im(Q_meas,cal)|")
    ax.set_title("Rapporto moduli (scala lineare)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.plot(table["k_rad_per_m"], table["ratio_abs_Im_00"], "o-", ms=4, label="Q₀₀")
    ax.plot(table["k_rad_per_m"], table["ratio_abs_Im_11"], "s-", ms=4, label="Q₁₁")
    ax.axhline(1.0, color="gray", ls="--", lw=1)
    ax.set_xlabel("k = bω/V [1/m]")
    ax.set_title("Overestimation vs k")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    plt.tight_layout()
    outfile = os.path.join(save_path, "ABRAMSON1965_ImQ_DLM_vs_measured.png")
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    if show_plot:
        plt.show()
    else:
        plt.close()
    return outfile


def run_imq_comparison(
    config,
    structural_results,
    coupling_results,
    aerogrid,
    *,
    save_path: str | None = None,
    show_plot: bool = False,
    V_for_k: float = 15.0,
    measured_csv: str | None = None,
) -> dict[str, Any]:
    """
    Full pipeline: table CSV, plot PNG, printed summary.

    Returns dict with keys ``table``, ``summary``, ``csv_path``, ``png_path``.
    """
    save_path = save_path or getattr(config, "output_dir", ".")
    os.makedirs(save_path, exist_ok=True)

    table = build_comparison_table(
        config,
        structural_results,
        coupling_results,
        aerogrid,
        V_for_k=V_for_k,
        measured_csv=measured_csv,
    )
    csv_path = os.path.join(save_path, "ABRAMSON1965_ImQ_DLM_vs_measured.csv")
    table.to_csv(csv_path, index=False, float_format="%.6e")

    b = float(config.chord) / 2.0
    _, inv_k_f = _flutter_reference_k(b=b)
    summary = summarize_overestimation(table, inv_k_flutter=inv_k_f)
    summary["inv_k_experimental_flutter"] = inv_k_f
    summary["k_experimental_flutter"] = amc.k_from_inv_k(inv_k_f, b)
    summary["V_for_Qjj_m_s"] = V_for_k

    png_path = plot_imq_comparison(
        table, save_path, inv_k_flutter=inv_k_f, show_plot=show_plot,
    )

    table_kv_path = None
    vf_csv = os.path.join(save_path, "vf_data.csv")
    if os.path.isfile(vf_csv):
        vf = pd.read_csv(vf_csv)
        v_col = "Velocity_m_s" if "Velocity_m_s" in vf.columns else vf.columns[0]
        f_col = "Frequency_Hz_Branch_2" if "Frequency_Hz_Branch_2" in vf.columns else None
        if f_col and f_col in vf.columns:
            V_arr = vf[v_col].to_numpy(dtype=float)
            omega_arr = 2.0 * np.pi * vf[f_col].to_numpy(dtype=float)
            tbl_kv = build_comparison_at_wet_k(
                config, structural_results, coupling_results, aerogrid,
                V_arr, omega_arr, mode_index=0,
                measured_csv=measured_csv,
            )
            table_kv_path = os.path.join(save_path, "ABRAMSON1965_ImQ_DLM_vs_measured_at_kV.csv")
            tbl_kv.to_csv(table_kv_path, index=False, float_format="%.6e")
            if tbl_kv.size:
                r_kv = tbl_kv["ratio_abs_Im_diag"].replace([np.inf, -np.inf], np.nan).dropna()
                if r_kv.size:
                    summary["median_ratio_abs_Im_bb_at_kV"] = float(np.median(r_kv))
                    summary["max_ratio_abs_Im_bb_at_kV"] = float(np.max(r_kv))

    print("\n" + "=" * 70)
    print("  Im(Q) DLM vs Abramson measured coefficients (curve 3, mid-span 50 %)")
    print("=" * 70)
    print(f"  Table: {csv_path}")
    print(f"  Plot:  {png_path}")
    print("  Calibration: |Q_DLM_ij|/|H_ij| at U=1 (k=1) maps H → Q_modal units")
    print(f"  Experimental flutter U=V/(bω) ≈ {inv_k_f:.3f}  (k ≈ {1.0/inv_k_f:.3f} 1/m)")
    for key, val in summary.items():
        if isinstance(val, float):
            print(f"  {key}: {val:.4f}")

    r_abs = summary.get("ratio_abs_Im_bb_at_flutter_U", np.nan)
    r_med = summary.get("median_ratio_abs_Im_bb", np.nan)
    if np.isfinite(r_abs) and r_abs > 0:
        summary["suggested_aero_im_Q_scale_from_abs_Im"] = 1.0 / r_abs
        print(f"  |Im(Q00)|_DLM/|Im(Q00)|_meas @ U_flutter ≈ {r_abs:.3f}")
        print(f"  → aero_im_Q_scale ≈ 1/ratio_abs ≈ {1.0/r_abs:.3f} (if mismatch is Im-only)")
    elif np.isfinite(r_med) and r_med > 0:
        summary["suggested_aero_im_Q_scale_from_abs_Im"] = 1.0 / r_med
        print(f"  median |Im| ratio (bb) ≈ {r_med:.3f}")

    out = {
        "table": table,
        "summary": summary,
        "csv_path": csv_path,
        "png_path": png_path,
    }
    if table_kv_path:
        out["csv_path_at_kV"] = table_kv_path
        print(f"  Table k(V): {table_kv_path}")
    return out
