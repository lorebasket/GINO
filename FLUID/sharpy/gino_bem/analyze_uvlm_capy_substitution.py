"""
UVLM apparent-mass removal vs Capytaine reinjection — Abramson 1965.

Compares ``added_mass_removal`` modes (none / gamma_dot / full):
  - modal projection of UVLM apparent mass (D-matrix feedthrough)
  - Css gamma_dot and Dss zeta_dot block norms (what the patch zeros)
  - frequency-ratio probe A_jj(V) = (omega_gd/omega_none)^2 - 1
  - Capytaine A(omega) in the aligned SHARPy modal basis

Usage:
  conda activate sharpy && cd sharpy
  python gino_bem/analyze_uvlm_capy_substitution.py --coarse
  python gino_bem/analyze_uvlm_capy_substitution.py --grid n20c10
  python gino_bem/analyze_uvlm_capy_substitution.py --grid-span 20 --grid-chord 10
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_SHARPY_ROOT = Path(__file__).resolve().parents[1]
_FSI_ROOT = _SHARPY_ROOT.parents[1]
if str(_SHARPY_ROOT) not in sys.path:
    sys.path.insert(0, str(_SHARPY_ROOT))

from gino_bem.capy_bem_bridge import capy_at_omega, check_capy_basis, default_capytaine_npz, load_capytaine_npz  # noqa: E402
from gino_bem.mode_alignment import align_capy_to_sharpy, sharpy_modal_displacements, strip_added_mass_modal  # noqa: E402
from gino_bem.run_abramson_pk_bem import (  # noqa: E402
    GRIDS,
    _dry_omegas_phys,
    _probe_uvlm_added_mass_diag,
    build_case,
)
from gino_bem.track_structural_modes import beam_state_indices  # noqa: E402

KN = 0.514444
MODE_NAMES = ("bending", "torsion")


def _uvlm_block_norms(ls) -> dict[str, float]:
    sys_uvlm = ls.uvlm.sys
    K, K_star, Kzeta = sys_uvlm.K, sys_uvlm.K_star, sys_uvlm.Kzeta
    C = np.asarray(sys_uvlm.SS.C)
    D = np.asarray(
        sys_uvlm.D_predictor if sys_uvlm.D_predictor is not None else sys_uvlm.SS.D
    )
    return {
        "css_gamma_dot": float(np.linalg.norm(C[:, K + K_star: 2 * K + K_star])),
        "dss_zeta_dot": float(np.linalg.norm(D[:, 3 * Kzeta: 6 * Kzeta])),
    }


def _M_modal_from_D(ls, nm: int) -> np.ndarray:
    """Mass-normalized modal apparent-mass matrix from coupled q_dot -> Q feedthrough."""
    D = np.asarray(ls.ss.D, dtype=float)
    return -D[:nm, nm: 2 * nm]


def _assemble_configs(
    nm: int,
    grid: tuple[int, int, int],
    *,
    rayleigh: bool,
    zetas: tuple[float, float],
    v_max_kn: float,
) -> dict[str, tuple]:
    out = {}
    for removal in ("none", "gamma_dot", "full"):
        tag = f"am_probe_{removal}"
        data, _ = build_case(
            nm, grid=grid, added_mass_removal=removal,
            rayleigh=rayleigh, zetas=zetas, v_max_kn=v_max_kn, case_tag=tag,
        )
        out[removal] = (data, data.linear.linear_system)
    return out


def _probe_diag_vs_V(ls_none, ls_gd, nm: int, v_list: np.ndarray) -> list[dict]:
    beam_idx_n = beam_state_indices(ls_none.ss, nm)
    beam_idx_g = beam_state_indices(ls_gd.ss, nm)
    rows = []
    for V in v_list:
        if V <= 0:
            continue
        try:
            diag = _probe_uvlm_added_mass_diag(
                ls_none, ls_gd, beam_idx_n, beam_idx_g, nm, float(V),
            )
        except RuntimeError as exc:
            print(f"[probe] V={V:.1f} m/s skipped: {exc}", flush=True)
            continue
        row = {"V_m_s": float(V), "V_kn": float(V) / KN}
        for j in range(min(nm, 2)):
            row[f"A_uvlm_removed_{MODE_NAMES[j]}"] = float(diag[j])
        rows.append(row)
    return rows


def _capy_diagonal(T: np.ndarray, capy: dict, omega: float, nm: int) -> np.ndarray:
    A, _ = capy_at_omega(capy, omega, mode_slice=slice(0, nm), warn_out_of_grid=False)
    Tn = np.asarray(T, dtype=float)[:nm, :nm]
    A = Tn.T @ A @ Tn
    return np.diag(A)


def plot_matrix_panels(
    matrices: dict[str, np.ndarray],
    labels: dict[str, str],
    out_path: Path,
    *,
    title: str,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    keys = list(matrices.keys())
    n = len(keys)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.8), squeeze=False)
    all_vals = np.concatenate([np.asarray(M).ravel() for M in matrices.values()])
    if vmin is None:
        vmin = float(np.min(all_vals))
    if vmax is None:
        vmax = float(np.max(all_vals))
    lim = max(abs(vmin), abs(vmax), 1e-6)

    for ax, key in zip(axes[0], keys):
        M = np.asarray(matrices[key], dtype=float)
        im = ax.imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim, origin="upper")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=9,
                        color="k" if abs(M[i, j]) < 0.55 * lim else "w")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(MODE_NAMES[: M.shape[1]])
        ax.set_yticklabels(MODE_NAMES[: M.shape[0]])
        ax.set_title(labels.get(key, key), fontsize=10)
    fig.colorbar(im, ax=axes[0], fraction=0.025, pad=0.04, label=r"$A_{ij}$ [modal]")
    fig.suptitle(title, fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_diagonal_vs_V(
    probe_rows: list[dict],
    capy_diag: dict[str, np.ndarray],
    out_path: Path,
    *,
    title: str,
) -> None:
    if not probe_rows:
        return
    V = np.array([r["V_m_s"] for r in probe_rows])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    colors = {"removed (none−γ̇)": "#d62728", "Capytaine A_jj": "#9467bd", "2D strip": "#8c564b"}
    for j, name in enumerate(MODE_NAMES):
        ax = axes[j]
        key = f"A_uvlm_removed_{name}"
        if key in probe_rows[0]:
            y = np.array([r[key] for r in probe_rows])
            ax.plot(V / KN, y, "o-", ms=3, color=colors["removed (none−γ̇)"],
                    label="UVLM removed (none−γ̇ probe)")
        if name in capy_diag:
            ax.axhline(float(capy_diag[name]), color=colors["Capytaine A_jj"], ls="--", lw=1.2,
                       label=f"Capytaine A_jj @ ω_wet")
        ax.set_ylabel(r"$A_{jj}$ [–]")
        ax.set_title(name)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_xlabel("V [kn]")
    axes[1].set_xlabel("V [kn]")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_substitution_balance(
    diag_removed: np.ndarray,
    A_capy: np.ndarray,
    M_full_diag: np.ndarray,
    out_path: Path,
    *,
    subtitle: str = "",
) -> None:
    """Bar chart: UVLM removed (freq-ratio probe) vs Capytaine reinjection."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(2)
    w = 0.2
    series = [
        ("UVLM removed (none−γ̇)", diag_removed[:2], "#d62728"),
        ("Capytaine A_jj", np.diag(A_capy)[:2], "#9467bd"),
        ("UVLM residual (full, D diag)", M_full_diag[:2], "#aec7e8"),
        ("net capy − UVLM_resid", np.diag(A_capy)[:2] - M_full_diag[:2], "#2ca02c"),
    ]
    for i, (label, vals, col) in enumerate(series):
        ax.bar(x + (i - 1.5) * w, vals, width=w, label=label, color=col)
    ax.set_xticks(x)
    ax.set_xticklabels(MODE_NAMES)
    ax.set_ylabel(r"Modal diagonal $A_{jj}$ [–]")
    title = "Added-mass substitution: removed UVLM vs Capytaine reinjection"
    if subtitle:
        title += f"\n{subtitle}"
    ax.set_title(title)
    ax.axhline(0, color="k", lw=0.6)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_summary_md(
    outdir: Path,
    *,
    grid_name: str,
    norms: dict,
    diag_probe: np.ndarray,
    A_capy_hi: np.ndarray,
    M: dict[str, np.ndarray],
    removed_gd: np.ndarray,
) -> None:
    lines = [
        f"# UVLM ↔ Capytaine added-mass substitution ({grid_name})",
        "",
        "## Patch effect on UVLM state-space blocks",
        "",
        "| mode | |Css γ̇ block| | |Dss ζ̇ block| |",
        "|------|----------------|---------------|",
    ]
    for rem in ("none", "gamma_dot", "full"):
        n = norms[rem]
        lines.append(
            f"| `{rem}` | {n['css_gamma_dot']:.4e} | {n['dss_zeta_dot']:.4e} |"
        )
    lines += [
        "",
        "- **`gamma_dot`**: zeros the Css coupling to `dtgamma_dot` (circulatory apparent-mass path).",
        "- **`full`**: additionally zeros the Dss body-velocity (`ζ̇`) feedthrough block.",
        "",
        "## Modal diagonal comparison (bending / torsion)",
        "",
        "| quantity | bend | torsion |",
        "|----------|------|---------|",
        f"| UVLM removed (freq-ratio none vs γ̇) | {diag_probe[0]:.3f} | {diag_probe[1]:.3f} |",
        f"| Capytaine A_jj (ω grid max) | {A_capy_hi[0,0]:.3f} | {A_capy_hi[1,1]:.3f} |",
        f"| UVLM D-feedthrough (`none`) | {M['none'][0,0]:.3f} | {M['none'][1,1]:.3f} |",
        f"| UVLM D-feedthrough (`full`) | {M['full'][0,0]:.3f} | {M['full'][1,1]:.3f} |",
        f"| D-diff removed by γ̇ | {removed_gd[0,0]:.3f} | {removed_gd[1,1]:.3f} |",
        "",
        "**Note:** the frequency-ratio probe `(ω_gd/ω_none)²−1` is the robust measure of",
        "effective UVLM apparent mass on each branch. The raw D-matrix feedthrough mixes",
        "non-circulatory and coupling paths and can differ in magnitude/sign after patching.",
        "",
        "## full vs gamma_dot",
        "",
    ]
    extra = M["gamma_dot"] - M["full"]
    if np.max(np.abs(extra)) < 1e-10:
        lines.append(
            "On this grid, **`full` ≡ `gamma_dot`**: the Dss ζ̇ block was already negligible "
            "before patching, so the extra `full` removal has no effect."
        )
    else:
        lines.append(
            f"Extra removal by `full` (diag): "
            f"bend={extra[0,0]:.3f}, torsion={extra[1,1]:.3f}."
        )
    (outdir / "ADDED_MASS_SUBSTITUTION.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coarse", action="store_true")
    ap.add_argument("--fine", action="store_true")
    ap.add_argument("--grid", type=str, default=None, choices=sorted(GRIDS.keys()))
    ap.add_argument("--grid-span", type=int)
    ap.add_argument("--grid-chord", type=int)
    ap.add_argument("--mstar-fact", type=int)
    ap.add_argument("--num-modes", type=int, default=4)
    ap.add_argument("--v-probe", type=float, default=4.5, help="velocity for D-matrix snapshot [m/s]")
    ap.add_argument("--v-max-kn", type=float, default=72.0)
    ap.add_argument("--no-rayleigh", action="store_true")
    ap.add_argument("--zeta-bend", type=float, default=0.076)
    ap.add_argument("--zeta-tors", type=float, default=0.039)
    ap.add_argument("--capy-npz", type=Path, default=None)
    ap.add_argument("--outdir", type=Path, default=None)
    args = ap.parse_args()

    nm = args.num_modes
    if args.grid_span is not None or args.grid_chord is not None:
        if args.grid_span is None or args.grid_chord is None:
            raise SystemExit("--grid-span and --grid-chord must be set together")
        Ms = args.mstar_fact if args.mstar_fact is not None else 6
        grid = (args.grid_chord, args.grid_span, Ms)
        grid_name = f"n{args.grid_span}c{args.grid_chord}"
    elif args.grid is not None:
        grid_name = args.grid
        grid = GRIDS[grid_name]
    else:
        grid_name = "coarse" if args.coarse else ("fine" if args.fine else "default")
        grid = GRIDS[grid_name]

    outdir = args.outdir or (
        _FSI_ROOT / "output_data" / "ABRAMSON1965" / "07_sharpy_pk_bem"
        / f"_added_mass_analysis_{grid_name}"
    )
    outdir.mkdir(parents=True, exist_ok=True)

    capy = check_capy_basis(load_capytaine_npz(args.capy_npz or default_capytaine_npz()))
    zetas = (args.zeta_bend, args.zeta_tors)

    print(f">>> grid {grid}  assembling none / gamma_dot / full ...", flush=True)
    configs = _assemble_configs(
        nm, grid, rayleigh=not args.no_rayleigh, zetas=zetas, v_max_kn=args.v_max_kn,
    )

    data0, ls0 = configs["none"]
    T, _ = align_capy_to_sharpy(capy, data0, nm)

    M = {}
    for rem, (_, ls) in configs.items():
        ls.update(float(args.v_probe))
        M[rem] = _M_modal_from_D(ls, nm)
    norms = {rem: _uvlm_block_norms(ls) for rem, (_, ls) in configs.items()}

    removed_gd = M["none"] - M["gamma_dot"]
    removed_full = M["gamma_dot"] - M["full"]

    # 2D strip reference (analytic flat plate)
    y_s, W_s, TH_s, _ = sharpy_modal_displacements(data0, nm)
    A_strip = strip_added_mass_modal(y_s, W_s, TH_s, a=-0.5)[:nm, :nm]

    # Capytaine at dry omegas (mean of modes 0,1) and at high-omega asymptote
    om_dry = _dry_omegas_phys(ls0, nm)
    omega_dry_mean = float(np.mean(om_dry[:2]))
    omega_hi = float(np.asarray(capy["omega"]).ravel()[-1])
    A_capy_dry, _ = capy_at_omega(capy, omega_dry_mean, mode_slice=slice(0, nm))
    A_capy_hi, _ = capy_at_omega(capy, omega_hi, mode_slice=slice(0, nm))
    Tn = T[:nm, :nm]
    A_capy_dry = Tn.T @ A_capy_dry @ Tn
    A_capy_hi = Tn.T @ A_capy_hi @ Tn

    # Measured diagonal via frequency-ratio probe (at v_probe and a small sweep)
    v_probe_list = np.array([args.v_probe, 8.0, 12.0, 16.0, 20.0])
    ls_n, ls_g = configs["none"][1], configs["gamma_dot"][1]
    probe_rows = _probe_diag_vs_V(ls_n, ls_g, nm, v_probe_list)
    try:
        diag_probe = _probe_uvlm_added_mass_diag(
            ls_n, ls_g,
            beam_state_indices(ls_n.ss, nm),
            beam_state_indices(ls_g.ss, nm),
            nm, args.v_probe,
        )
    except RuntimeError:
        diag_probe = np.full(nm, np.nan)
        if probe_rows:
            for j, name in enumerate(MODE_NAMES):
                key = f"A_uvlm_removed_{name}"
                diag_probe[j] = probe_rows[0].get(key, np.nan)

    # Rescale strip to measured diagonal at v_probe (same as delta mode)
    D_scale = np.diag(np.sqrt(np.maximum(np.nan_to_num(diag_probe, nan=1e-3), 1e-6)
                              / np.maximum(np.diag(A_strip), 1e-6)))
    A_uvlm_measured = D_scale @ A_strip @ D_scale

    summary = {
        "grid": grid,
        "grid_name": grid_name,
        "v_probe_m_s": args.v_probe,
        "block_norms": norms,
        "M_modal_D_coupled": {k: np.asarray(v).tolist() for k, v in M.items()},
        "removed_by_gamma_dot": np.asarray(removed_gd).tolist(),
        "removed_by_full_extra": np.asarray(removed_full).tolist(),
        "A_strip_2d": np.asarray(A_strip).tolist(),
        "A_uvlm_measured_diag_probe": np.asarray(A_uvlm_measured).tolist(),
        "diag_probe_at_v": np.asarray(diag_probe).tolist(),
        "A_capy_dry_mean_omega": np.asarray(A_capy_dry).tolist(),
        "A_capy_high_omega": np.asarray(A_capy_hi).tolist(),
        "dry_omega_rad_s": om_dry.tolist(),
        "fraction_removed_gamma_dot_diag": (np.diag(removed_gd) / np.maximum(np.diag(M["none"]), 1e-12)).tolist(),
        "fraction_removed_full_diag": (np.diag(M["none"] - M["full"]) / np.maximum(np.diag(M["none"]), 1e-12)).tolist(),
    }

    with (outdir / "added_mass_substitution_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    with (outdir / "uvlm_removed_diag_vs_V.csv").open("w", newline="") as f:
        if probe_rows:
            wri = csv.DictWriter(f, fieldnames=list(probe_rows[0].keys()))
            wri.writeheader()
            wri.writerows(probe_rows)

    # --- plots
    plot_matrix_panels(
        {
            "UVLM none": M["none"][:2, :2],
            "removed γ̇": removed_gd[:2, :2],
            "removed full": removed_full[:2, :2],
            "UVLM full": M["full"][:2, :2],
            "Capytaine (ω_hi)": A_capy_hi[:2, :2],
            "measured strip": A_uvlm_measured[:2, :2],
        },
        {
            "UVLM none": "UVLM apparent mass\n(none)",
            "removed γ̇": "Removed by\nγ̇ patch",
            "removed full": "Extra removed\nby full",
            "UVLM full": "UVLM left\n(full)",
            "Capytaine (ω_hi)": "Capytaine A\n(ω grid max)",
            "measured strip": "Measured UVLM\n(freq-ratio probe)",
        },
        outdir / "uvlm_added_mass_modal_projection.png",
        title=f"Modal added-mass projection — grid {grid_name} @ V={args.v_probe:.1f} m/s",
    )

    plot_substitution_balance(
        diag_probe, A_capy_hi, np.diag(M["full"]),
        outdir / "added_mass_substitution_balance.png",
        subtitle=f"grid {grid_name}  V_probe={args.v_probe:.1f} m/s",
    )

    write_summary_md(
        outdir, grid_name=grid_name, norms=norms,
        diag_probe=diag_probe, A_capy_hi=A_capy_hi, M=M, removed_gd=removed_gd,
    )

    capy_diag = {
        MODE_NAMES[j]: float(np.diag(A_capy_hi)[j]) for j in range(2)
    }
    plot_diagonal_vs_V(
        probe_rows, capy_diag,
        outdir / "uvlm_removed_diag_vs_V.png",
        title=f"UVLM apparent-mass diagonal removed (none−γ̇ probe) vs Capytaine — {grid_name}",
    )

    # Text report
    print("\n=== UVLM added-mass removal summary ===")
    print(f"grid: {grid}  V_probe = {args.v_probe:.2f} m/s ({args.v_probe/KN:.1f} kn)")
    for rem in ("none", "gamma_dot", "full"):
        n = norms[rem]
        print(f"\n[{rem}]")
        print(f"  |Css gamma_dot|_F = {n['css_gamma_dot']:.4e}  (0 => γ̇ path removed)")
        print(f"  |Dss zeta_dot|_F  = {n['dss_zeta_dot']:.4e}  (0 => full body-vel path removed)")
        print(f"  M_modal diag (D feedthrough) = {np.array2string(np.diag(M[rem][:2, :2]), precision=4)}")
    print(f"\nRemoved by gamma_dot (D-matrix difference) diag = "
          f"{np.array2string(np.diag(removed_gd[:2, :2]), precision=4)}")
    print(f"Extra removed by full diag = {np.array2string(np.diag(removed_full[:2, :2]), precision=4)}")
    print(f"Freq-ratio probe A_jj (none vs γ̇) @ V_probe = {np.array2string(diag_probe[:2], precision=4)}")
    print(f"Capytaine A_jj @ ω_hi = {np.array2string(np.diag(A_capy_hi[:2, :2]), precision=4)}")
    print(f"Net substitution (Capy − UVLM_full) diag = "
          f"{np.array2string(np.diag(A_capy_hi[:2, :2]) - np.diag(M['full'][:2, :2]), precision=4)}")
    print(f"\nWrote {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
