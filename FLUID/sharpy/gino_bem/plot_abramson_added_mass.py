#!/usr/bin/env python3
"""
Mode-specific added mass for Abramson 1965 — strip theory, reverse-engineered targets,
and (optional) Capytaine NPZ.

Reverse engineering (unit modal mass, diagonal):
    M_add,i / M_i = (f_dry,i / f_wet,i)^2 - 1
from dry vs fluid-at-rest natural frequencies.

  python gino_bem/plot_abramson_added_mass.py
  python gino_bem/plot_abramson_added_mass.py --capy-npz /path/to/modal_radiation_*.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_SHARPY_ROOT = Path(__file__).resolve().parents[1]
_FSI_ROOT = _SHARPY_ROOT.parents[1]
if str(_SHARPY_ROOT) not in sys.path:
    sys.path.insert(0, str(_SHARPY_ROOT))
if str(_FSI_ROOT) not in sys.path:
    sys.path.insert(0, str(_FSI_ROOT))

# Sharpy / literature dry reference (Modal solver, nm=2)
F_DRY_SHARPY = (9.93, 20.14)
# GINO FEA dry (2-mode reduction from eigendata mode 1 & 2)
F_DRY_GINO = (10.59074257, 20.65494330)
# Target wet (GINO Capytaine / Abramson tuning)
F_WET_CAPY = (8.0, 17.0)
# GINO strip-theory fluid-at-rest (T0_baseline log)
F_WET_STRIP = (8.30078772, 16.93899206)
# Sharpy UVLM coupled @ first stable V (3.5 m/s) — NOT true wet, shown for contrast
F_SHARPY_V35 = (7.270294426430277, 14.838195129835508)

MODE_LABELS = ("1st bending", "1st torsion")


def _ratio_mass(f_dry: float, f_wet: float) -> float:
    r = float(f_dry) / max(float(f_wet), 1e-12)
    return r * r - 1.0


def _reverse_engineer_mass_ratios(
    f_dry: tuple[float, float],
    f_wet: tuple[float, float],
) -> np.ndarray:
    return np.array([_ratio_mass(f_dry[i], f_wet[i]) for i in range(2)], dtype=float)


def _compute_strip_m_hat(cfg) -> np.ndarray | None:
    """GINO strip-theory M_hat_added = Phi^T M_strip Phi for modes 0–1."""
    sys.path[:0] = [
        str(_FSI_ROOT / "examples"),
        str(_FSI_ROOT / "examples" / "ABRAMSON1965"),
        str(_FSI_ROOT),
        str(_FSI_ROOT / "STRUCTURE"),
        str(_FSI_ROOT / "FLUID" / "capytaine"),
        str(_FSI_ROOT / "FLUID" / "PanelAero"),
    ]
    try:
        from FEA import structural_analysis, structural_model
        from FEA.structural_analysis import added_mass_projection
    except ImportError as e:
        print(f"[strip] skip: {e}")
        return None

    cfg.added_mass_strip_theory = True
    cfg.added_mass_strip_scale = float(getattr(cfg, "added_mass_strip_scale", 1.0))
    cfg.num_modes_egv = max(int(getattr(cfg, "num_modes_egv", 2)), 2)

    beam = structural_model.build(cfg)
    res = structural_analysis.run_dry_analysis(beam, cfg)
    if res is None:
        return None

    dihedral_deg = float(getattr(cfg, "dihedral_angle", 0.0))
    apply_dihedral = abs(dihedral_deg) > 1e-12 and bool(getattr(cfg, "dihedral_rotate_beam", True))
    if apply_dihedral:
        from FEA import post_dihedral_utils

        res, beam, _ = post_dihedral_utils.apply_structural_dihedral_about_x(
            beam, res, None, dihedral_deg, rotate_beam=True, rotate_aerogrid=False
        )

    mode_indices = [0, 1]
    M_hat, _ = added_mass_projection(
        res.M_global,
        res.dry_eigenvectors,
        mode_indices,
        cfg,
        res.total_dof,
        constrained_dofs=res.constrained_dofs,
        apply_structural_dihedral=apply_dihedral,
        method_label="strip theory",
    )
    return np.asarray(M_hat, dtype=float)


def _load_capy_m_hat(npz_path: Path, omega_rad_s: float) -> np.ndarray | None:
    try:
        from gino_bem.capy_bem_bridge import capy_at_omega, load_capytaine_npz
    except ImportError:
        from sharpy.gino_bem.capy_bem_bridge import capy_at_omega, load_capytaine_npz
    capy = load_capytaine_npz(npz_path)
    A, _ = capy_at_omega(capy, omega_rad_s, mode_slice=slice(0, 2))
    return np.asarray(A, dtype=float)


def _default_capy_npz() -> Path | None:
    capy_dir = _FSI_ROOT / "FLUID" / "capytaine" / "ABRAMSON1965" / "results_modal_radiation"
    for name in ("modal_radiation_AB_depth_0.0508.npz", "modal_radiation_AB.npz"):
        p = capy_dir / name
        if p.is_file():
            return p
    return None


def plot_added_mass(
    out_png: Path,
    *,
    strip_m: np.ndarray | None,
    capy_m: np.ndarray | None,
    reverse: dict[str, np.ndarray],
    capy_omega_hz: float | None,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # --- Left: diagonal generalized added mass / M_struct ---
    ax = axes[0]
    x = np.arange(2)
    w = 0.13
    series: list[tuple[str, np.ndarray, str]] = []
    for label, arr in reverse.items():
        series.append((label, arr, "o"))
    if strip_m is not None:
        series.append(("Strip ΦᵀMΦ diag", np.diag(strip_m), "s"))
    if capy_m is not None:
        series.append((f"Capytaine A(ω) diag @ {capy_omega_hz:.1f} Hz", np.diag(capy_m), "^"))

    for i, (label, vals, mk) in enumerate(series):
        offset = (i - 0.5 * (len(series) - 1)) * w
        ax.bar(x + offset, vals, width=w, label=label, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(MODE_LABELS)
    ax.set_ylabel("Generalized added mass / M_struct [–]")
    ax.set_title("Diagonal mode-specific added mass")
    ax.axhline(0, color="k", lw=0.6)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=7, loc="upper right")

    # --- Right: full 2×2 matrices (strip / Capytaine) ---
    ax2 = axes[1]
    panels = []
    if strip_m is not None:
        panels.append(("Strip M̂_add", strip_m))
    if capy_m is not None:
        panels.append((f"Capytaine A @ {capy_omega_hz:.1f} Hz", capy_m))
    if not panels:
        ax2.text(0.5, 0.5, "No strip/Capytaine matrix available\n(reverse-engineered only)",
                 ha="center", va="center", transform=ax2.transAxes)
        ax2.set_axis_off()
    elif len(panels) == 1:
        im = ax2.imshow(panels[0][1], cmap="RdBu_r", vmin=-0.5, vmax=1.0)
        ax2.set_title(panels[0][0])
        for (j, i), v in np.ndenumerate(panels[0][1]):
            ax2.text(i, j, f"{v:.3f}", ha="center", va="center", fontsize=10)
        ax2.set_xticks([0, 1]); ax2.set_yticks([0, 1])
        ax2.set_xticklabels(MODE_LABELS); ax2.set_yticklabels(MODE_LABELS)
        fig.colorbar(im, ax=ax2, fraction=0.046)
    else:
        from matplotlib.gridspec import GridSpecFromSubplotSpec
        inner = GridSpecFromSubplotSpec(1, 2, subplot_spec=ax2.get_subplotspec())
        ax2.set_visible(False)
        for col, (title, mat) in enumerate(panels):
            sub = fig.add_subplot(inner[col])
            im = sub.imshow(mat, cmap="RdBu_r", vmin=-0.5, vmax=1.0)
            sub.set_title(title, fontsize=9)
            for (j, i), v in np.ndenumerate(mat):
                sub.text(i, j, f"{v:.3f}", ha="center", va="center", fontsize=9)
            sub.set_xticks([0, 1]); sub.set_yticks([0, 1])
            sub.set_xticklabels(["bend", "tors"], fontsize=8)
            sub.set_yticklabels(["bend", "tors"], fontsize=8)
        fig.colorbar(im, ax=[fig.axes[-2], fig.axes[-1]], fraction=0.046, shrink=0.9)

    fig.suptitle(
        "Abramson 1965 — mode-specific added mass\n"
        "Reverse: M_add/M = (f_dry/f_wet)² − 1  |  Sharpy UVLM has no modal M_add",
        fontsize=11,
    )
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print("saved:", out_png)


def _load_abramson_config():
    sys.path.insert(0, str(_FSI_ROOT / "examples"))
    from config import get_config

    return get_config("ABRAMSON1965")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--capy-npz", type=Path, default=None)
    ap.add_argument("--capy-omega-hz", type=float, default=16.0,
                    help="frequency for Capytaine A(ω) interpolation [Hz]")
    ap.add_argument("--skip-strip", action="store_true",
                    help="skip GINO strip projection (no STRUCTURE env)")
    args = ap.parse_args()

    out_png = args.out or (
        _FSI_ROOT / "output_data" / "ABRAMSON1965" / "sharpy_uvlm_water"
        / "_baseline_nm4" / "abramson_mode_specific_added_mass.png"
    )

    reverse = {
        "Rev. GINO dry → wet (8, 17 Hz)": _reverse_engineer_mass_ratios(F_DRY_GINO, F_WET_CAPY),
        "Rev. GINO dry → strip wet": _reverse_engineer_mass_ratios(F_DRY_GINO, F_WET_STRIP),
        "Rev. Sharpy dry → wet target": _reverse_engineer_mass_ratios(F_DRY_SHARPY, F_WET_CAPY),
        "Rev. Sharpy dry → coupled @ 3.5 m/s": _reverse_engineer_mass_ratios(F_DRY_SHARPY, F_SHARPY_V35),
    }

    strip_m = None
    if not args.skip_strip:
        try:
            cfg = _load_abramson_config()
            strip_m = _compute_strip_m_hat(cfg)
            if strip_m is not None:
                print("Strip M_hat_added:\n", strip_m)
        except Exception as e:
            print(f"[strip] failed: {e}")

    capy_m = None
    capy_path = args.capy_npz or _default_capy_npz()
    if capy_path and capy_path.is_file():
        try:
            omega = 2.0 * np.pi * args.capy_omega_hz
            capy_m = _load_capy_m_hat(capy_path, omega)
            print(f"Capytaine A from {capy_path} @ {args.capy_omega_hz} Hz:\n", capy_m)
        except Exception as e:
            print(f"[capy] failed: {e}")
    elif capy_path:
        print(f"[capy] NPZ not found: {capy_path}")

    print("\nReverse-engineered M_add/M (diagonal):")
    for label, arr in reverse.items():
        print(f"  {label}: bend={arr[0]:.3f}, tors={arr[1]:.3f}")

    plot_added_mass(
        out_png,
        strip_m=strip_m,
        capy_m=capy_m,
        reverse=reverse,
        capy_omega_hz=args.capy_omega_hz,
    )


if __name__ == "__main__":
    main()
