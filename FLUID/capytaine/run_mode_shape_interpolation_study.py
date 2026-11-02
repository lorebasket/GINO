"""
Mode-shape interpolation study for Capytaine added mass A(ω) and damping B(ω).

For each blend parameter α ∈ [0, 1]:

    face_disp_α = (1 − α) · d_structural + α · d_target

where ``d_structural`` is the usual beam→mesh map and ``d_target`` at α = 1 is a single
Capytaine rigid-body unit field (Surge … Yaw) per structural mode, with matched peak
amplitude (``--target rigid_body``, default).

Runs ``run_modal_radiation.py`` with precomputed ``--modes`` NPZ per α and plots
A_ii, B_ii vs α at high ω.

Example (FSI root):
    python FLUID/capytaine/run_mode_shape_interpolation_study.py --case-name ABRAMSON1965

    python FLUID/capytaine/run_mode_shape_interpolation_study.py --case-name ABRAMSON1965 \\
        --omega-list 1,5,20,60 --target span_rigidify --alphas 0,0.5,1
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from typing import Iterable

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_FSI_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
_EXAMPLES_DIR = os.path.join(_FSI_ROOT, "examples")
if _EXAMPLES_DIR not in sys.path:
    sys.path.insert(0, _EXAMPLES_DIR)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from config import get_config  # noqa: E402
from mode_shape_interpolation import (  # noqa: E402
    build_structural_and_target_face_displacements,
    default_mode_indices,
    face_centers_before_depth_translation,
    interpolate_face_displacements,
    load_mesh,
    resolve_span_axis,
)
from run_modal_radiation import (  # noqa: E402
    _parse_depth_values,
    _plot_modal_projection_on_mesh,
    _translate_mesh_z,
)


def _parse_alpha_list(text: str) -> list[float]:
    vals = [float(x.strip()) for x in str(text).split(",") if x.strip()]
    if not vals:
        raise ValueError("Empty alpha list.")
    for a in vals:
        if not (0.0 <= a <= 1.0):
            raise ValueError(f"Each alpha must be in [0, 1]; got {a}")
    return vals


def _alpha_tag(alpha: float) -> str:
    return f"alpha_{alpha:.4f}".replace(".", "p")


def _run_radiation_for_modes(
    *,
    case_name: str,
    mesh_path: str,
    modes_npz: str,
    omega_list: str,
    phi_normalization: str,
    depth: float,
    results_dir: str,
) -> str:
    cmd = [
        sys.executable,
        os.path.join(_THIS_DIR, "run_modal_radiation.py"),
        "--case-name",
        case_name,
        "--mesh-path",
        mesh_path,
        "--modes",
        modes_npz,
        "--omega-list",
        omega_list,
        "--out-dir",
        results_dir,
        "--no-plots",
        "--no-debug-geometry-plot",
        "--phi-normalization",
        phi_normalization,
        "--depth",
        str(depth),
    ]
    subprocess.run(cmd, check=True, cwd=_FSI_ROOT)
    depth_tag = f"{depth:g}".replace(".", "p").replace("-", "m")
    src = os.path.join(results_dir, f"modal_radiation_AB_depth_{depth_tag}.npz")
    if not os.path.isfile(src):
        raise FileNotFoundError(f"Expected Capytaine output: {src}")
    return src


def _load_npz_summary(path: str) -> dict:
    from modal_radiation_io import load_modal_radiation_npz

    data = load_modal_radiation_npz(path)
    omega = data["omega"]
    A = data["added_mass"]
    B = data["added_damping"]
    n_tail = min(3, omega.size)
    return {
        "omega": omega,
        "mode_names": data["mode_names"],
        "added_mass": A,
        "added_damping": B,
        "A_asymptotic": np.mean(A[-n_tail:], axis=0),
        "B_asymptotic": np.mean(B[-n_tail:], axis=0),
        "omega_asymptotic": float(np.mean(omega[-n_tail:])),
    }


def _save_modes_npz(
    path: str,
    mode_names: list[str],
    face_disp: np.ndarray,
    *,
    alpha: float,
    meta: dict,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(
        path,
        mode_names=np.asarray(mode_names, dtype=object),
        face_displacements=np.asarray(face_disp, dtype=float),
        alpha=float(alpha),
        interpolation_meta=json.dumps(meta),
    )


def _plot_alpha_convergence(rows: list[dict], out_dir: str, omega_asym: float) -> None:
    import matplotlib.pyplot as plt

    alphas = np.array([r["alpha"] for r in rows], dtype=float)
    n_modes = len(rows[0]["mode_names"])
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_modes, 1)))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    for im in range(n_modes):
        diag_a = [r["A_asymptotic"][im, im] for r in rows]
        ax.plot(alphas, diag_a, "o-", color=colors[im], lw=2, label=f"mode {im + 1}")
    ax.set_xlabel("α (0 = structural, 1 = rigid target)")
    ax.set_ylabel("A_ii [kg]")
    ax.set_title(f"Added-mass diagonal (ω ≈ {omega_asym:.1f} rad/s)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    for im in range(n_modes):
        diag_b = [r["B_asymptotic"][im, im] for r in rows]
        ax.plot(alphas, diag_b, "o-", color=colors[im], lw=2, label=f"mode {im + 1}")
    ax.set_xlabel("α")
    ax.set_ylabel("B_ii [kg/s]")
    ax.set_title(f"Radiation damping diagonal (ω ≈ {omega_asym:.1f} rad/s)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle("Mode-shape interpolation: A, B vs blend parameter α", fontsize=11)
    fig.tight_layout()
    path = os.path.join(out_dir, "interpolation_A_B_vs_alpha.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Full A(ω) for a few α values
    fig2, axes2 = plt.subplots(2, 2, figsize=(10, 8), squeeze=False)
    pick = [0, len(rows) // 2, len(rows) - 1] if len(rows) >= 3 else list(range(len(rows)))
    for im in range(min(n_modes, 4)):
        ax2 = axes2[im // 2][im % 2]
        for idx in pick:
            r = rows[idx]
            omega = r["omega"]
            y = r["added_mass"][:, im, im]
            ax2.plot(omega, y, lw=1.6, label=f"α={r['alpha']:.3g}")
        ax2.set_title(f"A_{{ii}} mode {im + 1}")
        ax2.set_xlabel("ω [rad/s]")
        ax2.set_ylabel("A_ii [kg]")
        ax2.grid(True, alpha=0.3)
        if im == 0:
            ax2.legend(fontsize=8)
    for k in range(n_modes, 4):
        axes2[k // 2][k % 2].set_visible(False)
    fig2.suptitle("Added-mass diagonal vs ω (selected α)", fontsize=11)
    fig2.tight_layout()
    path2 = os.path.join(out_dir, "interpolation_A_omega_selected_alpha.png")
    fig2.savefig(path2, dpi=180, bbox_inches="tight")
    plt.close(fig2)


def _plot_mode_shapes_at_alphas(
    *,
    mesh,
    rows_modes: list[tuple[float, list[str], np.ndarray]],
    out_dir: str,
) -> None:
    """One debug projection PNG per mode at α = 0, mid, 1."""
    if not rows_modes:
        return
    pick_idx = [0, len(rows_modes) // 2, len(rows_modes) - 1]
    pick_idx = sorted(set(i for i in pick_idx if 0 <= i < len(rows_modes)))
    for idx in pick_idx:
        alpha, mode_names, face_disp = rows_modes[idx]
        tag = _alpha_tag(alpha)
        sub = os.path.join(out_dir, "mode_shapes", tag)
        os.makedirs(sub, exist_ok=True)
        _plot_modal_projection_on_mesh(
            mesh=mesh,
            face_disp=face_disp,
            mode_names=mode_names,
            out_dir=sub,
            depth_suffix="",
            quiver_scale=1.0,
            max_arrows=600,
            show=False,
        )


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Capytaine mode-shape interpolation study (structural → rigid target)."
    )
    parser.add_argument("--case-name", "--case_name", dest="case_name", type=str, required=True)
    parser.add_argument(
        "--alphas",
        type=str,
        default="0,0.167,0.333,0.5,0.667,0.833,1",
        help="Comma-separated blend parameters in [0, 1] (default: 7 cases).",
    )
    parser.add_argument(
        "--target",
        type=str,
        choices=("rigid_body", "rigid6_projection", "span_rigidify"),
        default="rigid_body",
        help="α=1 target per mode: single rigid-body DOF (default), LS 6-DOF mix, or span rigidify.",
    )
    parser.add_argument(
        "--no-match-peak-amplitude",
        action="store_true",
        help="At α=1 use unscaled Capytaine unit rigid fields (1 m / 1 rad) instead of matching structural peak |d|.",
    )
    parser.add_argument(
        "--ref-frac",
        type=float,
        default=0.5,
        help="Span station for span_rigidify target (0=root, 1=tip).",
    )
    parser.add_argument(
        "--omega-list",
        type=str,
        default="",
        help="Override omega [rad/s]; default: sparse subset of cfg.omega_list.",
    )
    parser.add_argument(
        "--mode-indices",
        type=str,
        default="",
        help="0-based CSV mode rows (default: num_modes_egv from config).",
    )
    parser.add_argument(
        "--phi-normalization",
        type=str,
        default="",
        help="Override capytaine_phi_normalization (default: case config).",
    )
    parser.add_argument("--depth", type=float, default=None, help="Submergence depth [m].")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="Study directory (default: <case>/mode_shape_interpolation).",
    )
    parser.add_argument(
        "--plot-mode-shapes",
        action="store_true",
        help="Save mode-projection PNGs at α=0, mid, 1.",
    )
    parser.add_argument(
        "--also-run-pure-rigid",
        action="store_true",
        help="After interpolation sweep, run a separate 6-DOF rigid-body radiation case.",
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip summary plots.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build interpolated modes NPZ + metadata only; skip Capytaine radiation solves.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    cfg = get_config(args.case_name)
    alphas = _parse_alpha_list(args.alphas)
    phi_norm = (
        str(args.phi_normalization).strip()
        or str(getattr(cfg, "capytaine_phi_normalization", "none")).strip()
        or "none"
    )
    depth = args.depth
    if depth is None:
        depth = float(np.asarray(cfg.depth).ravel()[int(getattr(cfg, "depth_index", 0))])

    mesh_path = getattr(cfg, "mesh_path", os.path.join(_THIS_DIR, args.case_name, f"{args.case_name}_mesh.vtu"))
    mesh_path = os.path.abspath(mesh_path)
    modal_dir = cfg.modal_dir[0] if isinstance(cfg.modal_dir, tuple) else cfg.modal_dir
    prefix = cfg.prefix

    if args.mode_indices.strip():
        mode_indices = [int(x.strip()) for x in args.mode_indices.split(",") if x.strip()]
    else:
        mode_indices = default_mode_indices(cfg)

    omega_list = args.omega_list.strip()
    if not omega_list:
        omega_full = np.asarray(cfg.omega_list, dtype=float).ravel()
        if omega_full.size >= 9:
            pick = np.linspace(0, omega_full.size - 1, 9, dtype=int)
            omega_list = ",".join(f"{omega_full[i]:.6g}" for i in pick)
        else:
            omega_list = ",".join(f"{x:.6g}" for x in omega_full)

    out_dir = args.out_dir.strip() or os.path.join(
        _THIS_DIR, args.case_name, "mode_shape_interpolation_rigid"
    )
    modes_dir = os.path.join(out_dir, "modes_npz")
    rad_dir = os.path.join(out_dir, "radiation")
    os.makedirs(modes_dir, exist_ok=True)
    os.makedirs(rad_dir, exist_ok=True)

    print(f"Case: {args.case_name}")
    print(f"Target at α=1: {args.target}")
    print(f"Alphas: {alphas}")
    print(f"Mode indices: {mode_indices}")
    print(f"Omega list: {omega_list}")
    print(f"phi_normalization: {phi_norm!r}")
    print(f"Depth: {depth} m")
    print(f"Output: {out_dir}")

    import capytaine as cpt

    mesh = load_mesh(cpt, mesh_path)
    mesh_for_modes = load_mesh(cpt, mesh_path)
    _translate_mesh_z(mesh_for_modes, -depth)
    face_centers = face_centers_before_depth_translation(mesh_for_modes, depth)
    span_axis = resolve_span_axis(cfg, None)
    rot_c = getattr(cfg, "rigid_body_rotation_center", None)
    rot_c = None if rot_c is None else np.asarray(rot_c, dtype=float)

    mode_names, d_struct, d_target, interp_meta = build_structural_and_target_face_displacements(
        cfg=cfg,
        face_centers=face_centers,
        modal_dir=modal_dir,
        prefix=prefix,
        mode_indices=mode_indices,
        span_axis=span_axis,
        phi_normalization=phi_norm,
        target=args.target,
        ref_frac=float(args.ref_frac),
        match_peak_amplitude=not bool(args.no_match_peak_amplitude),
        cpt=cpt,
        mesh=mesh_for_modes,
        rotation_center=rot_c,
    )
    meta_path = os.path.join(out_dir, "interpolation_targets_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(interp_meta, f, indent=2)
    print(f"Wrote target metadata: {meta_path}")
    for entry in interp_meta.get("per_mode", []):
        line = f"  {entry['mode_name']}:"
        if entry.get("rigid_dof_at_alpha_1"):
            line += (
                f" α=1 → rigid {entry['rigid_dof_at_alpha_1']}"
                f" (MAC²={entry.get('mac_squared_vs_structural', 0):.3f})"
            )
        if entry.get("shape_residual_rel") is not None:
            line += f"  shape residual vs struct {100 * entry['shape_residual_rel']:.1f} %"
        elif entry.get("projection_residual_rel") is not None:
            line += f"  projection residual {100 * entry['projection_residual_rel']:.1f} %"
        if entry.get("rigid6_coeffs"):
            line += f"  rigid6 coeffs={np.array2string(np.asarray(entry['rigid6_coeffs']), precision=3)}"
        print(line)

    rows: list[dict] = []
    rows_modes: list[tuple[float, list[str], np.ndarray]] = []

    for alpha in alphas:
        tag = _alpha_tag(alpha)
        print(f"\n=== α = {alpha:g} ({tag}) ===")
        d_alpha = interpolate_face_displacements(d_struct, d_target, alpha)
        modes_npz = os.path.join(modes_dir, f"modes_{tag}.npz")
        _save_modes_npz(
            modes_npz,
            mode_names,
            d_alpha,
            alpha=alpha,
            meta={**interp_meta, "alpha": alpha},
        )

        rows_modes.append((alpha, mode_names, d_alpha))
        if args.dry_run:
            print(f"  dry-run: wrote {modes_npz}")
            continue

        t0 = time.perf_counter()
        npz_path = _run_radiation_for_modes(
            case_name=args.case_name,
            mesh_path=mesh_path,
            modes_npz=modes_npz,
            omega_list=omega_list,
            phi_normalization=phi_norm,
            depth=depth,
            results_dir=os.path.join(rad_dir, tag),
        )
        solve_s = time.perf_counter() - t0
        print(f"  radiation solve: {solve_s:.1f} s -> {npz_path}")

        summary = _load_npz_summary(npz_path)
        row = {
            "alpha": alpha,
            "tag": tag,
            "solve_time_s": solve_s,
            "modes_npz": modes_npz,
            "npz_path": npz_path,
            "mode_names": summary["mode_names"],
            "omega": summary["omega"],
            "added_mass": summary["added_mass"],
            "added_damping": summary["added_damping"],
            "A_asymptotic": summary["A_asymptotic"],
            "B_asymptotic": summary["B_asymptotic"],
            "omega_asymptotic": summary["omega_asymptotic"],
        }
        rows.append(row)
        diag_a = np.diag(summary["A_asymptotic"])
        print(f"  A_ii @ high ω: {np.array2string(diag_a, precision=4)}")

    if args.dry_run:
        if args.plot_mode_shapes:
            _plot_mode_shapes_at_alphas(mesh=mesh_for_modes, rows_modes=rows_modes, out_dir=out_dir)
            print(f"Wrote mode-shape PNGs under {out_dir}/mode_shapes/")
        print("\nDry-run complete (modes NPZ only). Re-run without --dry-run for Capytaine solves.")
        return

    # CSV summary
    csv_path = os.path.join(out_dir, "interpolation_summary.csv")
    n_modes = len(mode_names)
    fieldnames = ["alpha", "tag", "solve_time_s", "omega_asymptotic_rad_s"]
    for i in range(n_modes):
        fieldnames.append(f"A_{i + 1}{i + 1}_kg")
    for i in range(n_modes):
        fieldnames.append(f"B_{i + 1}{i + 1}_kg_s")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()
        for r in rows:
            out_row = {
                "alpha": f"{r['alpha']:.6g}",
                "tag": r["tag"],
                "solve_time_s": f"{r['solve_time_s']:.2f}",
                "omega_asymptotic_rad_s": f"{r['omega_asymptotic']:.6g}",
            }
            A = r["A_asymptotic"]
            B = r["B_asymptotic"]
            for i in range(n_modes):
                out_row[f"A_{i + 1}{i + 1}_kg"] = f"{A[i, i]:.8e}"
            for i in range(n_modes):
                out_row[f"B_{i + 1}{i + 1}_kg_s"] = f"{B[i, i]:.8e}"
            wr.writerow(out_row)
    print(f"\nWrote {csv_path}")

    if not args.no_plot and rows:
        _plot_alpha_convergence(rows, out_dir, rows[-1]["omega_asymptotic"])
        print(f"Wrote plots in {out_dir}")

    if args.plot_mode_shapes:
        _plot_mode_shapes_at_alphas(mesh=mesh_for_modes, rows_modes=rows_modes, out_dir=out_dir)
        print(f"Wrote mode-shape PNGs under {out_dir}/mode_shapes/")

    if args.also_run_pure_rigid:
        print("\n=== Pure 6-DOF rigid-body radiation (reference) ===")
        rigid_dir = os.path.join(out_dir, "radiation_pure_rigid6")
        os.makedirs(rigid_dir, exist_ok=True)
        cmd = [
            sys.executable,
            os.path.join(_THIS_DIR, "run_modal_radiation.py"),
            "--case-name",
            args.case_name,
            "--mesh-path",
            mesh_path,
            "--omega-list",
            omega_list,
            "--out-dir",
            rigid_dir,
            "--no-plots",
            "--no-debug-geometry-plot",
            "--phi-normalization",
            phi_norm,
            "--depth",
            str(depth),
            "--rigid-body",
        ]
        subprocess.run(cmd, check=True, cwd=_FSI_ROOT)
        print(f"Rigid-body results: {rigid_dir}")


if __name__ == "__main__":
    main()
