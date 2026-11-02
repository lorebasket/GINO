"""
Mesh convergence study for Capytaine modal radiation (A, B vs mesh resolution).

Builds several hull meshes at increasing (n_span, n_chord) resolution, runs a
short omega sweep per mesh, and plots asymptotic added-mass / damping vs panel count.

Runtime scales roughly as n_faces * n_omega * n_modes per mesh.  Use a sparse
``--omega-list`` for screening; refine omega only on the mesh size you pick.

Example (from FSI root, GINO / capytaine env active):
    python FLUID/capytaine/run_mesh_convergence_study.py --case-name ABRAMSON1965

    python FLUID/capytaine/run_mesh_convergence_study.py --case-name ABRAMSON1965 \\
        --resolutions 8,12,16,24,30 --omega-list 1,5,20,60,120,150 --mode-indices 0,1
"""

from __future__ import annotations

import argparse
import csv
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

from config import get_config  # noqa: E402


def _parse_resolution_list(text: str) -> list[tuple[int, int]]:
    """'8,12,16' -> [(8,8), (12,12), ...]; '8x16,12x24' -> [(8,16), (12,24)]."""
    out: list[tuple[int, int]] = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        if "x" in part.lower():
            a, b = part.lower().split("x", 1)
            out.append((int(a.strip()), int(b.strip())))
        else:
            n = int(part)
            out.append((n, n))
    if not out:
        raise ValueError("Empty resolution list.")
    return out


def _mesh_tag(n_span: int, n_chord: int) -> str:
    return f"nspan{n_span}_nchord{n_chord}"


def _build_mesh(case_name: str, n_span: int, n_chord: int, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cmd = [
        sys.executable,
        os.path.join(_THIS_DIR, "build_capytaine_mesh.py"),
        "--case_name",
        case_name,
        "--n-span",
        str(n_span),
        "--n-perimeter",
        str(n_chord),
        "--out",
        out_path,
    ]
    subprocess.run(cmd, check=True, cwd=_FSI_ROOT)


def _run_radiation(
    *,
    case_name: str,
    mesh_path: str,
    omega_list: str,
    mode_indices: str,
    phi_normalization: str,
    depth: float | None,
    results_dir: str | None = None,
) -> str:
    cmd = [
        sys.executable,
        os.path.join(_THIS_DIR, "run_modal_radiation.py"),
        "--case-name",
        case_name,
        "--mesh-path",
        mesh_path,
        "--omega-list",
        omega_list,
        "--no-plots",
        "--no-debug-geometry-plot",
        "--phi-normalization",
        phi_normalization,
    ]
    if results_dir:
        cmd.extend(["--out-dir", results_dir])
    if mode_indices.strip():
        cmd.extend(["--mode-indices", mode_indices])
    if depth is not None:
        cmd.extend(["--depth", str(depth)])
    subprocess.run(cmd, check=True, cwd=_FSI_ROOT)
    depth_tag = f"{depth:g}".replace(".", "p").replace("-", "m") if depth is not None else None
    capy_dir = results_dir or getattr(
        get_config(case_name),
        "capytaine_results_dir",
        os.path.join(_THIS_DIR, case_name, "results_modal_radiation"),
    )
    if depth is not None:
        src = os.path.join(capy_dir, f"modal_radiation_AB_depth_{depth_tag}.npz")
    else:
        src = os.path.join(capy_dir, "modal_radiation_AB.npz")
    if not os.path.isfile(src):
        raise FileNotFoundError(f"Expected Capytaine output: {src}")
    return src


def _load_npz_summary(path: str) -> dict:
    sys.path.insert(0, _THIS_DIR)
    from modal_radiation_io import load_modal_radiation_npz

    data = load_modal_radiation_npz(path)
    omega = data["omega"]
    A = data["added_mass"]
    B = data["added_damping"]
    # High-omega asymptotic slice (mean of last 3 samples if available)
    n_tail = min(3, omega.size)
    A_hi = np.mean(A[-n_tail:], axis=0)
    B_hi = np.mean(B[-n_tail:], axis=0)
    return {
        "omega": omega,
        "mode_names": data["mode_names"],
        "added_mass": A,
        "added_damping": B,
        "A_asymptotic": A_hi,
        "B_asymptotic": B_hi,
        "omega_asymptotic": float(np.mean(omega[-n_tail:])),
    }


def _count_mesh_faces(vtu_path: str) -> int:
    import xml.etree.ElementTree as ET

    root = ET.parse(vtu_path).getroot()
    for el in root.iter():
        if el.tag.endswith("Piece"):
            return int(el.attrib.get("NumberOfCells", "0"))
    return 0


def _plot_convergence(rows: list[dict], out_path: str) -> None:
    import matplotlib.pyplot as plt

    n_faces = np.array([r["n_faces"] for r in rows], dtype=float)
    n_modes = len(rows[0]["mode_names"])
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), squeeze=False)
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_modes, 1)))

    for im in range(min(n_modes, 4)):
        ax_a = axes[im // 2][im % 2]
        diag_a = [r["A_asymptotic"][im, im] for r in rows]
        off_a = [r["A_asymptotic"][0, 1] if n_modes > 1 else np.nan for r in rows]
        ax_a.plot(n_faces, diag_a, "o-", color=colors[im], lw=2, label=f"A_{{ii}} mode {im + 1}")
        if n_modes > 1 and im == 0:
            ax_a.plot(n_faces, off_a, "s--", color="gray", lw=1.5, label="A_12")
        ax_a.set_xlabel("n_faces")
        ax_a.set_ylabel("A [kg]")
        ax_a.set_title(f"Added mass (ω ≈ {rows[-1]['omega_asymptotic']:.1f} rad/s)")
        ax_a.grid(True, alpha=0.3)
        ax_a.legend(fontsize=8)

    # Hide unused subplots
    for k in range(n_modes, 4):
        axes[k // 2][k % 2].set_visible(False)

    tags = ", ".join(f"{r['n_span']}×{r['n_chord']}" for r in rows)
    fig.suptitle(f"Mesh convergence — added mass vs n_faces\n({tags})", fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Damping figure
    fig2, axes2 = plt.subplots(2, 2, figsize=(10, 8), squeeze=False)
    for im in range(min(n_modes, 4)):
        ax = axes2[im // 2][im % 2]
        diag_b = [r["B_asymptotic"][im, im] for r in rows]
        ax.plot(n_faces, diag_b, "o-", color=colors[im], lw=2, label=f"B_{{ii}} mode {im + 1}")
        ax.set_xlabel("n_faces")
        ax.set_ylabel("B [kg/s]")
        ax.set_title(f"Radiation damping (ω ≈ {rows[-1]['omega_asymptotic']:.1f} rad/s)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    for k in range(n_modes, 4):
        axes2[k // 2][k % 2].set_visible(False)
    fig2.suptitle("Mesh convergence — radiation damping vs n_faces", fontsize=11)
    fig2.tight_layout()
    damp_path = out_path.replace("_added_mass", "_damping")
    if damp_path == out_path:
        damp_path = out_path.replace(".png", "_damping.png")
    fig2.savefig(damp_path, dpi=180, bbox_inches="tight")
    plt.close(fig2)


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Capytaine mesh convergence study.")
    parser.add_argument("--case-name", "--case_name", dest="case_name", type=str, required=True)
    parser.add_argument(
        "--resolutions",
        type=str,
        default="8,12,16,24,30",
        help="Comma list: '8,12,16' (square) or '8x16,12x24' (span×chord).",
    )
    parser.add_argument(
        "--omega-list",
        type=str,
        default="0.5,2,5,10,20,40,80,120,150",
        help="Sparse omega [rad/s] for convergence screening (default 9 points).",
    )
    parser.add_argument(
        "--mode-indices",
        type=str,
        default="",
        help="Optional mode CSV rows (default: all num_modes_egv from config).",
    )
    parser.add_argument(
        "--phi-normalization",
        type=str,
        default="",
        help="Override capytaine_phi_normalization (default: case config).",
    )
    parser.add_argument(
        "--depth",
        type=float,
        default=None,
        help="Submergence depth [m]; default from case config.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="Study output directory (default: <case>/mesh_convergence).",
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip convergence PNG plots.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    cfg = get_config(args.case_name)
    resolutions = _parse_resolution_list(args.resolutions)
    phi_norm = (
        str(args.phi_normalization).strip()
        or str(getattr(cfg, "capytaine_phi_normalization", "none")).strip()
        or "none"
    )
    depth = args.depth
    if depth is None:
        depth = float(np.asarray(cfg.depth).ravel()[int(getattr(cfg, "depth_index", 0))])

    out_dir = args.out_dir.strip() or os.path.join(_THIS_DIR, args.case_name, "mesh_convergence")
    mesh_dir = os.path.join(out_dir, "meshes")
    results_dir = os.path.join(out_dir, "radiation")
    os.makedirs(mesh_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    print(f"Case: {args.case_name}")
    print(f"Resolutions: {resolutions}")
    print(f"Omega list: {args.omega_list}")
    print(f"phi_normalization: {phi_norm!r}")
    print(f"Output: {out_dir}")

    rows: list[dict] = []
    for n_span, n_chord in resolutions:
        tag = _mesh_tag(n_span, n_chord)
        mesh_path = os.path.join(mesh_dir, f"{args.case_name}_{tag}.vtu")
        print(f"\n=== {tag} ===")
        if not os.path.isfile(mesh_path):
            print(f"Building mesh -> {mesh_path}")
            t0 = time.perf_counter()
            _build_mesh(args.case_name, n_span, n_chord, mesh_path)
            print(f"  build time: {time.perf_counter() - t0:.1f} s")
        else:
            print(f"Reusing mesh {mesh_path}")

        n_faces = _count_mesh_faces(mesh_path)
        n_omega = len([x for x in args.omega_list.split(",") if x.strip()])
        n_modes = len([x for x in (args.mode_indices or "").split(",") if x.strip()]) or int(
            getattr(cfg, "num_modes_egv", 4)
        )
        print(f"  n_faces={n_faces}, solves/mesh ≈ {n_omega * n_modes}")

        t0 = time.perf_counter()
        npz_src = _run_radiation(
            case_name=args.case_name,
            mesh_path=mesh_path,
            omega_list=args.omega_list,
            mode_indices=args.mode_indices,
            phi_normalization=phi_norm,
            depth=depth,
            results_dir=results_dir,
        )
        solve_s = time.perf_counter() - t0
        print(f"  radiation solve time: {solve_s:.1f} s")

        npz_dst = os.path.join(results_dir, f"modal_radiation_{tag}.npz")
        if os.path.abspath(npz_src) != os.path.abspath(npz_dst):
            import shutil

            shutil.copy2(npz_src, npz_dst)

        summary = _load_npz_summary(npz_dst)
        row = {
            "n_span": n_span,
            "n_chord": n_chord,
            "n_faces": n_faces,
            "solve_time_s": solve_s,
            "mesh_path": mesh_path,
            "npz_path": npz_dst,
            "mode_names": summary["mode_names"],
            "omega_asymptotic": summary["omega_asymptotic"],
            "A_asymptotic": summary["A_asymptotic"],
            "B_asymptotic": summary["B_asymptotic"],
        }
        rows.append(row)
        diag = np.diag(summary["A_asymptotic"])
        print(f"  A_ii @ high ω: {np.array2string(diag, precision=4)}")

    # CSV summary
    csv_path = os.path.join(out_dir, "mesh_convergence_summary.csv")
    mode_names = rows[0]["mode_names"]
    n_modes = len(mode_names)
    fieldnames = ["n_span", "n_chord", "n_faces", "solve_time_s", "omega_asymptotic_rad_s"]
    for i in range(n_modes):
        fieldnames.append(f"A_{i + 1}{i + 1}_kg")
    for i in range(n_modes):
        for j in range(n_modes):
            if i < j:
                fieldnames.append(f"A_{i + 1}{j + 1}_kg")
    for i in range(n_modes):
        fieldnames.append(f"B_{i + 1}{i + 1}_kg_s")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()
        for r in rows:
            out_row = {
                "n_span": r["n_span"],
                "n_chord": r["n_chord"],
                "n_faces": r["n_faces"],
                "solve_time_s": f"{r['solve_time_s']:.2f}",
                "omega_asymptotic_rad_s": f"{r['omega_asymptotic']:.6g}",
            }
            A = r["A_asymptotic"]
            B = r["B_asymptotic"]
            for i in range(n_modes):
                out_row[f"A_{i + 1}{i + 1}_kg"] = f"{A[i, i]:.8e}"
            for i in range(n_modes):
                for j in range(n_modes):
                    if i < j:
                        out_row[f"A_{i + 1}{j + 1}_kg"] = f"{A[i, j]:.8e}"
            for i in range(n_modes):
                out_row[f"B_{i + 1}{i + 1}_kg_s"] = f"{B[i, i]:.8e}"
            wr.writerow(out_row)
    print(f"\nWrote {csv_path}")

    if not args.no_plot and rows:
        plot_path = os.path.join(out_dir, "mesh_convergence_added_mass.png")
        _plot_convergence(rows, plot_path)
        print(f"Wrote {plot_path}")
        print(f"Wrote {plot_path.replace('_added_mass', '_damping')}")

    # Relative change finest vs previous
    if len(rows) >= 2:
        a_ref = rows[-1]["A_asymptotic"]
        a_prev = rows[-2]["A_asymptotic"]
        rel = np.max(np.abs(a_ref - a_prev) / np.maximum(np.abs(a_ref), 1e-12))
        print(
            f"\nMax |ΔA|/|A| between finest ({rows[-1]['n_span']}×{rows[-1]['n_chord']}) "
            f"and previous: {100 * rel:.2f} %"
        )


if __name__ == "__main__":
    main()
