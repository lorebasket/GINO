#!/usr/bin/env python3
"""UVLM aerogrid convergence for the Abramson 1965 case — matrix-level, no flutter sweep.

Compares the linearized UVLM itself between aerogrids: the modal generalized
aerodynamic force (GAF) transfer matrix

    Q(i omega) = H_q(i omega) + i omega_s H_qdot(i omega)

sampled from ``ls.uvlm.ss`` (already projected onto the structural modes by
LinearAssembler: inputs = modal q, q_dot; outputs = modal generalized forces)
at fixed flow speeds. Wake factor Ms is held constant across grids so only the
bound-panel discretization varies.

Per-grid run (subprocess-friendly, saves one NPZ):
  conda activate sharpy
  python gino_bem/uvlm_grid_convergence.py --grid-span 20 --grid-chord 10

Final comparison (reads all NPZs, writes CSV + plots):
  python gino_bem/uvlm_grid_convergence.py --analyze
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

_SHARPY_ROOT = Path(__file__).resolve().parents[1]
_FSI_ROOT = _SHARPY_ROOT.parents[1]
if str(_SHARPY_ROOT) not in sys.path:
    sys.path.insert(0, str(_SHARPY_ROOT))

KN = 0.514444
MS_WAKE = 6
NUM_MODES = 4
SPEEDS_KN = (20.0, 48.1)          # low-speed check + flutter-relevant speed
FREQS_HZ = np.linspace(1.0, 40.0, 40)
# studies: lists of (N span, M chord); last entry = convergence reference
STUDIES = {
    # proportional refinement (beam mesh follows N — includes small structural change)
    "both": [(12, 6), (16, 8), (20, 10), (24, 12), (32, 16)],
    # chordwise only: N fixed -> identical beam, pure UVLM effect
    "chord": [(20, 4), (20, 6), (20, 10), (20, 16)],
    # spanwise only: M fixed (beam mesh still follows N)
    "span": [(12, 10), (16, 10), (20, 10), (24, 10), (32, 10)],
}
GRID_LIST = STUDIES["both"]
MODE_NAMES = ("bending", "torsion", "mode3", "mode4")

OUTDIR = (_FSI_ROOT / "output_data" / "ABRAMSON1965" /
          "08_SHARPy_discretization_convergence_analysis" / "uvlm_gaf_comparison")


def grid_tag(nspan: int, mchord: int) -> str:
    return f"n{nspan}c{mchord}"


# --------------------------------------------------------------------------- per-grid run
def run_grid(nspan: int, mchord: int) -> Path:
    from gino_bem.run_abramson_pk_bem import build_case

    tag = grid_tag(nspan, mchord)
    t0 = time.perf_counter()
    data, _ = build_case(
        NUM_MODES, grid=(mchord, nspan, MS_WAKE), added_mass_removal="gamma_dot",
        rayleigh=False, zetas=(0.0, 0.0), v_max_kn=50.0,
        case_tag=f"uvlm_conv_{tag}",
    )
    t_build = time.perf_counter() - t0
    ls = data.linear.linear_system

    in_names = [v.name for v in ls.uvlm.ss.input_variables]
    out_names = [v.name for v in ls.uvlm.ss.output_variables]
    print(f"[{tag}] uvlm.ss inputs={in_names} outputs={out_names} "
          f"states={ls.uvlm.ss.states}", flush=True)

    nm = NUM_MODES
    Q_all = np.zeros((len(SPEEDS_KN), nm, nm, len(FREQS_HZ)), dtype=complex)
    t_freq = 0.0
    for iv, v_kn in enumerate(SPEEDS_KN):
        V = v_kn * KN
        ls.update(float(V))
        t_ref = float(ls.uvlm.sys.ScalingFacts["length"]) / float(V)
        wv = 2.0 * np.pi * FREQS_HZ * t_ref          # physical -> scaled rad/s
        t1 = time.perf_counter()
        H = ls.uvlm.ss.freqresp(wv)                   # (n_out, n_in, n_freq)
        t_freq += time.perf_counter() - t1
        if H.shape[0] < nm or H.shape[1] < 2 * nm:
            raise RuntimeError(f"unexpected uvlm.ss dims {H.shape}; expected "
                               f">=({nm}, {2*nm}) — check modal projection")
        for i, w_s in enumerate(wv):
            Q_all[iv, :, :, i] = (H[:nm, :nm, i]
                                  + 1j * w_s * H[:nm, nm:2 * nm, i])
        print(f"[{tag}] V={v_kn} kn done ({time.perf_counter() - t1:.1f} s)", flush=True)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    npz = OUTDIR / f"gaf_{tag}.npz"
    np.savez(
        npz, Q=Q_all, freqs_hz=FREQS_HZ, speeds_kn=np.asarray(SPEEDS_KN),
        nspan=nspan, mchord=mchord, ms_wake=MS_WAKE,
        n_states=ls.uvlm.ss.states, n_panels=nspan * mchord,
        t_build_s=t_build, t_freqresp_s=t_freq,
    )
    print(f"[{tag}] saved {npz}  (build {t_build:.0f} s, freqresp {t_freq:.0f} s)")
    return npz


# --------------------------------------------------------------------------- analysis
# dataviz fixed-order categorical palette (validated)
PALETTE = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7"]
TEXT_2ND = "#57565c"


def analyze(study: str = "both") -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid_list = STUDIES[study]
    pfx = f"{study}_"
    runs = []
    for nspan, mchord in grid_list:
        npz = OUTDIR / f"gaf_{grid_tag(nspan, mchord)}.npz"
        if not npz.exists():
            print(f"!! missing {npz} — run that grid first"); continue
        d = np.load(npz)
        runs.append(dict(tag=grid_tag(nspan, mchord), Q=d["Q"],
                         n_panels=int(d["n_panels"]), n_states=int(d["n_states"]),
                         t_build=float(d["t_build_s"]), t_freq=float(d["t_freqresp_s"]),
                         freqs=d["freqs_hz"], speeds=d["speeds_kn"]))
    if len(runs) < 2:
        print("need at least 2 grids to compare"); return 1
    runs.sort(key=lambda r: r["n_panels"])
    ref = runs[-1]
    freqs, speeds = ref["freqs"], ref["speeds"]
    print(f"reference grid: {ref['tag']} ({ref['n_panels']} panels)")

    # relative Frobenius error vs frequency, per speed
    rows = []
    for r in runs[:-1]:
        err = np.linalg.norm(r["Q"] - ref["Q"], axis=(1, 2)) / \
              np.linalg.norm(ref["Q"], axis=(1, 2))            # (n_speed, n_freq)
        r["err"] = err
        for iv, v_kn in enumerate(speeds):
            rows.append(dict(
                grid=r["tag"], n_panels=r["n_panels"], V_kn=float(v_kn),
                err_max_pct=100 * float(err[iv].max()),
                err_mean_pct=100 * float(err[iv].mean()),
                err_at_17p5Hz_pct=100 * float(np.interp(17.5, freqs, err[iv])),
                t_build_s=r["t_build"], t_freqresp_s=r["t_freq"],
            ))
    csv_path = OUTDIR / f"{pfx}uvlm_gaf_convergence_summary.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for row in rows:
            w.writerow(row)

    def style(ax):
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors=TEXT_2ND)

    # Fig 1: |Q_jk| vs frequency, 2x2 leading block, at the flutter-relevant speed
    iv = len(speeds) - 1
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
    for (j, k), ax in zip([(0, 0), (0, 1), (1, 0), (1, 1)], axes.ravel()):
        for r, c in zip(runs, PALETTE):
            lw = 2.2 if r is ref else 1.6
            ls_ = "--" if r is ref else "-"
            ax.plot(freqs, np.abs(r["Q"][iv, j, k]), ls_, color=c, lw=lw,
                    label=f"{r['tag']} ({r['n_panels']}p)")
        ax.set_title(f"|Q[{MODE_NAMES[j]}, {MODE_NAMES[k]}]|", fontsize=10)
        style(ax)
    for ax in axes[1]:
        ax.set_xlabel("frequency [Hz]")
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle(f"UVLM modal GAF magnitude vs aerogrid — V = {speeds[iv]} kn "
                 f"(Ms = {MS_WAKE})", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUTDIR / f"{pfx}gaf_magnitude_vs_grid.png", dpi=150)
    plt.close(fig)

    # Fig 2: relative Frobenius error vs frequency, one panel per speed
    fig, axes = plt.subplots(1, len(speeds), figsize=(11, 4.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ivv, (v_kn, ax) in enumerate(zip(speeds, axes)):
        for r, c in zip(runs[:-1], PALETTE):
            ax.semilogy(freqs, 100 * r["err"][ivv], color=c, lw=1.6,
                        label=f"{r['tag']} ({r['n_panels']}p)")
        ax.axvline(17.5, color=TEXT_2ND, ls=":", lw=1)
        ax.set_title(f"V = {v_kn} kn", fontsize=10)
        ax.set_xlabel("frequency [Hz]")
        style(ax)
    axes[0].set_ylabel(f"‖Q − Q_{ref['tag']}‖_F / ‖Q_{ref['tag']}‖_F  [%]")
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle(f"UVLM GAF matrix error vs reference grid {ref['tag']}", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUTDIR / f"{pfx}gaf_error_vs_frequency.png", dpi=150)
    plt.close(fig)

    # Fig 3: convergence + cost vs panel count (error at 17.5 Hz, flutter-relevant speed)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    npan = [r["n_panels"] for r in runs[:-1]]
    e175 = [100 * np.interp(17.5, freqs, r["err"][iv]) for r in runs[:-1]]
    ax1.loglog(npan, e175, "o-", color=PALETTE[0], lw=1.6, ms=6)
    for r, x, y in zip(runs[:-1], npan, e175):
        ax1.annotate(r["tag"], (x, y), textcoords="offset points", xytext=(6, 5),
                     fontsize=8, color=TEXT_2ND)
    ax1.set_xlabel("bound panels")
    ax1.set_ylabel(f"GAF error at 17.5 Hz, {speeds[iv]} kn [%]")
    style(ax1)
    ax2.plot([r["n_panels"] for r in runs], [r["t_build"] for r in runs], "o-",
             color=PALETTE[1], lw=1.6, ms=6, label="case build [s]")
    ax2.plot([r["n_panels"] for r in runs], [r["t_freq"] for r in runs], "o-",
             color=PALETTE[2], lw=1.6, ms=6, label="freqresp (80 solves) [s]")
    ax2.set_xlabel("bound panels")
    ax2.set_ylabel("wall time [s]")
    ax2.legend(fontsize=8, frameon=False)
    style(ax2)
    fig.suptitle("UVLM grid convergence and cost", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUTDIR / f"{pfx}gaf_convergence_vs_panels.png", dpi=150)
    plt.close(fig)

    print(f"\nsummary CSV: {csv_path}")
    for row in rows:
        print(f"  {row['grid']:>7s}  V={row['V_kn']:4.1f} kn  "
              f"max {row['err_max_pct']:6.2f}%  mean {row['err_mean_pct']:6.2f}%  "
              f"@17.5Hz {row['err_at_17p5Hz_pct']:6.2f}%")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid-span", type=int, default=None)
    ap.add_argument("--grid-chord", type=int, default=None)
    ap.add_argument("--analyze", type=str, default=None, choices=list(STUDIES))
    args = ap.parse_args()
    if args.analyze:
        return analyze(args.analyze)
    if args.grid_span is None or args.grid_chord is None:
        raise SystemExit("need --grid-span and --grid-chord (or --analyze)")
    run_grid(args.grid_span, args.grid_chord)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
