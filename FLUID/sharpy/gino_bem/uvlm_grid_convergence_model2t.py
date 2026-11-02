#!/usr/bin/env python3
"""UVLM aerogrid convergence for the Model 2T case — matrix-level, no flutter sweep.

Model2t twin of ``gino_bem/uvlm_grid_convergence.py`` (Abramson). Compares the
linearized UVLM between aerogrids through the modal GAF transfer matrix

    Q(i omega) = H_q(i omega) + i omega_s H_qdot(i omega)

sampled from ``ls.uvlm.ss`` (post-LinearAssembler: inputs modal q, q_dot;
outputs modal generalized forces) at fixed flow speeds. Wake factor Ms is held
constant so only the bound-panel discretization varies.

Model2t constraint: the dry-root mask needs a beam node on the waterline arc
(s = 0.2524 m = span/5), and the FlyingWing template needs an even element
count (num_elem = N//2) — so N must be a multiple of 10.

Stiffness: NOMINAL GJ/EI (gj_factor = ei_factor = 1), matching the
``_baseline_sharpy`` production line (the fairer SHARPy comparison — feeding
the BEM-calibrated GJ to SHARPy double-counts the softening).

Per-grid run (subprocess-friendly, saves one NPZ):
  conda activate sharpy
  python gino_bem/uvlm_grid_convergence_model2t.py --grid-span 20 --grid-chord 8

Final comparison (reads all NPZs, writes CSV + plots):
  python gino_bem/uvlm_grid_convergence_model2t.py --analyze both
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
SPEEDS_KN = (10.0, 18.1)          # low-speed check + experimental flutter speed
FREQS_HZ = np.linspace(0.5, 30.0, 40)
FREQ_MARK_HZ = 6.4                # Besch & Liu flutter frequency (config G)
V_REF_KN = 18.1                   # reference speed for UVLM GAF k-sweep precalc
K_MAX = 40.0
N_K_PRODUCTION = 81               # matches precompute_quvlm.py default
# studies: lists of (N span, M chord); last entry = convergence reference.
# N multiple of 10 (waterline node + even element count).
STUDIES = {
    # proportional refinement (beam mesh follows N — includes small structural change)
    "both": [(10, 4), (20, 8), (30, 12), (40, 16)],
    # chordwise only: N fixed -> identical beam, pure UVLM effect
    "chord": [(20, 4), (20, 6), (20, 8), (20, 12), (20, 16)],
    # spanwise only: M fixed (beam mesh still follows N)
    "span": [(10, 8), (20, 8), (30, 8), (40, 8)],
}
MODE_NAMES = ("bending", "torsion", "mode3", "mode4")

OUTDIR = (_FSI_ROOT / "output_data" / "Model2t" /
          "01_discretization_convergence_analysis" / "uvlm_gaf_comparison")


def production_k_list(k_max: float = K_MAX, n_k: int = N_K_PRODUCTION) -> np.ndarray:
    """k grid used in production ``precompute_quvlm.py`` (one-time GAF table)."""
    k_list = np.unique(np.concatenate([
        [0.0],
        np.linspace(0.02, 2.0, 34),
        np.linspace(2.2, 8.0, 22),
        np.linspace(8.5, k_max, n_k - 57),
    ]))
    return k_list


def time_gaf_precalc(ls, *, v_ref_kn: float = V_REF_KN) -> tuple[float, int]:
    """Wall time for production UVLM modal GAF table over the k-sweep at V_ref."""
    V = float(v_ref_kn) * KN
    ls.update(V)
    t_ref = float(ls.uvlm.sys.ScalingFacts["length"]) / V
    k_list = production_k_list()
    w_s = 2.0 * np.pi * k_list * V * t_ref
    dt_s = float(ls.uvlm.ss.dt)
    w_nyq = np.pi / dt_s
    if w_s[-1] > 0.9 * w_nyq:
        keep = w_s <= 0.9 * w_nyq
        k_list, w_s = k_list[keep], w_s[keep]
    t0 = time.perf_counter()
    ls.uvlm.ss.freqresp(w_s)
    return time.perf_counter() - t0, int(len(k_list))


def grid_tag(nspan: int, mchord: int) -> str:
    return f"n{nspan}c{mchord}"


# --------------------------------------------------------------------------- per-grid run
def build_case(nm: int, *, grid: tuple[int, int, int], case_tag: str):
    """Model2t case up to LinearAssembler (no stability sweep, full-order UVLM)."""
    from gino_bem import run_model2t_water as R
    from gino_bem.model2t_strut import Model2tStrut
    import sharpy.sharpy_main

    M, N, Ms = grid
    if N % 10:
        raise ValueError(f"N must be a multiple of 10 (waterline node), got {N}")
    route = _SHARPY_ROOT / "gino_bem"
    (route / "cases").mkdir(parents=True, exist_ok=True)
    out_root = route / "output"
    out_root.mkdir(exist_ok=True)
    ws = Model2tStrut(
        M=M, N=N, Mstar_fact=Ms, u_inf=1.0, alpha=0.0, rho=997.0,
        gj_factor=1.0, ei_factor=1.0,
        route=str(route / "cases"), case_name=case_tag,
    )
    ws.clean_test_files()
    ws.update_derived_params()
    ws.set_default_config_dict()
    ws.generate_aero_file()
    ws.generate_fem_file()
    vel = R._model2t_velocity_list_m_s()
    R._configure_sharpy_case(
        ws, num_modes=nm, velocity_list=vel,
        v_min=float(vel.min()), v_max=float(vel.max()), n_vel=len(vel),
        out_root=out_root, rom_settings=None, u_inf=1.0, k_points=[0.0],
    )
    # GAF sampling only: stop after LinearAssembler
    ws.config["SHARPy"]["flow"] = [
        "BeamLoader", "AerogridLoader", "StaticCoupled", "Modal", "LinearAssembler",
    ]
    ws.config.write()
    print(f">>> Model2t UVLM grid M,N,Ms = {grid} (nominal stiffness)", flush=True)
    data = sharpy.sharpy_main.main(["", ws.route + ws.case_name + ".sharpy"])
    return data


def run_grid(nspan: int, mchord: int) -> Path:
    tag = grid_tag(nspan, mchord)
    t0 = time.perf_counter()
    data = build_case(NUM_MODES, grid=(mchord, nspan, MS_WAKE),
                      case_tag=f"m2t_uvlm_conv_{tag}")
    t_build = time.perf_counter() - t0
    ls = data.linear.linear_system

    in_names = [v.name for v in ls.uvlm.ss.input_variables]
    out_names = [v.name for v in ls.uvlm.ss.output_variables]
    print(f"[{tag}] uvlm.ss inputs={in_names} outputs={out_names} "
          f"states={ls.uvlm.ss.states}", flush=True)

    t_gaf, n_k = time_gaf_precalc(ls, v_ref_kn=V_REF_KN)
    print(f"[{tag}] GAF precalc ({n_k} k @ {V_REF_KN} kn): {t_gaf:.1f} s", flush=True)

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
        t_build_s=t_build, t_gaf_precalc_s=t_gaf, t_freqresp_s=t_freq,
        n_k_precalc=n_k, v_ref_kn=V_REF_KN,
    )
    print(f"[{tag}] saved {npz}  (build {t_build:.0f} s, GAF precalc {t_gaf:.0f} s, "
          f"convergence freqresp {t_freq:.0f} s)")
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
                         t_build=float(d["t_build_s"]),
                         t_gaf=float(d["t_gaf_precalc_s"]) if "t_gaf_precalc_s" in d else np.nan,
                         t_freq=float(d["t_freqresp_s"]),
                         freqs=d["freqs_hz"], speeds=d["speeds_kn"]))
    if len(runs) < 2:
        print("need at least 2 grids to compare"); return 1
    runs.sort(key=lambda r: r["n_panels"])
    ref = runs[-1]
    freqs, speeds = ref["freqs"], ref["speeds"]
    print(f"reference grid: {ref['tag']} ({ref['n_panels']} panels)")

    fmark = FREQ_MARK_HZ
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
                err_at_flutter_freq_pct=100 * float(np.interp(fmark, freqs, err[iv])),
                t_build_s=r["t_build"], t_gaf_precalc_s=r["t_gaf"],
                t_freqresp_s=r["t_freq"],
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
    fig.suptitle(f"Model2t UVLM modal GAF magnitude vs aerogrid — V = {speeds[iv]} kn "
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
        ax.axvline(fmark, color=TEXT_2ND, ls=":", lw=1)
        ax.set_title(f"V = {v_kn} kn", fontsize=10)
        ax.set_xlabel("frequency [Hz]")
        style(ax)
    axes[0].set_ylabel(f"‖Q − Q_{ref['tag']}‖_F / ‖Q_{ref['tag']}‖_F  [%]")
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle(f"Model2t UVLM GAF matrix error vs reference grid {ref['tag']}", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUTDIR / f"{pfx}gaf_error_vs_frequency.png", dpi=150)
    plt.close(fig)

    # Fig 3: convergence + cost vs panel count (error at f_flutter, flutter-relevant speed)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    npan = [r["n_panels"] for r in runs[:-1]]
    e_f = [100 * np.interp(fmark, freqs, r["err"][iv]) for r in runs[:-1]]
    ax1.loglog(npan, e_f, "o-", color=PALETTE[0], lw=1.6, ms=6)
    for r, x, y in zip(runs[:-1], npan, e_f):
        ax1.annotate(r["tag"], (x, y), textcoords="offset points", xytext=(6, 5),
                     fontsize=8, color=TEXT_2ND)
    ax1.set_xlabel("bound panels")
    ax1.set_ylabel(f"GAF error at {fmark} Hz, {speeds[iv]} kn [%]")
    style(ax1)
    ax2.plot([r["n_panels"] for r in runs], [r["t_build"] for r in runs], "o-",
             color=PALETTE[1], lw=1.6, ms=6, label="case build [s]")
    ax2.plot([r["n_panels"] for r in runs],
             [r["t_gaf"] if np.isfinite(r["t_gaf"]) else 0.0 for r in runs], "o-",
             color=PALETTE[2], lw=1.6, ms=6, label=f"GAF precalc ({N_K_PRODUCTION} k) [s]")
    ax2.plot([r["n_panels"] for r in runs], [r["t_freq"] for r in runs], "o-",
             color=PALETTE[3], lw=1.6, ms=6, label="convergence eval [s]")
    ax2.set_xlabel("bound panels")
    ax2.set_ylabel("wall time [s]")
    ax2.legend(fontsize=8, frameon=False)
    style(ax2)
    fig.suptitle("Model2t UVLM grid convergence and cost", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUTDIR / f"{pfx}gaf_convergence_vs_panels.png", dpi=150)
    plt.close(fig)

    print(f"\nsummary CSV: {csv_path}")
    for row in rows:
        print(f"  {row['grid']:>7s}  V={row['V_kn']:4.1f} kn  "
              f"max {row['err_max_pct']:6.2f}%  mean {row['err_mean_pct']:6.2f}%  "
              f"@{fmark}Hz {row['err_at_flutter_freq_pct']:6.2f}%")
    return 0


def patch_gaf_precalc_timing(nspan: int, mchord: int) -> None:
    """Re-time production GAF precalc on an existing grid NPZ (no Q recomputation)."""
    tag = grid_tag(nspan, mchord)
    npz = OUTDIR / f"gaf_{tag}.npz"
    if not npz.is_file():
        raise FileNotFoundError(npz)
    t0 = time.perf_counter()
    data = build_case(NUM_MODES, grid=(mchord, nspan, MS_WAKE),
                      case_tag=f"m2t_uvlm_conv_{tag}_timing")
    t_build = time.perf_counter() - t0
    ls = data.linear.linear_system
    t_gaf, n_k = time_gaf_precalc(ls, v_ref_kn=V_REF_KN)
    with np.load(npz) as d:
        payload = {k: d[k] for k in d.files}
    payload.update(
        t_build_s=t_build, t_gaf_precalc_s=t_gaf, n_k_precalc=n_k, v_ref_kn=V_REF_KN,
    )
    np.savez(npz, **payload)
    print(f"[{tag}] patched timing: build {t_build:.1f} s, GAF precalc {t_gaf:.1f} s")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid-span", type=int, default=None)
    ap.add_argument("--grid-chord", type=int, default=None)
    ap.add_argument("--analyze", type=str, default=None, choices=list(STUDIES))
    ap.add_argument("--patch-timing", action="store_true",
                    help="re-time GAF k-sweep precalc on existing NPZs (no Q rebuild)")
    args = ap.parse_args()
    if args.patch_timing:
        if args.grid_span is None or args.grid_chord is None:
            for nspan, mchord in STUDIES["both"]:
                patch_gaf_precalc_timing(nspan, mchord)
        else:
            patch_gaf_precalc_timing(args.grid_span, args.grid_chord)
        return 0
    if args.analyze:
        return analyze(args.analyze)
    if args.grid_span is None or args.grid_chord is None:
        raise SystemExit("need --grid-span and --grid-chord (or --analyze)")
    run_grid(args.grid_span, args.grid_chord)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
