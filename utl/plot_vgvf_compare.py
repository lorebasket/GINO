#!/usr/bin/env python3
"""V-g / V-f overlay of several p-k run folders (vg_data.csv + vf_data.csv), any case.

Replaces the per-case comparison plots (methods, frameworks, baselines, sensitivity overlays).
Only converged points are drawn (Converged_Branch_N); branches can be cut at the first
frequency jump (--jump) and g = 0 crossings are marked.

    python utl/plot_vgvf_compare.py runA runB runC -l "UVLM" -l "UVLM+BEM" -l "DLM" \\
        --exp 18.1,6.4 --branches 1,2 -o output_data/Model2t/compare.png

A SHARPy velocity_eigs CSV (gino_bem output) can be added with --sharpy file.csv[:label].
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import branches as BR  # noqa: E402


def _sharpy(path):
    """SHARPy velocity_eigs CSV (V_m_s, V_kn, mode_index, real, imag, freq_Hz) -> (V_kn, g, f)."""
    a = np.genfromtxt(path, delimiter=",", names=True)
    keep = a["imag"] > 1e-3
    return a["V_kn"][keep], a["real"][keep] / a["imag"][keep], a["freq_Hz"][keep]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("-l", "--label", action="append", help="one per folder (default: folder name)")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--branches", help="comma list of branch numbers to draw (default all)")
    ap.add_argument("--jump", type=float, default=None, help="cut branches at this relative f jump")
    ap.add_argument("--exp", help="experimental flutter point V_kn,f_Hz")
    ap.add_argument("--sharpy", action="append", default=[], help="velocity_eigs.csv[:label]")
    ap.add_argument("--g-lim", default="-0.4,0.2")
    ap.add_argument("--f-lim")
    ap.add_argument("--title", default="")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = a.label or [os.path.basename(os.path.normpath(d)).lstrip("_") for d in a.dirs]
    want = {int(b) for b in a.branches.split(",")} if a.branches else None
    styles = ("-", "--", "-.", ":")
    fig, (ag, af) = plt.subplots(2, 1, sharex=True, figsize=(8, 8))
    cols = plt.cm.tab10(np.linspace(0, 1, 10))
    for i, (d, lab) in enumerate(zip(a.dirs, labels)):
        for r in BR.load(d, a.jump if a.jump else 1e9):
            if want and r["b"] not in want:
                continue
            m = r["m"]
            kw = dict(color=cols[i % 10], ls=styles[(r["b"] - 1) % 4], lw=1.6)
            ag.plot(r["V"][m], r["g"][m], label=f"{lab} — branch {r['b']}", **kw)
            af.plot(r["V"][m], r["f"][m], **kw)
            for vc, fc in r["cr"]:
                ag.plot(vc, 0, "o", color=cols[i % 10], mec="k", zorder=5)
                af.plot(vc, fc, "o", color=cols[i % 10], mec="k", zorder=5)
                print(f"{lab} branch {r['b']}: g=0 at {vc:.2f} kn, {fc:.2f} Hz")
    for j, s in enumerate(a.sharpy):
        p, _, lab = s.partition(":")
        V, g, f = _sharpy(p)
        c = cols[(len(a.dirs) + j) % 10]
        ag.scatter(V, g, s=5, color=c, label=lab or "SHARPy")
        af.scatter(V, f, s=5, color=c)
    if a.exp:
        ve, fe = (float(x) for x in a.exp.split(","))
        ag.axvline(ve, color="k", ls=":", lw=1)
        af.plot(ve, fe, "k*", ms=12, label="experiment")
        ag.plot([], [], "k*", ms=10, label=f"experiment {ve:g} kn / {fe:g} Hz")
    ag.axhline(0, color="k", lw=0.8)
    ag.set_ylim(*(float(x) for x in a.g_lim.split(",")))
    if a.f_lim:
        af.set_ylim(*(float(x) for x in a.f_lim.split(",")))
    ag.set_ylabel("g")
    af.set_ylabel("f [Hz]")
    af.set_xlabel("V [kn]")
    for ax in (ag, af):
        ax.grid(alpha=.3)
    ag.legend(fontsize=8, loc="best")
    fig.suptitle(a.title, fontsize=11)
    fig.tight_layout()
    fig.savefig(a.out, dpi=150)
    print(a.out)


if __name__ == "__main__":
    main()
