#!/usr/bin/env python3
"""V-g / V-f from self-consistent scan files (scan_*.txt from scan.py), any case.

Branches are rebuilt by continuity: from the lowest speed, each branch takes at the next speed
the closest root in frequency (gap < --tol, one root per branch).  Roots below --f-min (aero lag
roots) and with k = omega/V above --k-max (GAF table extrapolated) are discarded.
With --raw every root is drawn as a point, no branch reconstruction.

    python utl/plot_vgvf_scan.py base/scan_uvlm.txt base/scan_delta.txt -l "UVLM" -l "UVLM+BEM" -o vgvf.png
"""
import argparse
import os

import numpy as np

KN = 0.514444


def load(path, f_min, f_max, k_max):
    a = np.loadtxt(path, ndmin=2)
    V, f, s = a[:, 0], a[:, 1], a[:, 2]
    keep = (f > f_min) & (f < f_max) & (2 * np.pi * f / (V * KN) <= k_max)
    return {v: sorted((ff, ss / (2 * np.pi * ff)) for ff, ss in zip(f[keep & (V == v)], s[keep & (V == v)]))
            for v in np.unique(V)}


def branches(roots, tol, min_len=8):
    br = []
    for v in sorted(roots):
        free = list(roots[v])
        for b in br:
            if b["dead"]:
                continue
            fl = b["f"][-1]
            c = [r for r in free if abs(r[0] - fl) / fl < tol]
            if not c:
                b["dead"] = True
                continue
            r = min(c, key=lambda r: abs(r[0] - fl) + 0.5 * fl * abs(r[1] - b["g"][-1]))
            free.remove(r)
            b["V"].append(v); b["f"].append(r[0]); b["g"].append(r[1])
        br += [dict(V=[v], f=[r[0]], g=[r[1]], dead=False) for r in free]
    return [b for b in br if len(b["V"]) >= min_len]


def crossings(b):
    V, g, f = map(np.asarray, (b["V"], b["g"], b["f"]))
    out = []
    for i in range(len(V) - 1):
        if (g[i] < 0 <= g[i + 1]) or (g[i] >= 0 > g[i + 1]):
            t = -g[i] / (g[i + 1] - g[i])
            out.append((V[i] + t * (V[i + 1] - V[i]), f[i] + t * (f[i + 1] - f[i]), g[i] < 0))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("-l", "--label", action="append")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--raw", action="store_true")
    ap.add_argument("--f-min", type=float, default=2.0)
    ap.add_argument("--f-max", type=float, default=40.0)
    ap.add_argument("--k-max", type=float, default=40.0)
    ap.add_argument("--tol", type=float, default=0.12)
    ap.add_argument("--g-lim", default="-0.25,0.12")
    ap.add_argument("--title", default="")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = a.label or [os.path.relpath(p) for p in a.files]
    n = len(a.files)
    fig, axs = plt.subplots(2, n, figsize=(6 * n, 8), sharex=True, squeeze=False)
    cols = plt.cm.tab10(np.linspace(0, 1, 10))
    for i, (p, lab) in enumerate(zip(a.files, labels)):
        ag, af = axs[0, i], axs[1, i]
        R = load(p, a.f_min, a.f_max, a.k_max)
        txt = []
        if a.raw:
            for v, rr in R.items():
                for f, g in rr:
                    c = "r" if g > 0 else "k"
                    ag.plot(v, g, ".", color=c, ms=3); af.plot(v, f, ".", color=c, ms=3)
        else:
            br = sorted(branches(R, a.tol), key=lambda b: b["f"][0])
            for k, b in enumerate(br):
                c = cols[k % 10]
                ag.plot(b["V"], b["g"], color=c, lw=1.6); af.plot(b["V"], b["f"], color=c, lw=1.6)
                for vc, fc, up in crossings(b):
                    mk = "o" if up else "s"
                    ag.plot(vc, 0, mk, color=c, mec="k", ms=8, zorder=5)
                    af.plot(vc, fc, mk, color=c, mec="k", ms=8, zorder=5)
                    txt.append("%s %.1f kn @ %.1f Hz" % ("unstable from" if up else "stable from", vc, fc))
        ag.axhline(0, color="k", lw=1)
        ag.set_ylim(*(float(x) for x in a.g_lim.split(",")))
        af.set_ylim(0, a.f_max)
        af.set_xlabel("V [kn]")
        ag.set_title(lab, fontsize=10, fontweight="bold")
        if txt:
            ag.text(0.02, 0.97, "\n".join(txt), transform=ag.transAxes, va="top", fontsize=8,
                    bbox=dict(fc="lightyellow", ec="grey", alpha=.9))
        for ax in (ag, af):
            ax.grid(alpha=.3)
    axs[0, 0].set_ylabel("g = σ/ω  (> 0 unstable)")
    axs[1, 0].set_ylabel("f [Hz]")
    fig.suptitle(a.title or "self-consistent scan — o: becomes unstable, □: back to stable", fontsize=10)
    fig.tight_layout()
    fig.savefig(a.out, dpi=130)
    print(a.out)


if __name__ == "__main__":
    main()
