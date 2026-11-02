#!/usr/bin/env python3
"""Summary of every branch of a p-k run folder (vg_data.csv + vf_data.csv): convergence,
frequency jumps, max g, g = 0 crossings.  Reading rule: converged points only, each branch cut
at its first frequency jump > --jump (default 20 %%).

    python utl/branches.py output_data/<CASE>/<run> [<run> ...]
"""
import argparse
import csv
import os

import numpy as np

KN = 1 / 0.514444


def load(d, jump=0.2):
    vg = list(csv.DictReader(open(os.path.join(d, "vg_data.csv"))))
    vf = list(csv.DictReader(open(os.path.join(d, "vf_data.csv"))))
    V = np.array([float(r["Velocity_m_s"]) for r in vg]) * KN
    nb = sum(1 for k in vg[0] if k.startswith("Damping_Branch_"))
    col = lambda rows, k: np.array([float(r[k]) if r.get(k) not in (None, "", "nan") else np.nan for r in rows])
    out = []
    for b in range(1, nb + 1):
        g, f = col(vg, f"Damping_Branch_{b}"), col(vf, f"Frequency_Hz_Branch_{b}")
        conv = np.array([r.get(f"Converged_Branch_{b}", "1") == "1" for r in vg])
        ok = conv & np.isfinite(g) & np.isfinite(f)
        idx = np.flatnonzero(ok)
        cut, nj = len(V), 0
        for i, j in zip(idx[:-1], idx[1:]):
            if f[i] > 0 and abs(f[j] - f[i]) / f[i] > jump:
                cut, nj = min(cut, j), nj + 1
        m = ok.copy()
        m[cut:] = False
        ii = np.flatnonzero(m)
        cr = []
        for i, j in zip(ii[:-1], ii[1:]):
            if g[i] < 0 <= g[j]:
                t = -g[i] / (g[j] - g[i])
                cr.append((V[i] + t * (V[j] - V[i]), f[i] + t * (f[j] - f[i])))
        out.append(dict(b=b, V=V, g=g, f=f, m=m, conv=ok.mean(), vcut=V[cut] if cut < len(V) else None,
                        nj=nj, cr=cr))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--jump", type=float, default=0.2)
    a = ap.parse_args()
    for d in a.dirs:
        print("==", d)
        for r in load(d, a.jump):
            gm = np.nanmax(r["g"][r["m"]]) if r["m"].any() else np.nan
            f0 = r["f"][r["m"]][0] if r["m"].any() else np.nan
            print("branch %d  f0 %6.2f Hz  conv %5.1f%%  jumps %d (cut at %s kn)  g_max %+.3f  g=0: %s"
                  % (r["b"], f0, 100 * r["conv"], r["nj"], "-" if r["vcut"] is None else "%.1f" % r["vcut"], gm,
                     ", ".join("%.1f kn @ %.2f Hz" % c for c in r["cr"]) or "-"))


if __name__ == "__main__":
    main()
