#!/usr/bin/env python3
"""First PERSISTENT flutter in scan_*.txt files (from scan.py): a root with sigma/omega > g_min
and f_min < f < f_max that stays unstable (root within 15 %% in frequency) at >= n_pers speeds in
a window of `win` kn.  Isolated roots and near-zero-frequency (aero lag) roots are discarded,
as are roots with k = omega/V above the GAF table's k_max (extrapolated there).

    python utl/flutter_from_scan.py output_data/*/*/scan_delta.txt [--f-max 40 --k-max 40]
"""
import argparse

import numpy as np

KN = 0.514444


def first_flutter(path, g_min=1e-3, f_min=2.0, f_max=40.0, k_max=40.0, n_pers=3, win=5.0):
    a = np.loadtxt(path, ndmin=2)
    V = np.unique(a[:, 0])
    ok = lambda v, f: f_min < f < f_max and 2 * np.pi * f / (v * KN) <= k_max
    uns = {v: [(f, s / (2 * np.pi * f)) for _, f, s in a[a[:, 0] == v]
               if ok(v, f) and s / (2 * np.pi * f) > g_min] for v in V}
    for i, v in enumerate(V):
        for f, _g in uns[v]:
            hits = sum(1 for w in V[i + 1:] if w - v <= win and any(abs(ff - f) / f < 0.15 for ff, _ in uns[w]))
            if hits >= n_pers - 1:
                gmax = max(gg for w in V[i:] for ff, gg in uns[w] if abs(ff - f) / f < 0.3)
                return v, f, gmax
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--g-min", type=float, default=1e-3)
    ap.add_argument("--f-min", type=float, default=2.0)
    ap.add_argument("--f-max", type=float, default=40.0)
    ap.add_argument("--k-max", type=float, default=40.0)
    ap.add_argument("--n-pers", type=int, default=3)
    ap.add_argument("--win", type=float, default=5.0)
    a = ap.parse_args()
    for p in a.files:
        r = first_flutter(p, a.g_min, a.f_min, a.f_max, a.k_max, a.n_pers, a.win)
        print("%-70s %s" % (p.split("output_data/")[-1],
                            "no persistent flutter" if r is None else "flutter %.1f kn @ %.2f Hz (g max %.3f)" % r))


if __name__ == "__main__":
    main()
