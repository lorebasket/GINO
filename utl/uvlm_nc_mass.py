#!/usr/bin/env python3
"""Non-circulatory (apparent) mass of the UVLM from a GAF table: fit Re Q = a + (2/rho) A k^2
over the k range where the grid resolves the oscillation (k dx <~ 0.5) and the circulatory
part is already nearly constant (k b >~ 2).  Full matrix (one fit per entry), symmetrised.
Needed by the delta stage of scan.py (UVLM table intact + (A_BEM - A_UVLM_nc) in M).

    python utl/uvlm_nc_mass.py --case TFOIL_SIMPLE                       # paths()["gaf"] -> paths()["anc"]
    python utl/uvlm_nc_mass.py --gaf FLUID/PanelAero/Qjj/Quvlm_X.npz --out A_nc.npz --fit 10,20
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402


def fit_nc_mass(k, Q, rho, lo, hi):
    m = (k >= lo) & (k <= hi)
    if m.sum() < 3:
        raise SystemExit(f"only {m.sum()} k points in [{lo}, {hi}]: widen --fit")
    X = np.column_stack([np.ones(m.sum()), k[m] ** 2])
    Y = Q[m].real.reshape(m.sum(), -1)
    coef = np.linalg.lstsq(X, Y, rcond=None)[0]
    nm = Q.shape[1]
    A = 0.5 * rho * coef[1].reshape(nm, nm)
    res = np.abs(X @ coef - Y).max() / np.abs(Y).max()
    return 0.5 * (A + A.T), res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case")
    ap.add_argument("--gaf")
    ap.add_argument("--out")
    ap.add_argument("--fit", default="10,20", help="k range lo,hi [1/m] (default suits ~20 chord panels)")
    a = ap.parse_args()
    P = C.case_paths(a.case, gaf=a.gaf, anc=a.out) if a.case else dict(gaf=a.gaf, anc=a.out)
    if not P.get("gaf") or not P.get("anc"):
        raise SystemExit("give --case, or --gaf and --out")
    lo, hi = (float(x) for x in a.fit.split(","))
    q = np.load(P["gaf"])
    A, res = fit_nc_mass(q["k_list"], q["Q"], float(q["rho"]), lo, hi)
    np.savez(P["anc"], A_nc=A, k_fit=np.array([lo, hi]), gaf=P["gaf"], grid=P.get("grid", ""))
    print(f"A_UVLM_nc (fit k in [{lo:g}, {hi:g}], residual {res:.1e}) diag:", np.round(np.diag(A), 3))
    print("written", P["anc"])


if __name__ == "__main__":
    main()
