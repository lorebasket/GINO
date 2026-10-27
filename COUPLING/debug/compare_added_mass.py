#!/usr/bin/env python3
"""Modal added mass, UVLM vs BEM, for any case: diagonal of
  - BEM A(omega) (Capytaine modal_radiation npz: omega, added_mass, added_damping)
  - UVLM apparent mass 0.5 rho Re Q(k) / k^2 from the GAF table (all k), and its non-circulatory
    fit A_UVLM_nc (utl/uvlm_nc_mass.py) if available.

    python COUPLING/debug/compare_added_mass.py --case TFOIL_SIMPLE -o am.png
    python COUPLING/debug/compare_added_mass.py --gaf Q.npz --bem AB.npz [--anc A_nc.npz] --nm 4 -o am.png
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "utl"))
import _common as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case")
    for k in ("gaf", "bem", "anc"):
        ap.add_argument("--" + k)
    ap.add_argument("--nm", type=int)
    ap.add_argument("--chord", type=float, help="reference chord: x axis in reduced frequency k b")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()

    P = C.case_paths(a.case, gaf=a.gaf, bem=a.bem, anc=a.anc) if a.case else dict(gaf=a.gaf, bem=a.bem, anc=a.anc)
    q = np.load(P["gaf"])
    k, Q, rho = q["k_list"], q["Q"], float(q["rho"])
    NM = a.nm or int(P.get("nm") or Q.shape[1])
    A_uvlm = 0.5 * rho * np.real(Q[:, :NM, :NM]) / np.maximum(k, 1e-9)[:, None, None] ** 2
    bem = np.load(P["bem"]) if P.get("bem") and os.path.exists(P["bem"]) else None
    anc = np.load(P["anc"])["A_nc"][:NM, :NM] if P.get("anc") and os.path.exists(P["anc"]) else None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nc = min(NM, 4)
    nr = int(np.ceil(NM / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(4 * nc, 3.2 * nr), squeeze=False)
    print("%4s %12s %12s %12s" % ("mode", "A_BEM(inf)", "A_UVLM_nc", "ratio"))
    for i in range(NM):
        ax = axs.flat[i]
        x = k * (a.chord / 2) if a.chord else k
        ax.plot(x[1:], A_uvlm[1:, i, i], color="#E4572E", label="UVLM 0.5ρReQ/k²")
        if anc is not None:
            ax.axhline(anc[i, i], color="#E4572E", ls="--", lw=1, label="A_UVLM_nc fit")
        if bem is not None:
            ab = bem["added_mass"][:, i, i]
            ax2 = ax.twiny()
            ax2.plot(bem["omega"], ab, color="#13A89E", label="BEM A(ω)")
            ax2.set_xlabel("ω [rad/s]", fontsize=7, color="#13A89E")
            ax.axhline(ab[-1], color="#13A89E", ls=":", lw=1)
        ax.set_title(f"mode {i + 1}", fontsize=9)
        ax.set_xlabel("k b" if a.chord else "k [1/m]", fontsize=8)
        ax.set_xscale("log")
        ax.grid(alpha=.3)
        ref = bem["added_mass"][-1, i, i] if bem is not None else np.nan
        nci = anc[i, i] if anc is not None else np.nan
        print("%4d %12.4g %12.4g %12.3f" % (i + 1, ref, nci, nci / ref if ref else np.nan))
    for j in range(NM, nr * nc):
        axs.flat[j].axis("off")
    axs.flat[0].legend(fontsize=7)
    fig.suptitle(f"modal added mass — {a.case or os.path.basename(P['gaf'])}", fontsize=10)
    fig.tight_layout()
    fig.savefig(a.out, dpi=130)
    print(a.out)


if __name__ == "__main__":
    main()
