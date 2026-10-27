#!/usr/bin/env python3
"""Content of the dry modes in a dry_modes.npz (from export_modes.py): amplitude share per
sub-model and per DOF, so bending / torsion / chordwise modes can be told apart.

    python utl/classify_modes.py output_data/<CASE>/<variant>/dry_modes.npz [--nm 10] [--chord 0.3]

Rotations are weighted by a typical chord (--chord, m) to compare them with displacements.
Sub-models whose name ends in _port / _stbd are paired: the vertical DOF sign tells symmetric
from antisymmetric modes.
"""
import argparse

import numpy as np

DOF = ("ux", "uy", "uz", "rx", "ry", "rz")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz")
    ap.add_argument("--nm", type=int, default=10)
    ap.add_argument("--chord", type=float, default=0.3)
    ap.add_argument("--min-share", type=float, default=5.0, help="hide entries below this %%")
    a = ap.parse_args()

    d = np.load(a.npz, allow_pickle=True)
    pos, Phi, f = d["pos"], d["Phi"], d["f"]
    sub = d["sub"].astype(str) if "sub" in d else np.full(len(pos), "beam")
    U = Phi.reshape(len(pos), 6, -1)
    names = sorted(set(sub))
    pairs = [(n, n[:-5] + "_stbd") for n in names if n.endswith("_port") and n[:-5] + "_stbd" in names]
    for m in range(min(a.nm, Phi.shape[1])):
        share = {}
        for s in names:
            msk = sub == s
            for c in range(6):
                w = a.chord if c >= 3 else 1.0
                share[f"{s}:{DOF[c]}"] = w * np.sqrt((U[msk, c, m] ** 2).sum())
        tot = sum(share.values()) or 1.0
        txt = "  ".join(f"{k} {100 * v / tot:3.0f}%" for k, v in sorted(share.items(), key=lambda kv: -kv[1])
                        if 100 * v / tot >= a.min_share)
        for p, s in pairs:
            c = int(np.argmax([np.abs(U[sub == p, i, m]).sum() for i in range(3)]))
            sg = np.sign(U[sub == p, c, m].sum() * U[sub == s, c, m].sum())
            txt += f"  [{p[:-5]} {'sym' if sg > 0 else 'antisym' if sg < 0 else '-'}]"
        print(f"{m + 1:2d} {f[m]:8.2f} Hz  {txt}")


if __name__ == "__main__":
    main()
