#!/usr/bin/env python3
"""Dry modes of any case -> npz (pos, sub, Phi, f, M_hat, K_hat, C_hat) for the GAF / BEM precompute.

    conda run -n GINO python utl/export_modes.py TFOIL_SIMPLE            # -> paths()["dry"]
    conda run -n GINO python utl/export_modes.py Model2t --out my/dry_modes.npz
"""
import argparse
import contextlib
import io
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("case")
    ap.add_argument("--out", help="npz path (default: case paths()['dry'])")
    ap.add_argument("--n-print", type=int, default=12)
    a = ap.parse_args()

    os.chdir(C.REPO)
    cfg = C.get_config(a.case)
    out = a.out or C.case_paths(a.case)["dry"]
    from FEA import structural_model, structural_analysis
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        bm = structural_model.build(cfg)
        sr = structural_analysis.run_dry_analysis(bm, cfg)
    X = np.array([n["position"] for n in bm["nodes"]])
    sub = np.array([n.get("submodel", "?") for n in bm["nodes"]])
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    np.savez(out, pos=X, sub=sub, Phi=np.asarray(sr.dry_eigenvectors_full).real,
             f=np.asarray(sr.dry_frequencies), M_hat=np.asarray(sr.M_hat),
             K_hat=np.asarray(sr.K_hat), C_hat=np.asarray(sr.C_hat))
    print("saved", out, X.shape)
    print("dry f [Hz]", np.round(np.asarray(sr.dry_frequencies)[:a.n_print], 2))


if __name__ == "__main__":
    main()
