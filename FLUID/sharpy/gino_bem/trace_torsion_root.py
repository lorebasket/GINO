#!/usr/bin/env python3
"""
Trace the Model2t hybrid wet-torsion root across the full sweep with NO
participation/frequency filters — pure MAC continuity on the beam eigenvector.

Answers: where does the 1T branch physically go after ~13 kn — overdamped
(imag -> 0), dissolved into wake-dominated content (part -> 0 while
oscillating), or hybridized (frequency jump with falling MAC)?
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_FSI_ROOT = Path(__file__).resolve().parents[3]
OUTDIR = (_FSI_ROOT / "output_data" / "Model2t" / "00_baseline_sensitivity"
          / "_baseline_sharpy_removegammadot_plusbem")
KN = 0.514444


def mac(a, b):
    return float(abs(np.vdot(a, b)) ** 2
                 / max(np.vdot(a, a).real * np.vdot(b, b).real, 1e-300))


def main() -> int:
    npz = np.load(next(OUTDIR.glob("model2t_sharpy_eigvecs_nm*.npz")))
    V = np.asarray(npz["velocities"], float)
    lam = np.asarray(npz["lam"])          # (n_v, n_eig)
    vecs = np.asarray(npz["struct_vecs"])  # (n_v, n_beam, n_eig)
    part = np.asarray(npz["participation"])
    nm = vecs.shape[1] // 2

    # seed: wet torsion at the first velocity = 1T-dominant root near 8.7 Hz
    freq0 = lam[0].imag / (2 * np.pi)
    cands = np.flatnonzero((lam[0].imag > 1e-6) & (freq0 > 5) & (freq0 < 12)
                           & (part[0] > 1e-3))
    k0 = cands[np.argmax([np.abs(vecs[0][1, k]) / max(np.linalg.norm(vecs[0][:, k]), 1e-30)
                          for k in cands])]
    ref = vecs[0][:, k0].copy()
    print(f"seed at V={V[0]/KN:.1f} kn: f={freq0[k0]:.3f} Hz, "
          f"g={lam[0][k0].real/abs(lam[0][k0].imag):+.4f}, part={part[0][k0]:.5f}")
    print(f"{'V_kn':>6} {'f_Hz':>9} {'sigma':>10} {'g':>9} {'part':>10} "
          f"{'MAC':>6}  {'q (1B 1T 2B m4)':>22}")
    for iv in range(len(V)):
        # best-MAC root over the ENTIRE spectrum (both half-planes, no filters)
        macs = np.abs(np.conj(ref) @ vecs[iv]) ** 2 / (
            (np.conj(ref) @ ref).real
            * np.maximum(np.einsum("ij,ij->j", np.conj(vecs[iv]), vecs[iv]).real, 1e-300))
        # restrict to upper half plane to keep one of each conjugate pair; allow imag ~ 0
        upper = lam[iv].imag >= -1e-9
        macs = np.where(upper, macs.real, -1.0)
        k = int(np.argmax(macs))
        l = lam[iv][k]
        q = np.abs(vecs[iv][:nm, k])
        q = q / max(q.max(), 1e-300)
        g = l.real / max(abs(l.imag), 1e-30)
        print(f"{V[iv]/KN:6.1f} {l.imag/2/np.pi:9.4f} {l.real:+10.4f} {g:+9.4f} "
              f"{part[iv][k]:10.6f} {macs[k]:6.3f}  "
              f"[{q[0]:.2f} {q[1]:.2f} {q[2]:.2f} {q[3]:.2f}]")
        if macs[k] > 0.4:
            ref = vecs[iv][:, k].copy()   # follow the branch (refresh reference)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
