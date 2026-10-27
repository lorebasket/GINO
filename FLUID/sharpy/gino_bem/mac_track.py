#!/usr/bin/env python3
"""
MAC (Modal Assurance Criterion) mode tracking for the Abramson SHARPy flutter sweep.

SHARPy has no built-in flutter mode tracker; this is the gold-standard post-processing.
Reads the per-velocity eigenvectors saved by ``track_structural_modes.py``
(``abramson_sharpy_eigvecs_nm{nm}.npz``) and follows each structural branch
(bending, torsion) across velocity by MAXIMUM MODAL-ASSURANCE CORRELATION of the
structural eigenvector block, rather than by frequency band. Seeds are classified by
which structural DOF (q0=1st bending, q1=1st torsion) dominates the mode shape.

  python gino_bem/mac_track.py --npz <eigvecs.npz> --out <png>

MAC(a,b) = |a^H b|^2 / ((a^H a)(b^H b)),  a,b = structural (q, q_dot) sub-vectors.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SHARPY_ROOT = Path(__file__).resolve().parents[1]
if str(_SHARPY_ROOT) not in sys.path:
    sys.path.insert(0, str(_SHARPY_ROOT))

import numpy as np

from gino_bem.abramson_tracking_config import ABRAMSON_V_MIN_KN, at_or_above_m_s

V_EXP_KN, F_EXP_HZ = 48.1, 17.5


def mac(a: np.ndarray, b: np.ndarray) -> float:
    num = np.abs(np.vdot(a, b)) ** 2
    den = float(np.vdot(a, a).real) * float(np.vdot(b, b).real)
    return num / den if den > 1e-30 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--v-seed-kn", type=float, default=12.0,
                    help="velocity at which structural seeds are classified (clean spectrum)")
    ap.add_argument("--v-min-kn", type=float, default=ABRAMSON_V_MIN_KN,
                    help="do not plot or export points below this speed (wake-polluted region)")
    ap.add_argument("--part-floor", type=float, default=0.002)
    ap.add_argument("--mac-floor", type=float, default=0.25,
                    help="below this MAC the branch is considered lost at that velocity")
    ap.add_argument("--title", type=str, default="Abramson SHARPy UVLM water (MAC-tracked)")
    ap.add_argument("--v-max-kn", type=float, default=55.0, help="plot x-axis max [kn]")
    args = ap.parse_args()

    d = np.load(args.npz)
    vels = d["velocities"]
    lam = d["lam"]                 # (nV, Ns) continuous-time
    SV = d["struct_vecs"]          # (nV, nb, Ns)
    P = d["participation"]         # (nV, Ns)
    KN = float(d["kn"])
    nb = SV.shape[1]
    nm = nb // 2                   # q, q_dot -> displacement block = first nm rows

    order = np.argsort(vels)
    vels, lam, SV, P = vels[order], lam[order], SV[order], P[order]
    nV, Ns = lam.shape
    freq = np.abs(lam.imag) / (2 * np.pi)

    def candidates(i):
        return [k for k in range(Ns)
                if lam[i, k].imag > 1e-6 and P[i, k] >= args.part_floor
                and 1.0 <= freq[i, k] <= 40.0]

    # --- seed classification at v_seed ---
    seed_i = int(np.argmin(np.abs(vels / KN - args.v_seed_kn)))
    seeds = {}   # dominant-DOF index -> state index k
    for k in candidates(seed_i):
        disp = np.abs(SV[seed_i, :nm, k])        # displacement components q0..q_{nm-1}
        dom = int(np.argmax(disp))
        if dom not in seeds or P[seed_i, k] > P[seed_i, seeds[dom]]:
            seeds[dom] = k
    branch_seed = {}
    if 0 in seeds:
        branch_seed["bending"] = seeds[0]
    if 1 in seeds:
        branch_seed["torsion"] = seeds[1]
    if not branch_seed:
        raise SystemExit("No structural seeds found; lower --part-floor or change --v-seed-kn")
    print(f"Seeds @ {vels[seed_i]/KN:.1f} kn: " +
          ", ".join(f"{n}=state{k} (f={freq[seed_i,k]:.1f}Hz, p={P[seed_i,k]:.3f})"
                    for n, k in branch_seed.items()))

    series = {n: {} for n in branch_seed}       # name -> {i: k}
    for n, k in branch_seed.items():
        series[n][seed_i] = k

    def march(rng, refs):
        local = {n: refs[n].copy() for n in branch_seed}
        for i in rng:
            cand = candidates(i)
            if not cand:
                continue
            scored = []
            for n in branch_seed:
                for k in cand:
                    scored.append((mac(local[n], SV[i, :, k]), n, k))
            scored.sort(reverse=True, key=lambda t: t[0])
            used_k, used_n = set(), set()
            for m, n, k in scored:
                if n in used_n or k in used_k or m < args.mac_floor:
                    continue
                series[n][i] = k
                local[n] = SV[i, :, k].copy()
                used_n.add(n)
                used_k.add(k)

    refs = {n: SV[seed_i, :, k].copy() for n, k in branch_seed.items()}
    march(range(seed_i + 1, nV), refs)          # forward
    v_min_i = int(np.searchsorted(vels, args.v_min_kn * KN, side="left"))
    backward_rng = range(seed_i - 1, v_min_i - 1, -1)
    if len(backward_rng) > 0:
        march(backward_rng, refs)                # backward (not below v_min)

    # --- assemble branch curves ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = {"bending": "#1f77b4", "torsion": "#ff7f0e"}
    fig, (axg, axf) = plt.subplots(1, 2, figsize=(13, 5))
    vfl = None
    tracked = {}
    for n in branch_seed:
        idx = sorted(i for i in series[n] if at_or_above_m_s(vels[i]))
        v = np.array([vels[i] for i in idx])
        re = np.array([lam[i, series[n][i]].real for i in idx])
        fz = np.array([freq[i, series[n][i]] for i in idx])
        tracked[n] = (v, re, fz)
        axg.plot(v / KN, re, "-", color=cols[n], lw=2, label=n)
        axf.plot(v / KN, fz, "-", color=cols[n], lw=2, label=n)
        cx = np.where(np.diff(np.sign(re)) > 0)[0]
        if len(cx):
            j = cx[0]
            a = re[j] / (re[j] - re[j + 1])
            vc = (v[j] + a * (v[j + 1] - v[j])) / KN
            fc = fz[j] + a * (fz[j + 1] - fz[j])
            if vfl is None or vc < vfl[0]:
                vfl = (vc, fc, n)

    axg.axhline(0, color="k", lw=0.8)
    axg.axvline(V_EXP_KN, color="green", ls=":", lw=1.6, label=f"exp {V_EXP_KN} kn")
    if vfl:
        axg.axvline(vfl[0], color="red", ls="--", lw=1.3, label=f"flutter {vfl[0]:.1f} kn ({vfl[2]})")
    axg.set_xlabel("V [kn]"); axg.set_ylabel("damping: Re(λ) [1/s]")
    axg.set_title("V-g (MAC-tracked)"); axg.set_xlim(args.v_min_kn, args.v_max_kn); axg.set_ylim(-12, 8)
    axg.grid(alpha=0.3); axg.legend(fontsize=8)

    axf.plot(V_EXP_KN, F_EXP_HZ, "*", color="green", ms=15, label=f"exp {F_EXP_HZ} Hz")
    axf.axhline(20.14, color="gray", ls=":", lw=1, label="dry torsion 20.1 Hz")
    axf.axhline(9.93, color="lightgray", ls=":", lw=1, label="dry bending 9.9 Hz")
    axf.set_xlabel("V [kn]"); axf.set_ylabel("frequency [Hz]")
    axf.set_title("V-f (MAC-tracked)"); axf.set_xlim(args.v_min_kn, args.v_max_kn); axf.set_ylim(0, 25)
    axf.grid(alpha=0.3); axf.legend(fontsize=8)
    fig.suptitle(args.title)
    fig.tight_layout()
    out = args.out or args.npz.with_name(args.npz.stem.replace("eigvecs", "mac_tracked") + ".png")
    fig.savefig(out, dpi=140)

    import csv
    with open(out.with_suffix(".csv"), "w", newline="") as f:
        wri = csv.writer(f)
        wri.writerow(["branch", "V_m_s", "V_kn", "real_1_s", "freq_Hz"])
        for n, (v, re, fz) in tracked.items():
            for a, b, c in zip(v, re, fz):
                wri.writerow([n, a, a / KN, b, c])
    print("saved plot:", out)
    print("saved CSV: ", out.with_suffix(".csv"))
    if vfl:
        print(f"flutter (MAC): {vfl[0]:.2f} kn @ {vfl[1]:.2f} Hz ({vfl[2]})")


if __name__ == "__main__":
    main()
