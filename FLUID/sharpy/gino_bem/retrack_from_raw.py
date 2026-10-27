#!/usr/bin/env python3
"""
Offline re-tracking of Abramson SHARPy structural branches from a raw-modes CSV.

Decoupled from the expensive eig loop in ``track_structural_modes.py``: that script
dumps EVERY mode (freq 1-40 Hz) with its structural participation to
``abramson_sharpy_raw_modes_nm{nm}.csv``; here we (1) diagnose whether the first
zero-crossing is a real structural mode or a spurious wake mode, and (2) track the
bending & torsion branches by continuity and plot clean V-g / V-f.

  python gino_bem/retrack_from_raw.py --raw <raw_modes.csv> --out <png> [--branches 2]
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

_SHARPY_ROOT = Path(__file__).resolve().parents[1]
if str(_SHARPY_ROOT) not in sys.path:
    sys.path.insert(0, str(_SHARPY_ROOT))

import numpy as np

from gino_bem.abramson_tracking_config import ABRAMSON_V_MIN_KN

KN = 0.514444
V_EXP_KN, F_EXP_HZ = 48.1, 17.5


def load_raw(path):
    by_v = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            v = float(r["V_m_s"])
            by_v[v].append((float(r["real_1_s"]), float(r["freq_Hz"]), float(r["participation"])))
    return by_v


def diagnose(by_v):
    """First zero-crossing over ALL modes vs over participation-filtered modes."""
    vs = sorted(by_v)
    print("\n=== FLUTTER DIAGNOSIS ===")

    def first_cross(pmin):
        for v in vs:
            hits = [(re, fz, pa) for (re, fz, pa) in by_v[v] if pa >= pmin and re >= 0.0]
            if hits:
                re, fz, pa = max(hits, key=lambda t: t[0])
                return v, fz, pa
        return None

    for pmin in (0.0, 0.1, 0.3, 0.5, 0.7):
        r = first_cross(pmin)
        if r:
            v, fz, pa = r
            print(f"  participation >= {pmin:.1f}: first crossing at "
                  f"{v/KN:6.2f} kn ({v:5.2f} m/s), f={fz:5.2f} Hz, participation={pa:.3f}")
        else:
            print(f"  participation >= {pmin:.1f}: no crossing")
    print(f"  experiment: {V_EXP_KN} kn @ {F_EXP_HZ} Hz")


def track_bands(by_v, bands, v_min_kn=ABRAMSON_V_MIN_KN, part_floor=0.004):
    """Track one branch per frequency band by picking the max-participation mode
    in that band at each velocity. Robust because bending and torsion stay well
    separated in frequency (they do NOT coalesce in dense water).

    v_min_kn: skip the low-speed region below the first stabilized sweep point
    (default 3.5 m/s ≈ 6.8 kn) where wake modes pollute the bands.
    part_floor: require a minimum structural participation so a pure wake mode is
    never selected as a structural branch.
    """
    series = {name: [] for _, _, name in bands}
    for v in sorted(by_v):
        if v / KN < v_min_kn:
            continue
        for lo, hi, name in bands:
            cand = [(re, fz, pa) for (re, fz, pa) in by_v[v] if lo <= fz < hi and pa >= part_floor]
            if cand:
                re, fz, pa = max(cand, key=lambda t: t[2])  # most-structural in band
                series[name].append((v, re, fz, pa))
    return series


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--bend-lo", type=float, default=4.0)
    ap.add_argument("--bend-hi", type=float, default=13.0)
    ap.add_argument("--tors-lo", type=float, default=13.0)
    ap.add_argument("--tors-hi", type=float, default=24.0)
    ap.add_argument("--v-min-kn", type=float, default=ABRAMSON_V_MIN_KN,
                    help="first plotted velocity [kn] (default 3.5 m/s ≈ 6.8 kn)")
    ap.add_argument("--title", type=str, default="Abramson SHARPy UVLM water")
    ap.add_argument("--v-max-kn", type=float, default=55.0, help="plot x-axis max [kn]")
    args = ap.parse_args()

    by_v = load_raw(args.raw)
    diagnose(by_v)
    bands = [(args.bend_lo, args.bend_hi, "bending"),
             (args.tors_lo, args.tors_hi, "torsion")]
    series = track_bands(by_v, bands, v_min_kn=args.v_min_kn)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = {"bending": "#1f77b4", "torsion": "#ff7f0e"}
    fig, (axg, axf) = plt.subplots(1, 2, figsize=(13, 5))
    vfl = None
    for name, pts in series.items():
        if not pts:
            continue
        u = np.array([p[0] for p in pts])
        re = np.array([p[1] for p in pts])
        fz = np.array([p[2] for p in pts])
        axg.plot(u / KN, re, "-", color=cols[name], lw=2, label=name)
        axf.plot(u / KN, fz, "-", color=cols[name], lw=2, label=name)
        cx = np.where(np.diff(np.sign(re)) > 0)[0]
        if len(cx):
            i = cx[0]
            a = re[i] / (re[i] - re[i + 1])
            vc = (u[i] + a * (u[i + 1] - u[i])) / KN
            fc = fz[i] + a * (fz[i + 1] - fz[i])
            if vfl is None or vc < vfl[0]:
                vfl = (vc, fc, name)

    axg.axhline(0, color="k", lw=0.8)
    axg.axvline(V_EXP_KN, color="green", ls=":", lw=1.6, label=f"exp {V_EXP_KN} kn")
    if vfl:
        axg.axvline(vfl[0], color="red", ls="--", lw=1.3, label=f"flutter {vfl[0]:.1f} kn ({vfl[2]})")
    axg.set_xlabel("V [kn]"); axg.set_ylabel("damping: Re(λ) [1/s]")
    axg.set_title("V-g  (bending dives = heavily hydro-damped)")
    axg.set_xlim(args.v_min_kn, args.v_max_kn); axg.set_ylim(-12, 8)
    axg.grid(alpha=0.3); axg.legend(fontsize=8)

    axf.plot(V_EXP_KN, F_EXP_HZ, "*", color="green", ms=15, label=f"exp {F_EXP_HZ} Hz")
    axf.axhline(20.14, color="gray", ls=":", lw=1, label="dry torsion 20.1 Hz")
    axf.axhline(9.93, color="lightgray", ls=":", lw=1, label="dry bending 9.9 Hz")
    axf.set_xlabel("V [kn]"); axf.set_ylabel("frequency [Hz]")
    axf.set_title("V-f"); axf.set_xlim(args.v_min_kn, args.v_max_kn); axf.set_ylim(0, 25)
    axf.grid(alpha=0.3); axf.legend(fontsize=8)
    fig.suptitle(args.title)
    fig.tight_layout()
    out = args.out or args.raw.with_name(args.raw.stem.replace("raw_modes", "tracked") + ".png")
    fig.savefig(out, dpi=140)

    # Clean tracked branches CSV alongside the plot
    csv_out = out.with_suffix(".csv")
    with open(csv_out, "w", newline="") as f:
        wri = csv.writer(f)
        wri.writerow(["branch", "V_m_s", "V_kn", "real_1_s", "freq_Hz", "participation"])
        for name, pts in series.items():
            for u, re, fz, pa in pts:
                wri.writerow([name, u, u / KN, re, fz, pa])
    print("\nsaved plot:", out)
    print("saved tracked CSV:", csv_out)
    if vfl:
        print(f"flutter (first branch crossing): {vfl[0]:.2f} kn @ {vfl[1]:.2f} Hz ({vfl[2]})")


if __name__ == "__main__":
    main()
