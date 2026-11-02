#!/usr/bin/env python3
"""Flutter speed / frequency table over many p-k run folders (flutter_summary*.csv), any case,
with an optional bar chart (method matrix) or tornado chart (one-at-a-time sensitivity).

    python utl/summarize_runs.py output_data/Model2t/_method_matrix/* --exp 18.1,6.4 --bar matrix.png
    python utl/summarize_runs.py output_data/Model2t/sens/* --ref output_data/Model2t/base --tornado t.png
    python utl/summarize_runs.py ... --csv table.csv

The lowest flutter speed over the branches of each folder is reported.
"""
import argparse
import csv
import glob
import os

import numpy as np


def read_flutter(d):
    fs = sorted(glob.glob(os.path.join(d, "flutter_summary*.csv")))
    best = None
    for f in fs[:1]:
        for row in csv.DictReader(open(f)):
            try:
                kn, hz = float(row["Flutter_Speed_knots"]), float(row["Flutter_Frequency_Hz"])
            except (ValueError, KeyError, TypeError):
                continue
            if best is None or kn < best[0]:
                best = (kn, hz, int(row.get("Branch", 0) or 0))
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--exp", help="experimental V_kn,f_Hz")
    ap.add_argument("--ref", help="reference run folder (tornado / delta column)")
    ap.add_argument("--bar")
    ap.add_argument("--tornado")
    ap.add_argument("--csv")
    a = ap.parse_args()

    rows = [(os.path.basename(os.path.normpath(d)), read_flutter(d)) for d in a.dirs if os.path.isdir(d)]
    ref = read_flutter(a.ref) if a.ref else None
    exp = tuple(float(x) for x in a.exp.split(",")) if a.exp else None
    base = ref[0] if ref else (exp[0] if exp else None)
    print(f"{'run':45s} {'Vf [kn]':>9s} {'f [Hz]':>8s} {'branch':>6s} {'dV %':>7s}")
    for n, r in rows:
        dv = "%+.1f" % (100 * (r[0] - base) / base) if r and base else ""
        print(f"{n:45s} {('%.2f' % r[0]) if r else 'none':>9s} {('%.2f' % r[1]) if r else '':>8s} "
              f"{r[2] if r else '':>6} {dv:>7s}")
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["run", "Flutter_Speed_knots", "Flutter_Frequency_Hz", "Branch"])
            for n, r in rows:
                w.writerow([n, *(r if r else ("", "", ""))])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ok = [(n, r) for n, r in rows if r]
    if a.bar and ok:
        fig, ax = plt.subplots(figsize=(max(6, 0.8 * len(ok)), 4.5))
        ax.bar(range(len(ok)), [r[0] for _, r in ok], color="#4C78A8")
        for i, (_, r) in enumerate(ok):
            ax.text(i, r[0], "%.1f kn\n%.1f Hz" % r[:2], ha="center", va="bottom", fontsize=7)
        if exp:
            ax.axhline(exp[0], color="k", ls="--", lw=1, label="experiment %.1f kn" % exp[0])
            ax.legend(fontsize=8)
        ax.set_xticks(range(len(ok)))
        ax.set_xticklabels([n for n, _ in ok], rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("V_f [kn]")
        fig.tight_layout()
        fig.savefig(a.bar, dpi=150)
        print(a.bar)
    if a.tornado and ok and base:
        d = sorted(((n, 100 * (r[0] - base) / base) for n, r in ok), key=lambda t: abs(t[1]))
        fig, ax = plt.subplots(figsize=(7, 0.35 * len(d) + 1.5))
        ax.barh(range(len(d)), [v for _, v in d], color=["#E45756" if v < 0 else "#4C78A8" for _, v in d])
        ax.set_yticks(range(len(d)))
        ax.set_yticklabels([n for n, _ in d], fontsize=8)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_xlabel("ΔV_f vs reference [%]")
        fig.tight_layout()
        fig.savefig(a.tornado, dpi=150)
        print(a.tornado)


if __name__ == "__main__":
    main()
