#!/usr/bin/env python3
"""
Filter-free loci of ALL physical roots of the Model2t hybrid sweep.

Seeds one MAC chain per physical root at the first velocity (participation
> 1e-3: 1B, 1T, 2B, aliased m4) and follows each across the saved spectra with
best-MAC over the FULL eigenvalue set — no frequency window, no participation
floor. If the loci are smooth, nothing "disappears": roots only leave the
observation filters of the tracker/log.

Outputs (next to the NPZ): physical_root_loci.png, physical_root_loci.csv
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

_FSI_ROOT = Path(__file__).resolve().parents[3]
OUTDIR = (_FSI_ROOT / "output_data" / "Model2t" / "00_baseline_sensitivity"
          / "_baseline_sharpy_removegammadot_plusbem")
if len(sys.argv) > 1:                       # optional: sweep outdir to trace
    OUTDIR = Path(sys.argv[1])
KN = 0.514444
NAMES = ("1B", "1T", "2B", "m4")


def main() -> int:
    npz = np.load(next(OUTDIR.glob("model2t_sharpy_eigvecs_nm*.npz")))
    V = np.asarray(npz["velocities"], float)
    lam = np.asarray(npz["lam"])
    vecs = np.asarray(npz["struct_vecs"])
    part = np.asarray(npz["participation"])
    nm = vecs.shape[1] // 2

    freq0 = lam[0].imag / (2 * np.pi)
    seeds = np.flatnonzero((lam[0].imag > 1e-6) & (part[0] > 1e-3)
                           & (freq0 > 0.3) & (freq0 < 35.0))
    seeds = sorted(seeds, key=lambda k: lam[0][k].imag)
    chains = []
    for k in seeds:
        q = np.abs(vecs[0][:nm, k])
        chains.append(dict(name=NAMES[int(np.argmax(q))], ref=vecs[0][:, k].copy(),
                           rows=[]))
    print("chains seeded at V=%.1f kn: %s"
          % (V[0] / KN, [(c["name"], f"{lam[0][seeds[i]].imag/2/np.pi:.2f} Hz")
                         for i, c in enumerate(chains)]))

    for iv in range(len(V)):
        vb = vecs[iv]
        norms = np.maximum(np.einsum("ij,ij->j", np.conj(vb), vb).real, 1e-300)
        for c in chains:
            macs = np.abs(np.conj(c["ref"]) @ vb) ** 2 / (
                (np.conj(c["ref"]) @ c["ref"]).real * norms)
            macs = np.where(lam[iv].imag >= -1e-9, macs.real, -1.0)
            k = int(np.argmax(macs))
            l = lam[iv][k]
            c["rows"].append(dict(
                V_kn=V[iv] / KN, f_Hz=l.imag / (2 * np.pi), sigma=l.real,
                g=l.real / max(abs(l.imag), 1e-30), part=part[iv][k],
                mac=float(macs[k])))
            if macs[k] > 0.4:
                c["ref"] = vb[:, k].copy()

    csv_path = OUTDIR / "physical_root_loci.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["branch", "V_kn", "f_Hz", "sigma", "g", "part", "mac"])
        for c in chains:
            for r in c["rows"]:
                w.writerow([c["name"], f"{r['V_kn']:.3f}", f"{r['f_Hz']:.5f}",
                            f"{r['sigma']:.5f}", f"{r['g']:.6f}",
                            f"{r['part']:.3e}", f"{r['mac']:.4f}"])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    col = {"1B": "#2a78d6", "1T": "#eda100", "2B": "#1baf7a", "m4": "#c33d69"}
    fig, (axf, axg, axp) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    for c in chains:
        v = [r["V_kn"] for r in c["rows"]]
        axf.plot(v, [r["f_Hz"] for r in c["rows"]], "-", color=col[c["name"]],
                 lw=1.8, label=c["name"])
        axg.plot(v, [r["g"] for r in c["rows"]], "-", color=col[c["name"]], lw=1.8)
        axp.semilogy(v, [max(r["part"], 1e-7) for r in c["rows"]], "-",
                     color=col[c["name"]], lw=1.8)
    axg.axhline(0, color="k", lw=0.8)
    axp.axhline(1e-3, color="k", ls=":", lw=1, label="tracker floor 1e-3")
    axp.axhline(2e-4, color="gray", ls=":", lw=1, label="tracker floor 2e-4")
    axf.set_ylabel("frequency [Hz]")
    axg.set_ylabel("g = σ/ω [–]")
    axp.set_ylabel("beam participation [–]")
    axp.set_xlabel("V [kn]")
    axf.legend(fontsize=9, ncol=4)
    axp.legend(fontsize=8)
    for ax in (axf, axg, axp):
        ax.grid(alpha=0.3)
    fig.suptitle("Model2t hybrid — filter-free MAC loci of every physical root\n"
                 "(all curves continuous: 'disappearances' are only the "
                 "participation floors, bottom panel)")
    fig.tight_layout()
    fig.savefig(OUTDIR / "physical_root_loci.png", dpi=150)
    print(f"wrote {OUTDIR / 'physical_root_loci.png'}\n      {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
