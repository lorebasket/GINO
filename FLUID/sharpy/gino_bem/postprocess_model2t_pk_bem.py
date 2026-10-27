#!/usr/bin/env python3
"""
Post-process the Model2t hybrid p-k sweep spectrum NPZ: per-velocity root pool,
MAC continuity audit, new-root detection, and root-pool plots.

Reads ``model2t_sharpy_eigvecs_nm4.npz`` (full eigenvalues + beam-state
eigenvector rows + participation saved at every velocity) and the tracked CSV,
then for each velocity step prints a STEP E/F-style report:

  - top-N candidate roots (freq, g, participation, |q| modal content),
  - MAC match of each root pool entry against the previous velocity's pool,
  - which pool roots the tracked branches selected,
  - roots with no MAC parent at the previous velocity -> "NEW mode" (saved).

Outputs (next to the NPZ):
  root_pool_all.csv          every pool root at every velocity
  new_roots.csv              roots that appeared without a MAC parent
  root_pool_freq_damping.png pool scatter (colored by dominant q-DOF) + branches
  flutter_branch_content.png |q| content of the tracked flutter branch vs V

Usage:
  conda activate sharpy
  python gino_bem/postprocess_model2t_pk_bem.py [--outdir DIR] [--n-pool 10]
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

_FSI_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTDIR = (_FSI_ROOT / "output_data" / "Model2t" / "00_baseline_sensitivity"
                  / "_baseline_sharpy_removegammadot_plusbem")
KN = 0.514444
MODE_NAMES = ("1B", "1T", "2B", "m4")
FREQ_WINDOW_HZ = (0.3, 35.0)


def mac(a: np.ndarray, b: np.ndarray) -> float:
    num = abs(np.vdot(a, b)) ** 2
    den = (np.vdot(a, a).real * np.vdot(b, b).real)
    return float(num / max(den, 1e-300))


def load_tracked(csv_path: Path) -> dict[str, dict[float, tuple[float, float]]]:
    """branch -> {V_m_s: (freq_Hz, g)}"""
    out: dict[str, dict[float, tuple[float, float]]] = {}
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            out.setdefault(row["branch"], {})[round(float(row["V_m_s"]), 6)] = (
                float(row["freq_Hz"]), float(row["g"]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    ap.add_argument("--n-pool", type=int, default=10)
    ap.add_argument("--part-floor", type=float, default=3e-4)
    ap.add_argument("--mac-parent-floor", type=float, default=0.5,
                    help="min MAC to the previous-velocity pool to count as the same root")
    ap.add_argument("--quiet", action="store_true", help="skip per-velocity prints")
    args = ap.parse_args()

    npz_path = next(args.outdir.glob("model2t_sharpy_eigvecs_nm*.npz"))
    d = np.load(npz_path)
    V_all = np.asarray(d["velocities"], float)          # (n_v,)
    lam_all = np.asarray(d["lam"])                      # (n_v, n_eig)
    vecs_all = np.asarray(d["struct_vecs"])             # (n_v, n_beam, n_eig)
    part_all = np.asarray(d["participation"])           # (n_v, n_eig)
    nm = vecs_all.shape[1] // 2
    print(f"spectrum NPZ: {npz_path.name}  ({len(V_all)} velocities, "
          f"{lam_all.shape[1]} eigenvalues, {nm} modes)")

    tracked = load_tracked(next(args.outdir.glob("model2t_pk_bem_tracked_nm*.csv")))

    pool_rows = []
    new_rows = []
    prev_pool: list[dict] = []
    for iv, V in enumerate(V_all):
        lam = lam_all[iv]
        vecs = vecs_all[iv]
        part = part_all[iv]
        freq = lam.imag / (2 * np.pi)
        cand = np.flatnonzero((lam.imag > 1e-6)
                              & (freq >= FREQ_WINDOW_HZ[0]) & (freq <= FREQ_WINDOW_HZ[1])
                              & (part >= args.part_floor))
        pool_idx = cand[np.argsort(-part[cand])][: args.n_pool]
        pool_idx = pool_idx[np.argsort(lam[pool_idx].imag)]

        pool = []
        for k in pool_idx:
            q = np.abs(vecs[:nm, k])
            q = q / max(q.max(), 1e-300)
            g = lam[k].real / max(abs(lam[k].imag), 1e-300)
            # MAC parent in the previous velocity pool
            best_mac, parent = 0.0, None
            for p in prev_pool:
                m = mac(p["vec"], vecs[:, k])
                if m > best_mac:
                    best_mac, parent = m, p
            entry = dict(V=V, k=int(k), f_hz=float(freq[k]), g=float(g),
                         part=float(part[k]), q=q, vec=vecs[:, k].copy(),
                         mac_parent=best_mac,
                         parent_f=None if parent is None else parent["f_hz"])
            pool.append(entry)
            pool_rows.append(dict(
                V_m_s=float(V), V_kn=float(V) / KN, f_Hz=entry["f_hz"], g=entry["g"],
                part=entry["part"],
                q_1B=float(q[0]), q_1T=float(q[1]),
                q_2B=float(q[2]) if nm > 2 else 0.0,
                q_m4=float(q[3]) if nm > 3 else 0.0,
                dominant=MODE_NAMES[int(np.argmax(q))],
                mac_parent=best_mac,
                parent_f_Hz=entry["parent_f"] if entry["parent_f"] is not None else np.nan,
            ))
            if iv > 0 and best_mac < args.mac_parent_floor:
                new_rows.append(pool_rows[-1])

        if not args.quiet:
            print(f"\n--- VELOCITY STEP {iv + 1}/{len(V_all)}: V = {V:.2f} m/s "
                  f"({V / KN:.1f} kn) ---")
            print(f"  STEP E: root pool ({len(pool)} of top-{args.n_pool}, "
                  f"part >= {args.part_floor:g}, {FREQ_WINDOW_HZ[0]}-{FREQ_WINDOW_HZ[1]} Hz)")
            for e in pool:
                qs = " ".join(f"{MODE_NAMES[m]}={e['q'][m]:.2f}" for m in range(nm))
                print(f"    f={e['f_hz']:7.3f} Hz  g={e['g']:+8.4f}  part={e['part']:.5f}  "
                      f"[{qs}]")
            print("  STEP F: MAC continuity vs previous velocity pool")
            for e in pool:
                if iv == 0:
                    print(f"    f={e['f_hz']:7.3f} Hz : first velocity (no parent)")
                elif e["mac_parent"] >= args.mac_parent_floor:
                    df = (0.0 if not e["parent_f"]
                          else 100 * (e["f_hz"] - e["parent_f"]) / max(e["parent_f"], 1e-9))
                    print(f"    ✓ f={e['f_hz']:7.3f} Hz : MAC={e['mac_parent']:.3f}, "
                          f"f_prev={e['parent_f']:.3f}→{e['f_hz']:.3f} Hz ({df:+.1f}%)")
                else:
                    print(f"    [INFO] NEW root: f={e['f_hz']:7.3f} Hz, "
                          f"best_MAC={e['mac_parent']:.3f} (no parent) — saved")
            for name, pts in tracked.items():
                hit = pts.get(round(float(V), 6))
                if hit is not None:
                    fb, gb = hit
                    near = min(pool, key=lambda e: abs(e["f_hz"] - fb)) if pool else None
                    dnote = (f" -> pool root f={near['f_hz']:.3f} Hz "
                             f"[{MODE_NAMES[int(np.argmax(near['q']))]}-dominant]"
                             if near and abs(near["f_hz"] - fb) < 0.15 * max(fb, 1e-9) else "")
                    print(f"  tracked {name:>8s}: f={fb:7.3f} Hz  g={gb:+8.4f}{dnote}")
        prev_pool = pool

    # ------------------------------------------------------------------ CSVs
    for fn, rows in (("root_pool_all.csv", pool_rows), ("new_roots.csv", new_rows)):
        p = args.outdir / fn
        if rows:
            with p.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
        print(f"{fn}: {len(rows)} rows -> {p}")

    # ------------------------------------------------------------------ plots
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dom_col = {"1B": "#2a78d6", "1T": "#eda100", "2B": "#1baf7a", "m4": "#c33d69"}
    fig, (axf, axg) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for r in pool_rows:
        c = dom_col[r["dominant"]]
        axf.plot(r["V_kn"], r["f_Hz"], ".", ms=4, color=c, alpha=0.6)
        axg.plot(r["V_kn"], r["g"], ".", ms=4, color=c, alpha=0.6)
    for name, pts in tracked.items():
        vv = np.array(sorted(pts))
        ff = np.array([pts[v][0] for v in vv])
        gg = np.array([pts[v][1] for v in vv])
        ls = "-" if name == "bending" else "--"
        axf.plot(vv / KN, ff, ls, color="k", lw=1.4, label=f"tracked {name}")
        axg.plot(vv / KN, gg, ls, color="k", lw=1.4)
    for m, c in dom_col.items():
        axf.plot([], [], ".", color=c, label=f"{m}-dominant root")
    axg.axhline(0, color="k", lw=0.8)
    axf.set_ylabel("frequency [Hz]")
    axg.set_ylabel("g = σ/ω [–]")
    axg.set_xlabel("V [kn]")
    axf.legend(fontsize=8, ncol=2)
    for ax in (axf, axg):
        ax.grid(alpha=0.3)
    fig.suptitle("Model2t hybrid p-k — full root pool (color = dominant modal DOF) "
                 "vs tracked branches")
    fig.tight_layout()
    fig.savefig(args.outdir / "root_pool_freq_damping.png", dpi=150)
    plt.close(fig)

    # flutter-branch modal content vs V: MAC-match the tracked bending branch
    # root within the pool at each velocity (nearest frequency), plot |q|
    bend = tracked.get("bending", {})
    vv, q_hist = [], []
    for r in pool_rows:
        v_ms = r["V_m_s"]
        hit = bend.get(round(v_ms, 6))
        if hit is None or abs(r["f_Hz"] - hit[0]) > 0.02 * max(hit[0], 1e-9):
            continue
        vv.append(r["V_kn"])
        q_hist.append([r["q_1B"], r["q_1T"], r["q_2B"], r["q_m4"]])
    if vv:
        q_hist = np.asarray(q_hist)
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for m in range(q_hist.shape[1]):
            ax.plot(vv, q_hist[:, m], "-", color=list(dom_col.values())[m],
                    lw=1.8, label=MODE_NAMES[m])
        ax.set_xlabel("V [kn]")
        ax.set_ylabel("|q| content of tracked flutter branch (normalized)")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.suptitle("Tracked 'bending' branch — modal displacement content vs speed")
        fig.tight_layout()
        fig.savefig(args.outdir / "flutter_branch_content.png", dpi=150)
        plt.close(fig)
        print(f"flutter_branch_content.png: {len(vv)} matched points")

    n_new = len(new_rows)
    print(f"\nSummary: {n_new} NEW-root events (MAC parent < {args.mac_parent_floor}); "
          f"see new_roots.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
