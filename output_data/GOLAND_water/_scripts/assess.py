#!/usr/bin/env python3
"""Assess a GOLAND_water p-k result on the two criteria that matter:

  1. FLUTTER  -- the TORSION branch must cross g = 0 (positive damping).
  2. SMOOTH   -- no branch jumps: frequency must vary continuously with V,
                 and every point in the swept range must converge.

A frequency that leaps between consecutive velocity points means the tracker
swapped roots; any flutter speed read off such a curve is meaningless.
"""
import csv, os, sys, collections
import numpy as np


def assess(d, quiet=False):
    r = {"dir": os.path.basename(d)}
    vgp, vfp = d + "/vg_data.csv", d + "/vf_data.csv"
    if not (os.path.exists(vgp) and os.path.exists(vfp)):
        r["error"] = "no vg/vf data"
        return r
    vg = list(csv.DictReader(open(vgp)))
    vf = list(csv.DictReader(open(vfp)))
    V = np.array([float(x["Velocity_m_s"]) for x in vg])
    r["n"] = len(V)
    r["V_range"] = (float(V.min()), float(V.max()))

    def col(rows, name):
        out = []
        for x in rows:
            try:
                out.append(float(x[name]))
            except (ValueError, KeyError, TypeError):
                out.append(np.nan)
        return np.array(out)

    for br in (1, 2):
        g = col(vg, f"Damping_Branch_{br}")
        sig = col(vg, f"Sigma_rad_s_Branch_{br}")
        f = col(vf, f"Frequency_Hz_Branch_{br}")
        conv = np.array([x.get(f"Converged_Branch_{br}") == "1" for x in vg])
        ok = conv & np.isfinite(f) & np.isfinite(g)
        r[f"b{br}_conv_frac"] = float(ok.mean())

        # smoothness: largest relative frequency step between consecutive CONVERGED pts
        jumps, worstV = [], None
        for i in range(len(V) - 1):
            if ok[i] and ok[i + 1] and f[i] > 1e-9:
                rel = abs(f[i + 1] - f[i]) / max(f[i], 1e-9)
                jumps.append(rel)
                if worstV is None or rel > max(jumps[:-1] or [0]):
                    worstV = V[i]
        r[f"b{br}_max_jump"] = float(max(jumps)) if jumps else float("nan")
        r[f"b{br}_jump_at"] = float(worstV) if worstV is not None else float("nan")
        r[f"b{br}_njump_gt20pct"] = int(sum(1 for j in jumps if j > 0.20))

        # g = 0 crossing on converged points only
        cross = []
        for i in range(len(V) - 1):
            if ok[i] and ok[i + 1] and g[i] < 0 <= g[i + 1]:
                t = -g[i] / (g[i + 1] - g[i])
                cross.append(float(V[i] + t * (V[i + 1] - V[i])))
        r[f"b{br}_g0"] = cross
        r[f"b{br}_gmax"] = float(np.nanmax(np.where(ok, g, -np.inf))) if ok.any() else float("nan")
        r[f"b{br}_sigmax"] = float(np.nanmax(np.where(ok, sig, -np.inf))) if ok.any() else float("nan")

    # which slot the solver itself flagged unstable
    p = d + "/real_instability_events.csv"
    if os.path.exists(p):
        rows = list(csv.DictReader(open(p)))
        c = collections.Counter(int(x["mode_slot"]) for x in rows)
        first = {}
        for x in rows:
            s = int(x["mode_slot"]); v = float(x["Velocity_m_s"])
            if s not in first or v < first[s]:
                first[s] = v
        r["events"] = {s: (c[s], first[s]) for s in sorted(c)}
    else:
        r["events"] = {}

    # static divergence
    p = d + "/divergence_indicator.csv"
    r["V_div"] = None
    if os.path.exists(p):
        x = np.genfromtxt(p, delimiter=",", names=True)
        Vd, det = x["V_mps"], x["det_Keff_over_detK0"]
        idx = np.where(np.diff(np.sign(det)) != 0)[0]
        if len(idx):
            i = idx[0]
            r["V_div"] = float(Vd[i] - det[i] * (Vd[i + 1] - Vd[i]) / (det[i + 1] - det[i]))
    return r


def show(r):
    kn = 1 / 0.514444
    print(f"  {r['dir']}")
    if "error" in r:
        print(f"     {r['error']}")
        return
    dv = f"{r['V_div']*kn:.1f} kn" if r["V_div"] else "none"
    print(f"     V {r['V_range'][0]:.0f}-{r['V_range'][1]:.0f} m/s, {r['n']} pts   divergence: {dv}")
    for br, nm in ((1, "bend"), (2, "TORS")):
        print(f"     br{br} {nm}: conv {r[f'b{br}_conv_frac']*100:5.1f}%  "
              f"max Δf {r[f'b{br}_max_jump']*100:6.1f}%  "
              f"jumps>20% {r[f'b{br}_njump_gt20pct']:3d}  "
              f"g_max {r[f'b{br}_gmax']:+.4f}  "
              f"g=0 at {[f'{v*kn:.1f}kn' for v in r[f'b{br}_g0']] or '-'}")
    if r["events"]:
        for s, (n, v) in r["events"].items():
            print(f"     solver events slot {s} ({'TORSION' if s==1 else 'bending'}): "
                  f"{n} from {v*kn:.1f} kn")


if __name__ == "__main__":
    import glob
    pats = sys.argv[1:] or ["output_data/GOLAND_water/GOLAND_water_c0p3048*"]
    ds = []
    for p in pats:
        ds += sorted(glob.glob(p))
    for d in ds:
        if os.path.isdir(d):
            show(assess(d))
            print()
