#!/usr/bin/env python3
"""Merge the two validated GOLAND_water results into one figure:
   (a) A(omega; h) from Capytaine  -- the added mass really does move with depth
   (b) V_flutter(h/c)              -- but the flutter speed barely follows
   (c) why: k_w*h of each, on the same axis
"""
import glob, json, os, csv
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "/media/lorenzo/Seagate Por1/recupero dati Linux/lorebasket/FSI"
OUT  = f"{REPO}/output_data/GOLAND_water"
PROBE_C = 0.294          # chord of the BEM probe
kn = 1/0.514444

# ---- (a) added mass vs depth, from the probe -------------------------------
probe = {}
for d in sorted(glob.glob(f"{OUT}/bem_depth_probe/hc*")):
    hc = float(os.path.basename(d)[2:].replace("p", "."))
    npz = sorted(glob.glob(d+"/capytaine/modal_radiation_AB*.npz"))
    if not npz: continue
    z = np.load(npz[0], allow_pickle=True)
    probe[hc] = (np.asarray(z["omega"], float), np.asarray(z["added_mass"], float))
hcs = sorted(probe)

# ---- (b) the clean flutter depth sweep -------------------------------------
sw = f"{OUT}/sweeps/GOLAND_water_c0p1_s1p5_xea015_xcm075_ei1p5x_depth_sweep"
fl = []
for d in sorted(glob.glob(sw+"/hc*"), key=lambda x: float(os.path.basename(x)[2:].replace("p","."))):
    hc = float(os.path.basename(d)[2:].replace("p", "."))
    vg = list(csv.DictReader(open(d+"/vg_data.csv", encoding="utf-8", errors="replace")))
    vf = list(csv.DictReader(open(d+"/vf_data.csv", encoding="utf-8", errors="replace")))
    V  = np.array([float(x["Velocity_m_s"]) for x in vg])
    def c_(rows, n):
        o=[]
        for x in rows:
            try: o.append(float(x[n]))
            except Exception: o.append(np.nan)
        return np.array(o)
    g = c_(vg, "Damping_Branch_1"); f = c_(vf, "Frequency_Hz_Branch_1")
    ok = np.array([x.get("Converged_Branch_1")=="1" for x in vg]) & np.isfinite(g)
    for i in range(len(V)-1):
        if ok[i] and ok[i+1] and g[i] < 0 <= g[i+1]:
            t = -g[i]/(g[i+1]-g[i]); fl.append((hc, (V[i]+t*(V[i+1]-V[i]))*kn, f[i])); break
fl = np.array(fl)

fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.9))
fig.suptitle("GOLAND_water — the added mass moves with submergence; the flutter speed does not follow",
             fontsize=13, fontweight="bold")

# (a)
fref = np.array([0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 5.0, 8.0])
w = 2*np.pi*fref
deep = hcs[-1]
for hc in hcs:
    ww, A = probe[hc]
    a  = np.interp(w, ww, A[:, 0, 0])
    ad = np.interp(w, ww, probe[deep][1][:, 0, 0])
    ax[0].plot(fref, 100*(a-ad)/ad, "o-", label=f"h/c = {hc:g}")
ax[0].axhline(0, color="k", lw=.8)
ax[0].axvline(1.1, color="crimson", ls="--", lw=1.2)
ax[0].annotate("sign flip\n≈1.1 Hz", (1.1, 55), color="crimson", fontsize=9, ha="left")
ax[0].annotate("rigid-wall image\n(A increases)", (0.55, 70), fontsize=8, color="#444")
ax[0].annotate("pressure-release\n(A decreases)", (2.6, -42), fontsize=8, color="#444")
ax[0].set_xscale("log"); ax[0].set_xlabel("frequency [Hz]")
ax[0].set_ylabel(r"$\Delta A_{00}$ vs deep water  [%]")
ax[0].set_title("(a) added mass IS strongly depth-dependent", fontsize=10)
ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)

# (b)
if len(fl):
    ax[1].plot(fl[:,0], fl[:,1], "s-", color="darkgreen", ms=7)
    for hc, v, _ in fl:
        ax[1].annotate(f"{v:.2f}", (hc, v), textcoords="offset points",
                       xytext=(0,7), fontsize=8, ha="center")
    sp = 100*(fl[:,1].max()-fl[:,1].min())/fl[:,1].min()
    ax[1].set_title(f"(b) but $V_{{flutter}}$ moves only {sp:.1f}%", fontsize=10)
ax[1].set_xscale("log"); ax[1].set_xlabel("h/c"); ax[1].set_ylabel("$V_{flutter}$ [kn]")
ax[1].grid(alpha=.3)

# (c)
hgrid = np.logspace(-1, 0.7, 60)
for f0, col in ((0.75, "tab:blue"), (1.25, "tab:orange"), (2.5, "tab:green"), (10.6, "crimson")):
    kwh = (2*np.pi*f0)**2/9.81 * hgrid*PROBE_C
    ax[2].loglog(hgrid, kwh, color=col, label=f"{f0} Hz")
ax[2].axhspan(0.3, 3, color="gold", alpha=.25)
ax[2].annotate("$k_w h \\sim 1$:\ndepth matters", (0.11, 1.0), fontsize=9)
ax[2].axhline(1, color="k", lw=.8, ls=":")
ax[2].set_xlabel("h/c"); ax[2].set_ylabel("$k_w h = \\omega^2 h/g$")
ax[2].set_title("(c) why: the flutter case sits at 10.6 Hz", fontsize=10)
ax[2].legend(fontsize=8, title="flutter freq."); ax[2].grid(alpha=.3, which="both")

fig.tight_layout(rect=[0,0,1,0.94])
p = f"{OUT}/bem_depth_vs_flutter_summary.png"
fig.savefig(p, dpi=150)
print("saved", p)
if len(fl):
    print("\nV_flutter(h/c):")
    for hc, v, ff in fl: print(f"   h/c={hc:5.2f}   {v:6.2f} kn   f={ff:.2f} Hz")
