"""Test O su E3: V_f con la parte circolatoria scalata di f(h/c), stesso inviluppo (20-40 Hz) di figure_revE.py."""
import importlib.util, json, glob
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sp = importlib.util.spec_from_file_location("fe", HERE.parent / "report" / "revE" / "figure_revE.py")
fe = importlib.util.module_from_spec(sp); sp.loader.exec_module(fe)
BAND = (20, 40)
F = {r["hc"]: r for r in json.load(open(HERE / "fattore_immagine.json"))}
runs = fe.runs_of(1)
vu = fe.crossing(*fe.envelope(runs["uvlm"], BAND))[0]
out = {"uvlm": vu, "rows": []}
print("UVLM-only %.2f kn" % vu)
print(" h/c   f2D    f3D   | V_f BEM  | V_f +img2D      | V_f +img3D      | Fn_h")
for hc in [2.0, 1.0, 0.5, 0.4]:
    ht = ("%.1f" % hc).replace(".", "p")
    v0, f0 = fe.crossing(*fe.envelope(runs[hc], BAND))
    v2, w2 = fe.crossing(*fe.envelope(HERE / f"scan_hc{ht}_2d.txt", BAND))
    v3, w3 = fe.crossing(*fe.envelope(HERE / f"scan_hc{ht}_3d.txt", BAND))
    fnh = v0 * 0.5144 / np.sqrt(9.81 * hc * 0.294)
    out["rows"].append(dict(hc=hc, f2d=F[hc]["f2d"], f3d=F[hc]["f3d"], v0=v0, f0=f0, v2=v2, w2=w2, v3=v3, w3=w3, fnh=fnh))
    print("%4.1f  %.3f  %.3f | %.2f    | %.2f (%+.1f %%)  | %.2f (%+.1f %%)  | %.1f" % (hc, F[hc]["f2d"], F[hc]["f3d"], v0, v2, 100 * (v2 / vu - 1), v3, 100 * (v3 / vu - 1), fnh))
vi = fe.crossing(*fe.envelope(HERE / "scan_hc2p0_identita.txt", BAND))[0]
print("identita' h/c 2: %.3f kn (rev E %.3f)" % (vi, out["rows"][0]["v0"]))
json.dump(out, open(HERE / "risultati_testO.json", "w"), indent=1)

fig, ax = plt.subplots(figsize=(10, 4.6), dpi=200)
ax.axhline(0, color="#6B6B6B", lw=1)
for hc, col in [(2.0, "#2E6FB7"), (0.4, "#8E6BB8")]:
    ht = ("%.1f" % hc).replace(".", "p")
    for src, ls, lab in [(runs[hc], "-", "UVLM+BEM"), (HERE / f"scan_hc{ht}_2d.txt", "--", "+ immagine 2D"), (HERE / f"scan_hc{ht}_3d.txt", ":", "+ immagine 3D")]:
        V, s, f = fe.envelope(src, BAND)
        ax.plot(V, s / (2 * np.pi * f), ls, color=col, lw=2, label=f"h/c {hc:g}: {lab}")
ax.set_xlim(38, 50); ax.set_xlabel("V [kn]"); ax.set_ylabel("σ / ω del ramo di flutter  [−]")
ax.grid(True, color="#E3E3E3", lw=0.8); ax.legend(frameon=False, loc="upper left", fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(HERE / "testO_sigma.png")
