#!/usr/bin/env python3
import contextlib, io, os, sys, itertools
import numpy as np
REPO = "/media/lorenzo/Seagate Por1/recupero dati Linux/lorebasket/FSI"
for p in (REPO, REPO+"/STRUCTURE"):
    if p not in sys.path: sys.path.insert(0, p)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.update(dict(GOLAND_WATER_CHORD="0.294", GOLAND_WATER_BEAM_LENGTH="0.882",
    GOLAND_WATER_RHO_S="1600", GOLAND_WATER_E_MOD="70e9", GOLAND_WATER_NU="0.30"))
from examples import config as ec
from FEA import structural_model, structural_analysis

def run(xea, xcm, eif, gjf):
    os.environ.update({"GOLAND_WATER_XEA": f"{xea}", "GOLAND_WATER_XCM": f"{xcm}",
                       "GOLAND_WATER_EI_FRAC": f"{eif}", "GOLAND_WATER_GJ_FRAC": f"{gjf}"})
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        cfg = ec.get_config("GOLAND_water")
        bm  = structural_model.build(cfg)
        sr  = structural_analysis.run_dry_analysis(bm, cfg)
    f  = np.asarray(sr.dry_frequencies, float)
    Vf = np.asarray(sr.dry_eigenvectors_full, float)   # (ndof_full, nmodes)
    n  = Vf.shape[0] // 6
    Vn = Vf.reshape(n, 6, -1)
    tip = n - 1
    # Uz = plunge (dof 2), Rx/Ry = rotations; span axis found from which rotation
    # is largest across the whole wing for the clearly-torsional high mode.
    out = []
    for m in range(4):
        uz = abs(Vn[tip, 2, m]); rx = abs(Vn[tip, 3, m]); ry = abs(Vn[tip, 4, m])
        tw = max(rx, ry)
        out.append((f[m], uz, tw, tw*0.147/max(uz,1e-12)))   # twist*semichord / plunge
    return out

print(f"{'xea':>5}{'xcm':>6}{'e/c':>6}{'eif':>8}{'gjf':>9} | "
      f"{'f1':>7}{'t/p1':>7} {'f2':>7}{'t/p2':>7}  ratio  verdict")
best=[]
for xea, xcm in ((0.25,0.35),(0.25,0.45),(0.30,0.40),(0.35,0.45),(0.30,0.50)):
    for eif, gjf in ((0.008,0.0015),(0.030,0.0015),(0.060,0.0015),(0.030,0.0008),(0.060,0.0030)):
        try: o = run(xea,xcm,eif,gjf)
        except Exception as e: print(f"{xea:5.2f}{xcm:6.2f} FAILED {e}"); continue
        f1,_,_,tp1 = o[0]; f2,_,_,tp2 = o[1]
        r = f2/f1
        good = 1.05 <= r <= 1.7
        if good: best.append((r,xea,xcm,eif,gjf,f1,f2))
        print(f"{xea:5.2f}{xcm:6.2f}{xcm-xea:6.2f}{eif:8.3f}{gjf:9.5f} | "
              f"{f1:7.2f}{tp1:7.2f} {f2:7.2f}{tp2:7.2f}  {r:5.2f}  {'<== PAIR' if good else ''}")
print()
for b in sorted(best): print("  candidate:", b)
