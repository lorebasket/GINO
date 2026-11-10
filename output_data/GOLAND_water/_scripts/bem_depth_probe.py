#!/usr/bin/env python3
"""Measure the BEM depth effect directly: A(omega; h) from Capytaine at several
submergences, for one geometry. No UVLM, no flutter -- this isolates the quantity
that makes BEM matter.

Free-surface parameter k_w*h = omega^2*h/g:
  k_w*h >> 1 -> deep, A is depth-independent (BEM adds nothing over a constant)
  k_w*h ~ 1  -> transition, dA/dh largest
"""
import contextlib, math, os, subprocess, sys
from pathlib import Path
import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
REPO = Path("/media/lorenzo/Seagate Por1/recupero dati Linux/lorebasket/FSI")
for p in (REPO, REPO / "FLUID" / "capytaine", REPO / "STRUCTURE"):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

CHORD = float(os.environ.get("PROBE_CHORD", "0.294"))
SPAN = float(os.environ.get("PROBE_SPAN", "0.882"))
HC_LIST = [float(x) for x in os.environ.get("PROBE_HC", "0.25,0.5,1.0,2.0,4.0").split(",")]
OUT = REPO / "output_data" / "GOLAND_water" / os.environ.get("PROBE_TAG", "bem_depth_probe")
OUT.mkdir(parents=True, exist_ok=True)

base_env = {
    "GOLAND_WATER_CHORD": f"{CHORD}", "GOLAND_WATER_BEAM_LENGTH": f"{SPAN}",
    "GOLAND_WATER_XEA": os.environ.get("PROBE_XEA", "0.25"),
    "GOLAND_WATER_XCM": os.environ.get("PROBE_XCM", "0.50"),
    "GOLAND_WATER_RHO_S": os.environ.get("PROBE_RHO", "1600"),
    "GOLAND_WATER_E_MOD": os.environ.get("PROBE_E", "70e9"),
    "GOLAND_WATER_NU": "0.30",
    "GOLAND_WATER_EI_FRAC": os.environ.get("PROBE_EIF", "0.0031"),
    "GOLAND_WATER_GJ_FRAC": os.environ.get("PROBE_GJF", "0.0033"),
}
os.environ.update(base_env)

from examples import config as ec
from FEA import structural_model, structural_analysis
from capytaine_workflow import ensure_capytaine_matrices

results = {}
for hc in HC_LIST:
    os.environ["GOLAND_WATER_DEPTH_CHORD"] = f"{hc}"
    d = OUT / f"hc{hc:g}".replace(".", "p")
    d.mkdir(parents=True, exist_ok=True)
    mesh = d / "mesh.vtu"
    env = os.environ.copy()
    r = subprocess.run(
        [sys.executable, str(REPO / "FLUID" / "capytaine" / "build_capytaine_mesh.py"),
         "--case_name", "GOLAND_water", "--out", str(mesh)],
        env=env, capture_output=True, text=True)
    if r.returncode != 0 or not mesh.is_file():
        print(f"  h/c={hc}: mesh FAILED"); continue
    cfg = ec.get_config("GOLAND_water")
    cfg.mesh_path = str(mesh)
    cfg.capytaine_results_dir = str(d / "capytaine")
    cfg.modal_dir = str(d); cfg.output_dir = str(d)
    cfg.capytaine_auto_recalc = True
    with open(d / "log.txt", "w") as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
        bm = structural_model.build(cfg)
        sr = structural_analysis.run_dry_analysis(bm, cfg)
        ensure_capytaine_matrices(cfg, sr, bm, force=True)
    dry = [float(x) for x in sr.dry_frequencies[:2]]
    npz = sorted((d / "capytaine").glob("modal_radiation_AB*.npz"))
    if not npz:
        print(f"  h/c={hc}: no BEM npz"); continue
    z = np.load(npz[0], allow_pickle=True)
    results[hc] = (np.asarray(z["omega"], float), np.asarray(z["added_mass"], float), dry)
    print(f"  h/c={hc:4.2f} done   dry f = {dry[0]:.3f}/{dry[1]:.3f} Hz   depth={hc*CHORD*100:.1f} cm")

if not results:
    sys.exit("no results")

print(f"\nchord={CHORD} m  span={SPAN} m  rho_s={base_env['GOLAND_WATER_RHO_S']}")
hcs = sorted(results)
w_ref = results[hcs[0]][0]
probe_f = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0]
for mode in (0, 1):
    print(f"\n=== modal added mass A[{mode},{mode}] vs depth ===")
    hdr = f"{'f [Hz]':>8s}{'k_w*h(h/c=1)':>14s}" + "".join(f"h/c={h:g}".rjust(11) for h in hcs) + f"{'spread%':>10s}"
    print(hdr)
    for f in probe_f:
        w = 2 * math.pi * f
        if w < w_ref.min() or w > w_ref.max():
            continue
        vals = []
        for hc in hcs:
            ww, A, _ = results[hc]
            vals.append(float(np.interp(w, ww, A[:, mode, mode])))
        spread = 100.0 * (max(vals) - min(vals)) / abs(np.mean(vals)) if np.mean(vals) else float("nan")
        kwh = w * w / 9.81 * CHORD
        print(f"{f:8.2f}{kwh:14.2f}" + "".join(f"{v:11.4f}" for v in vals) + f"{spread:10.1f}")
