#!/usr/bin/env python3
"""Convergenza della mesh BEM sulla superficie libera (regola di copertura 14/10/2026).

Stesso caso (env GOLAND_WATER_*), mesh crescenti -> A(w), B(w) sui modi.
    MESHES="12x16,24x32,36x48" python bem_mesh_convergence.py <tag_uscita>
"""
import contextlib, os, subprocess, sys
from pathlib import Path
import numpy as np
os.environ.setdefault("MPLBACKEND", "Agg")
REPO = Path(__file__).resolve().parents[3]
for p in (REPO, REPO / "FLUID" / "capytaine", REPO / "STRUCTURE"):
    sys.path.insert(0, str(p))
from examples import config as ec
from FEA import structural_model, structural_analysis
from capytaine_workflow import ensure_capytaine_matrices

OUT = REPO / "output_data" / "GOLAND_water" / "freq_coverage" / sys.argv[1]
for m in os.environ.get("MESHES", "12x16,24x32").split(","):
    ns, nc = m.split("x")
    os.environ["GOLAND_WATER_MESH_NS"], os.environ["GOLAND_WATER_MESH_NC"] = ns, nc
    d = OUT / ("mesh" + m); d.mkdir(parents=True, exist_ok=True)
    mesh = d / "mesh.vtu"
    r = subprocess.run([sys.executable, str(REPO / "FLUID/capytaine/build_capytaine_mesh.py"),
                        "--case_name", "GOLAND_water", "--out", str(mesh)],
                       env=os.environ.copy(), capture_output=True, text=True)
    if r.returncode or not mesh.is_file():
        print("mesh", m, "FALLITA", r.stderr[-500:]); continue
    cfg = ec.get_config("GOLAND_water")
    cfg.mesh_path = str(mesh); cfg.capytaine_results_dir = str(d / "capytaine")
    cfg.modal_dir = cfg.output_dir = str(d); cfg.capytaine_auto_recalc = True
    with open(d / "log.txt", "w") as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
        bm = structural_model.build(cfg)
        sr = structural_analysis.run_dry_analysis(bm, cfg)
        ensure_capytaine_matrices(cfg, sr, bm, force=True)
    print("mesh", m, "fatta", flush=True)
