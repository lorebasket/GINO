"""SHARPy con l'ala immagine (gino_bem/image_surface.py) contro NACA TR 1232, figura 15: rapporto delle pendenze
a2/a1 = L(f/c) / L(fluido infinito) di un'ala rettangolare piana rigida, corda 1, A = 4, 6, 10.

Stessa costruzione del test P (template Goland a due semiali + add_image_surfaces), ma senza trave:
flusso BeamLoader -> AerogridLoader -> StaticUvlm, scia a ferro di cavallo (infinita), incidenza data dalla
direzione della corrente (terna dell'ala non ruotata), cosi' l'immagine sta esattamente sulla verticale a +2f.

    python sharpy_naca1232.py --one A F M     # un caso, stampa una riga JSON (F = 0: senza immagine)
    python sharpy_naca1232.py                 # tutti i casi (un sottoprocesso ciascuno) -> sharpy_naca1232.json
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SHARPY = HERE.parents[3] / "FLUID" / "sharpy"
ALPHA = 0.1  # deg: a 1 deg le due scie (entrambe inclinate di alpha, non speculari) rompono la simmetria di 0.5 punti
U = 10.0
FF = [0.25, 0.35, 0.5, 0.6, 0.75, 0.85, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
AA = [4, 6, 10]
GRID = {8: 8, 16: 16}  # M -> pannelli per corda in apertura (8 = come il VLM di riproduzione_naca1232.py)


def one(A, F, M):
    sys.path.insert(0, str(SHARPY))
    from gino_bem.run_goland_water_rho import _build_case
    import sharpy.sharpy_main

    N = 2 * int(round(GRID[M] * A / 2))
    N += N % 4  # due superfici, elementi a 3 nodi
    a = np.deg2rad(ALPHA)
    udir = np.array([np.cos(a), 0.0, np.sin(a)])
    tag = f"naca_A{A}_M{M}_F{F:g}".replace(".", "p")
    route = HERE / "_sharpy"
    ws, *_ = _build_case(rho=1000.0, u_inf=U, alpha_deg=0.0, M=M, N=N, M_star_fact=1, num_modes=4,
                         v_min=U, v_max=U, n_vel=1, velocity_list=None, route=route, case_tag=tag,
                         use_krylov=False, main_chord=1.0, wing_semispan=A / 2.0)
    ws.config["SHARPy"]["flow"] = ["BeamLoader", "AerogridLoader", "StaticUvlm"]
    ws.config["AerogridLoader"]["mstar"] = 1
    ws.config["AerogridLoader"]["freestream_dir"] = udir.tolist()
    ws.config["AerogridLoader"]["wake_shape_generator_input"]["u_inf_direction"] = udir.tolist()
    st = dict(ws.config["StaticCoupled"]["aero_solver_settings"])
    st["horseshoe"] = "on"
    st["num_cores"] = 1
    st["velocity_field_input"] = {"u_inf": U, "u_inf_direction": udir.tolist()}
    ws.config["StaticUvlm"] = st
    for k in ("StaticCoupled", "Modal", "LinearAssembler", "AsymptoticStability"):
        ws.config.pop(k, None)
    ws.config.write()
    if F > 0:
        from gino_bem.image_surface import add_image_surfaces
        add_image_surfaces(ws.route + "/" + ws.case_name, F)  # h = F * corda (corda 1)
    data = sharpy.sharpy_main.main(["", ws.route + ws.case_name + ".sharpy"])
    ta = data.aero.timestep_info[-1]
    n_real = 2
    lift = []
    for i, (z, f) in enumerate(zip(ta.zeta, ta.forces)):
        lift.append(float(-f[0].sum() * np.sin(a) + f[2].sum() * np.cos(a)))
        print(f"[naca] superficie {i}: z medio {z[2].mean():+.4f}, y [{z[1].min():+.3f}, {z[1].max():+.3f}], "
              f"L {lift[-1]:+.6g} N", file=sys.stderr)
    gam = [float(g.sum()) for g in ta.gamma]
    out = dict(A=A, F=F, M=M, N=N, L=sum(lift[:n_real]), L_img=sum(lift[n_real:]) if F > 0 else 0.0,
               gamma_ratio=(sum(gam[n_real:]) / sum(gam[:n_real])) if F > 0 else None,
               z_surf=[float(z[2].mean()) for z in ta.zeta])
    print("RESULT " + json.dumps(out))


def driver():
    res_file = HERE / "sharpy_naca1232.json"
    res = json.load(open(res_file)) if res_file.exists() else []
    done = {(r["A"], r["F"], r["M"]) for r in res}
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               PYTHONPATH=str(SHARPY))
    for M in GRID:
        for A in AA:
            for F in [0.0] + FF:
                if (A, F, M) in done:
                    continue
                p = subprocess.run(["nice", "-n", "19", sys.executable, __file__, "--one", str(A), str(F), str(M)],
                                   cwd=SHARPY, env=env, capture_output=True, text=True)
                line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
                if p.returncode or not line:
                    print(p.stdout[-3000:], p.stderr[-3000:])
                    raise SystemExit(f"caso A={A} F={F} M={M} fallito (rc {p.returncode})")
                r = json.loads(line[0][7:])
                res.append(r)
                json.dump(res, open(res_file, "w"), indent=1)
                print(f"A {A:2d} M {M:2d} f/c {F:4.2f}: L {r['L']:.6g}  Gimm/Gala {r['gamma_ratio']}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--one", nargs=3, type=float)
    a = ap.parse_args()
    if a.one:
        one(int(a.one[0]), float(a.one[1]), int(a.one[2]))
    else:
        driver()
