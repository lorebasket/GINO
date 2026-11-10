#!/usr/bin/env python3
"""Self-consistent p-k scan of a GOLAND_water p-k run, rebuilt from the run's own files
(GINO dry omegas + Rayleigh alpha/beta from flutter.log, GAF table, Capytaine table of the run),
so mode signs are exactly those of the p-k run.

    python gwscan.py <run_dir> <uvlm|delta> [--vmin-kn 20 --vmax-kn 60 --dv 0.1 --n-omega 6000]
"""
import argparse, contextlib, io, os, re, sys, glob, time
from pathlib import Path
import numpy as np

REPO = Path("/media/lorenzo/Seagate Por1/recupero dati Linux/lorebasket/FSI")
sys.path.insert(0, str(REPO / "utl"))
import _common as C  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("run"); ap.add_argument("stage", choices=("uvlm", "delta"))
ap.add_argument("--vmin-kn", type=float, default=20); ap.add_argument("--vmax-kn", type=float, default=60)
ap.add_argument("--dv", type=float, default=0.1, help="m/s"); ap.add_argument("--n-omega", type=int, default=6000)
ap.add_argument("--fit", default="14,24"); ap.add_argument("--rho", type=float, default=997.0)
ap.add_argument("--out")
ap.add_argument("--gaf", help="tabella GAF da usare al posto di quella del run (test P: GAF con ala immagine)")
ap.add_argument("--circ-factor", type=float, default=1.0,
                help="test O (T1.7): scala la parte circolatoria di Q, cioe' Q - (2/rho) A_nc k^2")
a = ap.parse_args()
run = Path(a.run).resolve()
log = (run / "flutter.log").read_text(errors="ignore")
om = np.array([float(x) for x in re.findall(r"mode \d: omega=([\d.eE+-]+) rad/s", log)])
if len(om) < 4:
    # Rayleigh «exact 2-mode» (caso B) stampa solo 2 omega: le 4 frequenze secche dalla tabella «Mode N: f [Hz]»
    om = 2 * np.pi * np.array([float(x) for x in re.findall(r"^\s+Mode \d: ([\d.]+)\s*$", log, re.M)[:4]])
al, be = (float(x) for x in re.search(r"alpha=([\d.eE+-]+), beta=([\d.eE+-]+)", log).groups())
gaf = re.search(r"modal GAF table (.*?\.npz)", log).group(1)
gaf = gaf if gaf.startswith("/") else str(REPO / gaf)
if a.gaf:
    gaf = str(Path(a.gaf).resolve())
    print("GAF:", gaf)
NM = len(om)
os.chdir(C.REPO)
with contextlib.redirect_stdout(io.StringIO()):
    cfg = C.get_config("GOLAND_water")
from COUPLING.hydroelastic_utl import aero_system, bem_delta
from COUPLING.hydroelastic_utl import pk_selfconsistent_scan as scs
cfg.quvlm_npz_path = gaf; cfg.aero_method = "UVLM"; cfg.aero_b2_removal = False
cfg.pk_modal_dof_indices = list(range(NM))
with contextlib.redirect_stdout(io.StringIO()):
    Qg = aero_system.make_Qg_func_uvlm(cfg)
M = np.eye(NM); K = np.diag(om ** 2); Cm = al * M + be * K
hydro = None
if a.stage == "delta":
    capy = [f for f in glob.glob(str(run / "capytaine" / "modal_radiation_AB_depth_*.npz"))]
    hb = np.load(max(capy, key=os.path.getmtime), allow_pickle=True)
    wh = np.asarray(hb["omega"], float); Ah = np.asarray(hb["added_mass"], float)[:, :NM, :NM]
    Bh = np.asarray(hb["added_damping"], float)[:, :NM, :NM]
    lo, hi = (float(x) for x in a.fit.split(","))
    with contextlib.redirect_stdout(io.StringIO()):
        Anc, _ = bem_delta.fit_uvlm_nc_mass(Qg, (lo, hi), a.rho, NM)
    sym = lambda X: 0.5 * (X + X.T)
    def hydro(w):
        w = float(np.clip(w, wh[0], wh[-1]))
        A = np.array([[np.interp(w, wh, Ah[:, i, j]) for j in range(NM)] for i in range(NM)])
        B = np.array([[np.interp(w, wh, Bh[:, i, j]) for j in range(NM)] for i in range(NM)])
        return sym(A) - Anc, sym(B)
    print("delta: diag A_BEM(w_max)-A_nc", np.round(np.diag(sym(Ah[-1]) - Anc), 4), "BEM w in", wh[[0, -1]])
if a.circ_factor != 1.0:
    # la massa non circolatoria (2/rho) A_nc k^2 resta intatta, cosi' il delta col BEM non cambia
    lo, hi = (float(x) for x in a.fit.split(","))
    with contextlib.redirect_stdout(io.StringIO()):
        Anc0, _ = bem_delta.fit_uvlm_nc_mass(Qg, (lo, hi), a.rho, NM)
    Qg0 = Qg
    def Qg(k, V, *r):
        Qnc = (2.0 / a.rho) * Anc0 * k ** 2
        return Qnc + a.circ_factor * (np.atleast_2d(Qg0(k, V, *r)) - Qnc)
    print("circ-factor %.4f sulla parte circolatoria di Q" % a.circ_factor)
V =np.arange(a.vmin_kn * C.KN, a.vmax_kn * C.KN + 1e-9, a.dv)
Mg = M + (hydro(1e9)[0] if hydro else 0)
om_max = 2.0 * np.max(np.sqrt(np.diag(K) / np.diag(Mg)))
t0 = time.time()
# omega_min di default (1 rad/s, assoluto) taglia via le radici delle ali grandi (s=256: flutter a 0.68 rad/s)
res = scs.sweep(V, a.rho, Qg, M, Cm, K, n_omega=a.n_omega, config=cfg, hydro_func=hydro,
                omega_min=0.2 * float(np.min(om)), omega_max=om_max)
out = a.out or str(run / f"scan_{a.stage}.txt")
with open(out, "w") as fh:
    for v in sorted(res):
        for w, s in res[v]:
            fh.write("%.4f %.4f %.6e\n" % (v / C.KN, w / 2 / np.pi, s))
Vs = sorted(res); smax = [max([s for _, s in res[v]], default=np.nan) for v in Vs]
j = next((i for i in range(1, len(Vs)) if smax[i - 1] < 0 <= smax[i]), None)
if j is None:
    print("%s: nessuna radice instabile fra %.1f e %.1f kn (%.0f s)" % (a.stage, a.vmin_kn, Vs[-1] / C.KN, time.time() - t0))
else:
    v0, v1 = Vs[j - 1], Vs[j]; vf = v0 - smax[j - 1] * (v1 - v0) / (smax[j] - smax[j - 1])
    w = [w for w, s in res[v1] if s == smax[j]][0]
    print("%s: V_f %.2f kn @ %.2f Hz  (sigma %.3f -> %.3f, %.0f s)" % (a.stage, vf / C.KN, w / 2 / np.pi, smax[j - 1], smax[j], time.time() - t0))
