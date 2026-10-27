#!/usr/bin/env python3
"""CHECK: which water mass really enters the p-k matrices, stage by stage, for any case.

In the p-k  [-omega^2 (M + M_hydro) + K - qinf Q(k)] q = 0,  qinf = rho V^2 / 2,  k = omega / V,
Re Q acts as an inertia: qinf Re Q / omega^2 = (rho/2) Re Q(k) / k^2.  So the effective modal
water mass at (V, omega) is   M_water = M_hydro(omega) + (rho/2) Re Q(k) / k^2.

Stages (same functions as production: aero_system.make_Qg_func_uvlm, Capytaine matrices)
  UVLM only   Q = table intact,            M_hydro = 0                -> ~ A_UVLM_nc
  B2 scheme   Q = table - k^2 B2(k_b2),   M_hydro = A_BEM(omega)      -> ~ A_BEM
  delta       Q = table intact,            M_hydro = A_BEM - A_UVLM_nc -> A_BEM by construction
Printed as ratios to A_BEM(inf) on the diagonal, after removing the circulatory part.

    python COUPLING/debug/check_hydro_mass_in_pk.py TFOIL_SIMPLE [--gaf .. --bem .. --anc .. --nm 10]
"""
import argparse
import contextlib
import io
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "utl"))
import _common as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("case")
    for k in ("gaf", "bem", "anc"):
        ap.add_argument("--" + k)
    ap.add_argument("--nm", type=int)
    ap.add_argument("--kb2", type=float, default=34.0)
    ap.add_argument("--speeds-kn", default="10,20,30,45")
    ap.add_argument("--freqs-hz", default="7.5,15,22")
    a = ap.parse_args()

    P = C.case_paths(a.case, gaf=a.gaf, bem=a.bem, anc=a.anc)
    with contextlib.redirect_stdout(io.StringIO()):
        cfg = C.get_config(a.case)
        from COUPLING.hydroelastic_utl import aero_system
        cfg.aero_method = "UVLM"
        cfg.quvlm_npz_path = P["gaf"]
        cfg.aero_b2_removal = False
        Qu = aero_system.make_Qg_func_uvlm(cfg)
    rho = float(np.load(P["gaf"])["rho"])
    NM = a.nm or int(P.get("nm") or np.load(P["gaf"])["Q"].shape[1])
    cfg.pk_modal_dof_indices = list(range(NM))
    B2 = -np.real(Qu(a.kb2, 1.0, None, 0, 0))[:NM, :NM] / a.kb2 ** 2
    B2 = 0.5 * (B2 + B2.T)
    cap = np.load(P["bem"])
    w_c, A_c = cap["omega"], cap["added_mass"][:, :NM, :NM]
    A_bem = lambda w: np.array([[np.interp(w, w_c, A_c[:, i, j]) for j in range(NM)] for i in range(NM)])
    Anc = np.load(P["anc"])["A_nc"][:NM, :NM]
    dA = 0.5 * ((A_c[-1] - Anc) + (A_c[-1] - Anc).T)
    eff = lambda k: 0.5 * rho * np.real(Qu(k, 1.0, None, 1, 1))[:NM, :NM] / k ** 2
    ref = np.diag(A_c[-1])
    modes = [i for i in range(NM) if abs(ref[i]) > 1e-6 * np.abs(ref).max()]
    r = lambda X: "".join("%9.2f" % (X[i, i] / ref[i]) for i in modes)
    print(f"{a.case} {P.get('tag', '')}, {NM} modes; diagonal modal water mass")
    print("%-40s%s" % ("", "".join("%9s" % f"m{i + 1}" for i in modes)))
    print("%-40s%s" % ("A_BEM(inf)", "".join("%9.3f" % ref[i] for i in modes)))
    print("%-40s%s" % ("A_UVLM_nc (fit)", "".join("%9.3f" % Anc[i, i] for i in modes)))
    print("\nnon-circulatory mass in the matrices / A_BEM(inf)  (UVLM ~ A_nc/A_BEM, others ~ 1)")
    for V_kn in (float(x) for x in a.speeds_kn.split(",")):
        V = V_kn * C.KN
        for f in (float(x) for x in a.freqs_hz.split(",")):
            w = 2 * np.pi * f
            k = w / V
            mu = eff(k)
            circ = mu - Anc
            mb = A_bem(w) + mu + 0.5 * rho * B2          # Q_B2 = Q + k^2 B2
            print("V=%4.0f kn f=%5.1f Hz k=%6.1f  UVLM only %s" % (V_kn, f, k, r(mu - circ)))
            print("%-34s B2 scheme %s" % ("", r(mb - circ)))
            print("%-34s delta     %s" % ("", r(dA + mu - circ)))


if __name__ == "__main__":
    main()
