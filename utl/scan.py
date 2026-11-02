#!/usr/bin/env python3
"""Self-consistent p-k scan (COUPLING/hydroelastic_utl/pk_selfconsistent_scan) on any case:
finds ALL the roots at every speed, no branch tracker.

Stages
  uvlm   UVLM GAF table only
  delta  UVLM table intact + (A_BEM(w) - A_UVLM_nc) in M + B_BEM(w) in C   (needs uvlm_nc_mass.py)
  bem    legacy B2 scheme: Re Q - k^2 B2(k_b2), + A_BEM(inf) in M  (unbalanced, kept for comparison)

    python utl/scan.py TFOIL_SIMPLE delta                     # files from the case paths()
    python utl/scan.py MYCASE uvlm --dry d.npz --gaf Q.npz --base output_data/MYCASE/v1 --nm 10
    python utl/scan.py TFOIL_SIMPLE delta --a-source uvlm    # identity test: must equal 'uvlm'

Writes <base>/scan_<stage><tag>.txt (V [kn], f [Hz], sigma [1/s] per root) and the .png.
"""
import argparse
import contextlib
import io
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402


def _interp(G, wh, w):
    w = float(np.clip(w, wh[0], wh[-1]))
    i = int(np.searchsorted(wh, w))
    if i == 0:
        return G[0]
    t = (w - wh[i - 1]) / (wh[i] - wh[i - 1])
    return (1 - t) * G[i - 1] + t * G[i]


def build_hydro(P, NM, a):
    """hydro(w) -> (dM, dC) for the delta stage."""
    from COUPLING.hydroelastic_utl import freq_coverage as fc
    hb = np.load(P["bem"])
    wh = np.asarray(hb["omega"], float)
    Ah = np.asarray(hb["added_mass"], float)[:, :NM, :NM]
    Bh = np.asarray(hb["added_damping"], float)[:, :NM, :NM]
    Anc = np.load(P["anc"])["A_nc"][:NM, :NM]
    use_B = not a.no_B
    if a.a_alpha != 1.0:                                   # stress test: A = A_nc + alpha (A_BEM - A_nc)
        Ah = Anc[None] + a.a_alpha * (Ah - Anc[None])
    if a.a_source == "uvlm":                               # identity: must reproduce UVLM only
        Ah = np.broadcast_to(Anc, Ah.shape).copy()
        use_B = False
    if a.a_inf:
        Ah = np.broadcast_to(Ah[-1], Ah.shape).copy()
    sym = lambda X: 0.5 * (X + X.T)
    w_fs = fc.free_surface_upper(wh, Bh) if use_B else None
    b_ext = a.b_extrapolate and w_fs is not None and w_fs < wh[-1]
    if b_ext:                                              # B ~ w^-n beyond the last free-surface omega
        i1 = int(np.searchsorted(wh, w_fs))
        with np.errstate(divide="ignore", invalid="ignore"):
            n_dec = np.log(np.abs(Bh[i1 - 1]) / np.abs(Bh[i1])) / np.log(wh[i1] / wh[i1 - 1])
        n_dec = np.where(np.isfinite(n_dec) & (n_dec > 0), n_dec, 6.0)

    def hydro(w):
        dM = sym(_interp(Ah, wh, w) - Anc)
        if not use_B:
            return dM, np.zeros_like(dM)
        Bw = Bh[i1] * (w_fs / w) ** n_dec if b_ext and w > w_fs else _interp(Bh, wh, w)
        return dM, sym(Bw)

    print("delta: A(w)%s - A_UVLM_nc, B(w) %s" % (" [A(inf)]" if a.a_inf else "", "yes" if use_B else "no"))
    print("  diag A(inf)-A_nc", np.round(np.diag(Ah[-1] - Anc), 3))
    return hydro, wh, Bh


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("case")
    ap.add_argument("stage", choices=("uvlm", "delta", "bem"))
    for k in ("dry", "gaf", "bem", "anc", "base"):
        ap.add_argument("--" + k, help=f"override paths()['{k}']")
    ap.add_argument("--nm", type=int)
    ap.add_argument("--rho", type=float, help="default: rho stored in the GAF table")
    ap.add_argument("--vmin", type=float, default=1.0, help="m/s")
    ap.add_argument("--vmax-kn", type=float, default=85.0)
    ap.add_argument("--dv", type=float, default=0.5, help="m/s")
    ap.add_argument("--n-omega", type=int, default=6000, help="900 misses roots (see memory note)")
    ap.add_argument("--tag", default="", help="suffix of the output file names")
    ap.add_argument("--kb2", type=float, default=34.0, help="bem stage: k where B2 is sampled")
    ap.add_argument("--a-source", choices=("bem", "uvlm"), default="bem")
    ap.add_argument("--a-scale", type=float, default=1.0, help="bem stage: scale on A_BEM")
    ap.add_argument("--a-alpha", type=float, default=1.0, help="delta stage: A_nc + alpha (A_BEM - A_nc)")
    ap.add_argument("--a-inf", action="store_true", help="delta stage: constant A(inf)")
    ap.add_argument("--no-B", action="store_true", help="delta stage: drop radiation damping")
    ap.add_argument("--b-extrapolate", action="store_true", help="delta stage: B ~ w^-n beyond the BEM range")
    ap.add_argument("--f-max-plot", type=float, default=80.0)
    a = ap.parse_args()

    os.chdir(C.REPO)
    P = C.case_paths(a.case, **{k: getattr(a, k) for k in ("dry", "gaf", "bem", "anc", "base")})
    NM = a.nm or int(P.get("nm") or 0) or None
    with contextlib.redirect_stdout(io.StringIO()):
        cfg = C.get_config(a.case)
    from COUPLING.hydroelastic_utl import aero_system
    from COUPLING.hydroelastic_utl import pk_selfconsistent_scan as scs
    from COUPLING.hydroelastic_utl import freq_coverage as fc
    if P.get("gaf"):
        cfg.quvlm_npz_path = P["gaf"]
    cfg.aero_method = "UVLM"
    cfg.aero_b2_removal = False
    gaf = np.load(cfg.quvlm_npz_path, allow_pickle=True)
    rho = a.rho or float(gaf["rho"])
    d = np.load(P["dry"])
    NM = NM or gaf["Q"].shape[1]
    cfg.pk_modal_dof_indices = list(range(NM))
    M, Cm, K = (d[k][:NM, :NM].copy() for k in ("M_hat", "C_hat", "K_hat"))
    Q0 = aero_system.make_Qg_func_uvlm(cfg)
    Qg, hydro, wh, Bh = Q0, None, None, None
    if a.stage == "bem":
        A = np.load(P["bem"])["added_mass"][-1][:NM, :NM]
        B2 = -np.real(Q0(a.kb2, 1.0, None, 0, 0)) / a.kb2 ** 2
        B2 = 0.5 * (B2 + B2.T)
        if a.a_source == "uvlm":
            A = -0.5 * rho * B2
        A = a.a_scale * A
        M = M + 0.5 * (A + A.T)
        print("UVLM mass removed diag", np.round(np.diag(-0.5 * rho * B2), 3))
        print("BEM mass added    diag", np.round(np.diag(A), 3))

        def Qg(k, V, p, s, j, **kw):
            return Q0(k, V, p, s, j) + float(k) ** 2 * B2
    elif a.stage == "delta":
        hydro, wh, Bh = build_hydro(P, NM, a)

    V = np.arange(a.vmin, a.vmax_kn * C.KN + 1e-9, a.dv)
    Mg = M + hydro(1e9)[0] if hydro else M
    om_struct = np.sqrt(np.abs(np.diag(K) / np.maximum(np.diag(M), 1e-30)))
    om_max = 2.0 * np.max(np.sqrt(np.abs(np.diag(K) / np.maximum(np.diag(Mg), 1e-30))))
    res = scs.sweep(V, rho, Qg, M, Cm, K, n_omega=a.n_omega, config=cfg, hydro_func=hydro, omega_max=om_max)
    first, _ = scs.report(res)
    roots = np.array([(w, v) for v in res for w, _s in res[v]])
    wet = np.array([w for w, _s in res[min(res)]])
    fc.report_and_save(cfg, roots, P["base"], uvlm_kmax=float(gaf["k_list"][-1]), wet_omega=wet,
                       bem_omega=wh, bem_B=Bh, omega_cap=2.0 * np.max(om_struct))

    tag = a.stage + a.tag
    os.makedirs(P["base"], exist_ok=True)
    out = os.path.join(P["base"], f"scan_{tag}.txt")
    with open(out, "w") as fh:
        for v in sorted(res):
            for w, s in res[v]:
                fh.write("%.3f %.4f %.6e\n" % (v / C.KN, w / 2 / np.pi, s))
    if first:
        v, (w, s) = first
        print("FIRST UNSTABLE ROOT (%s): %.2f kn @ %.2f Hz" % (tag, v / C.KN, w / 2 / np.pi))
    else:
        print("no unstable root (%s) up to %.1f kn" % (tag, V[-1] / C.KN))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    r = np.loadtxt(out, ndmin=2)
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(7, 7))
    col = np.where(r[:, 2] > 0, "r", "k")
    ax[0].scatter(r[:, 0], r[:, 1], s=4, c=col)
    ax[0].set_ylabel("f [Hz]")
    ax[0].set_ylim(0, a.f_max_plot)
    ax[1].scatter(r[:, 0], r[:, 2] / (2 * np.pi * np.maximum(r[:, 1], 1e-3)), s=4, c=col)
    ax[1].axhline(0, lw=.5)
    ax[1].set_ylabel("sigma/omega")
    ax[1].set_xlabel("V [kn]")
    ax[1].set_ylim(-1, .3)
    fig.suptitle(f"{a.case} {P.get('tag', '')}, {NM} modes, stage {tag}", fontsize=9)
    fig.savefig(out.replace(".txt", ".png"), dpi=130)


if __name__ == "__main__":
    main()
