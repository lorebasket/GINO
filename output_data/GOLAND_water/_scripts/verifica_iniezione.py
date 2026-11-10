"""Verifica INDIPENDENTE dell'iniezione UVLM + BEM nel p-k di main.py (GOLAND_water, rev A, metodo delta).

Per ogni run: ogni radice convergente del p-k (V, lambda) deve essere autovalore del sistema ricostruito da
zero dai file grezzi alla k = Im(lambda)/V: modi secchi (M = I, K = diag w^2), Rayleigh alpha/beta del log,
tabella UVLM (aero_imag_flip=False per GOLAND), A_UVLM_nc rifittata qui (k in [14, 24]), A(w), B(w) di Capytaine.
Stampa lo scarto e la massa d'acqua nelle matrici contro A_BEM.
    python output_data/GOLAND_water/_scripts/verifica_iniezione.py
"""
import glob, os, re
import numpy as np

GW = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FSI = os.path.abspath(os.path.join(GW, "..", ".."))
RHO = 997.0
out = []


def interp(x, G, xq):
    xq = float(np.clip(xq, x[0], x[-1])); i = min(max(int(np.searchsorted(x, xq)), 1), len(x) - 1)
    t = (xq - x[i - 1]) / (x[i] - x[i - 1]); return (1 - t) * G[i - 1] + t * G[i]


for d in sorted(glob.glob(os.path.join(GW, "caso*_" + os.environ.get("GW_REV", "revA"), "*_delta_*hc*"))):
    log = open(os.path.join(d, "flutter.log"), errors="ignore").read()
    al, be = (float(x) for x in re.search(r"alpha=([\d.eE+-]+), beta=([\d.eE+-]+)", log).groups())
    q_npz = re.search(r"modal GAF table (\S+\.npz)", log).group(1)
    tb = np.load(q_npz); kl = tb["k_list"]; Q = tb["Q"]; n = Q.shape[1]
    eg = np.genfromtxt(glob.glob(os.path.join(d, "GOLAND_water_dry_egv_eigendata.csv"))[0], delimiter=",", names=True)
    w0 = np.asarray(eg["Omega_rad_s"])[:n]
    M = np.eye(n); K = np.diag(w0 ** 2); C = al * M + be * K
    m = (kl >= 14) & (kl <= 24)
    Anc = np.array([[0.5 * RHO * np.polyfit(kl[m] ** 2, Q[m, i, j].real, 1)[0] for j in range(n)] for i in range(n)])
    Anc = 0.5 * (Anc + Anc.T)
    an_log = np.array([float(x) for x in re.search(r"A_UVLM_nc dal fit.*diag = \[([^\]]+)\]", log).group(1).split()])
    hb = np.load(os.path.join(d, "capytaine", "modal_radiation_AB.npz"), allow_pickle=True)
    wh, Ah, Bh = hb["omega"], hb["added_mass"][:, :n, :n], hb["added_damping"][:, :n, :n]
    ev = np.genfromtxt(os.path.join(d, "eigenvalue_data.csv"), delimiter=",", names=True)
    vg = np.genfromtxt(os.path.join(d, "vg_data.csv"), delimiter=",", names=True)
    conv = {(round(v, 6), 0): c1 for v, c1 in zip(vg["Velocity_m_s"], vg["Converged_Branch_1"])}
    conv.update({(round(v, 6), 1): c2 for v, c2 in zip(vg["Velocity_m_s"], vg["Converged_Branch_2"])})
    errs, emass, br = [], [], []
    for V, md, re_, im_ in zip(ev["Velocity_m_s"], ev["Mode"], ev["Real_Part_lambda_rad_s"], ev["Imag_Part_lambda_rad_s"]):
        if conv.get((round(V, 6), int(md))) != 1 or im_ <= 0.5:
            continue
        w = im_; k = w / V; qd = 0.5 * RHO * V ** 2
        Qk = Q[-1].real / kl[-1] ** 2 * k ** 2 + 1j * Q[-1].imag / kl[-1] * k if k > kl[-1] else interp(kl, Q, k)
        Ab = interp(wh, Ah, w); Bb = interp(wh, Bh, w)
        Mw = M + Ab - Anc; Cw = C + Bb - 0.5 * RHO * V / k * np.imag(Qk); Kw = K - qd * np.real(Qk)
        Z = np.zeros((n, n)); Ac = np.block([[Z, np.eye(n)], [-np.linalg.solve(Mw, Kw), -np.linalg.solve(Mw, Cw)]])
        e = np.linalg.eigvals(Ac); errs.append(np.min(np.abs(e - (re_ + 1j * im_))) / im_); br.append(int(md) if k <= kl[-1] else -1)
        if errs[-1] > 5e-3 and os.environ.get('VER_DEBUG'):
            print('   V %.2f m/s ramo %d k %.1f (k_max %.0f) scarto %.1e' % (V, md, k, kl[-1], errs[-1]))
        emass.append(np.abs((Mw - M + Anc) - Ab).max() / np.abs(Ab).max())
    name = os.path.basename(os.path.dirname(d)).split("_")[0] + " " + d.split("_hc")[-1]
    errs, br = np.array(errs), np.array(br); lo = np.array([e for e in errs])
    s = ("%-12s radici %3d (fuori tabella k>k_max: %d) | scarto mediano %.1e | ramo 2 (flutter): max %.1e | ramo 1 (flessione): max %.1e, punti > 1%% %d | "
         "A_nc log vs indip %.1e (arrotondamento log) | massa usata - A_BEM %.1e"
         % (name, len(errs), int((br == -1).sum()), np.median(errs), errs[br == 1].max(), errs[br == 0].max(), int((errs[br == 0] > 1e-2).sum()),
            np.abs(an_log - np.diag(Anc)).max(), max(emass)))
    print(s); out.append(s)
open(os.path.join(GW, "freq_coverage", "verifica_iniezione_goland.txt"), "w").write("\n".join(out) + "\n")
