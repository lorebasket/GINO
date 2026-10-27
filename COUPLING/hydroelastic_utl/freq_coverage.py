"""Copertura in frequenza delle tabelle fluide (BEM in omega, UVLM in k = omega/V).

REGOLA. Le matrici fluide sono piene (N x N fra modi): a una radice di frequenza
omega servono TUTTI i termini Q_ij(omega). Il range da precalcolare e' quindi
l'UNIONE delle bande attorno alle frequenze bagnate di tutti i modi:

    banda_i = [f_i - m_i, f_i + m_i],   m_i = max(margin_hz, margin_rel * f_i)

- BEM (Capytaine): nessuna velocita' di avanzamento, dipende solo da omega.
  La superficie libera deve essere risolta su TUTTA la banda (non solo A(inf), B=0).
- UVLM: tabella in k = omega/V [1/m]; il massimo richiesto e' omega_max / V_min.

Fuori tabella gli interpolatori saturano in silenzio (BEM) o estrapolano
l'asintoto (UVLM, k > k_max): per questo ogni radice va controllata e, se esce,
la tabella va ricalcolata su un range piu' ampio (`required_*`).
"""
from __future__ import annotations

import numpy as np

MARGIN_HZ = 10.0
MARGIN_REL = 0.30


def required_omega_band(wet_freqs_hz, margin_hz=MARGIN_HZ, margin_rel=MARGIN_REL, f_floor_hz=0.0):
    """-> (omega_lo, omega_hi) [rad/s]: unione delle bande attorno ai modi bagnati."""
    f = np.asarray(wet_freqs_hz, float).ravel()
    m = np.maximum(margin_hz, margin_rel * f)
    lo = max(float(np.min(f - m)), f_floor_hz)
    hi = float(np.max(f + m))
    return 2 * np.pi * lo, 2 * np.pi * hi


def free_surface_upper(omega, B):
    """Omega piu' alta a cui B e' stato calcolato davvero (oltre: B azzerato = A(inf)).

    Se B e' identicamente zero sopra un certo indice mentre sotto non lo e',
    quella e' una troncatura della griglia, non fisica. Si guardano solo gli zeri
    ESATTI: il decadimento fisico di B (grande omega, grande profondita') lascia
    valori piccoli ma non nulli.
    """
    w = np.asarray(omega, float)
    b = np.abs(np.diagonal(np.asarray(B, float), axis1=1, axis2=2)).max(axis=1)
    if not np.any(b > 0):
        return None                      # niente superficie libera: B nullo ovunque
    nz = np.nonzero(b != 0.0)[0]
    last = nz[-1]
    if last == len(w) - 1:
        return float(w[-1])
    # decaduto fino all'underflow (B ultimo << picco): fisico, la griglia copre tutto
    if b[last] < 1e-6 * b.max():
        return float(w[-1])
    return float(w[last])


def check_roots(roots, *, bem_omega=None, bem_B=None, uvlm_kmax=None, band=None, omega_cap=None):
    """Controlla un insieme di radici (omega [rad/s], V [m/s]).

    -> dict con, per ogni tabella, quante radici escono e il range minimo richiesto.
    `band` = (omega_lo, omega_hi) della regola sui modi: entra nel requisito.
    """
    r = np.asarray(roots, float).reshape(-1, 2)
    r = r[np.isfinite(r).all(1) & (r[:, 0] > 0) & (r[:, 1] > 0)]
    if omega_cap is not None:            # radici numeriche spurie (es. 4.8e4 Hz) fuori dal problema
        r = r[r[:, 0] <= omega_cap]
    w, V = r[:, 0], r[:, 1]
    out = {"n_roots": int(len(r))}
    w_need_hi = float(w.max()) if len(w) else 0.0
    if band is not None:
        w_need_hi = max(w_need_hi, band[1])
    if bem_omega is not None:
        g = np.asarray(bem_omega, float)
        fs = free_surface_upper(g, bem_B) if bem_B is not None else float(g[-1])
        hi_ok = g[-1] if fs is None else fs     # senza superficie libera B=0 e' fisico
        out["bem"] = dict(grid=(float(g[0]), float(g[-1])), free_surface_up_to=fs,
                          n_out_grid=int(np.sum((w < g[0]) | (w > g[-1]))),
                          n_out_free_surface=int(np.sum(w > hi_ok)),
                          omega_required=(float(min(g[0], w.min() if len(w) else g[0])), w_need_hi),
                          ok=bool(w_need_hi <= hi_ok and (not len(w) or w.min() >= g[0])))
    if uvlm_kmax is not None:
        k = w / V
        k_need = float(k.max()) if len(k) else 0.0
        if band is not None and len(V):
            k_need = max(k_need, band[1] / float(V.min()))
        out["uvlm"] = dict(k_max=float(uvlm_kmax), n_out=int(np.sum(k > uvlm_kmax)),
                           k_required=k_need, ok=bool(k_need <= uvlm_kmax),
                           V_at_out=sorted(set(np.round(V[k > uvlm_kmax], 3).tolist()))[:6])
    return out


def format_report(name, rep):
    s = ["[copertura] %s: %d radici" % (name, rep["n_roots"])]
    b = rep.get("bem")
    if b:
        s.append("  BEM  griglia %.2f-%.2f rad/s, superficie libera fino a %s; "
                 "radici fuori griglia %d, fuori superficie libera %d; serve fino a %.1f rad/s -> %s"
                 % (*b["grid"], "n/d" if b["free_surface_up_to"] is None else "%.1f" % b["free_surface_up_to"],
                    b["n_out_grid"], b["n_out_free_surface"], b["omega_required"][1],
                    "OK" if b["ok"] else ("RICALCOLARE" if b["n_out_grid"] + b["n_out_free_surface"]
                                          else "OK sulle radici, banda preventiva non coperta")))
    u = rep.get("uvlm")
    if u:
        s.append("  UVLM k_max %.1f 1/m; radici fuori %d (V %s); banda preventiva chiede k fino a %.1f -> %s"
                 % (u["k_max"], u["n_out"], u["V_at_out"], u["k_required"],
                    "OK" if u["ok"] else ("RICALCOLARE" if u["n_out"] else "OK sulle radici, banda preventiva non coperta")))
    return "\n".join(s)


def report_and_save(config, roots, out_dir, *, bem_omega=None, bem_B=None, uvlm_kmax=None,
                    wet_omega=None, omega_cap=None):
    """Controllo a fine p-k/scansione: stampa, scrive freq_coverage.txt, avvisa.

    Margini da config: freq_coverage_margin_hz / freq_coverage_margin_rel.
    Con config.freq_coverage_strict=True un'uscita dalla tabella solleva errore.
    """
    import os
    band = None
    if wet_omega is not None and len(wet_omega):
        # regola LV: [0, f_N + 20 Hz], N = min(20, modi del calcolo)
        band = (0.0, band_from_modes(
            np.asarray(wet_omega) / (2 * np.pi),
            n_modes=int(getattr(config, "freq_coverage_n_modes", N_MODES_BAND)),
            margin_top_hz=float(getattr(config, "freq_coverage_margin_top_hz", MARGIN_TOP_HZ))))
    rep = check_roots(roots, bem_omega=bem_omega, bem_B=bem_B, uvlm_kmax=uvlm_kmax, band=band,
                      omega_cap=omega_cap)
    name = str(getattr(config, "name", ""))
    txt = format_report(name, rep)
    if band is not None:
        txt += "\n  banda dai modi bagnati: %.2f-%.2f rad/s" % band
    print(txt)
    try:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "freq_coverage.txt"), "w") as fh:
            fh.write(txt + "\n")
    except OSError:
        pass
    # radici effettive fuori tabella = errore vero; la sola banda preventiva = avviso
    hard = (rep.get("bem", {}).get("n_out_free_surface", 0) + rep.get("bem", {}).get("n_out_grid", 0)
            + rep.get("uvlm", {}).get("n_out", 0))
    if hard:
        print("[copertura] ATTENZIONE: %d radici fuori dalle tabelle fluide precalcolate -> "
              "ricalcolare BEM/UVLM sul range indicato" % hard)
        if getattr(config, "freq_coverage_strict", False):
            raise RuntimeError("radici fuori dalle tabelle fluide (freq_coverage_strict)")
    return rep


# --- regola concordata con LV (14/10/2026) -----------------------------------
# banda = [0, f_N,bagnato + 20 Hz] con N = min(20, modi del calcolo); la superficie
# libera si risolve fin dove la mesh regge (>= 5 pannelli per lunghezza d'onda),
# sopra si usa A(inf), B=0 SOLO se B e' gia' decaduto (B/(w A) < 1%).
N_MODES_BAND = 20
MARGIN_TOP_HZ = 20.0
PANELS_PER_WAVE = 5.0
B_DECAY_TOL = 0.01
G = 9.81


def band_from_modes(wet_freqs_hz, n_modes=N_MODES_BAND, margin_top_hz=MARGIN_TOP_HZ):
    """-> omega_hi [rad/s] = 2 pi (f_N + margin), N = min(n_modes, len)."""
    f = np.sort(np.asarray(wet_freqs_hz, float).ravel())
    f = f[np.isfinite(f) & (f > 0)][:n_modes]
    return 2 * np.pi * (float(f[-1]) + margin_top_hz)


def omega_fs_resolved(panel_size_m, panels_per_wave=PANELS_PER_WAVE):
    """Omega massima con onda di superficie risolta: lambda = 2 pi g / w^2 >= n * dx."""
    return float(np.sqrt(2 * np.pi * G / (panels_per_wave * float(panel_size_m))))


def bem_omega_grid(omega_hi, omega_fs_max, n_lo=30, n_hi=30, omega_min=0.05):
    """Griglia BEM: geometrica fitta fino a omega_fs_max (superficie libera), lineare
    fino a omega_hi. -> (omega, mask_free_surface). omega_min > 0 perche' Capytaine
    non accetta omega=0; sotto omega_min l'interpolatore usa il primo punto (A, B
    gia' costanti a bassa frequenza)."""
    w_fs = min(float(omega_fs_max), float(omega_hi))
    lo = np.geomspace(omega_min, w_fs, n_lo)
    hi = np.linspace(w_fs, omega_hi, n_hi)[1:] if omega_hi > w_fs * 1.0001 else np.array([])
    w = np.unique(np.r_[lo, hi])
    return w, w <= w_fs * (1 + 1e-9)


def b_decay_ok(omega, A, B, omega_fs, tol=B_DECAY_TOL):
    """B/(w A) all'ultimo punto con superficie libera, per modo: deve essere < tol
    perche' sopra si possa usare A(inf), B=0. -> (ok, rapporto per modo)."""
    w = np.asarray(omega, float)
    i = int(np.argmin(np.abs(w - omega_fs)))
    a = np.abs(np.diag(np.asarray(A)[i])); b = np.abs(np.diag(np.asarray(B)[i]))
    r = b / np.maximum(w[i] * a, 1e-30)
    return bool(np.all(r < tol)), r
