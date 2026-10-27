"""Radiazione modale (Capytaine) del T-foil semplificato: massa aggiunta e smorzamento.

CORPI: due gusci chiusi generati dalla geometria (examples/TFOIL_SIMPLE/tfoil_simple_geom.py),
profilo NACA 00tt.
  timone     da z = -90 mm fino al dorso dell'elevatore (non lo compenetra), tappi in cima e in fondo
  elevatore  un solo guscio da punta a punta, tappi alle due punte
Superficie libera a z = 0 (oppure a TFS_WL), acqua profonda.

MODI: quelli di GINO (dry_modes.npz). Ogni pannello segue il nodo di trave piu' vicino DEL
SUO RAMO con la cinematica rigida della sezione, u + th x (r - r_nodo), come in
run_tfoil_radiation.py (AC38).

FRAME. GINO -> BEM (metri):  x_c = x_G,  y_c = z_G,  z_c = Z_ROOT - y_G   (R proprio, det +1)

OMEGA: come AC38. Sopra ~30 rad/s la lunghezza d'onda di superficie (2 pi g / w^2 < 7 cm)
scende sotto il pannello: li' si usa il limite di frequenza infinita (superficie a
pressione nulla, B = 0), che e' anche la fisica giusta (k_w L >> 1).

    conda run -n GINO python FLUID/capytaine/run_tfoil_simple_radiation.py --modes <dry_modes.npz> --nm 8 --out <npz>
"""
import argparse
import sys
import importlib.util
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
FSI = HERE.parents[1]
_s = importlib.util.spec_from_file_location("tfoil_simple_geom", FSI / "examples" / "TFOIL_SIMPLE" / "tfoil_simple_geom.py")
G = importlib.util.module_from_spec(_s); _s.loader.exec_module(G)

R_GC = np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]])
O_GC = np.array([0.0, 0.0, G.Z_ROOT])                  # BEM = O + R x_G
OMEGA_FS = np.array([0.5, 0.7, 0.99, 1.38, 1.94, 2.73, 3.83, 5.38, 7.55, 10.6, 14.88, 20.89])
OMEGA_INF = np.array([29.33, 41.18, 57.81, 81.17, 113.96, 160.0, 225.0, 320.0, 450.0])
OMEGA = np.concatenate([OMEGA_FS, OMEGA_INF])


def _profile(n):
    """Contorno chiuso del profilo su corda unitaria, n punti per lato (senza duplicati)."""
    xi = 0.5 * (1 - np.cos(np.linspace(0, np.pi, n + 1)))
    yt = G.naca_half_thickness(xi)
    up = np.column_stack([xi, yt])                     # bordo d'attacco -> uscita, dorso
    lo = np.column_stack([xi[::-1], -yt[::-1]])[1:-1]  # uscita -> attacco, ventre
    return np.vstack([up, lo])                         # 2n punti


def loft(stations, chords, place, n_chord, caps=(True, True)):
    """Guscio quadrilatero chiuso. place(s, xc, t) -> punti 3D della sezione alla stazione s."""
    P = _profile(n_chord); npts = len(P)
    V, F = [], []
    for s, c in zip(stations, chords):
        xc = (P[:, 0] - G.XEA) * c                     # asse elastico in x = 0
        V.append(place(s, xc, P[:, 1] * c))
    V = np.vstack(V); ns = len(stations)
    for i in range(ns - 1):
        for j in range(npts):
            a, b = i * npts + j, i * npts + (j + 1) % npts
            F.append([a, b, b + npts, a + npts])
    F = np.array(F)
    for i, sgn in [t for t, on in zip(((0, 1), (ns - 1, -1)), caps) if on]:   # tappi a ventaglio
        c0 = len(V); V = np.vstack([V, V[i * npts:(i + 1) * npts].mean(0)])
        for j in range(npts):
            a, b = i * npts + j, i * npts + (j + 1) % npts
            F = np.vstack([F, [a, b, c0, c0] if sgn < 0 else [b, a, c0, c0]])
    return V, F


def _graded(z_top, z_bot, n_uniform, dz0):
    """Stazioni da z_top a z_bot: primo pannello dz0 alla linea d'acqua, crescita
    geometrica (rapporto 1.15) fino al passo uniforme (z_top - z_bot)/n_uniform."""
    L = z_top - z_bot; du = L / n_uniform
    if not dz0 or dz0 >= du:
        return np.linspace(z_top, z_bot, n_uniform + 1)
    z = [z_top]; d = dz0
    while z_top - z[-1] + d < L and d < du:
        z.append(z[-1] - d); d *= 1.15
    rest = z[-1] - z_bot; n = max(1, int(round(rest / du)))
    return np.r_[z, np.linspace(z[-1], z_bot, n + 1)[1:]]


def build_meshes(cpt, n_span=40, n_chord=16, n_elev=16, wl_dz=0.0):
    t_root_e = G.TC * G.C_EROOT
    z_bot = G.Z_TIP + 0.5 * t_root_e                   # il timone si ferma sul dorso dell'elevatore
    # pelo libero sotto la radice: il timone buca la superficie, guscio aperto alla linea
    # d'acqua (convenzione standard dei solutori a pannelli con superficie libera)
    pierce = G.WL < G.Z_ROOT
    z_top = G.WL if pierce else G.Z_ROOT
    n = max(4, int(round(n_span * (z_top - z_bot) / (G.Z_ROOT - z_bot))))
    zr = _graded(z_top, z_bot, n, wl_dz)
    Vr, Fr = loft(zr, G.chord_rudder(-zr), lambda z, x, t: np.column_stack([x, t, np.full_like(x, z)]), n_chord,
                  caps=(not pierce, True))
    ye = np.linspace(-G.ELEV_SEMISPAN, G.ELEV_SEMISPAN, 2 * n_elev + 1)
    edx = float(getattr(G, "EDX", 0.0))
    Ve, Fe = loft(ye, G.chord_elev(ye), lambda y, x, t: np.column_stack([x + edx, np.full_like(x, y), G.Z_TIP + t]), n_chord)
    rud = cpt.Mesh(vertices=Vr, faces=Fr, name="rudder")
    ele = cpt.Mesh(vertices=Ve, faces=Fe, name="elevator")
    # normali verso l'esterno: il volume deve venire positivo (guscio aperto: solo l'elevatore)
    for m in ((ele,) if pierce else (rud, ele)):
        vol = (m.faces_centers * m.faces_normals * m.faces_areas[:, None]).sum() / 3
        if vol < 0:
            raise RuntimeError("normali del guscio %s verso l'interno" % m.name)
    return rud, ele


def face_displacements(centers, part, g, nm, torsion_sign=1.0):
    pos, sub, Phi = g["pos"], g["sub"].astype(str), g["Phi"][:, :nm]
    U = Phi.reshape(len(pos), 6, nm)
    rud = sub == "rudder"
    tipG = int(np.flatnonzero(rud)[np.argmax(pos[rud, 1])])
    el = np.flatnonzero(np.char.startswith(sub, "elevator") | (sub == "link"))
    near = el[np.abs(pos[el, 1] - pos[tipG, 1]) < 1e-9]
    rootG = int(near[np.argmin(np.abs(pos[near, 2]) + 1e3 * (np.abs(pos[near, 0] - pos[tipG, 0]) < 1e-9))]) \
        if near.size else tipG
    if np.abs(pos[rootG, 0] - pos[tipG, 0]) < 1e-9:
        rootG = tipG
    br = {lab: np.unique(np.concatenate([[rootG], np.flatnonzero(sub == lab)])) for lab in ("elevator_stbd", "elevator_port")}
    xg = (R_GC.T @ (centers - O_GC).T).T
    out = np.zeros((nm, len(centers), 3))
    for i, c in enumerate(xg):
        if part[i] == "rudder":
            idx = np.flatnonzero(rud); key = pos[idx, 1]; q = c[1]
        else:
            idx = br["elevator_port" if c[2] > 0 else "elevator_stbd"]; key = np.abs(pos[idx, 2]); q = abs(c[2])
        j = idx[int(np.argmin(np.abs(key - q)))]
        # rotazioni fisiche (destrorse), come la struttura con TFS_PHYS = 1 e come SHARPy. torsion_sign = -1 inverte la
        # rotazione attorno all'asse di apertura della parte (y timone, z elevatore): solo per diagnosi (21/10/2026)
        th = U[j, 3:, :].copy()
        th[1 if part[i] == "rudder" else 2] *= torsion_sign
        d = U[j, :3, :] + np.cross(th.T, (c - pos[j])[None, :]).T
        out[:, i, :] = (R_GC @ d).T
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", type=Path, required=True)
    ap.add_argument("--nm", type=int, default=8)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--free-surface", default="on", choices=["on", "off"], help="off: dominio illimitato (controllo)")
    ap.add_argument("--n-span", type=int, default=40)
    ap.add_argument("--n-chord", type=int, default=16)
    ap.add_argument("--n-elev", type=int, default=16)
    ap.add_argument("--wl-dz", type=float, default=0.0, help="[m] primo pannello del timone alla linea d'acqua (0 = uniforme)")
    ap.add_argument("--omega-list", type=str, default="", help="pulsazioni esplicite (prove di convergenza), tutte con superficie libera")
    ap.add_argument("--omega-max", type=float, default=0.0, help="[rad/s] tetto della griglia = banda [0, f_N + 20 Hz]")
    ap.add_argument("--omega-fs", type=float, default=0.0, help="[rad/s] superficie libera risolta fino a qui (0 = dalla mesh)")
    ap.add_argument("--torsion-sign", type=float, default=1.0,
                    help="+1 = rotazioni fisiche (default); -1 = torsione invertita, solo per diagnosi")
    a = ap.parse_args()
    import capytaine as cpt
    g = dict(np.load(a.modes, allow_pickle=True))
    rud, ele = build_meshes(cpt, a.n_span, a.n_chord, a.n_elev, a.wl_dz)
    mesh = rud + ele
    part = np.array(["rudder"] * rud.nb_faces + ["elevator"] * ele.nb_faces)
    D = face_displacements(mesh.faces_centers, part, g, a.nm, a.torsion_sign)
    wl = G.WL                                          # pelo libero (TFS_WL), Capytaine lo vuole a z = 0
    if wl:
        mesh = mesh.translated([0.0, 0.0, -wl])
    fs = 0.0 if a.free_surface == "on" else np.inf
    names = ["mode_%d" % (k + 1) for k in range(a.nm)]
    body = cpt.FloatingBody(mesh=mesh, dofs={n: D[k] for k, n in enumerate(names)}, name="tfoil_simple")
    # --- griglia in omega (regola di copertura LV, 14/10/2026) -----------------
    sys.path.insert(0, str(FSI))
    from COUPLING.hydroelastic_utl import freq_coverage as fc
    zc = mesh.faces_centers[:, 2]; size = np.sqrt(mesh.faces_areas)
    near = zc > -0.15                                  # pannelli entro 15 cm dalla superficie
    dx_fs = float(size[near].max() if near.any() else size.max())
    w_res = fc.omega_fs_resolved(dx_fs)
    if a.omega_list:
        omega = np.array([float(x) for x in a.omega_list.split(",")]); fs_mask = np.ones(len(omega), bool)
    elif a.omega_max > 0:
        w_fs = a.omega_fs or w_res
        omega, fs_mask = fc.bem_omega_grid(a.omega_max, w_fs)
    else:                                              # griglia storica (AC38)
        omega = OMEGA; fs_mask = np.isin(OMEGA, OMEGA_FS)
    if a.free_surface == "off":
        fs_mask[:] = False; fs_mask[0] = True
    print("pannelli: timone %d + elevatore %d = %d; modi %d; omega %d (%.2f-%.1f rad/s), superficie libera fino a %.1f; "
          "pannello alla superficie %.1f mm -> onda risolta fino a %.1f rad/s"
          % (rud.nb_faces, ele.nb_faces, mesh.nb_faces, a.nm, len(omega), omega[0], omega[-1],
             omega[fs_mask].max(), 1e3 * dx_fs, w_res), flush=True)
    solver = cpt.BEMSolver()

    def solve(w):
        probs = [cpt.RadiationProblem(body=body, radiating_dof=n, omega=w, water_depth=np.inf,
                                      free_surface=fs, rho=G.RHO_W, g=9.81) for n in names]
        Aw = np.zeros((a.nm, a.nm)); Bw = np.zeros_like(Aw)
        for r in solver.solve_all(probs, keep_details=False):
            i = names.index(r.radiating_dof)
            for j, n in enumerate(names):
                Aw[j, i] = r.added_masses[n]
                Bw[j, i] = 0.0 if np.isinf(w) else r.radiation_dampings[n]
        return Aw, Bw
    A = np.zeros((len(omega), a.nm, a.nm)); B = np.zeros_like(A)
    for iw in np.flatnonzero(fs_mask):
        A[iw], B[iw] = solve(float(omega[iw]))
        print("omega %7.2f  A_diag %s  B_diag %s" % (omega[iw], np.array2string(np.diag(A[iw])[:4], precision=3),
              np.array2string(np.diag(B[iw])[:4], precision=3)), flush=True)
    if not fs_mask.all():
        Ainf, _ = solve(np.inf)
        print("omega   inf    A_diag %s" % np.array2string(np.diag(Ainf)[:6], precision=3), flush=True)
        A[~fs_mask] = Ainf
        w_top = omega[fs_mask].max()
        ok, r = fc.b_decay_ok(omega, A, B, w_top)
        print("[copertura] B/(w A) a %.1f rad/s (ultima con superficie libera): max %.2e -> %s"
              % (w_top, r.max(), "A(inf), B=0 accettabili sopra" if ok else "B NON decaduto: rifinire la mesh"), flush=True)
    if a.free_surface == "off":                        # senza superficie A non dipende da omega
        A[:] = A[0]; B[:] = 0.0
    OMEGA_OUT = omega
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, omega=OMEGA_OUT, free_surface_mask=fs_mask, panel_fs_m=dx_fs, mode_names=np.array(names), added_mass=A, added_damping=B,
             radiation_damping=B, depth=np.array(-wl), case_name=np.array("TFOIL_SIMPLE"),
             rigid_body_motion=np.array(False), capytaine_modal_span_axis_used=np.array(-1),
             torsion_sign=np.array(a.torsion_sign),
             note=np.array("T semplificato: gusci NACA da tfoil_simple_geom, cinematica rigida di sezione; "
                           "superficie libera " + a.free_surface))
    print("scritto", a.out)


if __name__ == "__main__":
    main()
