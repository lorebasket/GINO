"""Radiazione modale (Capytaine) del timone AC38 a T: massa aggiunta e smorzamento.

CORPI (opzione A, decisa con LV): guscio chiuso del timone
(`AC38_RUDDER/AC38_RUDDER_IB500_mesh.vtu`) + guscio chiuso dell'ala
dell'elevatore (`2604.../out/aero_ELEV/elevator_shell.npz`, loft delle sezioni).
Il bulbo d'acciaio all'innesto e' escluso. Superficie libera a z = 0, acqua
profonda.

MODI. Quelli del T di GINO (`tfoil_dry_modes.npz`, frame di GINO prima del
diedro). `run_modal_radiation.py` ricostruisce gli spostamenti lungo un'unica
coordinata di span e non regge un T: qui ogni pannello segue il nodo di trave
piu' vicino DEL SUO RAMO (timone per quota, semiala per |y|) con la cinematica
rigida della sezione, u + th x (r - r_nodo).

FRAME. GINO -> CAD (metri):  x_c = 0.777 - x_G,  y_c = -z_G,  z_c = 0.565 - y_G.
Per i vettori (spostamenti, rotazioni) la parte lineare: diag(-1) con y<->z,
R = [[-1,0,0],[0,0,-1],[0,-1,0]], det +1.

    conda run -n GINO python run_tfoil_radiation.py --modes <npz> --nm 10
"""
import argparse, os
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
FSI = HERE.parents[1]
RUDDER_MESH = HERE / "AC38_RUDDER" / "AC38_RUDDER_IB500_mesh.vtu"
ELEV_SHELL = FSI / "2604 - Project Flutter" / "code" / "out" / "aero_ELEV" / "elevator_shell.npz"
OUT_DIR = HERE / "AC38_TFOIL" / "results_modal_radiation"
R_GC = np.array([[-1.0, 0, 0], [0, 0, -1.0], [0, -1.0, 0]])
O_GC = np.array([0.777, 0.0, 0.565])            # CAD = O + R x_G
OMEGA_FS = np.array([0.5, 0.7, 0.99, 1.38, 1.94, 2.73, 3.83, 5.38, 7.55, 10.6, 14.88, 20.89])
# Sopra ~30 rad/s le onde di superficie libera (lunghezza 2*pi*g/omega^2 = 7 cm a
# 30 rad/s) non sono risolte dalla mesh e si smorzano con la profondita' come
# exp(-omega^2 d/g): a 30 rad/s e 9 cm (inizio della carenatura bagnata) restano
# 3e-4. La superficie libera e' li' una parete a potenziale nullo: si usa il
# limite di frequenza infinita di Capytaine (B = 0) per tutte queste omega.
OMEGA_INF = np.array([29.33, 41.18, 57.81, 81.17, 113.96, 160.0, 225.0, 320.0, 450.0])
OMEGA = np.concatenate([OMEGA_FS, OMEGA_INF])


Z_TOP_BEM = -0.090    # quota superiore del guscio del timone [m], decisa con LV:
                      # stessa della superficie portante UVLM (Geometria_idro)


def cap_top(V, F, z_cut):
    """Taglia il guscio a z_cut tenendo la parte sotto e lo richiude con un coperchio.

    Si tengono i triangoli tutti sotto z_cut, si proiettano su z_cut i vertici
    dei triangoli a cavallo (il guscio e' quasi verticale li', l'errore e' di
    mezzo pannello), poi il bordo aperto si chiude con un ventaglio sul baricentro.
    """
    V = V.copy()
    zc = V[F].max(axis=1)[:, 2]
    keep = V[F][:, :, 2].min(axis=1) < z_cut
    F = F[keep]
    used = np.unique(F)
    V[used, 2] = np.minimum(V[used, 2], z_cut)
    # bordo aperto: lati che appartengono a un solo triangolo
    E = np.sort(np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
    u, n = np.unique(E, axis=0, return_counts=True)
    b = u[n == 1]
    b = b[np.all(np.abs(V[b][:, :, 2] - z_cut) < 1e-9, axis=1)]
    # ordina il ciclo
    adj = {}
    for a_, c_ in b:
        adj.setdefault(a_, []).append(c_); adj.setdefault(c_, []).append(a_)
    loop = [b[0][0]]; prev = None
    while True:
        nx = [k for k in adj[loop[-1]] if k != prev]
        if not nx or nx[0] == loop[0]:
            break
        prev = loop[-1]; loop.append(nx[0])
    c = len(V); V = np.vstack([V, V[loop].mean(0)])
    lid = np.array([[loop[i], loop[(i + 1) % len(loop)], c] for i in range(len(loop))])
    # coperchio con normale verso +z (esterno)
    nz = np.cross(V[lid[:, 1]] - V[lid[:, 0]], V[lid[:, 2]] - V[lid[:, 0]])[:, 2]
    if nz.mean() < 0:
        lid = lid[:, [1, 0, 2]]
    F = np.vstack([F, lid])
    used = np.unique(F); remap = np.full(len(V), -1); remap[used] = np.arange(len(used))
    return V[used], remap[F], len(loop)


def cut_open(V, F, z_cut):
    """Taglia il guscio a z_cut tenendo la parte sotto, SENZA coperchio: e' il
    corpo che attraversa il pelo libero, aperto sul piano d'acqua (convenzione
    standard dei solutori a pannelli con superficie libera)."""
    V = V.copy()
    keep = V[F][:, :, 2].min(axis=1) < z_cut - 1e-9
    F = F[keep]
    if len(F) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)
    used = np.unique(F)
    V[used, 2] = np.minimum(V[used, 2], z_cut)
    remap = np.full(len(V), -1); remap[used] = np.arange(len(used))
    return V[used], remap[F]


def rudder_points(m):
    """vertici del guscio del timone [m]; con AC38_RUD_SHIFT=<csv z_mm,dx_mm> ogni quota traslata di dx(z) verso prua
    (caso 32, asse elastico dritto): la traslazione dipende solo da z, quindi il guscio resta chiuso."""
    P = np.array(m.points, float)
    sh = os.environ.get("AC38_RUD_SHIFT")
    if sh:
        t = np.loadtxt(sh, delimiter=",", comments="#", ndmin=2)
        P[:, 0] += np.interp(-P[:, 2] * 1e3, -t[:, 0], t[:, 1]) * 1e-3
        print("guscio timone traslato per sezione da %s" % sh)
    # AC38_RUD_TSCALE=<csv z_mm,tc_orig,tc_nuovo,s> (caso 35): spessore scalato per quota, y -> s(z) y attorno al
    # piano di simmetria y = 0; corda e pianta invariate, il guscio resta chiuso
    ts = os.environ.get("AC38_RUD_TSCALE")
    if ts:
        t = np.loadtxt(ts, delimiter=",", comments="#", ndmin=2)
        P[:, 1] *= np.interp(-P[:, 2] * 1e3, -t[:, 0], t[:, 3])
        print("guscio timone scalato in spessore da %s" % ts)
    return P


def load_meshes(cpt, wl=None):
    """wl = None: caso base (guscio chiuso a Z_TOP_BEM, pelo libero a z = 0).
    wl [m]: pelo libero a z = wl; timone tagliato e aperto a wl. Le mesh restano
    nel frame CAD: la traslazione del pelo libero si fa nel problema BEM."""
    import meshio
    m = meshio.read(str(RUDDER_MESH))
    tri = np.vstack([c.data for c in m.cells if c.type == "triangle"])
    P = rudder_points(m)
    if wl is None:
        V, tri, nl = cap_top(P, tri, Z_TOP_BEM)
        print("guscio timone tagliato a z = %.3f m: %d triangoli (coperchio %d)" % (Z_TOP_BEM, len(tri), nl))
    else:
        V, tri = cut_open(P, tri, wl)
        print("pelo libero a z = %.3f m: guscio timone aperto, %d triangoli" % (wl, len(tri)))
    rud = cpt.Mesh(vertices=V, faces=np.column_stack([tri, tri[:, -1]]), name="rudder") if len(tri) else None
    e = np.load(ELEV_SHELL)
    Ve = np.array(e["V"], float)
    dx = float(os.environ.get("AC38_ELEV_DX", "0") or 0.0)   # [mm] verso prua (x CAD crescente)
    Ve[:, 0] += dx * 1e-3
    # AC38_ELEV_SHIFT=<csv y_mm,dx_mm>: ogni sezione traslata di dx(|y|) (asse elastico dritto); il guscio e' un
    # loft di stazioni a y costante, quindi la traslazione per vertice lo lascia chiuso
    sh = os.environ.get("AC38_ELEV_SHIFT")
    if sh:
        t = np.loadtxt(sh, delimiter=",", comments="#", ndmin=2)
        Ve[:, 0] += np.interp(np.abs(Ve[:, 1]) * 1e3, t[:, 0], t[:, 1]) * 1e-3
        print("guscio elevatore traslato per sezione da %s" % sh)
    ele = cpt.Mesh(vertices=Ve, faces=e["F"], name="elevator")
    return rud, ele


def refine_near(V, F, z_s, band, levels):
    """Suddivide 1 -> 4 i pannelli con baricentro entro `band` [m] dalla quota z_s (pelo libero),
    `levels` volte (fascia dimezzata a ogni livello). Facce quad [a,b,c,d] o triangoli
    degeneri [a,b,c,c]; nodi pendenti ammessi (metodo a pannelli costanti)."""
    V = [tuple(v) for v in np.asarray(V, float)]; F = np.asarray(F, int)
    mid = {}
    def m(i, j):
        k = (min(i, j), max(i, j))
        if k not in mid:
            mid[k] = len(V); V.append(tuple(0.5 * (np.array(V[i]) + np.array(V[j]))))
        return mid[k]
    for lev in range(int(levels)):
        Va = np.array(V); out = []
        for f in F:
            tri = f[2] == f[3]
            zc = Va[f[:3] if tri else f].mean(0)[2]
            if abs(zc - z_s) > band / 2 ** lev:
                out.append(f); continue
            if tri:
                a, b, c = f[:3]; ab, bc, ca = m(a, b), m(b, c), m(c, a)
                out += [[a, ab, ca, ca], [ab, b, bc, bc], [ca, bc, c, c], [ab, bc, ca, ca]]
            else:
                a, b, c, d = f; ab, bc, cd, da = m(a, b), m(b, c), m(c, d), m(d, a)
                ce = len(V); V.append(tuple(Va[f].mean(0)))
                out += [[a, ab, ce, da], [ab, b, bc, ce], [ce, bc, c, cd], [da, ce, cd, d]]
        F = np.array(out)
    return np.array(V), F


def face_displacements(centers, part, g, nm):
    """Spostamento (n_face, 3) di ogni pannello per ciascun modo, frame CAD."""
    pos, sub, Phi = g["pos"], g["sub"], g["Phi"][:, :nm]
    U = Phi.reshape(len(pos), 6, nm)
    out = np.zeros((nm, len(centers), 3))
    rud = sub == "rudder"
    br = {"elevator_port": sub == "elevator_port", "elevator_stbd": sub == "elevator_stbd"}
    xg = (R_GC.T @ (centers - O_GC).T).T           # centri dei pannelli nel frame di GINO
    for i, c in enumerate(xg):
        if part[i] == "rudder":
            m = rud; key = pos[m, 1]; q = c[1]
        else:
            lab = "elevator_port" if c[2] > 0 else "elevator_stbd"
            m = br[lab]; key = np.abs(pos[m, 2]); q = abs(c[2])
        j = np.flatnonzero(m)[int(np.argmin(np.abs(key - q)))]
        u = U[j, :3, :]; th = U[j, 3:, :]
        d = u + np.cross(th.T, (c - pos[j])[None, :]).T          # (3, nm), frame GINO
        out[:, i, :] = (R_GC @ d).T
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", type=Path, required=True)
    ap.add_argument("--nm", type=int, default=10)
    ap.add_argument("--out", type=Path, default=OUT_DIR / "modal_radiation_AB.npz")
    ap.add_argument("--wl", type=float, default=None,
                    help="quota del pelo libero [mm, CAD] per lo sweep di sommergenza")
    ap.add_argument("--refine-levels", type=int, default=0, help="suddivisioni 1->4 vicino al pelo libero")
    ap.add_argument("--refine-band", type=float, default=0.3, help="[m] fascia attorno al pelo libero (dimezzata a ogni livello)")
    ap.add_argument("--omega-list", type=str, default="", help="pulsazioni esplicite, tutte con superficie libera (convergenza)")
    ap.add_argument("--omega-max", type=float, default=0.0, help="[rad/s] tetto = banda [0, f_N + 20 Hz]")
    ap.add_argument("--omega-fs", type=float, default=0.0, help="[rad/s] superficie libera fino a qui (0 = dalla mesh)")
    ap.add_argument("--boundary", choices=["fs", "inf", "wall"], default="fs",
                    help="fs: superficie libera a z = 0 (default); inf: fluido infinito, nessuna superficie; "
                         "wall: fondo scafo rigido a z = Z_TOP_BEM (immagine speculare del timone, fluido infinito)")
    a = ap.parse_args()
    import capytaine as cpt
    g = dict(np.load(a.modes, allow_pickle=True))
    wl = None if a.wl is None else a.wl * 1e-3
    rud, ele = load_meshes(cpt, wl)
    if a.refine_levels > 0:                        # infittimento alla linea d'acqua (regola di copertura)
        zs = 0.0 if wl is None else wl
        if rud is not None:
            Vr, Fr = refine_near(rud.vertices, rud.faces, zs, a.refine_band, a.refine_levels)
            rud = cpt.Mesh(vertices=Vr, faces=Fr, name="rudder")
        Ve, Fe = refine_near(ele.vertices, ele.faces, zs, a.refine_band, a.refine_levels)
        ele = cpt.Mesh(vertices=Ve, faces=Fe, name="elevator")
    if a.boundary == "wall":
        # parete rigida a Z_TOP_BEM: timone aperto alla parete + sua immagine speculare (normali
        # riorientate invertendo l'ordine dei vertici). L'immagine si muove come lo specchio del
        # timone; le forze si leggono solo sui pannelli veri (gdl "real_k", vedi sotto).
        assert wl is None, "--boundary wall solo per il caso base"
        import meshio
        m_ = meshio.read(str(RUDDER_MESH)); tri_ = np.vstack([c.data for c in m_.cells if c.type == "triangle"])
        Vo, To = cut_open(np.asarray(m_.points, float), tri_, Z_TOP_BEM)
        Vi = Vo.copy(); Vi[:, 2] = 2 * Z_TOP_BEM - Vi[:, 2]
        rud = cpt.Mesh(vertices=Vo, faces=np.column_stack([To, To[:, -1]]), name="rudder")
        img = cpt.Mesh(vertices=Vi, faces=np.column_stack([To[:, ::-1], To[:, 0]]), name="image")
        print("parete a z = %.3f: timone aperto %d + immagine %d triangoli" % (Z_TOP_BEM, len(To), len(To)))
    mesh = ele if rud is None else rud + ele
    nr = 0 if rud is None else rud.nb_faces
    part = np.array(["rudder"] * nr + ["elevator"] * ele.nb_faces)
    D = face_displacements(mesh.faces_centers, part, g, a.nm)
    if a.boundary == "wall":
        ci = img.faces_centers.copy(); ci[:, 2] = 2 * Z_TOP_BEM - ci[:, 2]            # punto vero speculare
        Di = face_displacements(ci, np.array(["rudder"] * img.nb_faces), g, a.nm)
        Di[:, :, 2] *= -1.0                                                          # S d, S = diag(1, 1, -1)
        Dreal = np.concatenate([D, np.zeros_like(Di)], axis=1)
        D = np.concatenate([D, Di], axis=1)
        mesh = mesh + img
    if wl is not None:
        # pelo libero di Capytaine a z = 0: si alza tutto il corpo di -wl
        mesh = mesh.translated([0.0, 0.0, -wl])
    names = ["mode_%d" % (k + 1) for k in range(a.nm)]
    dofs = {n: D[k] for k, n in enumerate(names)}
    infl = names
    if a.boundary == "wall":
        infl = ["real_%d" % (k + 1) for k in range(a.nm)]       # stessi modi, solo sui pannelli veri
        dofs.update({n: Dreal[k] for k, n in enumerate(infl)})
    body = cpt.FloatingBody(mesh=mesh, dofs=dofs, name="ac38_tfoil")
    import sys
    sys.path.insert(0, str(FSI))
    from COUPLING.hydroelastic_utl import freq_coverage as fc
    zc = mesh.faces_centers[:, 2]; size = np.sqrt(mesh.faces_areas)
    near = zc > -0.15                                  # pannelli entro 15 cm dalla superficie (z = 0 dopo la traslazione)
    dx_fs = float(size[near].max() if near.any() else size.max())
    w_res = fc.omega_fs_resolved(dx_fs)
    if a.boundary != "fs":
        # senza superficie libera A non dipende da omega e B = 0: una sola soluzione
        a.omega_list = ""; a.omega_max = 0.0
        omega = np.array([0.1, 1.0e4]); fs_mask = np.array([False, False])
    elif a.omega_list:
        omega = np.array([float(x) for x in a.omega_list.split(",")]); fs_mask = np.isfinite(omega)
    elif a.omega_max > 0:
        omega, fs_mask = fc.bem_omega_grid(a.omega_max, a.omega_fs or w_res)
    else:                                              # griglia storica (rev A)
        omega = OMEGA; fs_mask = np.isin(OMEGA, OMEGA_FS)
    print("pannelli: timone %d + elevatore %d = %d; modi %d; omega %d (fino a %.1f), superficie libera fino a %.1f; "
          "pannello alla superficie %.1f mm -> onda risolta fino a %.1f rad/s"
          % (nr if rud is None else rud.nb_faces, ele.nb_faces, mesh.nb_faces, a.nm, len(omega), np.nanmax(omega[np.isfinite(omega)]),
             omega[fs_mask].max() if fs_mask.any() else 0.0, 1e3 * dx_fs, w_res), flush=True)
    solver = cpt.BEMSolver()
    A = np.zeros((len(omega), a.nm, a.nm)); B = np.zeros_like(A)
    def solve(w):
        fsurf = 0.0 if a.boundary == "fs" else np.inf
        probs = [cpt.RadiationProblem(body=body, radiating_dof=n, omega=w,
                                      water_depth=np.inf, free_surface=fsurf, rho=997.0, g=9.81)
                 for n in names]
        Aw = np.zeros((a.nm, a.nm)); Bw = np.zeros_like(Aw)
        for r in solver.solve_all(probs, keep_details=False):
            i = names.index(r.radiating_dof)
            for j, n in enumerate(infl):
                Aw[j, i] = r.added_masses[n]
                Bw[j, i] = 0.0 if np.isinf(w) else r.radiation_dampings[n]
        return Aw, Bw
    for iw in np.flatnonzero(fs_mask):
        A[iw], B[iw] = solve(float(omega[iw]))
        print("omega %7.2f  A_diag %s  B_diag %s" % (omega[iw], np.array2string(np.diag(A[iw])[:4], precision=3),
              np.array2string(np.diag(B[iw])[:4], precision=3)), flush=True)
    if not fs_mask.all():
        Ainf, _ = solve(np.inf if a.boundary == "fs" else 1.0)
        print("omega   inf    A_diag %s" % np.array2string(np.diag(Ainf)[:4], precision=3), flush=True)
        A[~fs_mask] = Ainf
        if a.boundary == "fs" and fs_mask.any() and np.isfinite(omega).all():
            ok, r = fc.b_decay_ok(omega, A, B, omega[fs_mask].max())
            print("[copertura] B/(w A) a %.1f rad/s (ultima con superficie libera): max %.2e -> %s"
                  % (omega[fs_mask].max(), r.max(), "A(inf), B=0 accettabili sopra" if ok else "B NON decaduto"), flush=True)
    OMEGA_OUT = omega
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, omega=OMEGA_OUT, free_surface_mask=fs_mask, panel_fs_m=dx_fs, mode_names=np.array(names), added_mass=A,
             added_damping=B, radiation_damping=B, depth=np.array(0.0 if wl is None else -wl),
             case_name=np.array("AC38_TFOIL"), rigid_body_motion=np.array(False),
             capytaine_modal_span_axis_used=np.array(-1),
             boundary=np.array(a.boundary),
             note=np.array("T: proiezione per ramo, cinematica rigida di sezione; bulbo escluso"))
    print("scritto", a.out)


if __name__ == "__main__":
    main()
