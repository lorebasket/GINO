"""Radiazione modale (Capytaine) del timone AC37: massa aggiunta e smorzamento.

CORPI: due gusci chiusi dalla geometria di examples/AC37_TFOIL/ac37_geom.py, profili simmetrici
NACA 00tt (timone 11.2 %, elevatore 9.5 %, slide 4-5 del deck AC37):
  timone     dal pelo libero (G.Z_WL, guscio aperto alla linea d'acqua) fino al dorso
             dell'elevatore; bordo d'attacco e corda dalla tabella (sotto l'ultima stazione: costanti)
  elevatore  un solo guscio da punta a punta, pianta E8 misurata, quota dei centri di taglio
             della tabella (diedro negativo alla radice), tappi alle punte
Superficie libera a z = 0, acqua profonda.

MODI: quelli di GINO (dry_modes.npz). Ogni pannello segue il nodo di trave piu' vicino DEL SUO
RAMO con la cinematica rigida della sezione, u + th x (r - r_nodo) (come run_tfoil_simple_radiation.py);
i pannelli del timone sotto l'ultima stazione (Z = 2.75) seguono quindi la punta.

FRAME. GINO -> BEM (metri):  x_c = x_G,  y_c = z_G,  z_c = Z_WL - y_G   (R proprio, det +1)

    python FLUID/capytaine/run_ac37_radiation.py --modes <dry_modes.npz> --nm 10 --omega-max <w> --out <npz>
"""
import argparse
import sys
import importlib.util
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
FSI = HERE.parents[1]
_s = importlib.util.spec_from_file_location("ac37_geom", FSI / "examples" / "AC37_TFOIL" / "ac37_geom.py")
G = importlib.util.module_from_spec(_s); _s.loader.exec_module(G)

R_GC = np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]])
O_GC = np.array([0.0, 0.0, G.Z_WL])                    # BEM = O + R x_G


def _half_t(xi, tc):
    return 5 * tc * (0.2969 * np.sqrt(xi) - 0.1260 * xi - 0.3516 * xi**2 + 0.2843 * xi**3 - 0.1036 * xi**4)


def _profile(n, tc):
    """Contorno chiuso su corda unitaria, 2n punti (attacco -> uscita sul dorso, ritorno sul ventre)."""
    xi = 0.5 * (1 - np.cos(np.linspace(0, np.pi, n + 1)))
    yt = _half_t(xi, tc)
    return np.vstack([np.column_stack([xi, yt]), np.column_stack([xi[::-1], -yt[::-1]])[1:-1]])


def loft(sections, n_chord, tc, caps=(True, True)):
    """Guscio quadrilatero chiuso. sections: lista di funzioni (xi, t) -> punti 3D (frame BEM)."""
    P = _profile(n_chord, tc); npts = len(P)
    V = np.vstack([f(P[:, 0], P[:, 1]) for f in sections]); ns = len(sections)
    F = []
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


def to_bem(xg, yg, zg):
    return np.column_stack([xg, zg, G.Z_WL - yg])


def build_meshes(cpt, n_span=24, n_chord=16, n_elev=18):
    y_bot = G.Z_ELEV_ROOT - 0.5 * G.TC_ELEV * float(G.chord_elev(0.0))     # dorso dell'elevatore
    yr = np.linspace(G.Z_WL, y_bot, n_span + 1)

    def rud_sec(y):
        c = float(G.chord_rudder(y)); xle = float(G.x_le_rudder(y))
        return lambda xi, t: to_bem(xle + xi * c, np.full_like(xi, y), t * c)
    secs_r = [rud_sec(y) for y in yr]
    Vr, Fr = loft(secs_r, n_chord, G.TC_RUD, caps=(False, True))
    Ye = np.linspace(-G.ELEV_SEMISPAN, G.ELEV_SEMISPAN, 2 * n_elev + 1)

    def elev_sec(Y):
        c = float(G.chord_elev(Y)); xle = float(G.x_le_elev(abs(Y))); ye = float(G.elev_axis(abs(Y))[1])
        return lambda xi, t: to_bem(xle + xi * c, ye - t * c, np.full_like(xi, Y))
    Ve, Fe = loft([elev_sec(Y) for Y in Ye], n_chord, G.TC_ELEV)

    def _vol(V, F):
        m = cpt.Mesh(vertices=V, faces=F)
        return (m.faces_centers * m.faces_normals * m.faces_areas[:, None]).sum() / 3
    # normali verso l'esterno: volume positivo (per il timone, aperto in cima, sulla versione chiusa)
    if _vol(*loft(secs_r, n_chord, G.TC_RUD)) < 0:
        Fr = Fr[:, ::-1]
    if _vol(Ve, Fe) < 0:
        Fe = Fe[:, ::-1]
    rud = cpt.Mesh(vertices=Vr, faces=Fr, name="rudder")
    ele = cpt.Mesh(vertices=Ve, faces=Fe, name="elevator")
    print("volumi: timone (chiuso) %.4f m3, elevatore %.4f m3" % (abs(_vol(*loft(secs_r, n_chord, G.TC_RUD))), _vol(Ve, Fe)))
    return rud, ele


def face_displacements(centers, part, g, nm):
    pos, sub, Phi = g["pos"], g["sub"].astype(str), g["Phi"][:, :nm]
    U = Phi.reshape(len(pos), 6, nm)
    rud = np.flatnonzero(sub == "rudder")
    el = np.flatnonzero(np.char.startswith(sub, "elevator"))
    rootG = int(el[np.argmin(np.abs(pos[el, 2]))])
    br = {lab: np.unique(np.concatenate([[rootG], np.flatnonzero(sub == lab)])) for lab in ("elevator_stbd", "elevator_port")}
    xg = (R_GC.T @ (centers - O_GC).T).T
    out = np.zeros((nm, len(centers), 3))
    for i, c in enumerate(xg):
        if part[i] == "rudder":
            idx = rud; key = pos[idx, 1]; q = c[1]
        else:
            idx = br["elevator_port" if c[2] > 0 else "elevator_stbd"]; key = np.abs(pos[idx, 2]); q = abs(c[2])
        j = idx[int(np.argmin(np.abs(key - q)))]
        d = U[j, :3, :] + np.cross(U[j, 3:, :].T, (c - pos[j])[None, :]).T
        out[:, i, :] = (R_GC @ d).T
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", type=Path, required=True)
    ap.add_argument("--nm", type=int, default=10)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--free-surface", default="on", choices=["on", "off"], help="off: dominio illimitato (controllo)")
    ap.add_argument("--n-span", type=int, default=24)
    ap.add_argument("--n-chord", type=int, default=16)
    ap.add_argument("--n-elev", type=int, default=18)
    ap.add_argument("--omega-max", type=float, default=300.0, help="[rad/s] tetto della griglia = banda [0, f_N + 20 Hz]")
    ap.add_argument("--omega-fs", type=float, default=0.0, help="[rad/s] superficie libera risolta fino a qui (0 = dalla mesh)")
    a = ap.parse_args()
    import capytaine as cpt
    g = dict(np.load(a.modes, allow_pickle=True))
    rud, ele = build_meshes(cpt, a.n_span, a.n_chord, a.n_elev)
    mesh = rud + ele
    part = np.array(["rudder"] * rud.nb_faces + ["elevator"] * ele.nb_faces)
    D = face_displacements(mesh.faces_centers, part, g, a.nm)
    fs = 0.0 if a.free_surface == "on" else np.inf
    names = ["mode_%d" % (k + 1) for k in range(a.nm)]
    body = cpt.FloatingBody(mesh=mesh, dofs={n: D[k] for k, n in enumerate(names)}, name="ac37")
    sys.path.insert(0, str(FSI))
    from COUPLING.hydroelastic_utl import freq_coverage as fc
    zc = mesh.faces_centers[:, 2]; size = np.sqrt(mesh.faces_areas)
    near = zc > -0.15
    dx_fs = float(size[near].max() if near.any() else size.max())
    w_res = fc.omega_fs_resolved(dx_fs)
    omega, fs_mask = fc.bem_omega_grid(a.omega_max, a.omega_fs or w_res)
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
        print("omega %7.2f  A_diag %s  B_diag %s" % (omega[iw], np.array2string(np.diag(A[iw])[:5], precision=3),
              np.array2string(np.diag(B[iw])[:5], precision=3)), flush=True)
    if not fs_mask.all():
        Ainf, _ = solve(np.inf)
        print("omega   inf    A_diag %s" % np.array2string(np.diag(Ainf)[:6], precision=3), flush=True)
        A[~fs_mask] = Ainf
        w_top = omega[fs_mask].max()
        ok, r = fc.b_decay_ok(omega, A, B, w_top)
        print("[copertura] B/(w A) a %.1f rad/s (ultima con superficie libera): max %.2e -> %s"
              % (w_top, r.max(), "A(inf), B=0 accettabili sopra" if ok else "B NON decaduto: rifinire la mesh"), flush=True)
    if a.free_surface == "off":
        A[:] = A[0]; B[:] = 0.0
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, omega=omega, free_surface_mask=fs_mask, panel_fs_m=dx_fs, mode_names=np.array(names), added_mass=A,
             added_damping=B, radiation_damping=B, depth=np.array(0.0), case_name=np.array("AC37_" + G.tag()),
             imm=np.array(G.IMM), z_wl=np.array(G.Z_WL),
             note=np.array("AC37: gusci NACA da ac37_geom, cinematica rigida di sezione; superficie libera " + a.free_surface))
    print("scritto", a.out)


if __name__ == "__main__":
    main()
