"""T-foil semplificato sul modello di AC38 — configurazione GINO.

Scopo: un esempio pulito di T-foil che vada in flutter, per confrontare UVLM-only
(massa apparente dell'UVLM, nessuna superficie libera) con UVLM + BEM (massa
aggiunta e radiazione di Capytaine, superficie libera a z = 0).

Geometria, materiale e sezioni: tfoil_simple_geom.py (unica fonte, variabili TFS_*).
Catena: run_chain.sh (modi -> GAF UVLM -> BEM -> p-k UVLM e UVLM+BEM).

Variabili d'ambiente:
  TFS_STAGE = uvlm (default) | bem
  TFS_NM    = numero di modi del p-k (default 8)
  TFS_WL    = [mm] pelo libero spostato (sweep di sommergenza; default 0)
  TFS_* di tfoil_simple_geom.py (materiale, assi, spessore)
"""
import os
import importlib.util
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_s = importlib.util.spec_from_file_location("tfoil_simple_geom", os.path.join(_HERE, "tfoil_simple_geom.py"))
G = importlib.util.module_from_spec(_s); _s.loader.exec_module(G)

NE_RUDDER, NE_ELEV = 52, 18
# incastro alla radice bagnata (z = -90 mm): tutti e sei i gdl del nodo 0
NODE_CONSTRAINTS = {0: ('ux', 'uy', 'uz', 'thx', 'thy', 'thz')}


def paths(FSI_path):
    """Percorsi di modi, GAF, BEM e uscite per la variante corrente (nome = G.tag())."""
    NM = int(os.environ.get("TFS_NM", "8"))
    WL = float(os.environ.get("TFS_WL", "0") or 0.0)
    t = G.tag() + ("_wl%04d" % round(WL) if WL else "")
    # griglia della tabella GAF (TFS_GRID): la n40c12e16 storica tiene il nome senza suffisso
    grid = os.environ.get("TFS_GRID", "n40c12e16")
    gs = "" if grid == "n40c12e16" else "_" + grid
    base = os.path.join(FSI_path, "output_data", "TFOIL_SIMPLE", t)
    return dict(
        tag=t, nm=NM, wl=WL, base=base,
        dry=os.path.join(base, "dry_modes.npz"),
        gaf=os.path.join(FSI_path, "FLUID", "PanelAero", "Qjj", "Quvlm_TFOIL_SIMPLE_%s_nm%d%s.npz" % (t, NM, gs)),
        grid=grid,
        # massa NON circolatoria dell'UVLM, dalla stessa tabella (stadio delta)
        anc=os.path.join(base, "A_uvlm_nc_nm%d%s.npz" % (NM, gs)),
        bem=os.path.join(FSI_path, "FLUID", "capytaine", "TFOIL_SIMPLE", t + "_nm%d" % NM, "modal_radiation_AB.npz"),
    )


def wet_seeds(P, bem):
    """Pulsazioni BAGNATE a V = 0 per il seed del tracker, con la massa aggiunta dello
    stadio: BEM (Capytaine, omega -> inf) oppure UVLM (0.5 rho Re Q / k^2 sul piatto in
    k). Qui l'acqua pesa 1-6 volte la struttura: partendo dalle frequenze secche il
    tracker aggancia radici spurie (ramo 1 convergente nello 0.8 % dei punti)."""
    if not os.path.exists(P["dry"]):
        return None
    NM = P["nm"]; d = np.load(P["dry"])
    M = np.asarray(d["M_hat"])[:NM, :NM]; K = np.asarray(d["K_hat"])[:NM, :NM]
    src = P["bem"] if bem else P["gaf"]
    if not os.path.exists(src):
        return list(2 * np.pi * d["f"][:NM])
    if bem:
        A = np.load(src)["added_mass"][-1][:NM, :NM]
    else:
        q = np.load(src); k = q["k_list"]; i = int(np.argmin(np.abs(k - 30.0)))
        A = 0.5 * G.RHO_W * q["Q"][i].real / k[i] ** 2
    A = 0.5 * (A + A.T)
    # lo slot i del p-k e' il modo secco i: a ciascuno la radice bagnata in cui quel
    # modo pesa di piu' (l'acqua rimescola l'ordine: il 3 secco scende sotto il 2)
    from scipy.linalg import eigh
    from scipy.optimize import linear_sum_assignment
    w2, X = eigh(K, M + A)
    W = np.abs(X) ** 2 / (np.abs(X) ** 2).sum(0)
    r, c = linear_sum_assignment(-W)
    return list(np.sqrt(np.maximum(w2[c[np.argsort(r)]], 1e-6)))


def get_config(AnalysisConfig, FSI_path, QJJ_path):
    P = paths(FSI_path)
    NM = P["nm"]
    bem = os.environ.get("TFS_STAGE", "uvlm") == "bem"
    out_dir = os.path.join(P["base"], "pk_uvlm_bem_nm%d" % NM if bem else "pk_uvlm_nm%d" % NM)
    seeds = wet_seeds(P, bem)
    # griglia a pannelli di AC38: serve a main.py solo come supporto (spline, grafici);
    # le forze vengono dalla tabella GAF UVLM
    qjj_dir = os.path.join(QJJ_path, "AC38_RUDDER_IB500_water_alpha0.0_nspan20_nchord10")
    V_max = float(os.environ.get("TFS_VMAX_KN", "85")) * 0.514444

    return AnalysisConfig(
        'TFOIL_SIMPLE',
        structural_model_type='tfoil_simple',
        beam_rotations_physical=G.PHYS,
        node_constraints=NODE_CONSTRAINTS,
        n_elements=NE_RUDDER,
        n_elements_elevator=NE_ELEV,
        beam_length=G.L_RUDDER,
        chord=0.5 * (G.C_ROOT + G.C_RTIP),
        chord_root=G.C_ROOT,
        chord_tip=G.C_RTIP,
        fluid='water',
        rho_s=G.RHO_S,
        alpha_deg=0.0,
        alpha_r=0.0,
        nspan=20, nchord=10,
        qjj_dir=qjj_dir,
        vjj_dir=os.path.join(QJJ_path, "vjj_AC38_RUDDER_IB500_water"),
        aerogrid_path=os.path.join(qjj_dir, "aerogrid.npz"),
        caero_x1=-G.XEA * G.C_ROOT, caero_y1=0.0, caero_z1=G.Z_ROOT,
        caero_x4=-G.XEA * G.C_RTIP, caero_y4=0.0, caero_z4=G.Z_TIP,
        caero_id=1, caero_pid=1, caero_igid=1,
        thickness_factor=G.TC,
        chordwise_strip_y=-1.5,
        dihedral_angle=-90,
        dihedral_rotate_beam=True,
        dihedral_rotate_aerogrid=False,
        beam_span_axis=2,
        xea_factor=G.XEA,
        aerogrid_xea_factor=G.XEA,
        decouple_xea_from_aerogrid=False,

        aero_method='UVLM',
        quvlm_npz_path=P["gaf"],
        aero_b2_removal=bem,            # con il BEM si toglie la massa apparente dell'UVLM
        quvlm_k_extrapolate=True,
        added_mass_strip_theory=False,
        pk_capytaine_matrices=bem,
        empirical_fluid_damping=False,

        num_modes_egv=NM + 4,
        num_modes_flutter_egv=NM,
        pk_modal_dof_indices=list(range(NM)),

        pk_method=True,
        roger_fit=False,
        RFA_PK_method=False,
        structural_damping_model='none',
        V_list=np.unique(np.concatenate([np.linspace(0.5, 5.0, 10), np.linspace(5.0, V_max, 120)])),
        pk_track_seed_omegas=seeds,
        pk_tracking_policy='frequency',
        pk_fRLX=0.25,
        pk_max_iter=400,
        mac_matching=True,
        last_converged_mode_matching=True,

        output_dir=out_dir,
        modal_dir=out_dir,
        prefix="TFOIL_SIMPLE_dry_egv",
        v_knots=True,
        vf_hertz=True,
        plot_vg_vf=True,
        plot_aerobeam=False,
        plot_mode_shapes=False,
        plot_deformed_modes=False,

        capytaine_results_dir=os.path.dirname(P["bem"]),
        mesh_path=os.path.join(FSI_path, "FLUID", "capytaine", "TFOIL_SIMPLE", "mesh_placeholder.vtu"),
        capytaine_auto_recalc=False,
        free_surface_elevation=0.0,
        capytaine_phi_normalization='geometric',
        omega_list=np.geomspace(0.5, 120.0, 16),
        depth=[0.0],
        depth_index=0,
        water_depth=np.inf,
    )


if __name__ == "__main__":
    # stampa un percorso per gli script di shell:  python config_TFOIL_SIMPLE.py <FSI> {tag|base|dry|gaf|bem}
    import sys
    print(paths(sys.argv[1])[sys.argv[2]])
