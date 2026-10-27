#!/usr/bin/env python3
"""Sample the SHARPy linear-UVLM modal GAF Q(k) for the GOLAND case (air).

Goland twin of ``precompute_quvlm.py`` with two case-specific differences:

* The GINO modal basis does not come from a Capytaine NPZ (no BEM in air):
  the mode shapes are read from the GINO FEA exports
  ``output_data/GOLAND/GOLAND_dry_egv_eigendata.csv`` + ``..._nodes.csv``.
* The stock SHARPy Goland template (``run_goland_water_rho._build_case``)
  models TWO mirrored surfaces with symmetric-mode reduction, while GINO
  models a single cantilever half-wing. After the C-congruence the
  generalized forces integrate both surfaces, i.e. exactly twice the
  half-wing GAF, hence ``--surface-factor 0.5`` (default).

Same calibrated normalization as precompute_quvlm.py: dimfac = ScalingFacts
length^2, conj() for the PanelAero harmonic convention, per-mode LSQ
projection C. Verify with ``config.quvlm_compare_dlm=True`` on the GINO side.

  conda activate sharpy
  cd sharpy
  python gino_bem/precompute_quvlm_goland.py                 # fine grid M16N32Ms10
  python gino_bem/precompute_quvlm_goland.py --grid default  # M8N16Ms6 (fast)
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

_SHARPY_ROOT = Path(__file__).resolve().parents[1]
_FSI_ROOT = _SHARPY_ROOT.parents[1]
if str(_SHARPY_ROOT) not in sys.path:
    sys.path.insert(0, str(_SHARPY_ROOT))

RHO_AIR = 1.02
GRIDS = {
    "coarse": (4, 8, 4),
    "default": (8, 16, 6),
    "fine": (16, 32, 10),      # literature / docs-notebook grid
    # k_max_usable ~= 1.546*M (Nyquist of the linear UVLM ROM's dt=2/M,
    # V_ref-independent) -- "fine" (M=16) tops out ~24.7 1/m, too narrow
    # for GOLAND_water's self-consistent k range (added mass in water
    # pushes k = omega/V much higher than air at the same V). M=48 -> ~74.
    "fineM32": (32, 32, 10),
    "fineM48": (48, 32, 10),
}
def load_gino_goland_modes(nm: int, gino_name: str = "GOLAND"):
    """GINO Goland dry modes from the FEA CSV exports.

    ``gino_name`` selects the ``output_data/<gino_name>/<gino_name>_dry_egv_*``
    export set, i.e. which structural config (and elastic-axis/CG placement)
    the mode shapes come from. Default "GOLAND" = original air axes
    (xea=0.33/xcm=0.43); pass "GOLAND_water" for the recalibrated axes
    (xea=0.28/xcm=0.38, see examples/GOLAND_water/config_GOLAND_water.py).

    Returns (y (n_nodes,), W (nm, n_nodes), TH (nm, n_nodes), omega (nm,) rad/s)
    on the free+clamped node stations (clamped root included, zero shape).
    Plunge = Uz, twist = Ry (rotation about the span axis Y).
    """
    # gino_name puo' essere un nome semplice ("GOLAND_water") oppure un percorso
    # relativo a output_data ("GOLAND_water/runs/<tag>"): cosi' i run di
    # GOLAND_water scrivono dentro la loro cartella di studio invece di
    # sporcare output_data/ e dover essere spostati a posteriori.
    gino_out = _FSI_ROOT / "output_data" / gino_name
    gino_name = gino_name.rsplit("/", 1)[-1]
    nodes_csv = gino_out / f"{gino_name}_dry_egv_nodes.csv"
    eig_csv = gino_out / f"{gino_name}_dry_egv_eigendata.csv"
    with nodes_csv.open() as f:
        rows = list(csv.DictReader(f))
    pos = np.array([[float(r["X_coord_m"]), float(r["Y_coord_m"]), float(r["Z_coord_m"])]
                    for r in rows])
    n_nodes = pos.shape[0]
    with eig_csv.open() as f:
        erows = list(csv.DictReader(f))
    if len(erows) < nm:
        raise RuntimeError(f"{eig_csv} has {len(erows)} modes < requested {nm}")
    # eigendata columns Node_i_* are the FREE nodes, keyed by the Node_Index of
    # the nodes CSV (node 0 = clamped root, zero shape)
    W = np.zeros((nm, n_nodes))
    TH = np.zeros((nm, n_nodes))
    omega = np.zeros(nm)
    for m in range(nm):
        r = erows[m]
        omega[m] = float(r["Omega_rad_s"])
        for i in range(1, n_nodes):
            W[m, i] = float(r.get(f"Node_{i}_Uz", 0.0))
            TH[m, i] = float(r.get(f"Node_{i}_Ry", 0.0))
        if f"Node_{n_nodes - 1}_Uz" not in r:
            raise RuntimeError("eigendata/nodes CSV node-count mismatch")
    span_axis = int(np.argmax(pos.max(axis=0) - pos.min(axis=0)))
    y = pos[:, span_axis]
    f_hz = omega / (2 * np.pi)
    print(f"[quvlm-goland] GINO modes: span_axis={span_axis}, n_nodes={n_nodes}, "
          f"dry f = {np.array2string(f_hz, precision=3)} Hz")
    return y, W, TH, omega


def main() -> int:
    from gino_bem.mode_alignment import (
        build_goland_congruence,
        goland_twist_sign,
        sharpy_modal_displacements,
        sharpy_modal_matrices,
    )
    from gino_bem.run_goland_water_rho import _build_case
    import sharpy.sharpy_main

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grid", type=str, default="fine", choices=sorted(GRIDS.keys()))
    ap.add_argument("--v-ref", type=float, default=165.0,
                    help="sampling speed [m/s]; Q(k) is V-independent")
    ap.add_argument("--num-modes-gino", type=int, default=2)
    ap.add_argument("--gino-name", type=str, default="GOLAND",
                    help="output_data/<name> dry-mode CSV set to project onto; "
                         "'GOLAND_water' uses the recalibrated xea=0.28/xcm=0.38 axes")
    ap.add_argument("--num-modes-sharpy", type=int, default=8,
                    help="must match official Goland notebook (8 → 4 after remove_sym_modes)")
    ap.add_argument("--k-max", type=float, default=24.0, help="max k_pa = omega/V [1/m]")
    ap.add_argument("--surface-factor", type=float, default=0.5,
                    help="0.5: two mirrored SHARPy surfaces vs GINO half-wing")
    ap.add_argument("--krylov", action="store_true",
                    help="Krylov ROM r=6 @ k=0 (SHARPy flutter parity; not for B2-fit QA)")
    ap.add_argument("--theta-scale", type=float, default=0.9144,
                    help="semichord [m] balancing w vs theta in MAC/LSQ vectors")
    ap.add_argument("--th-sign-sharpy", type=float, default=None, choices=[-1.0, 1.0],
                    help="sign convention of the SHARPy twist extraction relative "
                         "to plunge; default: auto (maximize pairing MAC). Verify "
                         "the choice against the DLM off-diagonal Re signs via "
                         "config.quvlm_compare_dlm on the GINO side.")
    ap.add_argument(
        "--projection",
        type=str,
        default="w_lsq",
        choices=("identity", "w_lsq", "full_lsq"),
        help="GINO←SHARPy congruence: identity=±1 permutation; w_lsq=plunge LSQ "
             "(parity default); full_lsq=legacy [w;θ] LSQ",
    )
    ap.add_argument(
        "--sharpy-structural-basis",
        action="store_true",
        help="Store Q in native SHARPy modal coords (C=I) plus M_hat/K_hat/omega "
             "for GINO config quvlm_sharpy_structural_basis=True",
    )
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--main-ea", type=float, default=0.33,
                    help="SHARPy reference-case elastic axis (chord fraction from LE); "
                         "must match the GINO structural config's xea_factor, else the "
                         "GINO<->SHARPy mode projection mismatches and can collapse "
                         "(see docs/notes/gino_passo0_gaf_table_stale_axes.md)")
    ap.add_argument("--main-cg", type=float, default=0.43,
                    help="SHARPy reference-case CG location (chord fraction from LE); "
                         "should match the GINO structural config's xcm_factor")
    ap.add_argument("--main-chord", type=float, default=1.8288,
                    help="SHARPy reference-case chord [m]; must match the GINO "
                         "structural config's chord, else the reference structure's "
                         "CG-EA offset and the ROM length-scaling both use the wrong "
                         "chord (see run_goland_water_rho._build_case)")
    ap.add_argument("--wing-semispan", type=float, default=6.096,
                    help="SHARPy reference-case semispan [m] (one surface); must "
                         "match the GINO structural config's beam_length")
    ap.add_argument("--sharpy-ei-scale", type=float, default=1.0,
                    help="multiplier on the SHARPy reference case's bending "
                         "stiffness (both planes); must match GINO's EI scale "
                         "or the GINO<->SHARPy mode-shape projection degrades")
    ap.add_argument("--sharpy-gj-scale", type=float, default=1.0,
                    help="multiplier on the SHARPy reference case's torsional "
                         "stiffness; must match GINO's GJ scale")
    ap.add_argument("--sharpy-mu", type=float, default=None,
                    help="SHARPy reference-case mass per unit length [kg/m]; must match "
                         "GINO's mu. The template hardcodes the stock Goland 35.71 -- "
                         "leaving it there while GINO uses a different mass gives the "
                         "two sides different mode ORDERING, which swaps the modal "
                         "pairing and corrupts the per-mode GAF calibration")
    ap.add_argument("--sharpy-i11", type=float, default=None,
                    help="SHARPy reference-case torsional inertia [kg.m]; match GINO i11")
    ap.add_argument("--sharpy-alpha", type=float, default=2.0,
                    help="AoA [deg] of the SHARPy StaticCoupled trim used to build the "
                         "linearisation point. 0.0 linearises about the UNDEFORMED shape "
                         "(what the gravity_on=off comment below intends) and is required "
                         "for very soft wings, whose 2-deg static solve diverges.")
    ap.add_argument("--sharpy-i22", type=float, default=None)
    ap.add_argument("--sharpy-i33", type=float, default=None)
    args = ap.parse_args()

    nm_g = args.num_modes_gino
    nm_s = args.num_modes_sharpy
    if args.sharpy_structural_basis:
        nm_g = nm_s
    M, N, Ms = GRIDS[args.grid]
    V_ref = float(args.v_ref)

    # ---- build the stock SHARPy Goland case, truncated after LinearAssembler --
    route = _SHARPY_ROOT / "gino_bem"
    ws, *_ = _build_case(
        rho=RHO_AIR, u_inf=V_ref, alpha_deg=float(args.sharpy_alpha), M=M, N=N, M_star_fact=Ms,
        num_modes=nm_s, v_min=V_ref, v_max=V_ref, n_vel=1, velocity_list=None,
        route=route, case_tag=f"quvlm_air{args.grid}",
        main_ea=args.main_ea, main_cg=args.main_cg,
        main_chord=args.main_chord, wing_semispan=args.wing_semispan,
        sharpy_ei_scale=args.sharpy_ei_scale, sharpy_gj_scale=args.sharpy_gj_scale,
        sharpy_mu=args.sharpy_mu, sharpy_i11=args.sharpy_i11,
        sharpy_i22=args.sharpy_i22, sharpy_i33=args.sharpy_i33,
    )
    ws.config["SHARPy"]["flow"] = [
        "BeamLoader", "AerogridLoader", "StaticCoupled", "Modal", "LinearAssembler",
    ]
    rom_settings = {
        "algorithm": "mimo_rational_arnoldi",
        "r": 6,
        "single_side": "observability",
        "frequency": np.array([0.0]),
        "tangent_input_file": ws.route + "/" + ws.case_name + ".rom.h5",
    }
    aero_settings = ws.config["LinearAssembler"]["linear_system_settings"]["aero_settings"]
    if args.krylov:
        aero_settings["rom_method"] = ["Krylov"]
        aero_settings["rom_method_settings"] = {"Krylov": rom_settings}
        print(f"[quvlm-goland] UVLM Krylov ROM r={rom_settings['r']} @ k=0")
    else:
        # Full-order ss (no Krylov ROM: its fit around k=0 corrupts the high-k
        # asymptote needed for the B2 fit)
        aero_settings["rom_method"] = []
    # gravity OFF: linearise about the undeformed state. With gravity on the
    # static-load stiffening terms (Kss et al.) are folded into the UVLM
    # feedthrough and contaminate the pure GAF that GINO expects (its DLM GAF
    # carries no static-equilibrium terms).
    ws.config["StaticCoupled"]["structural_solver_settings"]["gravity_on"] = "off"
    ws.config["LinearAssembler"]["linear_system_settings"]["beam_settings"]["gravity"] = "off"
    ws.config.write()
    data = sharpy.sharpy_main.main(["", ws.route + ws.case_name + ".sharpy"])
    ls = data.linear.linear_system

    # remove_sym_modes halves the retained set (2 mirrored surfaces): use the
    # actual modal size of the assembled beam, not the requested one
    nm_s = int(ls.beam.sys.num_modes)
    print(f"[quvlm-goland] SHARPy retained modes after sym reduction: {nm_s}")
    if args.sharpy_structural_basis:
        nm_g = nm_s
        print(f"[quvlm-goland] sharpy_structural_basis: Q and M/K in SHARPy modal coords (nm={nm_g})")

    semichord = float(args.theta_scale)
    m_hat_sh, k_hat_sh, omega_sh = sharpy_modal_matrices(
        ls.beam.sys, nm_s, v_ref=V_ref, semichord=semichord,
    )
    print(
        f"[quvlm-goland] SHARPy modal dry f = "
        f"{np.array2string(omega_sh / (2 * np.pi), precision=3)} Hz"
    )

    # ---- mode pairing + congruence (skipped for identity SHARPy basis) ---------
    ts = float(args.theta_scale)
    C = np.eye(nm_g) if args.sharpy_structural_basis else np.zeros((nm_g, nm_s))
    pair_of: list[tuple[int, int, float]] = []

    if not args.sharpy_structural_basis:
        y_g, W_g, TH_g, omega_g = load_gino_goland_modes(nm_g, gino_name=args.gino_name)
        y_s, W_s, TH_s, _ = sharpy_modal_displacements(data, nm_s)
        half = y_s >= -1e-9
        y_sh, W_sh, TH_sh = y_s[half], W_s[:, half], TH_s[:, half]
        th_sign = goland_twist_sign(
            y_sh, W_sh, TH_sh, y_g, W_g, TH_g,
            theta_scale=ts, forced=args.th_sign_sharpy,
        )
        print(f"[quvlm-goland] SHARPy twist sign convention: {th_sign:+g}")
        TH_sh = th_sign * TH_sh
        C, pair_of = build_goland_congruence(
            y_sh, W_sh, TH_sh, y_g, W_g, TH_g,
            theta_scale=ts, projection=args.projection,
        )
        for g, s, m in pair_of:
            print(
                f"[quvlm-goland] GINO mode {g} <-> SHARPy mode {s}: MAC={m:.3f}, "
                f"c={C[g, s]:+.4f} ({args.projection})"
            )
        order_g = np.argsort(y_g)
        for g, s, m in pair_of:
            wg = np.interp(y_sh, y_g[order_g], W_g[g][order_g])
            tg = np.interp(y_sh, y_g[order_g], TH_g[g][order_g])
            mac_w = (float(np.dot(wg, W_sh[s])) ** 2
                     / max(float(wg @ wg) * float(W_sh[s] @ W_sh[s]), 1e-30))
            mac_t = (float(np.dot(tg, TH_sh[s])) ** 2
                     / max(float(tg @ tg) * float(TH_sh[s] @ TH_sh[s]), 1e-30))
            print(
                f"[quvlm-goland]   pair g{g}-s{s}: MAC_w={mac_w:.3f} MAC_th={mac_t:.3f}"
            )
        if pair_of and min(m for _, _, m in pair_of) < 0.9:
            print("[quvlm-goland] WARNING: a pairing MAC < 0.9 — check mode ordering")

    # ---- frequency response ---------------------------------------------------
    ls.update(V_ref)
    facts = ls.uvlm.sys.ScalingFacts
    scaled = any(abs(float(facts[k]) - 1.0) > 1e-12 for k in ("length", "speed", "density"))
    t_ref = float(facts["length"]) / V_ref if scaled else 1.0
    dimfac = float(facts["length"]) ** 2 if scaled else 1.0
    print(f"[quvlm-goland] ScalingFacts length={facts['length']:g} speed={facts['speed']:g} "
          f"density={facts['density']:g} (scaled={scaled}) dimfac={dimfac:g}")

    k_list = np.unique(np.concatenate([
        [0.0],
        np.linspace(0.01, 1.5, 30),          # flutter band: k ~ 0.45 1/m
        np.linspace(1.6, 8.0, 14),
        np.linspace(8.5, args.k_max, 12),    # high-k asymptote for the B2 fit
    ]))
    w_phys = k_list * V_ref
    w_s = w_phys * t_ref
    dt_s = float(ls.uvlm.ss.dt)
    w_nyq = np.pi / dt_s
    if w_s[-1] > 0.9 * w_nyq:
        keep = w_s <= 0.9 * w_nyq
        print(f"[quvlm-goland] clipping k grid at Nyquist: {k_list[-1]:.1f} -> "
              f"{k_list[keep][-1]:.1f} 1/m")
        k_list, w_phys, w_s = k_list[keep], w_phys[keep], w_s[keep]

    # locate the modal input/output blocks by name (the control-surface template
    # carries extra inputs: cs deflection and rate)
    def _var_slice(varlist, name):
        for v in varlist:
            if v.name == name:
                return slice(v.first_position, v.end_position)
        raise RuntimeError(f"variable {name!r} not in {[v.name for v in varlist]}")

    sl_q = _var_slice(ls.uvlm.ss.input_variables, "q")
    sl_qd = _var_slice(ls.uvlm.ss.input_variables, "q_dot")
    sl_Q = _var_slice(ls.uvlm.ss.output_variables, "Q")
    print(f"[quvlm-goland] uvlm.ss states={ls.uvlm.ss.states}, "
          f"inputs={[v.name for v in ls.uvlm.ss.input_variables]}, "
          f"q={sl_q}, q_dot={sl_qd}, out Q={sl_Q}; "
          f"sampling {len(k_list)} k-points at V_ref={V_ref:g} m/s ...", flush=True)
    H = ls.uvlm.ss.freqresp(w_s)             # (n_out, n_in, n_k)

    sf = float(args.surface_factor)
    if args.sharpy_structural_basis:
        sf = 1.0
        print("[quvlm-goland] sharpy_structural_basis: surface_factor=1 (native sym-reduced SHARPy modes)")
    Q = np.zeros((len(k_list), nm_g, nm_g), dtype=complex)
    for i, wsc in enumerate(w_s):
        q_sh = H[sl_Q, sl_q, i] + 1j * wsc * H[sl_Q, sl_qd, i]
        q_dim = np.conj(dimfac * q_sh)
        if args.sharpy_structural_basis:
            Q[i] = q_dim[:nm_g, :nm_g] * sf
        else:
            Q[i] = np.conj(C @ q_dim @ C.T) * sf

    k_hi = k_list[-1]
    B2_diag = np.diag(-np.real(Q[-1]) / k_hi ** 2)
    print(f"[quvlm-goland] B2 diagonal at k={k_hi:.1f}: "
          f"{np.array2string(B2_diag, precision=5)} "
          "(expect NEGATIVE: UVLM k^2 term signed opposite to DLM)")
    i_lo = int(np.argmin(np.abs(k_list - 0.4)))
    print(f"[quvlm-goland] Q(k={k_list[i_lo]:.2f}) =")
    print(np.array2string(Q[i_lo], precision=4, suppress_small=True))

    axes_tag = "" if args.gino_name == "GOLAND" else f"_{args.gino_name.lower()}axes"
    geom_tag = ""
    if abs(args.main_chord - 1.8288) > 1e-9 or abs(args.wing_semispan - 6.096) > 1e-9:
        geom_tag = f"_c{args.main_chord:g}_s{args.wing_semispan:g}".replace(".", "p")
    if abs(args.sharpy_ei_scale - 1.0) > 1e-9 or abs(args.sharpy_gj_scale - 1.0) > 1e-9:
        geom_tag += f"_ei{args.sharpy_ei_scale:g}_gj{args.sharpy_gj_scale:g}".replace(".", "p")
    if args.sharpy_mu is not None:
        # different reference mass -> different modes -> different GAF; must not
        # share a cache entry with the stock-mass table
        geom_tag += f"_mu{args.sharpy_mu:g}".replace(".", "p")
    out = args.out or (_FSI_ROOT / "FLUID" / "PanelAero" / "Qjj" /
                       f"Quvlm_GOLAND_air_M{M}N{N}Ms{Ms}_nm{nm_g}"
                       f"{axes_tag}{geom_tag}"
                       f"{'_krylov' if args.krylov else ''}"
                       f"{'_sharpy_basis' if args.sharpy_structural_basis else ''}.npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    save_kw = dict(
        k_list=k_list, Q=Q,
        V_ref_m_s=V_ref, rho=RHO_AIR, qinf_ref=0.5 * RHO_AIR * V_ref ** 2,
        grid=f"M{M}N{N}Ms{Ms}", case="goland_air", nm=nm_g, C=C,
        gino_modes_source=args.gino_name,
        surface_factor=sf, dt_scaled=dt_s, n_states=ls.uvlm.ss.states,
        num_modes_sharpy=nm_s, krylov=bool(args.krylov), krylov_r=6 if args.krylov else 0,
        projection=str(args.projection),
        sharpy_structural_basis=bool(args.sharpy_structural_basis),
        M_hat_sharpy=m_hat_sh[:nm_g, :nm_g],
        K_hat_sharpy=k_hat_sh[:nm_g, :nm_g],
        omega_rad_sharpy=omega_sh[:nm_g],
        convention=("F_modal = qinf * Q(k_pa) * q_modal ; k_pa = omega/V [1/m]; "
                    "raw (pre aero_real_flip/aero_imag_flip) like DLM _Qg_raw; "
                    "surface_factor folds 2 mirrored SHARPy surfaces to the "
                    "GINO half-wing"),
    )
    np.savez(out, **save_kw)
    print(f"[quvlm-goland] saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
