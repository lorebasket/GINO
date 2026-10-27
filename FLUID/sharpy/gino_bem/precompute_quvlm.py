#!/usr/bin/env python3
"""Sample the SHARPy linear-UVLM modal GAF Q(k) for GINO's p-k (aero_method='UVLM').

GINO's whole downstream stack (p-k, Capytaine injection, tracking) consumes a
modal GAF ``Q(k_pa)`` with ``k_pa = omega/V`` [1/m] such that the harmonic
generalized aero force is ``F_modal = qinf * Q(k_pa) * q``. The linear UVLM is
LTI and incompressible, so its GAF depends on k only: sample it once at a
reference speed and reuse at every velocity.

    Q_sharpy(i w) = H_q(i w_s) + i w_s H_qdot(i w_s)     [ls.uvlm.ss freqresp]
    Q_gino(k)     = T @ Q_sharpy(w = k V_ref) @ T.T * (F_ref / qinf_ref)

``T`` is the signed permutation aligning SHARPy modes to the GINO/Capytaine
source modal basis (same basis the pk_capytaine A(omega),B(omega) live in).
``F_ref`` dimensionalizes the scaled ss outputs (ScalingFacts). The table is
mode-shape-dependent: regenerate after structural changes.

  conda activate sharpy
  cd sharpy
  python gino_bem/precompute_quvlm.py --case abramson --grid n12c16 --v-ref-kn 48.1
  python gino_bem/precompute_quvlm.py --case model2t --grid n30c12 --v-ref-kn 28.4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_SHARPY_ROOT = Path(__file__).resolve().parents[1]
_FSI_ROOT = _SHARPY_ROOT.parents[1]
if str(_SHARPY_ROOT) not in sys.path:
    sys.path.insert(0, str(_SHARPY_ROOT))

KN = 0.514444
RHO_WATER = 997.0

_CASE_SPECS = {
    "abramson": {
        "module": "gino_bem.run_abramson_pk_bem",
        "default_grid": "n12c16",
        "default_v_ref_kn": 48.1,
        "out_name": "Quvlm_ABRAMSON1965_water_{grid}_nm{nm}.npz",
        "capy_default": "default_capytaine_npz",
    },
    "ac38": {
        "module": "gino_bem.run_ac38_pk_bem",
        "default_grid": "n40c12",
        "default_v_ref_kn": 30.0,
        "out_name": "Quvlm_AC38_RUDDER_water_{grid}_nm{nm}.npz",
        "capy_default": "default_ac38_capy_npz",
    },
    # I_B_500 (RevD): stessa macchina, altra struttura e soprattutto altra
    # superficie portante — tutta quella bagnata, 0.834 m2 invece di 0.269.
    "ac38_ib500": {
        "module": "gino_bem.run_ac38_ib500_pk_bem",
        "default_grid": "n48c14",
        "default_v_ref_kn": 30.0,
        "out_name": "Quvlm_AC38_RUDDER_IB500_water_{grid}_nm{nm}.npz",
        "capy_default": "default_ac38_capy_npz",
    },
    "model2t": {
        "module": "gino_bem.run_model2t_pk_bem",
        "default_grid": "n30c12",
        "default_v_ref_kn": 28.4,
        "out_name": "Quvlm_Model2t_water_{grid}_nm{nm}.npz",
        "capy_default": "default_model2t_capy_npz",
    },
}


def _load_case(case: str):
    import importlib

    spec = _CASE_SPECS[case]
    mod = importlib.import_module(spec["module"])
    from gino_bem.capy_bem_bridge import check_capy_basis, load_capytaine_npz

    capy_fn = getattr(mod, spec["capy_default"])
    return mod.GRIDS, mod.build_case, capy_fn, check_capy_basis, load_capytaine_npz, spec


def main() -> int:
    from gino_bem.mode_alignment import (
        DEFAULT_THETA_SCALE,
        align_capy_to_sharpy,
        alignment_matrix,
        gino_modal_displacements,
        sharpy_modal_displacements,
    )

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", type=str, default="abramson", choices=sorted(_CASE_SPECS.keys()))
    ap.add_argument("--grid", type=str, default=None,
                    help="aerogrid tag (case-specific; see GRIDS in the case runner)")
    ap.add_argument("--v-ref-kn", type=float, default=None,
                    help="sampling speed [kn]; Q(k) is V-independent, this only sets "
                         "the physical omega = k*V_ref at which the ss is evaluated")
    ap.add_argument("--num-modes", type=int, default=4)
    ap.add_argument("--sharpy-modes", type=str, default=None,
                    help="indici (0-based) dei modi di SHARPy da usare, es '0,2,3,5'. "
                         "Il ROM si costruisce su tutti i modi fino al massimo "
                         "indicato, ma proiezione e allineamento usano solo questi: "
                         "serve quando la base di GINO salta dei modi (p.es. i "
                         "flessionali nel piano della corda, invisibili in "
                         "affondamento e torsione, che rendono il MAC degenere).")
    ap.add_argument("--k-max", type=float, default=40.0, help="max k_pa = omega/V [1/m]")
    ap.add_argument("--n-k", type=int, default=81)
    ap.add_argument("--capy-npz", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    GRIDS, build_case, default_capy_npz, check_capy_basis, load_capytaine_npz, spec = _load_case(args.case)
    grid_tag = args.grid or spec["default_grid"]
    if grid_tag not in GRIDS:
        raise SystemExit(f"--grid {grid_tag!r} not in {args.case} GRIDS: {sorted(GRIDS)}")
    v_ref_kn = float(args.v_ref_kn if args.v_ref_kn is not None else spec["default_v_ref_kn"])

    nm = args.num_modes
    sh_idx = ([int(x) for x in args.sharpy_modes.split(",")]
              if args.sharpy_modes else None)
    if sh_idx is not None:
        if len(sh_idx) != nm:
            raise SystemExit(f"--sharpy-modes ha {len(sh_idx)} indici ma "
                             f"--num-modes = {nm}: devono coincidere")
        nm_rom = max(sh_idx) + 1
    else:
        nm_rom = nm
    grid = GRIDS[grid_tag]
    V_ref = v_ref_kn * KN
    qinf_ref = 0.5 * RHO_WATER * V_ref ** 2

    capy = check_capy_basis(load_capytaine_npz(args.capy_npz or default_capy_npz()))

    data, _ = build_case(
        nm_rom, grid=grid, added_mass_removal="none",
        rayleigh=False, zetas=(0.0, 0.0), v_max_kn=max(v_ref_kn, 60.0),
        case_tag=f"quvlm_{args.case}_{grid_tag}",
    )
    ls = data.linear.linear_system
    if sh_idx is None:
        T, _report = align_capy_to_sharpy(capy, data, nm)
        y_s, W_s, TH_s, _f_s = sharpy_modal_displacements(data, nm)
    else:
        # Allineamento sul SOTTOINSIEME: i modi che la base di GINO non porta
        # restano fuori dalla tabella MAC, invece di comparirvi con |W| ~ 0 e
        # correlazioni spurie. Con tutti e sei i modi la riga del torsionale
        # preferiva la colonna di un flessionale in corda (MAC 0.89 contro 0.65)
        # e l'allineamento non produceva piu' una permutazione.
        y_s, W_s, TH_s, f_s = sharpy_modal_displacements(data, nm_rom)
        y_g, W_g, TH_g, f_g = gino_modal_displacements(capy)
        si = np.asarray(sh_idx, dtype=int)
        print(f"[quvlm] modi SHARPy usati: {sh_idx} di {nm_rom} nel ROM")
        T, _report = alignment_matrix(y_s, W_s[si], TH_s[si], f_s[si],
                                      y_g, W_g[:nm], TH_g[:nm], f_g[:nm])
        W_s, TH_s = W_s[si], TH_s[si]
    T = np.asarray(T, dtype=float)[:nm, :nm]

    # ---- per-mode projection normalization ----------------------------------
    # SHARPy and GINO are both mass-normalized in their own FE, but the same
    # physical shape carries slightly different modal amplitudes in each basis
    # (different discretizations). Enforce q_sharpy = c_g * q_gino by LSQ of the
    # GINO shape onto the paired SHARPy shape: c_g = <v_s, v_g>/<v_s, v_s>
    # (sign included — supersedes T's signs). Then Q_gino = C Q_sharpy C^T with
    # C[g, s(g)] = c_g. Verified against DLM: bare signed-permutation congruence
    # leaves a separable d_i*d_j ~ +/-11% residual (r00*r11 = r01*r10 to 0.3%).
    y_g, W_g, TH_g, _ = gino_modal_displacements(capy)
    order_g = np.argsort(y_g)
    C = np.zeros((nm, nm))
    for g in range(nm):
        s = int(np.argmax(np.abs(T[g, :])))          # sharpy partner of gino mode g
        v_s = np.concatenate([W_s[s], DEFAULT_THETA_SCALE * TH_s[s]])
        wg = np.interp(y_s, y_g[order_g], W_g[g][order_g])
        tg = np.interp(y_s, y_g[order_g], TH_g[g][order_g])
        v_g = np.concatenate([wg, DEFAULT_THETA_SCALE * tg])
        c_g = float(v_s @ v_g) / max(float(v_s @ v_s), 1e-30)
        C[g, s] = c_g
    print(f"[quvlm] projection coefficients c (gino mode <- sharpy partner): "
          f"{np.array2string(np.array([C[g].sum() for g in range(nm)]), precision=4)} "
          "(expect ~+/-1; deviation = cross-basis normalization)")

    ls.update(float(V_ref))
    facts = ls.uvlm.sys.ScalingFacts
    scaled = any(abs(float(facts[key]) - 1.0) > 1e-12 for key in ("length", "speed", "density"))
    F_ref = float(facts["dyn_pressure"]) * float(facts["length"]) ** 2 if scaled else 1.0
    t_ref = float(facts["length"]) / float(V_ref) if scaled else 1.0
    print(f"[quvlm] ScalingFacts: length={facts['length']:g} speed={facts['speed']:g} "
          f"density={facts['density']:g} -> F_ref={F_ref:g}, t_ref={t_ref:g} s (scaled={scaled})")

    # k grid: dense at low k (circulatory lag) + coverage to k_max for the B2 fit
    k_list = np.unique(np.concatenate([
        [0.0],
        np.linspace(0.02, 2.0, 34),
        np.linspace(2.2, 8.0, 22),
        np.linspace(8.5, args.k_max, args.n_k - 57),
    ]))
    w_phys = k_list * V_ref                      # rad/s
    w_s = w_phys * t_ref                         # scaled rad/s

    dt_s = float(ls.uvlm.ss.dt)
    w_nyq = np.pi / dt_s
    if w_s[-1] > 0.9 * w_nyq:
        keep = w_s <= 0.9 * w_nyq
        print(f"[quvlm] WARNING: clipping k grid at Nyquist: k_max {k_list[-1]:.1f} -> "
              f"{k_list[keep][-1]:.1f} 1/m (w_s_nyq={w_nyq:.2f}, dt_s={dt_s:.4f})")
        k_list, w_phys, w_s = k_list[keep], w_phys[keep], w_s[keep]

    in_names = [v.name for v in ls.uvlm.ss.input_variables]
    print(f"[quvlm] uvlm.ss inputs={in_names} states={ls.uvlm.ss.states}; "
          f"sampling {len(k_list)} k-points at V_ref={v_ref_kn} kn ...", flush=True)
    H = ls.uvlm.ss.freqresp(w_s)                 # (n_out, n_in, n_k)
    if H.shape[0] < nm_rom or H.shape[1] < 2 * nm_rom:
        raise RuntimeError(f"unexpected uvlm.ss dims {H.shape}; expected >=({nm_rom},{2*nm_rom}) "
                           "— check modal projection")

    # NORMALIZATION (calibrated against the DLM Qjj path via quvlm_compare_dlm):
    #  * dimensionalization = ScalingFacts length^2 (the assembly's dyn_pressure
    #    factor already absorbs 0.5*rho with speed fact 1; F_ref/dyn_pressure);
    #  * conj(): SHARPy freqresp uses the opposite harmonic convention to
    #    PanelAero — conjugation matches Re AND Im signs of the DLM raw GAF
    #    (element [0,1] Im keeps an opposite sign: UVLM/DLM physics difference
    #    in the lift-lag term, not a convention residual);
    #  * C congruence maps to the GINO modal basis incl. cross-basis mode
    #    normalization. Result matches DLM _Qg_raw in the CIRCULATORY content
    #    (Re and Im signs at low k); the UVLM's own k^2 apparent-mass term comes
    #    out with the OPPOSITE sign to the DLM's (Re>0 at high k) — the b2 fit
    #    B2 = -Re(Q(kb2))/kb2^2 cancels it regardless of sign, so aero_b2_removal
    #    (mandatory for this provider) works as-is; without b2removal this table
    #    would ADD apparent mass where DLM subtracts it.
    dimfac = F_ref / float(facts["dyn_pressure"]) if scaled else 1.0
    Q = np.zeros((len(k_list), nm, nm), dtype=complex)
    _ix = (np.ix_(np.asarray(sh_idx, dtype=int), np.asarray(sh_idx, dtype=int))
           if sh_idx is not None else None)
    for i, ws in enumerate(w_s):
        Q_sh = (H[:nm_rom, :nm_rom, i]
                + 1j * ws * H[:nm_rom, nm_rom:2 * nm_rom, i])     # per modal q, scaled force
        if _ix is not None:
            Q_sh = Q_sh[_ix]
        Q[i] = np.conj(C @ (dimfac * Q_sh) @ C.T)                 # GINO basis, DLM raw sign

    # ---- diagnostics --------------------------------------------------------
    # B2 fit exactly like the GINO b2removal: B2 = -Re(Q(k_hi))/k_hi^2. Expect
    # NEGATIVE diagonal for this table (UVLM k^2 term signed opposite to DLM's,
    # see normalization note); |B2| should be within ~a few x of the DLM ref
    # [5.095e-04, 9.206e-04, 1.746e-04, 1.805e-04] (n16-chord quartic table).
    k_hi = k_list[-1]
    B2_diag = np.diag(-np.real(Q[-1]) / k_hi ** 2)
    print(f"[quvlm] B2 diagonal at k={k_hi:.1f}: {np.array2string(B2_diag, precision=6)} "
          "(|.| vs DLM ref [5.1e-04, 9.2e-04, 1.7e-04, 1.8e-04])")
    i_lo = int(np.argmin(np.abs(k_list - 0.5)))
    print(f"[quvlm] Q(k={k_list[i_lo]:.2f}) =")
    print(np.array2string(Q[i_lo], precision=4, suppress_small=True))

    out = args.out or (_FSI_ROOT / "FLUID" / "PanelAero" / "Qjj" /
                       spec["out_name"].format(grid=grid_tag, nm=nm))
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        k_list=k_list, Q=Q,
        V_ref_m_s=V_ref, rho=RHO_WATER, qinf_ref=qinf_ref, F_ref=F_ref,
        grid=str(grid_tag), case=args.case, nm=nm, T=T,
        source_capy_npz=str(args.capy_npz or default_capy_npz()),
        dt_scaled=dt_s, n_states=ls.uvlm.ss.states,
        convention=("F_modal = qinf * Q(k_pa) * q_gino ; k_pa = omega/V [1/m]; "
                    "raw (pre aero_real_flip/aero_imag_flip) like DLM _Qg_raw"),
    )
    print(f"[quvlm] saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
