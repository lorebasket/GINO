#!/usr/bin/env python3
"""Export the raw (Krylov-reduced) SHARPy linear-UVLM aerodynamic ROM for GOLAND
(air) -- the state matrices themselves (A, B, C, D), not a frequency-sampled
Q(k) table -- for the "native coupling" method in
``COUPLING.hydroelastic_utl.native_state_coupling`` (no p-k, no Roger refit:
couples this ROM directly with an arbitrary modal M_hat/K_hat via the exact
same ``lingebm.newmark_ss`` + ``libss.couple`` recipe SHARPy uses internally).

Twin of ``precompute_quvlm_goland.py`` (same case construction: same grid,
same num_modes request -> sym-reduced count, same Krylov ROM), but exports
the ROM's own state matrices plus the (fixed, velocity-independent)
aero<->structure coupling gains Tas/Tsa, instead of sampling Q(k).

  conda activate sharpy
  cd sharpy
  python gino_bem/precompute_uvlm_rom_goland.py                 # fine grid M16N32Ms10
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

RHO_AIR = 1.02
GRIDS = {
    "coarse": (4, 8, 4),
    "default": (8, 16, 6),
    "fine": (16, 32, 10),
}


def main() -> int:
    from gino_bem.run_goland_water_rho import _build_case
    import sharpy.sharpy_main

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grid", type=str, default="fine", choices=sorted(GRIDS.keys()))
    ap.add_argument("--v-ref", type=float, default=165.0,
                    help="reference velocity for the static trim / ROM construction "
                         "[m/s]; the ROM's own dynamics are V-independent (nondimensional)")
    ap.add_argument("--num-modes-sharpy", type=int, default=8,
                    help="requested modes before remove_sym_modes (matches Quvlm build)")
    ap.add_argument("--newmark-damp", type=float, default=0.5e-4)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    M, N, Ms = GRIDS[args.grid]
    V_ref = float(args.v_ref)
    route = _SHARPY_ROOT / "gino_bem"
    ws, *_ = _build_case(
        rho=RHO_AIR, u_inf=V_ref, alpha_deg=2.0, M=M, N=N, M_star_fact=Ms,
        num_modes=args.num_modes_sharpy, v_min=V_ref, v_max=V_ref, n_vel=1,
        velocity_list=None, route=route, case_tag=f"uvlmrom_air{args.grid}",
        newmark_damp=float(args.newmark_damp),
    )
    # ROM export only needs LinearAssembler -- skip AsymptoticStability.
    ws.config["SHARPy"]["flow"] = [
        "BeamLoader", "AerogridLoader", "StaticCoupled", "Modal", "LinearAssembler",
    ]
    ws.config["StaticCoupled"]["structural_solver_settings"]["gravity_on"] = "off"
    ws.config["LinearAssembler"]["linear_system_settings"]["beam_settings"]["gravity"] = "off"
    ws.config.write()
    data = sharpy.sharpy_main.main(["", ws.route + ws.case_name + ".sharpy"])
    ls = data.linear.linear_system

    def _var_slice(varlist, name):
        for v in varlist:
            if v.name == name:
                return slice(v.first_position, v.end_position)
        raise RuntimeError(f"variable {name!r} not in {[v.name for v in varlist]}")

    sl_q = _var_slice(ls.uvlm.ss.input_variables, "q")
    sl_qd = _var_slice(ls.uvlm.ss.input_variables, "q_dot")
    sl_Q = _var_slice(ls.uvlm.ss.output_variables, "Q")
    nm = int(sl_q.stop - sl_q.start)

    A, B_full, C_full, D_full = ls.uvlm.ss.get_mats()
    for name, mat in (("A", A), ("B", B_full), ("C", C_full), ("D", D_full)):
        mat = np.asarray(mat)
        if np.iscomplexobj(mat):
            im_mag = np.max(np.abs(mat.imag)) if mat.size else 0.0
            if im_mag > 1e-8 * max(np.max(np.abs(mat.real)), 1e-30):
                raise RuntimeError(
                    f"UVLM ROM matrix {name!r} has non-negligible imaginary part "
                    f"(max|Im|={im_mag:.3e}); real-form assumption violated."
                )

    A = np.asarray(A, dtype=float)
    B_in = np.asarray(np.asarray(B_full)[:, sl_q.start:sl_qd.stop], dtype=float)
    C_out = np.asarray(np.asarray(C_full)[sl_Q, :], dtype=float)
    D_out = np.asarray(np.asarray(D_full)[sl_Q, sl_q.start:sl_qd.stop], dtype=float)
    dt_nondim = float(ls.uvlm.ss.dt)
    length_ref = float(ls.uvlm.sys.ScalingFacts["length"])

    Tas = ls.couplings["Tas"]
    Tsa = ls.couplings["Tsa"]
    Tas = np.asarray(Tas.value if hasattr(Tas, "value") else Tas, dtype=float)
    Tsa = np.asarray(Tsa.value if hasattr(Tsa, "value") else Tsa, dtype=float)
    K12 = Tas[: 2 * nm, :]
    K21 = Tsa

    nm_s = int(ls.beam.sys.num_modes)
    print(f"[uvlm-rom-goland] SHARPy retained modes after sym reduction: {nm_s}")
    print(f"[uvlm-rom-goland] aero ROM: states={A.shape[0]}, nm={nm}, "
          f"dt_nondim={dt_nondim:g}, length_ref={length_ref:g}, V_construction={V_ref:g}")
    print(f"[uvlm-rom-goland] K12 shape={K12.shape}, K21 diag={np.diag(K21)}")

    out = args.out or (
        _FSI_ROOT / "FLUID" / "PanelAero" / "Qjj"
        / f"Uvlm_ROM_GOLAND_air_M{M}N{N}Ms{Ms}_nm{nm}_krylov.npz"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        A=A, B_in=B_in, C_out=C_out, D_out=D_out,
        dt_nondim=dt_nondim, length_ref=length_ref, V_construction=V_ref,
        nm=nm, n_states=A.shape[0], K12=K12, K21=K21,
        grid=f"M{M}N{N}Ms{Ms}", num_modes_sharpy=nm_s,
    )
    print(f"[uvlm-rom-goland] saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
