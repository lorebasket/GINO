#!/usr/bin/env python3
"""Export the raw SHARPy linear-UVLM aerodynamic ROM for Model2t (water, config G)
-- the state matrices themselves (A, B, C, D), not a frequency-sampled Q(k)
table -- for the "native coupling" method in
``COUPLING.hydroelastic_utl.native_state_coupling``.

Model2t twin of ``precompute_uvlm_rom_goland.py``. Same case construction as
``run_model2t_water.py`` (production grid "n30c12" = M12N30Ms6, Krylov r=8,
single real shift k=0 -- confirmed elsewhere that Krylov ROM == full-order
for Model2t). ``gj_factor``/``ei_factor`` only affect the trim/linearization
point through StaticCoupled; gravity is off (undeformed-shape linearization,
matching Goland), so the nominal (1.0) stiffness is used here -- the aero
ROM itself does not depend on beam stiffness, only on geometry/grid. GINO's
own (SONATA-eventually) modal M_hat/K_hat, calibrated however needed, is
supplied separately at coupling time.

Single surface (no GolandControlSurface-style mirroring) -> remove_sym_modes
is off in this case, so num_modes requested == modes retained (no /2 as in
the Goland template).

  conda activate sharpy
  cd sharpy
  python gino_bem/precompute_uvlm_rom_model2t.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_SHARPY_ROOT = Path(__file__).resolve().parents[1]
_FSI_ROOT = _SHARPY_ROOT.parents[1]
if str(_SHARPY_ROOT) not in sys.path:
    sys.path.insert(0, str(_SHARPY_ROOT))

RHO_WATER = 997.0  # fresh water, matches run_model2t_water.py / Capytaine data
M, N, MS = 12, 30, 6          # "n30c12" production grid
NUM_MODES = 4
ROM_R = 8


def main() -> int:
    from gino_bem.model2t_strut import Model2tStrut
    from gino_bem.run_model2t_water import _configure_sharpy_case
    import sharpy.sharpy_main

    route = _SHARPY_ROOT / "gino_bem"
    case_name = f"uvlmrom_model2t_water_M{M}N{N}Ms{MS}_nm{NUM_MODES}"

    ws = Model2tStrut(
        M=M, N=N, Mstar_fact=MS,
        u_inf=1.0, alpha=0.0, rho=RHO_WATER,
        gj_factor=1.0, ei_factor=1.0,
        route=str(route / "cases"), case_name=case_name,
    )
    ws.clean_test_files()
    ws.update_derived_params()
    ws.set_default_config_dict()
    ws.generate_aero_file()
    ws.generate_fem_file()

    rom_settings = {"algorithm": "mimo_rational_arnoldi", "r": ROM_R, "single_side": "observability"}
    _configure_sharpy_case(
        ws, num_modes=NUM_MODES, velocity_list=None, v_min=1.0, v_max=1.0, n_vel=1,
        out_root=route / "output", rom_settings=rom_settings, u_inf=1.0, k_points=[0.0],
    )
    # ROM export only needs LinearAssembler -- skip AsymptoticStability.
    ws.config["SHARPy"]["flow"] = [
        "BeamLoader", "AerogridLoader", "StaticCoupled", "Modal", "LinearAssembler",
    ]
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
    print(f"[uvlm-rom-model2t] SHARPy retained modes: {nm_s} (no sym reduction)")
    print(f"[uvlm-rom-model2t] aero ROM: states={A.shape[0]}, nm={nm}, "
          f"dt_nondim={dt_nondim:g}, length_ref={length_ref:g} (semichord)")
    print(f"[uvlm-rom-model2t] K12 shape={K12.shape}, K21 diag={np.diag(K21)}")

    out = _FSI_ROOT / "FLUID" / "PanelAero" / "Qjj" / f"Uvlm_ROM_Model2t_water_M{M}N{N}Ms{MS}_nm{nm}_krylov.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        A=A, B_in=B_in, C_out=C_out, D_out=D_out,
        dt_nondim=dt_nondim, length_ref=length_ref, V_construction=1.0,
        nm=nm, n_states=A.shape[0], K12=K12, K21=K21,
        grid=f"M{M}N{N}Ms{MS}", num_modes_sharpy=nm_s,
    )
    print(f"[uvlm-rom-model2t] saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
