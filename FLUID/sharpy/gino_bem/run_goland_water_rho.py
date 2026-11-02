#!/usr/bin/env python3
"""
Stock SHARPy Goland flutter with a different fluid density (no BEM, no GINO).

Only ``rho`` (and the velocity sweep window) change vs the official Goland example.
Use this to see what UVLM predicts when the fluid is as dense as water (ρ ≈ 997 kg/m³).

  conda activate sharpy
  cd sharpy
  python gino_bem/run_goland_water_rho.py
  python gino_bem/run_goland_water_rho.py --rho 1.02 --v-max 180   # air reference
  python gino_bem/run_goland_water_rho.py --rho 997 --v-max 30 --coarse

Outputs under ``gino_bem/output/<case_name>/``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

_SHARPY_ROOT = Path(__file__).resolve().parents[1]
if str(_SHARPY_ROOT) not in sys.path:
    sys.path.insert(0, str(_SHARPY_ROOT))

import sharpy.cases.templates.flying_wings as wings  # noqa: E402
import csv

import sharpy.sharpy_main  # noqa: E402
from gino_bem.rayleigh_damping import (  # noqa: E402
    install_rayleigh_structural_damping,
    last_rayleigh_fit,
)

RHO_AIR = 1.02
RHO_WATER = 997.0
KN_TO_MS = 0.514444


def _goland_velocity_list_air() -> np.ndarray:
    """Dense sweep around Goland flutter (~166 m/s)."""
    return np.unique(
        np.concatenate(
            [
                np.linspace(120.0, 155.0, 25),
                np.linspace(155.0, 175.0, 45),
                np.linspace(175.0, 185.0, 15),
            ]
        )
    )


def _build_case(
    *,
    rho: float,
    u_inf: float,
    alpha_deg: float,
    M: int,
    N: int,
    M_star_fact: int,
    num_modes: int,
    v_min: float,
    v_max: float,
    n_vel: int,
    velocity_list: np.ndarray | None,
    route: Path,
    case_tag: str,
    use_krylov: bool = True,
    newmark_damp: float = 0.5e-4,
    main_ea: float = 0.33,
    main_cg: float = 0.43,
    main_chord: float = 1.8288,
    wing_semispan: float = 6.096,
    sharpy_ei_scale: float = 1.0,
    sharpy_gj_scale: float = 1.0,
    sharpy_mu: float | None = None,
    sharpy_i11: float | None = None,
    sharpy_i22: float | None = None,
    sharpy_i33: float | None = None,
):
    rom_settings = {
        "algorithm": "mimo_rational_arnoldi",
        "r": 6,
        "single_side": "observability",
    }
    frequency_continuous_k = np.array([0.0])
    case_name = f"goland_{case_tag}_M{M}N{N}Ms{M_star_fact}_nm{num_modes}"
    if not use_krylov:
        case_name += "_fulluvlm"

    ws = wings.GolandControlSurface(
        M=M,
        N=N,
        Mstar_fact=M_star_fact,
        u_inf=u_inf,
        alpha=alpha_deg,
        cs_deflection=[0, 0],
        rho=rho,
        sweep=0,
        physical_time=2,
        n_surfaces=2,
        main_chord=float(main_chord),
        # Goland/GolandControlSurface derive wing_span = aspect_ratio * main_chord
        # (default aspect_ratio is a fixed literal tied to the stock 1.8288 m
        # chord / 6.096 m semispan, NOT recomputed from main_chord) -- pass it
        # explicitly so a custom chord doesn't silently drag the span with it.
        aspect_ratio=(2.0 * float(wing_semispan)) / float(main_chord),
        route=str(route / "cases"),
        case_name=case_name,
    )
    # GolandControlSurface.__init__ hardcodes main_ea=0.33/main_cg=0.43
    # (stock air-Goland axes); override here so the SHARPy reference case's
    # elastic axis / CG offset matches the GINO structural config instead of
    # silently drifting from it (docs/notes/gino_passo0_gaf_table_stale_axes.md,
    # 2026-09-03: mismatch collapsed the mode-1 GINO<->SHARPy projection when
    # GINO's xea moved away from 0.33 without SHARPy's reference following).
    ws.main_ea = float(main_ea)
    ws.main_cg = float(main_cg)
    # GolandControlSurface.__init__ also hardcodes c_ref=1.8288 as a bare
    # literal (NOT tied to main_chord) -- it feeds the aeroelastic ROM's
    # length scaling (ScalingDict["length"] below) AND, inside
    # update_mass_stiff(), the reference structural model's own CG-EA
    # offset (self.c_ref * (main_cg - main_ea)). Left stale, a custom
    # main_chord would non-dimensionalise the linear system at the wrong
    # length AND build the SHARPy-side reference structure with the old
    # chord's offset -- exactly the kind of mismatch that collapsed the
    # GINO<->SHARPy projection in the main_ea/main_cg case above, here via
    # chord instead of axis fraction.
    ws.c_ref = float(main_chord)
    ws.clean_test_files()
    ws.update_derived_params()
    if sharpy_ei_scale != 1.0 or sharpy_gj_scale != 1.0:
        # update_mass_stiff() (called by update_derived_params() above) hardcodes
        # gj=0.987581e6/eiy=9.77221e6 (stock air-Goland section) -- if GINO's
        # own EIxx/GJ are scaled down (e.g. to hit a target dry frequency) and
        # the SHARPy reference structure is left at stock stiffness, the two
        # sides' mode SHAPES/ordering drift apart and the GINO<->SHARPy modal
        # projection MAC degrades or collapses (2026-09-21: seen dropping to
        # ~0.01-0.23 on the untouched higher modes when GINO's EI was
        # softened ~160x for a chord=0.3/span=1.5 case). Rescale the SHARPy
        # reference section stiffness to match: index 3 = GJ (torsion),
        # index 4 = EIy (the bending plane paired with GJ), index 5 = EIz
        # (the other bending plane; GINO's EIzz tracks its EIxx 1:1 too).
        ws.stiffness[:, 3, 3] *= float(sharpy_gj_scale)
        ws.stiffness[:, 4, 4] *= float(sharpy_ei_scale)
        ws.stiffness[:, 5, 5] *= float(sharpy_ei_scale)
    if sharpy_mu is not None:
        # update_mass_stiff() also hardcodes the stock air-Goland section MASS
        # (m_unit=35.71 kg/m, j_tors=8.64 kg.m). Syncing only the stiffness leaves the
        # two sides with different mass distributions, hence different mode shapes AND
        # ordering. Measured 2026-09-29 with GINO on physical aluminum mass (mu=72.14,
        # i11=1.019 -> SHARPy's torsional inertia 8.5x larger): the GINO<->SHARPy
        # pairing came out SWAPPED (g0<->s1, g1<->s0) with MAC 0.39-0.80 and a per-mode
        # calibration coefficient of +39.5 on the torsion mode, which drove that root to
        # zero in the p-k spectrum -- no torsion root, so no flutter was possible for
        # ANY structure. Mirror GINO's mass here. Frame: xb spans, so index 3 is torsion.
        from sharpy.utils import algebra as _algebra

        m = float(sharpy_mu)
        i11 = float(sharpy_i11) if sharpy_i11 is not None else ws.mass[0, 3, 3]
        i22 = float(sharpy_i22) if sharpy_i22 is not None else 0.1 * i11
        i33 = float(sharpy_i33) if sharpy_i33 is not None else 0.9 * i11
        pos_cg_b = np.array([0.0, ws.c_ref * (ws.main_cg - ws.main_ea), 0.0])
        m_chi_cg = _algebra.skew(m * pos_cg_b)
        ws.mass = np.zeros((1, 6, 6))
        ws.mass[0, :, :] = np.diag([m, m, m, i11, i22, i33])
        ws.mass[0, :3, 3:] = m_chi_cg
        ws.mass[0, 3:, :3] = -m_chi_cg
        ws.elem_mass = np.zeros((ws.num_elem_tot,), dtype=int)
        print(f"[sharpy-mass-sync] mu={m:.4f} i11={i11:.6f} i22={i22:.6f} i33={i33:.6f}")
    ws.set_default_config_dict()

    frequency_continuous_w = 2 * u_inf * frequency_continuous_k / ws.c_ref
    rom_settings["frequency"] = frequency_continuous_w
    rom_settings["tangent_input_file"] = ws.route + "/" + ws.case_name + ".rom.h5"

    ws.generate_aero_file()
    ws.generate_fem_file()

    out_root = route / "output"
    ws.config["SHARPy"] = {
        "flow": [
            "BeamLoader",
            "AerogridLoader",
            "StaticCoupled",
            "Modal",
            "LinearAssembler",
            "AsymptoticStability",
        ],
        "case": ws.case_name,
        "route": ws.route,
        "write_screen": "on",
        "write_log": "on",
        "log_folder": str(out_root) + "/",
        "log_file": ws.case_name + ".log",
    }

    ws.config["BeamLoader"] = {"unsteady": "off", "orientation": ws.quat}
    ws.config["AerogridLoader"] = {
        "unsteady": "off",
        "aligned_grid": "on",
        "mstar": ws.Mstar_fact * ws.M,
        "freestream_dir": ws.u_inf_direction,
        "wake_shape_generator": "StraightWake",
        "wake_shape_generator_input": {
            "u_inf": ws.u_inf,
            "u_inf_direction": ws.u_inf_direction,
            "dt": ws.dt,
        },
    }
    ws.config["StaticCoupled"] = {
        "print_info": "on",
        "max_iter": 200,
        "n_load_steps": 1,
        "tolerance": 1e-10,
        "relaxation_factor": 0.0,
        "aero_solver": "StaticUvlm",
        "aero_solver_settings": {
            "rho": ws.rho,
            "print_info": "off",
            "horseshoe": "off",
            "num_cores": 4,
            "n_rollup": 0,
            "rollup_dt": ws.dt,
            "rollup_aic_refresh": 1,
            "rollup_tolerance": 1e-4,
            "velocity_field_generator": "SteadyVelocityField",
            "velocity_field_input": {
                "u_inf": ws.u_inf,
                "u_inf_direction": ws.u_inf_direction,
            },
        },
        "structural_solver": "NonLinearStatic",
        "structural_solver_settings": {
            "print_info": "off",
            "max_iterations": 150,
            "num_load_steps": 4,
            "delta_curved": 1e-1,
            "min_delta": 1e-10,
            "gravity_on": "on",
            "gravity": 9.81,
        },
    }
    ws.config["Modal"] = {
        "NumLambda": 20,
        "rigid_body_modes": "off",
        "print_matrices": "on",
        "save_data": "on",
        "rigid_modes_cg": "off",
        "continuous_eigenvalues": "off",
        "dt": 0,
        "plot_eigenvalues": False,
        "max_rotation_deg": 15.0,
        "max_displacement": 0.15,
        "write_modes_vtk": False,
        "use_undamped_modes": True,
    }
    ws.config["LinearAssembler"] = {
        "linear_system": "LinearAeroelastic",
        "linear_system_settings": {
            "beam_settings": {
                "modal_projection": "on",
                "inout_coords": "modes",
                "discrete_time": "on",
                "newmark_damp": float(newmark_damp),
                "discr_method": "newmark",
                "dt": ws.dt,
                "proj_modes": "undamped",
                "use_euler": "off",
                "num_modes": num_modes,
                "print_info": "on",
                "gravity": "on",
                "remove_sym_modes": "on",
                "remove_dofs": [],
            },
            "aero_settings": {
                "dt": ws.dt,
                "ScalingDict": {
                    "length": 0.5 * ws.c_ref,
                    "speed": u_inf,
                    "density": rho,
                },
                "integr_order": 2,
                "density": rho,
                "remove_predictor": "on",
                "use_sparse": "on",
                "remove_inputs": ["u_gust"],
                "rom_method": (["Krylov"] if use_krylov else []),
                "rom_method_settings": ({"Krylov": rom_settings} if use_krylov else {}),
            },
        },
    }
    ws.config["AsymptoticStability"] = {
        "print_info": True,
        "velocity_analysis": [v_min, v_max, n_vel],
        "velocity_list": velocity_list.tolist() if velocity_list is not None else [],
        "export_eigenvalues": True,
        "display_root_locus": False,
        "num_evals": max(24, 6 * num_modes),
    }
    ws.config.write()
    return ws, v_min, v_max, n_vel


def _find_velocity_dat(case_out: Path) -> Path | None:
    stability = case_out / "stability"
    if not stability.is_dir():
        return None
    files = sorted(stability.glob("velocity_analysis_*.dat"), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def _parse_velocity_dat(path: Path) -> dict:
    res = np.loadtxt(path)
    if res.ndim == 1:
        res = res.reshape(1, -1)
    u = res[:, 0]
    re = res[:, 1]
    im = res[:, 2]
    u_unique = np.unique(u)
    rows = []
    for ui in u_unique:
        mask = np.isclose(u, ui)
        re_u = re[mask]
        im_u = im[mask]
        for j in range(len(re_u)):
            rows.append(
                {
                    "V_m_s": float(ui),
                    "V_kn": float(ui) / KN_TO_MS,
                    "mode_index": j,
                    "real": float(re_u[j]),
                    "imag": float(im_u[j]),
                    "freq_Hz": float(abs(im_u[j]) / (2.0 * np.pi)),
                }
            )
    return {"rows": rows}


def _parse_flutter(velocity_dat: Path) -> dict:
    res = np.loadtxt(velocity_dat)
    if res.ndim == 1:
        res = res.reshape(1, -1)
    u = res[:, 0]
    re = res[:, 1]
    im = res[:, 2]
    # Track mode with largest real part at highest V (crude flutter estimate)
    n_modes = len(u) // len(np.unique(u)) if len(u) else 0
    if n_modes < 1:
        return {"flutter_V": None, "note": "empty velocity file"}
    u_unique = np.unique(u)
    re_end = re[-n_modes:]
    im_end = im[-n_modes:]
    i_crit = int(np.argmax(re_end))
    return {
        "V_at_max_real": float(u_unique[-1]),
        "max_real_part": float(re_end[i_crit]),
        "omega_at_max_real": float(np.hypot(re_end[i_crit], im_end[i_crit])),
        "freq_Hz_at_max_real": float(np.hypot(re_end[i_crit], im_end[i_crit]) / (2 * np.pi)),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Goland flutter — stock SHARPy, variable rho")
    p.add_argument("--rho", type=float, default=RHO_WATER, help=f"fluid density [kg/m³] (water={RHO_WATER})")
    p.add_argument("--v-min", type=float, default=None, help="velocity sweep min [m/s]")
    p.add_argument("--v-max", type=float, default=None, help="velocity sweep max [m/s]")
    p.add_argument("--n-vel", type=int, default=25, help="number of velocities in sweep")
    p.add_argument("--coarse", action="store_true", help="M=4 N=8 (fast test)")
    p.add_argument("--fine", action="store_true", help="M=16 N=32 (slow, literature grid)")
    p.add_argument("--uniform-sweep", action="store_true", help="linspace v-min..v-max instead of dense list (air)")
    p.add_argument("--no-krylov", action="store_true", help="full-order UVLM (no Krylov ROM)")
    p.add_argument("--rayleigh", action="store_true", help="GINO-matched Rayleigh C on modes 0,1")
    p.add_argument("--zeta-bend", type=float, default=0.0, help="Rayleigh zeta on bending mode 0")
    p.add_argument("--zeta-tors", type=float, default=0.0, help="Rayleigh zeta on torsion mode 1")
    p.add_argument("--newmark-damp", type=float, default=0.5e-4, help="Newmark beam damping (no --rayleigh)")
    p.add_argument(
        "--out-subdir",
        type=str,
        default="sharpy_uvlm",
        help="subfolder under output_data/GOLAND for CSV export",
    )
    args = p.parse_args()

    rho = float(args.rho)
    if args.coarse:
        M, N, M_star_fact, num_modes = 4, 8, 4, 4
    elif args.fine:
        M, N, M_star_fact, num_modes = 16, 32, 10, 8
    else:
        M, N, M_star_fact, num_modes = 8, 16, 6, 4

    if args.v_min is None:
        v_min = 3.0 if rho > 100 else 120.0
    else:
        v_min = args.v_min
    if args.v_max is None:
        v_max = 35.0 if rho > 100 else 185.0
    else:
        v_max = args.v_max

    velocity_list = None
    if rho <= 100 and not args.uniform_sweep:
        velocity_list = _goland_velocity_list_air()
        v_min = float(np.min(velocity_list))
        v_max = float(np.max(velocity_list))
        n_vel_use = len(velocity_list)
    else:
        n_vel_use = args.n_vel

    tag = "water" if rho > 100 else "air"
    route = _SHARPY_ROOT / "gino_bem"
    route.mkdir(parents=True, exist_ok=True)
    (route / "cases").mkdir(exist_ok=True)
    (route / "output").mkdir(exist_ok=True)

    print(f"SHARPy root: {_SHARPY_ROOT}")
    print(f"rho = {rho} kg/m³  ({tag})")
    print(f"grid M={M} N={N}  velocity [{v_min}, {v_max}] m/s  n={n_vel_use}")

    ws, vl, vu, nv = _build_case(
        rho=rho,
        u_inf=1.0,
        alpha_deg=2.0,
        M=M,
        N=N,
        M_star_fact=M_star_fact,
        num_modes=num_modes,
        v_min=v_min,
        v_max=v_max,
        n_vel=n_vel_use,
        velocity_list=velocity_list,
        route=route,
        case_tag=tag,
        use_krylov=not args.no_krylov,
        newmark_damp=float(args.newmark_damp),
    )

    rayleigh_zetas = (float(args.zeta_bend), float(args.zeta_tors))
    if args.rayleigh:
        install_rayleigh_structural_damping(
            mode_ids=(0, 1),
            target_zetas=rayleigh_zetas,
        )
        print(
            f"  Rayleigh structural damping: "
            f"ζ_bend={rayleigh_zetas[0]:.4g}, ζ_tors={rayleigh_zetas[1]:.4g} (modes 0,1)"
        )

    sharpy.sharpy_main.main(["", ws.route + ws.case_name + ".sharpy"])

    case_out = route / "output" / ws.case_name
    vel_file = _find_velocity_dat(case_out)
    parsed = _parse_velocity_dat(vel_file) if vel_file and vel_file.is_file() else {}
    summary = _parse_flutter(vel_file) if vel_file and vel_file.is_file() else {"error": str(vel_file)}

    out_data = _SHARPY_ROOT.parents[1] / "output_data" / "GOLAND" / args.out_subdir
    out_data.mkdir(parents=True, exist_ok=True)
    csv_path = out_data / f"{ws.case_name}_velocity_eigs.csv"
    if parsed.get("rows"):
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(parsed["rows"][0].keys()))
            w.writeheader()
            w.writerows(parsed["rows"])

    fit = last_rayleigh_fit() if args.rayleigh else None

    print("\n--- Summary ---")
    print(f"Case: {ws.case_name}")
    print(f"Velocity analysis file: {vel_file}")
    if args.rayleigh and fit:
        print(
            f"  Rayleigh: alpha={fit['alpha']:.4e}, beta={fit['beta']:.4e}, "
            f"zeta_bend={rayleigh_zetas[0]:.4g}, zeta_tors={rayleigh_zetas[1]:.4g}"
        )
    else:
        print(f"  Newmark damp: {args.newmark_damp:.4e}")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    if parsed.get("rows"):
        print(f"CSV: {csv_path}")
        print(f"Plot: python gino_bem/plot_sharpy_vgvf.py --raw-modes --csv {csv_path}")
    print(
        "\nNote: this is UVLM with higher rho only — not a submerged hydrofoil BEM model. "
        "Compare qualitatively with GINO, not Abramson experiment."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
