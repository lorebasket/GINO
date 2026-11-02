#!/usr/bin/env python3
"""
Besch & Liu Model 2T (config G) — stock SHARPy (UVLM + GEBM), rho = 997, no BEM.

Mirrors gino_bem/run_abramson_water.py. Tip pod enters structurally only
(lumped mass); UVLM cannot model the pod hydrodynamics or the free surface.

  conda activate sharpy
  cd sharpy
  python gino_bem/run_model2t_water.py
  python gino_bem/run_model2t_water.py --coarse
  python gino_bem/run_model2t_water.py --gj-factor 1.0   # nominal (uncalibrated) GJ
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

from gino_bem.model2t_strut import GJ_FACTOR_HEADLINE, Model2tStrut  # noqa: E402
from gino_bem.rayleigh_damping import (  # noqa: E402
    install_rayleigh_structural_damping,
    last_rayleigh_fit,
)
import sharpy.sharpy_main  # noqa: E402

RHO_WATER = 997.0
V_EXP_KN = 18.1  # Besch & Liu Fig. 11b flutter point (config G)
F_EXP_HZ = 6.4
KN_TO_MS = 0.514444

# GINO Model2t baseline (config_Model2T.py): zeta=0.05 on the first two modes
MODEL2T_RAYLEIGH_MODE_IDS = (0, 1)
MODEL2T_RAYLEIGH_ZETAS = (0.05, 0.05)


def _model2t_velocity_list_m_s(*, v_max_m_s: float = 20.58) -> np.ndarray:
    """GINO sweep is linspace(0.5, 20.58, 80) = 0-40 kn (Besch & Liu Fig. 11).
    Denser around the experimental flutter point (~9.3 m/s); extendable."""
    parts = [
        np.linspace(0.5, 6.0, 15),
        np.linspace(6.0, 13.0, 45),
        np.linspace(13.0, min(20.58, v_max_m_s), 25),
    ]
    if v_max_m_s > 20.58:
        parts.append(np.linspace(20.58, v_max_m_s, 30))
    return np.unique(np.concatenate(parts))


def _configure_sharpy_case(
    ws: Model2tStrut,
    *,
    num_modes: int,
    velocity_list: np.ndarray | None,
    v_min: float,
    v_max: float,
    n_vel: int,
    out_root: Path,
    rom_settings: dict,
    u_inf: float,
    k_points: list[float],
):
    # Krylov ROM shifts: real axis at the reduced frequencies (see the long
    # comment in run_abramson_water.py — mimo_rational_arnoldi validates the
    # basis with a transpose check, so imaginary shifts deflate spuriously).
    if rom_settings is not None:
        rom_settings = dict(rom_settings)
        if len(k_points) == 1 and float(k_points[0]) == 0.0:
            rom_settings["frequency"] = np.array([0.0])
        else:
            rom_settings["frequency"] = [float(k) for k in k_points]
        rom_settings["tangent_input_file"] = ws.route + "/" + ws.case_name + ".rom.h5"

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
        "n_load_steps": 4,
        "tolerance": 1e-8,
        "relaxation_factor": 0.7,
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
            "gravity_on": "off",
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
        "write_modes_vtk": True,
        "use_undamped_modes": True,
    }
    ws.config["LinearAssembler"] = {
        "linear_system": "LinearAeroelastic",
        "linear_system_settings": {
            "beam_settings": {
                "modal_projection": "on",
                "inout_coords": "modes",
                "discrete_time": "on",
                "newmark_damp": 0.5e-4,
                "discr_method": "newmark",
                "dt": ws.dt,
                "proj_modes": "undamped",
                "use_euler": "off",
                "num_modes": num_modes,
                "print_info": "on",
                "gravity": "off",
                "remove_sym_modes": "off",
                "remove_dofs": [],
            },
            "aero_settings": {
                "dt": ws.dt,
                "ScalingDict": {
                    "length": 0.5 * ws.c_ref,
                    "speed": u_inf,
                    "density": ws.rho,
                },
                "integr_order": 2,
                "density": ws.rho,
                "remove_predictor": "on",
                "use_sparse": "on",
                "remove_inputs": ["u_gust"],
            },
        },
    }
    if rom_settings is not None:
        aero = ws.config["LinearAssembler"]["linear_system_settings"]["aero_settings"]
        aero["rom_method"] = ["Krylov"]
        aero["rom_method_settings"] = {"Krylov": rom_settings}
    ws.config["AsymptoticStability"] = {
        "print_info": True,
        "velocity_analysis": [v_min, v_max, n_vel],
        "velocity_list": velocity_list.tolist() if velocity_list is not None else [],
        "export_eigenvalues": True,
        "display_root_locus": False,
        "num_evals": max(24, 6 * num_modes),
    }
    ws.config.write()


def _read_modal_frequencies_hz(modal_dir: Path) -> list[float]:
    p = modal_dir / "frequencies.dat"
    if not p.is_file():
        return []
    omega = np.loadtxt(p)
    omega = np.atleast_1d(omega).astype(float)
    return sorted((omega / (2.0 * np.pi)).tolist())[:6]


def _parse_velocity_dat(path: Path) -> dict:
    res = np.loadtxt(path)
    if res.ndim == 1:
        res = res.reshape(1, -1)
    u = res[:, 0]
    re = res[:, 1]
    im = res[:, 2]
    u_unique = np.unique(u)
    n_modes = len(u) // len(u_unique) if len(u_unique) else 0
    if n_modes < 1:
        return {}
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
    by_v = {}
    for r in rows:
        v = r["V_m_s"]
        by_v.setdefault(v, []).append(r)
    v_sorted = sorted(by_v)
    flutter_V = None
    for v in v_sorted:
        if max(x["real"] for x in by_v[v]) >= 0.0:
            flutter_V = v
            break
    return {
        "rows": rows,
        "flutter_V_m_s": flutter_V,
        "flutter_V_kn": None if flutter_V is None else flutter_V / KN_TO_MS,
    }


def _find_velocity_dat(case_out: Path) -> Path | None:
    stability = case_out / "stability"
    if not stability.is_dir():
        return None
    files = sorted(stability.glob("velocity_analysis_*.dat"), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def main() -> int:
    p = argparse.ArgumentParser(description="Model 2T SHARPy UVLM, rho water, no BEM")
    p.add_argument("--rho", type=float, default=RHO_WATER)
    p.add_argument("--v-min", type=float, default=0.5, help="m/s (used if --uniform-sweep)")
    p.add_argument("--v-max", type=float, default=20.58, help="m/s (20.58 = 40 kn, Fig. 11 range)")
    p.add_argument("--n-vel", type=int, default=90)
    p.add_argument("--uniform-sweep", action="store_true", help="linspace v-min..v-max instead of GINO-like list")
    p.add_argument("--coarse", action="store_true")
    p.add_argument("--fine", action="store_true")
    p.add_argument("--grid-m", type=int, default=None, help="chordwise panels M (overrides presets)")
    p.add_argument("--grid-n", type=int, default=None, help="spanwise panels N, multiple of 10 (overrides presets)")
    p.add_argument("--grid-ms", type=int, default=None, help="wake length factor Mstar_fact (overrides presets)")
    p.add_argument(
        "--k-points",
        type=str,
        default=None,
        help="Comma-separated reduced frequencies for a multi-point Krylov ROM, "
        "e.g. '0,0.3,0.6,0.9'. k=0 is always included. Default: single point k=0.",
    )
    p.add_argument("--rom-r", type=int, default=8, help="Krylov moments matched per interpolation point")
    p.add_argument(
        "--no-rom",
        action="store_true",
        help="Skip the Krylov ROM and eigen-solve the full-order linear UVLM.",
    )
    p.add_argument(
        "--gj-factor",
        type=float,
        default=GJ_FACTOR_HEADLINE,
        help=f"GJ calibration factor (default {GJ_FACTOR_HEADLINE} = GINO headline "
        "wet-1T calibration; use 1.0 for the nominal Besch & Liu value)",
    )
    p.add_argument("--ei-factor", type=float, default=1.0, help="EI calibration factor (default 1.0)")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Summary/CSV output directory "
        "(default: output_data/Model2t/sharpy_uvlm_water)",
    )
    p.add_argument("--num-modes", type=int, default=4, help="modes retained in the linear beam")
    p.add_argument(
        "--rayleigh",
        action="store_true",
        default=True,
        help="Apply GINO-matched Rayleigh structural damping (default: on)",
    )
    p.add_argument(
        "--no-rayleigh",
        action="store_false",
        dest="rayleigh",
        help="Disable structural Rayleigh damping (undamped beam)",
    )
    p.add_argument(
        "--zeta-1",
        type=float,
        default=MODEL2T_RAYLEIGH_ZETAS[0],
        help="Target damping ratio on mode 1 (default: 0.05, GINO Model2t)",
    )
    p.add_argument(
        "--zeta-2",
        type=float,
        default=MODEL2T_RAYLEIGH_ZETAS[1],
        help="Target damping ratio on mode 2 (default: 0.05, GINO Model2t)",
    )
    args = p.parse_args()

    # N multiple of 10 puts a node at s=0.2524 m ~ the 0.254 m waterline arc.
    if args.coarse:
        M, N, M_star_fact = 6, 10, 4
    elif args.fine:
        M, N, M_star_fact = 12, 30, 8
    else:
        M, N, M_star_fact = 8, 20, 6
    if args.grid_m is not None:
        M = int(args.grid_m)
    if args.grid_n is not None:
        N = int(args.grid_n)
    if args.grid_ms is not None:
        M_star_fact = int(args.grid_ms)
    if N % 10:
        raise SystemExit(f"--grid-n must be a multiple of 10 (waterline node + even elements), got {N}")
    num_modes = int(args.num_modes)

    if args.k_points:
        k_pts = sorted({round(float(x), 6) for x in args.k_points.split(",") if x.strip() != ""})
        if 0.0 not in k_pts:
            k_pts = [0.0] + k_pts
        multi_k = len(k_pts) > 1
    else:
        k_pts = [0.0]
        multi_k = False

    velocity_list = None if args.uniform_sweep else _model2t_velocity_list_m_s(v_max_m_s=float(args.v_max))
    v_min = float(args.v_min if args.uniform_sweep else np.min(velocity_list))
    v_max = float(args.v_max if args.uniform_sweep else np.max(velocity_list))
    n_vel = int(args.n_vel if args.uniform_sweep else len(velocity_list))

    route = _SHARPY_ROOT / "gino_bem"
    (route / "cases").mkdir(parents=True, exist_ok=True)
    out_root = route / "output"
    out_root.mkdir(exist_ok=True)

    rom_settings = {
        "algorithm": "mimo_rational_arnoldi",
        "r": args.rom_r,
        "single_side": "observability",
    }
    if args.no_rom:
        rom_settings = None

    case_name = f"model2t_water_M{M}N{N}Ms{M_star_fact}_nm{num_modes}"
    if abs(args.gj_factor - 1.0) > 1e-12:
        case_name += f"_gj{args.gj_factor:.4f}".replace(".", "p")
    if abs(args.ei_factor - 1.0) > 1e-12:
        case_name += f"_ei{args.ei_factor:.4f}".replace(".", "p")
    if args.no_rom:
        case_name += "_fulluvlm"
    elif multi_k:
        case_name += f"_mk{len(k_pts)}r{args.rom_r}"
    if args.rayleigh:
        case_name += "_rayleigh"

    ws = Model2tStrut(
        M=M,
        N=N,
        Mstar_fact=M_star_fact,
        u_inf=1.0,
        alpha=0.0,
        rho=float(args.rho),
        gj_factor=float(args.gj_factor),
        ei_factor=float(args.ei_factor),
        route=str(route / "cases"),
        case_name=case_name,
    )
    ws.clean_test_files()
    ws.update_derived_params()
    ws.set_default_config_dict()
    ws.generate_aero_file()
    ws.generate_fem_file()

    _configure_sharpy_case(
        ws,
        num_modes=num_modes,
        velocity_list=velocity_list,
        v_min=v_min,
        v_max=v_max,
        n_vel=n_vel,
        out_root=out_root,
        rom_settings=rom_settings,
        u_inf=1.0,
        k_points=k_pts,
    )

    rayleigh_zetas = (float(args.zeta_1), float(args.zeta_2))
    if args.rayleigh:
        install_rayleigh_structural_damping(
            mode_ids=MODEL2T_RAYLEIGH_MODE_IDS,
            target_zetas=rayleigh_zetas,
        )
        print(
            f"  Rayleigh structural damping: "
            f"zeta1={rayleigh_zetas[0]:.4g}, zeta2={rayleigh_zetas[1]:.4g} "
            f"(modes {list(MODEL2T_RAYLEIGH_MODE_IDS)})"
        )

    n_wet = int(np.count_nonzero(ws.aero_node))
    print(f"Running SHARPy Model 2T case: {case_name}")
    print(
        f"  strut: span {ws.span:.4f} m (EA, 15 deg sweep), chord {ws.c_ref:.4f} m, "
        f"EA@{ws.main_ea:.3f}c, CG@{ws.main_cg:.3f}c"
    )
    print(
        f"  GJ factor {args.gj_factor} -> GJ={args.gj_factor * 5940.9:.1f} N m^2, "
        f"EI factor {args.ei_factor} -> EI={args.ei_factor * 3710.91:.1f} N m^2"
    )
    print(
        f"  wetted span: {n_wet}/{ws.num_node_tot} aero nodes "
        f"(dry root {ws.wetted_from_root:.3f} m masked); pod = structural lumped mass only"
    )
    if args.no_rom:
        print("  Full-order linear UVLM (no Krylov ROM)")
    else:
        print(f"  Krylov ROM: k={k_pts}, r={args.rom_r} moments each (real shifts)")
    print(
        f"  rho={args.rho} kg/m^3  V=[{v_min:.2f}, {v_max:.2f}] m/s  "
        f"({v_min / KN_TO_MS:.1f}-{v_max / KN_TO_MS:.1f} kn)  n={n_vel}"
    )
    sharpy.sharpy_main.main(["", ws.route + ws.case_name + ".sharpy"])

    case_out = out_root / case_name
    vel_file = _find_velocity_dat(case_out)
    parsed = _parse_velocity_dat(vel_file) if vel_file and vel_file.is_file() else {}
    modal_hz = _read_modal_frequencies_hz(case_out / "beam_modal_analysis")

    summary_dir = args.out_dir or (_FSI_ROOT / "output_data" / "Model2t" / "sharpy_uvlm_water")
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = summary_dir / f"{case_name}_velocity_eigs.csv"
    if parsed.get("rows"):
        with summary_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(parsed["rows"][0].keys()))
            w.writeheader()
            w.writerows(parsed["rows"])

    report = summary_dir / f"{case_name}_summary.txt"
    fit = last_rayleigh_fit() if args.rayleigh else None
    lines = [
        f"case: {case_name}",
        f"rho: {args.rho} kg/m^3 (stock SHARPy UVLM, no BEM, no free surface, pod structural only)",
        f"stiffness: GJ factor {args.gj_factor}, EI factor {args.ei_factor}",
        (
            f"structural Rayleigh: zeta1={rayleigh_zetas[0]:.4g}, "
            f"zeta2={rayleigh_zetas[1]:.4g}"
            + (
                f"  alpha={fit['alpha']:.4e}, beta={fit['beta']:.4e}"
                if fit
                else "  (fit pending — AsymptoticStability did not assemble beam)"
            )
            if args.rayleigh
            else "structural Rayleigh: off (undamped beam, C=0)"
        ),
        (
            "aero reduction: full-order linear UVLM (no ROM)"
            if args.no_rom
            else f"Krylov ROM interpolation points (reduced freq k): {k_pts}  (r={args.rom_r})"
        ),
        f"velocity file: {vel_file}",
        f"experimental: V_f = {V_EXP_KN} kn ({V_EXP_KN * KN_TO_MS:.2f} m/s), f = {F_EXP_HZ} Hz "
        "(Besch & Liu Fig. 11b, config G)",
        f"GINO headline (pk_capytaine, GJ 0.4953): V_f = 18.76 kn",
        f"dry modal frequencies (Hz) SHARPy structural: {modal_hz}  "
        "(GINO dry refs: 1T ~11.3, 2B ~25; 1B lower)",
        f"flutter estimate (UVLM): V = {parsed.get('flutter_V_kn')} kn ({parsed.get('flutter_V_m_s')} m/s)",
        "",
        "Compare with GINO: output_data/Model2t/ (pk_capytaine baselines)",
    ]
    report.write_text("\n".join(lines), encoding="utf-8")

    print("\n--- Summary ---")
    for line in lines:
        print(line)
    print(f"\nCSV: {summary_csv if parsed.get('rows') else '(no velocity file)'}")
    print(f"Report: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
