#!/usr/bin/env python3
"""
Abramson 1965 — stock SHARPy (UVLM + GEBM), rho = 997, no BEM.

  conda activate sharpy
  cd sharpy
  python gino_bem/run_abramson_water.py
  python gino_bem/run_abramson_water.py --coarse
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

from gino_bem.abramson1965_wing import Abramson1965  # noqa: E402
from gino_bem.rayleigh_damping import (  # noqa: E402
    ABRAMSON_RAYLEIGH_MODE_IDS,
    ABRAMSON_RAYLEIGH_ZETAS,
    install_rayleigh_structural_damping,
    last_rayleigh_fit,
)
import sharpy.sharpy_main  # noqa: E402

RHO_WATER = 997.0
V_EXP_KN = 48.1
F_EXP_HZ = 17.5
KN_TO_MS = 0.514444


def _abramson_velocity_list_m_s(*, v_max_m_s: float = 32.0) -> np.ndarray:
    """Dense sweep aligned with GINO Abramson config; extend with ``v_max_m_s`` [m/s]."""
    parts = [
        np.linspace(0.5, 5.0, 10),
        np.linspace(5.0, 18.0, 52),
        np.linspace(18.0, 28.0, 65),
        np.linspace(28.0, min(32.0, v_max_m_s), 25),
    ]
    if v_max_m_s > 32.0:
        parts.append(np.linspace(32.0, min(v_max_m_s, 42.0), 40))
    if v_max_m_s > 42.0:
        parts.append(np.linspace(42.0, min(v_max_m_s, 55.0), 35))
    if v_max_m_s > 55.0:
        parts.append(np.linspace(55.0, v_max_m_s, 40))
    return np.unique(np.concatenate(parts))


def _kn_to_m_s(v_kn: float) -> float:
    return float(v_kn) * KN_TO_MS


def _configure_sharpy_case(
    ws: Abramson1965,
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
    # --- Krylov ROM interpolation points ---------------------------------
    # The ROM is built on the *non-dimensionalised* UVLM system (ScalingDict
    # active below: length=b, speed=u_inf, density=rho).  In that scaled system
    # the continuous-time frequency axis IS the reduced frequency
    #     k = omega * b / U          (b = semichord = 0.5 * c_ref),
    # because time is scaled by ScalingFacts['time'] = length/speed = b/U.
    # We place the interpolation shifts on the REAL axis at magnitudes equal to
    # the target reduced frequencies:  s0 = k.
    #
    # Why real and not imaginary (s0 = i*k, the "physical" harmonic point)?
    # SHARPy's mimo_rational_arnoldi validates the Krylov basis with
    # check_eye(w, w.T) -- transpose, NOT conjugate-transpose. For real shifts
    # the basis is real and w.T == w^H, so the check passes. For imaginary
    # shifts the basis is complex, orthonormality is Hermitian (w^H w = I), and
    # w.T w != I makes the check fail spuriously -> the space deflates to rank 1
    # and the reduction aborts. Real-shift rational Krylov still broadens the
    # frequency coverage across [0, max(k)] vs a single expansion at k=0.
    #
    # NB: the legacy single-point formula  w = 2*u_inf*k/c_ref  returns the
    # *physical* angular frequency and is only correct when scaling is OFF.
    # It is exact for k = 0 (both give 0), which is why the dormant mismatch
    # never showed up in the validated single-point run.
    if rom_settings is not None:
        rom_settings = dict(rom_settings)
        if len(k_points) == 1 and float(k_points[0]) == 0.0:
            # Legacy validated path: keep identical real-array serialization "[0.]".
            rom_settings["frequency"] = np.array([0.0])
        else:
            # Multi-point real shifts; comma-separated list serializes cleanly.
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
                # ROM keys injected below only when use_rom is True.
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
    """SHARPy Modal ``frequencies.dat`` (rad/s) → Hz."""
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
    # Flutter heuristic: first V where any mode has real >= 0 (track mode 0 or max real at each V)
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
    return {"rows": rows, "flutter_V_m_s": flutter_V, "flutter_V_kn": None if flutter_V is None else flutter_V / KN_TO_MS}


def _find_velocity_dat(case_out: Path) -> Path | None:
    stability = case_out / "stability"
    if not stability.is_dir():
        return None
    files = sorted(stability.glob("velocity_analysis_*.dat"), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def main() -> int:
    p = argparse.ArgumentParser(description="Abramson 1965 SHARPy UVLM, rho water, no BEM")
    p.add_argument("--rho", type=float, default=RHO_WATER)
    p.add_argument("--v-min", type=float, default=0.5, help="m/s (used if --uniform-sweep)")
    p.add_argument("--v-max", type=float, default=32.0)
    p.add_argument("--n-vel", type=int, default=150)
    p.add_argument("--uniform-sweep", action="store_true", help="linspace v-min..v-max instead of GINO-like list")
    p.add_argument("--coarse", action="store_true")
    p.add_argument("--fine", action="store_true")
    p.add_argument(
        "--k-points",
        type=str,
        default=None,
        help="Comma-separated reduced frequencies for a multi-point Krylov ROM, "
        "e.g. '0,0.3,0.6,0.9'. k=0 is always included. Default: single point k=0 (legacy).",
    )
    p.add_argument("--rom-r", type=int, default=8, help="Krylov moments matched per interpolation point")
    p.add_argument(
        "--no-rom",
        action="store_true",
        help="Skip the Krylov ROM and eigen-solve the full-order linear UVLM "
        "(exact at all k, no moment-matching artifacts). Best for coarse grids.",
    )
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
        help="Disable structural Rayleigh damping (legacy undamped beam)",
    )
    p.add_argument(
        "--zeta-bend",
        type=float,
        default=ABRAMSON_RAYLEIGH_ZETAS[0],
        help="Target damping ratio on 1st bending mode (default: 0.076)",
    )
    p.add_argument(
        "--zeta-tors",
        type=float,
        default=ABRAMSON_RAYLEIGH_ZETAS[1],
        help="Target damping ratio on 1st torsion mode (default: 0.039)",
    )
    args = p.parse_args()

    if args.coarse:
        M, N, M_star_fact, num_modes = 6, 12, 4, 4
    elif args.fine:
        M, N, M_star_fact, num_modes = 12, 24, 8, 4
    else:
        M, N, M_star_fact, num_modes = 8, 16, 6, 4

    # ROM interpolation points (reduced frequency). k=0 anchors the steady point.
    if args.k_points:
        k_pts = sorted({round(float(x), 6) for x in args.k_points.split(",") if x.strip() != ""})
        if 0.0 not in k_pts:
            k_pts = [0.0] + k_pts
        multi_k = len(k_pts) > 1
    else:
        k_pts = [0.0]
        multi_k = False

    velocity_list = None if args.uniform_sweep else _abramson_velocity_list_m_s()
    v_min = float(args.v_min if args.uniform_sweep else np.min(velocity_list))
    v_max = float(args.v_max if args.uniform_sweep else np.max(velocity_list))
    n_vel = int(args.n_vel if args.uniform_sweep else len(velocity_list))

    route = _SHARPY_ROOT / "gino_bem"
    route.mkdir(parents=True, exist_ok=True)
    (route / "cases").mkdir(exist_ok=True)
    out_root = route / "output"
    out_root.mkdir(exist_ok=True)

    rom_settings = {
        "algorithm": "mimo_rational_arnoldi",
        "r": args.rom_r,
        "single_side": "observability",
    }
    if args.no_rom:
        rom_settings = None  # full-order linear UVLM, no reduction
    case_name = f"abramson_water_M{M}N{N}Ms{M_star_fact}_nm{num_modes}"
    if args.no_rom:
        case_name += "_fulluvlm"
    elif multi_k:
        case_name += f"_mk{len(k_pts)}r{args.rom_r}"
    if args.rayleigh:
        case_name += "_rayleigh"

    ws = Abramson1965(
        M=M,
        N=N,
        Mstar_fact=M_star_fact,
        u_inf=1.0,
        alpha=0.0,
        rho=float(args.rho),
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

    rayleigh_zetas = (float(args.zeta_bend), float(args.zeta_tors))
    if args.rayleigh:
        install_rayleigh_structural_damping(
            mode_ids=ABRAMSON_RAYLEIGH_MODE_IDS,
            target_zetas=rayleigh_zetas,
        )
        print(
            f"  Rayleigh structural damping: "
            f"ζ_bend={rayleigh_zetas[0]:.4g}, ζ_tors={rayleigh_zetas[1]:.4g} "
            f"(modes {list(ABRAMSON_RAYLEIGH_MODE_IDS)})"
        )

    print(f"Running SHARPy Abramson case: {case_name}")
    if args.no_rom:
        print("  Full-order linear UVLM (no Krylov ROM) — exact at all reduced frequencies")
    else:
        print(
            f"  Krylov ROM: {len(k_pts)} interpolation point(s) at reduced freq k={k_pts}, "
            f"r={args.rom_r} moments each (scaled real shift s0 = k)"
        )
    n_pts = n_vel if velocity_list is None else len(velocity_list)
    print(
        f"  rho={args.rho} kg/m³  V=[{v_min:.2f}, {v_max:.2f}] m/s  "
        f"({v_min/KN_TO_MS:.1f}–{v_max/KN_TO_MS:.1f} kn)  n={n_pts}"
    )
    sharpy.sharpy_main.main(["", ws.route + ws.case_name + ".sharpy"])

    case_out = out_root / case_name
    vel_file = _find_velocity_dat(case_out)
    parsed = _parse_velocity_dat(vel_file) if vel_file and vel_file.is_file() else {}
    modal_hz = _read_modal_frequencies_hz(case_out / "beam_modal_analysis")

    summary_dir = _FSI_ROOT / "output_data" / "ABRAMSON1965" / "sharpy_uvlm_water"
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
        f"rho: {args.rho} kg/m³ (stock SHARPy UVLM, no BEM)",
        (
            f"structural Rayleigh: zeta_bend={rayleigh_zetas[0]:.4g}, "
            f"zeta_tors={rayleigh_zetas[1]:.4g}"
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
            else f"Krylov ROM interpolation points (reduced freq k): {k_pts}  (r={args.rom_r} moments each)"
        ),
        f"velocity file: {vel_file}",
        f"experimental: V_f = {V_EXP_KN} kn ({V_EXP_KN * KN_TO_MS:.2f} m/s), f = {F_EXP_HZ} Hz",
        f"dry modal frequencies (Hz) SHARPy structural: {modal_hz}  (GINO/air ref: ~10, ~20.5)",
        f"flutter estimate (UVLM): V = {parsed.get('flutter_V_kn')} kn ({parsed.get('flutter_V_m_s')} m/s)",
        "",
        "Compare with GINO: output_data/ABRAMSON1965/.../vg_vf_combined.png",
        "Phase 4 (later): gino_bem/capy_bem_bridge.py",
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
