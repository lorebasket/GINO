#!/usr/bin/env python3
"""
Validate Capytaine added-mass computation against known analytical values.

Tier 1 — exact benchmarks (infinite fluid, no free surface):
  • Sphere  : A_surge = A_sway = A_heave = (2/3) π ρ R³
  • Horizontal cylinder (axis ‖ X, heave in Z):
              A_heave ≈ π ρ R² L   (slender-body / 2-D strip limit per unit length × L)

Tier 2 — wing / airfoil (order-of-magnitude check):
  • Abramson1965 hull, rigid Heave & Sway vs strip estimate m_a' L with m_a' = ρ π c² / 4
    (3-D BEM + partial immersion ≠ strip; reported as indicative only).

Tier 3 — workflow check (Abramson mesh):
  • Rigid-body Heave (Capytaine built-in) vs one custom face DOF with the *same*
    face displacement field — added mass must agree (validates --modes / run_modal_radiation path).

Usage (FSI root, sonata-env):
    python FLUID/capytaine/validate_added_mass_benchmarks.py
    python FLUID/capytaine/validate_added_mass_benchmarks.py --case-name ABRAMSON1965
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
_FSI = os.path.dirname(os.path.dirname(_THIS))
_EXAMPLES = os.path.join(_FSI, "examples")
if _EXAMPLES not in sys.path:
    sys.path.insert(0, _EXAMPLES)
if _THIS not in sys.path:
    sys.path.insert(0, _THIS)

from config import get_config  # noqa: E402
from run_modal_radiation import (  # noqa: E402
    _apply_rigid_body_dofs,
    _build_floating_body,
    _mesh_area_centroid,
    _resolve_mesh_loader,
    _translate_mesh_z,
)


def _solve_rigid_diagonal(cpt, body, *, omega: float, rho: float, dofs: list[str]) -> dict[str, float]:
    solver = cpt.BEMSolver()
    out: dict[str, float] = {}
    for dof in dofs:
        problem = cpt.RadiationProblem(
            body=body,
            radiating_dof=dof,
            omega=float(omega),
            rho=float(rho),
            water_depth=np.inf,
            free_surface=np.inf,
        )
        result = solver.solve(problem)
        out[str(dof)] = float(result.added_masses[dof])
    return out


def _pct_err(num: float, ref: float) -> float:
    if abs(ref) < 1e-30:
        return np.nan
    return 100.0 * (num - ref) / ref


def benchmark_sphere(*, radius: float, rho: float, resolution: tuple[int, int], omega: float) -> dict:
    import capytaine as cpt

    mesh = cpt.mesh_sphere(radius=radius, center=(0.0, 0.0, 0.0), resolution=resolution)
    body = _build_floating_body(cpt, mesh, name="sphere")
    body.add_all_rigid_body_dofs()
    dofs = ["Surge", "Sway", "Heave"]
    A_num = _solve_rigid_diagonal(cpt, body, omega=omega, rho=rho, dofs=dofs)
    A_ref = (2.0 / 3.0) * np.pi * rho * radius**3
    return {
        "geometry": "sphere",
        "radius_m": radius,
        "resolution": resolution,
        "omega_rad_s": omega,
        "A_ref_kg": A_ref,
        "A_num_kg": A_num,
        "pct_error": {d: _pct_err(A_num[d], A_ref) for d in dofs},
    }


def benchmark_cylinder(
    *,
    radius: float,
    length: float,
    rho: float,
    resolution: tuple[int, int, int],
    omega: float,
) -> dict:
    import capytaine as cpt

    mesh = cpt.mesh_horizontal_cylinder(
        length=length,
        radius=radius,
        center=(0.0, 0.0, 0.0),
        resolution=resolution,
    )
    body = _build_floating_body(cpt, mesh, name="cylinder")
    body.add_all_rigid_body_dofs()
    A_num = _solve_rigid_diagonal(cpt, body, omega=omega, rho=rho, dofs=["Heave"])
    A_ref = np.pi * rho * radius**2 * length
    return {
        "geometry": "horizontal_cylinder",
        "radius_m": radius,
        "length_m": length,
        "resolution": resolution,
        "omega_rad_s": omega,
        "A_heave_ref_kg": A_ref,
        "A_heave_num_kg": A_num["Heave"],
        "pct_error_heave": _pct_err(A_num["Heave"], A_ref),
    }


def _case_fluid_settings(cfg) -> tuple[float, float, float, float]:
    fluid = getattr(cfg, "fluid", "water")
    rho = float(getattr(cfg, "rho_f", {}).get(fluid, 997.0))
    depth = float(np.asarray(cfg.depth).ravel()[int(getattr(cfg, "depth_index", 0))])
    free_surface = getattr(cfg, "free_surface_elevation", 0.0)
    fs = float(free_surface) if free_surface is not None else 0.0
    water_depth_cfg = getattr(cfg, "water_depth", np.inf)
    wd = np.inf if str(water_depth_cfg).lower() == "inf" else float(water_depth_cfg)
    return rho, depth, fs, wd


def _load_case_mesh_submerged(cpt, cfg, case_name: str):
    depth = float(np.asarray(cfg.depth).ravel()[int(getattr(cfg, "depth_index", 0))])
    mesh_path = getattr(cfg, "mesh_path", os.path.join(_THIS, case_name, f"{case_name}_mesh.vtu"))
    mesh = _resolve_mesh_loader(cpt, mesh_path)
    _translate_mesh_z(mesh, -depth)
    return mesh, depth


def benchmark_abramson_heave_rigid_vs_custom(
    case_name: str,
    *,
    omega_list: list[float],
) -> dict:
    """
  Compare built-in rigid Heave vs a single custom DOF carrying the identical face field.

  This is the sanity check for ``run_modal_radiation.py --modes`` (face_displacements NPZ).
  """
    import capytaine as cpt

    cfg = get_config(case_name)
    rho, depth, fs, wd = _case_fluid_settings(cfg)
    rot_c = getattr(cfg, "rigid_body_rotation_center", None)
    rot_c = None if rot_c is None else np.asarray(rot_c, dtype=float)

    mesh_rigid, _ = _load_case_mesh_submerged(cpt, cfg, case_name)
    body_rigid = _build_floating_body(cpt, mesh_rigid, name="rigid_heave_ref")
    _, _ = _apply_rigid_body_dofs(body_rigid, rotation_center=rot_c)
    heave_faces = np.asarray(body_rigid.dofs["Heave"], dtype=float).copy()
    com = getattr(body_rigid, "center_of_mass", None)
    com = np.asarray(com, dtype=float).ravel() if com is not None else _mesh_area_centroid(mesh_rigid)

    mesh_custom, _ = _load_case_mesh_submerged(cpt, cfg, case_name)
    body_custom = _build_floating_body(cpt, mesh_custom, name="custom_heave")
    body_custom.dofs["Heave_custom"] = heave_faces

    mesh_custom_field = np.asarray(body_custom.dofs["Heave_custom"], dtype=float)
    diff_norm = float(np.linalg.norm(heave_faces - mesh_custom_field))
    ref_norm = float(np.linalg.norm(heave_faces))
    field_match_rel = diff_norm / max(ref_norm, 1e-30)

    solver = cpt.BEMSolver()
    rows: list[dict] = []
    for omega in omega_list:
        prob_rigid = cpt.RadiationProblem(
            body=body_rigid,
            radiating_dof="Heave",
            omega=float(omega),
            rho=rho,
            water_depth=wd,
            free_surface=fs,
        )
        prob_custom = cpt.RadiationProblem(
            body=body_custom,
            radiating_dof="Heave_custom",
            omega=float(omega),
            rho=rho,
            water_depth=wd,
            free_surface=fs,
        )
        A_rigid = float(solver.solve(prob_rigid).added_masses["Heave"])
        A_custom = float(solver.solve(prob_custom).added_masses["Heave_custom"])
        rows.append(
            {
                "omega_rad_s": float(omega),
                "A_rigid_kg": A_rigid,
                "A_custom_kg": A_custom,
                "pct_error": _pct_err(A_custom, A_rigid),
            }
        )

    max_pct = max(abs(r["pct_error"]) for r in rows)
    return {
        "geometry": f"{case_name} Heave rigid vs custom face DOF",
        "depth_m": depth,
        "rho": rho,
        "free_surface": fs,
        "water_depth": wd,
        "rotation_center_m": com.tolist(),
        "n_faces": int(heave_faces.shape[0]),
        "field_match_rel": field_match_rel,
        "rows": rows,
        "max_abs_pct_error": max_pct,
    }


def benchmark_wing_case(case_name: str, *, omega: float) -> dict:
    """Rigid Heave/Sway on case hydro mesh vs strip m_a' · L (indicative)."""
    import capytaine as cpt

    cfg = get_config(case_name)
    rho, depth, fs, wd = _case_fluid_settings(cfg)
    chord = float(cfg.chord)
    span = float(cfg.beam_length)

    mesh, _ = _load_case_mesh_submerged(cpt, cfg, case_name)
    body = _build_floating_body(cpt, mesh, name="wing")
    body.add_all_rigid_body_dofs()

    solver = cpt.BEMSolver()
    A_num: dict[str, float] = {}
    for dof in ("Heave", "Sway", "Surge"):
        problem = cpt.RadiationProblem(
            body=body,
            radiating_dof=dof,
            omega=float(omega),
            rho=float(rho),
            water_depth=wd,
            free_surface=fs,
        )
        result = solver.solve(problem)
        A_num[dof] = float(result.added_masses[dof])

    m_a_prime = rho * np.pi * chord**2 / 4.0
    A_strip_scale = m_a_prime * span

    return {
        "geometry": f"wing ({case_name})",
        "chord_m": chord,
        "span_m": span,
        "depth_m": depth,
        "rho": rho,
        "omega_rad_s": omega,
        "free_surface": fs,
        "water_depth": getattr(get_config(case_name), "water_depth", np.inf),
        "A_num_kg": A_num,
        "strip_m_a_prime_kg_per_m": m_a_prime,
        "strip_scale_m_a_prime_times_L_kg": A_strip_scale,
        "note": (
            "Strip m_a'·L is a 2-D infinite-span sectional scale, NOT an exact 3-D rigid-body "
            "added mass. Partial immersion and free surface lower Capytaine values."
        ),
    }


def _print_sphere(r: dict) -> None:
    print("\n=== Sphere (infinite fluid, A = 2/3 π ρ R³) ===")
    print(f"  R = {r['radius_m']} m, mesh {r['resolution']}, ω = {r['omega_rad_s']} rad/s")
    print(f"  A_ref = {r['A_ref_kg']:.4f} kg")
    for dof in ("Surge", "Sway", "Heave"):
        print(f"  {dof:5s}: A = {r['A_num_kg'][dof]:.4f} kg  ({r['pct_error'][dof]:+.2f} %)")


def _print_cylinder(r: dict) -> None:
    print("\n=== Horizontal cylinder Heave (A ≈ π ρ R² L) ===")
    print(f"  R = {r['radius_m']} m, L = {r['length_m']} m, mesh {r['resolution']}")
    print(f"  A_ref = {r['A_heave_ref_kg']:.4f} kg")
    print(f"  A_num = {r['A_heave_num_kg']:.4f} kg  ({r['pct_error_heave']:+.2f} %)")


def _print_heave_consistency(r: dict) -> None:
    print(f"\n=== {r['geometry']} ===")
    print(f"  depth = {r['depth_m']} m, ρ = {r['rho']} kg/m³, n_faces = {r['n_faces']}")
    print(f"  rotation center [m]: {r['rotation_center_m']}")
    print(f"  face field copy error (rel): {r['field_match_rel']:.3e}")
    print(f"  {'omega':>8s}  {'A_rigid':>12s}  {'A_custom':>12s}  {'err %':>8s}")
    for row in r["rows"]:
        print(
            f"  {row['omega_rad_s']:8.3f}  {row['A_rigid_kg']:12.6f}  "
            f"{row['A_custom_kg']:12.6f}  {row['pct_error']:+8.4f}"
        )
    print(f"  max |error| = {r['max_abs_pct_error']:.4f} %")


def _print_wing(r: dict) -> None:
    print(f"\n=== Wing / airfoil: {r['geometry']} ===")
    print(f"  chord = {r['chord_m']} m, span = {r['span_m']} m, depth = {r['depth_m']} m")
    print(f"  ω = {r['omega_rad_s']} rad/s, ρ = {r['rho']} kg/m³")
    print(f"  Capytaine rigid-body diagonal A_ii [kg]:")
    for dof, val in r["A_num_kg"].items():
        print(f"    {dof:5s}: {val:.4f}")
    print(f"  Strip 2-D scale m_a' = {r['strip_m_a_prime_kg_per_m']:.2f} kg/m")
    print(f"  Strip order-of-magnitude m_a'·L = {r['strip_scale_m_a_prime_times_L_kg']:.2f} kg")
    print(f"  Note: {r['note']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Capytaine added-mass analytical benchmarks.")
    parser.add_argument("--case-name", type=str, default="ABRAMSON1965", help="Wing case for tier-2 check.")
    parser.add_argument("--rho", type=float, default=1000.0, help="Fluid density for analytic benchmarks [kg/m³].")
    parser.add_argument("--omega", type=float, default=5.0, help="Radiation frequency [rad/s] (high enough for plateau).")
    parser.add_argument("--sphere-res", type=int, default=30, help="Sphere mesh resolution (n_theta, n_phi).")
    parser.add_argument("--cyl-res", type=str, default="12,24,8", help="Cylinder resolution nx,ntheta,nlength.")
    parser.add_argument(
        "--heave-omegas",
        type=str,
        default="1,5,20,60",
        help="ω [rad/s] for Abramson Heave rigid vs custom DOF test.",
    )
    parser.add_argument(
        "--skip-heave-test",
        action="store_true",
        help="Skip Abramson Heave rigid vs custom face DOF consistency check.",
    )
    args = parser.parse_args()

    cyl_parts = [int(x) for x in args.cyl_res.split(",")]
    if len(cyl_parts) != 3:
        raise ValueError("--cyl-res must be 'nx,ntheta,nlength'.")

    print("Capytaine added-mass validation")
    print(f"ρ = {args.rho} kg/m³, ω = {args.omega} rad/s")

    r_sphere = benchmark_sphere(
        radius=1.0,
        rho=args.rho,
        resolution=(args.sphere_res, args.sphere_res),
        omega=args.omega,
    )
    _print_sphere(r_sphere)

    r_cyl = benchmark_cylinder(
        radius=0.5,
        length=4.0,
        rho=args.rho,
        resolution=tuple(cyl_parts),
        omega=args.omega,
    )
    _print_cylinder(r_cyl)

    try:
        r_wing = benchmark_wing_case(args.case_name, omega=args.omega)
        _print_wing(r_wing)
    except Exception as exc:
        print(f"\n=== Wing case {args.case_name} skipped: {exc} ===")

    ok_heave = True
    if not args.skip_heave_test:
        try:
            heave_omegas = [float(x.strip()) for x in args.heave_omegas.split(",") if x.strip()]
            r_heave = benchmark_abramson_heave_rigid_vs_custom(
                args.case_name, omega_list=heave_omegas
            )
            _print_heave_consistency(r_heave)
            ok_heave = r_heave["max_abs_pct_error"] < 0.01 and r_heave["field_match_rel"] < 1e-12
        except Exception as exc:
            ok_heave = False
            print(f"\n=== Abramson Heave consistency test failed: {exc} ===")

    # Pass/fail summary for exact benchmarks
    ok_sphere = all(abs(r_sphere["pct_error"][d]) < 5.0 for d in ("Surge", "Sway", "Heave"))
    ok_cyl = abs(r_cyl["pct_error_heave"]) < 8.0
    print("\n--- Summary ---")
    print(f"  Sphere (±5 %): {'PASS' if ok_sphere else 'CHECK MESH / refine --sphere-res'}")
    print(f"  Cylinder Heave (±8 %): {'PASS' if ok_cyl else 'CHECK MESH / refine --cyl-res'}")
    if not args.skip_heave_test:
        print(f"  Abramson Heave rigid vs custom DOF (±0.01 %): {'PASS' if ok_heave else 'FAIL'}")
    print("  Wing strip scale: qualitative only.")


if __name__ == "__main__":
    main()
