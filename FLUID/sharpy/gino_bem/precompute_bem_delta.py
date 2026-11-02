#!/usr/bin/env python3
"""Precompute the UVLM apparent-mass offset A_uvlm for GINO ``pk_bem_mode='delta'``.

SHARPy delta injection keeps the intact UVLM circulatory GAF and injects only
``A_capy(omega) - A_uvlm`` into the structural mass.  The offset is measured by
the gamma_dot probe in ``build_delta_offset`` (same path as
``run_model2t_pk_bem.py --bem-mode delta``).

Writes an NPZ with ``A_uvlm_gino = T @ A_uvlm_sharpy @ T.T`` so GINO can subtract
it from the Capytaine A(omega) already expressed in the FEA modal basis.

  conda activate sharpy
  cd sharpy
  python gino_bem/precompute_bem_delta.py --case model2t --grid n30c12
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np

_SHARPY_ROOT = Path(__file__).resolve().parents[1]
_FSI_ROOT = _SHARPY_ROOT.parents[1]
if str(_SHARPY_ROOT) not in sys.path:
    sys.path.insert(0, str(_SHARPY_ROOT))

KN = 0.514444

_CASE_SPECS = {
    "abramson": {
        "module": "gino_bem.run_abramson_pk_bem",
        "default_grid": "n12c16",
        "out_dir": _FSI_ROOT / "FLUID" / "capytaine" / "ABRAMSON1965" / "bem_delta",
        "out_name": "bem_delta_A_uvlm_{grid}_nm{nm}.npz",
        "capy_default": "default_capytaine_npz",
        "strip_a_default": -0.5,
        "strip_b_default": None,
        "v_probe_default": "3.0,4.5,6.0",
        "v_max_kn_default": 60.0,
        "wetted_mask_fn": None,
        "probe_tag": "abramson_pk_probe_gd",
        "probe_part_floor": 0.005,
        "probe_part_ceiling": None,
        "require_all_modes": False,  # modes 0,2,3 often miss at probe V; strip-ratio fill
        "rayleigh_zetas": (0.0, 0.0),
    },
    "model2t": {
        "module": "gino_bem.run_model2t_pk_bem",
        "default_grid": "n30c12",
        "out_dir": _FSI_ROOT / "FLUID" / "capytaine" / "Model2t" / "bem_delta",
        "out_name": "bem_delta_A_uvlm_{grid}_nm{nm}.npz",
        "capy_default": "default_model2t_capy_npz",
        "strip_a_default": None,  # 2*(XEA-0.5) resolved at runtime
        "strip_b_default": None,  # CHORD/2 resolved at runtime
        "v_probe_default": "0.5,1.0,1.5",
        "v_max_kn_default": 40.0,
        "wetted_mask_fn": "wetted_span_mask",
        "probe_tag": "model2t_pk_probe_gd",
        "probe_part_floor": 1e-4,
        "probe_part_ceiling": 0.5,
        "require_all_modes": False,
        "rayleigh_zetas": None,  # MODEL2T_RAYLEIGH_ZETAS at runtime
    },
}


def _load_case(case: str):
    spec = _CASE_SPECS[case]
    mod = importlib.import_module(spec["module"])
    from gino_bem.capy_bem_bridge import check_capy_basis, load_capytaine_npz

    capy_fn = getattr(mod, spec["capy_default"])
    wetted_fn = None
    wetted_name = spec.get("wetted_mask_fn")
    if wetted_name:
        wetted_fn = getattr(mod, wetted_name)

    strip_a = spec["strip_a_default"]
    strip_b = spec["strip_b_default"]
    if case == "model2t":
        from gino_bem.model2t_strut import CHORD, XEA

        if strip_a is None:
            strip_a = 2.0 * (XEA - 0.5)
        if strip_b is None:
            strip_b = CHORD / 2.0
        if spec["rayleigh_zetas"] is None:
            from gino_bem.run_model2t_water import MODEL2T_RAYLEIGH_ZETAS

            rayleigh_zetas = MODEL2T_RAYLEIGH_ZETAS
        else:
            rayleigh_zetas = spec["rayleigh_zetas"]
    else:
        rayleigh_zetas = spec["rayleigh_zetas"]

    from gino_bem.run_abramson_pk_bem import build_delta_offset
    from gino_bem.mode_alignment import align_capy_to_sharpy

    return (
        mod.GRIDS,
        mod.build_case,
        capy_fn,
        check_capy_basis,
        load_capytaine_npz,
        build_delta_offset,
        align_capy_to_sharpy,
        wetted_fn,
        strip_a,
        strip_b,
        rayleigh_zetas,
        spec,
    )


def measure_delta_offset(
    *,
    case: str,
    grid_tag: str,
    nm: int,
    capy_npz: Path | None = None,
    strip_a: float | None = None,
    strip_b: float | None = None,
    v_probe: str | None = None,
    v_max_kn: float | None = None,
    rayleigh: bool = True,
    zetas: tuple[float, float] | None = None,
) -> dict:
    (
        GRIDS,
        build_case,
        default_capy_npz,
        check_capy_basis,
        load_capytaine_npz,
        build_delta_offset,
        align_capy_to_sharpy,
        wetted_fn,
        strip_a_def,
        strip_b_def,
        rayleigh_zetas_def,
        spec,
    ) = _load_case(case)

    if grid_tag not in GRIDS:
        raise ValueError(f"grid {grid_tag!r} not in {case} GRIDS: {sorted(GRIDS)}")
    grid = GRIDS[grid_tag]
    capy_path = Path(capy_npz or default_capy_npz())
    capy = check_capy_basis(load_capytaine_npz(capy_path))

    strip_a = float(strip_a if strip_a is not None else strip_a_def)
    strip_b = strip_b if strip_b is not None else strip_b_def
    v_probe_list = [
        float(x) for x in (v_probe or spec["v_probe_default"]).split(",") if x.strip()
    ]
    v_max_kn = float(v_max_kn if v_max_kn is not None else spec["v_max_kn_default"])
    zetas = tuple(zetas if zetas is not None else rayleigh_zetas_def)

    print(f"[precompute_bem_delta] case={case} grid={grid_tag} nm={nm} capy={capy_path.name}")
    data, _ = build_case(
        nm,
        grid=grid,
        added_mass_removal="none",
        rayleigh=rayleigh,
        zetas=zetas,
        v_max_kn=v_max_kn,
        case_tag=f"gino_bem_delta_{case}_{grid_tag}_nm{nm}",
    )
    ls = data.linear.linear_system
    T, _ = align_capy_to_sharpy(capy, data, nm)
    wetted_mask = wetted_fn(data) if wetted_fn is not None else None

    kw = dict(
        grid=grid,
        rayleigh=rayleigh,
        zetas=zetas,
        v_max_kn=v_max_kn,
        a_ea=strip_a,
        v_probe=v_probe_list,
        build_case_fn=build_case,
        probe_tag=spec["probe_tag"],
        probe_part_floor=spec["probe_part_floor"],
        probe_part_ceiling=spec["probe_part_ceiling"],
        require_all_modes=spec["require_all_modes"],
        wetted_mask=wetted_mask,
    )
    if strip_b is not None:
        kw["strip_b"] = float(strip_b)
    A_uvlm_sharpy = build_delta_offset(data, capy, T, nm, ls, **kw)

    Tn = np.asarray(T, dtype=float)[:nm, :nm]
    A_sh = np.asarray(A_uvlm_sharpy, dtype=float)[:nm, :nm]
    A_uvlm_gino = Tn @ A_sh @ Tn.T

    print("[precompute_bem_delta] A_uvlm_sharpy:")
    print(np.array2string(A_sh, precision=3, suppress_small=True))
    print("[precompute_bem_delta] A_uvlm_gino (T @ A_sh @ T.T):")
    print(np.array2string(A_uvlm_gino, precision=3, suppress_small=True))

    return {
        "A_uvlm_sharpy": A_sh,
        "A_uvlm_gino": A_uvlm_gino,
        "T": Tn,
        "case": case,
        "grid": grid_tag,
        "nm": nm,
        "capy_npz": str(capy_path),
        "strip_a": strip_a,
        "strip_b": np.nan if strip_b is None else float(strip_b),
        "v_probe": np.asarray(v_probe_list, dtype=float),
        "v_max_kn": v_max_kn,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", choices=sorted(_CASE_SPECS), default="model2t")
    ap.add_argument("--grid", default=None)
    ap.add_argument("--num-modes", type=int, default=4)
    ap.add_argument("--capy-npz", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--strip-a", type=float, default=None)
    ap.add_argument("--strip-b", type=float, default=None)
    ap.add_argument("--delta-v-probe", type=str, default=None)
    ap.add_argument("--v-max-kn", type=float, default=None)
    ap.add_argument("--no-rayleigh", action="store_true")
    args = ap.parse_args()

    spec = _CASE_SPECS[args.case]
    grid_tag = args.grid or spec["default_grid"]
    out = args.out
    if out is None:
        out_dir = spec["out_dir"]
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / spec["out_name"].format(grid=grid_tag, nm=args.num_modes)

    result = measure_delta_offset(
        case=args.case,
        grid_tag=grid_tag,
        nm=args.num_modes,
        capy_npz=args.capy_npz,
        strip_a=args.strip_a,
        strip_b=args.strip_b,
        v_probe=args.delta_v_probe,
        v_max_kn=args.v_max_kn,
        rayleigh=not args.no_rayleigh,
    )
    np.savez_compressed(
        out,
        A_uvlm_sharpy=result["A_uvlm_sharpy"],
        A_uvlm_gino=result["A_uvlm_gino"],
        T=result["T"],
        case=result["case"],
        grid=result["grid"],
        nm=result["nm"],
        capy_npz=result["capy_npz"],
        strip_a=result["strip_a"],
        strip_b=result["strip_b"],
        v_probe=result["v_probe"],
        v_max_kn=result["v_max_kn"],
    )
    print(f"[precompute_bem_delta] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
