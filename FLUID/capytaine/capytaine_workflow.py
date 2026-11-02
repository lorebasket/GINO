"""
Capytaine matrix cache: fingerprint structural inputs, reuse or recompute.

Used by ``main.py`` when ``RFA_PK_method`` or ``pk_capytaine_matrices`` is enabled.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from typing import Any

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_FSI_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))

_SOURCE_ARRAY_KEYS = (
    "source_mode_indices",
    "source_eigenvalues",
    "source_eigenvectors",
    "source_beam_positions",
    "source_omega_list",
    "source_depth",
    "source_mesh_path",
    "source_rho",
    "source_phi_normalization",
    "source_dihedral_deg",
    "source_alpha_deg",
    "source_wing_dihedral_deg",
    "source_water_depth",
    "source_free_surface",
    "source_rigid_body_motion",
)

_OPTIONAL_SOURCE_KEYS = frozenset({"source_face_displacements", "source_bem_torsion_sign"})


def needs_capytaine_matrices(config) -> bool:
    """True when the flutter solver will load precomputed Capytaine A(ω), B(ω)."""
    return bool(
        getattr(config, "RFA_PK_method", False)
        or getattr(config, "pk_capytaine_matrices", False)
    )


def resolve_capytaine_npz_path(config) -> str:
    """Path to the modal radiation NPZ for the active depth index."""
    capy_dir = getattr(config, "capytaine_results_dir", "")
    if not capy_dir:
        case_name = getattr(config, "name", "case")
        capy_dir = os.path.join(_FSI_ROOT, "FLUID", "capytaine", case_name, "results_modal_radiation")
    depth_value = float(np.asarray(config.depth).ravel()[int(getattr(config, "depth_index", 0))])
    depth_tag = f"{depth_value:g}".replace(".", "p").replace("-", "m")
    capy_npz = getattr(
        config,
        "capytaine_modal_radiation_file",
        os.path.join(capy_dir, f"modal_radiation_AB_depth_{depth_tag}.npz"),
    )
    if not os.path.exists(capy_npz):
        legacy = os.path.join(capy_dir, f"modal_radiation_AB_depth_{depth_value:g}.npz")
        if os.path.exists(legacy):
            return legacy
        legacy2 = os.path.join(capy_dir, "modal_radiation_AB.npz")
        if os.path.exists(legacy2):
            return legacy2
    return capy_npz


def _mode_indices_from_config(config) -> list[int]:
    n = int(getattr(config, "num_modes_egv", 1))
    if n < 1:
        raise ValueError(f"num_modes_egv must be >= 1, got {n}")
    return list(range(n))


def _fluid_rho(config) -> float:
    fluid = getattr(config, "fluid", "water")
    rho_f = getattr(config, "rho_f", {})
    if isinstance(rho_f, dict):
        return float(rho_f.get(fluid, 997.0))
    return float(rho_f)


def _active_depth(config) -> float:
    return float(np.asarray(config.depth).ravel()[int(getattr(config, "depth_index", 0))])


def _water_depth_scalar(config) -> float:
    wd = getattr(config, "water_depth", np.inf)
    if isinstance(wd, str) and wd.lower() == "inf":
        return np.inf
    return float(wd)


def collect_source_inputs(
    config,
    structural_results,
    beam_model,
    *,
    face_displacements: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Structural / BEM inputs that define Capytaine modal radiation matrices.

    Arrays are stored in NPZ metadata so cached results can be validated on reload.
    """
    mode_indices = _mode_indices_from_config(config)
    n_modes = len(mode_indices)

    ev = np.asarray(structural_results.dry_eigenvalues, dtype=float).ravel()
    if ev.size < n_modes:
        raise ValueError(
            f"dry_eigenvalues has {ev.size} entries but num_modes_egv={n_modes}."
        )
    eigenvalues = ev[:n_modes]

    phi_full = np.asarray(structural_results.dry_eigenvectors_full, dtype=float)
    if phi_full.ndim != 2 or phi_full.shape[1] < n_modes:
        raise ValueError(
            f"dry_eigenvectors_full shape {phi_full.shape} incompatible with "
            f"num_modes_egv={n_modes}."
        )
    eigenvectors = phi_full[:, :n_modes].T.copy()

    nodes = beam_model["nodes"]
    beam_positions = np.asarray([node["position"] for node in nodes], dtype=float)

    omega_list = np.asarray(getattr(config, "omega_list"), dtype=float).ravel()
    mesh_path = os.path.abspath(
        getattr(config, "mesh_path", os.path.join(_FSI_ROOT, "FLUID", "capytaine", f"{config.name}_mesh.vtu"))
    )

    out: dict[str, Any] = {
        "source_mode_indices": np.asarray(mode_indices, dtype=int),
        "source_eigenvalues": eigenvalues,
        "source_eigenvectors": eigenvectors,
        "source_beam_positions": beam_positions,
        "source_omega_list": omega_list,
        "source_depth": np.asarray([_active_depth(config)], dtype=float),
        "source_mesh_path": np.asarray(mesh_path, dtype=object),
        "source_rho": np.asarray([_fluid_rho(config)], dtype=float),
        "source_phi_normalization": np.asarray(
            str(getattr(config, "capytaine_phi_normalization", "geometric")), dtype=object
        ),
        "source_dihedral_deg": np.asarray([float(getattr(config, "dihedral_angle", 0.0))], dtype=float),
        "source_alpha_deg": np.asarray([float(getattr(config, "alpha_deg", 0.0))], dtype=float),
        "source_wing_dihedral_deg": np.asarray(
            [float(getattr(config, "wing_dihedral_angle", 0.0))], dtype=float
        ),
        "source_water_depth": np.asarray([_water_depth_scalar(config)], dtype=float),
        "source_free_surface": np.asarray(
            [float(getattr(config, "free_surface_elevation", 0.0))], dtype=float
        ),
        "source_rigid_body_motion": np.asarray(
            [bool(getattr(config, "rigid_body_motion", False))], dtype=bool
        ),
        "source_bem_torsion_sign": np.asarray([float(getattr(config, "bem_torsion_sign", 1.0))], dtype=float),
    }
    if face_displacements is not None:
        out["source_face_displacements"] = np.asarray(face_displacements, dtype=float)
    return out


def _scalar_meta_equal(a, b, *, key: str) -> bool:
    sa = np.asarray(a).ravel()
    sb = np.asarray(b).ravel()
    if sa.size != sb.size:
        return False
    if sa.dtype == object or sb.dtype == object:
        return str(sa.flat[0]) == str(sb.flat[0])
    if np.issubdtype(sa.dtype, np.bool_) or np.issubdtype(sb.dtype, np.bool_):
        return bool(sa.flat[0]) == bool(sb.flat[0])
    if not np.all(np.isfinite(sa)) or not np.all(np.isfinite(sb)):
        return bool(np.all(sa == sb))
    return bool(np.allclose(sa, sb, rtol=1e-9, atol=1e-12))


def source_inputs_match(stored: dict[str, Any], current: dict[str, Any], *, rtol: float = 1e-9, atol: float = 1e-10) -> tuple[bool, str]:
    """Return (match, reason) comparing stored NPZ metadata to current structural inputs."""
    # caches written before bem_torsion_sign existed were built with +1
    ts_old = float(np.asarray(stored.get("source_bem_torsion_sign", [1.0])).ravel()[0])
    ts_new = float(np.asarray(current.get("source_bem_torsion_sign", [1.0])).ravel()[0])
    if ts_old != ts_new:
        return False, f"source_bem_torsion_sign {ts_old:+g} != {ts_new:+g}"
    keys = set(_SOURCE_ARRAY_KEYS) | (_OPTIONAL_SOURCE_KEYS & set(stored.keys()))
    for key in keys:
        if key not in stored:
            if key in current and key not in _OPTIONAL_SOURCE_KEYS:
                return False, f"cached file missing {key}"
            continue
        if key not in current:
            if key in _OPTIONAL_SOURCE_KEYS:
                continue
            return False, f"current run missing {key}"
        a = np.asarray(stored[key])
        b = np.asarray(current[key])
        if a.shape != b.shape:
            return False, f"{key} shape {a.shape} != {b.shape}"
        if a.dtype == object or b.dtype == object:
            if str(np.asarray(a).ravel()[0]) != str(np.asarray(b).ravel()[0]):
                return False, f"{key} path/string mismatch"
            continue
        if np.issubdtype(a.dtype, np.bool_) or np.issubdtype(b.dtype, np.bool_):
            if bool(a.flat[0]) != bool(b.flat[0]):
                return False, f"{key} bool mismatch"
            continue
        if not np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True):
            return False, f"{key} values differ"
    return True, "ok"


def extract_source_inputs_from_npz(capy_data: dict[str, Any]) -> dict[str, Any]:
    keys = set(_SOURCE_ARRAY_KEYS) | _OPTIONAL_SOURCE_KEYS
    return {k: capy_data[k] for k in keys if k in capy_data}


def _load_modal_radiation_io():
    io_path = os.path.join(_THIS_DIR, "modal_radiation_io.py")
    spec = importlib.util.spec_from_file_location("modal_radiation_io", io_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_capytaine_bundle(config) -> dict[str, Any]:
    """
    Load Capytaine A(ω), B(ω) and the structural inputs they were built from.
    """
    io = _load_modal_radiation_io()
    npz_path = resolve_capytaine_npz_path(config)
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(f"Capytaine modal radiation file not found: {npz_path}")
    capy_data = io.load_modal_radiation_npz(npz_path)
    return {
        "path": npz_path,
        "matrices": capy_data,
        "source_inputs": extract_source_inputs_from_npz(capy_data),
    }


def _export_modal_csv(config, structural_results, beam_model) -> None:
    """Write dry modal CSVs under modal_dir for run_modal_radiation beam mapping."""
    from FEA.fea_utl.analysis import _save_modal_data_to_csv

    modal_dir = getattr(config, "modal_dir", getattr(config, "output_dir", "."))
    os.makedirs(modal_dir, exist_ok=True)
    # Keep output_dir aligned so _save_modal_data_to_csv writes where Capytaine reads.
    prev_output_dir = getattr(config, "output_dir", None)
    config.output_dir = modal_dir

    total_dof = int(structural_results.total_dof)
    constrained = list(structural_results.constrained_dofs)
    free_dofs = sorted(set(range(total_dof)) - set(constrained))
    n_modes = int(getattr(config, "num_modes_egv", 1))
    phi_full = np.asarray(structural_results.dry_eigenvectors_full, dtype=float)
    dry_vectors = phi_full[free_dofs, :n_modes]
    dry_values = np.asarray(structural_results.dry_eigenvalues, dtype=float).ravel()[:n_modes]

    try:
        _save_modal_data_to_csv(
            beam_model,
            total_dof,
            constrained,
            dry_values,
            dry_vectors,
            config,
            output_suffix="_dry_egv",
        )
    finally:
        if prev_output_dir is not None:
            config.output_dir = prev_output_dir


def _run_modal_radiation(config) -> None:
    capy_path = os.path.join(_THIS_DIR, "run_modal_radiation.py")
    if _THIS_DIR not in sys.path:
        sys.path.insert(0, _THIS_DIR)
    spec = importlib.util.spec_from_file_location("run_modal_radiation", capy_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    phi_norm = str(getattr(config, "capytaine_phi_normalization", "geometric"))
    argv = [
        "--case-name",
        str(config.name),
        "--out-dir",
        os.path.abspath(getattr(config, "capytaine_results_dir", resolve_capytaine_npz_path(config))),
        "--no-plots",
        "--phi-normalization",
        phi_norm,
    ]
    n_modes = int(getattr(config, "num_modes_egv", 1))
    if n_modes >= 1:
        argv.extend(["--mode-indices", ",".join(str(i) for i in range(n_modes))])
    mesh_path = getattr(config, "mesh_path", None)
    if mesh_path:
        argv.extend(["--mesh-path", str(mesh_path)])
    modal_dir = getattr(config, "modal_dir", None)
    if modal_dir:
        argv.extend(["--modal-dir", os.path.abspath(str(modal_dir))])
    prefix = getattr(config, "prefix", None)
    if prefix:
        argv.extend(["--prefix", str(prefix)])
    mod.main(argv)


def capytaine_cache_valid(config, structural_results, beam_model) -> tuple[bool, str]:
    """Check whether an on-disk Capytaine NPZ matches current structural inputs."""
    npz_path = resolve_capytaine_npz_path(config)
    if not os.path.isfile(npz_path):
        return False, f"file not found: {npz_path}"

    io = _load_modal_radiation_io()
    try:
        capy_data = io.load_modal_radiation_npz(npz_path)
    except Exception as exc:
        return False, f"could not load {npz_path}: {exc}"

    stored = extract_source_inputs_from_npz(capy_data)
    if not stored:
        return False, "NPZ has no source-input metadata (pre-workflow cache)"

    current = collect_source_inputs(config, structural_results, beam_model)
    return source_inputs_match(stored, current)


def ensure_capytaine_matrices(
    config,
    structural_results,
    beam_model,
    *,
    force: bool = False,
) -> str:
    """
    Ensure Capytaine modal radiation matrices exist and match current structural state.

    Returns the path to the modal radiation NPZ.
    """
    if not needs_capytaine_matrices(config):
        return ""

    npz_path = resolve_capytaine_npz_path(config)
    out_dir = os.path.dirname(npz_path) or getattr(config, "capytaine_results_dir", ".")
    os.makedirs(out_dir, exist_ok=True)

    force = force or bool(getattr(config, "capytaine_force_recalc", False))
    auto = bool(getattr(config, "capytaine_auto_recalc", True))
    if not auto and not force:
        if not os.path.isfile(npz_path):
            raise FileNotFoundError(
                f"Capytaine matrices missing at {npz_path} and capytaine_auto_recalc=False."
            )
        print(f"[Capytaine] Using existing matrices (capytaine_auto_recalc=False): {npz_path}")
        return npz_path

    if not force:
        valid, reason = capytaine_cache_valid(config, structural_results, beam_model)
        if valid:
            print(f"[Capytaine] Reusing cached matrices (inputs unchanged): {npz_path}")
            return npz_path
        print(f"[Capytaine] Cache stale or missing ({reason}); recomputing…")

    print(f"[Capytaine] Exporting modal CSV → {getattr(config, 'modal_dir', config.output_dir)}")
    _export_modal_csv(config, structural_results, beam_model)

    print(f"[Capytaine] Running BEM radiation solve → {out_dir}")
    _run_modal_radiation(config)

    if not os.path.isfile(npz_path):
        raise RuntimeError(f"Capytaine run finished but NPZ not found: {npz_path}")

    bundle = load_capytaine_bundle(config)
    face_disp = bundle["matrices"].get("face_displacements")
    if face_disp is None:
        face_disp = bundle["matrices"].get("source_face_displacements")
    stamp_source_inputs_on_npz(
        npz_path,
        config,
        structural_results,
        beam_model,
        face_displacements=face_disp,
    )

    valid_after, reason_after = capytaine_cache_valid(config, structural_results, beam_model)
    if not valid_after:
        raise RuntimeError(
            f"Capytaine matrices saved but source-input fingerprint still mismatches: {reason_after}"
        )

    print(f"[Capytaine] Saved matrices and source inputs: {npz_path}")
    return npz_path


def attach_source_inputs_to_metadata(
    meta: dict[str, Any],
    config,
    structural_results,
    beam_model,
    *,
    face_displacements: np.ndarray | None = None,
) -> dict[str, Any]:
    """Merge fingerprint arrays into modal_radiation NPZ metadata before save."""
    meta.update(
        collect_source_inputs(
            config,
            structural_results,
            beam_model,
            face_displacements=face_displacements,
        )
    )
    return meta


def collect_source_inputs_from_modal_csv(
    config,
    *,
    modal_dir: str | None = None,
    prefix: str | None = None,
    mode_indices: list[int] | None = None,
    face_displacements: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Build source-input fingerprint from dry modal CSVs (standalone ``run_modal_radiation``).
    """
    from run_modal_radiation import (  # noqa: WPS433  same package
        _constrained_global_dofs,
        _load_beam_positions,
        _load_phi_full_row,
        _mode_indices_from_config,
    )

    modal_dir = modal_dir or getattr(config, "modal_dir", getattr(config, "output_dir", "."))
    prefix = prefix or getattr(config, "prefix", f"{config.name}_dry_egv")
    mode_indices = mode_indices if mode_indices is not None else _mode_indices_from_config(config)

    nodes_csv = os.path.join(modal_dir, f"{prefix}_nodes.csv")
    constrained_csv = os.path.join(modal_dir, f"{prefix}_constrained_dofs.csv")
    eigendata_csv = os.path.join(modal_dir, f"{prefix}_eigendata.csv")
    for path in (nodes_csv, constrained_csv, eigendata_csv):
        if not os.path.isfile(path):
            raise FileNotFoundError(path)

    beam_positions = _load_beam_positions(nodes_csv)
    n_nodes = int(beam_positions.shape[0])
    constrained = _constrained_global_dofs(constrained_csv)

    eigenvalues = []
    eigenvectors = []
    import csv

    with open(eigendata_csv, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        _ = next(reader, None)
        rows = [row for row in reader if len(row) >= 5]

    for mode_idx in mode_indices:
        if mode_idx >= len(rows):
            raise IndexError(
                f"mode index {mode_idx} out of range in {eigendata_csv} ({len(rows)} modes)."
            )
        row = rows[mode_idx]
        eigenvalues.append(float(row[3]))
        eigenvectors.append(
            _load_phi_full_row(eigendata_csv, n_nodes, constrained, mode_idx)
        )

    omega_list = np.asarray(getattr(config, "omega_list"), dtype=float).ravel()
    mesh_path = os.path.abspath(
        getattr(config, "mesh_path", os.path.join(_FSI_ROOT, "FLUID", "capytaine", f"{config.name}_mesh.vtu"))
    )

    out: dict[str, Any] = {
        "source_mode_indices": np.asarray(mode_indices, dtype=int),
        "source_eigenvalues": np.asarray(eigenvalues, dtype=float),
        "source_eigenvectors": np.asarray(eigenvectors, dtype=float),
        "source_beam_positions": beam_positions,
        "source_omega_list": omega_list,
        "source_depth": np.asarray([_active_depth(config)], dtype=float),
        "source_mesh_path": np.asarray(mesh_path, dtype=object),
        "source_rho": np.asarray([_fluid_rho(config)], dtype=float),
        "source_phi_normalization": np.asarray(
            str(getattr(config, "capytaine_phi_normalization", "geometric")), dtype=object
        ),
        "source_dihedral_deg": np.asarray([float(getattr(config, "dihedral_angle", 0.0))], dtype=float),
        "source_alpha_deg": np.asarray([float(getattr(config, "alpha_deg", 0.0))], dtype=float),
        "source_wing_dihedral_deg": np.asarray(
            [float(getattr(config, "wing_dihedral_angle", 0.0))], dtype=float
        ),
        "source_water_depth": np.asarray([_water_depth_scalar(config)], dtype=float),
        "source_free_surface": np.asarray(
            [float(getattr(config, "free_surface_elevation", 0.0))], dtype=float
        ),
        "source_rigid_body_motion": np.asarray(
            [bool(getattr(config, "rigid_body_motion", False))], dtype=bool
        ),
        "source_bem_torsion_sign": np.asarray([float(getattr(config, "bem_torsion_sign", 1.0))], dtype=float),
    }
    if face_displacements is not None:
        out["source_face_displacements"] = np.asarray(face_displacements, dtype=float)
    return out


def stamp_source_inputs_on_npz(
    npz_path: str,
    config,
    structural_results,
    beam_model,
    *,
    face_displacements: np.ndarray | None = None,
) -> None:
    """Append / overwrite source-input metadata on an existing radiation NPZ."""
    data = dict(np.load(npz_path, allow_pickle=True))
    data.update(
        collect_source_inputs(
            config,
            structural_results,
            beam_model,
            face_displacements=face_displacements,
        )
    )
    os.makedirs(os.path.dirname(npz_path) or ".", exist_ok=True)
    np.savez_compressed(npz_path, **data)
