"""
UVLM apparent-mass offset for GINO ``pk_bem_mode='delta'``.

Loads (or subprocess-precomputes via SHARPy) ``A_uvlm_gino`` and embeds it on the
PK modal DOFs so ``M_hydro = A_capy(omega) - A_uvlm`` matches SHARPy delta injection.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SHARPY_PY = os.environ.get(
    "SHARPY_PYTHON",
    "/home/lorenzo/anaconda3/envs/sharpy/bin/python",
)

_CASE_SLUGS = {
    "model2t": "model2t",
    "abramson1965": "abramson",
}

_DEFAULT_OUT_DIRS = {
    "model2t": _REPO_ROOT / "FLUID" / "capytaine" / "Model2t" / "bem_delta",
    "abramson": _REPO_ROOT / "FLUID" / "capytaine" / "ABRAMSON1965" / "bem_delta",
}

_DEFAULT_GRIDS = {
    "model2t": "n30c12",
    "abramson": "n12c16",
}


def _case_slug(config) -> str:
    name = str(getattr(config, "name", "")).lower().replace(" ", "")
    for key, slug in _CASE_SLUGS.items():
        if key in name:
            return slug
    raise ValueError(
        f"pk_bem_mode='delta' has no case mapping for config.name={config.name!r}; "
        f"supported: {sorted(_CASE_SLUGS)}"
    )


def _default_npz_path(config) -> Path:
    slug = _case_slug(config)
    grid = str(getattr(config, "pk_bem_delta_grid", None) or _DEFAULT_GRIDS[slug])
    nm = int(getattr(config, "num_modes_egv", 4))
    out_dir = _DEFAULT_OUT_DIRS[slug]
    return out_dir / f"bem_delta_A_uvlm_{grid}_nm{nm}.npz"


def _run_precompute(config, capy_npz: str | Path, out_npz: Path) -> None:
    script = _REPO_ROOT / "FLUID" / "sharpy" / "gino_bem" / "precompute_bem_delta.py"
    if not script.is_file():
        raise FileNotFoundError(f"Missing SHARPy precompute script: {script}")
    slug = _case_slug(config)
    grid = str(getattr(config, "pk_bem_delta_grid", None) or _DEFAULT_GRIDS[slug])
    nm = int(getattr(config, "num_modes_egv", 4))
    cmd = [
        _SHARPY_PY,
        str(script),
        "--case",
        slug,
        "--grid",
        grid,
        "--num-modes",
        str(nm),
        "--out",
        str(out_npz),
        "--capy-npz",
        str(capy_npz),
    ]
    strip_a = getattr(config, "pk_bem_delta_strip_a", None)
    if strip_a is not None:
        cmd.extend(["--strip-a", str(float(strip_a))])
    strip_b = getattr(config, "pk_bem_delta_strip_b", None)
    if strip_b is not None:
        cmd.extend(["--strip-b", str(float(strip_b))])
    v_probe = getattr(config, "pk_bem_delta_v_probe", None)
    if v_probe is not None:
        if isinstance(v_probe, str):
            cmd.extend(["--delta-v-probe", v_probe])
        else:
            cmd.extend(["--delta-v-probe", ",".join(str(float(x)) for x in np.atleast_1d(v_probe))])
    v_max_kn = getattr(config, "pk_bem_delta_v_max_kn", None)
    if v_max_kn is not None:
        cmd.extend(["--v-max-kn", str(float(v_max_kn))])
    if not bool(getattr(config, "pk_bem_delta_rayleigh", True)):
        cmd.append("--no-rayleigh")
    print(f"[pk_bem_mode=delta] precomputing A_uvlm via SHARPy: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=str(_REPO_ROOT / "FLUID" / "sharpy"))


def _load_block(npz_path: Path, nm: int) -> tuple[np.ndarray, dict]:
    d = np.load(npz_path, allow_pickle=True)
    A_gino = np.asarray(d["A_uvlm_gino"], dtype=float)
    n_file = int(d.get("nm", A_gino.shape[0]))
    if n_file != nm:
        raise ValueError(
            f"{npz_path} has nm={n_file} but config.num_modes_egv={nm}; "
            "regenerate with matching --num-modes"
        )
    meta = {
        "npz": str(npz_path),
        "case": str(d.get("case", "")),
        "grid": str(d.get("grid", "")),
        "strip_a": float(d.get("strip_a", np.nan)),
        "v_probe": np.asarray(d.get("v_probe", []), dtype=float).tolist(),
        "capy_npz": str(d.get("capy_npz", "")),
    }
    return A_gino[:nm, :nm], meta


def resolve_uvlm_offset_embedded(
    config,
    capy_npz: str | Path,
    n_modal: int,
    apply_indices,
    source_indices,
) -> tuple[np.ndarray, dict]:
    """Return (A_embedded, meta) with shape (n_modal, n_modal) in GINO basis."""
    from . import hydro_system

    nm = int(getattr(config, "num_modes_egv", n_modal))
    npz_path = Path(getattr(config, "pk_bem_delta_npz", None) or _default_npz_path(config))
    force = bool(getattr(config, "pk_bem_delta_force_recalc", False))
    if force or not npz_path.is_file():
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        _run_precompute(config, capy_npz, npz_path)

    A_block, meta = _load_block(npz_path, nm)
    B_zero = np.zeros_like(A_block)
    A_emb, _ = hydro_system.embed_capytaine_on_modal_dofs(
        A_block,
        B_zero,
        n_modal,
        apply_indices,
        source_indices,
    )
    print(
        f"[pk_bem_mode=delta] A_uvlm offset loaded from {npz_path.name}; "
        f"diag(gino)={np.array2string(np.diag(A_block), precision=3)}"
    )
    A_hf = None
    try:
        capy_data = np.load(capy_npz, allow_pickle=True)
        A_h = np.asarray(capy_data["added_mass"], dtype=float)
        if A_h.ndim == 3 and A_h.shape[0] > 0:
            A_hf = A_h[-1][:nm, :nm]
            ev = np.linalg.eigvalsh(np.eye(nm) + A_hf - A_block)
            print(
                f"[pk_bem_mode=delta] eig(I + A_capy(ω_max) - A_uvlm) = "
                f"{np.array2string(ev, precision=3)}"
            )
    except Exception as exc:
        print(f"[pk_bem_mode=delta] diagnostic eig check skipped: {exc}")

    meta["A_uvlm_gino_block"] = A_block
    if A_hf is not None:
        meta["A_capy_hf_gino"] = A_hf
    return A_emb, meta


def fit_uvlm_nc_mass(Qg_func, k_fit, rho, n_modal, n_pts=21):
    """Massa apparente NON circolatoria dell'UVLM dalla Q(k) usata dal p-k.

    Re Q = a + (2/rho) A k^2 nel tratto k in [lo, hi] (alto, dove la parte
    circolatoria e' quasi costante e la griglia risolve ancora l'oscillazione).
    Con K_eff = K - q Re Q, il termine q (2/rho) A k^2 = A w^2 e' massa: A si somma a M.
    Matrice completa, simmetrizzata. -> (A, meta)
    """
    lo, hi = (float(x) for x in k_fit)
    ks = np.linspace(lo, hi, int(n_pts))
    Q = np.array([np.real(np.atleast_2d(Qg_func(k, 1.0, None, 0, 0)))[:n_modal, :n_modal] for k in ks])
    X = np.column_stack([np.ones_like(ks), ks ** 2])
    coef = np.linalg.lstsq(X, Q.reshape(len(ks), -1), rcond=None)[0]
    A = 0.5 * float(rho) * coef[1].reshape(n_modal, n_modal)
    A = 0.5 * (A + A.T)
    res = np.abs(X @ coef - Q.reshape(len(ks), -1)).max() / max(np.abs(Q).max(), 1e-30)
    print(f"[pk_bem_mode=delta] A_UVLM_nc dal fit della GAF, k in [{lo:g}, {hi:g}], residuo {res:.1e}; "
          f"diag = {np.array2string(np.diag(A), precision=4)}")
    return A, {"npz": f"fit GAF k[{lo:g},{hi:g}]", "fit_residual": float(res), "A_uvlm_gino_block": A}
