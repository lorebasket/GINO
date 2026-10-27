"""
Bridge GINO Capytaine modal radiation NPZ → SHARPy linear beam (modal M, C).

Lives under ``sharpy/gino_bem/`` (FSI repo). Optional — only if you inject BEM on top
of stock SHARPy. For ``rho``-only water tests use ``run_goland_water_rho.py``.
"""

from __future__ import annotations

import importlib.util
import warnings
from functools import wraps
from pathlib import Path
from typing import Any

import numpy as np

_FSI_ROOT = Path(__file__).resolve().parents[3]
_CAPY_IO = _FSI_ROOT / "FLUID" / "capytaine" / "modal_radiation_io.py"


def _load_modal_radiation_io():
    spec = importlib.util.spec_from_file_location("modal_radiation_io", _CAPY_IO)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def load_capytaine_npz(path: str | Path) -> dict[str, Any]:
    io = _load_modal_radiation_io()
    return io.load_modal_radiation_npz(str(path))


def check_capy_basis(capy: dict[str, Any]) -> dict[str, Any]:
    """Validate modal basis and strip non-finite omega grid rows in place.

    The injected matrices must be in the mass-normalized modal basis to be
    added directly to SHARPy's ``Phi.T M Phi = I`` beam matrices.
    """
    basis = str(capy.get("added_mass_modal_basis", ""))
    if basis != "mass_normalized":
        raise ValueError(
            f"Capytaine NPZ modal basis is '{basis or '<missing>'}', expected 'mass_normalized'. "
            "Re-run FLUID/capytaine/run_modal_radiation.py with mass-normalized projection "
            "(or convert via phi_bem_scale_factors) before injecting into SHARPy."
        )

    omega = np.asarray(capy["omega"], dtype=float).ravel()
    A = np.asarray(capy["added_mass"], dtype=float)
    B = np.asarray(capy["added_damping"], dtype=float)
    finite = np.isfinite(A).all(axis=(1, 2)) & np.isfinite(B).all(axis=(1, 2))
    n_bad = int((~finite).sum())
    if n_bad:
        warnings.warn(
            f"Capytaine NPZ: dropping {n_bad} omega grid point(s) with non-finite A/B "
            f"(omegas: {omega[~finite][:5]})",
            RuntimeWarning,
        )
        capy["omega"] = omega[finite]
        capy["added_mass"] = A[finite]
        capy["added_damping"] = B[finite]
    if len(np.asarray(capy["omega"]).ravel()) < 2:
        raise ValueError("Capytaine NPZ: fewer than 2 finite omega grid points after NaN strip.")
    return capy


def interp_modal_matrix(omega_grid: np.ndarray, stack: np.ndarray, omega_query: float) -> np.ndarray:
    w = float(np.clip(omega_query, omega_grid[0], omega_grid[-1]))
    i_hi = int(np.searchsorted(omega_grid, w, side="right"))
    if i_hi <= 0:
        return np.asarray(stack[0], dtype=float)
    if i_hi >= len(omega_grid):
        return np.asarray(stack[-1], dtype=float)
    i_lo = i_hi - 1
    w0, w1 = float(omega_grid[i_lo]), float(omega_grid[i_hi])
    alpha = 0.0 if abs(w1 - w0) < 1e-14 else (w - w0) / (w1 - w0)
    return (1.0 - alpha) * stack[i_lo] + alpha * stack[i_hi]


_CLIP_WARNED: set[str] = set()


def capy_at_omega(
    capy: dict[str, Any],
    omega_rad_s: float,
    *,
    mode_slice: slice | None = None,
    warn_out_of_grid: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    omega_h = np.asarray(capy["omega"], dtype=float).ravel()
    if warn_out_of_grid:
        side = "above" if omega_rad_s > omega_h[-1] else ("below" if omega_rad_s < omega_h[0] else None)
        if side is not None and side not in _CLIP_WARNED:
            _CLIP_WARNED.add(side)
            warnings.warn(
                f"capy_at_omega: query omega={omega_rad_s:.2f} rad/s is {side} the Capytaine grid "
                f"[{omega_h[0]:.2f}, {omega_h[-1]:.2f}] rad/s — clipping to grid edge "
                "(warning shown once per direction; consider regenerating the NPZ with a wider grid).",
                RuntimeWarning,
            )
    A = interp_modal_matrix(omega_h, capy["added_mass"], omega_rad_s)
    B = interp_modal_matrix(omega_h, capy["added_damping"], omega_rad_s)
    if mode_slice is not None:
        A = A[mode_slice, mode_slice]
        B = B[mode_slice, mode_slice]
    return np.asarray(A, dtype=float), np.asarray(B, dtype=float)


def patch_beam_assemble_with_bem(beam_flex_sys) -> None:
    if getattr(beam_flex_sys, "_gino_bem_patch_active", False):
        return

    import sharpy.linear.src.lingebm as lingebm

    orig_assemble = beam_flex_sys.assemble

    @wraps(orig_assemble)
    def assemble_with_bem(Nmodes=None):
        M_bem = getattr(beam_flex_sys, "_bem_M_modal", None)
        C_bem = getattr(beam_flex_sys, "_bem_C_modal", None)
        if M_bem is None and C_bem is None:
            return orig_assemble(Nmodes)

        orig_newmark = lingebm.newmark_ss

        @wraps(orig_newmark)
        def newmark_with_bem(M, C, K, dt, num_damp=1e-4, M_is_SPD=False):
            M_out = np.array(M, dtype=float, copy=True)
            C_out = np.array(C, dtype=float, copy=True)
            n = M_out.shape[0]
            if M_bem is not None:
                M_out[:n, :n] += np.asarray(M_bem, dtype=float)[:n, :n]
            if C_bem is not None:
                C_out[:n, :n] += np.asarray(C_bem, dtype=float)[:n, :n]
            return orig_newmark(M_out, C_out, K, dt, num_damp=num_damp, M_is_SPD=M_is_SPD)

        lingebm.newmark_ss = newmark_with_bem
        try:
            return orig_assemble(Nmodes)
        finally:
            lingebm.newmark_ss = orig_newmark

    beam_flex_sys.assemble = assemble_with_bem
    beam_flex_sys._gino_bem_patch_active = True


def set_bem_modal(beam_flex_sys, M_add: np.ndarray | None, C_add: np.ndarray | None) -> None:
    patch_beam_assemble_with_bem(beam_flex_sys)
    beam_flex_sys._bem_M_modal = None if M_add is None else np.asarray(M_add, dtype=float)
    beam_flex_sys._bem_C_modal = None if C_add is None else np.asarray(C_add, dtype=float)


def apply_bem_from_capy(
    linear_aeroelastic,
    capy: dict[str, Any],
    omega_rad_s: float,
    *,
    n_modes: int | None = None,
    T: np.ndarray | None = None,
    t_ref: float | None = None,
    A_offset: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Set Capytaine A(omega), B(omega) on the beam ahead of the next assemble().

    ``T`` is the signed permutation mapping SHARPy modal coords -> GINO modal
    coords (``q_gino = T q_sharpy``); matrices are congruence-transformed
    ``T.T @ A @ T``. ``t_ref`` is SHARPy's time-normalisation reference
    (ScalingFacts['length']/V): the beam is assembled in scaled time where
    ``Cstr = C_phys * t_ref`` (lingebm.update_matrices_time_scale), so the
    physical radiation damping B must be injected as ``t_ref * B``. Added
    mass is invariant under the time scaling and is injected raw.

    ``A_offset`` (SHARPy modal basis) is subtracted after the transform —
    used for delta injection: A_inj = A_capy(omega) - A_uvlm_flat_plate, so
    the intact UVLM keeps its forward-speed non-circulatory damping while its
    2D apparent mass is corrected to the 3D BEM value.
    """
    n = n_modes if n_modes is not None else int(linear_aeroelastic.beam.sys.num_modes)
    sl = slice(0, n)
    A, B = capy_at_omega(capy, omega_rad_s, mode_slice=sl)
    if T is not None:
        T = np.asarray(T, dtype=float)[:n, :n]
        A = T.T @ A @ T
        B = T.T @ B @ T
    if A_offset is not None:
        A = A - np.asarray(A_offset, dtype=float)[:n, :n]
    if t_ref is None:
        warnings.warn(
            "apply_bem_from_capy: t_ref not provided — injecting physical damping into the "
            "time-scaled beam over-weights it by 1/t_ref. Pass t_ref = ScalingFacts['length']/V.",
            RuntimeWarning,
        )
        C_add = B
    else:
        C_add = float(t_ref) * B
    set_bem_modal(linear_aeroelastic.beam.sys, A, C_add)
    return A, B


def velocity_sweep_with_bem(
    linear_aeroelastic,
    capy: dict[str, Any],
    v_list: np.ndarray,
    *,
    n_modes: int,
    omega_init_rad_s: float = 100.0,
    relax: float = 0.5,
    T: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    import scipy.linalg as sclalg

    results: list[dict[str, Any]] = []
    omega_est = float(omega_init_rad_s)

    for V in np.asarray(v_list, dtype=float).ravel():
        t_ref = linear_aeroelastic.uvlm.sys.ScalingFacts["length"] / float(V)
        apply_bem_from_capy(linear_aeroelastic, capy, omega_est, n_modes=n_modes, T=T, t_ref=t_ref)
        ss = linear_aeroelastic.update(float(V))
        eigs, _ = sclalg.eig(ss.A)
        uvlm = linear_aeroelastic.uvlm
        dt_dim = uvlm.sys.ScalingFacts["length"] / float(V) * ss.dt
        eigs_ct = np.log(eigs) / dt_dim
        idx = int(np.argmax(eigs_ct.real))
        lam = eigs_ct[idx]
        omega_new = float(abs(lam.imag))
        if omega_new > 1e-6:
            omega_est = (1.0 - relax) * omega_est + relax * omega_new
        results.append(
            {
                "V": float(V),
                "g": float(lam.real),
                "omega_rad_s": omega_new,
                "omega_est_next": omega_est,
            }
        )
    return results


def default_capytaine_npz() -> Path:
    capy_dir = _FSI_ROOT / "FLUID" / "capytaine" / "ABRAMSON1965" / "results_modal_radiation"
    candidates = (
        "modal_radiation_AB_depth_0p0508.npz",
        "modal_radiation_AB_depth_0.0508.npz",
        "modal_radiation_AB.npz",
    )
    for name in candidates:
        p = capy_dir / name
        if p.is_file():
            print(f"[capy_bem_bridge] using Capytaine NPZ: {p.name}")
            if name == "modal_radiation_AB.npz":
                warnings.warn(
                    "Falling back to legacy modal_radiation_AB.npz which lacks meta/source keys "
                    "(no basis check, no mode alignment possible).",
                    RuntimeWarning,
                )
            return p
    raise FileNotFoundError(f"No modal_radiation NPZ under {capy_dir}")
