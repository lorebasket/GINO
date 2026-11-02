"""
GINO-matched Rayleigh structural damping for SHARPy linear flutter (Abramson 1965).

Adds C = alpha*M + beta*K on the structural tangent matrices **before** SHARPy
snapshots them for non-dimensional time scaling (``scale_system_normalised_time``).
Without that ordering, ``update(u)`` restores C=0 at every velocity sample.
"""

from __future__ import annotations

from functools import wraps
from typing import Sequence

import numpy as np

# Abramson GINO baseline (config_ABRAMSON1965.py)
ABRAMSON_RAYLEIGH_MODE_IDS = (0, 1)
ABRAMSON_RAYLEIGH_ZETAS = (0.076, 0.039)

_installed = False
_last_fit: dict[str, float] | None = None


def compute_rayleigh_ab_from_modes(
    omegas: Sequence[float],
    zetas: Sequence[float],
    *,
    cond_limit: float = 1e12,
) -> tuple[float, float]:
    """Fit alpha, beta so zeta_i = alpha/(2 omega_i) + beta omega_i/2."""
    w = np.asarray(omegas, dtype=float).ravel()
    z = np.asarray(zetas, dtype=float).ravel()
    if w.shape != z.shape or w.size < 2:
        raise ValueError("Need at least two (omega, zeta) pairs")
    if np.any(w <= 1e-12):
        raise ValueError("Rayleigh target frequencies must be positive")
    A = np.column_stack([1.0 / (2.0 * w), w / 2.0])
    if w.size == 2:
        try:
            if np.linalg.cond(A) <= cond_limit:
                alpha, beta = np.linalg.solve(A, z)
                return float(alpha), float(beta)
        except Exception:
            pass
    alpha, beta = np.linalg.lstsq(A, z, rcond=None)[0]
    return float(alpha), float(beta)


def rayleigh_c_from_omegas(
    M: np.ndarray,
    K: np.ndarray,
    omegas: Sequence[float],
    zetas: Sequence[float],
) -> tuple[np.ndarray, float, float]:
    alpha, beta = compute_rayleigh_ab_from_modes(omegas, zetas)
    M_a = np.asarray(M, dtype=float)
    K_a = np.asarray(K, dtype=float)
    return alpha * M_a + beta * K_a, alpha, beta


def _dry_omegas_rad_s(
    flex_sys,
    mode_ids: Sequence[int],
    *,
    omega_rad_s: Sequence[float] | None,
) -> tuple[float, float]:
    if omega_rad_s is not None:
        w = tuple(float(x) for x in omega_rad_s)
        return w[0], w[1]
    fn = flex_sys.freq_natural
    if fn is None:
        raise RuntimeError(
            "Rayleigh damping: Modal frequencies unavailable; "
            "run Modal before LinearAssembler or pass omega_rad_s."
        )
    fn = np.asarray(fn, dtype=float).ravel()
    mids = [int(i) for i in mode_ids if 0 <= int(i) < len(fn)]
    if len(mids) < 2:
        raise RuntimeError(
            f"Rayleigh damping: need modes {mode_ids}, got {len(fn)} frequencies"
        )
    return float(fn[mids[0]]), float(fn[mids[1]])


def inject_rayleigh_on_flex(
    flex_sys,
    *,
    mode_ids: Sequence[int] = ABRAMSON_RAYLEIGH_MODE_IDS,
    target_zetas: Sequence[float] = ABRAMSON_RAYLEIGH_ZETAS,
    omega_rad_s: Sequence[float] | None = None,
    verbose: bool = True,
) -> dict[str, float] | None:
    """Add Rayleigh C to ``flex_sys.Cstr`` once (idempotent)."""
    global _last_fit
    if getattr(flex_sys, "_gino_rayleigh_done", False):
        return _last_fit

    w0, w1 = _dry_omegas_rad_s(flex_sys, mode_ids, omega_rad_s=omega_rad_s)
    C_ray, alpha, beta = rayleigh_c_from_omegas(
        flex_sys.Mstr, flex_sys.Kstr, (w0, w1), target_zetas[:2],
    )
    if flex_sys.Cstr is None:
        flex_sys.Cstr = C_ray
    else:
        flex_sys.Cstr = np.asarray(flex_sys.Cstr, dtype=float) + C_ray

    z_ach = [
        alpha / (2.0 * w0) + beta * w0 / 2.0,
        alpha / (2.0 * w1) + beta * w1 / 2.0,
    ]
    _last_fit = {
        "alpha": alpha,
        "beta": beta,
        "omega0_rad_s": w0,
        "omega1_rad_s": w1,
        "zeta0": z_ach[0],
        "zeta1": z_ach[1],
    }
    flex_sys._gino_rayleigh_done = True

    if verbose:
        print(
            "[Rayleigh] SHARPy structural damping: "
            f"alpha={alpha:.4e}, beta={beta:.4e}"
        )
        for mid, w, z_tgt, z_ach in zip(
            mode_ids[:2], (w0, w1), target_zetas[:2], z_ach,
        ):
            print(
                f"  mode {mid}: omega={w:.3f} rad/s "
                f"({w / (2 * np.pi):.2f} Hz), "
                f"zeta_target={z_tgt:.4g}, zeta_achieved={z_ach:.4g}"
            )
    return _last_fit


def install_rayleigh_structural_damping(
    *,
    mode_ids: Sequence[int] = ABRAMSON_RAYLEIGH_MODE_IDS,
    target_zetas: Sequence[float] = ABRAMSON_RAYLEIGH_ZETAS,
    omega_rad_s: Sequence[float] | None = None,
    verbose: bool = True,
) -> None:
    """
    Patch ``LinearBeam.assemble`` so Rayleigh C is injected **before**
    ``scale_system_normalised_time`` snapshots the reference matrices.
    """
    global _installed
    if _installed:
        return

    from sharpy.linear.assembler import linearbeam

    mode_ids_t = tuple(int(i) for i in mode_ids)
    zetas_t = tuple(float(z) for z in target_zetas)
    fixed_omegas = None if omega_rad_s is None else tuple(float(w) for w in omega_rad_s)
    orig_assemble = linearbeam.LinearBeam.assemble

    @wraps(orig_assemble)
    def assemble_with_rayleigh(self, t_ref=None):
        inject_rayleigh_on_flex(
            self.sys,
            mode_ids=mode_ids_t,
            target_zetas=zetas_t,
            omega_rad_s=fixed_omegas,
            verbose=verbose,
        )
        return orig_assemble(self, t_ref)

    linearbeam.LinearBeam.assemble = assemble_with_rayleigh
    _installed = True


def last_rayleigh_fit() -> dict[str, float] | None:
    return _last_fit
