"""
Empirical fluid damping beyond Capytaine radiation (potential flow).

Abramson (1964/65) Fig. 5 reports an exponential decay factor δ [s⁻¹] from free-decay
tests. For underdamped modal motion with envelope ∝ exp(−δ t),

    δ ≈ ζ ω_n,   C_modal = 2 ζ M ω = 2 δ M.

The experiment lies ~3–4 s⁻¹ above the "measured 3D coefficients" curve at subcritical
speed; use ``empirical_fluid_delta_add_s`` to target that gap while keeping structural
Rayleigh damping separate.
"""

from __future__ import annotations

import numpy as np


def _as_length_n(values, n_modes: int, name: str) -> np.ndarray:
    arr = np.atleast_1d(np.asarray(values, dtype=float)).ravel()
    if arr.size == 1:
        return np.full(n_modes, float(arr[0]))
    if arr.size != n_modes:
        raise ValueError(f"{name} length {arr.size} != n_modes {n_modes}")
    return arr


def modal_empirical_C(
    config,
    V: float,
    M_eff: np.ndarray,
    *,
    omega_ref: float | None = None,
) -> np.ndarray:
    """
  Build modal empirical fluid damping matrix [kg/s] added to ``C_eff``.

  Config (all optional except ``empirical_fluid_damping=True``):

  - ``empirical_fluid_model``: ``'abramson_delta'`` (default), ``'constant_kg_s'``,
    ``'velocity_linear'``, or ``'delta_at_omega'``.
  - ``empirical_fluid_delta_add_s``: extra δ [s⁻¹] → ``C_ii = 2 δ M_ii`` (Abramson gap).
  - ``empirical_fluid_C_kg_s``: diagonal (N,) or scalar [kg/s] for ``constant_kg_s``.
  - ``empirical_fluid_C_per_ms``: ``C_ii += scale * V`` [kg/s per (m/s)].
  - ``empirical_fluid_omega_ref_rad_s``: for ``delta_at_omega``: ``C_ii = 2 δ M_ii ω_ref``.
    """
    if not bool(getattr(config, "empirical_fluid_damping", False)):
        return np.zeros_like(M_eff, dtype=float)

    n_modes = int(M_eff.shape[0])
    M_diag = np.maximum(np.diag(np.asarray(M_eff, dtype=float)), 1e-12)
    model = str(getattr(config, "empirical_fluid_model", "abramson_delta")).strip().lower()
    C = np.zeros((n_modes, n_modes), dtype=float)

    if model == "constant_kg_s":
        c_diag = _as_length_n(
            getattr(config, "empirical_fluid_C_kg_s", 15.0),
            n_modes,
            "empirical_fluid_C_kg_s",
        )
        np.fill_diagonal(C, c_diag)
    elif model == "velocity_linear":
        scale = float(getattr(config, "empirical_fluid_C_per_ms", 0.5))
        np.fill_diagonal(C, scale * float(V))
    elif model == "delta_at_omega":
        delta = float(getattr(config, "empirical_fluid_delta_add_s", 4.0))
        w_ref = float(omega_ref if omega_ref is not None else getattr(
            config, "empirical_fluid_omega_ref_rad_s", 70.0
        ))
        np.fill_diagonal(C, 2.0 * delta * M_diag * max(w_ref, 1e-6))
    else:
        # 'abramson_delta': C_ii = 2 * delta_add * M_ii  (δ–M link at current M_eff)
        if model != "abramson_delta":
            raise ValueError(
                f"Unknown empirical_fluid_model={model!r}; "
                "use abramson_delta, constant_kg_s, velocity_linear, or delta_at_omega."
            )
        delta = float(getattr(config, "empirical_fluid_delta_add_s", 4.0))
        np.fill_diagonal(C, 2.0 * delta * M_diag)

    c_v = float(getattr(config, "empirical_fluid_C_per_ms", 0.0))
    if abs(c_v) > 0.0:
        C += c_v * float(V) * np.eye(n_modes)

    return C


def describe_empirical_C(config, V: float, M_eff: np.ndarray, C_emp: np.ndarray) -> str:
    """One-line log string for the enabled empirical model."""
    model = getattr(config, "empirical_fluid_model", "abramson_delta")
    diag = np.diag(C_emp)
    return (
        f"[empirical fluid] model={model}, V={V:.3g} m/s, "
        f"C_emp diagonal [kg/s]={np.array2string(diag, precision=4)}"
    )
