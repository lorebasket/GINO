"""
Newmark-beta discrete structural state matrix for p-k flutter (SHARPy parity).

Ports the Ass assembly from ``sharpy.linear.src.lingebm.newmark_ss`` without
depending on SHARPy.  The free-vibration eigenvalues ``mu`` of ``Ass`` map to
continuous poles via ``s = log(mu) / dt``.
"""

from __future__ import annotations

import numpy as np
from scipy import linalg as la


def newmark_discrete_ss(
    M: np.ndarray,
    C: np.ndarray,
    K: np.ndarray,
    dt: float,
    num_damp: float = 1e-4,
    *,
    M_is_spd: bool = False,
) -> np.ndarray:
    """
    Discrete-time state matrix ``Ass`` for ``M qdd + C qd + K q = f``.

    ``num_damp`` is SHARPy's ``newmark_damp`` (alpha in gamma = 1/2 + alpha).
    Returns ``Ass`` with state ``[q; qdot]`` (2n x 2n), same layout as the
    continuous companion matrix used elsewhere in the p-k engine.
    """
    dt = float(dt)
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")

    th1 = 0.5 + float(num_damp)
    th2 = 0.0625 + 0.25 * (th1 + th1 ** 2)

    dt2 = dt ** 2
    a1 = th2 * dt2
    a0 = 0.5 * dt2 - a1
    b1 = th1 * dt
    b0 = dt - b1

    n = K.shape[0]
    imat = np.eye(n)

    if M_is_spd:
        m_factors = la.cho_factor(M)
        m_solve = lambda b: la.cho_solve(m_factors, b)
    else:
        m_factors = la.lu_factor(M)
        m_solve = lambda b: la.lu_solve(m_factors, b)

    minv_k = m_solve(K)
    minv_c = m_solve(C)

    ass0 = np.block(
        [
            [imat - a0 * minv_k, dt * imat - a0 * minv_c],
            [-b0 * minv_k, imat - b0 * minv_c],
        ]
    )
    ass1 = np.block(
        [
            [imat + a1 * minv_k, a1 * minv_c],
            [b1 * minv_k, imat + b1 * minv_c],
        ]
    )

    ass1_factors = la.lu_factor(ass1)
    return la.lu_solve(ass1_factors, ass0)


def discrete_poles_to_continuous(mu: np.ndarray, dt: float) -> np.ndarray:
    """Map discrete eigenvalues ``mu`` to continuous poles ``s = log(mu)/dt``."""
    dt = float(dt)
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")
    return np.log(np.asarray(mu, dtype=complex)) / dt
