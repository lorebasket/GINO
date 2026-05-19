"""
Scale Im(Q) only when assembling modal aero damping (C_eff / C_aero).

Roger RFA fitting in ``flutter_solver`` always uses unscaled ``Q_modal`` data.
Apply ``config.aero_im_Q_scale`` afterward on the imaginary / B1 damping path.
Optionally apply the same scale to Roger lag-state aerodynamic forcing.
"""

from __future__ import annotations

import numpy as np


def aero_im_Q_scale(config) -> float:
    """Scalar multiplier for aerodynamic damping from Im(Q) or Roger B1."""
    if config is None:
        return 1.0
    return float(getattr(config, "aero_im_Q_scale", 1.0))


def scale_roger_lag_forcing(config) -> bool:
    """Whether ``aero_im_Q_scale`` should also multiply Roger lag-state forcing."""
    if config is None:
        return False
    return bool(getattr(config, "aero_im_Q_scale_lags", False))


def roger_lag_force_scale(config) -> float:
    """Multiplier for ``q_dyn * Blag`` lag forcing in Roger/RFA augmented systems."""
    return aero_im_Q_scale(config) if scale_roger_lag_forcing(config) else 1.0


def scale_Qhh_imag_for_damping(Qhh: np.ndarray, config) -> np.ndarray:
    """
    Return Q with Re(Q) unchanged and Im(Q) multiplied by ``aero_im_Q_scale``.

    Used in PK ``Ac`` so the RFA fit (if any) can remain on unscaled data while
    each PK iteration applies the scale only in C_aero = (q/2) Im(Q).
    """
    s = aero_im_Q_scale(config)
    if abs(s - 1.0) < 1e-14:
        return Qhh
    Qhh = np.asarray(Qhh, dtype=complex)
    return np.real(Qhh) + 1j * s * np.imag(Qhh)


def log_aero_im_Q_scale_once(config, *, context: str = "") -> None:
    """Print scale notice once per analysis (guarded by config attribute)."""
    s = aero_im_Q_scale(config)
    lag_msg = ", lag forcing also scaled" if scale_roger_lag_forcing(config) else ""
    if abs(s - 1.0) < 1e-14 and not lag_msg:
        return
    if getattr(config, "_aero_im_Q_scale_logged", False):
        return
    msg = (
        f"[aero Im(Q) scale] aero_im_Q_scale={s:g} applied when assembling "
        f"C_eff / C_aero{lag_msg} (RFA fit uses unscaled Q)."
    )
    if context:
        msg = f"{msg} ({context})"
    print(msg)
    setattr(config, "_aero_im_Q_scale_logged", True)
